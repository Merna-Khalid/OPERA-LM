"""Fenwick-incremental decoding: O(log T) per generated token.

The naive sampler (see opera_generate.py) re-runs the full forward for
every generated token: O(L * T log T) compose nodes per token. That is
pure waste, because the model is exactly causal:

- Tree node (k, j) covers the FIXED span [j*2^k, (j+1)*2^k) and depends
  only on its own leaves, so appending one token never mutates an
  existing node -- it only ADDS <= log2(T) new ancestors per layer.
- Position T's prefix state folds only the <= log2(T+1) Fenwick blocks
  of its own prefix -- all of them already computed.
- Cross-layer mixing (cross_mlp / blend_gate) is position-wise, so the
  next layer's new leaf needs only the new position's prefix state.

`OperaDecoder` caches the per-layer trees and pays O(L log T) compose
nodes per appended token (~T/x less than re-forwarding; ~100x at
T=512), which is what makes CPU inference interactive.

Scope: fold_mode='left' (the incumbent) with shared fold rotors,
use_metal=False, batch size 1, eval mode. Every compose/gate/norm call
is dispatched to the model's own modules in the same order as the
batched path, so per-position logits match a full forward up to float
op-reordering (asserted in opera_lm.selftest).
"""
import torch
import torch.nn.functional as F

from .model import (sinusoidal_pos_enc, rotor_pos_tables, apply_rotor_pe,
                    level_sin_enc)


def fenwick_blocks_of(L):
    """Fenwick decomposition of the prefix of length L: [(level, node)],
    largest block first -- row L-1 of model.fenwick_blocks' table,
    computed without building the whole table."""
    blocks = []
    start = 0
    rem = L
    k = rem.bit_length() - 1
    while rem > 0:
        size = 1 << k
        if size <= rem:
            blocks.append((k, start >> k))
            start += size
            rem -= size
        k -= 1
    return blocks


class OperaDecoder:
    """Incremental decoder for OperaSpinorFenwickTree (fold_mode='left').

    Usage:
        dec = OperaDecoder(model)
        for tid in prompt_ids:
            logits = dec.append(tid)      # logits for the NEXT token
        # sample from logits, then dec.append(sampled) and repeat

    `append` returns the final-layer logits [vocab_size] at the position
    just appended (i.e. the distribution over the token that follows).
    """

    def __init__(self, model, device=None):
        if model.fold_mode != 'left':
            raise NotImplementedError(
                f"OperaDecoder supports fold_mode='left' only "
                f"(got {model.fold_mode!r})")
        # readout_mode='multistate' and mem_mode='delta' ARE supported:
        # the multistate reduction is a static function of the fold's
        # blocks, and the delta memory is incremental by design (M_t
        # updates rank-one per token). Same ops, same order as the
        # batched path -> exactness asserted in opera_lm.selftest.
        if model.rot_free_fold is not None:
            # The batched prefix_states switches fold rotations only on
            # quat_fold (model.py:1022); the free+separate-rotors case is
            # ambiguous upstream, so refuse rather than diverge.
            raise NotImplementedError(
                "fold_rotors='separate' with rot_mode='free' is not "
                "supported by OperaDecoder")
        if model.use_metal:
            raise NotImplementedError(
                "OperaDecoder runs the eager compose path only; "
                "set use_metal=False for inference")
        self.model = model
        self.device = (device if device is not None
                       else next(model.parameters()).device)
        model.eval()
        # Rotations are deterministic functions of the parameters;
        # resolve them once per layer (tree and fold) instead of per node.
        self.rots = [model.get_rotations(l) for l in range(model.num_layers)]
        if model.quat_fold is not None:
            # Mirror model.py:1022 exactly (fold rotors only via quat_fold).
            self.fold_rots = [model.get_rotations(l, fold=True)
                              for l in range(model.num_layers)]
        else:
            self.fold_rots = self.rots
        self.reset()

    def reset(self):
        """Drop all cached state (new conversation / context truncation)."""
        self.t = 0
        # nodes[layer][level] = [n_level, d] tensor of completed tree nodes
        # (level 0 = leaves). Append-only; never mutated in place.
        self.nodes = [[self.model.word_emb.weight.new_zeros(0, self.model.d)]
                      for _ in range(self.model.num_layers)]
        # T1.4: per-layer delta-rule memory state M (fp32, as in the
        # batched path). None when the arm is off.
        if getattr(self.model, 'mem_mode', 'none') == 'delta':
            dm = self.model.mem_dim
            self.mem = [torch.zeros(dm, dm, dtype=torch.float32,
                                    device=self.device)
                        for _ in range(self.model.num_layers)]
        else:
            self.mem = None

    def _embed_position(self, token_id, t):
        """Embedding + positional contribution for position t (eval: no
        dropout). Matches model.forward's embedding stage for one row."""
        m = self.model
        tok = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        x = m.word_emb(tok)[0, 0]                                  # [d]
        if m.pe_mode == 'sin':
            x = x + sinusoidal_pos_enc(t + 1, m.d, self.device)[t]
        elif m.pe_mode == 'rotor':
            cos, sin = rotor_pos_tables(t + 1, m.nb, self.device)
            x = apply_rotor_pe(x.reshape(1, 1, -1),
                               cos[t:], sin[t:], m.nb)[0, 0]
        return x

    def _fold_twist(self, blk, level, layer_idx):
        """--fold-scale: a block of span 2^k enters the fold twisted by
        angle k*theta_b about its z-axis (model.py:1036-1049), one row."""
        m = self.model
        theta = m.fold_theta[layer_idx]                            # [nb]
        ang = level * theta
        c, s = torch.cos(ang), torch.sin(ang)
        h = blk.reshape(m.nb, 4)
        sc, vx, vy, vz = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
        vx2 = vx * c - vy * s
        vy2 = vx * s + vy * c
        return torch.stack([sc, vx2, vy2, vz], dim=-1).reshape(m.d)

    @torch.no_grad()
    def append(self, token_id):
        m = self.model
        t = self.t
        d = m.d
        x = self._embed_position(int(token_id), t)
        prefix = None
        for l in range(m.num_layers):
            R_L, R_R, R_O = self.rots[l]
            levels = self.nodes[l]

            # Insert the new leaf and build its (new) ancestor nodes.
            # Node (k, j) is composed the moment its span completes and
            # never changes afterwards -> append-only cache.
            ins = x
            k = 0
            while True:
                while len(levels) <= k:
                    levels.append(x.new_zeros(0, d))
                n = levels[k].shape[0]
                levels[k] = torch.cat([levels[k], ins.unsqueeze(0)], 0)
                if (n + 1) % 2 == 0:
                    left = levels[k][n - 1].unsqueeze(0)
                    ins = m._compose(left, ins.unsqueeze(0),
                                     l, R_L, R_R, R_O)[0][0]
                    k += 1
                else:
                    break

            # Fold the Fenwick blocks of prefix length t+1, largest block
            # first, with the fold's own rotations/bias -- the same op
            # sequence as the batched left fold for this one position.
            fR_L, fR_R, fR_O = self.fold_rots[l]
            fgb = (m.fold_gate_bias[l]
                   if m.fold_gate_bias is not None else None)
            acc = None
            blocks = []            # (level, block state) for the readout
            for (lvl, j) in fenwick_blocks_of(t + 1):
                blk = levels[lvl][j]
                if m.fold_theta is not None:
                    blk = self._fold_twist(blk, lvl, l)
                blocks.append((lvl, blk))
                if acc is None:
                    acc = blk
                else:
                    acc = m._compose(acc.unsqueeze(0), blk.unsqueeze(0),
                                     l, fR_L, fR_R, fR_O,
                                     gate_bias=fgb)[0][0]
            prefix = acc

            # T0.4: static multi-state reduction over the same blocks
            # (batched path reads the post-twist `gathered`; so do we).
            if getattr(m, 'readout_mode', 'none') == 'multistate':
                red = None
                for s, (lvl, blk) in enumerate(blocks):
                    le = level_sin_enc(
                        torch.tensor([[lvl]], device=self.device),
                        m.d)[0, 0].to(blk.dtype)
                    term = m.readout_gate[l, s] * (blk + le)
                    red = term if red is None else red + term
                prefix = prefix + m.readout_out[l](red)

            # T1.4: rank-one delta-rule update of this layer's memory
            # from the new leaf, then content query from the prefix.
            if self.mem is not None:
                k = F.normalize(m.mem_k[l](x), dim=-1).float()
                v = m.mem_v[l](x).float()
                beta = torch.sigmoid(m.mem_beta[l](x)).float()
                M = self.mem[l]
                M += beta * ((v - M @ k).unsqueeze(-1) @ k.unsqueeze(0))
                q = m.mem_q[l](prefix).float()
                y = (M @ q).to(prefix.dtype)
                g = torch.sigmoid(m.mem_gate[l](
                    torch.cat([prefix, y], dim=-1)))
                prefix = prefix + g * m.mem_out[l](y)

            # Position-wise cross-layer mixing (model.py:1654-1658).
            mixed = m.cross_mlp[l](prefix)
            gate = m.blend_gate[l](prefix)
            x = gate * mixed + (1.0 - gate) * x

        self.t += 1
        return m.apply_head(prefix.reshape(1, 1, d))[0, 0]       # [vocab]

    def prefill(self, token_ids):
        """Feed a prompt; returns the logits after its last token."""
        logits = None
        for tid in token_ids:
            logits = self.append(int(tid))
        return logits
