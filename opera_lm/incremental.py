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
                    level_sin_enc, inject_geometry)


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
        if getattr(model, 'fold_grade', None) is not None:
            # The graded fold masks each step's accumulator write per
            # slot group; this decoder folds unmasked and would silently
            # decode a different function.
            raise NotImplementedError(
                "fold_grade (scale-graded readout) is not supported by "
                "OperaDecoder yet")
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
        # tree_disent_rank: the RAW (pre-disentangle) nodes, whose left
        # neighbour feeds the next node's disentangler update.
        self.raw = [[self.model.word_emb.weight.new_zeros(0, self.model.d)]
                    for _ in range(self.model.num_layers)]
        # hmem: per-layer running memory M [n, 4] (fp32) and the previous
        # position's unit key (zeros before the first token: no binding).
        if getattr(self.model, 'hmem_nb', 0):
            n = self.model.hmem_nb
            self.hmem_M = [torch.zeros(n, 4, device=self.device)
                           for _ in range(self.model.num_layers)]
            self.hmem_kprev = [torch.zeros(n, 4, device=self.device)
                               for _ in range(self.model.num_layers)]
            # hmem_conv: the last K raw projections (row 0 = newest)
            self.hmem_yhist = ([None] * self.model.num_layers
                               if getattr(self.model, 'hmem_conv', 0) else None)
        else:
            self.hmem_M = self.hmem_kprev = None
        # T1.4: per-layer delta-rule memory state M (fp32, as in the
        # batched path). None when the arm is off.
        if getattr(self.model, 'mem_mode', 'none') == 'delta':
            dm = self.model.mem_dim
            self.mem = [torch.zeros(dm, dm, dtype=torch.float32,
                                    device=self.device)
                        for _ in range(self.model.num_layers)]
        else:
            self.mem = None

    def _embed_position(self, token_id, t, geom=None, geom_block=0):
        """Embedding + positional contribution for position t (eval: no
        dropout). Matches model.forward's embedding stage for one row.

        geom (optional [3] tensor): mirrors forward()'s geom/geom_mask --
        non-LM callers only, splices a literal vector into block
        `geom_block` of this ONE leaf, after PE, before it enters the
        tree. A single append() call is always "this position or not",
        so unlike forward() there's no separate mask arg: geom is not
        None means inject here, with an all-True (scalar) mask."""
        m = self.model
        tok = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        x = m.word_emb(tok)[0, 0]                          # [d_model]
        if m.pe_mode == 'sin':
            x = x + sinusoidal_pos_enc(t + 1, m.d_model, self.device)[t]
        elif m.pe_mode == 'rotor':
            cos, sin = rotor_pos_tables(t + 1, m.nb_model, self.device)
            x = apply_rotor_pe(x.reshape(1, 1, -1),
                               cos[t:], sin[t:], m.nb_model)[0, 0]
        if geom is not None:
            mask = torch.ones((), dtype=torch.bool, device=self.device)
            x = inject_geometry(x, geom, mask, m.nb_model, geom_block)
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
    def append(self, token_id, geom=None, geom_block=0):
        m = self.model
        t = self.t
        d = m.d                        # tree width
        x = self._embed_position(int(token_id), t, geom, geom_block)
        head_in = None
        for l in range(m.num_layers):
            R_L, R_R, R_O = self.rots[l]
            levels = self.nodes[l]

            # The layer's leaf: the stream, pre-normed (resid_mode='add')
            # and lifted to tree width (state_mult) -- as in forward().
            leaf = x
            if m.tree_in_norm is not None:
                leaf = m.tree_in_norm[l](leaf)
            if m.tree_in is not None:
                leaf = m.tree_in[l](leaf)

            # Insert the new leaf and build its (new) ancestor nodes.
            # Node (k, j) is composed the moment its span completes and
            # never changes afterwards -> append-only cache.
            ins = leaf
            raws = self.raw[l]
            k = 0
            while True:
                while len(levels) <= k:
                    levels.append(leaf.new_zeros(0, d))
                    raws.append(leaf.new_zeros(0, d))
                n = levels[k].shape[0]
                raws[k] = torch.cat([raws[k], ins.unsqueeze(0)], 0)
                if m.tree_disent_up is not None:
                    # causal disentangler from the RAW left neighbour, as
                    # in build_tree (node 0 of a level has none)
                    if n >= 1:
                        ins = ins + m._lowrank_corr(raws[k][n - 1],
                                                    'tree_disent', l)
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
            # fold_h0: start from the learned accumulator and compose
            # every block (the batched fold's step 0), else seed with
            # the first block.
            acc = (m.fold_h0[l].to(leaf.dtype) if m.fold_h0 is not None
                   else None)
            blocks = []            # (level, block state) for the readout
            for (lvl, j) in fenwick_blocks_of(t + 1):
                blk = levels[lvl][j]
                if m.fold_theta is not None:
                    blk = self._fold_twist(blk, lvl, l)
                blocks.append((lvl, blk))
                if acc is None:
                    acc = blk
                else:
                    acc = m._compose(
                        m._fold_innov(acc, blk, l).unsqueeze(0),
                        blk.unsqueeze(0), l, fR_L, fR_R, fR_O,
                        gate_bias=fgb, fold=True)[0][0]
            prefix = acc
            # fold_dir='both': counterclockwise (right-nested) fold over the
            # same blocks, older block as the LEFT child, joined by the
            # per-slot gain -- same ops/order as model._right_fold.
            if getattr(m, 'fold_join_gain', None) is not None:
                acc_r = blocks[-1][1]
                for (_lv, blk) in reversed(blocks[:-1]):
                    acc_r = m._compose(blk.unsqueeze(0), acc_r.unsqueeze(0),
                                       l, fR_L, fR_R, fR_O, fold=True)[0][0]
                g = m.fold_join_gain[l].to(acc.dtype)
                prefix = (prefix.reshape(m.nb, 4)
                          + g[:, None] * acc_r.reshape(m.nb, 4)).reshape(d)

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
                k = F.normalize(m.mem_k[l](leaf), dim=-1).float()
                v = m.mem_v[l](leaf).float()
                beta = torch.sigmoid(m.mem_beta[l](leaf)).float()
                M = self.mem[l]
                M += beta * ((v - M @ k).unsqueeze(-1) @ k.unsqueeze(0))
                q = m.mem_q[l](prefix).float()
                y = (M @ q).to(prefix.dtype)
                g = torch.sigmoid(m.mem_gate[l](
                    torch.cat([prefix, y], dim=-1)))
                prefix = prefix + g * m.mem_out[l](y)

            if m.tree_out is not None:
                prefix = m.tree_out[l](prefix)

            # hmem: bind (previous key, this value), accumulate, unbind with
            # this position's key -- same ops as model._hmem_read, one row.
            if self.hmem_M is not None:
                from .model import quat_mul, quat_conj
                if self.hmem_yhist is not None:
                    # short conv: convolve the newest raw projection with
                    # the K-1 before it, then the same parts as the kernel
                    y = m._hmem_proj(x, l, conv=False)
                    K = m.hmem_conv
                    h = self.hmem_yhist[l]
                    if h is None:
                        h = y.new_zeros(K, y.shape[-1])
                    h = torch.cat([y.unsqueeze(0), h[:K - 1]], 0)
                    self.hmem_yhist[l] = h
                    yc = m._hmem_conv_row(h, l).float()
                    n = m.hmem_nb
                    k = F.normalize(yc[:4 * n].reshape(n, 4), dim=-1)
                    v = yc[4 * n:8 * n].reshape(n, 4)
                    w = torch.sigmoid(yc[8 * n:8 * n + 1])
                    if m.hmem_decay == 'gated':
                        lam = torch.sigmoid(yc[8 * n + 1:] + m.hmem_decay_logit[l].float())
                    elif m.hmem_decay == 'fixed':
                        lam = torch.sigmoid(m.hmem_decay_logit[l].float())
                    else:
                        lam = torch.ones(n, device=yc.device)
                    lam = lam.unsqueeze(-1)
                else:
                    k, v, w = m._hmem_parts(x, l)                # x: stream in
                    lam = (m._hmem_lambda(x, l).unsqueeze(-1)     # [n, 1]
                           if getattr(m, 'hmem_decay', 'none') != 'none' else None)
                upd = quat_mul(self.hmem_kprev[l], v) * w
                if lam is not None:
                    self.hmem_M[l] = lam * self.hmem_M[l] + upd
                else:
                    self.hmem_M[l] = self.hmem_M[l] + upd
                from .model import quat_mul_conj_a
                r = quat_mul_conj_a(k, self.hmem_M[l])
                prefix = prefix + m._hmem_out(r, l, prefix.dtype)
                self.hmem_kprev[l] = k

            # Position-wise cross-layer mixing, as in forward().
            mixed = m.cross_mlp[l](prefix)
            if m.resid_mode == 'add':
                x = x + mixed
            else:
                gate = m.blend_gate[l](prefix)
                x = gate * mixed + (1.0 - gate) * x
            head_in = (m.stream_norm[l](x) if m.head_mode == 'stream'
                       else prefix)

        self.t += 1
        return m.apply_head(head_in.reshape(1, 1, -1))[0, 0]     # [vocab]

    def prefill(self, token_ids):
        """Feed a prompt; returns the logits after its last token."""
        logits = None
        for tid in token_ids:
            logits = self.append(int(tid))
        return logits
