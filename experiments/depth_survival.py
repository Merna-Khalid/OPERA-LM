"""EXPLORATORY diagnostic (no gate, no adoption): information survival
from the first layer to the last in the OPERA stack.

Motivation (Merna, 2026-09-09): after the scale-tied operator problem
closed (beta^l, LO-Muon, level-conditioning all falsified), the unasked
question is the ACROSS-layer one. Each OPERA layer is

    prefix  = Fenwick_readout(build_tree(current))   # all new info
    gate    = sigmoid(MLP(prefix))                   # scalar/position
    current = gate * MLP(prefix) + (1-gate) * current

so the ONLY carry-forward channel for previous-layer state is a scalar
sigmoid gate, and every layer's new information enters through a fold-sum
over ~log T Fenwick nodes. This probe measures, on a TRAINED incumbent:

  D1  blend-gate saturation per layer + the per-position identity
      attenuation prod_l (1 - gate_l) -- how much of the embedding
      highway survives to the last layer;
  D2  linear decodability of the INPUT byte at t from each stream
      (leaf states in, Fenwick prefix, states out) per layer;
  D3  linear decodability of the byte at t-k (recency profile) from the
      final prefix and final states, k in {1..511} -- is the readout
      sharpest on recent bytes the way attention's softmax is?
  D4  position decodability (32 bins) from embeddings (chance by
      construction: pe=none) vs tree readout -- where does position
      live, if anywhere;
  D5  effective rank (participation ratio) of each stream's covariance;
  D6  per-position model CE vs the Fenwick prefix node count count(t)
      -- systematic position-dependent capacity from the readout shape;
  D7  per-layer CE of the aux heads (how the layer-wise readouts
      improve with depth) + next-byte linear probe as head calibration.

All probes are closed-form ridge (XtX d x d, scatter XtY), batch-level
80/20 train/test split, lambda grid {0.03, 0.3, 3.0} reported at best.
Exploratory: any architecture change this motivates requires its own
pre-registration before it can enter Path A.

Usage:
  python experiments/depth_survival.py --tag rx_fgate_wd \
      [--ckpt runs_reprs/rx_fgate_wd/<...>.pt] [--layers 2]
  python experiments/depth_survival.py --tag control_init --control
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opera_lm.model import OperaSpinorFenwickTree  # noqa: E402
from opera_lm.train import make_batch_full         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'runs_reprs')

# Match the incumbent arm family (repr_study ARMS bytes_* rows).
D, NB, VOCAB, T = 512, 128, 259, 1024
LAMBDAS = (0.03, 0.3, 3.0)
LAGS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 511)
POS_BINS = 32


def load_corpus():
    cache = os.path.join(ROOT, 'assets',
                         f'corpus_bytes_T{T}_a20000.pkl')
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache}")
    with open(cache, 'rb') as f:
        return pickle.load(f)


def build_model(layers, ckpt=None, seed=42):
    torch.manual_seed(seed)
    m = OperaSpinorFenwickTree(
        vocab_size=VOCAB, d=D, nb=NB, num_layers=layers,
        pe_mode='none', fold_mode='left', rot_mode='free')
    if ckpt:
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        m.load_state_dict(sd, strict=True)
    return m.eval()


@torch.no_grad()
def capture(model, seqs, device, batch=16):
    """Run the exact forward() layer loop, capturing every stream.

    Returns dict of fp16 CPU tensors [N, T, d] (or [N, T] for gates)
    plus logits-derived CE/argmax. N = len(seqs) padded to full batches.
    """
    L = model.num_layers
    streams = {f'S_in_{l}': [] for l in range(L)}
    streams.update({f'P_{l}': [] for l in range(L)})
    streams.update({f'S_out_{l}': [] for l in range(L)})
    gates = {l: [] for l in range(L)}
    ces = {l: [] for l in range(L)}
    correct, total = 0, 0
    fenwick_count = None

    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        token_ids, lengths = make_batch_full(chunk, T)
        token_ids = token_ids.to(device)
        lengths_d = lengths.to(device)
        B, Tt = token_ids.shape

        model._need_locks = False
        model._need_energy = False
        model._rot_dense_cache.clear()

        current = model.word_emb(token_ids)
        for l in range(L):
            streams[f'S_in_{l}'].append(
                current.detach().to('cpu', torch.float16))
            R_L, R_R, R_O = model.get_rotations(l)
            levels, _, _ = model.build_tree(current, l, R_L, R_R, R_O)
            if fenwick_count is None:
                offs, off = [], 0
                for lv in levels:
                    offs.append(off)
                    off += lv.shape[1]
                _, count, _, _, _ = model._fenwick_indices(
                    Tt, len(levels), offs, device)
                fenwick_count = count.cpu()
            prefix = model.prefix_states(levels, Tt, l, R_L, R_R, R_O)
            streams[f'P_{l}'].append(
                prefix.detach().to('cpu', torch.float16))
            logits = model.apply_head(prefix)
            lp = F.log_softmax(logits.float(), dim=-1)
            tgt = token_ids[:, 1:]
            ce = -lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            valid = (torch.arange(Tt - 1, device=device)[None, :]
                     < (lengths_d[:, None] - 1))
            ces[l].append(torch.where(valid, ce,
                                      torch.zeros_like(ce)).cpu())
            if l == L - 1:
                pred = logits[:, :-1].argmax(-1)
                correct += ((pred == tgt) & valid).sum().item()
                total += valid.sum().item()
            mixed = model.cross_mlp[l](prefix)
            gate = model.blend_gate[l](prefix)
            gates[l].append(gate.detach().squeeze(-1).cpu())
            current = gate * mixed + (1 - gate) * current
            streams[f'S_out_{l}'].append(
                current.detach().to('cpu', torch.float16))

    out = {k: torch.cat(v) for k, v in streams.items()}
    out['gates'] = {l: torch.cat(v).numpy() for l, v in gates.items()}
    out['ces'] = {l: torch.cat(v).numpy() for l, v in ces.items()}
    out['argmax_acc'] = correct / max(total, 1)
    out['fenwick_count'] = fenwick_count.numpy()
    return out


class Ridge:
    """Closed-form ridge accumulator: XtX (d x d) + class-scattered XtY."""

    def __init__(self, d, C):
        self.d, self.C = d, C
        self.Sxx = torch.zeros(d, d)
        self.Sxy = torch.zeros(C, d)
        self.sx = torch.zeros(d)
        self.sy = torch.zeros(C)
        self.n = 0

    def add(self, X, y):
        X = X.float()
        self.Sxx += X.T @ X
        self.Sxy.index_add_(0, y, X)
        self.sx += X.sum(0)
        self.sy += torch.bincount(y, minlength=self.C).float()
        self.n += X.shape[0]

    def _centered(self):
        n = max(self.n, 1)
        Sxx = self.Sxx - torch.outer(self.sx, self.sx) / n
        Sxy = self.Sxy - torch.outer(self.sy, self.sx) / n
        return Sxx, Sxy

    def accuracy(self, X, y):
        X = X.float()
        best = None
        Sxx, Sxy = self._centered()
        mu = self.sx / max(self.n, 1)
        tr = Sxx.diagonal().sum().clamp_min(1e-8) / self.d
        for lam in LAMBDAS:
            A = Sxx + torch.eye(self.d) * (lam * tr)
            W = torch.linalg.solve(A, Sxy.T)          # [d, C]
            pred = ((X - mu) @ W).argmax(-1)
            acc = (pred == y).float().mean().item()
            best = acc if best is None else max(best, acc)
        return best


def tokens_of(stream, ids, lengths, min_t=0, shift=0):
    """Flatten stream [N, T, d] -> X [n, d], y [n] = ids[b, t+shift].

    Valid positions are the predictable ones (t < len-1, the same mask
    the model's own CE uses) with t >= min_t (lag floor); the target
    index t+shift is clipped into [0, T-1]."""
    N, Tt, d = stream.shape
    ar = np.arange(Tt)
    ok = (ar[None, :] >= min_t) & (ar[None, :] < (lengths - 1)[:, None])
    b_idx, t_idx = np.nonzero(ok)
    tt = np.clip(t_idx + shift, 0, Tt - 1)
    Xi = stream[torch.from_numpy(b_idx), torch.from_numpy(t_idx)]
    yy = torch.from_numpy(ids[b_idx, tt])
    return Xi, yy


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float('nan')


def probe_stream(stream, ids, lengths, name, C=VOCAB, shift=0, min_t=0,
                 n_train_batches=0.8):
    """Batch-level 80/20 split ridge probe. Returns test accuracy."""
    N = stream.shape[0]
    cut = max(1, int(N * n_train_batches))
    tr = Ridge(stream.shape[-1], C)
    for b in range(cut):
        X, y = tokens_of(stream[b:b + 1], ids[b:b + 1],
                         lengths[b:b + 1], min_t=min_t, shift=shift)
        tr.add(X, y)
    Xte, yte = tokens_of(stream[cut:], ids[cut:], lengths[cut:],
                         min_t=min_t, shift=shift)
    if tr.n == 0 or Xte.shape[0] == 0:
        return float('nan')
    return tr.accuracy(Xte, yte)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--control', action='store_true',
                   help='untrained seed-42 model of the same shape')
    p.add_argument('--n-seqs', type=int, default=128)
    p.add_argument('--all-layers', type=int, default=0,
                   help='0: lags on final+layer0 streams only; 1: all '
                        'prefix streams P_l; 2: also all state streams')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.device is None:
        args.device = ('mps' if torch.backends.mps.is_available()
                       else 'cpu')
    if args.control and args.ckpt:
        raise SystemExit('--control and --ckpt are exclusive')

    data, meta = load_corpus()
    train_data, test_short, test_long, _ = data
    seqs = test_short[:args.n_seqs]
    print(f"[{args.tag}] {len(seqs)} eval seqs, T={T}, "
          f"L={args.layers}, device={args.device}"
          f"{' (UNTRAINED CONTROL)' if args.control else ''}")

    t0 = time.time()
    model = build_model(args.layers,
                        None if args.control else args.ckpt, seed=42)
    model.to(args.device)
    cap = capture(model, seqs, args.device)
    L = model.num_layers
    ids = np.concatenate([make_batch_full(seqs[i:i + 16], T)[0].numpy()
                          for i in range(0, len(seqs), 16)])
    lens = np.concatenate([make_batch_full(seqs[i:i + 16], T)[1].numpy()
                           for i in range(0, len(seqs), 16)])
    print(f"  forward+capture {time.time() - t0:.0f}s; "
          f"model argmax acc {cap['argmax_acc']:.3f}")

    res = {'tag': args.tag, 'layers': L, 'argmax_acc': cap['argmax_acc'],
           'seqs': len(seqs)}

    # D1: gate saturation + identity attenuation
    gates = {}
    for l in range(L):
        g = cap['gates'][l]                       # [N, T]
        gates[f'layer{l}'] = {
            'mean': float(g.mean()), 'p10': float(np.percentile(g, 10)),
            'p90': float(np.percentile(g, 90)),
            'frac_gt_0.9': float((g > 0.9).mean())}
    atten = np.ones(T)
    for l in range(L):
        atten *= (1.0 - cap['gates'][l].mean(axis=0))
    gates['identity_attenuation_mean'] = float(atten.mean())
    gates['identity_attenuation_p10'] = float(
        np.percentile(atten, 10))
    res['gates'] = gates
    print(f"  D1 gates: " + " ".join(
        f"L{l}={gates[f'layer{l}']['mean']:.3f}" for l in range(L))
        + f" | prod(1-g) mean {gates['identity_attenuation_mean']:.2e}")

    # D2: input-byte identity decode per stream
    ident = {}
    for key in [f'S_in_{l}' for l in range(L)] + \
               [f'P_{l}' for l in range(L)] + [f'S_out_{l}' for l in range(L)]:
        acc = probe_stream(cap[key], ids, lens, key)
        ident[key] = acc
        print(f"  D2 identity from {key:9s}: {acc:.3f}")
    res['identity_acc'] = ident

    # D3: recency profile from final prefix + final states (+ layer-0 prefix)
    recency = {}
    if args.all_layers:
        srcs = ([f'P_{l}' for l in range(L)]
                + ([f'S_out_{l}' for l in range(L)]
                   if args.all_layers > 1 else []))
        lags = (1, 2, 4, 8, 16)
    else:
        srcs = (f'P_{L - 1}', f'S_out_{L - 1}', 'P_0')
        lags = LAGS
    for src in srcs:
        row = {}
        for k in lags:
            row[k] = probe_stream(cap[src], ids, lens, f'{src}@lag{k}',
                                  shift=-k)
        recency[src] = row
        print(f"  D3 lags from {src}: " + " ".join(
            f"k={k}:{v:.2f}" for k, v in row.items()))
    res['recency_acc'] = recency

    # D4: position decode (32 bins), predictable positions only
    pos = {}
    edges = np.linspace(0, T, POS_BINS + 1).astype(int)
    bins = torch.from_numpy(np.clip(
        np.digitize(np.arange(T), edges) - 1, 0, POS_BINS - 1))
    ar = np.arange(T)
    ok = ar[None, :] < (lens - 1)[:, None]
    b_idx, t_idx = np.nonzero(ok)
    cut = max(1, int(ids.shape[0] * 0.8))
    tr_m = torch.from_numpy(b_idx) < cut
    yb = bins[torch.from_numpy(t_idx)]
    for src in ('S_in_0', f'P_{L - 1}', f'S_out_{L - 1}'):
        stream = cap[src]
        X = stream[torch.from_numpy(b_idx), torch.from_numpy(t_idx)]
        tr = Ridge(stream.shape[-1], POS_BINS)
        tr.add(X[tr_m], yb[tr_m])
        pos[src] = tr.accuracy(X[~tr_m], yb[~tr_m])
        print(f"  D4 position bins from {src}: {pos[src]:.3f} "
              f"(chance {1 / POS_BINS:.3f})")
    res['position_acc'] = pos

    # D5: effective rank per stream
    rank = {}
    for key in cap:
        if key not in ('gates', 'ces', 'argmax_acc', 'fenwick_count'):
            X = cap[key].reshape(-1, cap[key].shape[-1]).float()
            X = X[torch.randperm(X.shape[0])[:60000]]
            X = X - X.mean(0, keepdim=True)
            ev = torch.linalg.eigvalsh(X.T @ X / X.shape[0])
            ev = ev.clamp_min(0)
            pr = (ev.sum() ** 2 / (ev ** 2).sum()).item()
            rank[key] = {'participation_ratio': pr,
                         'top1_frac': (ev[-1] / ev.sum()).item()}
            print(f"  D5 rank of {key:9s}: PR {pr:6.1f}/{D} "
                  f"top1 {rank[key]['top1_frac']:.2f}")
    res['rank'] = rank

    # D6: per-position CE vs Fenwick prefix node count
    ce_final = cap['ces'][L - 1].mean(axis=0)      # [T-1]
    cnt = cap['fenwick_count'][:T - 1].astype(float)
    res['ce_pos_vs_count'] = {
        'spearman': spearman(ce_final, cnt),
        'ce_mean': float(ce_final.mean()),
        'ce_first32': float(ce_final[:32].mean()),
        'ce_last32': float(ce_final[-32:].mean()),
        'count_min': int(cnt.min()), 'count_max': int(cnt.max()),
        'by_count': {int(c): float(ce_final[cnt == c].mean())
                     for c in np.unique(cnt)}}
    print(f"  D6 CE vs fenwick-count spearman "
          f"{res['ce_pos_vs_count']['spearman']:+.3f} "
          f"(count {int(cnt.min())}..{int(cnt.max())})")

    # D7: per-layer CE + next-byte probe calibration
    res['per_layer_ce'] = {str(l): float(cap['ces'][l].sum()
                                        / max((lens - 1).sum(), 1))
                           for l in range(L)}
    nxt = probe_stream(cap[f'P_{L - 1}'], ids, lens, 'next', shift=+1)
    res['next_probe_acc'] = nxt
    print(f"  D7 per-layer CE " + " ".join(
        f"L{l}={res['per_layer_ce'][str(l)]:.3f}" for l in range(L))
        + f" | next-byte probe {nxt:.3f} vs argmax "
          f"{cap['argmax_acc']:.3f}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'depth_survival_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {path} ({time.time() - t0:.0f}s total)")


if __name__ == '__main__':
    main()
