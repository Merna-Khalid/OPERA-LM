"""Phase-1b instrument: the captured-MI curve (L²M's direct estimator
applied to OPERA; docs/OPERA_Redundancy_Research_2026-09-24.md §2.3, §4).
Exploratory; no gates.

For a held-out document x and a prefix length l, let Y = x[l : l+m]
(m bytes; default 32) and Y' = Y without its first byte. With the model
q as the density estimator (Chen et al., NeurIPS 2025, eq. 6):

    I(l) = E[ log q(Y' | x[P-l : P+1]) - log q(Y' | x[P]) ]    (bits)

with the window fixed at P and only the context length l varied (paired
design, see curve()) -- how many bits about the next m-1 bytes the model
extracts from the l bytes of context before the window (conditioning both terms on the
window's first byte, because the model never predicts a sequence's first
token). This is the MI the MODEL CAPTURES, not the language's true MI.

L²M's prediction for a history state that is too small: the curve bends
flat with l (natural language's bipartite MI keeps growing ~ l^beta), and
it lifts when the state is enlarged. We report the curve, a log-log slope
over [16, train length], and the growth over the extrapolation region.

Usage:
  python experiments/mi_curve.py --arms st_off [x_tie2 ...] \
      [--n-seqs 64] [--m 32] [--lmax 2048] [--threads 3]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arm_utils import OUT, load_arm, load_corpus

L_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
          384, 512, 768, 1024, 1536, 2048)


@torch.no_grad()
def token_logprobs(model, ids):
    """lp[b, t] = log q(ids[b, t+1] | ids[b, :t+1]) (natural log)."""
    B, T = ids.shape
    lens = torch.full((B,), T, dtype=torch.long)
    logits = model(ids, lens, head_last_only=True).logits[-1].float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    return lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # [B, T-1]


def curve(model, seqs, grid, m):
    """Per-document I(l) in bits for every l in grid: [n_docs, len(grid)],
    plus the conditional per-byte CE of the window (bits).

    PAIRED design: the target window is FIXED at the end of each document's
    usable span (Y = x[P : P+m], P = max(grid)); only the amount of
    context before it varies, X = x[P-l : P]. Every l is scored on the
    same bytes, so content variance cancels in the l-to-l comparison. The
    model sees the window at position l of its input (its Fenwick
    decomposition changes with l -- position is structure; that is part
    of what is being measured)."""
    P = max(grid)
    I = np.zeros((len(seqs), len(grid)))
    ce = np.zeros((len(seqs), len(grid)))
    for i, s in enumerate(seqs):
        x = torch.tensor(s[:P + m], dtype=torch.long)
        marg = token_logprobs(model, x[P:P + m][None])[0].sum().item()
        for j, l in enumerate(grid):
            lp = token_logprobs(model, x[P - l:P + m][None])[0]
            cond = lp[l:l + m - 1].sum().item()               # x_{P+1..P+m-1}
            I[i, j] = (cond - marg) / math.log(2)
            ce[i, j] = -cond / (m - 1) / math.log(2)
    return I, ce


def loglog_slope(ls, vals, lo, hi):
    sel = [(l, v) for l, v in zip(ls, vals) if lo <= l <= hi and v > 0]
    if len(sel) < 3:
        return None
    x = np.log([l for l, _ in sel])
    y = np.log([v for _, v in sel])
    return float(np.polyfit(x, y, 1)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--n-seqs', type=int, default=64)
    p.add_argument('--m', type=int, default=32)
    p.add_argument('--lmax', type=int, default=2048)
    p.add_argument('--train-len', type=int, default=1024)
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    data, _ = load_corpus()
    grid = [l for l in L_GRID if l <= args.lmax]
    seqs = [s for s in data[2] if len(s) >= max(grid) + args.m][:args.n_seqs]
    print(f"{len(seqs)} test_long documents (len >= {max(grid) + args.m}); "
          f"window m={args.m}; train length {args.train_len}")
    results = {}
    for name in args.arms:
        t0 = time.time()
        model, kw = load_arm(name)
        I, ce = curve(model, seqs, grid, args.m)
        mean, sem = I.mean(0), I.std(0, ddof=1) / math.sqrt(len(seqs))
        slope = loglog_slope(grid, mean, 16, args.train_len)
        tl = args.train_len
        g_in = (mean[grid.index(tl)] / mean[grid.index(tl // 2)]
                if tl in grid and tl // 2 in grid else None)
        g_ex = (mean[grid.index(2 * tl)] / mean[grid.index(tl)]
                if 2 * tl in grid and tl in grid else None)
        results[name] = {
            'grid': grid, 'I_bits_mean': mean.tolist(),
            'I_bits_sem': sem.tolist(),
            'window_ce_bits': ce.mean(0).tolist(),
            'loglog_slope_16_to_train': slope,
            'growth_train_half_to_train': g_in,
            'growth_train_to_2x': g_ex,
            'n_docs': len(seqs), 'm': args.m}
        print(f"\n[{name}] ({time.time() - t0:.0f}s)  slope[16..{tl}] "
              f"{slope if slope is None else round(slope, 3)}  "
              f"growth x2 in-length {g_in and round(g_in, 3)}  "
              f"growth x2 extrapolated {g_ex and round(g_ex, 3)}")
        for l, mu, se, c in zip(grid, mean, sem, ce.mean(0)):
            mark = '  <- train length' if l == tl else ''
            print(f"  l={l:5d}  I={mu:7.3f} ± {se:5.3f} bits   "
                  f"window CE {c:.3f} b/byte{mark}")
    tag = '_'.join(args.arms)
    path = os.path.join(OUT, f'mi_curve_{tag}.json')
    json.dump(results, open(path, 'w'), indent=2)
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
