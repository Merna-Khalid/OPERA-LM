"""EXPLORATORY calibration for depth_survival.py: what context length of
byte n-gram matches the trained OPERA model's CE on the same eval
positions?

If the model's CE sits at the level of a context-k n-gram trained on the
same corpus, its effective conditioning window is ~k bytes regardless of
the 1024-byte context the Fenwick readout could in principle use.
Stupid-backoff (factor 0.3), hashed counts (2^26 bins, int32), train
split of the same corpus. Context lengths {1,2,3,4,6,8,12}
(k=1 is a bigram model). Eval = test_short[:128], model's own mask
(t < len-1), so CE is directly comparable to depth_survival D7.
"""
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EVAL = 128
BINS = 1 << 26
BACKOFF = 0.3
ORDERS = (1, 2, 3, 4, 6, 8, 12)

with open(os.path.join(ROOT, 'assets', 'corpus_bytes_T1024_a20000.pkl'),
          'rb') as f:
    data, meta = pickle.load(f)
train_data, test_short = data[0], data[1]

SEP = meta['vocab_size'] - 1        # last special as boundary marker
stream = np.concatenate(
    [np.asarray(s, dtype=np.int64) for s in train_data[:20000]]
    + [np.array([SEP], dtype=np.int64)] * len(train_data[:20000]))
V = meta['vocab_size'] + 1
print(f"train stream {len(stream):,} ids "
      f"({meta['units_per_byte']:.2f} units/byte)")


def gram_hashes(a, k):
    """exact rolling base-V hash of every length-k window (uint64)."""
    x = np.zeros(len(a) - k + 1, dtype=np.uint64)
    for j in range(k):
        x = x * np.uint64(V) + a[j:len(a) - k + 1 + j].astype(np.uint64)
    return x * np.uint64(0x9E3779B97F4A7C15) \
        ^ (x * np.uint64(0xC2B2AE3D27D4EB4F) >> np.uint64(29))


class NG:
    """exact counts for context length k (order k+1 n-gram model):
    sorted-unique hash tables + searchsorted lookups (no collisions)."""

    def __init__(self, stream, k):
        self.k = k
        u, c = np.unique(gram_hashes(stream, k), return_counts=True)
        self.u_ctx, self.c_ctx = u, c.astype(np.float64)
        u, c = np.unique(gram_hashes(stream, k + 1), return_counts=True)
        self.u_nxt, self.c_nxt = u, c.astype(np.float64)

    def counts(self, q_ctx, q_nxt):
        i = np.searchsorted(self.u_ctx, q_ctx)
        i = np.minimum(i, len(self.u_ctx) - 1)
        c1 = np.where(self.u_ctx[i] == q_ctx, self.c_ctx[i], 0.0)
        j = np.searchsorted(self.u_nxt, q_nxt)
        j = np.minimum(j, len(self.u_nxt) - 1)
        c2 = np.where(self.u_nxt[j] == q_nxt, self.c_nxt[j], 0.0)
        return c1, c2


models = {k: NG(stream, k) for k in ORDERS}
N_TOT = float(len(stream))


def ce_with_ctx(ids, kmax):
    """stupid-backoff CE [bits/unit] using context lengths kmax..1.

    ps[t-1] = P(arr[t]); level k covers positions t >= k+1 (its first
    full-context prediction is t = k+1, index k)."""
    arr = np.asarray(ids, dtype=np.int64)
    n = len(arr)
    ps = np.full(n - 1, np.nan)
    assigned = np.zeros(n - 1, dtype=bool)
    weight = 1.0
    for k in [q for q in ORDERS if q <= kmax][::-1]:
        m = models[k]
        ctx = gram_hashes(arr[:-1], k)          # j-th: arr[j..j+k-1]
        tgt = gram_hashes(arr, k + 1)           # j-th: arr[j..j+k]
        c1, c2 = m.counts(ctx, tgt)
        idx = np.arange(n - k) + k - 1          # ps index of arr[j+k]
        sel = (~assigned[idx]) & (c2 > 0) & (c1 > 0)
        ps[idx[sel]] = weight * c2[sel] / c1[sel]
        assigned[idx[sel]] = True
        weight *= BACKOFF
    # unigram floor for positions nothing covered / always-missed
    cnt = np.bincount(arr, minlength=V).astype(np.float64)
    uni = (cnt[arr[1:]] + 0.01) / (N_TOT + V * 0.01)
    miss = ~assigned
    ps[miss] = uni[miss] * BACKOFF ** kmax
    return -np.log2(ps)


def main():
    seqs = test_short[:N_EVAL]
    rows = {}
    for kmax in ORDERS:
        tot, n = 0.0, 0
        for s in seqs:
            if len(s) <= kmax + 1:
                continue
            ce = ce_with_ctx(s, kmax)
            m = len(s) - 1
            tot += float(ce[:m].sum())
            n += m
        rows[kmax] = tot / n
        print(f"  ctx {kmax:>2}: CE {rows[kmax]:.4f} bits/unit "
              f"over {n:,} positions")
    out = os.path.join(ROOT, 'runs_reprs', 'ngram_ce.json')
    with open(out, 'w') as f:
        json.dump({'ngram_ce_bits_by_ctx': rows,
                   'model_ce_bits': 1.30}, f, indent=2)
    print("  model (depth_survival D7, same seqs/mask): CE 1.30 bits/unit")
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
