"""Phase-1a instrument: redundancy / unique / synergy between Fenwick
blocks, and what the fold keeps (docs/OPERA_Redundancy_Research_2026-09-24.md
§2.5, §4). Exploratory; no gates.

For positions t whose prefix decomposes into exactly C Fenwick blocks,
take the block states of one layer (slot 0 = oldest/largest, slot C-1 =
newest) and a target Y (the next byte, or the byte at t-k). Mutual
information is estimated as the cross-entropy drop of a probe below the
unigram baseline:

    I(Y; X) ~= H_unigram(Y) - CE_probe(Y | X)          (bits, lower bound)

Reported per target:
  * single sources: every slot s, I(Y; block_s)
  * MMI partial information decomposition of (A = oldest, B = newest):
      Red = min(I_A, I_B); Unq_A = I_A - Red; Unq_B = I_B - Red;
      Syn = I_AB - max(I_A, I_B)
  * Darwinism curves: I(Y; newest m blocks) and I(Y; oldest m blocks),
    m = 1..C (plateau = redundant record; steady climb = unique content)
  * block-anchored single-copy targets (fixed offset inside a block):
    A_last = last byte of the oldest block, s1_first =
    first byte of slot 1, sB2_first = first byte of the second-newest
    block -- their address is determined by the block, so the block
    probe is a fair reader and the fold's retention of them is the clean
    "single-copy record" test
  * doc_style: a POINTER variable (document-level byte-histogram
    cluster, k=16) -- Darwinism predicts it is redundant across blocks
    (the oldest block's FIRST byte is not used as a target: the oldest
    block always starts at position 0, so it is one value per document)
  * fold retention: I(Y; all C blocks) vs I(Y; fold readout P) vs
    I(Y; head input). P is a deterministic function of the blocks, so a
    gap I(all) > I(P) is information the fold discarded (as far as the
    probe class can see; a linear probe can under-read the concatenation,
    so I(P) > I(all) is possible and is reported, not clipped).

Layer 0 is the clean case: its leaves are token embeddings, so the blocks
are DISJOINT fragments of the prefix (the Quantum-Darwinism setting). At
layer >= 1 leaves already carry prefix context, so blocks overlap in what
they can know; reported with that caveat.

Usage:
  python experiments/block_pid.py --arm st_off [--layer 0] [--count 5]
      [--n-seqs 400] [--per-seq 100] [--probe linear|mlp] [--threads 3]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arm_utils import OUT, layer_streams, load_arm, load_corpus
from opera_lm.model import fenwick_blocks
from opera_lm.train import make_batch_full

LAGS = (1, 4, 8, 16)


def doc_clusters(seqs, V, k=16, seed=0, iters=50):
    """POINTER-VARIABLE target: a document-level style/topic label, the
    k-means cluster (k=16) of each document's normalized byte histogram.
    Quantum-Darwinism prediction: such a variable is recorded redundantly
    (decodable from ANY single block, flat Darwinism curve), unlike byte
    identity. Unsupervised on the documents' bytes only (no model states),
    fit on all probe documents."""
    H = np.stack([np.bincount(np.asarray(s), minlength=V)[:V] for s in seqs])
    H = H / H.sum(1, keepdims=True)
    H = np.sqrt(H)                                 # Hellinger geometry
    g = np.random.default_rng(seed)
    cent = H[g.choice(len(H), k, replace=False)]
    for _ in range(iters):
        lab = ((H[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for c in range(k):
            if (lab == c).any():
                cent[c] = H[lab == c].mean(0)
    return lab


def collect(model, seqs, layer, C, per_seq, T, seed, V=259):
    """Sample positions with exactly C blocks; return fp16 block states
    [N, C, d], fold readout [N, dm], head input [N, dm], and targets."""
    g = np.random.default_rng(seed)
    doc_cluster = doc_clusters(seqs, V, seed=seed)
    table, _ = fenwick_blocks(T)
    # block-anchored addresses: for prefix t (length t+1), slot s's span
    # starts at node*2^level. Targets at a FIXED offset inside a block
    # have an address the block itself determines (a lag-k target's
    # offset inside its block varies with t, which no block encodes).
    starts = np.zeros((T, C), dtype=np.int64)
    ends = np.zeros((T, C), dtype=np.int64)
    for t in range(T):
        if len(table[t]) == C:
            for s_, (lv, node) in enumerate(table[t]):
                starts[t, s_] = node << lv
                ends[t, s_] = ((node + 1) << lv) - 1
    blocks, pref, head, y_next, y_lag, y_anch = [], [], [], [], [], []
    for i in range(0, len(seqs), 8):
        chunk = seqs[i:i + 8]
        ids, lens = make_batch_full(chunk, T)
        out = None
        for st in layer_streams(model, ids):
            if st['layer'] == layer:
                out = st
                break
        count = out['count'].numpy()
        for b in range(ids.shape[0]):
            n = int(lens[b])
            ok = np.nonzero((count == C) & (np.arange(T) >= max(LAGS))
                            & (np.arange(T) + 1 < n))[0]
            if len(ok) == 0:
                continue
            pick = g.choice(ok, size=min(per_seq, len(ok)), replace=False)
            pick_t = torch.from_numpy(pick)
            blocks.append(out['gathered'][b, pick_t, :C].half())
            pref.append(out['prefix'][b, pick_t].half())
            head.append(out['head_in'][b, pick_t].half())
            row = ids[b]
            y_next.append(row[pick_t + 1])
            y_lag.append(torch.stack([row[pick_t - k] for k in LAGS], -1))
            st_, en_ = starts[pick], ends[pick]
            y_anch.append(torch.stack([
                torch.full((len(pick),), doc_cluster[i + b],
                           dtype=torch.long),            # document style/topic
                row[torch.from_numpy(en_[:, 0])],        # oldest block, last byte
                row[torch.from_numpy(st_[:, 1])],        # slot 1, 1st byte
                row[torch.from_numpy(st_[:, C - 2])],    # 2nd-newest, 1st byte
            ], -1))
    return (torch.cat(blocks), torch.cat(pref), torch.cat(head),
            torch.cat(y_next), torch.cat(y_lag), torch.cat(y_anch))


def entropy_unigram(ytr, yte, V):
    """Cross-entropy (bits) of the add-one-smoothed train unigram on test:
    the H(Y) baseline the probes are measured against."""
    cnt = torch.bincount(ytr, minlength=V).float() + 1.0
    lp = torch.log(cnt / cnt.sum())
    return float(-lp[yte].mean() / math.log(2))


def probe_ce(Xtr, ytr, Xte, yte, V, kind='linear', seed=0, epochs=40):
    """Train a softmax probe (linear or 1-hidden-layer MLP), early-stopped
    on a 10% split of the train set; return test CE in bits."""
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-5
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    nv = max(1, Xtr.shape[0] // 10)
    Xv, yv, Xt, yt = Xtr[:nv], ytr[:nv], Xtr[nv:], ytr[nv:]
    D = Xtr.shape[1]
    if kind == 'mlp':
        net = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.GELU(),
                                  torch.nn.Linear(512, V))
    else:
        net = torch.nn.Linear(D, V)
    # Start EXACTLY at the unigram baseline: zero output weights, bias =
    # log train prior (add-one). The probe can then only report
    # information the features actually add; a random init starts ~ln V
    # and early stopping on small data would report negative MI.
    last = net[-1] if kind == 'mlp' else net
    with torch.no_grad():
        cnt = torch.bincount(ytr, minlength=V).float() + 1.0
        last.weight.zero_()
        last.bias.copy_(torch.log(cnt / cnt.sum()))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    with torch.no_grad():
        best = F.cross_entropy(net(Xv), yv).item()      # = unigram on val
    best_state = {k: t.clone() for k, t in net.state_dict().items()}
    bad = 0
    for ep in range(epochs):
        perm = torch.randperm(Xt.shape[0])
        net.train()
        for j in range(0, Xt.shape[0], 1024):
            sel = perm[j:j + 1024]
            loss = F.cross_entropy(net(Xt[sel]), yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = F.cross_entropy(net(Xv), yv).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return F.cross_entropy(net(Xte), yte).item() / math.log(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--count', type=int, default=5,
                   help='positions whose prefix has exactly this many '
                        'Fenwick blocks (5 is the modal count at T=1024)')
    p.add_argument('--n-seqs', type=int, default=400)
    p.add_argument('--per-seq', type=int, default=100)
    p.add_argument('--probe', default='linear', choices=('linear', 'mlp'))
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--pca', type=int, default=128,
                   help='per-source PCA dims (0 = raw features)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    model, kw = load_arm(args.arm)
    data, meta = load_corpus()
    V = meta['vocab_size']
    seqs = data[1][:args.n_seqs]                 # test_short
    C, T = args.count, 1024
    blocks, pref, head, y_next, y_lag, y_anch = collect(
        model, seqs, args.layer, C, args.per_seq, T, args.seed)
    N = blocks.shape[0]
    print(f"[{args.arm}] layer {args.layer}, count {C}: {N} positions "
          f"from {len(seqs)} seqs ({time.time() - t0:.0f}s)")
    # sequence-blocked split is approximated by order: collect() walks
    # sequences in order, so the last 20% of rows are unseen sequences.
    cut = int(0.8 * N)
    targets = {'next': y_next}
    for j, k in enumerate(LAGS):
        targets[f'lag{k}'] = y_lag[:, j]
    for j, nm in enumerate(('doc_style', 'A_last', 's1_first', 'sB2_first')):
        targets[nm] = y_anch[:, j]

    # Per-source PCA (fit on the train rows only) so that probes over 1
    # vs C blocks are not dominated by feature count: each block (and P,
    # head input) is reduced to --pca dims. Trained OPERA states have a
    # participation ratio of ~27-50 (F4), so 128 dims keep nearly all
    # their variance. --pca 0 disables.
    def pca_fit(X, k):
        Xc = X[:cut].float()
        mu = Xc.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xc - mu, full_matrices=False)
        return mu, Vh[:k].T

    def reduce(X, fit):
        if fit is None:
            return X.float()
        mu, W = fit
        return (X.float() - mu) @ W

    k = args.pca
    blk_fit = [pca_fit(blocks[:, s], k) if k else None for s in range(C)]
    red_blocks = torch.stack([reduce(blocks[:, s], blk_fit[s])
                              for s in range(C)], 1)
    pref_r = reduce(pref, pca_fit(pref, k) if k else None)
    head_r = reduce(head, pca_fit(head, k) if k else None)

    def feats(slots):
        return red_blocks[:, list(slots)].reshape(N, -1)

    res = {'arm': args.arm, 'layer': args.layer, 'count': C, 'N': N,
           'probe': args.probe, 'pca': args.pca, 'model_kwargs': {k: str(v) for k, v in kw.items()},
           'targets': {}}
    for tname, y in targets.items():
        ytr, yte = y[:cut], y[cut:]
        H = entropy_unigram(ytr, yte, V)

        memo = {}

        def I(X):
            ce = probe_ce(X[:cut], ytr, X[cut:], yte, V, args.probe,
                          args.seed)
            return H - ce

        def Islots(slots):
            key = tuple(slots)
            if key not in memo:
                memo[key] = I(feats(list(slots)))
            return memo[key]

        single = [Islots([s]) for s in range(C)]
        I_A, I_B = single[0], single[C - 1]
        I_AB = Islots([0, C - 1])
        red = min(I_A, I_B)
        pid = {'I_A_oldest': I_A, 'I_B_newest': I_B, 'I_AB': I_AB,
               'Red': red, 'Unq_A': I_A - red, 'Unq_B': I_B - red,
               'Syn': I_AB - max(I_A, I_B)}
        newest = [Islots(range(C - m, C)) for m in range(1, C + 1)]
        oldest = [Islots(range(0, m)) for m in range(1, C + 1)]
        I_all = newest[-1]
        I_P = I(pref_r)
        I_H = I(head_r)
        r = {'H_unigram_bits': H, 'single_slot': single, 'pid_oldest_newest': pid,
             'darwin_newest_m': newest, 'darwin_oldest_m': oldest,
             'I_all_blocks': I_all, 'I_fold_readout': I_P,
             'I_head_input': I_H, 'fold_retention': I_P / I_all if I_all > 0
             else None}
        res['targets'][tname] = r
        print(f"  {tname:6s} H={H:.2f}b | slots(old->new) "
              + " ".join(f"{v:.3f}" for v in single)
              + f" | PID red {red:.3f} unqA {pid['Unq_A']:.3f} "
              f"unqB {pid['Unq_B']:.3f} syn {pid['Syn']:+.3f}"
              + f" | all {I_all:.3f} fold {I_P:.3f} head {I_H:.3f}"
              + f" | newest-m " + " ".join(f"{v:.3f}" for v in newest),
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'block_pid_{args.arm}_L{args.layer}'
                             f'_c{C}_{args.probe}_pca{args.pca}.json')
    json.dump(res, open(path, 'w'), indent=2)
    print(f"  wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
