"""Multi-query associative recall (MQAR; Arora et al., Zoology, arXiv
2312.04927) for OPERA -- the direct test of in-context recall
(docs/OPERA_Recall_Research_2026-09-24.md §5 step 1). Measurement only.

Sequence layout (length T):
    k1 v1 k2 v2 ... kN vN | filler ... q ... filler ... q ...
The first 2N tokens bind N distinct keys to N distinct values. The rest is
random filler tokens, with each of the N keys re-inserted once as a query
at a random later position; the target at a query position (the next
token) is that key's value. Loss and accuracy are computed ONLY at query
positions. Vocabularies are disjoint: keys [KEY0, KEY0+NK), values
[VAL0, VAL0+NV), filler [FIL0, FIL0+NF); 0 = pad.

Reading the accuracy:
    ~1/NV : the model does not use the context at all
    ~1/N  : it knows the answer is one of the N values it saw, but cannot
            bind the queried key to its value (no recall)
    ~1    : recall

Usage:
  python experiments/mqar.py [--d 256] [--layers 2] [--steps 4000]
      [--T 256] [--pairs 4 8 16 32] [--batch 64] [--device mps]
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from opera_lm import OperaSpinorFenwickTree            # noqa: E402
from opera_lm.muon import Muon, split_muon_params      # noqa: E402

NK, NV, NF = 512, 512, 64
KEY0, VAL0 = 1, 1 + NK
FIL0 = VAL0 + NV
VOCAB = FIL0 + NF


def make_batch(B, T, N, g, device):
    """ids [B, T], query mask [B, T] (True where the NEXT token is a
    recalled value), and the key->query distance at each query."""
    ids = torch.randint(FIL0, FIL0 + NF, (B, T), generator=g)
    qmask = torch.zeros(B, T, dtype=torch.bool)
    dist = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        keys = torch.randperm(NK, generator=g)[:N] + KEY0
        vals = torch.randperm(NV, generator=g)[:N] + VAL0
        ids[b, 0:2 * N:2] = keys
        ids[b, 1:2 * N:2] = vals
        # query positions: distinct EVEN offsets after the pairs, so every
        # answer slot (p+1) is odd and can never be another query's slot
        slots = (T - 2 * N - 1) // 2
        assert slots >= N, "sequence too short for this many pairs"
        qpos = 2 * torch.randperm(slots, generator=g)[:N] + 2 * N
        for i in range(N):
            p = int(qpos[i])
            ids[b, p] = keys[i]
            ids[b, p + 1] = vals[i]          # the answer, also as next input
            qmask[b, p] = True
            dist[b, p] = p - 2 * i           # from the key's first position
    return ids.to(device), qmask.to(device), dist.to(device)


def evaluate(model, T, N, device, n=512, seed=123, control=False):
    g = torch.Generator().manual_seed(seed + N)
    correct = total = 0
    bydist = {}
    model.eval()
    with torch.no_grad():
        for _ in range(n // 64):
            ids, qm, dist = make_batch(64, T, N, g, 'cpu')
            ids_d = ids.to(device)
            lens = torch.full((64,), T, device=device)
            logits = model(ids_d, lens, head_last_only=True).logits[-1].float().cpu()
            pred = logits.argmax(-1)
            tgt = (torch.roll(ids, 1, dims=1) if control
                   else torch.roll(ids, -1, dims=1))
            ok = (pred == tgt) & qm
            correct += ok.sum().item()
            total += qm.sum().item()
            for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 256), (256, 10 ** 9)):
                sel = qm & (dist >= lo) & (dist < hi)
                if sel.any():
                    c, t = bydist.get(lo, (0, 0))
                    bydist[lo] = (c + (ok & sel).sum().item(), t + sel.sum().item())
    model.train()
    return correct / max(total, 1), {k: c / t for k, (c, t) in sorted(bydist.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--T', type=int, default=256)
    p.add_argument('--pairs', type=int, nargs='+', default=[4, 8, 16, 32])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--device', default='mps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='opera_base')
    p.add_argument('--hmem-nb', type=int, default=0,
                   help='quaternion holographic memory slots per layer (0 = off)')
    p.add_argument('--hmem-decay', default='none', choices=['none', 'fixed', 'gated'])
    p.add_argument('--hmem-conv', type=int, default=0,
                   help='short causal conv width on the memory projection (0 = off)')
    p.add_argument('--control', action='store_true',
                   help='pipeline sanity check: at the same query positions '
                        'the target is the PREVIOUS token (purely local copy, '
                        'no key->value binding needed)')
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    model = OperaSpinorFenwickTree(
        vocab_size=VOCAB, d=args.d, nb=args.d // 4, num_layers=args.layers,
        pe_mode='none', fold_mode='left', rot_mode='free',
        head_mode='stream', fold_impl='downsweep',
        use_metal=(dev == 'mps'), hmem_nb=args.hmem_nb,
        hmem_decay=args.hmem_decay, hmem_conv=args.hmem_conv).to(dev)
    mu, ad = split_muon_params(model, ('fusion_gate',))
    opt = Muon([{'params': [q for _, q in mu], 'names': [n for n, _ in mu],
                 'use_muon': True, 'lr': 0.02, 'weight_decay': 0.01},
                {'params': [q for _, q in ad], 'use_muon': False, 'lr': 1e-3}],
               lr=1e-3)
    base_lr = [grp['lr'] for grp in opt.param_groups]
    g = torch.Generator().manual_seed(args.seed + 1)
    warm = 200
    t0 = time.time()
    print(f"[{args.tag}] OPERA d={args.d} L={args.layers} hmem={args.hmem_nb} "
          f"decay={args.hmem_decay} conv={args.hmem_conv} T={args.T} "
          f"pairs={args.pairs} vocab={VOCAB} steps={args.steps}", flush=True)
    for step in range(args.steps):
        f = (step + 1) / warm if step < warm else \
            0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm)))
        for grp, lr0 in zip(opt.param_groups, base_lr):
            grp['lr'] = lr0 * f
        N = args.pairs[step % len(args.pairs)]
        ids, qm, _ = make_batch(args.batch, args.T, N, g, dev)
        lens = torch.full((args.batch,), args.T, device=dev)
        with torch.autocast(dev if dev != 'cpu' else 'cpu', dtype=torch.bfloat16,
                            enabled=(dev != 'cpu')):
            logits = model(ids, lens, head_last_only=True).logits[-1]
        tgt = (torch.roll(ids, 1, dims=1) if args.control
               else torch.roll(ids, -1, dims=1))
        loss = F.cross_entropy(logits.float()[qm], tgt[qm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  query loss {loss.item():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = {'tag': args.tag, 'd': args.d, 'layers': args.layers, 'T': args.T,
           'hmem_nb': args.hmem_nb, 'hmem_decay': args.hmem_decay,
           'hmem_conv': args.hmem_conv,
           'steps': args.steps, 'results': {}}
    print("\n  accuracy at query positions (chance 1/512; 'knows the value set' ~1/N):")
    for N in args.pairs:
        acc, bd = evaluate(model, args.T, N, dev, control=args.control)
        res['results'][N] = {'acc': acc, 'by_distance': bd, 'one_over_N': 1 / N}
        print(f"  N={N:3d}  acc {acc:.3f}   (1/N = {1 / N:.3f})   by key->query distance: "
              + " ".join(f"[{k},...):{v:.2f}" for k, v in bd.items()), flush=True)
    out = os.path.join(ROOT, 'runs_reprs', f'mqar_{args.tag}.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
