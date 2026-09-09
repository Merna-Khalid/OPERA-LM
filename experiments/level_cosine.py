"""level_cosine.py -- the OPERA_Optimizer_prereg.md stage-1 kill-switch.

THE QUESTION (prereg 4.1): do the tree levels' whitened gradients of the
scale-tied fusion weight already AGREE? If yes, LO-Muon is a no-op by
construction (NS(sum G_l) == NS(sum NS(G_l)) when the G_l agree), the
design is falsified before construction, and stage 1 does not run.

    median inter-level cos( NS5(G_l), NS5(G_l') ) >= 0.95  ->  KILL
    median < 0.95                                         ->  build LO-Muon

Also reports, because they are needed either way:
  - the per-level gradient-share table (sanity against the measured
    34.1/25.3/... shares at T=256; this rung is T=1024 so counts
    differ, ordering should not), and
  - the per-level input-RMS table x_l -- the statistic the DERIVED
    weighting (w_l ~ 1/x_l) normalizes by.

Run on the CURRENT byte-rung incumbent checkpoint (Muon-trained, so
the gradient structure is that of the recipe LO will run under):

    python experiments/level_cosine.py --ckpt-dir runs_reprs/rx_muon
"""
import argparse
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.model import (OperaSpinorFenwickTree, enable_level_capture,  # noqa: E402
                            disable_level_capture, level_capture_dict)
from opera_lm.muon import zeropower_via_newtonschulz5 as NS5              # noqa: E402

CORPUS = 'assets/corpus_bytes_T1024_a20000.pkl'


def find_ckpt(d):
    c = sorted(f for f in os.listdir(d) if f.endswith('.pt'))
    assert c, f"no checkpoint in {d}"
    return os.path.join(d, c[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', default='runs_reprs/rx_muon')
    p.add_argument('--corpus', default=CORPUS)
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', default='mps')
    p.add_argument('--kill', type=float, default=0.95)
    a = p.parse_args()

    with open(a.corpus, 'rb') as f:
        d = pickle.load(f)
    if isinstance(d, tuple) and len(d) == 2 and isinstance(d[1], dict):
        d, _meta = d            # (data, meta) -- reprs.build_corpus cache
    if isinstance(d, dict):
        train, V = d['train'], d['vocab_size']
    else:
        train, V = d[0], d[3]
    T = 1024
    pool = [s for s in train if len(s) == T]
    print(f"corpus: {len(pool):,} full-length ({T}) seqs, V={V}")

    model = OperaSpinorFenwickTree(
        V, d=512, nb=128, num_layers=2, pe_mode='none', fold_mode='left',
        rot_mode='free', tie=False)
    ck = find_ckpt(a.ckpt_dir)
    model.load_state_dict(torch.load(ck, map_location='cpu',
                                     weights_only=True))
    device = torch.device(a.device)
    model = model.to(device).eval()
    print(f"model: byte d512 incumbent <- {ck}")

    torch.manual_seed(42)
    cap = enable_level_capture()
    all_cos, fold_cos, shares, rms_hist = [], [], [], cap['rms']
    g = torch.Generator().manual_seed(42)
    n_layers = model.num_layers
    for bi in range(a.batches):
        idx = torch.randint(len(pool), (a.batch,), generator=g).tolist()
        ids = torch.tensor([pool[i] for i in idx], dtype=torch.long,
                           device=device)
        lens = torch.full((a.batch,), T, dtype=torch.long, device=device)
        cap['grads'] = {}
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            lg = model(ids, lens).logits[-1]
            loss = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
            loss.backward()
        # correctness of the router itself: buffers must sum to .grad
        if bi == 0:
            for l in range(n_layers):
                name = f'fusion_gate.{l}.weight'
                w = model.fusion_gate[l].weight
                tot = sum(v for (n, _), v in cap['grads'].items() if n == name)
                rel = ((tot - w.grad).norm() / w.grad.norm()).item()
                assert rel < 1e-3, f"router mismatch {name}: {rel}"
            print("router check: per-level buffers sum to .grad (ok)")
        # per-level whitened gradients + pairwise cosines, per layer.
        # The FOLD bucket (scale-mixed applications of the same weight)
        # is reported separately: the registered median is over TREE
        # LEVEL pairs only.
        for l in range(n_layers):
            name = f'fusion_gate.{l}.weight'
            lv = sorted(i for (n, i) in cap['grads']
                        if n == name and isinstance(i, int))
            fold = cap['grads'].get((name, 'fold'))
            G = {i: cap['grads'][(name, i)] for i in lv}
            if fold is not None:
                G['fold'] = fold
            norms = {i: G[i].norm().item() for i in G}
            tot = sum(norms.values())
            shares.append((l, {i: norms[i] / tot for i in G}))
            Wd = {i: NS5(G[i].float()).flatten() for i in G}
            for x in range(len(lv)):
                for y in range(x + 1, len(lv)):
                    u, v = Wd[lv[x]], Wd[lv[y]]
                    all_cos.append((u @ v /
                                    (u.norm() * v.norm() + 1e-12)).item())
            if fold is not None:
                for i in lv:
                    u, v = Wd[i], Wd['fold']
                    fold_cos.append((u @ v /
                                     (u.norm() * v.norm() + 1e-12)).item())
    disable_level_capture()

    all_cos.sort()
    med = all_cos[len(all_cos) // 2]
    print(f"\ninter-level whitened-gradient cosines (n={len(all_cos)}): "
          f"median {med:.3f} | p10 {all_cos[len(all_cos)//10]:.3f} | "
          f"p90 {all_cos[-len(all_cos)//10]:.3f} | "
          f"min {all_cos[0]:.3f} max {all_cos[-1]:.3f}")
    if fold_cos:
        fold_cos.sort()
        fmed = fold_cos[len(fold_cos) // 2]
        print(f"fold-vs-tree-level cosines (n={len(fold_cos)}, "
              f"informational): median {fmed:.3f} | "
              f"p10 {fold_cos[len(fold_cos)//10]:.3f} | "
              f"min {fold_cos[0]:.3f} max {fold_cos[-1]:.3f}")
    print("\nper-context gradient share (fusion_gate, T=1024):")
    for l, sh in shares[:n_layers]:
        row = " ".join(f"{('fold' if i == 'fold' else f'L{i}')}:"
                       f"{sh[i]*100:5.1f}%" for i in sorted(sh, key=str))
        print(f"  layer {l}: {row}")
    print("\nper-level input RMS x_l (derived weighting normalizes by):")
    for i in sorted(rms_hist, key=str):
        print(f"  level {i}: {rms_hist[i]:.3f}")

    verdict = 'KILL' if med >= a.kill else 'PROCEED'
    print(f"\nVERDICT ({a.kill} threshold): {verdict}")
    out = {'median_cosine': med, 'n_pairs': len(all_cos),
           'p10': all_cos[len(all_cos)//10], 'p90': all_cos[-len(all_cos)//10],
           'min': all_cos[0], 'max': all_cos[-1],
           'gradient_shares': {f"layer{l}": sh for l, sh in shares},
           'input_rms': rms_hist, 'verdict': verdict, 'ckpt': ck}
    os.makedirs('runs_reprs', exist_ok=True)
    with open('runs_reprs/level_cosine.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("wrote runs_reprs/level_cosine.json")


if __name__ == '__main__':
    main()
