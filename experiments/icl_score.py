"""In-context learning score (Olsson et al. 2022: loss at the 500th token
minus loss at the 50th), on held-out byte sequences, plus the per-position
loss profile. More negative = the model gains more from its context.
Windows (±10 positions around 50 and 500) are averaged for stability.

Usage:
  python experiments/icl_score.py --arms b2m_base b2m_hmem128 [--n-seqs 400]
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from arm_utils import OUT, load_arm, load_corpus
from opera_lm.train import make_batch_full


@torch.no_grad()
def per_position_loss(model, seqs, device, T=1024, batch=16):
    tot = torch.zeros(T - 1)
    cnt = torch.zeros(T - 1)
    for i in range(0, len(seqs), batch):
        ids, lens = make_batch_full(seqs[i:i + batch], T)
        ids, lens = ids.to(device), lens.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device != 'cpu')):
            lg = model(ids, lens, head_last_only=True).logits[-1]
        ce = F.cross_entropy(lg.float()[:, :-1].reshape(-1, lg.shape[-1]),
                             ids[:, 1:].reshape(-1), reduction='none').reshape(ids.shape[0], -1)
        valid = (torch.arange(T - 1, device=device)[None] < (lens[:, None] - 1)).float()
        tot += (ce * valid).sum(0).cpu()
        cnt += valid.sum(0).cpu()
    return (tot / cnt.clamp_min(1)).numpy() / math.log(2)          # bits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--n-seqs', type=int, default=400)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    data, _ = load_corpus()
    seqs = [s for s in data[1] if len(s) >= 1024][:args.n_seqs]
    res = {}
    for name in args.arms:
        model, _ = load_arm(name, device=args.device)
        prof = per_position_loss(model, seqs, args.device)
        l50, l500 = prof[40:61].mean(), prof[490:511].mean()
        bands = {f'{a}-{b}': float(prof[a:b].mean())
                 for a, b in ((0, 16), (16, 64), (64, 128), (128, 256),
                              (256, 512), (512, 1023))}
        res[name] = {'icl_score_bits': float(l500 - l50), 'loss_at_50': float(l50),
                     'loss_at_500': float(l500), 'bands_bits': bands}
        print(f"{name:14s} ICL score {l500 - l50:+.4f} bits (loss@50 {l50:.3f}, @500 {l500:.3f})  bands: "
              + " ".join(f"{k}:{v:.3f}" for k, v in bands.items()), flush=True)
    out = os.path.join(OUT, 'icl_' + '_'.join(args.arms) + '.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
