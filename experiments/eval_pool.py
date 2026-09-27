"""Score finished arms on BOTH test sets (Recall research §8b):

  a20k   the unchanged a20000 test set (first 5,000 in-length chunks),
         the number repr_study reports -- recomputed here as a check;
  pool   a seeded uniform sample of 5,000 in-length test chunks from
         every test-split article in the full corpus
         (experiments/build_packed.py -> <pool>.test.pkl);
  new    the pool sample restricted to articles past the first 20,000
         (text no a20000-trained model has a test set for).

Bytes are 1 unit each, so BPB = ln(PPL) / ln 2. Also reports the
1025-2048 extrapolation bucket on the pool's long chunks.

  PYTHONPATH=.. python eval_pool.py --arms b2_long w768_hmem192_gated_full
"""
import argparse
import json
import math
import os
import pickle

import torch

from arm_utils import OUT, ROOT, load_arm, load_corpus
from opera_lm.train import _eval_batch, _oom_backstop, compute_perplexity


def bpb(model, seqs, T, device):
    ppl = _oom_backstop(lambda b: compute_perplexity(model, seqs, T, b, device),
                        _eval_batch(32, T), device)
    return math.log(ppl) / math.log(2)


def test_sets(pool, T=1024, extra=None):
    """extra: another .test.pkl (e.g. FineWeb-Edu, build_fineweb_bytes.py),
    added as 'fineweb' / 'fineweb_long_1025_2048'."""
    (_, a20k_short, _, _), _ = load_corpus(T)
    pt = pickle.load(open(os.path.join(ROOT, pool + '.test.pkl'), 'rb'))
    sets = {'a20k': a20k_short[:5000],
            'pool': [c for _, c in pt['test_short']],
            'new': [c for a, c in pt['test_short'] if a >= 20000],
            'long_1025_2048': [c for _, c in pt['test_long'] if T < len(c) <= 2 * T]}
    if extra:
        ft = pickle.load(open(extra, 'rb'))
        sets['fineweb'] = [c for _, c in ft['test_short']]
        sets['fineweb_long_1025_2048'] = [c for _, c in ft['test_long']
                                          if T < len(c) <= 2 * T]
    return sets


@torch.no_grad()
def score_arm(name, sets, device, T=1024):
    model, _ = load_arm(name, device=device)
    r = {k: bpb(model, v, 2 * T if k.startswith('long') else T, device)
         for k, v in sets.items()}
    del model
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--pool', default='assets/packed_bytes_T1024_all')
    p.add_argument('--device', default='mps')
    p.add_argument('--T', type=int, default=1024)
    p.add_argument('--extra-test', default=None,
                   help='another .test.pkl, e.g. the FineWeb-Edu one')
    args = p.parse_args()
    sets = test_sets(args.pool, args.T, args.extra_test)
    print("test sets: " + "  ".join(f"{k} {len(v)}" for k, v in sets.items()), flush=True)

    out_path = os.path.join(OUT, 'pool_eval.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name in args.arms:
        r = res[name] = score_arm(name, sets, args.device, args.T)
        print(f"{name:28s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()),
              flush=True)
        json.dump(res, open(out_path, 'w'), indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
