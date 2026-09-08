"""H4: tokenization invariance on digit strings.

THE IDEA, AND WHERE IT COMES FROM
---------------------------------
The Kirby-calculus intuition (2026-09-08) is that meaning is not a
representation but an *equivalence class* of representations under some
set of moves. The calculus itself has no referent in OPERA -- there is
no base manifold and no connection problem -- but this much is concrete
and testable: **a model should not care how a string was cut up.**

BPE cares. Segmentation of digit runs is essentially arbitrary: "4839"
may merge as [48][39] while "4831" goes [4][831], depending on which
pairs the merge table happened to learn on its training corpus. Two
strings of identical structure and identical difficulty therefore get
different unit counts, different context consumption, and different
per-character cost. Byte-level has no such freedom -- one string, one
encoding -- so the invariance is not learned, it is structural.

WHAT IS MEASURED
----------------
For random digit strings of fixed length embedded in a fixed carrier
sentence, the **bits per character over the digit span**:

    bits(span) = sum over units overlapping the span of -log2 p(unit)

summed over units and divided by the number of CHARACTERS, so the
quantity is comparable across tokenizations (the same normalisation
argument as BPB). Then:

  * mean  -- how expensive digits are for this model;
  * SD across strings -- **the invariance measure**. Every string is
    drawn from the same uniform distribution over digits, so all of them
    are equally hard in principle. Residual spread is the model's
    sensitivity to how the string happened to be cut.

Prediction (pre-registered in docs/OPERA_Repr_prereg.md §4 H4, no gate):
the byte arm shows lower SD. Reported as a first measurement, not a
decision -- at 0.9M parameters and 3000 steps neither model can actually
do arithmetic, and this probe does not claim otherwise. It measures
representational stability, not competence.

    python experiments/invariance_probe.py
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.garden_path import load_opera                 # noqa: E402
from opera_lm.gp_benchmark import subword_encode            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
CARRIER = "The number is {}."


@torch.no_grad()
def span_bits_per_char(model, sentence, span, repr_mode, tok):
    """Bits spent on the characters of `span`=(c0,c1), normalised per
    character so byte and BPE arms are directly comparable."""
    ids, offs = subword_encode(sentence, repr_mode, tok)
    t = torch.tensor([ids])
    lp = F.log_softmax(
        model(t, torch.tensor([len(ids)])).logits[-1].float(), dim=-1)
    # unit j is predicted from prefix j-1, so its cost is at lp[0, j-1]
    c0, c1 = span
    bits = 0.0
    for j, (a, b) in enumerate(offs):
        if j == 0 or a >= c1 or b <= c0:
            continue
        bits += -lp[0, j - 1, t[0, j]].item() / math.log(2)
    return bits / (c1 - c0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--digits', type=int, default=7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default=os.path.join(ROOT, 'runs_reprs',
                                                  'invariance.json'))
    args = p.parse_args()

    rng = random.Random(args.seed)
    strings = [''.join(rng.choice('0123456789')
                       for _ in range(args.digits))
               for _ in range(args.n)]

    arms = {
        'bytes_d256': dict(repr_mode='bytes', vocab_size=259, d=256, nb=64,
                           ckpt='runs_reprs/opera_v8_0_pe-none_rotfree_'
                                'amp_foreach_cmp-default_gpudata_aux0.25.pt'),
        'bpe_d256': dict(repr_mode='bpe', vocab_size=16384, d=256, nb=64,
                         ckpt=None),
    }
    # Resolve checkpoints by VOCAB SIZE, scanning runs_reprs recursively.
    # Recursive because train()'s checkpoint name is built from the config
    # tag, which does not encode vocab_size or max_len -- so arms are kept
    # in per-arm subdirectories to stop them overwriting each other.
    # Matching on the embedding's vocab dimension is unambiguous here
    # (259 vs 16384) and survives any future renaming.
    import glob
    cks = sorted(glob.glob(os.path.join(ROOT, 'runs_reprs', '**', '*.pt'),
                           recursive=True))
    cks = [c for c in cks if 'train_ckpt' not in c]

    out = {}
    for name, cfg in arms.items():
        V = cfg['vocab_size']
        match = [c for c in cks
                 if torch.load(c, map_location='cpu', weights_only=False)
                 ['word_emb.weight'].shape[0] == V]
        if not match:
            print(f"  [{name}] no checkpoint with vocab {V} in runs_reprs; "
                  f"skipping")
            continue
        ckpt = match[0]
        tok = None
        if cfg['repr_mode'] == 'bpe':
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(
                os.path.join(ROOT, 'opera-chat', 'tokenizer.json'))
        model = load_opera(ckpt, vocab_size=V, d=cfg['d'], nb=cfg['nb'],
                           layers=2, rot_mode='free')
        vals, ntoks = [], []
        for s in strings:
            sent = CARRIER.format(s)
            c0 = sent.index(s)
            vals.append(span_bits_per_char(model, sent, (c0, c0 + len(s)),
                                           cfg['repr_mode'], tok))
            ids, offs = subword_encode(sent, cfg['repr_mode'], tok)
            ntoks.append(sum(1 for a, b in offs
                             if a < c0 + len(s) and b > c0))
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        tm = sum(ntoks) / len(ntoks)
        tsd = math.sqrt(sum((v - tm) ** 2 for v in ntoks)
                        / (len(ntoks) - 1))
        out[name] = dict(ckpt=os.path.basename(ckpt), mean_bpc=m, sd_bpc=sd,
                         cv=sd / m, mean_units=tm, sd_units=tsd,
                         n=len(vals), digits=args.digits)
        print(f"  {name:<11} bits/char {m:6.3f} ± {sd:.3f}  "
              f"(CV {sd / m:.4f})   units/span {tm:.2f} ± {tsd:.2f}")

    if len(out) == 2:
        a, b = out['bytes_d256'], out['bpe_d256']
        print(f"\n  unit-count spread (the segmentation freedom itself): "
              f"bytes SD {a['sd_units']:.2f} vs bpe SD {b['sd_units']:.2f}")
        print(f"  cost spread (the consequence):  bytes CV {a['cv']:.4f} "
              f"vs bpe CV {b['cv']:.4f}")
        print(f"  -> byte arm is {'MORE' if a['cv'] < b['cv'] else 'LESS'} "
              f"invariant on this probe")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.json}")


if __name__ == '__main__':
    main()
