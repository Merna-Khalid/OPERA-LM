"""Build the matched BPE / byte corpora for the representation study.

MATCHING (the whole point -- see docs/reprs_prereg.md):
bytes are 1.00 raw bytes/unit and BPE 3.53 on this corpus (measured, not
assumed), so

    bytes  T=1024, batch 8  -> 8192 raw bytes of context per step
    bpe    T=290,  batch 8  -> 8195 raw bytes of context per step

Both arms therefore see the same amount of TEXT per step and the same
context window in TEXT, which makes step counts and bits-per-byte
directly comparable. Equal `max_len` would not be equal content.

`--max-articles` is a memory control (byte mode is ~8x more Python ints
per article than word mode); it must be identical across arms.

    python experiments/build_reprs.py --max-articles 20000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.reprs import build_corpus     # noqa: E402

# Measured on this corpus by experiments/build_reprs.py --ratios.
BYTES_PER_BPE = 3.53
CTX_BYTES = 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-articles', type=int, default=20000)
    p.add_argument('--ctx-bytes', type=int, default=CTX_BYTES)
    p.add_argument('--out-dir', default='assets')
    p.add_argument('--eval-mult', type=int, default=4,
                   help='eval_max_len = eval_mult * max_len')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plan = {
        'bytes': args.ctx_bytes,
        'bpe': int(round(args.ctx_bytes / BYTES_PER_BPE)),
    }
    print(f"context target: {args.ctx_bytes} raw bytes")
    for m, T in plan.items():
        print(f"  {m:<6} max_len={T}")

    for mode, T in plan.items():
        cache = os.path.join(args.out_dir,
                             f'corpus_{mode}_T{T}_a{args.max_articles}.pkl')
        data, meta = build_corpus(
            mode, max_len=T, eval_max_len=T * args.eval_mult,
            max_articles=args.max_articles, cache=cache)
        tr, ts, tl, V = data
        ctx = T / meta['units_per_byte']
        print(f"  -> {mode}: train {len(tr):,} test_short {len(ts):,} "
              f"test_long {len(tl):,} V={V}")
        print(f"     context {ctx:.0f} raw bytes; train units "
              f"{meta['units_train']:,}; raw train bytes "
              f"{meta['raw_bytes_train']:,}")


if __name__ == '__main__':
    main()
