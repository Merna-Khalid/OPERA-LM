"""Builds a concatenation-augmented training set for a short length-
generalization fine-tune phase (State Passing, Buitrago Ruiz & Gu 2025,
arXiv:2507.02782 -- the intervention that beat noise-based state init).

train_data in the source pkl is capped at max_len (mean length well below
that), so there is almost no real content past training length to learn
from. Concatenating several UNRELATED training sequences end-to-end
synthesizes genuine deep compose-node inputs (Fenwick tree spans beyond
max_len) using only real tokens, without needing naturally long documents
-- the same trick behind document packing in standard LM pretraining, here
specifically targeting OPERA's weight-tied compose function's unexplored
deep-recursion states.

test_short/test_long/vocab_size pass through unchanged: test_long already
has genuine long conversations spanning the eval buckets used by the
existing length-extrapolation report, so no eval-side changes needed.

Usage:
  python prep_statepassing_data.py --src data_chat.pkl \
      --dst data_chat_statepassing.pkl
  python prep_statepassing_data.py --src data_chat_full.pkl \
      --dst data_chat_full_statepassing.pkl --n-synth 50000 --max-len 1024
"""
import argparse
import pickle
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data_chat.pkl")
    p.add_argument("--dst", default="data_chat_statepassing.pkl")
    p.add_argument("--n-synth", type=int, default=20000,
                   help="concatenated examples to generate")
    p.add_argument("--max-len", type=int, default=1024,
                   help="cap, should match the base run's eval_max_len")
    p.add_argument("--min-chain", type=int, default=2)
    p.add_argument("--max-chain", type=int, default=6,
                   help="sequences chained per synthetic example")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed)
    with open(a.src, "rb") as f:
        d = pickle.load(f)
    train = d["train"]

    synth = []
    for _ in range(a.n_synth):
        k = random.randint(a.min_chain, a.max_chain)
        seq = []
        for _ in range(k):
            seq.extend(random.choice(train))
            if len(seq) >= a.max_len:
                break
        synth.append(seq[:a.max_len])

    lens = [len(s) for s in synth]
    print(f"synthesized {len(synth)} concatenated examples")
    print(f"lens: min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.1f}")
    step = a.max_len // 4
    for lo in range(0, a.max_len, step):
        hi = lo + step
        c = sum(1 for l in lens if lo < l <= hi or (lo == 0 and l == 0))
        print(f"  bucket ({lo},{hi}): {c} ({100*c/len(lens):.1f}%)")

    out = {"train": synth, "test_short": d["test_short"],
           "test_long": d["test_long"], "vocab_size": d["vocab_size"]}
    with open(a.dst, "wb") as f:
        pickle.dump(out, f)
    print(f"saved -> {a.dst}")


if __name__ == "__main__":
    main()
