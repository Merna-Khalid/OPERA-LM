"""pack_data.py -- split a prepare_*.py output pkl into an eval-only pkl
plus a packed (mmap-able) train pool for --packed-data.

Input pkl (prepare_data.py / prepare_fineweb.py contract):
    {"train": [...], "test_short": [...], "test_long": [...],
     "vocab_size": int, ["special": {...}]}

Outputs (prefix P, max_len L):
    P.eval.pkl         same dict with train=[] (fast session-start load)
    P.tokens.npy       int32 flat ids (sequences truncated to L)
    P.offsets.npy      int64 [N+1]
    P.pack.json        stats

Usage:
  python kaggle/pack_data.py --pkl data_fineweb_512.pkl \
      --prefix data_fineweb_512 --max-len 512
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opera_lm.packed import pack_pool, packed_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--max-len", type=int, required=True)
    a = p.parse_args()

    with open(a.pkl, "rb") as f:
        d = pickle.load(f)
    train = d["train"]
    assert train, f"{a.pkl} has an empty train list (already packed?)"
    n_tok = sum(min(len(s), a.max_len) for s in train)
    stats = pack_pool(train, a.prefix, a.max_len)
    assert stats["total_tokens"] == n_tok

    d["train"] = []
    with open(f"{a.prefix}.eval.pkl", "wb") as f:
        pickle.dump(d, f)
    stats["source_pkl"] = a.pkl
    stats["eval_pkl"] = f"{a.prefix}.eval.pkl"
    with open(f"{a.prefix}.pack.json", "w") as f:
        json.dump(stats, f, indent=2)

    hdr = packed_stats(a.prefix)
    print(f"packed {hdr['n_seqs']:,} seqs / {hdr['total_tokens']:,} tokens "
          f"(max_len {a.max_len}) -> {a.prefix}.[tokens|offsets].npy")
    print(f"eval-only pkl (train=[]) -> {a.prefix}.eval.pkl")


if __name__ == "__main__":
    main()
