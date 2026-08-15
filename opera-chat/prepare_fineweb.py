"""prepare_fineweb.py -- BPE-tokenize a FineWeb-Edu slice for Stage 2's
raw-text pretraining phase, using the SAME tokenizer.json as the chat
pipeline (so the resulting checkpoint can be continued into smoltalk SFT
via train_chat.py's --init-weights-from, same mechanism as the existing
state-passing fine-tune).

Output pickle matches train()'s injected-data contract, same as
prepare_data.py: train sequences <= max_len, test_short <= max_len,
test_long in (max_len, eval_max_len]. Chunking reuses opera_lm.data's
existing doc_chunks (the same deterministic schedule the library's own
Wikipedia "docs-en" pretraining mode uses), long_first=True so long-
document buckets are populated by construction rather than requiring
articles many multiples of train_max_len long. Split hygiene: the
train/test draw happens per DOCUMENT before chunking, so consecutive
chunks of one document never straddle the split.

Usage:
  python prepare_fineweb.py --smoke
  python prepare_fineweb.py --tokenizer opera-chat/tokenizer.json \
      --out data_fineweb.pkl --max-tokens 500000000
"""
import argparse
import pickle
import random

from chat_common import load_tokenizer, encode_ids, ensure_opera_lm

ensure_opera_lm()

MIN_TOKENS = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="opera-chat/tokenizer.json")
    p.add_argument("--out", default="data_fineweb.pkl")
    p.add_argument("--fw-config", default="sample-10BT",
                   help="HuggingFaceFW/fineweb-edu config name")
    p.add_argument("--max-tokens", type=int, default=500_000_000,
                   help="stop after streaming ~this many BPE tokens "
                        "(500M is a deliberately modest default for a "
                        "single Colab session, not the full 10B-token "
                        "sample -- raise it explicitly for a bigger run)")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-max-len", type=int, default=2048)
    p.add_argument("--smoke", action="store_true",
                   help="cap at 5M tokens, streaming")
    a = p.parse_args()
    max_tokens = 5_000_000 if a.smoke else a.max_tokens

    tok = load_tokenizer(a.tokenizer)
    vocab_size = tok.get_vocab_size()

    from datasets import load_dataset
    from opera_lm.data import doc_chunks
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=a.fw_config,
                      split="train", streaming=True)

    rng = random.Random(42)
    cap = a.eval_max_len
    train, test_short, test_long = [], [], []
    seen_docs = 0
    total_tokens = 0
    for doc in ds:
        text = doc.get("text", "").strip()
        if not text:
            continue
        ids = encode_ids(tok, text)
        seen_docs += 1
        total_tokens += len(ids)
        if len(ids) >= MIN_TOKENS:
            is_test = rng.random() < 0.05      # split BEFORE chunking
            for c in doc_chunks(ids, a.max_len, cap, long_first=True):
                if len(c) < MIN_TOKENS:
                    continue
                if is_test:
                    (test_short if len(c) <= a.max_len else test_long).append(c)
                elif len(c) <= a.max_len:
                    train.append(c)
        if seen_docs % 5000 == 0:
            print(f"  {seen_docs} docs, ~{total_tokens:,} tokens seen: "
                  f"train={len(train)} short={len(test_short)} "
                  f"long={len(test_long)}", flush=True)
        if total_tokens >= max_tokens:
            break

    out = {"train": train, "test_short": test_short,
           "test_long": test_long, "vocab_size": vocab_size}
    with open(a.out, "wb") as f:
        pickle.dump(out, f)

    print(f"saved {a.out}", flush=True)
    print(f"docs seen: {seen_docs}, ~{total_tokens:,} BPE tokens", flush=True)
    for name, pool in [("train", train), ("test_short", test_short),
                       ("test_long", test_long)]:
        if not pool:
            print(f"  {name}: 0 sequences  <-- WARNING: empty", flush=True)
            continue
        lens = [len(s) for s in pool]
        print(f"  {name}: {len(pool)} seqs, {sum(lens)} tokens, "
              f"len mean={sum(lens)/len(lens):.1f} "
              f"min={min(lens)} max={max(lens)}", flush=True)


if __name__ == "__main__":
    main()
    import os, sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
