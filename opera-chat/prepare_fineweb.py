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

  # optional SmolLM2-style mix (all three expose a "text" field, so the
  # tokenize/chunk loop below is unchanged either way):
  python prepare_fineweb.py --mix fineweb,finemath,stackedu \
      --mix-weights 0.85,0.10,0.05 --max-tokens 500000000
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
    p.add_argument("--mix", default=None,
                   help="comma-separated source names to interleave "
                        "instead of plain FineWeb-Edu, e.g. "
                        "'fineweb,finemath,stackedu' (choices: fineweb, "
                        "finemath, stackedu)")
    p.add_argument("--mix-weights", default=None,
                   help="comma-separated sampling weights matching "
                        "--mix, e.g. '0.85,0.10,0.05' (default: equal)")
    p.add_argument("--finemath-config", default="finemath-3plus",
                   help="HuggingFaceTB/finemath config name")
    p.add_argument("--stackedu-lang", default="Python",
                   help="HuggingFaceTB/stack-edu language config")
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

    if a.mix:
        from datasets import interleave_datasets
        sources = [s.strip() for s in a.mix.split(",")]
        weights = ([float(w) for w in a.mix_weights.split(",")]
                   if a.mix_weights else [1.0 / len(sources)] * len(sources))
        assert len(sources) == len(weights), \
            "--mix and --mix-weights must have the same length"
        # all three expose a "text" field, so the tokenize/chunk loop
        # below needs no per-source special-casing
        source_loaders = {
            "fineweb": lambda: load_dataset(
                "HuggingFaceFW/fineweb-edu", name=a.fw_config,
                split="train", streaming=True),
            "finemath": lambda: load_dataset(
                "HuggingFaceTB/finemath", name=a.finemath_config,
                split="train", streaming=True),
            "stackedu": lambda: load_dataset(
                "HuggingFaceTB/stack-edu", name=a.stackedu_lang,
                split="train", streaming=True),
        }
        unknown = set(sources) - set(source_loaders)
        assert not unknown, \
            f"unknown --mix source(s) {unknown}, choose from {list(source_loaders)}"
        ds = interleave_datasets([source_loaders[s]() for s in sources],
                                 probabilities=weights, seed=42,
                                 stopping_strategy="all_exhausted")
        print(f"  mixing: {dict(zip(sources, weights))}", flush=True)
    else:
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
