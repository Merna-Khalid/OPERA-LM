"""train_tokenizer.py -- train the byte-level BPE for the chat pipeline.

Streams HuggingFaceTB/smoltalk (config "all", split "train") and yields
each message's text as a training line, capped at --docs lines. Special
tokens come first in the vocab, so their ids are 0..4 (see chat_common).

Usage:
  python train_tokenizer.py --smoke                       # quick check
  python train_tokenizer.py --out opera-chat/tokenizer.json
"""
import argparse
import os

from chat_common import SPECIAL_TOKENS

SMOKE_DOCS = 30000


def line_iter(docs):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train",
                      streaming=True)
    n = 0
    for ex in ds:
        for m in ex["messages"]:
            txt = m["content"].strip()
            if not txt:
                continue
            yield txt
            n += 1
            if n >= docs:
                return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="opera-chat/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--docs", type=int, default=2000000)
    p.add_argument("--smoke", action="store_true",
                   help=f"cap at {SMOKE_DOCS} lines")
    a = p.parse_args()
    docs = SMOKE_DOCS if a.smoke else a.docs

    from tokenizers import ByteLevelBPETokenizer
    print(f"training byte-level BPE: vocab={a.vocab_size}, "
          f"docs<={docs} (streaming smoltalk)", flush=True)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(line_iter(docs), vocab_size=a.vocab_size,
                            min_frequency=2,
                            special_tokens=SPECIAL_TOKENS)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)

    print(f"saved {a.out}", flush=True)
    print(f"vocab size: {tok.get_vocab_size()}", flush=True)
    # specials must land at ids 0..4 (pad first -- opera_lm zero-pads)
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i, f"{name} id != {i}"
    # byte-level BPE decode round-trips whitespace; verify on samples
    for s in ["Hello, how are you today?",
              "  indented   and\nmultiline text.",
              "The quick brown fox jumps over 13 lazy dogs!"]:
        rt = tok.decode(tok.encode(s).ids)
        assert rt == s, f"roundtrip failed: {s!r} -> {rt!r}"
    print("roundtrip sanity check: OK (3/3)", flush=True)


if __name__ == "__main__":
    main()
    # HF streaming leaves downloader threads alive after main() returns
    # (observed: process hangs for minutes after the artifacts are saved).
    # The tokenizer is on disk at this point; exit hard so chained
    # pipelines (run_full.sh) proceed.
    import os, sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
