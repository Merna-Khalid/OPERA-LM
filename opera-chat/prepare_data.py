"""prepare_data.py -- tokenize smoltalk conversations into train/test pools.

Output pickle matches the contract opera_lm.train.train() expects for
injected data: train sequences <= max_len, test_short <= max_len,
test_long in (max_len, eval_max_len]. The conversation-level 95/5 split
happens BEFORE chunking, so chunks of one conversation stay together.

Usage:
  python prepare_data.py --smoke
  python prepare_data.py --tokenizer opera-chat/tokenizer.json \
      --out opera-chat/data_chat.pkl
"""
import argparse
import pickle
import random
from itertools import islice

from chat_common import load_tokenizer, ROLE_TOKEN, encode_ids, END_TOKEN

KEEP_ROLES = ("system", "user", "assistant")
MIN_TOKENS = 5


def turns_of(messages, tok):
    """One token-id list per message (role token + content [+ <|end|>])."""
    turns = []
    for m in messages:
        ids = [tok.token_to_id(ROLE_TOKEN[m["role"]])]
        ids.extend(encode_ids(tok, m["content"].strip()))
        if m["role"] == "assistant":
            ids.append(tok.token_to_id(END_TOKEN))
        turns.append(ids)
    return turns


def chunk_turns(turns, eval_max_len):
    """Greedy turn-boundary chunking: accumulate whole turns until the
    next turn would exceed eval_max_len, emit, continue. A single turn
    longer than eval_max_len is hard-split into eval_max_len pieces."""
    chunks = []
    cur = []
    for turn in turns:
        if len(cur) + len(turn) <= eval_max_len:
            cur.extend(turn)
            continue
        if cur:
            chunks.append(cur)
        if len(turn) > eval_max_len:
            for i in range(0, len(turn), eval_max_len):
                chunks.append(turn[i:i + eval_max_len])
            cur = []
        else:
            cur = list(turn)
    if cur:
        chunks.append(cur)
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="opera-chat/tokenizer.json")
    p.add_argument("--out", default="opera-chat/data_chat.pkl")
    p.add_argument("--n-convs", type=int, default=150000)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-max-len", type=int, default=1024)
    p.add_argument("--smoke", action="store_true",
                   help="2000 conversations, streaming")
    a = p.parse_args()
    n_convs = 2000 if a.smoke else a.n_convs

    tok = load_tokenizer(a.tokenizer)
    vocab_size = tok.get_vocab_size()

    from datasets import load_dataset
    if a.smoke:
        ds = islice(load_dataset("HuggingFaceTB/smoltalk", "all",
                                 split="train", streaming=True), n_convs)
    else:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")

    rng = random.Random(42)
    train, test_short, test_long = [], [], []
    seen = dropped = 0
    for ex in ds:
        if seen >= n_convs:
            break
        msgs = [m for m in ex["messages"] if m["role"] in KEEP_ROLES]
        if not msgs:
            continue
        seen += 1
        turns = turns_of(msgs, tok)
        total = sum(len(t) for t in turns)
        if total < MIN_TOKENS:
            dropped += 1
            continue
        is_test = rng.random() < 0.05        # split BEFORE chunking
        chunks = ([ [t for turn in turns for t in turn] ]
                  if total <= a.eval_max_len
                  else chunk_turns(turns, a.eval_max_len))
        for c in chunks:
            if len(c) < MIN_TOKENS:
                continue
            if is_test:
                (test_short if len(c) <= a.max_len else test_long).append(c)
            elif len(c) <= a.max_len:
                train.append(c)
        if seen % 10000 == 0:
            print(f"  {seen} convs: train={len(train)} "
                  f"short={len(test_short)} long={len(test_long)}",
                  flush=True)

    out = {"train": train, "test_short": test_short,
           "test_long": test_long, "vocab_size": vocab_size,
           "special": {name: tok.token_to_id(name) for name in
                       ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>",
                        "<|system|>"]}}
    with open(a.out, "wb") as f:
        pickle.dump(out, f)

    print(f"saved {a.out}", flush=True)
    print(f"conversations seen: {seen} (dropped <{MIN_TOKENS} tokens: "
          f"{dropped})", flush=True)
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
    # HF streaming leaves downloader threads alive after main() returns
    # (observed: process hangs for minutes after the pkl is saved).
    # Exit hard so chained pipelines (run_full.sh) proceed.
    import os, sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
