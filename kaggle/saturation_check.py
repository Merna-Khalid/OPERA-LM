"""saturation_check.py -- prereg §5.1: does a 2M-line-trained BPE beat
the incumbent 500k-line one at 16,384 vocab, or is the corpus saturated?

Samples --convs conversations from the smoltalk stream (in-distribution
sample, see prereg §5.1 wording), tokenizes each message with both
tokenizers, and writes {"adopt_new": bool, ...} per the pre-committed
rule: adopt the new tokenizer only if median tokens/conversation
improves (i.e., drops) by >= --threshold (default 1%).
"""
import argparse
import json
import statistics
from itertools import islice


def convo_iter(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train",
                      streaming=True)
    for ex in islice(ds, n):
        text = "\n".join(m["content"] for m in ex["messages"]
                         if m.get("content"))
        if text.strip():
            yield text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--convs", type=int, default=1000)
    p.add_argument("--threshold", type=float, default=0.01)
    a = p.parse_args()

    from tokenizers import Tokenizer
    old_t, new_t = Tokenizer.from_file(a.old), Tokenizer.from_file(a.new)
    assert old_t.get_vocab_size() == new_t.get_vocab_size(), \
        "vocab sizes differ -- adoption rule assumes matched vocab"
    n_old, n_new = [], []
    for text in convo_iter(a.convs):
        n_old.append(len(old_t.encode(text).ids))
        n_new.append(len(new_t.encode(text).ids))

    med_old, med_new = statistics.median(n_old), statistics.median(n_new)
    tot_old, tot_new = sum(n_old), sum(n_new)
    improve = (med_old - med_new) / med_old
    res = {
        "n_convs": len(n_old),
        "median_tokens_old": med_old, "median_tokens_new": med_new,
        "total_tokens_old": tot_old, "total_tokens_new": tot_new,
        "median_improvement_frac": round(improve, 5),
        "threshold": a.threshold,
        "adopt_new": bool(improve >= a.threshold),
        "rule": "adopt new iff median tokens/conversation improves >= 1%",
        "sample": "first smoltalk conversations, in-distribution (prereg §5.1)",
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
