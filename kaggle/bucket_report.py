"""bucket_report.py -- prereg §5.2: report the extrapolation-bucket and
length-band populations of the Path A eval pools BEFORE any training, so
thin buckets are known (and marked in every later table) rather than
discovered after the fact.
"""
import argparse
import json
import pickle
from collections import Counter


def bucket_counts(seqs, max_len, eval_max):
    c = Counter()
    for s in seqs:
        n = min(len(s), eval_max)
        if n <= max_len:
            continue
        lo = max_len + 1
        while lo <= eval_max:
            hi = min(lo + max_len - 1, eval_max)
            if lo <= n <= hi:
                c[f"{lo}-{hi}"] += 1
                break
            lo += max_len
    return dict(sorted(c.items(), key=lambda kv: int(kv[0].split("-")[0])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoltalk", required=True, help="eval pkl path")
    p.add_argument("--fineweb", required=True, help="eval pkl path")
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--eval-max-len", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    report = {"max_len": a.max_len, "eval_max_len": a.eval_max_len}
    for name, path in (("smoltalk", a.smoltalk), ("fineweb", a.fineweb)):
        with open(path, "rb") as f:
            d = pickle.load(f)
        tl = d["test_long"]
        lens = [min(len(s), a.eval_max_len) for s in tl]
        reach = {f">= {k}": sum(1 for n in lens if n >= k)
                 for k in (1024, 2048, 4096, 8192)}
        report[name] = {
            "test_long_seqs": len(tl),
            "test_short_seqs": len(d["test_short"]),
            "buckets": bucket_counts(tl, a.max_len, a.eval_max_len),
            "threshold_note": "buckets with < 50 seqs are marked thin "
                              "and excluded from gate arithmetic (prereg §5.2)",
            "thin_buckets": {k: v for k, v in
                             bucket_counts(tl, a.max_len, a.eval_max_len).items()
                             if v < 50},
            "reach": reach,
        }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
