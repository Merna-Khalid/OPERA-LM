"""Compare repr_study arms against a same-session control.

Reads runs_reprs/repr_summary.json (in-length BPB, params, minutes) and
each arm's results.jsonl (extrapolation buckets), and prints every arm's
delta vs the control. Exploratory read-out only -- no gates.

Noise reference (same incumbent recipe, different sessions):
rx_fgate_wd 2.0371 / lc_off 2.0406 / relaxM_off 2.0411 BPB -> ~0.2%
spread, so single-run deltas under ~0.3-0.5% are not interpretable.

Usage:
  python experiments/compare_arms.py --control st_off \
      --arms st_head x_sam x_nmix st_auxw0 st_wide x_tie2
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'runs_reprs')


def extrap_of(arm_dir):
    """Last results.jsonl row's extrapolation buckets {bucket: ppl}."""
    path = os.path.join(arm_dir, 'opera_v8_0_results.jsonl')
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[-1].get('extrapolation', {}) if rows else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', default='st_off')
    p.add_argument('--arms', nargs='+', required=True)
    args = p.parse_args()
    summ = json.load(open(os.path.join(OUT, 'repr_summary.json')))
    if args.control not in summ:
        raise SystemExit(f"control {args.control} not finished yet")
    c = summ[args.control]
    c_ext = extrap_of(os.path.join(OUT, args.control))
    buckets = sorted(c_ext, key=lambda b: int(str(b).split('-')[0])
                     if str(b).split('-')[0].isdigit() else 0)
    hdr = (f"{'arm':12s} {'params':>10s} {'BPB':>7s} {'dBPB%':>7s} "
           f"{'min':>5s}  " + "  ".join(f"ext{b}" for b in buckets))
    print(hdr)
    print('-' * len(hdr))
    for name in [args.control] + args.arms:
        if name not in summ:
            print(f"{name:12s} (not finished)")
            continue
        r = summ[name]
        d = 100.0 * (r['bpb'] / c['bpb'] - 1.0)
        ext = extrap_of(os.path.join(OUT, name))
        ecols = []
        for b in buckets:
            if b in ext and b in c_ext:
                ecols.append(f"{ext[b]:.3f}({100 * (ext[b] / c_ext[b] - 1):+.1f}%)")
            else:
                ecols.append('-')
        print(f"{name:12s} {r['params']:>10,} {r['bpb']:7.4f} {d:+7.2f} "
              f"{r['minutes']:5.0f}  " + "  ".join(ecols))


if __name__ == '__main__':
    main()
