"""Compute-optimal scaling ladder for the gated-memory OPERA model
(Recall research §8d).

Each rung is a model fw_d{d}_L{L} trained on RATIO bytes per parameter
(default 20, the Chinchilla tokens-per-parameter rule applied to bytes)
from the FineWeb-Edu byte pool, then scored on the FineWeb held-out set
(and the Simple-Wikipedia sets). The fit of loss against training compute
C = 6 N D is the result: its slope says whether more compute is worth
spending on this architecture.

  python experiments/ladder.py plan --pool <prefix> [--bench runs/bench.jsonl]
  python experiments/ladder.py run  --pool <prefix> [--rungs 512x2 768x2 ...]
  python experiments/ladder.py fit  [--out ladder_fit.png]

Default rungs: a width ladder at L=2 (5M, 11M, 19M, 43M, 77M params) and
a depth check, d1024 x L4 (37M), next to d1536 x L2 (43M). All rungs use
the same batch, recipe and seed; learning-rate schedules scale with each
rung's own step count (cosine, 500 warmup steps).

--model selects the architecture trained at each rung:
  opera    (default) OperaSpinorFenwickTree via experiments/repr_study.py.
  tf_byte  matched transformer baseline (opera-chat/opera_transformer_baseline_v2.py,
           pe=nope -- the direct comparison arm for OPERA's pe-none) on the
           SAME byte --pool as the opera ladder, via experiments/train_tf_pool.py.
  tf_tok   the same transformer baseline, pe=rope (the standard strong
           setup), on a BPE token --pool (build_fineweb_tokens.py); vocab_size
           and the BPB conversion factor are read from that pool's meta.json.
Each model's rows land in the same repr_summary.json under a model-specific
arm prefix (fw_ / tfb_ / tft_), so `fit --model <m>` fits/plots that one
family. The transformer arms are the LADDER ONLY (5M-77M) -- intentionally
no long run at the OPERA demo model's ~172M scale (fw_d3072_L2), since that
single rung already costs ~18h and a baseline only needs the ladder's trend.
"""
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RUNS = os.environ.get('OPERA_RUNS', os.path.join(ROOT, 'runs_reprs'))
DEFAULT_RUNGS = ['512x2', '768x2', '1024x2', '1536x2', '2048x2', '1024x4']
T = 1024
ARM_PREFIX = {'opera': 'fw_', 'tf_byte': 'tfb_', 'tf_tok': 'tft_'}


def arm_name(d, L, ratio, model='opera', suffix=''):
    return f"{ARM_PREFIX[model]}d{d}_L{L}_r{ratio:g}{suffix}"


def tf_arm_suffix(args):
    """Distinguishes a qk_norm/attn_logit_cap re-run's arm names from an
    already-written, unfixed tfb_/tft_ entry in repr_summary.json -- so a
    controlled before/after comparison (docs/OPERA_Mechanisms_Guide.md §8:
    one variable per run, same-session incumbent) keeps BOTH rows instead
    of the fixed run silently overwriting the broken one under the same
    name."""
    s = ''
    if getattr(args, 'qk_norm', False):
        s += '_qknorm'
    cap = getattr(args, 'attn_logit_cap', 0.0)
    if cap:
        s += f'_cap{cap:g}'
    return s


def n_params(d, L, model='opera', vocab_size=259):
    import torch
    if model == 'opera':
        from opera_lm import OperaSpinorFenwickTree
        kw = dict(vocab_size=259, d=d, nb=d // 4, num_layers=L, pe_mode='none',
                  fold_mode='left', rot_mode='free', head_mode='stream',
                  fold_impl='downsweep', hmem_nb=d // 4, hmem_decay='gated')
        if L > 2:
            kw.update(resid_mode='add', resid_init_scale='auto')
        with torch.device('meta'):
            m = OperaSpinorFenwickTree(**kw)
        return sum(p.numel() for p in m.parameters())
    sys.path.insert(0, os.path.join(ROOT, 'opera-chat'))
    from opera_transformer_baseline_v2 import TransformerBaseline
    nheads = max(1, d // 64)
    with torch.device('meta'):
        m = TransformerBaseline(vocab_size, d=d, nheads=nheads,
                                num_layers=L, tie=False)
    return sum(p.numel() for p in m.parameters())


def mean_seq_len(pool):
    """Average training-sequence length of the pool (chunks are <= T; a
    document's trailing chunk is shorter), so bytes seen = steps*B*this."""
    meta = json.load(open(pool + '.meta.json'))
    return meta['total_tokens'] / meta['n_seqs']


def load_bench(path):
    rows = []
    if path and os.path.exists(path):
        for line in open(path):
            rows.append(json.loads(line))
    return rows


def est_ms(bench, d, L, B, params):
    """ms/step from the A100 benchmark: the exact (d, L, B) if measured,
    else the nearest measured size with the same L and B scaled linearly in
    parameters (per-step compute ~ params at fixed tokens/step)."""
    same = [r for r in bench if r['L'] == L and r['B'] == B]
    for r in same:
        if r['d'] == d:
            return r['ms'], 'measured'
    pool = same or [r for r in bench if r['B'] == B] or bench
    if not pool:
        return None, 'no benchmark'
    r = min(pool, key=lambda r: abs(math.log(r['params'] / params)))
    return r['ms'] * params / r['params'] * B / r['B'], f"scaled from d{r['d']} L{r['L']} B{r['B']}"


def plan(args):
    model = getattr(args, 'model', 'opera')
    msl = mean_seq_len(args.pool)
    pm = json.load(open(args.pool + '.meta.json'))
    vocab_size = pm.get('vocab_size', 259) if model != 'opera' else 259
    units_per_byte = pm.get('units_per_byte', 1.0) or 1.0
    bench = load_bench(args.bench)
    suffix = tf_arm_suffix(args) if model != 'opera' else ''
    rows, tot_h, tot_bytes = [], 0.0, 0
    for g in args.rungs:
        d, L = (int(x) for x in g.split('x'))
        N = n_params(d, L, model=model, vocab_size=vocab_size)
        ms, how = est_ms(bench, d, L, args.batch, N)
        if args.hours:
            # fixed time budget: as many bytes as the hours allow
            assert ms, '--hours needs the benchmark (bench.jsonl)'
            steps = int(args.hours * 3.6e6 / ms)
            D = steps * args.batch * msl
            ratio = round(D / N)
        else:
            ratio = args.ratio
            D = ratio * N
            steps = math.ceil(D / (args.batch * msl))
        h = steps * ms / 3.6e6 if ms else None
        tot_h += h or 0
        tot_bytes += D
        rows.append(dict(arm=arm_name(d, L, ratio, model=model, suffix=suffix),
                         d=d, L=L, params=N, bytes=D, steps=steps, hours=h,
                         est=how, vocab_size=vocab_size,
                         units_per_byte=units_per_byte))
    print(f"model={model}  batch {args.batch} x {T}, mean sequence {msl:.0f} bytes, "
          + (f"{args.hours:g} h per rung" if args.hours else f"{args.ratio:g} bytes/param"))
    print(f"{'arm':22s} {'params':>8s} {'bytes':>8s} {'steps':>8s} {'~hours':>7s}  estimate")
    for r in rows:
        hs = f"{r['hours']:7.2f}" if r['hours'] is not None else '      ?'
        print(f"{r['arm']:22s} {r['params'] / 1e6:7.1f}M {r['bytes'] / 1e9:7.2f}G "
              f"{r['steps']:8d} {hs}  {r['est']}")
    print(f"total ~{tot_h:.1f} GPU-hours (training only; add ~5 min/rung for "
          f"compile and evaluation), {tot_bytes / 1e9:.2f}G bytes; the pool "
          f"has {pm['total_tokens'] / 1e9:.2f}G, so the largest rung sees "
          f"{max(r['bytes'] for r in rows) / pm['total_tokens']:.2f} passes")
    return rows


def run(args):
    model = getattr(args, 'model', 'opera')
    rows = plan(args)
    py = sys.executable
    for r in rows:
        if model == 'opera':
            cmd = [py, os.path.join(ROOT, 'experiments', 'repr_study.py'),
                   '--arms', r['arm'], '--steps', str(r['steps']),
                   '--batch', str(args.batch), '--packed', args.pool,
                   '--test-pkl', args.pool + '.test.pkl',
                   '--save-every', str(args.save_every), '--resume',
                   '--seed', str(args.seed)]
            if args.pool_test:
                cmd += ['--pool-test', args.pool_test]
        else:
            pe = 'nope' if model == 'tf_byte' else 'rope'
            cmd = [py, os.path.join(ROOT, 'experiments', 'train_tf_pool.py'),
                   '--arm', r['arm'], '--pe', pe,
                   '--vocab-size', str(r['vocab_size']),
                   '--units-per-byte', str(r['units_per_byte']),
                   '--d', str(r['d']), '--num-layers', str(r['L']),
                   '--steps', str(r['steps']), '--batch', str(args.batch),
                   '--packed', args.pool, '--test-pkl', args.pool + '.test.pkl',
                   '--save-every', str(args.save_every), '--resume',
                   '--seed', str(args.seed),
                   '--summary', os.path.join(RUNS, 'repr_summary.json')]
            if getattr(args, 'qk_norm', False):
                cmd += ['--qk-norm']
            if getattr(args, 'attn_logit_cap', 0.0):
                cmd += ['--attn-logit-cap', str(args.attn_logit_cap)]
            if getattr(args, 'log_attn_every', 0):
                cmd += ['--log-attn-every', str(args.log_attn_every)]
        if args.device:
            cmd += ['--device', args.device]
        if args.muon_lr_scale is not None:
            cmd += ['--muon-lr-scale', str(args.muon_lr_scale)]
        if getattr(args, 'max_lr_scale', None) is not None:
            cmd += ['--max-lr-scale', str(args.max_lr_scale)]
        if args.grad_clip is not None:
            cmd += ['--grad-clip', str(args.grad_clip)]
        print(f"\n### rung {r['arm']}: {r['steps']} steps", flush=True)
        log = open(os.path.join(RUNS, r['arm'] + '.log'), 'a')
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, cwd=ROOT)
        for line in p.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
        if p.wait() != 0:
            sys.exit(f"rung {r['arm']} failed (log: {log.name}); re-run to resume")
    fit(args)


def fit_power_law(C, y):
    """y = E + A * C^-alpha, by grid search over the irreducible term E
    (0 .. min(y)) with a log-log least-squares fit of the rest. With E = 0
    it is a pure power law. Returns (E, A, alpha, rmse)."""
    import numpy as np
    C, y = np.asarray(C, float), np.asarray(y, float)
    best = None
    for E in np.linspace(0, y.min() * 0.999, 400):
        X = np.stack([np.ones_like(C), np.log(C)], 1)
        coef, *_ = np.linalg.lstsq(X, np.log(y - E), rcond=None)
        pred = E + np.exp(coef[0]) * C ** coef[1]
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best[3]:
            best = (float(E), float(np.exp(coef[0])), float(-coef[1]), rmse)
    return best


def fit(args):
    prefix = ARM_PREFIX[getattr(args, 'model', 'opera')]
    summ = json.load(open(os.path.join(RUNS, 'repr_summary.json')))
    pts = []
    for name, v in summ.items():
        if not name.startswith(prefix) or v.get('test_pkl') is None:
            continue
        msl = mean_seq_len(v['packed'])
        D = v['steps'] * v['batch'] * msl
        pts.append(dict(arm=name, L=v['layers'], N=v['params'], D=D,
                        C=6 * v['params'] * D, bpb=v['bpb'],
                        pool=(v.get('pool_eval') or {})))
    if not pts:
        print('no finished ladder rungs yet')
        return
    pts.sort(key=lambda p: p['C'])
    print(f"\n{'arm':22s} {'params':>8s} {'bytes':>8s} {'C (FLOPs)':>10s} "
          f"{'FineWeb':>8s} {'SimpleWiki':>10s}")
    for p in pts:
        sw = p['pool'].get('pool')
        print(f"{p['arm']:22s} {p['N'] / 1e6:7.1f}M {p['D'] / 1e9:7.2f}G "
              f"{p['C']:10.2e} {p['bpb']:8.4f} "
              + (f"{sw:10.4f}" if sw is not None else f"{'-':>10s}"))
    # the width fit uses only the ladder's own ratio (runs at other
    # bytes/param, e.g. an iso-FLOP check, are listed but not fitted)
    tag = f"_r{args.ratio:g}{tf_arm_suffix(args) if getattr(args, 'model', 'opera') != 'opera' else ''}"
    excl = set(args.exclude or [])
    width = [p for p in pts if p['L'] == 2 and p['arm'].endswith(tag)
             and p['arm'] not in excl]
    res = {}
    if len(width) >= 3:
        E, A, a, rmse = fit_power_law([p['C'] for p in width], [p['bpb'] for p in width])
        print(f"\nwidth ladder (L=2, {len(width)} rungs): BPB = {E:.3f} + "
              f"{A:.3g} * C^-{a:.4f}   (rmse {rmse:.4f})")
        if len(width) < 5:
            print("  NOTE: 3-4 points constrain a 3-parameter fit weakly; "
                  "read alpha as indicative only")
        res = dict(E=E, A=A, alpha=a, rmse=rmse, n=len(width))
        for p in pts:
            if p not in width:
                pred = E + A * p['C'] ** -a
                kind = ('depth check' if p['L'] != 2 else
                        'excluded (unstable), not fitted' if p['arm'] in excl
                        else 'off-ratio')
                print(f"  {kind} {p['arm']}: {p['bpb']:.4f} vs width-fit "
                      f"{pred:.4f} at equal compute ({100 * (p['bpb'] / pred - 1):+.1f}%)")
    model = getattr(args, 'model', 'opera')
    fit_tag = 'ladder_fit' if model == 'opera' else f'ladder_fit_{model}{tf_arm_suffix(args)}'
    json.dump(dict(points=pts, width_fit=res),
              open(os.path.join(RUNS, f'{fit_tag}.json'), 'w'), indent=2)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 4))
        for L in sorted({p['L'] for p in pts}):
            q = [p for p in pts if p['L'] == L]
            ax.plot([p['C'] for p in q], [p['bpb'] for p in q], 'o',
                    label=f"L={L}")
            for p in q:
                n = p['N'] / 1e6
                ax.annotate(f"{n:.1f}M" if n < 10 else f"{n:.0f}M", (p['C'], p['bpb']),
                            textcoords='offset points', xytext=(4, 4), fontsize=8)
        if res:
            cs = np.logspace(np.log10(pts[0]['C']) - 0.2, np.log10(pts[-1]['C']) + 0.5, 100)
            ax.plot(cs, res['E'] + res['A'] * cs ** -res['alpha'], '-', lw=1,
                    label=f"fit: {res['E']:.2f} + A C^-{res['alpha']:.3f}")
        ax.set_xscale('log')
        ax.set_xlabel('training compute C = 6ND (FLOPs)')
        ax.set_ylabel('FineWeb-Edu held-out BPB')
        title = {'opera': 'OPERA + holographic memory: compute-optimal ladder',
                 'tf_byte': 'Transformer baseline (byte, pe=nope): compute-optimal ladder',
                 'tf_tok': 'Transformer baseline (BPE, pe=rope): compute-optimal ladder'}[model]
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, which='both')
        out = args.out or os.path.join(RUNS, f'{fit_tag}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"plot -> {out}")
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['plan', 'run', 'fit'])
    p.add_argument('--model', default='opera', choices=['opera', 'tf_byte', 'tf_tok'],
                   help='opera (default): OperaSpinorFenwickTree via repr_study.py. '
                        'tf_byte/tf_tok: the matched transformer baseline via '
                        'train_tf_pool.py (pe=nope/rope respectively) -- see module '
                        'docstring')
    p.add_argument('--pool', help='byte pool prefix (build_fineweb_bytes.py) for '
                        '--model opera/tf_byte, or token pool prefix '
                        '(build_fineweb_tokens.py) for --model tf_tok')
    p.add_argument('--rungs', nargs='+', default=DEFAULT_RUNGS, help='dxL entries')
    p.add_argument('--ratio', type=float, default=20.0, help='training bytes per parameter')
    p.add_argument('--hours', type=float, default=0,
                   help='instead of --ratio: train each rung for this many '
                        'hours (steps from the benchmark), i.e. as many bytes '
                        'as the time allows')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--bench', default=os.path.join(RUNS, 'bench.jsonl'))
    p.add_argument('--pool-test', default='assets/packed_bytes_T1024_all',
                   help='Simple-Wikipedia pool test sets (eval_pool); "" to skip')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default=None)
    p.add_argument('--muon-lr-scale', type=float, default=None,
                   help='run only: passed through to repr_study.py '
                        '--muon-lr-scale (safe to vary across --resume)')
    p.add_argument('--max-lr-scale', type=float, default=None,
                   help='run only: passed through to repr_study.py --max-lr-scale '
                        '(Adam-group LR; pair with --muon-lr-scale when running '
                        '--batch above the protocol 32 — keep constant across '
                        'resumes of the same run)')
    p.add_argument('--grad-clip', type=float, default=None,
                   help='run only: passed through to repr_study.py '
                        '--grad-clip (safe to vary across --resume)')
    p.add_argument('--qk-norm', action='store_true',
                   help='--model tf_byte/tf_tok only: per-head QK-norm '
                        '(train_tf_pool.py --qk-norm), the fix for the '
                        'attention-logit collapse observed in the '
                        'unfixed tfb_ ladder. Appends _qknorm to every '
                        'arm name in this run so it never collides with '
                        'an existing unfixed entry (plan/run/fit must all '
                        'pass the same flags to address the same arms)')
    p.add_argument('--attn-logit-cap', type=float, default=0.0,
                   help='--model tf_byte/tf_tok only: Gemma-2-style soft '
                        'cap on raw attention logits (0 = off); appends '
                        '_cap<v> to every arm name')
    p.add_argument('--log-attn-every', type=int, default=0,
                   help='--model tf_byte/tf_tok, run only: diagnostic -- '
                        'log max |attention logit| every N steps to '
                        '<arm>/attn_logits.jsonl (0 = off, no extra cost)')
    p.add_argument('--exclude', nargs='+', default=[],
                   help='fit only: arm names to keep OUT of the width fit '
                        '(still listed and plotted). Use for rungs that '
                        'diverged mid-training, e.g. the tfb_ widths whose '
                        'attention collapsed -- a diverged rung is an '
                        'instability observation, not a scaling point')
    p.add_argument('--out', default=None, help='plot path (fit)')
    args = p.parse_args()
    if args.cmd in ('plan', 'run') and not args.pool:
        p.error('--pool is required for plan / run')
    {'plan': plan, 'run': run, 'fit': fit}[args.cmd](args)


if __name__ == '__main__':
    main()
