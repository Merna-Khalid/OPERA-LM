"""Evaluate the pre-registered gates of docs/OPERA_Bistable_prereg.md.

Run AFTER `repr_study.py --arms bist_off bist_mono bist_bi` completes.
Reads the three checkpoints and reports H1-H4 verbatim against the
thresholds fixed in the registration, so the verdict is arithmetic
rather than judgement.

    H1 (primary)  p99.9/median of the step distribution:
                  PASS iff  bi >= 1.60  AND  bi - mono >= 0.10
    H2            |partial r(step, surprisal | emb norm)|:
                  PASS iff  |bi| exceeds |off| and |mono| by >= 0.10
    H3 (no gate)  fraction of blocks with gain a > 1, i.e. the arm
                  reporting whether the objective actually reached for
                  bistability. 'bi' with a <= 1 everywhere is a null by
                  cold parameters, NOT a failed mechanism.
    H4 (guard)    BPB must not regress > 5% vs bytes_d512 (2.1317).

Everything is measured on the same held-out byte sequences for all
three arms.

    python experiments/bistable_gates.py
"""
import json
import math
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.garden_path import (load_opera, _restore, prefix_states,   # noqa: E402
                                  surprisal, pearson, fisher_mean)
from opera_lm.step_nulls import state_geometry, summarize               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
ARMS = ['bist_off', 'bist_mono', 'bist_bi', 'bist_bi_r1',
        'bist_bi_r4', 'relax_under', 'relax_over']
RELAX = {'relax_under': 'under', 'relax_over': 'over'}
RANK = {'bist_bi_r1': 1, 'bist_bi_r4': 4}
BPB_REF = 2.1317          # bytes_d512, the same rung
CORPUS = 'assets/corpus_bytes_T1024_a20000.pkl'


def find_ckpt(arm):
    d = os.path.join(ROOT, 'runs_reprs', arm)
    if not os.path.isdir(d):
        return None
    c = [f for f in os.listdir(d)
         if f.endswith('.pt') and 'train_ckpt' not in f]
    return os.path.join(d, c[0]) if c else None


def partial_r(a, b, c):
    """r(a,b) controlling for c, Fisher-z aggregated over positions."""
    out = []
    for t in range(len(a)):
        if len(a[t]) < 8:
            continue
        rab, rac, rbc = pearson(a[t], b[t]), pearson(a[t], c[t]), \
            pearson(b[t], c[t])
        den = math.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 1e-12))
        out.append((rab - rac * rbc) / den)
    return fisher_mean(out)[0]


@torch.no_grad()
def measure(arm, seqs, T=256, batch=8, mode='off', rank=0,
            relax='off'):
    ck = find_ckpt(arm)
    if ck is None:
        return None
    m = load_opera(ck, vocab_size=259, d=512, nb=128, layers=2,
                   rot_mode='free', fold_bistable=mode,
                   bist_rank=rank, fold_relax=relax)
    steps_all = []
    S, U, N = [[] for _ in range(T - 1)], [[] for _ in range(T - 1)], \
        [[] for _ in range(T - 1)]
    W = _restore(m).weight.detach()
    for i in range(0, len(seqs), batch):
        ch = seqs[i:i + batch]
        ids = torch.tensor([s[:T] for s in ch])
        ln = torch.full((len(ch),), T, dtype=torch.long)
        e0 = _restore(m)(ids).detach()
        st = prefix_states(m, e0, ids, ln)
        _restore(m)
        sup = surprisal(m, ids, ln, e0)
        _restore(m)
        _, step, _ = state_geometry(st.float())
        steps_all.append(step)
        nrm = W[ids[:, 1:]].norm(dim=-1)
        for t in range(T - 1):
            S[t] += step[:, t].tolist()
            U[t] += sup[:, t].tolist()
            N[t] += nrm[:, t].tolist()
    step = torch.cat(steps_all, 0)
    res = summarize(step, arm)
    res['partial_r_step_surprisal'] = partial_r(S, U, N)

    # H3: did the objective reach for bistability?
    #
    # MUST be measured on REAL text. The gain is content-dependent, so
    # feeding it Gaussian noise reports what it does to noise, not to
    # language -- and the two differ enormously here (noise: a_mean
    # 1.001, frac(a>1) 0.50, i.e. indistinguishable from sitting on the
    # bifurcation point; real text: a_mean 1.270, median 1.646,
    # frac(a>1) 0.66). Reporting the noise number would have inverted
    # the verdict of the whole study.
    if mode != 'off':
        caps = []
        orig = m._bistable_update

        def spy(acc, nxt, composed, layer_idx):
            h = torch.cat([acc, nxt], dim=-1)
            z = m.bist_a[layer_idx](h)
            a_ = (1.0 + torch.tanh(z) if mode == 'bi'
                  else torch.sigmoid(z))
            caps.append(a_.detach().flatten())
            return orig(acc, nxt, composed, layer_idx)

        m._bistable_update = spy
        ids = torch.tensor([s[:T] for s in seqs[:batch]])
        m(ids, torch.full((ids.shape[0],), T, dtype=torch.long))
        m._bistable_update = orig
        a_all = torch.cat(caps)
        qs = torch.quantile(a_all, torch.tensor([0.5, 0.9, 0.99]))
        res['gain'] = {
            'a_mean': a_all.mean().item(), 'a_sd': a_all.std().item(),
            'a_median': qs[0].item(), 'a_p90': qs[1].item(),
            'a_max': a_all.max().item(),
            'frac_bistable': (a_all > 1.0).float().mean().item(),
            'frac_deep': (a_all > 1.5).float().mean().item(),
            'n': a_all.numel(),
        }
    return res


def main():
    with open(os.path.join(ROOT, CORPUS), 'rb') as f:
        (_, ts, _, _), _ = pickle.load(f)
    seqs = [s for s in ts if len(s) >= 257][:64]

    summ_path = os.path.join(ROOT, 'runs_reprs', 'repr_summary.json')
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}

    print('=' * 76)
    print('BISTABLE FOLD — pre-registered gates '
          '(docs/OPERA_Bistable_prereg.md)')
    print('=' * 76)

    res = {}
    for arm in ARMS:
        if arm in RELAX:
            mode, rank, rlx = 'off', 0, RELAX[arm]
        else:
            mode = 'off' if arm == 'bist_off' else (
                'mono' if arm == 'bist_mono' else 'bi')
            rank, rlx = RANK.get(arm, 0), 'off'
        r = measure(arm, seqs, mode=mode, rank=rank, relax=rlx)
        # R-H2: gamma usage on REAL text (never on noise -- that
        # estimator inverted the bistable verdict once already).
        if r is not None and arm in RELAX:
            mm = load_opera(find_ckpt(arm), vocab_size=259, d=512, nb=128,
                            layers=2, rot_mode='free', fold_relax=RELAX[arm])
            caps = []
            orig = mm._relax_update

            def spy(acc, nxt, composed, li, _o=orig, _m=mm, _a=arm):
                z = torch.tanh(_m.relax_g[li](
                    torch.cat([acc, nxt], dim=-1)))
                if RELAX[_a] == 'under':
                    z = torch.clamp(z, max=0.0)
                caps.append((1.0 + z).detach().flatten())
                return _o(acc, nxt, composed, li)

            mm._relax_update = spy
            ids0 = torch.tensor([s[:256] for s in seqs[:8]])
            mm(ids0, torch.full((ids0.shape[0],), 256, dtype=torch.long))
            g_all = torch.cat(caps)
            r['gain'] = {
                'a_mean': g_all.mean().item(),
                'a_median': g_all.median().item(),
                'a_max': g_all.max().item(),
                'frac_bistable': (g_all > 1.0).float().mean().item(),
                'frac_deep': (g_all > 1.5).float().mean().item(),
                'n': g_all.numel()}
        if r is None:
            print(f"  [{arm}] no checkpoint yet — skipping")
            continue
        r['bpb'] = summ.get(arm, {}).get('bpb')
        res[arm] = r

    if len(res) < 3:
        print('\n  incomplete: rerun once all three arms have finished')
        return

    print(f"\n  {'arm':<11}{'BPB':>8}{'step med':>10}{'p99.9/med':>11}"
          f"{'partial r':>11}{'frac a>1':>10}")
    for a in ARMS:
        r = res[a]
        fr = r.get('gain', {}).get('frac_bistable')
        print(f"  {a:<11}{(r['bpb'] or float('nan')):>8.4f}"
              f"{r['median']:>10.3f}{r['p999_over_median']:>11.3f}"
              f"{r['partial_r_step_surprisal']:>11.4f}"
              f"{('—' if fr is None else f'{fr:.3f}'):>10}")

    bi, mo, off = res['bist_bi'], res['bist_mono'], res['bist_off']
    print('\n  ADDENDUM A-H1 — does CORRELATED commitment move the tail?')
    print(f"    {'arm':<13}{'rank':>5}{'p99.9/med':>11}{'vs bi':>9}"
          f"{'frac a>1':>10}{'BPB':>9}")
    for a in ('bist_bi', 'bist_bi_r1', 'bist_bi_r4'):
        if a not in res:
            continue
        r = res[a]
        fr = r.get('gain', {}).get('frac_bistable', float('nan'))
        print(f"    {a:<13}{RANK.get(a, 0):>5}{r['p999_over_median']:>11.3f}"
              f"{r['p999_over_median'] - bi['p999_over_median']:>+9.3f}"
              f"{fr:>10.3f}{(r['bpb'] or float('nan')):>9.4f}")
    ah1 = [a for a in ('bist_bi_r1', 'bist_bi_r4') if a in res
           and res[a]['p999_over_median'] >= 1.60
           and res[a]['p999_over_median'] - bi['p999_over_median'] >= 0.10]
    print(f"    A-H1 (>=1.60 AND >= bi+0.10): "
          f"{'PASS: ' + ', '.join(ah1) if ah1 else 'FAIL for every rank'}")
    if not ah1 and all(res[a].get('gain', {}).get('frac_bistable', 0) > 0.01
                       for a in ('bist_bi_r1', 'bist_bi_r4') if a in res):
        print('    -> gains WERE driven past 1, so this is not a cold-'
              'parameter null.\n       The post-mortem explanation of the '
              'per-block failure (independent\n       commitment averages '
              'out) is itself FALSIFIED: correlating the\n       gains '
              'changes nothing either.')
    print('\n  GATES')
    t_bi, t_mo = bi['p999_over_median'], mo['p999_over_median']
    h1 = t_bi >= 1.60 and (t_bi - t_mo) >= 0.10
    print(f"    H1  tail: bi {t_bi:.3f} (need >=1.60), "
          f"bi-mono {t_bi - t_mo:+.3f} (need >=0.10)  -> "
          f"{'PASS' if h1 else 'FAIL'}")
    rb, rm, ro = (abs(x['partial_r_step_surprisal'])
                  for x in (bi, mo, off))
    h2 = (rb - rm) >= 0.10 and (rb - ro) >= 0.10
    print(f"    H2  |partial r|: bi {rb:.4f} vs mono {rm:.4f} vs off "
          f"{ro:.4f}  -> {'PASS' if h2 else 'FAIL'}")
    fr = bi.get('gain', {}).get('frac_bistable', 0.0)
    am = bi.get('gain', {}).get('a_mean', float('nan'))
    print(f"    H3  gain used: frac(a>1) {fr:.3f}, a_mean {am:.3f} "
          f"(no gate; reports usage)")
    h4 = bi['bpb'] is not None and bi['bpb'] <= BPB_REF * 1.05
    print(f"    H4  BPB {bi['bpb']:.4f} vs {BPB_REF} x1.05 = "
          f"{BPB_REF * 1.05:.4f}  -> {'PASS' if h4 else 'FAIL'}")

    if 'relax_over' in res and 'relax_under' in res:
        ov, un = res['relax_over'], res['relax_under']
        print('\n  RELAX (docs/OPERA_Relax_prereg.md) — does EXTRAPOLATION '
              'move the tail?')
        print(f"    {'arm':<13}{'BPB':>8}{'step med':>10}{'p99.9/med':>11}"
              f"{'partial r':>11}{'frac g>1':>10}{'mean g':>8}")
        for a in ('bist_off', 'relax_under', 'relax_over'):
            r = res[a]; gn = r.get('gain', {})
            print(f"    {a:<13}{(r['bpb'] or float('nan')):>8.4f}"
                  f"{r['median']:>10.3f}{r['p999_over_median']:>11.3f}"
                  f"{r['partial_r_step_surprisal']:>11.4f}"
                  f"{gn.get('frac_bistable', float('nan')):>10.3f}"
                  f"{gn.get('a_mean', float('nan')):>8.3f}")
        t_o, t_u = ov['p999_over_median'], un['p999_over_median']
        rh1 = t_o >= 1.60 and (t_o - t_u) >= 0.10
        print(f"    R-H1 tail: over {t_o:.3f} (need >=1.60), over-under "
              f"{t_o - t_u:+.3f} (need >=0.10) -> "
              f"{'PASS' if rh1 else 'FAIL'}")
        fo = ov.get('gain', {}).get('frac_bistable', 0.0)
        print(f"    R-H2 overshoot used: frac(g>1) {fo:.3f}, "
              f"mean g {ov.get('gain', {}).get('a_mean', float('nan')):.3f}")
        if not rh1 and fo > 0.01:
            print('    -> extrapolation was AVAILABLE, TRAINED and USED, and '
                  'still produced\n       no excursions. With the four '
                  'gating negatives this means displacement\n       is '
                  'inaccessible by ANY local update rule on the fold '
                  'accumulator.')
        elif not rh1:
            print('    -> the objective DECLINED overshoot (g stayed <=1): '
                  'a training\n       finding, not a mechanism failure.')

    print('\n  VERDICT (pre-stated in §6)')
    if h1 and fr > 0.01:
        print('    Bistability moved the trajectory. First mechanism in '
              'this project\n    to change the step distribution.')
    elif not h1 and fr > 0.01:
        print('    Bistability was AVAILABLE, TRAINED, and did not '
              'produce excursions.\n    The mechanism class is wrong — '
              'the line closes. Clean negative.')
    else:
        print('    The objective DECLINED bistability (a stayed <= 1). '
              'This is a\n    training/objective finding, NOT a '
              'mechanism failure — do not\n    report it as one.')
    print('=' * 76)

    with open(os.path.join(ROOT, 'runs_reprs', 'bistable_gates.json'),
              'w') as f:
        json.dump(res, f, indent=2, default=float)
    print('  wrote runs_reprs/bistable_gates.json')


if __name__ == '__main__':
    main()
