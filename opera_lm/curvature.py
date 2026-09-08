"""Representation curvature: does OPERA's record of falsified geometric
constraints have a single mechanistic explanation?

BACKGROUND
----------
"Gating Enables Curvature: A Geometric Expressivity Gap in Attention"
(arXiv:2604.14702) proves that ungated attention -- outputs confined to
affine combinations of value vectors -- induces INTRINSICALLY FLAT
representation manifolds, and that multiplicative gating breaks the
affine structure and permits strictly positive curvature. It measures
curvature with a finite-difference proxy for local nonlinearity and
reports r = 0.79 between curvature and accuracy, with no benefit on
linear control tasks (the advantage is task-geometry dependent).

THE HYPOTHESIS THIS SCRIPT TESTS
--------------------------------
OPERA offers a RIGID (flat / isometric / equivariant) and a BENDABLE
option on four independent knobs, and the pre-registered record says
the rigid one lost every time:

    rot_mode    so3 (exact rotation)      vs  free       -> free won (so3 nulled twice)
    fold_mode   rack (isometric conj.)    vs  left       -> left won
    norm_mode   blockrms (SO(3)-equivar.) vs  layer      -> layer won (+4 PPL)
    act_mode    linear (geometry-clean)   vs  tanh       -> tanh is the incumbent

H: those are not four unrelated nulls. They are one result -- curvature
is load-bearing, and imposing flat geometry costs performance.

PREDICTION (stated before running): within each knob, the configuration
that WON on perplexity has HIGHER measured curvature. A knob where the
ranking reverses falsifies the account for that knob; a majority
reversal kills it outright.

Nothing here trains. Curvature is a property of the architecture's
realizable geometry, so it is measured at initialization across seeds --
which also avoids the confound of different configurations training to
different quality.

    python -m opera_lm.curvature --validate     # proxy sanity checks only
    python -m opera_lm.curvature                # the four-knob comparison
"""
import argparse
import json

import torch
import torch.nn as nn

from .model import OperaSpinorFenwickTree


def second_difference_ratio(f, x, n_dirs=16, h_frac=1e-2, seed=0):
    """Dimensionless finite-difference curvature proxy.

        rho = || f(x+hu) - 2 f(x) + f(x-hu) ||  /  || f(x+hu) - f(x-hu) ||

    Numerator ~ h^2 ||g''||, denominator ~ 2h ||g'|| along the curve
    g(t) = f(x + t u), so rho ~ (h/2) * ||g''|| / ||g'|| -- the local
    deviation from affine behaviour, RELATIVE to the linear response.

    Two properties make it usable as a cross-configuration metric:
      * it is exactly 0 for any affine f, at any scale;
      * it is invariant to rescaling f (c*f leaves rho unchanged), so
        configurations whose outputs have different magnitudes -- which
        these very much do -- stay comparable.

    Averaged over n_dirs random unit directions u, with step size h set
    as a fixed FRACTION of ||x|| so the perturbation is relative.
    """
    g = torch.Generator().manual_seed(seed)
    x = x.detach()
    h = h_frac * x.norm()
    f0 = f(x)
    out = []
    for _ in range(n_dirs):
        u = torch.randn(x.shape, generator=g)
        u = u / u.norm()
        fp, fm = f(x + h * u), f(x - h * u)
        num = (fp - 2 * f0 + fm).norm()
        den = (fp - fm).norm()
        if den > 1e-12:
            out.append((num / den).item())
    return float(torch.tensor(out).mean()), float(torch.tensor(out).std())


def validate_proxy():
    """The metric must read 0 on affine maps, >0 on curved ones, and
    scale ~linearly in h. Reported before any model number is trusted."""
    torch.manual_seed(0)
    d = 64
    A = torch.randn(d, d) / d ** 0.5
    b = torch.randn(d)
    x = torch.randn(d)

    print('  PROXY VALIDATION')
    aff, _ = second_difference_ratio(lambda z: A @ z + b, x)
    print(f"    affine  f(x) = Ax + b            rho = {aff:.3e}   "
          f"(must be ~0)")
    tanh, _ = second_difference_ratio(lambda z: torch.tanh(3 * (A @ z)), x)
    print(f"    curved  f(x) = tanh(3Ax)         rho = {tanh:.3e}")
    ln = nn.LayerNorm(d)
    lnv, _ = second_difference_ratio(lambda z: ln(A @ z), x)
    print(f"    curved  f(x) = LayerNorm(Ax)     rho = {lnv:.3e}")
    print('    h-scaling of the curved map (rho should be ~linear in h):')
    for hf in [4e-3, 8e-3, 1.6e-2, 3.2e-2]:
        r, _ = second_difference_ratio(
            lambda z: torch.tanh(3 * (A @ z)), x, h_frac=hf)
        print(f"      h_frac={hf:<7.4f}  rho = {r:.4e}   "
              f"rho/h = {r / hf:.4f}")
    ok = aff < 1e-5 and tanh > 1e-3
    print(f"    VALIDATION: {'PASS' if ok else 'FAIL'}")
    return ok


def h_linearity_check(measure_fn, hs=(2.5e-3, 5e-3, 1e-2, 2e-2),
                      max_drift=1.35):
    """rho is only a second-order (curvature) measure while rho/h is
    roughly constant. If it drifts, the configuration is outside the
    small-h regime and its rho is NOT comparable to another config's.

    This is not a formality. Measured here: the `rack` fold drifts 62x
    and its rho DECREASES with h -- the signature of the denominator
    ||f(x+hu) - f(x-hu)|| collapsing because rack's conjugation
    q*acc*q^-1 is quadratic in q, so the odd (first-order) term vanishes
    and the map is locally EVEN. rho then diverges as h -> 0 for a
    reason that has nothing to do with curvature. Reporting that number
    as 'rack is 12x more curved' would have been an artifact.
    """
    vals = [measure_fn(h) for h in hs]
    ratios = [v / h for v, h in zip(vals, hs)]
    drift = max(ratios) / min(ratios)
    return drift <= max_drift, drift, vals


class _FixedEmb(nn.Module):
    """Feeds a supplied embedding tensor to the model instead of a
    lookup, so the representation map can be perturbed continuously."""

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, ids):
        return self.t


def representation_curvature(cfg, T=64, batch=4, d=128, nb=32, layers=2,
                             vocab=64, seed=0, n_dirs=16, h_frac=1e-2):
    """Curvature of the map (token embeddings) -> (final-layer prefix
    states) -- the analogue of the paper's X -> Y(X), measured on the
    representation the LM head actually reads."""
    torch.manual_seed(seed)
    model = OperaSpinorFenwickTree(
        vocab_size=vocab, d=d, nb=nb, num_layers=layers, pe_mode='none',
        tie=False, **cfg).eval()
    ids = torch.randint(0, vocab, (batch, T))
    lengths = torch.full((batch,), T, dtype=torch.long)
    x0 = model.word_emb(ids).detach()

    def f(e):
        model.word_emb = _FixedEmb(e)
        with torch.no_grad():
            return model(ids, lengths, return_states=True).states[-1]

    return second_difference_ratio(f, x0, n_dirs=n_dirs, h_frac=h_frac,
                                   seed=seed + 1)


# Each knob: (name, rigid config, bendable config, which won on PPL).
# The PPL winner is taken from the project's own falsification record
# and is FIXED before measurement -- it is not read off these numbers.
KNOBS = [
    ('rot_mode',
     dict(rot_mode='so3'), dict(rot_mode='free'),
     'bendable', 'so3 nulled twice (pre-registered rematch)'),
    ('norm_mode',
     dict(norm_mode='blockrms'), dict(norm_mode='layer'),
     'bendable', 'blockrms measured +4 PPL'),
    ('act_mode',
     dict(act_mode='linear'), dict(act_mode='tanh'),
     'bendable', 'tanh is the incumbent'),
    ('fold_mode',
     dict(fold_mode='rack'), dict(fold_mode='left'),
     'bendable', 'rack slower at fixed budget'),
]

BASE = dict(rot_mode='free', norm_mode='layer', act_mode='tanh',
            fold_mode='left')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--train-len', type=int, default=64)
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--nb', type=int, default=32)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--n-dirs', type=int, default=16)
    p.add_argument('--h-frac', type=float, default=1e-2)
    p.add_argument('--validate', action='store_true',
                   help='run proxy sanity checks and exit')
    p.add_argument('--json', default=None)
    args = p.parse_args()

    print('=' * 74)
    print('OPERA REPRESENTATION CURVATURE — rigid vs bendable, 4 knobs')
    print('=' * 74)
    if not validate_proxy():
        print('\n  proxy failed validation; not reporting model numbers')
        return
    if args.validate:
        return

    print(f"\n  d={args.d} nb={args.nb} layers={args.layers} T="
          f"{args.train_len} seeds={args.seeds} dirs={args.n_dirs} "
          f"h_frac={args.h_frac}")
    print('  measured at INITIALISATION (architectural bias, not learned)')
    print('\n  PREDICTION (fixed in advance): the PPL winner is the more '
          'curved config.\n')
    print(f"  {'knob':<11} {'rigid':<22} {'bendable':<20} "
          f"{'ratio':>7}  {'pred':>5}")
    print('  ' + '-' * 70)

    results, hits, measurable = {}, 0, 0
    for name, rigid, bend, ppl_winner, note in KNOBS:
        kwv = dict(T=args.train_len, d=args.d, nb=args.nb,
                   layers=args.layers, seed=0, n_dirs=8)
        ok_r, dr, _ = h_linearity_check(
            lambda h: representation_curvature(
                {**BASE, **rigid}, h_frac=h, **kwv)[0])
        ok_b, db, _ = h_linearity_check(
            lambda h: representation_curvature(
                {**BASE, **bend}, h_frac=h, **kwv)[0])
        rk, bk = list(rigid.values())[0], list(bend.values())[0]
        if not (ok_r and ok_b):
            bad = f"{rk} drift {dr:.1f}x" if not ok_r else ''
            bad += (', ' if bad else '') + (f"{bk} drift {db:.1f}x"
                                            if not ok_b else '')
            results[name] = dict(measurable=False, detail=bad, note=note)
            print(f"  {name:<11} NOT MEASURABLE — {bad}")
            continue

        measurable += 1
        rs, bs = [], []
        for s in range(args.seeds):
            kw = dict(T=args.train_len, d=args.d, nb=args.nb,
                      layers=args.layers, seed=s, n_dirs=args.n_dirs,
                      h_frac=args.h_frac)
            rs.append(representation_curvature({**BASE, **rigid}, **kw)[0])
            bs.append(representation_curvature({**BASE, **bend}, **kw)[0])
        rt, bt = torch.tensor(rs), torch.tensor(bs)
        ratio = (bt.mean() / rt.mean()).item()
        # Separated only if the gap clears the pooled spread; a ratio
        # above 1 inside overlapping error bars is not a result.
        sep = abs(bt.mean() - rt.mean()) > (rt.std() + bt.std())
        ok = ratio > 1.0 and sep
        hits += ok
        results[name] = dict(measurable=True, rigid_mean=rt.mean().item(),
                             rigid_std=rt.std().item(),
                             bendable_mean=bt.mean().item(),
                             bendable_std=bt.std().item(),
                             ratio=ratio, separated=bool(sep),
                             prediction_held=bool(ok), note=note)
        verdict = 'HELD' if ok else ('OVERLAP' if ratio > 1 else 'FAIL')
        print(f"  {name:<11} {rk + ':':<9}{rt.mean():.4f}±{rt.std():.4f}  "
              f"{bk + ':':<7}{bt.mean():.4f}±{bt.std():.4f}  "
              f"{ratio:>6.2f}x  {verdict:>7}")

    print('  ' + '-' * 70)
    print(f"\n  measurable knobs: {measurable}/{len(KNOBS)}   "
          f"prediction held (and separated): {hits}/{measurable}")
    if measurable < len(KNOBS):
        print("  -> INCONCLUSIVE. The proxy cannot measure every knob, and")
        print("     the unmeasurable ones are the informative ones. This is")
        print("     neither support for the account nor evidence against it.")
    elif hits == measurable:
        print("  -> curvature ranks every knob the way perplexity did.")
    else:
        print("  -> ACCOUNT FALSIFIED on the measurable knobs.")
    print('=' * 74)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'config': vars(args), 'knobs': results,
                       'measurable': measurable, 'hits': hits}, f,
                      indent=2, default=float)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
