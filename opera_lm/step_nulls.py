"""Is constant-speed integration FORCED by geometry, or LEARNED?

THE GAP THIS FILLS
------------------
`trajectory.py` established that OPERA, a matched RoPE transformer and
Mamba all integrate at near-constant speed (step p99.9/median 1.44-1.94,
no surprisal coupling surviving controls), and that no model moves its
state *more* on informative tokens. Phase 0 called this "the accelerator
is empty."

But that measurement has no null model, and every one of these
architectures normalizes its state. A normalized state lives on a sphere
of radius R, so

    step_t^2 = R_t^2 + R_{t-1}^2 - 2 R_t R_{t-1} cos(theta_t)

Step size is then a deterministic function of radius and turn angle, and
is bounded above by R_t + R_{t-1}. If normalization pins R, step
variance is *purely angular* -- and "constant speed" might be a
statement about the norm layer rather than about the model's integration
policy. Before designing a mechanism to fix the deficiency we need to
know whether there is a deficiency to fix.

WHAT IS COMPUTED
----------------
1. **Radius**: mean and CV of ||h_t||. Near-zero CV means R is pinned by
   normalization and cannot contribute to step variance.

2. **Variance decomposition.** Two counterfactual step distributions
   rebuilt from the same trajectory:
     * `step_angle_only`  -- observed theta, R frozen at its mean;
     * `step_radius_only` -- observed R, theta frozen at its mean.
   Their variances say how much of the real spread each channel could
   account for on its own.

3. **Isotropic null.** If successive states were at *random* directions
   at the observed radius, what would step look like? In d dimensions
   the cosine between random unit vectors has density proportional to
   (1 - c^2)^((d-3)/2): concentrated at 90 degrees with SD ~ 1/sqrt(d).
   Sampled directly rather than approximated. This is the "geometry
   forces it" hypothesis in its strongest form.

4. **Shuffled-token control.** The same model on scrambled input. This
   keeps the architecture, the weights and the normalization, and
   destroys the linguistic structure. If the step distribution is
   unchanged, trajectory shape is a property of the network, not of
   language.

5. **Headroom.** observed mean step / (R_t + R_{t-1}), the fraction of
   the geometrically available range actually used. Near 1.0 means the
   model is at the ceiling and no mechanism can amplify without breaking
   norm conservation. Well below 1.0 means the flatness is a policy, not
   a cage -- and a bistable/switching mechanism is the right lever.

READING THE RESULT
------------------
* observed spread ~= isotropic null  -> geometry forces it. A new
  mechanism must break norm conservation (let the state grow during
  reanalysis) or move to a lower-dimensional bottleneck.
* observed spread << isotropic null, headroom low -> the objective is
  actively SUPPRESSING excursions. The lever is bistability: two
  interpretations as two attractors, disambiguation as a basin switch.
  This also explains the phase-1 cold-gate nulls -- a smooth sigmoid
  gate has no reason to switch.
* shuffled ~= real -> whatever the shape is, it is not about language.

    python -m opera_lm.step_nulls --ckpt <path> --repr bytes
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from .garden_path import load_opera, _restore, prefix_states


def state_geometry(states):
    """states [B,T,d] -> radius [B,T], step [B,T-1], theta [B,T-1]."""
    R = states.norm(dim=-1)
    step = (states[:, 1:] - states[:, :-1]).norm(dim=-1)
    cos = F.cosine_similarity(states[:, 1:], states[:, :-1], dim=-1)
    theta = cos.clamp(-1 + 1e-7, 1 - 1e-7).arccos()
    return R, step, theta


def _law(Ra, Rb, theta):
    """step from the law of cosines -- the identity everything rests on."""
    return (Ra ** 2 + Rb ** 2
            - 2 * Ra * Rb * theta.cos()).clamp(min=0).sqrt()


def isotropic_cos(n, d, generator=None):
    """Cosine between two random unit vectors in R^d, sampled exactly
    (not approximated): normalize two independent Gaussians."""
    a = torch.randn(n, d, generator=generator)
    b = torch.randn(n, d, generator=generator)
    return F.cosine_similarity(a, b, dim=-1)


def summarize(x, name):
    x = x.flatten().float()
    q = torch.quantile(x, torch.tensor([0.5, 0.99, 0.999]))
    return {
        'name': name, 'mean': x.mean().item(), 'sd': x.std().item(),
        'cv': (x.std() / x.mean().clamp(min=1e-9)).item(),
        'median': q[0].item(), 'p99': q[1].item(), 'p999': q[2].item(),
        'p999_over_median': (q[2] / q[0].clamp(min=1e-9)).item(),
    }


def _row(s):
    return (f"    {s['name']:<22} mean {s['mean']:8.3f}  sd {s['sd']:7.3f}"
            f"  CV {s['cv']:6.4f}  p99.9/med {s['p999_over_median']:5.3f}")


@torch.no_grad()
def analyze(model, ids, lengths, d_model, seed=0, shuffle_ids=None):
    emb = _restore(model)
    e0 = emb(ids).detach()
    states = prefix_states(model, e0, ids, lengths)
    _restore(model)
    R, step, theta = state_geometry(states.float())
    Ra, Rb = R[:, 1:], R[:, :-1]
    out = {}

    out['radius'] = summarize(R, 'radius ||h||')
    out['step_observed'] = summarize(step, 'step OBSERVED')
    out['theta_deg'] = summarize(theta * 180 / math.pi, 'turn angle (deg)')

    # (2) variance decomposition
    Rm = R.mean()
    out['step_angle_only'] = summarize(
        _law(Rm.expand_as(theta), Rm.expand_as(theta), theta),
        'step: angle only (R fixed)')
    tm = theta.mean().expand_as(theta)
    out['step_radius_only'] = summarize(
        _law(Ra, Rb, tm), 'step: radius only (th fixed)')

    # (3) isotropic null at the OBSERVED radii
    g = torch.Generator().manual_seed(seed)
    c = isotropic_cos(theta.numel(), d_model, g).reshape(theta.shape)
    out['step_isotropic'] = summarize(
        _law(Ra, Rb, c.clamp(-1 + 1e-7, 1 - 1e-7).arccos()),
        'step: ISOTROPIC null')
    out['theta_isotropic_deg'] = summarize(
        c.clamp(-1 + 1e-7, 1 - 1e-7).arccos() * 180 / math.pi,
        'turn: isotropic (deg)')

    # (4) shuffled-token control
    if shuffle_ids is not None:
        es = _restore(model)(shuffle_ids).detach()
        ss = prefix_states(model, es, shuffle_ids, lengths)
        _restore(model)
        _, sstep, sth = state_geometry(ss.float())
        out['step_shuffled'] = summarize(sstep, 'step: SHUFFLED tokens')
        out['theta_shuffled_deg'] = summarize(sth * 180 / math.pi,
                                              'turn: shuffled (deg)')

    # (5) headroom
    head = step / (Ra + Rb).clamp(min=1e-9)
    out['headroom'] = summarize(head, 'step / (Ra+Rb)  [max 1]')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--repr', default='bytes',
                   choices=('bytes', 'bpe', 'word'))
    p.add_argument('--corpus', default=None,
                   help='cached corpus pkl (bytes/bpe); word uses vocab_docs')
    p.add_argument('--vocab-size', type=int, default=259)
    p.add_argument('--d', type=int, default=512)
    p.add_argument('--nb', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--rot-mode', default='free')
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--n-seqs', type=int, default=64)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    import pickle
    if args.repr == 'word':
        with open('assets/vocab_docs.pkl', 'rb') as f:
            _, ts, _, _, _, _ = pickle.load(f)
    else:
        with open(args.corpus, 'rb') as f:
            (_, ts, _, _), _ = pickle.load(f)
    seqs = [s for s in ts if len(s) >= args.seq_len][:args.n_seqs]
    if not seqs:
        raise SystemExit('no sequences long enough')

    model = load_opera(args.ckpt, vocab_size=args.vocab_size, d=args.d,
                       nb=args.nb, layers=args.layers,
                       rot_mode=args.rot_mode)
    print('=' * 78)
    print('STEP-SIZE NULL MODELS — is constant speed forced or learned?')
    print('=' * 78)
    print(f"  ckpt {args.ckpt.split('/')[-1][:52]}")
    print(f"  repr={args.repr} d={args.d} T={args.seq_len} "
          f"n={len(seqs)}")

    g = torch.Generator().manual_seed(args.seed)
    acc = []
    for i in range(0, len(seqs), args.batch):
        chunk = seqs[i:i + args.batch]
        ids = torch.tensor([s[:args.seq_len] for s in chunk])
        lengths = torch.full((len(chunk),), args.seq_len,
                             dtype=torch.long)
        perm = torch.stack([ids[b][torch.randperm(args.seq_len,
                                                  generator=g)]
                            for b in range(ids.shape[0])])
        acc.append(analyze(model, ids, lengths, args.d,
                           seed=args.seed + i, shuffle_ids=perm))

    # average the summaries across batches
    keys = acc[0].keys()
    res = {}
    for k in keys:
        res[k] = {kk: (sum(a[k][kk] for a in acc) / len(acc)
                       if kk != 'name' else acc[0][k]['name'])
                  for kk in acc[0][k]}

    print('\n  [1] RADIUS — is it pinned by normalization?')
    print(_row(res['radius']))
    pinned = res['radius']['cv'] < 0.05
    print(f"      -> radius {'PINNED' if pinned else 'FREE'} "
          f"(CV {res['radius']['cv']:.4f})")

    print('\n  [2] VARIANCE DECOMPOSITION — which channel carries spread?')
    for k in ['step_observed', 'step_angle_only', 'step_radius_only']:
        print(_row(res[k]))
    va, vr = res['step_angle_only']['sd'], res['step_radius_only']['sd']
    tot = max(va + vr, 1e-9)
    print(f"      -> angle accounts for {100 * va / tot:.0f}% of the "
          f"reconstructible spread, radius {100 * vr / tot:.0f}%")

    print('\n  [3] ISOTROPIC NULL — what would random directions give?')
    for k in ['theta_deg', 'theta_isotropic_deg', 'step_observed',
              'step_isotropic']:
        print(_row(res[k]))
    ratio = res['step_observed']['sd'] / max(res['step_isotropic']['sd'],
                                             1e-9)
    print(f"      -> observed step SD is {ratio:.2f}x the isotropic null")

    if 'step_shuffled' in res:
        print('\n  [4] SHUFFLED-TOKEN CONTROL — is the shape about '
              'language?')
        for k in ['step_observed', 'step_shuffled', 'theta_deg',
                  'theta_shuffled_deg']:
            print(_row(res[k]))
        dd = abs(res['step_shuffled']['mean'] - res['step_observed']['mean'])
        print(f"      -> shuffled vs real mean step differs by "
              f"{dd:.3f} ({100 * dd / res['step_observed']['mean']:.1f}%)")

    print('\n  [5] HEADROOM — how much of the available range is used?')
    print(_row(res['headroom']))
    h = res['headroom']['mean']
    print(f"      -> mean step is {100 * h:.1f}% of the maximum "
          f"compatible with these radii")

    print('\n  VERDICT')
    if ratio > 1.2:
        print('    Observed spread EXCEEDS the isotropic null: the '
              'trajectory is not\n    at a geometric ceiling and '
              'flatness is not forced by normalization.')
    elif ratio < 0.8:
        print('    Observed spread is BELOW the isotropic null: '
              'excursions are being\n    actively suppressed -> '
              'bistability is the lever, not more capacity.')
    else:
        print('    Observed spread MATCHES the isotropic null: step size '
              'is whatever\n    the geometry hands you -> breaking norm '
              'conservation is required.')
    if h < 0.5:
        print(f'    Headroom {100 * h:.0f}% means amplification is '
              f'geometrically available.')
    print('=' * 78)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == '__main__':
    main()
