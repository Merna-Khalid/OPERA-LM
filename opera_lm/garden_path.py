"""Do words that change meaning destabilise the composition?

THE HYPOTHESIS (the user's, stated sharply)
-------------------------------------------
Language is not a smooth manifold. Most words refine meaning
incrementally, but some force REINTERPRETATION of what came before --
a garden path ("the horse raced past the barn FELL"), a negation, a
discourse pivot. If that is real, the map from input to composed prefix
state should be locally NON-SMOOTH at exactly those words: a small
change in the input produces a disproportionately nonlinear change in
the state. A "blow-up."

That is a measurable claim, and `opera_lm.curvature`'s finite-difference
proxy measures exactly it. Here it is applied PER TOKEN:

    rho_t = || f_t(e + h u_t) - 2 f_t(e) + f_t(e - h u_t) ||
            -----------------------------------------------
                    || f_t(e + h u_t) - f_t(e - h u_t) ||

where u_t perturbs ONLY position t's embedding and f_t is the prefix
state at position t. Zero for an affine map, scale-invariant, and (per
`curvature.h_linearity_check`) only valid while rho/h stays constant.

THE CONFOUND THAT WOULD FAKE A RESULT
-------------------------------------
OPERA's prefix state at position t is computed by a circuit whose SHAPE
depends on popcount(t+1) -- position 7 folds 3 Fenwick blocks, position
8 folds 1. So rho varies with position for purely STRUCTURAL reasons,
with no linguistic content whatsoever. Any analysis that pools across
positions will find "structure" that is just the binary expansion of t.

Both experiments below control for it by construction:

  A. CURVATURE vs SURPRISAL (high power). Fixed-length windows, so
     position t has an IDENTICAL circuit in every sentence. Correlation
     is computed WITHIN each position across sentences, then aggregated
     by Fisher z. Position is held exactly constant, not regressed out.

  B. GARDEN-PATH MINIMAL PAIRS (sharp, low power). Sentences identical
     up to the critical word, which is swapped for one that does or does
     not force reanalysis. Same length, same position, same prefix, same
     circuit -- the only difference is which word arrives.

Experiment A is the real test; B is the vivid one. B on a small
word-level model may simply be underpowered -- garden-path effects
require syntax the model may not have -- and that is reported, not
hidden.
"""
import argparse
import json
import math
import pickle

import torch
import torch.nn.functional as F

from .model import OperaSpinorFenwickTree


class _FixedEmb(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, ids):
        return self.t


def load_opera(ckpt, vocab_size=10000, d=640, nb=160, layers=4,
               rot_mode='so3', **model_kw):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    m = OperaSpinorFenwickTree(vocab_size=vocab_size, d=d, nb=nb,
                               num_layers=layers, pe_mode='none',
                               fold_mode='left', rot_mode=rot_mode,
                               **model_kw)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    return m.eval()


@torch.no_grad()
def prefix_states(model, e, ids, lengths):
    model.word_emb = _FixedEmb(e)
    return model(ids, lengths, return_states=True).states[-1]


@torch.no_grad()
def surprisal(model, ids, lengths, emb):
    """Per-token NLL of the ACTUAL next token: surprisal at position t is
    -log p(token_{t+1} | tokens_0..t), i.e. how unexpected the word that
    arrives after prefix t is. Aligned to rho_t, which measures the
    stability of that same prefix state."""
    model.word_emb = _FixedEmb(emb)
    lg = model(ids, lengths).logits[-1]
    lp = F.log_softmax(lg.float(), dim=-1)
    tgt = ids[:, 1:]
    return -lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [B,T-1]


def _restore(model):
    """Put the real embedding back. per_token_curvature swaps in a
    _FixedEmb to perturb the input; every entry point restores first so a
    later call can never silently reuse a previous batch's embeddings."""
    if isinstance(model.word_emb, _FixedEmb):
        model.word_emb = model._true_emb
    else:
        model._true_emb = model.word_emb
    return model.word_emb


@torch.no_grad()
def per_token_curvature(model, ids, lengths, n_dirs=2, h_frac=1e-2,
                        seed=0):
    """rho at every position; perturbs ONE position at a time."""
    g = torch.Generator().manual_seed(seed)
    emb = _restore(model)
    e0 = emb(ids).detach()
    B, T, d = e0.shape
    base = prefix_states(model, e0, ids, lengths)
    rho = torch.zeros(B, T)
    for t in range(T):
        acc = []
        h = h_frac * e0[:, t].norm(dim=-1, keepdim=True)      # [B,1]
        for _ in range(n_dirs):
            u = torch.randn(B, d, generator=g)
            u = u / u.norm(dim=-1, keepdim=True)
            ep, em = e0.clone(), e0.clone()
            ep[:, t] = ep[:, t] + h * u
            em[:, t] = em[:, t] - h * u
            fp = prefix_states(model, ep, ids, lengths)[:, t]
            fm = prefix_states(model, em, ids, lengths)[:, t]
            f0 = base[:, t]
            num = (fp - 2 * f0 + fm).norm(dim=-1)
            den = (fp - fm).norm(dim=-1).clamp(min=1e-12)
            acc.append(num / den)
        rho[:, t] = torch.stack(acc).mean(0)
    _restore(model)
    return rho


def fisher_mean(rs):
    """Average correlations properly (Fisher z), not by naive mean."""
    rs = [r for r in rs if r is not None and abs(r) < 0.999
          and not math.isnan(r)]
    if not rs:
        return float('nan'), 0
    z = [0.5 * math.log((1 + r) / (1 - r)) for r in rs]
    zb = sum(z) / len(z)
    return (math.exp(2 * zb) - 1) / (math.exp(2 * zb) + 1), len(rs)


def pearson(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / den)


def experiment_a(model, seqs, T=32, batch=32, n_batches=4, n_dirs=2,
                 h_frac=1e-2, seed=0):
    """Curvature vs surprisal, correlated WITHIN each position."""
    usable = [s for s in seqs if len(s) >= T + 1]
    print(f"    {len(usable):,} sequences of length >= {T + 1}")
    per_pos_rho, per_pos_sup = [[] for _ in range(T - 1)], \
                               [[] for _ in range(T - 1)]
    n = 0
    for bi in range(n_batches):
        chunk = usable[bi * batch:(bi + 1) * batch]
        if len(chunk) < 2:
            break
        ids = torch.tensor([s[:T] for s in chunk])
        lengths = torch.full((len(chunk),), T, dtype=torch.long)
        e0 = _restore(model)(ids).detach()
        rho = per_token_curvature(model, ids, lengths, n_dirs=n_dirs,
                                  h_frac=h_frac, seed=seed + bi)
        sup = surprisal(model, ids, lengths, e0)
        _restore(model)
        for t in range(T - 1):
            per_pos_rho[t] += rho[:, t].tolist()
            per_pos_sup[t] += sup[:, t].tolist()
        n += len(chunk)
        print(f"      batch {bi + 1}/{n_batches} ({n} seqs)", flush=True)

    rs = []
    print(f"\n    within-position correlation (n={n} per position)")
    print(f"    {'pos':>4}  {'mean rho':>9}  {'mean surp':>9}  {'r':>7}")
    for t in range(T - 1):
        if len(per_pos_rho[t]) < 8:
            continue
        r = pearson(per_pos_rho[t], per_pos_sup[t])
        rs.append(r)
        mr = sum(per_pos_rho[t]) / len(per_pos_rho[t])
        ms = sum(per_pos_sup[t]) / len(per_pos_sup[t])
        if t % 4 == 0 or abs(r) > 0.3:
            print(f"    {t:>4}  {mr:>9.4f}  {ms:>9.4f}  {r:>7.3f}")
    agg, k = fisher_mean(rs)
    pos_frac = sum(1 for r in rs if r > 0) / max(len(rs), 1)
    print(f"\n    Fisher-z aggregate r = {agg:+.4f}  over {k} positions")
    print(f"    positions with r > 0: {pos_frac * 100:.0f}%")
    return dict(aggregate_r=agg, n_positions=k, n_seqs=n,
                frac_positive=pos_frac, per_position_r=rs)


# Reduced-relative garden paths. Each pair shares an IDENTICAL prefix and
# length; only the final word differs, so the Fenwick circuit at the
# critical position is bitwise the same for both members. The garden-path
# member forces reanalysis of the preceding participle from main verb to
# reduced relative ("raced past the house" -> "[that was] raced past the
# house"); the control member does not. All words verified in-vocabulary
# (no <unk>), since an <unk> would confound the comparison entirely.
GP_PAIRS = [
    ("the horse raced past the house", "fell",   "today"),
    ("the soldier sent to the front",  "died",   "lines"),
    ("the ship sailed down the river", "sank",   "again"),
    ("the man given the book",         "left",   "today"),
    ("the woman brought the flowers",  "died",   "home"),
    ("the player passed the ball",     "fell",   "back"),
    ("the dog walked in the park",     "died",   "again"),
    ("the boy told the story",         "left",   "again"),
]


@torch.no_grad()
def experiment_b(model, w2i, n_dirs=8, h_frac=1e-2, seed=0):
    """Garden-path minimal pairs: curvature at the critical word."""
    rows = []
    print(f"    {'prefix':<34} {'crit':<7} {'rho':>7} {'ctl':<7} "
          f"{'rho':>7} {'ratio':>6} {'surp GP':>8} {'surp CT':>8}")
    for prefix, gp_w, ct_w in GP_PAIRS:
        toks = prefix.split()
        if any(w not in w2i for w in toks + [gp_w, ct_w]):
            print(f"    SKIP (oov): {prefix} / {gp_w} / {ct_w}")
            continue
        seqs = [[w2i[w] for w in toks] + [w2i[gp_w]],
                [[w2i[w] for w in toks] + [w2i[ct_w]]][0]]
        ids = torch.tensor(seqs)
        T = ids.shape[1]
        lengths = torch.full((2,), T, dtype=torch.long)
        crit = T - 1
        e0 = _restore(model)(ids).detach()
        rho = per_token_curvature(model, ids, lengths, n_dirs=n_dirs,
                                  h_frac=h_frac, seed=seed)
        sup = surprisal(model, ids, lengths, e0)
        _restore(model)
        g_r, c_r = rho[0, crit].item(), rho[1, crit].item()
        g_s, c_s = sup[0, crit - 1].item(), sup[1, crit - 1].item()
        rows.append(dict(prefix=prefix, gp=gp_w, ctl=ct_w, rho_gp=g_r,
                         rho_ctl=c_r, ratio=g_r / max(c_r, 1e-12),
                         surp_gp=g_s, surp_ctl=c_s))
        print(f"    {prefix:<34} {gp_w:<7} {g_r:>7.4f} {ct_w:<7} "
              f"{c_r:>7.4f} {g_r / max(c_r, 1e-12):>6.2f} {g_s:>8.3f} "
              f"{c_s:>8.3f}")
    if rows:
        wins = sum(1 for r in rows if r['rho_gp'] > r['rho_ctl'])
        mr = sum(r['ratio'] for r in rows) / len(rows)
        ds = sum(r['surp_gp'] - r['surp_ctl'] for r in rows) / len(rows)
        print(f"\n    GP more curved in {wins}/{len(rows)} pairs; "
              f"mean ratio {mr:.2f}x")
        print(f"    mean surprisal delta (GP - control): {ds:+.3f} nats")
        print(f"    NOTE: {len(rows)} pairs is not a powered test. A sign "
              f"count this small\n          is descriptive only.")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default='/Users/mernahafez/Documents/'
                   'OPERA-spinor/runs/opera_v8_0_pe-none_data-docs_metal'
                   '_amp_foreach_cmp-default_gpudata_aux0.25.pt')
    p.add_argument('--vocab', default='/private/tmp/claude-501/-Users-'
                   'mernahafez-Documents-OPERA-spinor-opera-lm/'
                   'e1d30ac9-a0cb-45dd-99d6-d0a63b7bcff7/scratchpad/'
                   'vocab_docs.pkl')
    p.add_argument('--d', type=int, default=640)
    p.add_argument('--nb', type=int, default=160)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--rot-mode', default='so3')
    p.add_argument('--train-len', type=int, default=32)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--n-batches', type=int, default=4)
    p.add_argument('--n-dirs', type=int, default=2)
    p.add_argument('--h-frac', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    tr, ts, tl, vocab, w2i, i2w = pickle.load(open(args.vocab, 'rb'))
    model = load_opera(args.ckpt, d=args.d, nb=args.nb,
                       layers=args.layers, rot_mode=args.rot_mode)
    print('=' * 74)
    print('GARDEN-PATH / BLOW-UP TEST — is composition unstable at '
          'meaning-changing words?')
    print('=' * 74)
    print(f"  model: d={args.d} nb={args.nb} L={args.layers} "
          f"rot={args.rot_mode}  "
          f"{sum(q.numel() for q in model.parameters()):,} params")

    # h-validity: rho must be second-order here too, or nothing is
    # comparable (the rack artifact from opera_lm.curvature).
    print('\n  [0] h-validity check (rho/h must be ~constant)')
    ids = torch.tensor([s[:args.train_len] for s in ts
                        if len(s) >= args.train_len + 1][:8])
    lengths = torch.full((ids.shape[0],), args.train_len,
                         dtype=torch.long)
    ratios = []
    for hf in [2.5e-3, 5e-3, 1e-2, 2e-2]:
        r = per_token_curvature(model, ids, lengths, n_dirs=2,
                                h_frac=hf, seed=0).mean().item()
        ratios.append(r / hf)
        print(f"      h_frac={hf:<7.4f}  mean rho={r:.5f}  "
              f"rho/h={r / hf:.3f}")
    drift = max(ratios) / min(ratios)
    print(f"      drift {drift:.2f}x  -> "
          f"{'VALID' if drift < 1.35 else '** INVALID, ABORTING **'}")
    if drift >= 1.35:
        return

    print('\n  [A] CURVATURE vs SURPRISAL (within-position, real text)')
    a = experiment_a(model, ts, T=args.train_len, batch=args.batch,
                     n_batches=args.n_batches, n_dirs=args.n_dirs,
                     h_frac=args.h_frac, seed=args.seed)

    print('\n  [B] GARDEN-PATH MINIMAL PAIRS')
    b = experiment_b(model, w2i, h_frac=args.h_frac, seed=args.seed)
    print('=' * 74)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'config': vars(args), 'experiment_a': a,
                       'experiment_b': b}, f, indent=2, default=float)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
