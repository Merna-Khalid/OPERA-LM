"""Per-token state-trajectory statistics: does the state MOVE differently
when the arriving token is informative?

THE HYPOTHESIS (stated sharply)
-------------------------------
Two days of probing established that OPERA's prefix-state trajectory
advances at near-constant speed per token regardless of token
informativeness (p99.9/median of per-token step ~= 1.46, no heavy tail).
The literature (Barenholtz et al. 2026, arXiv:2606.05346) says the
trajectory statistic that predicts human reading times and garden-path
cost is DIRECTION-CHANGE -- extrapolation error -- not step magnitude.
So the open question is not "does the state jump further on surprising
tokens" (apparently not) but "does it TURN more". This module measures
both, plus controls, on OPERA and a matched RoPE transformer:

    step[t]   = || f_{t+1} - f_t ||            -- magnitude of the jump
    turn[t]   = 1 - cos(f_t - f_{t-1},
                        f_{t+1} - f_t)         -- direction change, t >= 1
    extrap[t] = || f_{t+1} - (f_t + (f_t - f_{t-1})) ||
                / || f_t - f_{t-1} ||          -- constant-velocity
                                                  extrapolation error,
                                                  t >= 1, denominator
                                                  clamped at 1e-12
    surp[t]   = -log p(x_{t+1} | x_0..t)       -- how surprising the
                                                  ARRIVING token is

ALIGNMENT SEMANTICS (exact)
---------------------------
For states f_0..f_{T-1} and tokens x_0..x_{T-1}: surp[t], step[t],
turn[t] and extrap[t] ALL describe the integration of token x_{t+1}
into the prefix state. step is defined for t in [0, T-2]; turn and
extrap need a previous step, so t in [1, T-2]. When correlating
turn/extrap against surprisal the surprisal vector is sliced [:, 1:]
so position indices line up exactly. This matches garden_path.surprisal
(surprisal at t is the NLL of token t+1 given the prefix ending at t).

THE POSITION CONFOUND (controlled by construction, not regression)
------------------------------------------------------------------
OPERA's prefix state at position t is computed by a circuit whose SHAPE
depends on popcount(t+1), so every one of these statistics varies with
position for purely structural reasons. Exactly as in
garden_path.experiment_a: FIXED-LENGTH windows (default T=32) so
position t has an identical circuit in every sequence; correlations are
computed WITHIN each position across sequences and aggregated by Fisher
z (fisher_mean, reused from garden_path); tail statistics and the
pooled correlation matrix are computed on metrics z-scored WITHIN
position before pooling. No cross-position pooling of raw values
anywhere.

CONTROLS
--------
A correlation between step/turn/extrap and surprisal could be driven by
two uninteresting mechanisms, both partialled out (linear partial
correlation, per position, via the inverse correlation matrix):
  * embedding norm: the arriving token's embedding vector norm -- big
    vectors move states more without any "meaning".
  * log unigram frequency of the arriving token, counted on the TRAIN
    split -- rare words are surprising for boring reasons.

Nothing here trains. Everything is measured at inference on frozen
checkpoints.

MAMBA (--arch mamba)
--------------------
External yardstick only: pretrained state-spaces/mamba-130m (129M params)
via HF transformers' pure-PyTorch path, tokenizer EleutherAI/gpt-neox-20b
(the mamba repos ship no tokenizer files). Tokenization is BPE, NOT the
word-level vocab of the other two arches, and detokenized test text
contains literal '<unk>' strings where the word-level vocab was OOV --
so all Mamba statistics are computed on its own tokenization and the
cross-architecture comparison is DISTRIBUTIONAL (tails, correlation
signs), never aligned per-word. No frequency control for Mamba (BPE
unigrams are not comparable to the word-level counts). Additionally
captures per-token delta = softplus(dt_proj output) at every layer --
Mamba's content-adaptive integration step size -- and reports its
within-position correlation with surprisal, the check of whether a
trained adaptive step actually fires on informative tokens.

transformers 5.6.2 note: the repo's config.json uses the original
(d_model, n_layer) naming which this version no longer maps, so the
config is built explicitly (hidden_size=768, num_hidden_layers=24,
vocab_size=50280 -- the checkpoint pads the vocab to a multiple of 8)
and MambaForCausalLM is used because its `backbone.*` structure matches
the original checkpoint's key prefix. Embeddings verified bitwise equal
to the checkpoint's backbone.embedding.weight (tied to lm_head).

Usage:
    python -m opera_lm.trajectory --arch opera --n-batches 8
    python -m opera_lm.trajectory --arch rope  --n-batches 8
    python -m opera_lm.trajectory --arch mamba --n-batches 8
"""
import argparse
import importlib.util
import json
import math
import pickle
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .garden_path import (load_opera, prefix_states, surprisal,
                          fisher_mean, pearson, _restore)

OPERA_CKPT = ('/Users/mernahafez/Documents/OPERA-spinor/runs/'
              'opera_v8_0_pe-none_data-docs_metal_amp_foreach_cmp-default'
              '_gpudata_aux0.25.pt')
ROPE_CKPT = ('/Users/mernahafez/Documents/OPERA-spinor/runs/'
             'transformer_v2_pe-rope_data-docs.pt')


# ----------------------------------------------------------------------------
# Core metric
# ----------------------------------------------------------------------------

def trajectory_metrics(states):
    """The core statistic. states: [N, T, D] prefix states f_0..f_{T-1}.

    Returns dict with
      step   [N, T-1]  step[t]   = ||f_{t+1} - f_t||,        t in [0,T-2]
      turn   [N, T-2]  turn[t]   = 1 - cos(d_{t-1}, d_t),    t in [1,T-2]
      extrap [N, T-2]  normalized constant-velocity error,   t in [1,T-2]
    where d_t = f_{t+1} - f_t is the step vector. turn/extrap are
    indexed by the ARRIVAL position t, i.e. turn[:, k] is turn at
    position t = k+1, aligned with surp[:, k+1].
    """
    f = states.float()
    d = f[:, 1:] - f[:, :-1]                            # [N, T-1, D]
    step = d.norm(dim=-1)                               # [N, T-1]
    prev, cur = d[:, :-1], d[:, 1:]                     # d_{t-1}, d_t
    cos = ((prev * cur).sum(-1)
           / (prev.norm(dim=-1) * cur.norm(dim=-1)).clamp(min=1e-12))
    turn = 1.0 - cos                                    # [N, T-2]
    pred = f[:, 1:-1] + prev                            # f_t + (f_t - f_{t-1})
    extrap = ((f[:, 2:] - pred).norm(dim=-1)
              / prev.norm(dim=-1).clamp(min=1e-12))     # [N, T-2]
    return dict(step=step, turn=turn, extrap=extrap)


# ----------------------------------------------------------------------------
# Position-controlled statistics
# ----------------------------------------------------------------------------

def _within_pos_r(x, y, min_n=8):
    """Pearson r per position. x, y: [N, P]. Returns list (len P)."""
    return [pearson(x[:, t], y[:, t]) if x.shape[0] >= min_n
            else float('nan') for t in range(x.shape[1])]


def _within_pos_partial_r(x, y, zs, min_n=8):
    """Partial r(x, y | zs) per position via the inverse correlation
    matrix (exact for linear partial correlation). Positions with a
    singular/degenerate correlation matrix are skipped (nan)."""
    P = x.shape[1]
    out = []
    for t in range(P):
        cols = [x[:, t], y[:, t]] + [z[:, t] for z in zs]
        M = torch.stack([c.float() for c in cols], dim=1)   # [N, k]
        if M.shape[0] < min_n or M.std(dim=0).min() < 1e-12:
            out.append(float('nan'))
            continue
        Mc = M - M.mean(dim=0, keepdim=True)
        R = (Mc.T @ Mc) / (Mc.norm(dim=0).unsqueeze(1)
                           * Mc.norm(dim=0).unsqueeze(0)).clamp(min=1e-12)
        try:
            Pm = torch.linalg.inv(R + 1e-8 * torch.eye(R.shape[0]))
        except Exception:
            out.append(float('nan'))
            continue
        den = (Pm[0, 0] * Pm[1, 1]).clamp(min=1e-12).sqrt()
        out.append(float((-Pm[0, 1] / den).clamp(-1.0, 1.0)))
    return out


def _zscore_within_pos(x):
    """z-score each column (position) separately; [N, P] -> [N, P]."""
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-12)
    return (x - mu) / sd


def _tails(z, raw):
    """Percentiles of pooled position-z-scored values, plus tail RATIOS
    computed on raw values WITHIN each position (then the median across
    positions). The ratio is not computed on the z-scored pool because
    a z-scored median is ~0 by construction, making p99/median a
    division by noise; the within-position raw ratio keeps the position
    control AND a meaningful scale (comparable to the p99.9/median
    ~= 1.46 constant-speed figure from the raw-space probing)."""
    v = z.flatten().float()
    v = v[torch.isfinite(v)]
    q = torch.tensor([0.5, 0.95, 0.99, 0.999])
    p = torch.quantile(v, q)
    r99, r999 = [], []
    for t in range(raw.shape[1]):
        col = raw[:, t].float()
        if col.numel() < 20:
            continue
        pq = torch.quantile(col, torch.tensor([0.5, 0.99, 0.999]))
        if pq[0] > 1e-12:
            r99.append((pq[1] / pq[0]).item())
            r999.append((pq[2] / pq[0]).item())
    r99t = torch.tensor(r99) if r99 else torch.tensor([float('nan')])
    r999t = torch.tensor(r999) if r999 else torch.tensor([float('nan')])
    return dict(p50=p[0].item(), p95=p[1].item(), p99=p[2].item(),
                p999=p[3].item(),
                p99_over_med=r99t.median().item(),
                p999_over_med=r999t.median().item())


def analyze(step, turn, extrap, surp, embn, freq=None, label='',
            delta=None):
    """All statistics for one architecture.

    step/embn/surp/freq: [N, T-1] (indexed by arrival position t).
    turn/extrap:         [N, T-2] (arrival position t = index+1).
    freq may be None (no frequency control).
    delta (Mamba only):  [N, T-1] mean per-token integration step size,
                         aligned to surp (index k integrated token k+1).
    """
    res = {'label': label, 'n_seqs': step.shape[0]}
    # surp/embn/freq sliced to turn/extrap's position range (t >= 1)
    sup1, emb1 = surp[:, 1:], embn[:, 1:]
    frq1 = freq[:, 1:] if freq is not None else None

    print(f"\n  [{label}] n={step.shape[0]} sequences")
    print(f"    {'metric':<7} {'agg r':>8} {'%r>0':>5} {'r|emb':>8} "
          + (f"{'r|freq':>8} {'r|both':>8}" if freq is not None else ""))
    metrics = [('step', step, surp, embn, freq),
               ('turn', turn, sup1, emb1, frq1),
               ('extrap', extrap, sup1, emb1, frq1)]
    corr = {}
    for name, m, s, e, fq in metrics:
        rs = _within_pos_r(m, s)
        agg, k = fisher_mean(rs)
        frac = sum(1 for r in rs if r == r and r > 0) / max(k, 1)
        pe = _within_pos_partial_r(m, s, [e])
        agg_e, _ = fisher_mean(pe)
        row = dict(agg_r=agg, n_positions=k, frac_positive=frac,
                   partial_r_emb=agg_e)
        line = f"    {name:<7} {agg:>+8.3f} {frac * 100:>4.0f}% {agg_e:>+8.3f} "
        if fq is not None:
            pf = _within_pos_partial_r(m, s, [fq])
            pb = _within_pos_partial_r(m, s, [e, fq])
            agg_f, _ = fisher_mean(pf)
            agg_b, _ = fisher_mean(pb)
            row.update(partial_r_freq=agg_f, partial_r_both=agg_b)
            line += f" {agg_f:>+8.3f}  {agg_b:>+8.3f}"
        print(line)
        corr[name] = row
    res['correlations_vs_surprisal'] = corr

    # Tails on position-z-scored metrics
    zs, zt, ze = (_zscore_within_pos(m) for m in (step, turn, extrap))
    tails = {n: _tails(z, m) for (n, z), m in
             zip((('step', zs), ('turn', zt), ('extrap', ze)),
                 (step, turn, extrap))}
    res['tails_zscored'] = tails
    print(f"\n    tails (percentiles on position-z-scored pool; ratios "
          f"within-position raw)")
    print(f"    {'metric':<7} {'p50':>7} {'p95':>7} {'p99':>7} "
          f"{'p99.9':>7} {'p99/med':>8} {'p99.9/med':>9}")
    for n, tl in tails.items():
        print(f"    {n:<7} {tl['p50']:>7.3f} {tl['p95']:>7.3f} "
              f"{tl['p99']:>7.3f} {tl['p999']:>7.3f} "
              f"{tl['p99_over_med']:>8.2f} {tl['p999_over_med']:>9.2f}")

    # Disproportionality: top/bottom 5% surprisal tokens, common t>=1 range
    zsup = _zscore_within_pos(sup1).flatten()
    pool = {'step': zs[:, 1:].flatten(), 'turn': zt.flatten(),
            'extrap': ze.flatten()}
    k5 = max(1, int(0.05 * zsup.numel()))
    top_i = zsup.argsort(descending=True)[:k5]
    bot_i = zsup.argsort()[:k5]
    contrast = {}
    print(f"\n    disproportionality (top vs bottom 5% surprisal, "
          f"{k5} tokens each, z-scored means)")
    print(f"    {'metric':<7} {'top5%':>8} {'bot5%':>8} {'delta':>8}")
    for n, z in pool.items():
        hi, lo = z[top_i].mean().item(), z[bot_i].mean().item()
        contrast[n] = dict(top5=hi, bottom5=lo, delta=hi - lo)
        print(f"    {n:<7} {hi:>8.3f} {lo:>8.3f} {hi - lo:>+8.3f}")
    res['disproportionality'] = contrast

    if delta is not None:
        rs = _within_pos_r(delta, surp)
        agg, k = fisher_mean(rs)
        pe = _within_pos_partial_r(delta, surp, [embn])
        agg_e, _ = fisher_mean(pe)
        zd = _zscore_within_pos(delta).flatten()
        zsup_full = _zscore_within_pos(surp).flatten()
        hi = zd[zsup_full.argsort(descending=True)[:k5]].mean().item()
        lo = zd[zsup_full.argsort()[:k5]].mean().item()
        res['delta_vs_surprisal'] = dict(
            agg_r=agg, n_positions=k, partial_r_emb=agg_e,
            top5_delta_z=hi, bottom5_delta_z=lo)
        print(f"\n    delta (Mamba step size) vs surprisal: agg r "
              f"{agg:+.3f} over {k} positions, r|emb {agg_e:+.3f}")
        print(f"    delta z at top-5% surprisal {hi:+.3f} / bottom-5% "
              f"{lo:+.3f}")

    # Pooled correlation matrix on the common t>=1 position range
    names = ['surprisal', 'emb_norm'] + (['log_freq'] if freq is not None
                                         else []) + ['step', 'turn', 'extrap']
    cols = [_zscore_within_pos(sup1).flatten(),
            _zscore_within_pos(emb1).flatten()]
    if frq1 is not None:
        cols.append(_zscore_within_pos(frq1).flatten())
    cols += [zs[:, 1:].flatten(), zt.flatten(), ze.flatten()]
    C = torch.stack(cols, dim=1)
    Cc = C - C.mean(dim=0, keepdim=True)
    R = (Cc.T @ Cc) / (Cc.norm(dim=0).unsqueeze(1)
                       * Cc.norm(dim=0).unsqueeze(0)).clamp(min=1e-12)
    res['pooled_corr'] = dict(names=names, matrix=R.tolist())
    print(f"\n    pooled correlation (position-z-scored, t>=1)")
    print(f"    {'':>10}" + ''.join(f"{n[:8]:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"    {n[:10]:>10}" + ''.join(f"{R[i, j]:>9.3f}"
                                             for j in range(len(names))))
    return res


# ----------------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------------

def log_unigram_freq(train_seqs, vocab_size):
    """log unigram frequency per token id from the TRAIN split,
    add-0.5 smoothed. Indexed by token id."""
    counts = torch.zeros(vocab_size)
    for s in train_seqs:
        counts += torch.bincount(torch.tensor(s), minlength=vocab_size
                                 ).float()[:vocab_size]
    total = counts.sum() + 0.5 * vocab_size
    return ((counts + 0.5) / total).log()


def _pick_windows(seqs, T, batch, n_batches, seed):
    usable = [s for s in seqs if len(s) >= T + 1]
    random.Random(seed).shuffle(usable)     # seed selects WHICH windows
    return usable[:batch * n_batches]


@torch.no_grad()
def collect_opera(model, seqs, T=32, batch=32, n_batches=8, seed=0):
    """Prefix states, surprisal, embedding norms for OPERA windows.
    Uses garden_path.prefix_states / surprisal, which swap in a
    _FixedEmb; _restore is called after every batch so word_emb is
    never left patched."""
    wins = _pick_windows(seqs, T, batch, n_batches, seed)
    F_all, S_all, E_all, I_all = [], [], [], []
    for bi in range(0, len(wins), batch):
        chunk = wins[bi:bi + batch]
        if len(chunk) < 2:
            break
        ids = torch.tensor([s[:T] for s in chunk])
        lengths = torch.full((len(chunk),), T, dtype=torch.long)
        emb = _restore(model)
        e0 = emb(ids).detach()
        f = prefix_states(model, e0, ids, lengths)      # [B, T, D]
        sup = surprisal(model, ids, lengths, e0)        # [B, T-1]
        _restore(model)
        F_all.append(f.float())
        S_all.append(sup.float())
        E_all.append(e0[:, 1:].norm(dim=-1).float())    # arriving token
        I_all.append(ids[:, 1:])
        print(f"      batch {bi // batch + 1} "
              f"({sum(x.shape[0] for x in F_all)} seqs)", flush=True)
    return (torch.cat(F_all), torch.cat(S_all), torch.cat(E_all),
            torch.cat(I_all))


def load_rope(ckpt, vocab_size=10000, d=512, nheads=8, num_layers=4):
    """Load the matched RoPE transformer baseline. Its module lives in
    the hyphenated opera-chat/ directory, so it is loaded by file path.
    The checkpoint is a raw state_dict (saved via
    torch.save(model.state_dict()))."""
    path = (Path(__file__).resolve().parent.parent
            / 'opera-chat' / 'opera_transformer_baseline_v2.py')
    spec = importlib.util.spec_from_file_location('tf_baseline_v2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.TransformerBaseline(vocab_size=vocab_size, d=d, nheads=nheads,
                                num_layers=num_layers, pe_mode='rope')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    sd = sd.get('model', sd)
    m.load_state_dict(sd, strict=True)
    return m.eval()


@torch.no_grad()
def collect_rope(model, seqs, T=32, batch=32, n_batches=8, seed=0):
    """Same measurements for the RoPE transformer. f_t is the post-ln_f
    residual stream (captured with a forward hook); surprisal is the NLL
    of token t+1 under its logits, identically aligned to OPERA's."""
    wins = _pick_windows(seqs, T, batch, n_batches, seed)
    buf = []
    hook = model.ln_f.register_forward_hook(
        lambda m, i, o: buf.append(o.detach()))
    F_all, S_all, E_all, I_all = [], [], [], []
    try:
        for bi in range(0, len(wins), batch):
            chunk = wins[bi:bi + batch]
            if len(chunk) < 2:
                break
            ids = torch.tensor([s[:T] for s in chunk])
            lengths = torch.full((len(chunk),), T, dtype=torch.long)
            buf.clear()
            logits = model(ids, lengths)[0]             # one-element list
            f = buf[0]                                  # [B, T, d]
            lp = F.log_softmax(logits.float(), dim=-1)
            sup = -lp[:, :-1].gather(
                -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            F_all.append(f.float())
            S_all.append(sup.float())
            E_all.append(model.word_emb(ids)[:, 1:]
                         .norm(dim=-1).detach().float())
            I_all.append(ids[:, 1:])
            print(f"      batch {bi // batch + 1} "
                  f"({sum(x.shape[0] for x in F_all)} seqs)", flush=True)
    finally:
        hook.remove()
    return (torch.cat(F_all), torch.cat(S_all), torch.cat(E_all),
            torch.cat(I_all))


MAMBA_HF_ID = 'state-spaces/mamba-130m'
MAMBA_TOK_ID = 'EleutherAI/gpt-neox-20b'


def load_mamba():
    """Pretrained mamba-130m, external yardstick. Config built explicitly
    because transformers 5.6.2 no longer maps the repo's (d_model,
    n_layer) config naming and would silently build a 32-layer model
    (see module docstring). Returns (model, tokenizer)."""
    from transformers import MambaForCausalLM, MambaConfig, AutoTokenizer
    cfg = MambaConfig(hidden_size=768, num_hidden_layers=24,
                      vocab_size=50280)
    m = MambaForCausalLM.from_pretrained(MAMBA_HF_ID, config=cfg)
    tok = AutoTokenizer.from_pretrained(MAMBA_TOK_ID)
    return m.eval(), tok


@torch.no_grad()
def collect_mamba(model, tok, seqs, i2w, T=32, batch=32, n_batches=8,
                  seed=0, word_cap=48):
    """Same measurements for Mamba on its OWN BPE tokenization. Test
    sequences are detokenized with i2w ('<unk>' appears literally where
    the word-level vocab was OOV -- recorded in the module docstring),
    re-tokenized, and sliced to fixed T-token windows so the within-
    position protocol is identical to the other arches. Also captures
    per-token delta (softplus of each layer's dt_proj output, averaged
    over the intermediate dim and over layers) -- Mamba's learned
    integration step size."""
    wins = _pick_windows(seqs, T, batch, n_batches, seed)
    # dt_proj output already includes the bias; softplus gives delta.
    dbuf = []
    hooks = [lay.mixer.dt_proj.register_forward_hook(
        lambda m, i, o: dbuf.append(o.detach().float()))
        for lay in model.backbone.layers]
    F_all, S_all, E_all, I_all, D_all = [], [], [], [], []
    try:
        n_done = 0
        for bi in range(0, len(wins), batch):
            chunk = wins[bi:bi + batch]
            if len(chunk) < 2:
                break
            texts = [' '.join(i2w[i] for i in s[:word_cap]) for s in chunk]
            toks = [tok(t).input_ids[:T] for t in texts]
            toks = [t for t in toks if len(t) >= T]
            if len(toks) < 2:
                continue
            ids = torch.tensor(toks)
            dbuf.clear()
            out = model(ids, output_hidden_states=True)
            f = model.backbone.norm_f(out.hidden_states[-1])
            lp = F.log_softmax(out.logits.float(), dim=-1)
            sup = -lp[:, :-1].gather(
                -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # delta: per layer [B, T, inter] -> mean over inter, then
            # over layers; slice [:, 1:] so index k is the step size
            # that integrated token x_{k+1}, aligned with surp[:, k].
            dl = torch.stack([F.softplus(d) for d in dbuf])  # [L,B,T,I]
            dl = dl.mean(dim=-1).mean(dim=0)[:, 1:]          # [B,T-1]
            F_all.append(f.float())
            S_all.append(sup.float())
            E_all.append(model.backbone.embeddings(ids)[:, 1:]
                         .norm(dim=-1).float())
            I_all.append(ids[:, 1:])
            D_all.append(dl)
            n_done += len(ids)
            print(f"      batch {bi // batch + 1} ({n_done} seqs)",
                  flush=True)
            if n_done >= batch * n_batches:
                break
    finally:
        for h in hooks:
            h.remove()
    return (torch.cat(F_all), torch.cat(S_all), torch.cat(E_all),
            torch.cat(I_all), torch.cat(D_all))


def sanity_check(res, step, surp):
    """Invariants that must hold for the numbers to mean anything."""
    ok = True
    if not (torch.isfinite(surp).all() and (surp > 0).all()):
        print("    ** FAIL: surprisal not finite-and-positive")
        ok = False
    if not (torch.isfinite(step).all() and (step >= 0).all()):
        print("    ** FAIL: step not finite-and-nonnegative")
        ok = False
    for name, row in res['correlations_vs_surprisal'].items():
        r = row['agg_r']
        if r == r and abs(r) > 1.0:
            print(f"    ** FAIL: aggregate r for {name} outside [-1,1]")
            ok = False
    print(f"    sanity: {'PASS' if ok else 'FAIL'} "
          f"(surprisal finite>0, step finite>=0, |agg r|<=1)")
    return ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch', choices=('opera', 'rope', 'mamba'),
                   required=True)
    p.add_argument('--ckpt', default=None,
                   help='default: the project 22M checkpoint for the arch')
    p.add_argument('--vocab', default='assets/vocab_docs.pkl')
    p.add_argument('--T', type=int, default=32)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--n-batches', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', default=None)
    # arch=opera model config (defaults = the 22M headline checkpoint;
    # arm-eval checkpoints are the 5.9M rung: --d 256 --nb 64 --layers 2
    # --rot-mode free, plus their arm flag)
    p.add_argument('--d', type=int, default=640)
    p.add_argument('--nb', type=int, default=160)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--rot-mode', default='so3')
    p.add_argument('--homeo', default='off')
    p.add_argument('--node-paths', type=int, default=3)
    p.add_argument('--fold-adapt', default='off')
    args = p.parse_args()

    tr, ts, tl, vocab, w2i, i2w = pickle.load(open(args.vocab, 'rb'))
    print('=' * 74)
    print(f'TRAJECTORY STATISTICS — {args.arch}  '
          f'(T={args.T}, batch={args.batch}, n_batches={args.n_batches}, '
          f'seed={args.seed})')
    print('=' * 74)
    print('  measured at inference on a frozen checkpoint; nothing trains')

    if args.arch == 'opera':
        ckpt = args.ckpt or OPERA_CKPT
        model = load_opera(ckpt, d=args.d, nb=args.nb, layers=args.layers,
                           rot_mode=args.rot_mode, homeo_mode=args.homeo,
                           node_paths=args.node_paths,
                           fold_adapt=args.fold_adapt)
        print(f"  model: {ckpt}\n         "
              f"{sum(q.numel() for q in model.parameters()):,} params")
        F_, S_, E_, I_ = collect_opera(model, ts, T=args.T, batch=args.batch,
                                       n_batches=args.n_batches,
                                       seed=args.seed)
        # the real embedding must be back in place after collection
        from .garden_path import _FixedEmb
        assert not isinstance(model.word_emb, _FixedEmb), \
            'word_emb left patched!'
        D_ = None
        logf = log_unigram_freq(tr, len(vocab))
        freq = logf[I_]                              # [N, T-1]
    elif args.arch == 'rope':
        ckpt = args.ckpt or ROPE_CKPT
        model = load_rope(ckpt)
        print(f"  model: {ckpt}\n         "
              f"{sum(q.numel() for q in model.parameters()):,} params")
        F_, S_, E_, I_ = collect_rope(model, ts, T=args.T, batch=args.batch,
                                      n_batches=args.n_batches,
                                      seed=args.seed)
        D_ = None
        logf = log_unigram_freq(tr, len(vocab))
        freq = logf[I_]
    else:
        model, tok = load_mamba()
        print(f"  model: {MAMBA_HF_ID} (tokenizer {MAMBA_TOK_ID})\n"
              f"         {sum(q.numel() for q in model.parameters()):,} "
              f"params")
        F_, S_, E_, I_, D_ = collect_mamba(model, tok, ts, i2w, T=args.T,
                                           batch=args.batch,
                                           n_batches=args.n_batches,
                                           seed=args.seed)
        freq = None     # BPE unigrams not comparable to word counts

    m = trajectory_metrics(F_)
    res = analyze(m['step'], m['turn'], m['extrap'], S_, E_, freq=freq,
                  label=args.arch, delta=D_)
    res['raw_step_median'] = m['step'].median().item()
    res['raw_step_mean'] = m['step'].mean().item()
    res['raw_surp_mean'] = S_.mean().item()
    res['sanity_pass'] = sanity_check(res, m['step'], S_)
    print(f"\n    raw step: median {res['raw_step_median']:.3f}, "
          f"mean {res['raw_step_mean']:.3f}; "
          f"raw surprisal mean {res['raw_surp_mean']:.3f} nats")
    print('=' * 74)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'config': vars(args), 'results': res}, f,
                      indent=2, default=float)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
