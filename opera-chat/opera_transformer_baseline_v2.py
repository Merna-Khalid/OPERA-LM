"""
MATCHED TRANSFORMER BASELINE v2 -- the anchor for every OPERA v8.0 claim.

Decoder-only pre-LN transformer trained under the EXACT OPERA v8.0
protocol. Identity is enforced by construction, not by promise: this
file IMPORTS load_data, lm_loss, compute_perplexity, extrapolation_eval,
get_lr, make_batch_full and count_params from opera_v8_spinor.py
(must sit in the same directory), so the corpus, the split, the vocab,
the loss masking, the PPL definition, the buckets, the lr schedule and
the seed protocol are the same OBJECTS, and cannot drift.

t3 (2026-07-19), for the v8.4 SCALE RUNG:
  --data docs-en accepted (full English Wikipedia via the shared opt6+
    loader; long-first chunk schedule powers the 1025-4096 buckets).
  --lr <f> / --warmup <n>  (defaults 1e-3 / 500 = historical recipe).
    The scale gate REQUIRES the recipe matched across arms: run both
    S1 and S2 with --lr 3e-4 --warmup 2000. Tags record non-defaults so
    checkpoint filenames distinguish recipes.
  --docs-limit <n> threaded to the shared loader (default 100000,
    matching S1) and recorded in the results jsonl.
  RESUME FIX (latent, same class as OPERA's opt7b): cuda_rng restored
    via .cpu() -- torch.load(map_location=device) moves the saved RNG
    ByteTensor to CUDA and set_rng_state requires CPU.
  NOTE for P2 (efficiency): S2 runs eager with SDPA (the standard
    strong transformer implementation); S1 runs torch.compile. Both are
    "best readily available" for their architecture; state this beside
    any timing claim.

PE arms (one flag each):
  --pe rope     (default) rotary position embeddings -- the strong
                baseline; extrapolates by formula.
  --pe nope     NO positional encoding (Haviv et al. 2022; Kazemnejad
                et al. 2023). THE comparison arm for OPERA pe-none:
                NoPE's position source is implicit/emergent from the
                causal mask; OPERA's is explicit/constructive from the
                Fenwick decomposition. Same claim, different mechanism.
  --pe sin      additive sinusoidal (the classic extrapolation liability).
  --pe learned  learned absolute embeddings, table sized eval_max_len;
                positions past train length are simply untrained.

Parameter matching: at V=10000, d=512, 8 heads, 4 layers, ffn 4x,
untied embedding + biased head (matching OPERA's untied head+bias):
~22.8M vs OPERA's 22.25M (left) / 22.58M (attend) -- matched within
~3%. Scale rung: V=32768, d=1024, 16 heads, 6 layers -> ~142.9M vs
OPERA S1's 140.5M (+1.7%). The run banner prints both so every log
self-documents the match. Use --ffn-mult to trim if exact matching is
wanted.

PROBE MODE: --probe CKPT runs the v8.0 interrogation probe + position
curve on a trained baseline checkpoint (imports the machinery from
opera_v8_probe.py; the transformer returns [logits] so the probe code
runs unchanged). This produces the transformer anchor for the
prefix-255 sensitivity ratios -- the number that decides whether
OPERA's early-token collapse is an architecture property or a corpus
property.

Scale-rung quickstart (opera_v8_spinor.py = opt7b+ alongside):
  !python opera_transformer_baseline_v2.py --selftest
  !python opera_transformer_baseline_v2.py 30000 8 1024 32768 1024 16 6 4096 \
      --pe nope --data docs-en --lr 3e-4 --warmup 2000 --amp --seed 1 \
      --out /content/drive/MyDrive/opera --save-every 1000 --resume

Usage:
  python3 opera_transformer_baseline_v2.py <steps> <batch> <max_len> <vocab_size> <d> <nheads> <num_layers> <eval_max_len> [flags]
  defaults: steps=20000, batch=16, max_len=256, vocab_size=10000, d=512, nheads=8, num_layers=4, eval_max_len=2048
  NOTE the arg order differs from OPERA in one slot: <nheads> where OPERA has <nb>.

Flags: --pe rope|nope|sin|learned  --seed <n>  --data sentences|docs|docs-en
       --docs-limit <n>  --lr <f>  --warmup <n>
       --ffn-mult <f>  --dropout <p>  --amp
       --out <dir>  --save-every <n>  --resume
       --probe <ckpt>  --prefixes a,b,c  --nseq <n>  --resamples <n>  --tag <s>
"""
BASELINE_VERSION = "2026-07-19-t3"
print(f"[baseline {BASELINE_VERSION}] opera_transformer_baseline_v2.py", flush=True)
import os
import sys
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# load_data / make_batch_full / lm_loss / compute_perplexity /
# extrapolation_eval / get_lr / count_params used to come from the legacy
# opera_v8_spinor_optimized.py (not part of this repo). TransformerBaseline
# itself only needs sinusoidal_pos_enc (below); the other names were only
# used by this file's own standalone main()/selftest(), which callers of
# TransformerBaseline (e.g. train_tf_chat.py) don't invoke -- they bring
# their own copies of those functions from opera_lm.train/opera_lm.losses.


def sinusoidal_pos_enc(T, d, device):
    position = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                         * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
    return pe


# ============================================================================
# ROTARY EMBEDDINGS
# ============================================================================

def rope_tables(T, head_dim, device, base=10000.0):
    freqs = base ** (-torch.arange(0, head_dim, 2, device=device,
                                   dtype=torch.float32) / head_dim)
    t = torch.arange(T, device=device, dtype=torch.float32)
    ang = torch.outer(t, freqs)                                  # [T, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, hd]. Rotate consecutive even/odd pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None]                                          # [1,1,T,hd/2]
    s = sin[None, None]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


# ============================================================================
# MODEL
# ============================================================================

class Block(nn.Module):
    def __init__(self, d, nheads, ffn_mult, dropout):
        super().__init__()
        self.nheads = nheads
        self.hd = d // nheads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        f = int(d * ffn_mult)
        self.ffn = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))
        self.dropout = dropout

    def forward(self, x, rope):
        B, T, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nheads, self.hd)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # [B,H,T,hd]
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        att = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        att = att.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(att)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerBaseline(nn.Module):
    """Decoder-only pre-LN transformer. forward() returns [logits] -- a
    ONE-ELEMENT LIST -- so OPERA's lm_loss / compute_perplexity /
    extrapolation_eval and the v8.0 probe run on it unchanged."""

    def __init__(self, vocab_size, d=512, nheads=8, num_layers=4,
                 pe_mode='rope', max_pe_len=2048, ffn_mult=4.0, dropout=0.0,
                 tie=False):
        super().__init__()
        assert d % nheads == 0
        assert pe_mode in ('rope', 'nope', 'sin', 'learned')
        self.pe_mode = pe_mode
        self.d = d
        self.nheads = nheads
        self.hd = d // nheads
        self.word_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        if pe_mode == 'learned':
            self.pos_emb = nn.Embedding(max_pe_len, d)
        else:
            self.pos_emb = None
        self.blocks = nn.ModuleList(
            [Block(d, nheads, ffn_mult, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)     # untied + bias, like OPERA
        if tie:
            # tie=True: share the embedding weight with the head, with the
            # same fix OPERA uses (model.py): a learnable logit scale init
            # at d**-0.5, because nn.Embedding's N(0,1) init makes raw
            # tied logits huge (observed: loss 463 at init vs ln(V) 9.7).
            self.head.weight = self.word_emb.weight
            self.logit_scale = nn.Parameter(torch.tensor(d ** -0.5))
        else:
            self.logit_scale = None
        self.dropout = dropout
        self._rope_cache = {}

    def forward(self, token_ids, lengths, **kwargs):
        B, T = token_ids.shape
        device = token_ids.device
        x = self.word_emb(token_ids)
        if self.pe_mode == 'sin':
            x = x + sinusoidal_pos_enc(T, self.d, device).unsqueeze(0)
        elif self.pe_mode == 'learned':
            pos = torch.arange(T, device=device).clamp(
                max=self.pos_emb.num_embeddings - 1)
            x = x + self.pos_emb(pos).unsqueeze(0)
        # 'rope' enters inside attention; 'nope' adds nothing
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        rope = None
        if self.pe_mode == 'rope':
            key = (T, str(device))
            if key not in self._rope_cache:
                self._rope_cache[key] = rope_tables(T, self.hd, device)
            rope = self._rope_cache[key]
        for blk in self.blocks:
            x = blk(x, rope)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        return [logits]                         # list: protocol-compatible


# ============================================================================
# SELF-TEST (no dataset needed)
# ============================================================================

def selftest():
    torch.manual_seed(0)
    print("=== transformer baseline v2 self-test (t3) ===")

    # parameter accounting vs OPERA references (22M rung + SCALE rung)
    for d, nh, L, fm, V, ref, refname in (
            (512, 8, 4, 4.0, 10000, 22254488, '22M-rung OPERA left'),
            (1024, 16, 6, 4.0, 32768, 140514700, 'S1 (140M scale rung)')):
        m = TransformerBaseline(vocab_size=V, d=d, nheads=nh,
                                num_layers=L, ffn_mult=fm)
        n = count_params(m)
        print(f"  d={d} h={nh} L={L} V={V}: {n:,} params "
              f"(vs {refname} {ref:,}; delta {100*(n-ref)/ref:+.1f}%)")
        assert abs(n - ref) / ref < 0.05, "param match drifted past 5%"

    # rope: rotation preserves per-pair norm; identity at t=0
    cos, sin = rope_tables(16, 64, 'cpu')
    x = torch.randn(2, 4, 16, 64)
    xr = apply_rope(x, cos, sin)
    nrm = (x.norm(dim=-1) - xr.norm(dim=-1)).abs().max().item()
    t0 = (x[:, :, 0] - xr[:, :, 0]).abs().max().item()
    print(f"  rope: isometry err {nrm:.2e}, identity-at-t0 err {t0:.2e}")
    assert nrm < 1e-4 and t0 < 1e-6

    # every PE arm: forward/backward finite + EXACT causality, list output
    for pm in ('rope', 'nope', 'sin', 'learned'):
        m = TransformerBaseline(vocab_size=211, d=64, nheads=4, num_layers=2,
                                pe_mode=pm, max_pe_len=64)
        tok = torch.randint(1, 211, (3, 13))
        lens = torch.tensor([13, 7, 5])
        out = m(tok, lens)
        assert isinstance(out, list) and len(out) == 1
        loss, _, _ = lm_loss(out, tok, lens)
        loss.backward()
        assert torch.isfinite(loss)
        m.eval()
        tok = torch.randint(1, 211, (1, 12))
        lens = torch.tensor([12])
        with torch.no_grad():
            a = m(tok, lens)[-1][0, :8].clone()
            tok2 = tok.clone(); tok2[0, 10] = (tok2[0, 10] + 5) % 210 + 1
            b = m(tok2, lens)[-1][0, :8]
        err = (a - b).abs().max().item()
        print(f"  pe={pm}: loss {loss.item():.3f}, causality err {err:.2e}")
        assert err < 1e-5

    # probe interface compatibility (imports the real probe machinery);
    # SKIPPED gracefully if opera_v8_probe.py is absent (train mode does
    # not need it)
    try:
        from opera_v8_probe import interrogation_probe, position_curve
    except ImportError:
        print("  (opera_v8_probe.py not present; probe-interface test "
              "skipped -- training mode unaffected)")
    else:
        m = TransformerBaseline(vocab_size=211, d=64, nheads=4, num_layers=2,
                                pe_mode='rope')
        m.eval()
        rng = random.Random(7)
        seqs = [[rng.randrange(4, 211) for _ in range(40)] for _ in range(16)]
        curve, B = interrogation_probe(m, seqs, Lp=31, vocab_size=211,
                                       device='cpu', n_seq=16, resamples=2)
        assert all(np.isfinite(v) and v > 0 for v in curve.values())
        bands, n = position_curve(m, seqs, max_pos=40, device='cpu',
                                  n_seq=16, band=16)
        assert all(np.isfinite(v) for v in bands.values())
        print(f"  probe machinery runs on transformer: {len(curve)} "
              f"positions, {len(bands)} bands -- OK")

    print("  ALL PASS")


# ============================================================================
# TRAINING (mirrors OPERA v8.0 train() step for step)
# ============================================================================

def train(steps, batch, max_len, vocab_size, d, nheads, num_layers,
          eval_max_len, device, pe_mode='rope', ffn_mult=4.0, dropout=0.0,
          seed=42, data_mode='docs', docs_limit=100000, out_dir='.',
          save_every=0, resume=False, use_amp=False,
          max_lr=1e-3, warmup_steps=500):
    tag = [f'pe-{pe_mode}']
    if data_mode != 'sentences': tag.append(f'data-{data_mode}')
    if data_mode == 'docs-en' and docs_limit != 100000:
        tag.append(f'lim{docs_limit}')
    if seed != 42: tag.append(f'seed{seed}')
    if dropout > 0: tag.append(f'drop{dropout}')
    if ffn_mult != 4.0: tag.append(f'ffn{ffn_mult}')
    if use_amp: tag.append('amp')
    if max_lr != 1e-3: tag.append(f'lr{max_lr}')
    if warmup_steps != 500: tag.append(f'wu{warmup_steps}')
    tag = '+'.join(tag)

    print(f"=== TRANSFORMER BASELINE v2 [{tag}] ===", flush=True)
    print(f"  steps={steps}, batch={batch}, train max_len={max_len}, "
          f"eval_max_len={eval_max_len}", flush=True)
    print(f"  d={d}, nheads={nheads}, num_layers={num_layers}, "
          f"ffn_mult={ffn_mult}, device={device}", flush=True)
    print(f"  pe={pe_mode}, seed={seed}, data={data_mode}, "
          f"lr={max_lr}, warmup={warmup_steps}", flush=True)
    print(f"  OPERA v8.0 references (same protocol, docs, 20k steps): "
          f"so3+left 42.58 | free+left 41.53 | so3+attend 43.78 (5k eval)",
          flush=True)

    os.makedirs(out_dir, exist_ok=True)
    train_data, test_short, test_long, vocab, w2i, i2w = load_data(
        vocab_size, max_len, eval_max_len, data_mode=data_mode,
        docs_limit=docs_limit)
    V = len(vocab)

    # Variation seed AFTER load_data (identical rule to OPERA r15).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = TransformerBaseline(V, d=d, nheads=nheads, num_layers=num_layers,
                                pe_mode=pe_mode, max_pe_len=eval_max_len,
                                ffn_mult=ffn_mult, dropout=dropout).to(device)
    npar = count_params(model)
    print(f"  Model params: {npar:,} "
          f"(OPERA: 22,254,488 left @22M rung / 140,514,700 S1 @scale rung)",
          flush=True)

    warmup = warmup_steps
    opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    amp_dtype = torch.bfloat16 if device == 'cuda' else torch.float16
    if use_amp:
        print(f"  AMP dtype: {amp_dtype}", flush=True)
    import contextlib
    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type=device, dtype=amp_dtype)
        return contextlib.nullcontext()

    train_ckpt = os.path.join(
        out_dir, f'transformer_v2_{tag.replace("+","_")}_train_ckpt.pt')
    start_step = 0
    if resume and os.path.exists(train_ckpt):
        st = torch.load(train_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        start_step = st['step'] + 1
        random.setstate(st['py_rng'])
        np.random.set_state(st['np_rng'])
        torch.set_rng_state(st['torch_rng'].cpu())
        if device == 'cuda' and st.get('cuda_rng') is not None:
            # t3 FIX (same class as OPERA opt7b): map_location moved the
            # saved RNG ByteTensor to CUDA; set_rng_state needs CPU.
            torch.cuda.set_rng_state(st['cuda_rng'].cpu())
        print(f"  RESUMED from {train_ckpt} at step {start_step}", flush=True)

    init_ppl = compute_perplexity(model, test_short[:200], max_len, 32, device)
    print(f"  Initial per-token perplexity: {init_ppl:.2f} (chance ~ {V})",
          flush=True)

    t0 = time.time()
    for step in range(start_step, steps):
        lr = get_lr(step, warmup, steps, max_lr)
        for g in opt.param_groups:
            g['lr'] = lr

        batch_sents = random.sample(train_data, min(batch, len(train_data)))
        token_ids, lengths = make_batch_full(batch_sents, max_len)
        token_ids, lengths = token_ids.to(device), lengths.to(device)

        with amp_ctx():
            all_logits = model(token_ids, lengths)
            loss, _, _ = lm_loss(all_logits, token_ids, lengths)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            done = step - start_step + 1
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.5f}  "
                  f"({elapsed:.1f}s, {elapsed/done:.2f}s/step)", flush=True)

        if step % 1000 == 0 and step > 0:
            ppl = compute_perplexity(model, test_short[:200], max_len, 32, device)
            print(f"    per-token perplexity: {ppl:.2f}", flush=True)

        if save_every and step > 0 and step % save_every == 0:
            torch.save({
                'model': model.state_dict(), 'opt': opt.state_dict(),
                'step': step, 'tag': tag,
                'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': (torch.cuda.get_rng_state()
                             if device == 'cuda' else None),
            }, train_ckpt)
            print(f"    checkpoint -> {train_ckpt}", flush=True)

    print(f"\n=== Final Evaluation ===", flush=True)
    t_eval0 = time.time()
    ppl_1k = compute_perplexity(model, test_short[:1000], max_len, 32, device)
    ppl = compute_perplexity(model, test_short[:5000], max_len, 32, device)
    print(f"  In-length per-token PPL (<= {max_len}): {ppl:.2f} on 5k test "
          f"sentences", flush=True)
    print(f"  (legacy 1k eval: {ppl_1k:.2f})", flush=True)
    print(f"  OPERA v8.0 same protocol: so3+left 42.58 | free+left 41.53 | "
          f"so3+attend 43.78", flush=True)

    print(f"\n=== Length Extrapolation (train <= {max_len}) ===", flush=True)
    t_ex0 = time.time()
    extrap = extrapolation_eval(model, test_long, max_len, eval_max_len,
                                16, device)
    t_ex = time.time() - t_ex0
    for bucket, (bppl, n) in extrap.items():
        if bppl is None:
            print(f"  len {bucket}: insufficient data (n={n})", flush=True)
        else:
            print(f"  len {bucket}: PPL {bppl:.2f}  (n={n}, "
                  f"{bppl/ppl:.2f}x in-length)", flush=True)
    print(f"  (extrapolation eval wall-clock: {t_ex:.1f}s -- P2 timing "
          f"half; compare same-GPU vs S1's)", flush=True)

    ckpt_path = os.path.join(out_dir,
                             f'transformer_v2_{tag.replace("+", "_")}.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}", flush=True)

    results = {
        'model': 'transformer_baseline_v2', 'config': tag, 'pe': pe_mode,
        'seed': seed, 'data': data_mode,
        'docs_limit': (docs_limit if data_mode == 'docs-en' else None),
        'lr': max_lr, 'warmup': warmup_steps,
        'params': npar, 'd': d,
        'nheads': nheads, 'num_layers': num_layers, 'ffn_mult': ffn_mult,
        'vocab_size': V, 'max_len': max_len, 'eval_max_len': eval_max_len,
        'steps': steps, 'final_loss': loss.item(),
        'test_perplexity_in_length': ppl, 'test_perplexity_1k_legacy': ppl_1k,
        'extrapolation': {k: v[0] for k, v in extrap.items()},
        'extrap_eval_seconds': t_ex,
        'init_perplexity': init_ppl,
    }
    results_path = os.path.join(out_dir, 'transformer_v2_results.jsonl')
    with open(results_path, 'a') as f:
        f.write(json.dumps(results) + '\n')
    print(f"Saved to {results_path}", flush=True)
    return results


# ============================================================================
# PROBE MODE (transformer anchor for the v8.0 probe measurements)
# ============================================================================

def run_probe(ckpt_path, cfg, device):
    from opera_v8_probe import interrogation_probe, position_curve
    train_data, test_short, test_long, vocab, w2i, i2w = load_data(
        cfg['vocab'], cfg['max_len'], cfg['eval_max_len'],
        data_mode=cfg['data'], docs_limit=cfg.get('docs_limit', 100000))
    V = len(vocab)
    model = TransformerBaseline(
        V, d=cfg['d'], nheads=cfg['nheads'], num_layers=cfg['layers'],
        pe_mode=cfg['pe'], max_pe_len=cfg['eval_max_len'],
        ffn_mult=float(cfg['ffn_mult']))
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print(f"Loaded transformer ({count_params(model):,} params, vocab {V})",
          flush=True)

    tag = cfg['tag'] or ('tf_' + os.path.basename(ckpt_path).replace('.pt', ''))
    results = {'probe_version': 'transformer-anchor', 'tag': tag,
               'ckpt': ckpt_path, 'arch': 'transformer', 'pe': cfg['pe'],
               'curves': {}, 'ratios': {}, 'position_bands': None}

    prefixes = sorted({int(x) for x in str(cfg['prefixes']).split(',')})
    pool = test_short + test_long
    print("\n=== Interrogation probe (transformer anchor) ===", flush=True)
    for Lp in prefixes:
        out = interrogation_probe(model, pool, Lp, V, device,
                                  n_seq=cfg['nseq'],
                                  resamples=cfg['resamples'])
        if out is None:
            print(f"  prefix {Lp}: insufficient sequences", flush=True)
            continue
        curve, B = out
        last_p = max(curve)
        ratio = curve[1] / curve[last_p]
        results['curves'][str(Lp)] = curve
        results['ratios'][str(Lp)] = ratio
        print(f"  prefix {Lp} (n={B}):", flush=True)
        for p in sorted(curve):
            bar = '#' * max(1, int(60 * curve[p] / max(curve.values())))
            print(f"    p={p:4d}  JSD {curve[p]:.5f}  {bar}", flush=True)
        print(f"    >>> pp1/ppLast (p=1 vs p={last_p}): {ratio:.3f}",
              flush=True)

    print("\n=== Position curve (transformer anchor) ===", flush=True)
    max_pos = min(cfg['eval_max_len'], 2 * cfg['max_len'])
    pc = position_curve(model, test_long, max_pos, device)
    if pc is not None:
        bands, n = pc
        results['position_bands'] = bands
        print(f"  n={n} sequences:", flush=True)
        for k, v in bands.items():
            marker = ' <- beyond training length' \
                if int(k.split('-')[0]) >= cfg['max_len'] else ''
            print(f"    pos {k:>9}: CE {v:.3f}  (PPL {math.exp(v):7.2f})"
                  f"{marker}", flush=True)

    out_path = os.path.join(cfg['out'], 'opera_v8_probe_results.jsonl')
    os.makedirs(cfg['out'], exist_ok=True)
    with open(out_path, 'a') as f:
        f.write(json.dumps(results) + '\n')
    print(f"\nSaved to {out_path}", flush=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--selftest' in argv:
        selftest()
        sys.exit(0)

    VAL = {'--pe': ('rope', 'nope', 'sin', 'learned'),
           '--data': ('sentences', 'docs', 'docs-en'),
           '--docs-limit': None,
           '--lr': None, '--warmup': None,
           '--seed': None, '--dropout': None, '--ffn-mult': None,
           '--out': None, '--save-every': None,
           '--probe': None, '--prefixes': None, '--nseq': None,
           '--resamples': None, '--tag': None}
    BARE = {'--resume', '--amp', '--selftest'}

    pe_mode = 'rope'
    data_mode = 'docs'
    docs_limit = 100000
    max_lr = 1e-3
    warmup_steps = 500
    seed = 42
    dropout = 0.0
    ffn_mult = 4.0
    out_dir = '.'
    save_every = 0
    resume = False
    use_amp = False
    probe_ckpt = None
    prefixes = '31,63,127,255'
    nseq = 64
    resamples = 4
    ptag = ''
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--'):
            if a in BARE:
                if a == '--resume': resume = True
                if a == '--amp': use_amp = True
                i += 1
            elif a in VAL:
                if i + 1 >= len(argv):
                    sys.exit(f"ERROR: {a} requires a value")
                v = argv[i + 1]
                allowed = VAL[a]
                if allowed is not None and v not in allowed:
                    sys.exit(f"ERROR: {a} must be one of {allowed}, got {v}")
                if a == '--pe': pe_mode = v
                if a == '--data': data_mode = v
                if a == '--docs-limit': docs_limit = int(v)
                if a == '--lr': max_lr = float(v)
                if a == '--warmup': warmup_steps = int(v)
                if a == '--seed': seed = int(v)
                if a == '--dropout': dropout = float(v)
                if a == '--ffn-mult': ffn_mult = float(v)
                if a == '--out': out_dir = v
                if a == '--save-every': save_every = int(v)
                if a == '--probe': probe_ckpt = v
                if a == '--prefixes': prefixes = v
                if a == '--nseq': nseq = int(v)
                if a == '--resamples': resamples = int(v)
                if a == '--tag': ptag = v
                i += 2
            else:
                sys.exit(f"ERROR: unknown flag {a}. Known: "
                         f"{sorted(BARE | set(VAL))}")
        else:
            args.append(a)
            i += 1

    steps = int(args[0]) if len(args) > 0 else 20000
    batch = int(args[1]) if len(args) > 1 else 16
    max_len = int(args[2]) if len(args) > 2 else 256
    vocab_size = int(args[3]) if len(args) > 3 else 10000
    d = int(args[4]) if len(args) > 4 else 512
    nheads = int(args[5]) if len(args) > 5 else 8
    num_layers = int(args[6]) if len(args) > 6 else 4
    eval_max_len = int(args[7]) if len(args) > 7 else 2048

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device}", flush=True)

    if probe_ckpt is not None:
        cfg = {'pe': pe_mode, 'data': data_mode, 'docs_limit': docs_limit,
               'd': d, 'nheads': nheads,
               'layers': num_layers, 'vocab': vocab_size, 'max_len': max_len,
               'eval_max_len': eval_max_len, 'ffn_mult': ffn_mult,
               'prefixes': prefixes, 'nseq': nseq, 'resamples': resamples,
               'tag': ptag, 'out': out_dir}
        run_probe(probe_ckpt, cfg, device)
        sys.exit(0)

    train(steps, batch, max_len, vocab_size, d, nheads, num_layers,
          eval_max_len, device, pe_mode=pe_mode, ffn_mult=ffn_mult,
          dropout=dropout, seed=seed, data_mode=data_mode,
          docs_limit=docs_limit, out_dir=out_dir,
          save_every=save_every, resume=resume, use_amp=use_amp,
          max_lr=max_lr, warmup_steps=warmup_steps)