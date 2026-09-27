# OPERA speed diagnosis + attention-free design directions (2026-09-24)

Companions: `OPERA_Redundancy_Research_2026-09-24.md` (L²M, Phase-1
instruments), `OPERA_Literature_Scan_2026-09-24.md`. Standing constraint:
no attention in any form.

---

## 1. Why OPERA was slower than a transformer

Measured on the byte model (d512 / nb128 / L2, T=1024):

| cause | measurement |
|---|---|
| The left fold re-folds every prefix's ~log T blocks from scratch | 4,097 composes/layer vs 1,023 for the tree — **80% of compose work, 75% of layer time** (CPU: fold 83 ms, tree 19 ms, cross_mlp 9 ms) |
| Rotations run as dense GEMMs on the block-diagonal matrix (`_dense_rot`; chosen because MPS handles one GEMM better than many small bmm's) | 0.88 MFLOP per compose vs 0.007 MFLOP actually needed — more than the fusion gate (0.79 MFLOP) |
| Net | ~20.9 MFLOP/token forward vs ~10.5 for a parameter-matched transformer (d256/L4) at T=1024 — **2.0× more arithmetic**, in ~500 small kernel launches per layer, eager on MPS |

The asymptotic argument (O(T log T) vs O(T²)) was correct but did not
apply at this size: per token, a layer cost ~5 composes of a 1.5·d²
gate, vs attention's ~4·T·d. Attention only becomes the larger term
beyond T ≈ d·log T ≈ 5k tokens.

## 2. Fix 1 — the downsweep fold (`fold_impl='downsweep'`, implemented)

The fold for every prefix is computed top-down over the tree, sharing
intermediate results between prefixes (the static-scan idea of
Prefix-Scannable Models, arXiv 2506.10918):

    E_k[2j]   = E_{k+1}[j]
    E_k[2j+1] = compose(E_{k+1}[j], node_k[2j])     (j ≥ 1;  E_k[1] = node_k[0])
    prefix state of position t = E_0[t+1]

Each prefix gets exactly the same chain of composes (same operands,
same order, same fold gate / bias / innovation) as the compacted fold.

- **Equivalence:** logits within ≤2e-6 and gradients within ≤5e-7 of
  `compact` across T ∈ {1, …, 100}, with fold gate bias, separate fold
  gate, innovation fold, and stream head + state_tie
  (`test_downsweep_fold`).
- **Work:** 1,013 fold composes/layer at T=1024 vs 4,097. Training cost
  is now **O(T)** per layer, not O(T log T) — per-token cost no longer
  grows with context.
- **Measured speed:** fold 5× faster on CPU (19 vs 97 ms). A full
  training step on MPS (fwd + bwd + optimizer, B=8, T=1024) is **2.6–2.7×
  faster**: 0.811 → 0.297 s (incumbent), 0.841 → 0.325 s (stream head).
- **Scope for now:** the plain left fold plus fold_gate_bias, fold_gate,
  fold_innov_rank, stream head and state_mult/state_tie. Grading, h0 and
  the falsified accumulator arms still use `compact`.

**Remaining speed items:**
1. True block-diagonal rotations on CUDA (grouped/batched 3×3), keeping
   the dense form only where it measurably wins (MPS).
2. Fused compose kernel (the Triton "KERNEL V2" exists) + CUDA graphs,
   against launch overhead.
3. Cheaper gates (low-rank or grouped fusion gate), now the dominant
   FLOP.

Analytic estimate after item 1: ~0.7× a transformer's FLOPs/token at
T=1024, ~0.2× at T=8192. To be verified with CUDA timings.

## 3. Quality without attention — attention's three jobs, three borrowed tools

**A. Memory that grows with context → Fast Multipole Method
(computational physics).**
- L²M: a model's history state must grow like T^β.
- FMM computes all-pairs interactions on a tree and gives *far* cells
  more expansion terms, because they summarize more. OPERA gives a node
  covering 512 tokens the same width as a single leaf.
- **Multipole OPERA:** node width grows with level, w_ℓ ∝ 2^{βℓ}.
  - per-prefix history state (the Fenwick roots) ∝ T^β — satisfies L²M;
  - total memory stays O(T); compute stays near-linear for β ≤ ½;
  - the readout gives each level a width-matched slot group — the
    principled completion of the scale-graded readout.
- **Sequenced after `b2_tie2`:** if doubling width *uniformly* does not
  lift the ~100-byte MI plateau, level-growing width is unlikely to;
  if it does, Multipole is the efficient way to buy it.

**B. Exact recall of single-copy content → LZ77 / PPM pointers (data
compression). DEFERRED (Merna, 2026-09-24).**
- Exact-match pointer memory: the previous occurrence of the current
  n-gram, read what followed it.
- Relatives: pointer-sentinel / copy mechanisms, kNN-LM, ∞-gram, Engram.
  Transformer induction heads learn nearly this algorithm.
- Why deferred:
  - exact match is brittle (no paraphrase / case / spelling
    generalization);
  - gains concentrate on repetitive text;
  - it sidesteps rather than answers whether the geometric tree itself
    can carry long-range information.
- If revisited: a separate, clearly ablated module, results split into
  repeated vs novel tokens.

**C. Weighted fusion of evidence → the redundancy matrix (structural
mechanics; Forster & von Scheven, arXiv 2405.06294).**
- Definitions: R = I − A K⁻¹ Aᵀ C, K = AᵀCA. R is an idempotent
  (oblique) projection; tr R = the degree of static indeterminacy;
  R_kk ∈ [0,1] is member k's share of it (0 = single point of failure,
  1 = fully corroborated). Same object as geodesy's redundancy numbers
  (Baarda's reliability theory: r_i = 1 − leverage).
- Mapping onto OPERA:
  - members = the prefix's Fenwick blocks, each "measuring" the latent
    prefix state;
  - readout = the weighted least-squares / equilibrium solve
    d̂ = K⁻¹ Aᵀ C s;
  - per-block precision C comes from each block's own state (no query).
  - In information form this is a sum → associative → tree-exact and
    downsweep-compatible.
- Readings:
  - the PID finding (the oldest block's unique content is erased by the
    fold) is a zero-redundancy member failing;
  - the paper's Woodbury update for removing a member is the maths of
    fragment dropout;
  - R's diagonal is a ready-made diagnostic of which blocks the readout
    depends on alone.
- Expected gain: modest (a principled readout, not new capacity).

**Order:** measure `b2_tie2` → A (Multipole) → C (equilibrium readout);
B only on an explicit decision.

## 4. Queue 3 (running) — base 2 = stream head + downsweep

`b2_base`, `b2_tie2`, `b2_innov`, `b2_disent`, `b2_nmix` (all
single-variable on base 2), then `st_wide` (compact fold, fold head,
d640: the live-parameter control attributing `st_head`'s −1.79% BPB),
then the MI curve on every arm and the block-PID probe on the b2 arms.
Log: `runs_reprs/queue3_2026-09-24.log`.

### 4.1 First result and a noise-floor correction

`b2_base` (stream head + downsweep, seed 42): **2.0198 BPB**, extrapolation
3.992 — vs `st_head` (same model, compact fold) 2.0058 / 3.964: +0.70%.
Checked directly: with `st_head`'s trained weights on MPS under bf16
autocast, both fold implementations give the same loss to 6 decimals
(1.343216), logits within 0.008, gradients within 0.5–0.7% relative
(cosine ≥ 0.99997). The gap is therefore training-trajectory variance
(bf16 rounding compounding over 3000 steps — effectively a different
seed), not a systematic difference.

Consequence: single-run noise for this setup is larger than the 0.25%
spread seen across incumbent runs — two samples of the same config
differ by 0.7%. Queue 4 (`b2_base_s43`, `b2_base_s44`) measures the
base-2 noise band; arm effects are judged against it, and anything under
~2× that band needs a second seed.

### 4.2 Architecture arms on base 2 — all three lost, and why (a Muon artifact)

| arm | BPB | vs b2_base 2.0198 |
|---|---|---|
| b2_tie2 (2× tied state) | 2.0026 | −0.85% (inside the 0.7% base spread) |
| b2_innov (innovation fold, r64) | 2.0453 | +1.3% |
| b2_nmix (cross-slot mixing, r64) | 2.0764 | +2.8% |
| b2_disent (causal disentangler, r64) | 2.1758 | +7.7% |

MI curves: `b2_tie2` lifts short/mid-range information (~+1 bit at
4–48 bytes of context) and the plateau by ~0.4–0.5 bits, but the plateau
onset does not move (both flat from ~100–250 bytes to 2048). Doubling the
state does not extend context use at this rung → Multipole OPERA (A) is
not supported yet.

**The three low-rank arms did not test their mechanisms.** Their U factors
were zero-initialized and routed to Muon, whose steps have a fixed size
regardless of gradient magnitude (×√(512/64) ≈ 2.8 for these tall
matrices). The learned corrections grew far past "small refinement":
median |correction| / |input| = 1.3× (innov), 6.6× (disent), 30×
(node_mix); top singular values of U·Vᵀ 21 / 69 / 71.

**General lesson:** "interior init" (a zero-init factor giving a bitwise
incumbent) is only gentle under optimizers whose step scales with the
gradient. Under Muon a parameter that should stay small still gets a
constant-size push every step. Earlier Muon-era falsifications of
zero-init arms carry this caveat: `relax_g` (over-relaxation; positive
under AdamW, `relaxM_over` +2.5% under Muon) and `lc_U`/`lc_V`
(level-conditioned weights) were all routed to Muon.

**Fix — `lowrank_gain=True`:** correction = gain · (x·V)·Uᵀ · √d / ‖V·Uᵀ‖_F.
U and V are random and live in Muon (only their direction matters, their
norms cancel); a per-layer scalar gain starts at 0 and lives in AdamW —
bitwise incumbent at init, gradient reaches the gain at step 0, and the
gain equals the typical relative size of the correction (selftest: rms
ratio 0.499 at gain 0.5, independent of ‖U‖). Reruns `b2g_innov`,
`b2g_disent`, `b2g_nmix` are queued (queue 6).

### 4.3 Counterclockwise fold (`fold_dir='both'`) and its precedents

**Mechanism.** The left fold ((B1∘B2)∘B3)∘B4 composes the oldest block
first and squashes it through every later compose — the block probe
measured its single-copy content going from 4.6 bits in-block to 0 after
the fold. The right-nested ("counterclockwise") fold B1∘(B2∘(B3∘B4))
composes the newest blocks first and joins the oldest last. The readout is
P = P_left + gain ⊙ P_right (per-slot gain, zero-init, AdamW): bitwise
incumbent at init, and every block has a short path to the readout from
one side. Causal (every block lies in the prefix), attention-free.
Selftested: matches a naive per-position right fold to 9e-7, causal,
decoder-exact under both fold implementations. Cost: Σ(popcount−1)
composes (no downsweep sharing for right-nested folds). Arm `b2_ccw`
(queue 7).

**Inspiration.** Path splitting in color-coding subgraph counting
(Chakaravarthy et al., arXiv 1602.04478): a cycle between two boundary
nodes is split into its clockwise and counterclockwise paths, both built
edge by edge, then joined.

**Precedents (the principle, not this exact mechanism):**
- Shi et al., *On Tree-Based Neural Sentence Modeling* (EMNLP 2018):
  trivial trees (balanced, left-branching, right-branching) match or beat
  syntactic trees on ten tasks; tree models do better "when crucial words
  are closer to the final representation". This is the same lever: the
  nesting decides each block's path length to the readout.
- Sutskever et al., seq2seq (NeurIPS 2014): reversing the source order
  created short-term dependencies and improved translation markedly —
  path length matters.
- Teng & Zhang, *Head-Lexicalized Bidirectional Tree LSTMs* (TACL 2017):
  a bottom-up and a top-down pass over the same tree beat either alone.
- Bidirectional RNNs combine two traversal directions, but over the whole
  sequence (non-causal). The OPERA version is causal because both
  traversals stay inside the prefix.
- Prefix-Scannable Models (arXiv 2506.10918): with non-associative
  operators the prefix output depends on the parenthesization; `b2_ccw`
  uses two parenthesizations deliberately.
- Reverse language models (LEDOM, arXiv 2507.01335; *Reverse Modeling in
  LLMs*, NAACL 2025) show forward/backward complementarity, but reverse
  the whole sequence and are used for reranking, not as a causal readout.

A handful of searches found no prior use of combining left- and
right-nested folds of prefix blocks in a causal language model (not an
exhaustive search).

### 4.4 Measured noise band (base 2 = stream head + downsweep)

Four runs of the base-2 model: st_head 2.0058 (seed 42, compact fold),
b2_base 2.0198 (seed 42), b2_base_s43 2.0366, b2_base_s44 2.0284 →
**mean 2.0227, sd 0.0131 (0.65% per run)**. The four earlier incumbent
runs (2.0371 / 2.0406 / 2.0411 / 2.0423, sd 0.11%) all used seed 42 and
therefore measured only float-level trajectory noise, not seed variance —
the "0.25% noise floor" used earlier today understated single-run noise
~3×. Many past single-seed ~1% gate decisions in the project fall inside
this band.

Reading rules from here: a single run must differ from the base mean by
> ~1.3% (2 sd) to count; smaller effects need ≥2 seeds. Stream head vs
incumbent: −0.86% (t ≈ −2.6 against the seed-42-only incumbent spread;
~−1.9 if the incumbent has the same seed variance) — probably real,
modest. b2_tie2 2.0026 = −1.5 sd from the base mean — suggestive; second
seed `b2_tie2_s43` queued (queue 8).

### 4.5 The ~100-byte plateau — diagnosis in progress

Every arm so far (incumbent, stream head, 2× state, the runaway low-rank
arms) extracts no further information about the next 31 bytes from
context beyond ~100–130 bytes. Four candidate causes, each with a test:

1. **The data has nothing beyond 100 bytes — RULED OUT.** On the 252
   held-out long documents, the next 8 bytes already appeared in the
   prefix at 15.0% of positions: within the last 100 bytes at 4.1%, and
   **only further back at 11.0%**, spread evenly from 128 to 4096 bytes
   back (0.9 / 2.7 / 2.8 / 2.5 / 1.7 / 0.4% per octave). Recovering even
   part of that is worth ~0.1 bits/byte (~5% BPB).
2. **Undertraining** (~25M training bytes per run): `b2_long`, the base
   trained 4× longer (12k steps), MI curve compared.
3. **The fold erases old content** (block probe: 4.6 → 0 bits):
   `b2_ccw` (counterclockwise fold) and `b2g_innov` (innovation fold,
   gain-parameterized).
4. **The objective barely rewards far context:** `b2_fut`, a
   training-only head predicting the hashed byte-trigrams (K=1024) of the
   next 128 bytes from the final head input (weight 0.1). The past–future
   MI grows with the future window (L²M), so a window-level target pays
   the state to keep content that will recur. Forward and inference
   unchanged (selftested bitwise); targets verified against brute-force
   counts.

### 4.6 Gain-parameterized reruns (base mean 2.0227 ± 0.0131)

| arm | BPB | vs base mean | MI plateau (bits) | oldest-block last byte: in-block → fold |
|---|---|---|---|---|
| b2g_nmix (cross-slot mixing) | **1.9967** | **−1.29% (≈ −2 sd)** | 11.4–11.5 | — |
| b2g_innov (innovation fold) | 2.0147 | −0.4% (noise) | 11.2 | 4.24 → 0.00 |
| b2g_disent (disentangler) | 2.0691 | +2.3% (harmful) | 9.5 | — |
| (b2_tie2, for reference) | 2.0026 | −1.0% | 12.4 | 4.46 → 0.00 |

- The gain fix confirmed the Muon diagnosis: node mixing went from +2.8%
  (runaway) to −1.3% (bounded). Second seed `b2g_nmix_s43` queued (queue 10).
- Its gain is local: MI plateau height within seed noise (base seeds
  11.9 / 10.8 / 11.1 bits), onset unchanged.
- No arm so far moves the plateau onset (~128 bytes) or lets the oldest
  block's single-copy content survive the fold (0.00 bits in every arm).
- The disentangler is harmful even when bounded — dropped.

### 4.7 Plateau tests: objective and training length

| arm | BPB | vs base mean | extrap. 1025–2048 | MI plateau (bits) | onset |
|---|---|---|---|---|---|
| b2_fut (future-content objective) | 2.0005 | −1.1% (1.7 sd) | 3.962 | 12.0 | ~128 |
| **b2_long (4× training, 12k steps)** | **1.8179** | **−10.1%** | **3.467 (−13%)** | 12.9–13.0 | ~128–256 |
| b2_ccw (counterclockwise fold) | 1.9961 | −1.3% (≈ −2 sd) | 3.937 | 11.7–11.9 | ~128 |

- **OPERA is heavily undertrained at 3000 steps**: 4× the training is
  worth −10% BPB, far more than any architecture arm (best ≈ −1.3%).
- **The ~128-byte onset survives everything tested**: data (11% far-only
  repeats exist), state size (tie2), a short path for old blocks (ccw,
  learned join gains stay small: mean |g| ≈ 0.04), a window objective
  (fut), 4× training (long). Old single-copy content reaches the fold
  readout at 0.00 bits in every arm — "availability is not use",
  measured directly.
- Working hypothesis: **gradient starvation** — positions where only far
  context helps are rare (4.7% of targets complete an 8-gram seen only
  >100 bytes back) and their gradient is swamped by the local majority.

### 4.8 Gradient-side arms (queue 11)

- `b2_far` — far-repeat loss weighting (`far_repeat_mask`, n=8, D=100;
  weight 1+6 on marked targets, renormalized to mean 1: the marked 4.7%
  carry ~26% of the final-layer loss). Lineage: Rho-1 selective language
  modeling, focal loss.
- `b2_bdrop` — byte dropout 15% on inputs only (mask = the unused EOS id,
  learnable). Lineage: word dropout against posterior collapse (Bowman et
  al. 2016).
- `b2_sam` — SAMuon-lite γ=3.54: amplifies the spectral "bulk" (weak
  directions) relative to the head — the optimizer-level counter to
  starvation.
Judged primarily by the MI-curve onset and by fold retention of old
single-copy content (block probe on b2_far / b2_bdrop), then BPB.
`b2_tie2_s43` dropped (state size does not move the onset);
`b2g_nmix_s43` runs after.

### 2.1 Fix 2 — the fused Metal compose kernel (`use_metal=True`)

Interleaved MPS benchmark of the compose node (fwd + bwd, N = 8,192 rows,
d512/nb128): dense-GEMM rotations (current) 54.1 ms; per-block einsum
209.0 ms; elementwise 3×3 123.1 ms; **fused Metal kernel 17.9 ms (3.0×)**.
The dense GEMM was the right choice among unfused options (MPS punishes
many small kernels); the fused kernel sidesteps the question.

- Equivalence (selftest `test_metal_equivalence`, fp32): loss identical,
  logits ≤ 2e-6, gradients ≤ 3e-5 relative, for stream head + downsweep,
  bounded node mixing, and fold gate bias. At d512: logits ≤ 1e-5,
  gradients ≤ 1.6e-4 relative.
- Full training step (B=8, T=1024, stream + downsweep, measured under
  contention): eager 0.614 s → metal 0.295 s (**2.1×**). Combined with the
  downsweep: ~5× faster than the original 0.81 s/step.
- Adoption: validation run `b2m_base` (queue 12) must land inside the
  base-2 band before metal becomes the default for new arms.
- Note: the README's "requires rot_mode='so3'" is outdated — the fused
  backward recomputes rather than inverting R_O and supports `free`.

### 4.9 Gradient-side arms, Metal validation, and a regression check

| arm | BPB | verdict |
|---|---|---|
| b2_sam (SAMuon-lite γ=3.54) | 2.1439 (+6.0%) | hurts — its training loss diverges from b2_base exactly as γ ramps (same init/data: step 0 identical); bulk steps too large at batch 8 |
| b2_far (far-repeat weighting ×7) | 2.1465 (+6.1%) | hurts — oldest-block content still 0.00 bits after the fold |
| b2_bdrop (15% byte dropout) | 2.1457 (+6.1%) | hurts — train/test mismatch; MI plateau lower (~10 bits) |
| b2g_nmix_s43 | 2.0485 | two-seed mean 2.0226 ≈ base mean → the −1.3% was noise |
| st_wide (fold head, d640, width control) | 2.0069 | ≈ st_head 2.0058 → the stream-head gain is largely reclaimed capacity |
| **b2m_base (Metal kernel)** | **2.0354** | inside the base band → **Metal adopted as default** |

Regression check (the three +6% arms looked suspiciously alike): current
code re-evaluates every checkpoint with a constant offset (no eval bug),
and a rerun of the b2_base config with current code tracks the original
log (step 600: 2.157 vs 2.158). The results are real; the similarity is
coincidence.

**Plateau status.** No intervention moved the ~128-byte onset: data
(far-only repeats exist), state size, a short path for old blocks, node
capacity, a window objective, 4× training, loss reweighting, input
dropout, optimizer. Old single-copy content reaches the fold readout at
0.00 bits in every arm, while the tree blocks hold 4+ bits of it.
Reading: the fold's content-blind compression discards single-copy far
content at this scale; retrieving a specific old byte needs
content-addressed access (attention — excluded — or the deferred
exact-match lookup). Open decision (Merna): (1) accept the horizon and
scale training (4× = −10% BPB); (2) revisit the exact-match lookup as a
separate, ablated module; (3) try depth (L=8) first.

### 4.10 Counterclockwise fold, second seed

b2m_ccw_s43 (Metal, seed 43): 2.0419. Two-seed mean 2.0190 vs the
five-run base mean 2.0252 ± 0.0127 → −0.3%, t ≈ −0.6: **noise**. With
seed noise measured, no architecture arm tested today beats the base
(node mixing and the counterclockwise fold both regressed to the mean on
their second seed). The day's real effects: stream head (−0.9% vs the old
incumbent, largely reclaimed capacity per st_wide), 4× training (−10%),
and ~5× faster training steps (downsweep + Metal kernel).

### 4.11 Depth (L=8): first attempt failed on initialization, not on depth

| arm | params | BPB | vs L=2 Metal base 2.0354 |
|---|---|---|---|
| d8_add (8 × d512, additive, default init) | 11.9M | 3.0266 | +49% (loss spike 2.99 → 4.42 at step 400) |
| d8_blend (8 × d512, blend, default init) | 12.4M | 2.8142 | +38% |
| d8_add_d256 (8 × d256, param-matched) | 3.05M | 2.2017 | +8% |

MI curves collapsed accordingly (plateau 5–9 bits); onset still ~128.

**Diagnosis.** Downsweep vs compact gradients disagreed by up to 33% at
L=8 (~1e-7 at L=2). In float64 the forward difference grows smoothly
4e-16 → 9e-12 over 8 layers — the algorithms are identical, and the
**stack is expansive at init**: perturbations grow ~5× per layer.
Per-layer perturbation growth (float64, L=8, d512): blend/default
4.84×; add/default 1.38×; blend + 1/√(2L) output scaling 3.01×;
**add + 1/√(2L) 1.05×** (GPT-2 residual scaling); add + zero-init 1.00×.
The blend residual is expansive at depth regardless of init (cf. G1).

**Fix:** `resid_init_scale='auto'` scales each cross_mlp output
projection by 1/√(2L) and zeroes its bias (in place, no RNG — off is
bitwise). Retry: `d8s_add` (8 × d512) and `d8s_add_d256` (8 × d256,
param-matched), queue 14.
