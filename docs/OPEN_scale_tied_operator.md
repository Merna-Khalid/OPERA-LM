# OPEN PROBLEM — optimizing a weight set tied across SCALE

**Status:** open. Not a registered study; a problem statement plus the
evidence gathered so far and the two experiments that would attack it.
**Opened:** 2026-09-09, from the level-gradient work of 2026-09-08.

---

## 1. The problem

Every modern optimizer framework — Muon, Scion, Gluon, modular norms —
is built on a **layer-wise** decomposition. Each layer has its own
weight matrix, its own smoothness constant, its own norm choice, and
**one gradient contribution per forward pass**. The entire prescription
is "match the norm and step size to each layer's geometry."

OPERA's compose operator is not a layer. It is **one weight set applied
at ⌈log₂ T⌉ tree levels**:

    dL/dW = Σ_ℓ (contribution from level ℓ)

and those applications differ from one another in two ways at once:

- **Count** — level 0 has T nodes, level ℓ has T/2^ℓ. Geometrically
  decreasing.
- **Input distribution** — level 0 consumes token embeddings
  (mean ‖h‖ 10.68); levels ≥ 1 consume composed states (‖h‖ 6.93),
  measured 2026-09-07 in `docs/trajectory_phase2_nulls.md` lineage.

So "what step size is correct for W?" is not well posed under layer-wise
theory, because W is not a layer — it is an operator whose gradient is a
sum over structurally *different* applications.

## 2. What is measured (all on byte d512/nb128/L2, `bist_off`)

Gradient share reaching the shared compose weights, by tree level
(T=256, layer 0, measured by hooking each level's output):

| level | span | share of dL/dW | per-node |
|---|---|---|---|
| 0 | 1 | 34.1% | 0.000021 |
| 1 | 2 | 25.3% | 0.000031 |
| 2 | 4 | 14.8% | 0.000036 |
| 3 | 8 | 9.6% | 0.000047 |
| 4 | 16 | 6.3% | 0.000061 |
| 5 | 32 | 4.4% | 0.000086 |
| 6 | 64 | 3.1% | 0.000121 |
| 7 | 128 | 2.3% | 0.000180 |
| 8 | 256 | **0.0%** | **0** |

- **15:1 imbalance** across trained levels. Node counts halve (2× decay)
  but per-node gradient *grows* 8.6× with depth, so the net decay is
  ~1.5× per level, not 2×. The optimizer is already partly
  compensating.
- **The root receives exactly zero gradient.** The only prefix whose
  Fenwick decomposition reads it is L = T, whose prediction target is
  the token *after* the sequence, which the loss masks. `msup_loss`
  cannot fix this either: it breaks at `span >= T`, and its target for
  the root lies outside the sequence by definition.

## 3. What has already been ruled out

**Level-balanced gradient — FALSIFIED 2026-09-08**
(`docs/OPERA_LevelGrad_prereg.md`). Rescaling each level's contribution
to dL/dW by β^ℓ (normalised to mean 1, so it rebalances rather than
rescales):

| β | BPB vs incumbent | extrapolation vs incumbent |
|---|---|---|
| 1.25 | +0.74% | +0.72% |
| 1.5 | +1.50% | +1.71% |
| 2.0 | +3.02% | +4.37% |

Monotone in the **wrong** direction across four points, no capacity
confound (identical parameter count). The likely reason is in §2: the
per-node gradient already grows 8.6× with depth, so the residual 15:1 is
plausibly the *correct* allocation — level 0 performs 256× more distinct
compositions and should receive more gradient. Forcing equality is
over-correction.

**Methodological note worth keeping:** *measurable* + *unique to the
architecture* ≠ *harmful*. Three separate claims; the first two were
established and the third was assumed.

## 4. Adjacent prior art, and why it does not cover this

- **Weight tying across DEPTH** — Universal Transformer, ALBERT,
  Mixture-of-Recursions (arXiv:2507.10524). The tied layer is applied N
  times to a *same-length* sequence: equal node counts per application,
  no imbalance can arise.
- **Weight tying across TIME** — RNNs. Same weights applied T times,
  gradient summed over steps; exhaustively studied (BPTT, exploding/
  vanishing, clipping). But RNN applications are *uniform*: same count
  per step, roughly stationary input distribution.
- **Gradient rescaling for imbalance** — multi-task GradNorm; cell-level
  rescaling in NAS/lookup networks. General technique, not indexed by
  scale in a weight-tied hierarchy.
- **LMO/norm frameworks** — Scion (arXiv, per-layer norms: spectral for
  weight matrices, ℓ1→∞ for embeddings), Gluon (arXiv:2505.13416,
  layer-wise smoothness model with convergence guarantees).

None of these has a story for an operator whose applications differ in
**both count and input distribution**. That combination is what is
unusual, and we are not aware of another architecture that has it.

Caveat, stated plainly: this is a **framing**, not a theorem. RNNs are
close enough that the novelty claim needs care.

## 5. Two experiments that would attack it

They pull in opposite directions, which is what makes the pair
informative.

### 5a. Untie the compose weights across levels
Give shallow / mid / deep levels their own compose parameters (or fully
untie). Directly tests whether scale-tying costs anything.

- **Cost:** parameters grow ~log T for the compose node. `fusion_gate`
  is 393K/layer at d=512; fully untied over 10 levels is 3.9M/layer.
  Grouping into 2–3 bands is the practical version.
- **Breaks length extrapolation.** At eval 8192 you would need weights
  for levels 10–13 that were never trained — the level-exposure problem
  of 2026-09-07 made worse. That is a real tradeoff, not a flaw in the
  experiment; measure both in-length and extrapolation.
- **Confound:** untying adds parameters. Needs a capacity-matched
  control (e.g. a wider tied model at equal params).

### 5b. Level-conditioned weights (extrapolation-safe)
    W_ℓ = W + U · diag(f(ℓ)) · V
with `f` **smooth in ℓ** (the `level_sin_enc` convention already used by
`fold_scale` and the attend fold, and already declared
extrapolation-safe). Each level gets its own effective operator while
the weights stay *defined* at unseen depths.

- Preserves I1: level is scale, not position.
- Low-rank, so the parameter cost is `r(2d + 3nb)` rather than a full
  copy per level.
- **Do NOT use a per-level embedding or one-hot** — rows for levels
  ≥ 10 would arrive at eval never having been trained. Smoothness in ℓ
  is the whole point.
- Zero-init `U` (or `V`) so β=0 is bitwise the incumbent and gradients
  still flow — the LoRA convention, per the failures catalogued in
  `docs/OPERA_Bistable_prereg.md` §4.1.

Not aware of prior art for 5b.

## 6. Gates to pre-register before running either

The level-balance study's gate structure applies: the deficiency is at
deep levels, which dominate longer sequences, so the **primary gate is
length extrapolation, not in-length BPB**, and a small BPB regression
with extrapolation improving should be declared a success *in advance*.

Also required, given §3: an explicit usage/effect check that the
intervention took effect, so a null is distinguishable from a
no-op — the distinction that made the bistable outcomes interpretable.

## 7. Why this is parked rather than running

As of 2026-09-09 the measured levers are elsewhere and larger:

- **Muon** on the byte rung: BPB 2.1308 → **2.0508 (−3.75%)**,
  extrapolation **−6.73%**. Larger than every mechanism arm combined.
- **`fusion_gate` is excluded from the Muon partition by a substring
  match** on `'gate'` in `EXCLUDE_SUBSTR` — 28% of all 2-D parameters
  routed to AdamW by accident. Unfixed.

Seven mechanism arms have been tested and one succeeded (over-relaxation,
`docs/OPERA_Relax_prereg.md`). The recipe axis has produced a larger
effect on the first attempt than the architecture axis produced in a
day. Finish that axis before reopening this one.
