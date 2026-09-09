# OPERA scale-aware optimizer (LO-Muon) — pre-registered protocol

**Version:** 1.0 — 2026-09-09
**Status:** PRE-REGISTERED before any stage-1 implementation or run.
Stage 0 items are ungated recipe measurements (the
`OPERA_Recipe_results.md` precedent). Deviations are appended to the
Amendments log, never rewritten.
**Hardware:** MacBook M4, 16 GB, MPS. No Kaggle quota. **Path A is
untouched** — its optimizer is pinned to the §4.8 recipe by
`OPERA_PathA_prereg.md`, and nothing in this document modifies it.

---

## 1. The problem, from this week's own measurements

OPERA's compose operator is **one weight set applied at ⌈log₂T⌉ tree
levels** (`docs/OPEN_scale_tied_operator.md`). The level-tied 2-D
tensors are `fusion_gate.{l}.weight` (3nb × 2d) and `rot_free`
(batched 3×3 per block). Everything modern — Muon included — is
layer-wise: one weight matrix, one gradient, one norm choice. Here

    dL/dW = Σ_ℓ G_ℓ ,   ℓ = 1..⌈log₂T⌉

with applications differing in **count** (level ℓ has T/2^ℓ nodes;
measured share 34.1% → 2.3%, root exactly 0) and **input
distribution** (level 1 consumes embeddings, ‖h‖ ≈ 10.68; levels ≥ 2
composed states, ‖h‖ ≈ 6.93).

**What is already falsified.** β^ℓ rescaling of the summed gradient,
monotonically worse at every dose (`OPERA_LevelGrad_prereg.md`). That
was a *magnitude* intervention under **AdamW**, which is
element-wise and magnitude-sensitive.

**The observation this study is built on.** Newton–Schulz maps a
matrix toward its orthogonal polar factor and is approximately
scale-invariant: NS₅(c·G) ≈ NS₅(G). So under Muon, the 15:1 magnitude
imbalance is *mostly discarded already* — which is consistent with the
β^ℓ null being about AdamW, and means it does **not** bind here. What
Muon cannot undo is *which level's geometry dominates the directions
of the sum*: the top singular directions of Σ G_ℓ track its largest
contributors (level 0, embedding-input). Orthogonalizing the sum
optimizes a level-0-dominated direction at every step. Formally,

    NS(Σ_ℓ G_ℓ)  ≠  NS(Σ_ℓ NS(G_ℓ))

and the two agree only when the levels' gradient directions already
agree. The right-hand side gives every level an **equal-magnitude
direction vote** instead of a magnitude vote. That operator —
orthogonalize per application context, then sum — does not exist in
any layer-wise framework, because no layer-wise framework has a weight
that is not a layer.

## 2. Prior art, checked

Muon/Scion/Gluon/Moonshot: layer-wise by construction. Weight tying
across **depth** (Universal Transformer, Mixture-of-Recursions) or
**time** (RNN BPTT): uniform application counts, no imbalance.
GradNorm and multi-task rescaling: magnitude-only. **No prior art
found for per-application-context orthogonalization of a scale-tied
operator.** (If a reviewer finds some, this section gets an amendment,
not a defence.)

## 3. Stage 0 — pin the honest incumbent (ungated)

The mechanism study must not win by recovering known misallocations.
Stage 0 items may already be running in a parallel session
(`OPERA_Recipe_results.md` "Next" §1–2); results slot into the
incumbent table regardless of who runs them.

**0a. `fusion_gate` into the Muon partition.** Single-variable spec:
add an explicit include-list evaluated BEFORE the exclusion substrings
—

    include = ('fusion_gate',)
    muon ⇐ p.ndim ≥ 2 and (name matches include
                            or no EXCLUDE_SUBSTR matches)

— leaving every other name's routing byte-identical (side effects on
`fold_adapt_w` / `bist_*` are thereby avoided; they keep today's
routing). At d=512 this moves 917,760 2-D parameters (28%) from AdamW
to Muon.

**0b. Weight decay on the Muon side** (`weight_decay=0.01`, the
Moonshot scaling recipe; the whole model currently runs wd = 0.0 at
`train.py:703`). AdamW side unchanged, single variable.

**0c. Three-way understanding run** (survey §4.2, still unrun):
Muon-all / Muon-with-`rot_free`-moved-to-AdamW / AdamW-everywhere.
Answers whether Muon's win is the rotation manifold arriving by the
update rule. No gate; it is context for §5's interpretation, and it
bounds the ceiling of stage 2.

**New incumbent** = best BPB among {`rx_muon`, 0a, 0b, 0a+0b} at the
byte d512 rung (T=1024, batch 8, 3000 steps, seed 42 — the series
standard). Every stage-1 arm runs with exactly the incumbent's recipe;
no stacking (the `rx_full` lesson).

## 4. Stage 1 — LO-Muon, the mechanism

**Level-orthogonalized Muon.** For each level-tied tensor W
(`fusion_gate.{l}.weight`, `rot_free`):

1. **Router** (generalizes `_GradScale`, proven through activation
   checkpointing `use_reentrant=False`): at `build_tree`'s per-level
   `_compose` call sites, W is consumed via
   `LevelCapture.apply(W, ℓ)` — identity forward; backward **copies**
   g into `buffers[W][ℓ]` and returns it unchanged. Total `.grad` is
   byte-identical to today: an incumbent run with the router active
   and LO off must reproduce the incumbent exactly (asserted).
2. **Per-level momentum:** m_ℓ ← μ·m_ℓ + g_ℓ. All buffers decay every
   step whether or not their level occurred (uniform cadence; short
   curriculum sequences decay high levels silently).
3. **Per-level orthogonalization, then sum, then one final NS₅:**

       u  = NS₅( Σ_ℓ  w_ℓ · NS₅(m_ℓ) ),   w_ℓ = 1  (v1: uniform)

   The sum happens **in whitened space**; the final NS₅ restores
   Muon's unit-RMS update discipline. NS₅∘NS₅ is near-idempotent —
   the idempotence error is measured and reported (it bounds how much
   the final NS could dilute the level mixing).
4. `W ← W − lr · max(1, m/n)^0.5 · u`, Muon-side weight decay per the
   incumbent recipe.

**No new parameters.** Capacity exactly matched by construction (the
level-grad study's advantage, kept). The forward is untouched — this
is an optimizer-side intervention only; checkpoints stay
architecturally identical.

Invariants: I1–I4 untouched (no forward change at all).

### 4.1 Stage 1 step 0 — the 20-minute kill-switch, run BEFORE building

If the levels' gradient directions already agree, LO-Muon is a no-op
by construction and must not be built. Using the existing per-level
hook machinery, measure the principal-angle cosine between whitened
level gradients, cos(NS(G_ℓ), NS(G_ℓ′)), on real batches:

- **median inter-level cosine ≥ 0.95 → the design is falsified before
  construction.** Recorded as such; stage 1 does not run; §1's
  operator is predicted to reduce to the incumbent. (This is the
  cheapest possible outcome of the study and is counted as a result.)
- median < 0.95 → proceed. The measured cosine table is published with
  the outcomes either way — it is the quantity that makes the whole
  story interpretable.

## 5. Arms and gates (stage 1)

Arms: **incumbent** (post-stage-0 Muon) vs **lo_muon** (uniform w_ℓ).
Byte d512/nb128/L2, T=1024, batch 8, 3000 steps, seed 42. One dose,
one treatment — the fishing guard.

**O-H1 (primary).** BPB improves ≥ **1.0%** vs the incumbent. (Rung
reproducibility floor is ΔBPB 0.00088 ≈ 0.04%; Muon's own effect is
3.75%; the gate sits 25× above noise and below any effect that would
matter.)

**O-H2 (extrapolation, decision-bearing, pre-registered as a
trade-off branch — mirroring the level-grad registration).**
Extrapolation bucket (1025–2048) improves ≥ 2.0% with BPB regression
≤ 1.0% ⇒ **PASS-with-tradeoff**. Deep levels weigh more at eval; an
optimizer that equalises the level vote should show there first if
anywhere.

**O-H3 (usage — no gate; the house rule that makes nulls
interpretable).** Report ‖u_ℓ‖ shares next to the gradient-share table
(34.1/25.3/14.8/…). If the update shares replicate the gradient
shares, routing failed and nothing is concluded. Also report the §4.1
cosine table and the NS₅ idempotence error.

**O-H4 (guards).** Zero new parameters (assert); router-on/LO-off
reproduces the incumbent bitwise; divergence-guard skip counts
reported (a tail of NaN skips would make any BPB win unreadable).

## 6. Pre-stated interpretation

- **O-H1 pass** → the first optimizer result specific to scale-tied
  composition; per-application-context orthogonalization is a real
  lever; the OPEN problem's weights-side attack (5b, level-conditioned
  low-rank deltas) becomes the motivated sibling. The paper gains an
  optimizer half of the scale-tying story to match the measurement
  half.
- **O-H1 fail with O-H3 confirming routing** → magnitude rebalancing
  (β^ℓ, AdamW) AND direction rebalancing (LO, Muon) are both dead.
  The optimizer side of the scale-tied problem is closed, and OPEN
  5b is the only remaining attack. The OPEN problem gains a closed
  branch — a stronger document, not a weaker one.
- **§4.1 fires** → the levels already agree, the tied operator is
  pulling one direction after all, and BOTH the optimizer and
  weights-side attacks lose their motivation at this rung. Cheapest
  outcome; recorded with the cosine table as the evidence.

**One pre-registered retry, only if null-with-routing-confirmed:** the
per-level momentum placement (m_ℓ per level vs shared momentum on the
summed whitened gradient) is the one free design choice v1 fixed
arbitrarily. If v1 nulls cleanly, the shared-momentum variant may run
once, as an amendment logged before it starts. No third attempt; two
nulls close the line.

## 7. Stage 2 — exact polar retraction for the 3×3 rotors (cheap, expected null)

NS₅'s orthogonality error at 3×3 is 0.5594 (vs 0.15–0.21 at the large
shapes it was tuned for). Replace NS₅ on `rot_free` with batched exact
polar (3×3 SVD; closed form optional). **Registered as a measurement,
not a hope:** "Muon is Not That Special" (arXiv:2605.11181) predicts a
null, and `rot_free` is 0.3% of the partition — the ceiling is low and
that is stated up front. Value: it closes the exact-vs-iterative
question for the one tensor where the geometry is the point. One arm,
no gate, one paragraph of outcomes.

## 8. Limitations at registration

Single seed (project convention; BPB repro floor 0.00088 measured at
this rung, but the variance of a *difference* between optimizers is
not established). One rung, 3000 steps ≈ 0.56 epochs — under-trained
by construction, and optimizer interactions can change sign with
budget; any pass is re-confirmed at 10k steps before it is claimed in
the paper. MPS. Router memory is ⌈log₂T⌉ extra copies of two tensors
per layer (≈4 MB at this rung) — trivial here, noted for porting.
Under DDP the pass-through capture must happen before gradient
all-reduce; not needed on this rung, flagged for whoever ports it.
Stage 0c's AdamW-everywhere arm re-uses the falsified incumbent
recipe on purpose (it is the control leg, not a proposal).

## Amendments log (append-only)

**2026-09-09 — implementation scope: LO v1 covers `fusion_gate` only;
`rot_free` deferred.** Building the router revealed that `rot_free`'s
only per-level decomposition point is the R_L/R_R/R_O rotation tensors
(assembled once per layer, before the level loop) — whitening would
happen in R-space, not in the parameter space where NS5 operates, and
the two are not interchangeable. `fusion_gate.{l}.weight` decomposes
natively (captured at the point of consumption, per level). rot_free
is 0.3% of the partition; its per-level treatment is deferred to a v2
if LO passes. Relatedly: the router captures the FOLD's applications of
the shared weight (36–38% of dL/dW at T=1024, measured by the
kill-switch) as a dedicated scale-mixed bucket with its own momentum —
dropping it would have silently discarded a third of the signal.

**2026-09-09 — KILL-SWITCH RESULT: PROCEED (runs_reprs/level_cosine.json).
Median inter-level whitened cosine 0.012 against the 0.95 threshold
(n=360 pairs, 4×4 batches at T=1024 on the rx_muon checkpoint); the
levels are near-orthogonal, so NS(Σ G_ℓ) and NS(Σ NS(G_ℓ)) are
maximally different operators. Measured alongside: level-1 input RMS
1.020 vs 0.63–0.70 deeper levels (the 1.5× gap the derived weighting
corrects), and gradient shares L1 30–33% decaying monotonically to ~0
at the root — consistent with the 2026-09-07 audit at T=256.**

**2026-09-09 — literature survey (docs/OPERA_Optimizer_survey.md)
landed before any stage ran; three protocol updates.** (1) Stage 1 now
has TWO treatment arms, one mechanism: `lo_uniform` (w_ℓ = 1, the
original v1) and `lo_derived` (w_ℓ ∝ 1/x̄_ℓ, the running mean input
RMS at level ℓ) — the latter is *derived*, not guessed: Bernstein's
RMS→RMS operator-norm steepest-descent derivation assumes one input
distribution per layer, and repairing that assumption for a
scale-tied operator yields exactly the per-level inverse-norm
weighting (survey §2; Gluon's layer-wise smoothness is the
layer-granularity version of the same correction). `lo_derived` is
predicted-primary by theory; `lo_uniform` is the theory-free ablation.
O-H1 applies to each arm separately with the same 1.0% bar. The
pre-registered single retry (momentum placement) is unchanged and
still one. (2) Stage 2's exact-polar arm may be replaced by a
Scion-style spectral-norm **constraint** on the 3×3 blocks if that is
cheaper to implement — same gate structure (measurement, expected
null); whichever variant runs is named in the outcomes. (3)
Implementation note, not a protocol change: per-level NS should use
Gram Newton–Schulz (Dao-AILab, 40–50% off the orthogonalization step)
if cost is ever measurable; at this rung it will not be. The §4.1
kill-switch runs first and binds both treatment arms.
