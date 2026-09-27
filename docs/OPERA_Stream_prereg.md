# OPERA stream & tree-width arms — pre-registration DRAFT (2026-09-24)

**Status: DRAFT — NOT REGISTERED.** Implemented and selftested
(`test_stream_arms`), never trained. The gates below are *proposals*;
this document becomes a registration only when Merna signs off on §4 and
dates it. Running any arm before that makes it exploratory, not
confirmatory. Origin: architecture review of 2026-09-24 (review items
1–5 below); exploratory ancestors `OPERA_Depth_Survival_diagnostic.md`
(F1–F7, G1–G4) and `OPERA_Readout_survey.md`.

**Sequencing with the registered scale-graded study.**
`OPERA_ScaleGraded_prereg.md` runs FIRST and exactly as registered
(`grade_off` / `grade_r24` / `grade_full`, flags unchanged). Nothing here
alters its protocol: every new flag defaults to the incumbent, and
flags-off is bitwise `HEAD` (parameters, logits, gradients, RNG stream —
verified across seven configurations including dropout). If an SG arm
is adopted, it enters Stage 2's base by amendment.

---

## 1. Measured premises (all on the incumbent, reproducible from the repo)

**P1 — a third of the incumbent's parameters are dead.** The head reads
each layer's fold output, so the last layer's `cross_mlp` and
`blend_gate` compute a stream nothing reads. Backward pass on the byte
incumbent config (d512/nb128/L2, vocab 259): **1,115,907 of 3,293,447
parameters (33.9%) get no gradient**; 9.0% at L=8; ~11.8M of 154.8M in
the Path A OPERA arm. The live incumbent is 2.18M, not 3.29M. Selftest
`test_stream_arms (a)` pins this.

**P2 — the residual stream is a convex scalar blend, and the head never
sees it.** `current = g·MLP(P) + (1−g)·current` (G1: ∏(1−g) = 0.021 at
L=8, highway closed). The head consumes the most compressed readout in
the stack (F7/G3: layer-0 recency k=1 0.83 → final 0.68).

**P3 — every layer is trained as a standalone LM through the shared
head.** `aux_weight=0.5` on every non-final layer (never ablated): the
final layer carries 67% of the loss at L=2 and **22% at L=8**. The
shared head also forces every P_l into one output basis — a candidate
cause of G2 (identical recency cliff at every layer).

**P4 — values never move between quaternion slots inside a layer.** The
node's value path is block-diagonal (slot k of the parent is built from
slot k of the children); all within-tree value transformation is
`rot_free` (6,912 params); the 787,200 fusion-gate params only modulate.
Jacobian ∂parent/∂left-child on the trained `rx_fgate_wd` checkpoint:
**54–77% of energy in the diagonal 4×4 blocks** (1/128 of the entries);
the cross-block part is multiplicative (block j rescales block k, never
writes into it). Per-prefix state = d = 512 numbers, no expansion.
Proposed mechanism for F3 (cliff) and F4 (rank collapse); the
scale-graded readout is a hand-wired instance of the missing routing.

**P5 — the fold's two jobs share one gate; zero-fold readouts exist.**
Tree nodes compose equal-span siblings at one level; the fold composes a
large accumulator with a small block, scales mixed per call (36–38% of
the gate's dL/dW). Only the bias is fold-specific today. Separately, at
t+1 = 2^a the readout is a raw tree node (F6: +0.18…+0.28 bits above
neighbors), and graded groups start from zero (no neutral element).

## 2. Mechanisms (flags on `OperaSpinorFenwickTree`, all default = incumbent)

| flag | change | params (byte d512/L2) | init |
|---|---|---|---|
| `head_mode='stream'` | head reads `LN_l(current_l)` after each layer's update (aux heads per layer) | +2,048 (LayerNorms); live 2.18M → 3.30M | deterministic; NOT bitwise |
| `resid_mode='add'` | `current += cross_mlp(P)`; pre-norm on each tree input; no blend gate. Requires `'stream'` | −129,538 vs stream | NOT bitwise |
| `aux_weight` (train) | aux-layer loss weight 0.5 → w | 0 | loss-side only |
| `fold_gate='separate'` | left fold gets its own fusion gate, initialized as a COPY of the tree gate | +787,200 | **bitwise incumbent at init** (selftested), interior |
| `fold_h0=True` | every position folds from a learned per-layer h0 over ALL its blocks; with grading, unrouted groups start at h0 | +1,024 | zero-init; NOT bitwise; +T compositions/layer |
| `state_mult=k` | tree/fold at width k·d behind per-layer in/out projections; stream, head, MLP stay at d | k=2: 3.29M → 7.76M total | variance-preserving projections; NOT bitwise |

Invariants: I1 ✓ (no positional signal added) · I2 ✓ (no scores, no
softmax over positions) · I3 ✓ (compose node untouched; `state_mult`
runs the same node at larger width; projections are position-wise) ·
I4 ✓ (same Fenwick prefix set; causality selftested for every arm and
for `h0 + fold_grade`). The incremental decoder is exact for every new
flag (≤ 2.6e-6); it refuses `fold_grade` (pre-existing gap: it never
applied the graded write masks).

## 3. Protocol (identical to the SG family)

`experiments/repr_study.py`, bytes T=1024, d=512, nb=128, L=2, 3000
steps, batch 8, seed 42, Muon 0.02 / include fusion_gate / wd 0.01,
eval to 2048 (`--eval-cap 2048`). Serialize runs (16 GB). Same-session
controls in every stage. Probe battery for mechanism gates:
`experiments/depth_survival.py` (k-lag decodability, PR, F6 texture),
invoked with the arm's flags, e.g. `--arch '{"head_mode": "stream"}'`
(and `--d 640` / `--d 920` for the width controls). For stream-head
models the "final readout" in every gate is `H_{L-1}` — the head's
actual input, recorded as `head_src` in the probe JSON — not the fold
output `P_{L-1}`. The probe's reconstructed forward matches `forward()`
to ≤ 1e-6 per-position CE on the incumbent and on the arms.

**Stage 1 — stream side** (arms defined in `repr_study.ARCH`):

| arm | config |
|---|---|
| `st_off` | incumbent, fresh control |
| `st_head` | `head_mode='stream'` |
| `st_add` | `head_mode='stream', resid_mode='add'` |
| `st_auxw0` | `aux_weight=0.0` |
| `st_wide` | incumbent at d640/nb160 — **live-param-matched control for `st_head`** (3.32M vs 3.30M live) |

**Stage 1b — the residual at depth:** `st8_head`, `st8_add` with
`--layers 8 --steps 1500` (the `l8_probe` budget).

**Stage 2 — fold/tree side**, on `STAGE2_BASE` (set by amendment to the
Stage-1 winner, incumbent if none):

| arm | config |
|---|---|
| `s2_base` | base, fresh control |
| `s2_foldgate` | + `fold_gate='separate'` |
| `s2_h0` | + `fold_h0=True` |
| `s2_sx2` | + `state_mult=2` |
| `s2_sx2_wide` | base at d920 (fold head) / d800 (stream head) — live-param-matched control for `s2_sx2` (±0.5%) |

Timing: not yet measured for the new arms. `s2_sx2` runs the tree at 2×
width (gate FLOPs ~4×); `st_wide`/`s2_sx2_wide` are wider everywhere.
Expect those three to exceed the ~75–95 min/arm of the d512 family.

## 4. Gates — PROPOSED (Merna to confirm or edit before registering)

- **ST-H1 (head, adopt):** `st_head` BPB ≤ 0.99 × `st_off` AND ≤
  `st_wide` BPB AND 1025–2048 extrapolation PPL not worse than `st_off`
  by >1%. Beating `st_wide` is the capacity guard: a win explained by the
  1.1M newly-live params alone must not be credited to the head.
  *Mechanism (non-binding):* final-readout k=1/k=2 decodability ≥
  `st_off`'s (the head now sees mid-stack detail).
- **ST-H2 (residual):** at L=2, `st_add` BPB ≤ 1.005 × `st_head`
  (non-inferiority — little depth for a residual to matter). Adopt only
  if Stage 1b also shows `st8_add` BPB ≤ 0.99 × `st8_head`.
  *Mechanism (binding at L=8):* S_in identity decodability at the last
  layer ≥ `st8_head`'s.
- **ST-H3 (aux weight, adopt):** `st_auxw0` BPB ≤ 0.99 × `st_off`.
  Pre-stated: a pass is evidence for P3's shared-basis reading; a fail
  at L=2 does NOT close the question at L=8 (the final layer holds 67%
  of the loss at L=2 vs 22% at L=8) — revival needs an L=8 registration.
- **FG-H1 (fold gate):** `s2_foldgate` BPB ≤ 0.99 × `s2_base`. Disclosed:
  +787k params (+24% of the live stream-head model) and no width control
  proposed — **open decision for Merna:** add one, or require a larger
  margin (e.g. 0.985).
- **H0-H1 (learned h0):** `s2_h0` BPB ≤ 0.995 × `s2_base` AND
  *mechanism kill-switch:* the power-of-two CE bump (F6 matched-position
  measure) shrinks by ≥ 50%. BPB pass without the bump shrinking → no
  adoption (noise).
- **SX-H1 (state width, adopt):** `s2_sx2` BPB ≤ 0.99 × `s2_sx2_wide`
  AND extrapolation not worse than `s2_base` by >1%.
  *Mechanism kill-switch:* k=8 byte decodability from the final readout
  ≥ 0.20 (incumbent ≈ 0.105). SX-H1 pass with kill-switch fail → no
  adoption.
- **Guards (all arms):** `python -m opera_lm.selftest` ALL PASS; param
  counts logged per arm; single dose per arm as listed.

## 5. Pre-stated interpretations

- ST-H1 passes, `st_wide` beats `st_off` by a similar margin → the
  incumbent was capacity-starved by its own dead parameters; the head
  fix is the cheapest way to reclaim them.
- ST-H1 fails *only* on the `st_wide` clause → the gain is capacity,
  not topology; still adopt the fix as a parameter-accounting correction
  (it makes nominal = live), but make no mechanism claim.
- SX-H1 passes, `s2_sx2_wide` does not beat `s2_base` → state width, not
  parameter count, is what binds: P4's mechanism is the explanation.
- SX-H1 kill-switch passes but BPB does not → addressability is
  expressible but does not pay at this rung (same reading as SG's H1-fail/H2-pass).

## 6. Risks (honest)

- `resid_mode='add'` at L=2 is nearly untestable; Stage 1b is where it
  lives or dies.
- Pre-norm plus additive residual changes leaf statistics at layer 0
  (the tree now sees LN(embedding)); a small tax is possible.
- `state_mult` projections are random-init: the tree sees a random
  linear mix of the stream at step 0, unlike the incumbent. The width
  control shares the protocol, not this init.
- Same-session MPS noise ≈ 0.04%; cross-session drift ~0.2% — fresh
  controls in every stage for that reason.

## Amendments log

*(empty)*
