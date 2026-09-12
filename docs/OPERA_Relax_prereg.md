# OPERA over-relaxed fold — pre-registered protocol

**Version:** 1.0 — 2026-09-08. PRE-REGISTERED before any run.
**Hardware:** M4, 16 GB, MPS. Same rung as every arm in this series.

---

## 1. The argument, from four negatives

Four mechanisms have now failed to move the step-size distribution:

| mechanism | form | tail (p99.9/med) |
|---|---|---|
| incumbent | — | 1.200 |
| `fold_adapt` (sigmoid) | gate | 1.43–1.45 (22M rung) |
| Mamba Δ (softplus) | gate | flat, brake-only |
| bistable per-block | gate | 1.217 |
| bistable low-rank r1 / r4 | gate | 1.174 / 1.184 |

They differ in smoothness, granularity, and stability structure. They
share **one** property: every update is a **convex blend**,

    acc <- (1 - c)*X + c*Y ,   c in (0,1)

Both coefficients are positive, so the result is pinned strictly
*between* X and Y. A gate chooses **where in that interval** to land; it
can never pass either endpoint. **No gating scheme can therefore
increase displacement.** That is arithmetic, not a training failure, and
it is the most economical explanation of all four negatives.

Meanwhile `docs/trajectory_phase2_nulls.md` measured **41–58% of the
geometrically available step range going unused**, with the radius free
(CV 0.05) and observed variance 1.75× the isotropic null. The room
exists; nothing tried so far can reach into it.

## 2. Mechanism

Successive over-relaxation on the fold accumulator, per block:

    acc <- (1 - g)*acc + g*composed ,   g = 1 + tanh(z)  in (0,2)

- **g = 1** → `acc <- composed`, exactly the incumbent.
- **g < 1** → under-relaxation; lands short of `composed`.
- **g > 1** → **(1 − g) is NEGATIVE**: the state is pushed *away* from
  `acc`, past `composed`. A negative coefficient is what extrapolation
  means, and it is precisely what no gate can produce.

Written as `(1-g)*acc + g*composed` rather than the algebraically equal
`acc + g*(composed-acc)`: at g=1 the first is bitwise exact, the second
rounds, since `(a + (b-a)) != b` in floating point. That exactness is
load-bearing (§3).

## 3. Why this design avoids both failure modes of the previous arms

- **The incumbent is INTERIOR to the parameter space.** z zero-init →
  g = 1 → flags-on is **bitwise the incumbent**, verified in the
  selftest, while gradients flow in *both* directions (measured 4.2e-02
  at init). Phase 1's `homeo`/`quotient` nulled behind cold gates whose
  own gradient was suppressed ~400×; the bistable arms had to abandon
  bitwise-incumbent init to get warm gates. This arm needs neither
  compromise.
- **The control is exact.** `under` clamps the same expression at 1
  (`g = 1 + min(tanh z, 0)`): identical parameters, identical
  bitwise-incumbent init, identical dynamics at step 0. The **only**
  difference is whether g can exceed 1. The contrast isolates
  **overshoot**, not capacity.

Invariants: I1 ✓ (g reads content, never position) · I2 ✓ (no scores,
no softmax over positions) · I3 ✓ (compose node untouched) · I4 ✓
(same Fenwick blocks; causality asserted at 0.00e+00).

## 4. Arms

`relax_off` (incumbent) · `relax_under` (control, g ≤ 1) ·
`relax_over` (treatment, g ∈ (0,2)). Byte d512/nb128/L2, T=1024,
batch 8, 3000 steps, seed 42 — identical to the bistable series, so
`bist_off` (BPB 2.1308, tail 1.200) is a second independent incumbent.

## 5. Gates

**R-H1 (primary).** `p99.9/median ≥ 1.60` for `over`, AND
`≥ under + 0.10`. **The threshold does not move because the mechanism
changed** — it is the same bar the four falsified arms were held to.

**R-H2 (usage, no gate).** frac(g > 1) and mean g on **real text**
(never on Gaussian noise — that estimator inverted the bistable
verdict once already). `over` that never overshoots is a cold-parameter
null, not a failed mechanism.

**R-H3 (content coupling).** |partial r(step, surprisal | emb norm)|
for `over` exceeds `under` and `off` by ≥ 0.10.

**R-H4 (quality guard).** BPB within 5% of `bist_off`'s 2.1308.

**R-H5 (stability, new hazard).** g > 1 compounds over the fold's
≤ log T steps and, unlike the bistable arm, has **no `tanh` bounding
it**. Divergence-guard skip counts are reported; a run with > 1% skipped
steps is reported as unstable rather than as a clean result.

## 6. Pre-stated interpretation

- **R-H1 pass with g>1 used** → displacement in this architecture is
  reachable, and it required leaving the convex class. That is a
  positive mechanism result and the first of the series.
- **R-H1 fail with g>1 used** → extrapolation was available, trained,
  used, and still produced no excursions. Combined with the four gating
  negatives this would mean **displacement is inaccessible in this
  architecture by any local update rule on the fold accumulator** — a
  far stronger and more publishable statement than any individual null.
- **R-H1 fail with g ≤ 1 everywhere** → the objective actively prefers
  under-relaxation. Report as a training/objective finding; note that
  the LM loss appears to reward *damped* integration, which would
  explain the whole series at once.

## 7. Limitations at registration

Single seed. One rung. 3000 steps ≈ 0.56 epochs. The fold is per-prefix,
so this is over-relaxation *within* a prefix's composition, not across
tokens. No `tanh` bound on the positive feedback (§R-H5).

---

## Outcomes — 2026-09-08

| arm | BPB | step median | p99 | p99.9 | **p99.9/med** | partial r | frac(g>1) | mean g |
|---|---|---|---|---|---|---|---|---|
| `bist_off` (incumbent) | 2.1308 | 19.420 | 22.435 | 23.309 | 1.200 | −0.2249 | — | — |
| `relax_under` (control) | 2.1176 | 18.751 | 21.753 | 22.571 | 1.204 | −0.2150 | 0.000 | 0.916 |
| **`relax_over`** | 2.1470 | **21.134** | **31.571** | **35.003** | **1.656** | **−0.3069** | **0.636** | 1.100 |

### R-H1 — **PASS.** The first mechanism in the series to move the tail.

`over` reaches **1.656** (threshold ≥ 1.60) and beats the control by
**+0.452** (threshold ≥ 0.10). For scale, every previously tested
mechanism sat in 1.17–1.22 and the isotropic null is ~1.10.

The distribution *shape* changed, not just its scale: p99.9 went
23.309 → 35.003 (**+50%**) while the median moved 19.420 → 21.134
(+9%). The control, with identical parameters and identical
bitwise-incumbent init, stayed at 1.204 — indistinguishable from the
incumbent. **The entire effect is attributable to g being allowed
above 1.**

### R-H2 — overshoot genuinely used
frac(g>1) = **0.636**, mean 1.100, median 1.141, max 2.000, and 19.8%
of updates above g = 1.5. Measured on real text, not noise.

### R-H3 — **FAIL**, narrowly, and this is the honest limitation.
|partial r| = 0.3069 for `over` vs 0.2150 (under) and 0.2249 (off) —
gaps of **+0.092** and **+0.082** against a required 0.10. Content
coupling is the strongest of any arm measured and moved 36% in
magnitude, but did not clear the bar.

**What this means, stated plainly: the excursions are real but not yet
well-targeted.** The state can now move far; it is not yet moving far
*specifically at the informative tokens*. Displacement and its
content-alignment are separate problems, and this arm solves the first.

### R-H4 — **PASS.** BPB 2.1470 vs the 2.2373 guard. Note it is *worse*
than the incumbent's 2.1308 by 0.76%: overshoot costs a little quality.

### R-H5 (stability) — **PASS, and load-bearing.**
**0 divergence skips out of 3000 steps in both arms** (0.000%), final
losses 1.7254 / 1.7365. This was the main way a positive could have been
fake — g > 1 is unbounded positive feedback with no `tanh` containing
it, and a tail produced by numerical blow-up would look identical in the
p99.9 statistic. It did not happen.

### Verdict — the convexity diagnosis was correct

Five mechanisms failed because every one was a gate on a convex blend,
which is pinned between its endpoints and therefore cannot increase
displacement. The single mechanism that can produce a **negative
coefficient** — pushing the state past `composed` rather than toward
it — moves the tail by 38% on the first attempt, with a
capacity-matched control that does not move at all.

That is a positive mechanism result, and the argument for it was derived
from this project's own accumulated negatives rather than imported.

### Caveats

Single seed; the whole series is single-seed and the same-config
reproducibility floor measured earlier is ΔBPB 0.00088, which says
nothing about the variance of a tail statistic. One rung, 3000 steps
(~0.56 epochs). R-H3 failed, so this is displacement without
demonstrated content-targeting. Quality cost is real if small. And the
fold is per-prefix, so this is over-relaxation *within* a prefix's
composition, not across tokens.

### The next question, now well-posed

Not "can the state move" — it can. **"Can the movement be aimed?"**
R-H3 is the gate that failed, and it is the one that matters for the
garden-path claim: excursions must land on the reinterpretation points,
not merely exist. That needs its own registration.

---

# ADDENDUM 2 — pre-registered re-validation under Muon (2026-09-09)

**Registered BEFORE any run.** The 2026-09-08 outcomes above were
measured on the AdamW baseline that `OPERA_Recipe_results.md` later
showed was 3.75% off its achievable quality. The recipe axis has since
moved twice (Muon; then fusion_gate routing + wd 0.01 → BPB 2.0371).
Before over-relaxation can be considered for Path A it must be
re-validated under the current incumbent recipe — both because a
positive measured on a mis-trained baseline is unproven, and because
optimizer–mechanism interactions are real (LO-Muon's falsification
showed the update rule changes what the fold mechanisms do).

**Arms:** `relaxM_off` / `relaxM_under` / `relaxM_over` — identical to
the 2026-09-08 arms except the recipe is `rx_fgate_wd` (Muon lr 0.02,
include fusion_gate, wd 0.01). Same rung, seed, steps, data.

**Gates:**

- **RM-H1 (the original falsifier, unchanged):** `over` p99.9/median
  ≥ 1.60 and ≥ `under` + 0.10. If the tail effect was an AdamW
  artifact, it dies here and the mechanism line closes for good.
- **RM-H2 (Path-A relevance — new, decision-bearing):** the quantity
  Path A actually gates on is length extrapolation. *PASS* if `over`
  improves the 1025–2048 bucket ≥ 1.5% vs `relaxM_off` with BPB
  regression ≤ 1.0%. **This is the adopt-into-Path-A gate**: PASS →
  over-relaxation is proposed as the OPERA arm's fold (amendment to
  the Path A prereg, before training); FAIL → it stays a trajectory
  result, interesting and closed.
- **RM-H3 (guard):** BPB within 1.5% of `relaxM_off` (the 2026-09-08
  cost was −0.76%; more than ~1.5% under the better optimizer means
  the trade is no longer worth it at this rung).

**Pre-stated interpretation.** RM-H1 pass + RM-H2 pass → the one
positive mechanism of the series survives its re-validation AND
targets the study's extrapolation axis; strongest architecture
candidate for Path A. RM-H1 pass + RM-H2 fail → displacement is real
but does not buy extrapolation; recorded, not adopted. RM-H1 fail →
the 2026-09-08 positive was baseline-dependent; line closes.

**Limitations:** single seed; one rung; the 1025–2048 bucket is 2×
training length, not 16× — adoption for Path A is a transfer bet
stated openly, and the Path A position-curve instrument itself is the
final arbiter of H1 there.

## Outcomes (Addendum 2) — 2026-09-12

| arm | BPB | extrap | p99.9/med |
|---|---|---|---|
| relaxM_off | 2.0411 | 4.0562 | **1.859** |
| relaxM_under | 2.0451 | 4.0517 | 1.833 |
| relaxM_over | 2.0927 | 4.1763 | **1.958** |

### RM-H1 — **PASS.** The tail effect survives the better optimizer:
1.958 ≥ 1.60, and over − under = +0.125 ≥ 0.10.

### RM-H2 — **FAIL, decisively.** Extrapolation is 2.96% *worse*
(needed ≥ 1.5% better); the displacement does not buy extrapolation.

### RM-H3 — **FAIL.** BPB cost 2.52% (guard 1.5%). The trade worsened
under Muon (it was −0.76% under AdamW): the better the optimizer, the
more over-relaxation costs.

### Verdict (pre-stated): **RM-H1 pass + RM-H2 fail → displacement is
real but does not buy extrapolation; recorded, not adopted.** The
one positive mechanism of the series is now a trajectory-only result.

### Unregistered observation, load-bearing for the trajectory
program: **Muon itself fattened the tails.** The AdamW incumbent's
tail was 1.200; the identical-architecture Muon incumbent measures
**1.859** — the recipe change alone moved the trajectory statistic
past the 1.60 bar the entire mechanism series was held to. Part of
the "constant-speed deficiency" the program set out to fix was an
optimizer artifact. Any future trajectory claim must state its
recipe, and the phase-0/2 constant-speed measurements carry an
AdamW caveat in the writeup.
