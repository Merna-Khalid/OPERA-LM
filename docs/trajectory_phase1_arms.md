# Phase-1 Arm Record: Three Native Mechanisms at the 5.9M Rung

Date: 2026-09-08. Follows `docs/trajectory_phase0_verdict.md`.
Rung: d=256, nb=64, L=2, pe-none, fold left, rot free, norm layer, act
tanh; steps=3000, batch=16, max_len=256, seed 42, docs word-level data
(`assets/vocab_docs.pkl`). Driver: `experiments/trajectory_arms.py`
(rerun-safe). Checkpoints + JSONs: `runs_arms/`.
Arms: `homeo_mode='on'` (per-level anchor spinor + gated slerp
relaxation), `node_paths=4` (quotient path h_L ⊗ h_R⁻¹, separate
`quotient_gate`, eager-only), `fold_adapt='on'` (fold twist angle read
from the block's own state, zero-init w). All flags-off-bitwise or
~-incumbent at init; selftest arms `test_homeo_anchor`,
`test_quotient_path`, `test_fold_adapt` (37/37 suite pass).

## Results

| | incumbent | homeo | quotient | fold_adapt |
|---|---|---|---|---|
| PPL (in-length, 5k sents) | 188.64 | 188.26 | 189.40 | 187.64 |
| params | 5,890,324 | +10,240 | +65,664 | +512 |
| step r vs surprisal | −0.252 | −0.253 | −0.248 | −0.259 |
| step p99.9/median | 1.45 | 1.43 | 1.43 | 1.43 |
| disproport. step Δ (top−bot 5%) | −1.03σ | −1.02σ | −1.01σ | −1.04σ |
| GP surprisal d (critical) | 0.356 | 0.438 | 0.373 | **0.599** |
| GP extrap d (spillover+1) | 0.337 | 0.251 | 0.288 | **0.622** |
| arm params moved? | — | no (gate 0.00247 vs 0.00248 init) | barely (0.00258 vs 0.00248) | **yes (w norm 2.06 / 0.98 by layer, from 0)** |

(PPL noise band ±1.5 per train.py's own note: all four tie.)

## Verdicts

1. **fold_adapt — mechanism live, direction promising, unproven.**
   The only arm training reached for. Its trajectory statistics are
   unchanged at this budget (no tail/coupling movement), but on the
   garden-path benchmark it shows the largest surprisal separation
   (d 0.60 vs 0.36 incumbent) AND a near-doubled direction-change
   separation at spillover+1 (extrap d 0.62 vs 0.34) — exactly the
   region where humans show the largest effects. n=12 cells, so this is
   a lean, not a result. Next step for this arm: longer budget (the
   mechanism is cheap: +512 params) and the full-coverage GP eval
   question (word-level OOV limits) — see below.
2. **homeo — recorded null at this budget; cold gates.** The relaxation
   never opened (bias −6 ≈ 0.0025 stays put over 3000 steps). Either
   the init is too cold for the budget or the objective has no use for
   a fixed attractor at this scale. If retried: warmer init (−2) or a
   level-annealed gate; do not silently tune — the point of the arm is
   whether the objective WANTS homeostasis.
3. **quotient — recorded null at this budget; same cold-gate pattern.**
   The algebraic gap is real (the node cannot express relative
   transforms without it) but the objective did not reach for it at
   3000 steps. Same retry rule as homeo.
4. **Method note:** zero/near-zero gate inits make "did training use
   it?" directly measurable from the checkpoint — the arm reports its
   own usage. That convention (already house style) is what makes these
   nulls informative rather than ambiguous.

## Cross-cutting observations

- At the 5.9M/3000-step scale, extrap-vs-surprisal is slightly
  NEGATIVE (−0.08) where the 22M model was +0.13 raw — the
  direction-change coupling emerges with scale/training, consistent
  with Barenholtz et al.'s effect growing GPT-2 S→L. The arms' target
  phenomenon is scale-dependent; iteration at small rungs can rank
  mechanisms but cannot confirm the effect.
- The disproportionality inversion (surprising tokens move the state
  LESS) reproduces at this rung too (−1.0σ), so it is not a 22M
  artifact.

## Correction (2026-09-08): paired per-item re-analysis

Verdict 1 above cites "a near-doubled direction-change separation at
spillover+1 (extrap d 0.62 vs 0.34)". **That comparison does not
survive.** The `standardized` effects divide by the metric's SD *within
each model*, so comparing d across arms compares different denominators.
All four arms scored the SAME 12 items, so a paired per-item test is
available and strictly more powerful.

Per-item GP effect (ambiguous − unambiguous), arm minus incumbent,
n=12 paired items:

| metric | homeo | quotient | fold_adapt |
|---|---|---|---|
| surp | +0.044 (t=+0.37) | −0.024 (t=−0.31) | **+0.224 (t=+2.26, 9/12)** |
| extrap_sp1 | −0.019 (t=−1.06) | −0.024 (t=−1.13) | +0.017 (t=+0.85) |
| turn_sp1 | +0.014 (t=+1.06) | −0.026 (t=−1.18) | +0.034 (t=+1.90) |
| step_sp1 | −0.013 (t=−0.09) | +0.075 (t=+0.40) | +0.151 (t=+0.87) |

So: **extrap_sp1 does not move** (t=+0.85). fold_adapt's one reliable
effect is on **surprisal** (t=+2.26, nominal p≈0.045 — and one of 12
comparisons, so not surviving correction).

This matters for the program's logic, and it cuts against the arm.
Phase 0's central finding is that garden paths separate in the *readout*
(surprisal) while the *geometry* stays silent — the Phantom Transitions
split at inference. An arm whose only reliable effect is a better
surprisal separation has improved the readout, which is where the effect
already lived. The geometry metrics are the ones that needed to move.

Revised verdict 1: **fold_adapt — mechanism live, training reaches for
it (w norm 2.06/0.98 from zero, +512 params), direction positive on all
four metrics, but the only separated effect is on the readout and no
metric survives multiple-comparison correction at n=12.**

Before the 20k run, pre-register ONE primary metric, and make it a
*geometry* metric (extrap or turn at spillover+1) with surprisal
secondary — otherwise a surprisal-only win cannot be distinguished from
a readout improvement and does not answer the question the program asks.

## What would change the verdicts

- fold_adapt at 20k steps / 22M rung: if the GP spillover separation
  survives power, it is a result; if it washes out, recorded null.
- A BPE or larger-vocab OPERA variant would lift the 25% GP coverage
  limit (currently 12/72 cells; NP/Z structurally invisible).
