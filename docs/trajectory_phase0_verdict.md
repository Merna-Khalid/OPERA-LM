# Phase-0 Verdict: Trajectory Dynamics of OPERA vs RoPE vs Mamba

Date: 2026-09-07. Instrument: `opera_lm/trajectory.py` (per-token step,
turn, constant-velocity extrapolation error; position-controlled,
embedding-norm and frequency partialled). Benchmark:
`opera_lm/gp_benchmark.py` (SAP ClassicGP stimuli + N=2000 human SPR
effect sizes, `assets/gp_stimuli/`). Checkpoints: OPERA 22M word-level
(d=640, nb=160, L=4, so3), matched RoPE transformer 22.9M, Mamba-130m
(HF, external yardstick, BPE — not a matched control).
Results JSONs: `runs_traj_{opera,rope,mamba}.json`, `runs_gp.json`.

## A. Natural-text trajectory statistics (n=1,280 fixed windows each)

| metric vs surprisal | OPERA | RoPE | Mamba |
|---|---|---|---|
| step, raw r | −0.03 | −0.13 | −0.03 |
| turn, raw r | −0.16 | −0.16 | +0.04 |
| extrap, raw r | +0.13 (100% pos) | +0.07 (100% pos) | 0.00 |
| extrap, r \| emb+freq | +0.03 | +0.02 | +0.01 |
| step p99.9/median | 1.44 | 1.58 | 1.94 |
| extrap p99.9/median | 2.12 | 3.34 | 4.61 |
| state movement at top-5% surprising tokens | −0.76σ (less) | −1.18σ (less) | −0.05σ (same) |

Findings:

1. **No architecture moves its state MORE on informative tokens.** The
   disproportionality test is flat (Mamba) or inverted (OPERA, RoPE).
   The accelerator half of content-adaptive integration is empty in all
   three.
2. **The raw direction-change signal (extrap vs surprisal, +0.13/+0.07)
   is confound**: partialling embedding norm and frequency leaves
   +0.01–0.03. Same confound class that faked earlier probes; the
   instrument's controls caught it.
3. **Constant-speed confirmed again** for step (p99.9/med ≤ 1.94
   everywhere). Mamba has modestly heavier extrap tails (4.61) but zero
   surprisal coupling.
4. **Mamba's trained Δ gates content asymmetrically** (Δ vs surprisal
   r=+0.077, +0.085 after emb control; the 5% most predictable tokens
   get Δ at −0.70σ, the most surprising get ≈average). Content-adaptive
   integration exists at scale as a *brake* on the predictable — not as
   amplification of the informative. To our knowledge this is the first
   published per-token Δ-vs-surprisal measurement in language.

## B. Garden-path benchmark (SAP ClassicGP, human RT ground truth)

| | OPERA (12 cells*) | RoPE (12 cells*) | Mamba (72 cells) |
|---|---|---|---|
| surprisal GP effect, crit (std units) | +0.48 | +0.78 | **+1.26** |
| surprisal GP effect, spill+1 | −0.02 | +0.00 | +0.48 |
| best trajectory metric, crit | extrap +0.21 | extrap +0.33 | turn +0.04 |
| human construction ordering r (surprisal) | n/a (2 constr.) | n/a | **+0.83** |
| item-level tracking, surprisal | +0.43 | −0.04 | +0.06 |
| item-level tracking, turn | +0.70 | −0.55 | +0.13 |

\* word-level models: 25% stimulus coverage after OOV filtering; NP/Z
unmeasurable (comma disambiguation is invisible to punctuation-free
tokenization — measured, not assumed: exactly zero effect on every
metric). n=6 items/construction → OPERA/RoPE item-level numbers are
descriptive only.

Findings:

5. **Surprisal is the only signal that separates garden-path from
   control sentences** in all three architectures; Mamba's surprisal
   even reproduces the human construction ordering (r=+0.83, 3 points).
6. **No trajectory metric tracks human per-item GP cost** in any model
   (all |r| ≤ 0.13 at full coverage), consistent with Huang et al.
   2024's null for surprisal and extending it to geometry metrics.
   Barenholtz et al.'s positive direction-change result is on larger
   BPE models (GPT-2/Pythia) with human RT regression — a scale/class
   gap, not a contradiction yet.

## Verdict (pre-registered style, before Phase 1)

Outcome class (a): **deficiency confirmed, with a sharpened shape.**
The gap is not "states should move more on surprising tokens" (they
move *less*, everywhere). It is: (i) no model amplifies state movement
for informative content — the accelerator is empty; (ii) the only
working GP signal is the readout (surprisal), the geometry is silent —
exactly the Phantom Transitions split, now shown at inference time on
garden paths; (iii) content-adaptive step sizing demonstrably trains
stably at scale but only as a brake (Mamba Δ).

Phase 1 arms are therefore motivated with defined target metrics:
- Arm success ≠ PPL only. Targets: (1) positive, control-surviving
  coupling of trajectory movement to surprisal; (2) a geometric GP
  effect at the critical word beyond surprisal's (standardized d on
  turn/extrap > surprisal's d); (3) no regression on the
  disproportionality test.
- Arm 1 (content-adaptive fold transport) now has a specific design
  prior: braking trains stably (Mamba); the open question is whether
  amplification can be trained at all. Both directions must be logged.
- Arm 2 (spinor homeostasis) target: excursion-then-return structure —
  turn/extrap at reinterpretation without permanent drift.
- Arm 3 (quotient path) target: any GP geometric separation at all.

## Caveats

- Mamba is 6x larger, differently tokenized and trained: yardstick only.
- OPERA/RoPE GP results rest on 12 cells; direction-change item-level
  numbers at that power are noise-level (OPERA +0.70 vs RoPE −0.55 on
  the same metric shows it).
- All correlations are per-token events; human RT aggregates spillover
  regions, which we match only coarsely (ROI 0/+1).
