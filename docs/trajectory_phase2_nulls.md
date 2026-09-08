# Phase-2: Null models for constant-speed integration

Date: 2026-09-08. Instrument: `opera_lm/step_nulls.py`. Checkpoints:
OPERA 22M word-level (d=640, nb=160, L=4, so3 — the phase-0 model) and
OPERA byte-level (d=512, nb=128, L=2, free — the phase-1r winner).
JSONs: `runs_reprs/step_nulls_{word22M,bytes_d512}.json`.

## Why this exists

Phase 0 established constant-speed integration across three
architectures and called it "the accelerator is empty." That
measurement had **no null model**, and every architecture involved
normalizes its state. Since

    step_t^2 = R_t^2 + R_{t-1}^2 - 2 R_t R_{t-1} cos(theta_t)

step size is a deterministic function of radius and turn angle and is
bounded by R_t + R_{t-1}. If normalization pins R, "constant speed"
could be a statement about the norm layer rather than about the model's
integration policy. Designing a mechanism before settling that risked
building against an artifact.

Identity verified to 7.6e-06; the isotropic sampler reproduces
SD = 1/sqrt(d) to three decimals (0.0394 measured vs 0.0395 expected at
d=640).

## Results

| quantity | word 22M | bytes d512 |
|---|---|---|
| radius CV | 0.059 (**free**) | 0.052 (**free**) |
| step spread: angle / radius | 74% / 26% | 71% / 29% |
| observed turn angle | 49.1° ± 7.2 | 70.3° ± 5.5 |
| isotropic turn angle | 90.0° ± 2.2 | 90.0° ± 2.5 |
| observed step SD ÷ isotropic step SD | **1.76×** | **1.75×** |
| shuffled-token mean step vs real | −1.8% | −4.5% |
| **headroom** (step / (R_a+R_b)) | **41.5%** | **57.6%** |

## Findings

1. **Constant speed is NOT forced by normalization.** Radius is not
   pinned (CV ≈ 0.05–0.06), 41–58% of the geometrically available step
   range is unused, and observed step variance *exceeds* the isotropic
   null by 1.75–1.76× in both models. There is no cage. The two
   architectures, two representations, two rot_modes and two depths
   agree to within 0.01 on the isotropic ratio, which is a stronger
   agreement than we expected.

2. **The trajectory is a strongly correlated walk, not a random one.**
   Successive states sit 49° (word) / 70° (bytes) apart where random
   directions would give 90°. The model moves *less far* than chance
   while being *more variable* than chance — it is following a
   direction, not diffusing.

3. **The step-size distribution is almost input-independent.**
   Scrambling the token order changes mean step by **1.8%** (word) and
   **4.5%** (bytes). Turn-angle distributions are likewise nearly
   unchanged (49.1° → 48.7°; 70.3° → 67.9°).

## The diagnosis, restated

Phase 0: *"no model moves its state more on informative tokens."*

Phase 2 sharpens this. It is not that the accelerator is empty — the
pedal has 42–58% of travel left and the engine is not governed. It is
that **step size is nearly decoupled from the input**: destroying all
linguistic structure moves it by a few percent. Step size is close to a
constant of the architecture rather than a variable of the computation.

*Scope, precisely.* This is a statement about the **marginal
distribution** of step sizes, not about trajectories. Two very different
trajectories can share a step-size distribution, and the shuffled
control does not show that scrambled and real text follow the same path
— only that the *lengths of the increments* are drawn from nearly the
same distribution. That is exactly the quantity phase 0 correlated
against surprisal, so the two results are consistent and this one
explains the other: the correlation was near zero because the quantity
is nearly constant.

## Consequence for mechanism design

Because flatness is **not** geometrically forced, a new mechanism does
*not* need to break norm conservation, and does not need a
lower-dimensional bottleneck. The capacity to move is already there and
unused. What is missing is a coupling from content to step size.

Smooth content-adaptive gating has now been tried twice and does not
supply it:

- **Mamba's Δ** (phase 0): trains stably at scale, but only as a
  *brake* on the predictable (−0.70σ at bottom-5% surprisal); never an
  amplifier (top-5% ≈ average).
- **OPERA's `fold_adapt`** (phase 1): the arm training actually reached
  for (w norm 2.06/0.98 from zero), yet its trajectory statistics were
  **unchanged** — step r vs surprisal −0.259 vs incumbent −0.252,
  p99.9/median 1.43 vs 1.45.

Both are smooth (sigmoid/softplus) modulations of an otherwise smooth
flow, and both leave the trajectory's shape alone. That is the argument
for **bistability** specifically rather than more adaptivity: a smooth
gate has no reason to switch, which is also the most plausible reading
of the phase-1 cold-gate nulls (`homeo`, `quotient` — neither opened).
The cusp formulation — two interpretations as two attractors,
disambiguation as a basin switch, lingering misanalysis as hysteresis —
is the mechanism class that has *not* been tried, and it is the one the
catastrophe-theoretic semantics literature (Thom; Petitot 1985;
Wildgen 1982) has been describing for forty years.

## What would falsify the design before it is built

The prediction is that a bistable fold gate changes the trajectory's
step-size distribution (raises p99.9/median, introduces a tail) where
smooth gating did not. If a bistable gate also leaves the distribution
at ~1.4, the deficiency is not in the gate's smoothness and the
mechanism class is wrong.

## Caveats

Single seed per checkpoint. Two OPERA configurations only — the RoPE
and Mamba checkpoints have **not** been run through this instrument, so
"architecture-independent" is inherited from phase 0 and not yet
established for the null-model result. Sequence lengths 128 (word) and
256 (bytes) rather than full context. The shuffled control preserves the
token *distribution* of each sequence; a cross-sequence control (tokens
drawn from other documents) would be a stronger test.
