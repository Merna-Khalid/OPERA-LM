# OPERA bistable fold — pre-registered protocol

**Version:** 1.0 — 2026-09-08
**Status:** PRE-REGISTERED. Written before any line of the mechanism was
implemented and before any run. Deviations go in the Amendments log.
**Hardware:** MacBook M4, 16 GB, MPS. No Kaggle quota.

---

## 1. The measured deficiency this is designed against

`docs/trajectory_phase2_nulls.md` (2026-09-08) established, on two OPERA
checkpoints (22M word/so3/L4 and byte d512/free/L2):

- Constant-speed integration is **not** forced by normalization: radius
  is free (CV 0.052–0.059), **41–58% of the geometrically available step
  range is unused**, and observed step variance *exceeds* the isotropic
  null by 1.75–1.76× in both models.
- **Step size is nearly decoupled from the input**: scrambling token
  order changes mean step by 1.8% (word) / 4.5% (bytes).

So the capacity to move is present and unused, and the missing thing is
a coupling from content to step size.

**Smooth gating has now failed to supply that coupling twice**, in two
architectures:

| mechanism | outcome |
|---|---|
| Mamba Δ (softplus), phase 0 | a *brake* on the predictable (−0.70σ at bottom-5% surprisal); never an amplifier |
| OPERA `fold_adapt` (sigmoid), phase 1 | training reached for it (w norm 2.06/0.98 from zero) yet trajectory statistics were **unchanged**: step r vs surprisal −0.259 vs incumbent −0.252; p99.9/median 1.43 vs 1.45 |

Both are smooth modulations of a smooth flow. The hypothesis is that
**smoothness is the limiting property**, and that a gate with two stable
states is required for a discrete commitment to appear in the geometry.

## 2. Prior art, and what is actually new

**Bistable Recurrent Cell** (Vecoven, Ernst & Drion, arXiv:2006.05252;
PLOS One 2021) is the closest published mechanism:

```
h_t = c_t·h_{t−1} + (1 − c_t)·tanh(a_t·h_{t−1} + U_x·x_t)
a_t = 1 + tanh(U_a·x_t + w_a⊙h_{t−1})   ∈ (0, 2)
```

Bistable exactly when `a_t > 1`: above 1 the feedback map acquires a
region of negative slope and two stable fixed points appear; at or below
1 it is monotonic and monostable. Two findings of theirs bind our design:

1. **Learned-constant bistability failed** — training `a` and `c` as
   plain SGD parameters "resulted in lack of representational power";
   *neuromodulation* (making them input-dependent) was required. Our `a`
   is therefore read from content, never a free constant.
2. It is **cellular/diagonal** and was evaluated only on toy memory
   tasks (copy-first-input, denoising, permuted MNIST). Never language,
   never scaled.

Other relevant lines, none of which implement this: fold bifurcations
underlying line attractors (Phys. Rev. E 96:052308); catastrophic
forgetting as saddle-node bifurcation (arXiv:2508.10765); dynamical-
systems parsing, where syntactic reanalysis is modelled as *"switching
of the control parameter, in analogy to phase transitions"* (beim
Graben et al.) — the right framing with no neural implementation;
catastrophe-theoretic semantics (Thom 1972; Petitot 1985; Wildgen 1982).
Modern LM gating is uniformly smooth: Gated Attention (arXiv:2505.06708,
NeurIPS 2025 Oral, sigmoid, validated to 15B MoE / 3.5T tokens), Mamba
Δ (softplus), `fold_adapt` (sigmoid).

**Novel here:** bistable gating in a modern sequence model, placed in a
hierarchical composition operator rather than a diagonal recurrent cell,
designed against a *measured* trajectory deficiency, and evaluated
against human garden-path data.

## 3. Mechanism

In the left fold's accumulator loop (`model.py`, `fold_mode == 'left'`),
replacing `acc ← composed`:

```
a = A(acc, nxt)                                  # per block, [N, nb]
c = sigmoid(W_c·[acc; nxt] + b_c)                # per block, [N, nb]
acc ← (1 − c)·composed + c·tanh(a ⊙ acc)
```

with `composed = node(acc, nxt)` exactly as now. The gain is applied to
the accumulator *inside* a `tanh`, so the positive feedback `a > 1`
cannot diverge — the bound is structural, not a clamp.

Placement rationale: `composed` is post-`comp_norm`, so a gain applied
*before* the node would be renormalized away. Applying it in the fold's
accumulator update is the only place in this path where the gain
survives.

Invariants: I1 ✓ (`a`, `c` read content, never position) · I2 ✓ (gates,
no scores, no softmax over positions) · I3 ✓ (the compose node is
untouched) · I4 ✓ (the fold still consumes exactly the prefix's Fenwick
blocks; causality unchanged).

### 3.1 Scope note — this is commitment, not temporal hysteresis

OPERA's fold is **per-prefix**: every position's accumulator is rebuilt
from that prefix's own blocks, so there is no accumulator persisting
across tokens. Bistability here is therefore a discrete commitment
*during composition*, not hysteresis over time.

Discrete change across positions still follows, by a different route:
adjacent prefixes share most of their Fenwick blocks, so a committed
block propagates to every prefix that reads it, and a change in the
decomposition flips the inherited commitment — which appears in the
trajectory as a jump. This is the mechanism by which the arm is expected
to move the step-size distribution, and it is stated in advance so the
claim is not retrofitted.

## 4. Arms — bistability is isolated by a matched control

| arm | `fold_bistable` | gain range | bistable reachable |
|---|---|---|---|
| incumbent | `'off'` | — | — |
| **control** | `'mono'` | `a = sigmoid(z) ∈ (0,1)` | **no** |
| **treatment** | `'bi'` | `a = 1 + tanh(z) ∈ (0,2)` (BRC) | **yes, iff a>1** |

`mono` and `bi` have **identical parameter count, identical
initialization, identical capacity** and differ only in whether the gain
can exceed 1. The contrast `bi − mono` therefore isolates *bistability*,
not "a new module." This is the control phase 1 lacked.

### 4.1 Gates start WARM — the correction to phase 1

Phase 1's `homeo` and `quotient` arms both nulled with gates that never
opened (bias −6 → 0.00247, unmoved after 3000 steps). A multiplicative
sigmoid gate at 0.0025 has its gradient suppressed by `g(1−g) ≈ 400×`,
so a cold gate is self-fulfilling: it gets no gradient *because* it is
cold. That is a property of the initialization, not a verdict on the
mechanism.

Both arms here initialize `b_c = 0` (c ≈ 0.5) and `z ≈ 0` (`a ≈ 1` in
`bi`, `a ≈ 0.5` in `mono`). **Flags-off is therefore NOT bitwise the
incumbent**, deliberately. The incumbent is recovered by the `'off'`
arm, and `mono` is the capacity-matched control. This is a considered
departure from the house rule in `OPERA_Mechanisms_Guide.md` §7, made
because the rule produced two uninformative nulls, and it is recorded as
a departure rather than a silent change.

## 5. Hypotheses and gates

Rung: byte d512/nb128/L2 (the phase-1r representation winner), 3000
steps, batch 8, T=1024, seed 42 — identical to the `bytes_d512` run so
the incumbent number already exists (BPB 2.1317).

**H1 (the falsifier, pre-committed in phase 2) — primary.**
The bistable arm raises the step-size tail.
*PASS* if `p99.9/median (bi) ≥ 1.60` AND `> mono` by ≥ 0.10.
Incumbent/`fold_adapt` sit at 1.43–1.45; the isotropic null is ~1.10.
**FAIL closes the mechanism class** — if a gate with two stable states
also leaves the distribution at ~1.4, smoothness was not the limiting
property and the bistability line is retired.

**H2 (content coupling).** Bistability couples step size to content.
*PASS* if `|partial r(step, surprisal | emb norm)|` for `bi` exceeds
both incumbent (−0.095 at 22M) and `mono` by ≥ 0.10 in magnitude.

**H3 (the gain is actually used).** Fraction of blocks with `a > 1` at
convergence, and its correlation with surprisal. **No gate — this is the
arm reporting its own usage**, the convention that made phase 1's nulls
interpretable. `bi` with `a ≤ 1` everywhere is a null by cold parameters
and must be reported as such rather than as a mechanism failure.

**H4 (quality guard).** BPB must not regress > 5% vs `bytes_d512`'s
2.1317. A tie with H1 passing is a success: the trajectory instrument
measures something BPB cannot see.

**H5 (garden path, exploratory).** Standardized GP effect on
`turn`/`extrap` at spillover+1, 72 cells, paired per-item vs `mono`.
Reported, no gate — every arm at this budget has shown near-null GP
effects because these runs see ~4× less text than the phase-1 rung.

## 6. Pre-stated interpretation

- **H1 pass + H3 shows `a>1` used** → the first mechanism in this project
  to move the trajectory's shape; bistability is the lever and the
  catastrophe framing earns its place in the paper.
- **H1 fail with `a>1` used** → bistability was available, trained, and
  did not produce excursions. The mechanism class is wrong. This is the
  cleanest possible negative result and closes the line.
- **H1 fail with `a≤1` everywhere** → the objective declined
  bistability. Not the same as the mechanism failing; report as a
  training/objective finding and note the LM loss appears to reward
  smooth integration.

## 7. Risks acknowledged at registration

`a > 1` is positive feedback; the `tanh` bounds it structurally but
training instability is the main hazard and the divergence guard stays
on with skip counts reported. BRC never scaled past toy tasks and there
may be an unpublished reason. Single seed. One rung. The fold is
per-prefix, so this tests commitment-in-composition, not temporal
hysteresis (§3.1). Warm-gate init means `bi`/`mono` are not bitwise the
incumbent, so `off` must be run in the same session rather than quoted
from the earlier table.

---

## Outcomes — 2026-09-08

| arm | params | BPB | step median | **p99.9/med** | partial r | frac(a>1) |
|---|---|---|---|---|---|---|
| bist_off | 3,293,447 | 2.1308 | 19.420 | 1.200 | −0.2249 | — |
| bist_mono | 3,818,247 | 2.1264 | 19.480 | 1.194 | −0.2052 | 0.000 |
| **bist_bi** | 3,818,247 | **2.1228** | 18.554 | **1.217** | −0.2443 | **0.660** |

Control validity: `bist_off` reproduces the independent `bytes_d512` run
to ΔBPB **0.00088** (2.1308 vs 2.1317), which is also the first
same-config reproducibility floor this project has measured.

### H1 — **FAIL.** The tail did not appear.

`bi` p99.9/median = **1.217**, against a pre-registered threshold of
**≥ 1.60**; `bi − mono` = **+0.023** against **≥ 0.10**. For scale, the
incumbent and `fold_adapt` sit at 1.43–1.45 and the isotropic null is
~1.10. Bistability moved the tail by about a fortieth of what was
required.

### H2 — **FAIL.** `bi` has the largest |partial r| (0.2443 vs mono
0.2052, off 0.2249) but by 0.02–0.04, not the required 0.10. And the
sign is **negative** in every arm: more surprising tokens still move the
state *less*.

### H3 — bistability was **heavily used**. No gate; this is the number
that decides *which* negative result this is.

Measured on real text (6.3M block updates, layer 0):

| statistic | value |
|---|---|
| gain a: mean | 1.270 |
| gain a: **median** | **1.646** |
| gain a: p90 / max | 1.993 / 2.000 |
| frac(a > 1.0) | **0.660** |
| frac(a > 1.5) | 0.557 |

Median gain 1.646 implies attractors at ≈ ±0.89 — deep bistability, not
jitter around the critical point. The objective reached for the
mechanism hard and immediately.

### H4 — **PASS**, and better than that: `bi` has the **best BPB of the
three** (2.1228 < 2.1264 < 2.1308). Bistability slightly *helped*
language modelling.

### Verdict (pre-stated in §6, applied)

**H1 fail with `a>1` used → the mechanism class is wrong. The line
closes.** Bistability was available, was trained, was used on two thirds
of all block updates at a median gain of 1.65, improved perplexity — and
produced no trajectory excursions whatsoever.

This is the clean negative §6 asked for, and it is a *stronger* negative
than a null would have been: the mechanism worked as designed and the
designed effect did not follow.

### Why it failed — mechanistic reading (post-hoc, not pre-registered)

`tanh` saturates. With median gain 1.65 and post-`comp_norm` components
of magnitude ~1, `tanh(1.65·x) ≈ ±0.93`: the bistable term collapses to
a **per-component sign function**. The state commits to a corner of a
hypercube — commitment is real and is happening.

But the commitments are **independent across the nb=128 blocks**, each
with its own gate. In high dimension, independent per-component sign
flips do not produce large displacements: the corners of a hypercube are
all at comparable distance, so switching corners moves the state by
about the same amount as any other update. Independent commitment
averages out in the norm. The step median in fact *fell* (18.554 vs
19.42/19.48), consistent with saturation compressing the state.

**An excursion requires correlated commitment** — many blocks flipping
together, i.e. a low-rank or shared switch rather than nb independent
ones. That is a different mechanism, and this result does not test it.
Recorded as a hypothesis, explicitly not as a rescue of the current one:
the pre-registered line is closed and any shared-switch arm needs its
own registration and its own gate.

### Method note — a measurement bug that would have inverted the verdict

H3 was first computed by calling `bistable_stats` on **Gaussian noise**,
which reported a_mean 1.001 and frac(a>1) = 0.50 — indistinguishable
from a gain sitting on the bifurcation point, i.e. "the objective
declined bistability." On real text the same model gives a_mean 1.270,
median 1.646, frac 0.660. The gain is content-dependent, so noise
measures what it does to noise. Caught before the outcome was written;
`experiments/bistable_gates.py` now captures gains from the live fold on
held-out sequences. Reporting the noise number would have produced the
opposite conclusion about *why* the arm failed.

## Amendments log (append-only)

**2026-09-08 — H3 measured on real text, not Gaussian noise** (see
Method note above). No hypothesis or threshold changed; only the
estimator for the H3 usage statistic, which carries no gate.

---

# ADDENDUM — pre-registered follow-up: low-rank (correlated) gain

**Registered 2026-09-08, after the per-block outcome above and before
the follow-up ran.** This is a **new hypothesis with its own gate**, not
a re-opening of the falsified arm. The per-block result stands: bistable
gains were used heavily and produced no excursions. What is tested here
is the *stated reason* for that failure.

## A1. Hypothesis

The per-block post-mortem argued that nb=128 **independent**
commitments cannot displace the state, because independent sign flips in
high dimension average out in the norm. If that is right, forcing the
gains to move **together** should produce the excursions the per-block
arm did not.

Independent support: metastable-attractor models (Recanatesi et al.,
*Neuron* 2021) generate lingering-then-abrupt dynamics by coupling a
high-dimensional state to a **low-dimensional** modulator — the same
correlation this arm imposes, arrived at from neuroscience rather than
from our post-mortem.

Counter-evidence, recorded up front: the SSM literature holds that
**per-channel gating is more expressive** than shared gating, and treats
shared gates as a compute concession. This arm therefore trades
expressivity for correlation and may lose on quality; H4 guards that.

## A2. Mechanism

The gain head becomes a rank-r bottleneck: `Linear(2d → r) →
Linear(r → nb)`. At r=1 a single scalar drives all nb gains through one
fixed profile — maximally correlated commitment. Everything else
(retention gate `c`, the `tanh` bound, the update law, warm init, the
gain sitting exactly on a=1 at start) is unchanged.

LoRA init convention: the down projection keeps its random init, only
the up projection is zeroed. Zeroing both would leave the down factor
with exactly zero gradient forever — the cold-start failure that made
phase 1's nulls uninterpretable. Verified: up-projection gradient is
non-zero at step 0.

**Arms:** `bist_bi_r1` (r=1) and `bist_bi_r4` (r=4), against the
existing `bist_bi` (per-block) and `bist_mono` (control) at the same
rung and seed.

## A3. Gates

**A-H1 (primary).** `p99.9/median ≥ 1.60` and `≥ bist_bi + 0.10`.
Same threshold as H1 — the bar does not move because the mechanism
changed. Per-block reached 1.217.

**A-H2.** frac(a>1) reported on real text, as before. An arm that
declines bistability cannot test correlation.

**A-H3 (quality guard).** BPB within 5% of `bist_bi`'s 2.1228.

## A4. Confound, stated in advance

Low-rank has **fewer** parameters (66,249 / 67,113 vs 70,057 at the
selftest rung; the same ordering holds at d=512). The asymmetry that
follows is deliberate and must be honored in the write-up:

- **A pass is unconfounded** — more effect from fewer parameters cannot
  be a capacity artifact.
- **A fail is partially confounded** — it could be correlation failing,
  or the bottleneck simply having too little capacity to express a
  useful gain at all. A-H2 partially separates these: an arm that still
  drives a>1 heavily has not been starved of the ability to use the
  mechanism.

## A5. Pre-stated interpretation

- **A-H1 pass** → correlated commitment is the missing ingredient; the
  metastable two-timescale design (slow high-tree-level modulator over
  fast low-level composition) becomes motivated and earns its own
  registration.
- **A-H1 fail with a>1 used** → correlation is *not* the missing
  ingredient either. The post-mortem explanation of the per-block
  failure is then itself falsified, and bistable commitment — correlated
  or not — does not move this architecture's trajectory. That closes the
  broader line, not just one parameterization.

## Outcomes — 2026-09-08

| arm | rank | BPB | step med | **p99.9/med** | vs bi | partial r | frac(a>1) |
|---|---|---|---|---|---|---|---|
| bist_off | — | 2.1308 | 19.420 | 1.200 | −0.017 | −0.2249 | — |
| bist_mono | — | 2.1264 | 19.480 | 1.194 | −0.023 | −0.2052 | 0.000 |
| bist_bi | 0 | 2.1228 | 18.554 | 1.217 | — | −0.2443 | 0.660 |
| bist_bi_r1 | 1 | 2.1362 | 19.809 | **1.174** | **−0.043** | −0.1269 | **0.827** |
| bist_bi_r4 | 4 | 2.1313 | 19.495 | **1.184** | **−0.033** | −0.1517 | **0.863** |

### A-H1 — **FAIL at every rank**, and in the wrong direction

Required ≥ 1.60 and ≥ `bi` + 0.10. Measured **1.174** (r=1) and **1.184**
(r=4) — both *below* the per-block arm. Correlating the gains made the
tail **smaller**.

### A-H2 — not a cold-parameter null. The opposite.

The rank arms drove the gain past 1 on **83%** (r=1) and **86%** (r=4) of
block updates, versus 66% for per-block. Constraining the gains made the
objective use bistability **more**, not less. There is no reading of
this where the mechanism was unavailable or untrained.

### A-H3 — quality guard passes (both within 5% of 2.1228), though
low-rank is *worse* on BPB than per-block (2.1362 / 2.1313 vs 2.1228),
consistent with the SSM literature's position that per-channel gating is
more expressive and that shared gating is a compute concession.

### Verdict — the post-mortem is falsified, and the broader line closes

Per §A5's pre-stated interpretation: **A-H1 fail with a>1 used means the
explanation offered for the per-block failure is itself wrong.** The
claim was that nb=128 independent commitments average out in the norm
and that correlating them would produce excursions. Correlating them
produced *less* tail, more usage, and weaker content coupling
(|partial r| 0.2443 → 0.1269).

So bistable commitment does not move this architecture's trajectory in
**any** parameterization tested — independent or correlated, shallow or
deep, at three ranks. That closes the line as a whole, not one
variant. The metastable two-timescale design that A5 said would "become
motivated" by a pass is **not** motivated and is not being built.

### What the accumulated negatives now say (post-hoc, not registered)

Four mechanisms have now failed to move the step distribution:
`fold_adapt` (sigmoid), Mamba's Δ (softplus), bistable per-block, and
bistable low-rank. They differ in smoothness, in granularity, and in
whether they admit two stable states — and they share exactly one
structural property: **every one is a gate on an interpolation.** The
update is always

    acc <- (1 - c) * X + c * Y

which places the new state strictly *between* X and Y. A gate can choose
where in that interval to land; it can never overshoot past either
endpoint. Meanwhile phase 2 measured 41–58% of the geometrically
available step range going unused.

The next hypothesis worth registering is therefore not another gate but
an **extrapolating** update — one that can overshoot the composed value
rather than interpolate toward it (`acc + γ(composed − acc)` with γ > 1,
or a reflection). That is a different class of operation and this study
says nothing about it. It needs its own registration, its own gate, and
its own control; it is recorded here only so the reasoning is on the
record, not as a licence to keep going.
