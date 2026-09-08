# Trajectory Dynamics: Synthesis and Positioning

Date: 2026-09-08. Builds on `trajectory_phase0_verdict.md` (the
measurement) and `trajectory_phase1_arms.md` (the arms). This document
is the framing for the paper and the research direction — it contains
no new measurements.

## The diagnosis in one sentence

OPERA — like the matched RoPE transformer and like Mamba — integrates
language at near-constant speed regardless of content: the state
trajectory is **homeorhetic** (a stabilized *drift*, Kaneko's term for
what Waddington's landscape actually stabilizes), not **homeostatic**
(no attractor the state departs from and returns to). Human language
processing is neither: garden-path reanalysis is a large, content-
triggered excursion — in the forty-year-old vocabulary nobody in ML
has revived (Thom 1972; Petitot 1985; Wildgen 1982), a **catastrophe**:
the cusp, where two competing interpretations are two attractors and
disambiguation is a basin switch, with hysteresis (the lingering
misanalysis Christianson et al. 2001 measure in humans) as the
signature.

## What is now measured, not asserted

1. Constant-speed integration across three architectures
   (step p99.9/median 1.44–1.94; no surprisal coupling that survives
   controls) — `trajectory.py`, n=1,280 windows each.
2. The accelerator is empty everywhere: no model moves its state *more*
   on informative tokens (OPERA/RoPE move *less*, −0.76σ/−1.18σ;
   Mamba flat). The one adaptive-step mechanism at scale, Mamba's Δ,
   is a *brake* on the predictable (−0.70σ at bottom-5% surprisal),
   never an amplifier (top-5% ≈ average; Δ-surprisal r = +0.077). First
   per-token Δ-vs-surprisal measurement in language we are aware of.
3. Garden paths separate in the readout, not the geometry: surprisal
   reproduces the human construction ordering (Mamba r=+0.83); no
   trajectory metric tracks human per-item cost in any model
   (|r| ≤ 0.13 at full coverage) — `gp_benchmark.py` on SAP ClassicGP
   with N=2000 human SPR effect sizes.
4. Three native mechanisms (homeostatic anchors, quotient path,
   content-adaptive fold transport) at small budget: PPL ties, one arm
   (fold transport) is used by training and leans right on GP
   separation; two arms' gates never opened (cold-start, recorded
   honestly).

## Positioning (OPERA stays OPERA)

- **Versor (arXiv:2602.10195)** is the closest published relative and
  is already kernel-strategy-credited in `triton_kernel.py`. The
  distinction: Versor evolves rotors *sequentially* in Cl(4,1) with
  scale = physical object size; OPERA composes Spin(3) states *up a
  Fenwick tree* with scale = syntactic span. The trajectory-dynamics
  program (this document) has no counterpart in Versor. Nothing from
  Versor enters the architecture.
- **Mamba** is a baseline and a citation (its Δ is prior art for
  content-adaptive step size — as a brake). Nothing from Mamba enters
  the architecture either: fold_adapt is a rotor twist read from the
  block's own state, in OPERA's own algebra, extending the project's
  own level-adaptive fold-transport arm.
- **Tractor calculus / conformal machinery** (the user's pointer, Sep
  7): the calculus has no referent here (no base manifold, no
  connection problem); the *algebra* survives only as the observation
  that the compose node spans rotations+product but not the quotient —
  tested as Arm 3. No Cl(4,1) upgrade: CATr's lesson (bigger group ≠
  better) plus this repo's four falsified geometric constraints both
  argue against; the hyperbolic-LM literature is the graveyard warning.
  The Fenwick tree already is the scale sweep the "tractor sweep"
  intuition wanted.

## Threats (stated before reviewers find them)

- **Phantom Transitions (arXiv:2606.07559)**: apparent discontinuities
  can live entirely in the softmax readout (shown for training-time
  fine-tuning transitions, embeddings frozen). Our claim is about
  inference-time token trajectories; and our own GP result — surprisal
  separates, geometry doesn't — is exactly their split, at inference.
  The claim to defend is not "models show jumps" but "human processing
  has excursions current integration geometry cannot express; an
  architecture that can express them is a better model of language
  processing" — a modeling claim about human alignment, not a
  soft-max artifact.
- **Barenholtz et al. (arXiv:2606.05346)**: direction-change predicts
  human RT beyond surprisal in GPT-2/Pythia — we adopt their metric
  family (turn/extrap) and add the architecture comparison. Our
  small-scale models show the coupling only weakly/raw-confounded;
  their effect grows with scale, ours is measured at 5.9M–22M. Not a
  contradiction; a scale boundary to state.
- **Garden-path magnitude gap** (van Schijndel & Linzen 2021; Arehalli
  et al. 2022; Huang et al. 2024): surprisal underestimates human GP
  cost — our benchmark reproduces the *existence* of the gap at small
  scale (surprisal separates, but item-level tracking is null
  everywhere). 2026 follow-ups (parse-multiplicity, syntactic belief
  update) are active debate; cite as unsettled.

## Next moves, in order

1. fold_adapt at real budget (20k steps, then 22M rung if the GP
   spillover separation survives): the one arm training uses.
2. Warmer-gate retries for homeo/quotient are NOT scheduled — the
   cold-gate null is the answer at this budget; revisit only if
   fold_adapt shows the objective wants excursions, at which point an
   attractor to return *to* becomes motivated again.
3. BPE/larger-vocab OPERA variant would lift GP coverage from 25% to
   full and make NP/Z measurable — the single biggest measurement
   upgrade available.
4. The paper: catastrophe framing (cusp = disambiguation, hysteresis =
   lingering misanalysis), the homeorhesis diagnosis, the empty-
   accelerator table, and fold_adapt's lean — each labeled with its
   power level, as in the two companion docs.
