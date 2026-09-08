# OPERA optimizer survey — the 2025–26 LMO/spectral-descent family

**Date:** 2026-09-09. Web-researched at the user's direction; feeds
`docs/OPERA_Optimizer_prereg.md` (amendment logged there). Every claim
below is sourced; links at the end. Items already cited in
`OPERA_Improvement_Survey_2026-09.md` §4.2 (Moonshot Muon scaling,
Dion3, "How Much Orthogonalization", Hierarchical Muon, "Muon is Not
That Special") are not repeated except where new detail changes the
OPERA read.

---

## 1. The family map

| optimizer | one-line mechanism | OPERA relevance |
|---|---|---|
| **Muon** (Jordan; Moonlight 2502.16982 scaled it) | orthogonalize the momentum buffer with Newton–Schulz; RMS-matched step | incumbent; −3.75% BPB on the byte rung |
| **Scion** (Pethick et al., 2502.07529, ICML 2025) | norm-**constrained** spectral descent: Sign→Spectral→Sign pipeline; weight decay re-derived as a constraint, not a heuristic; spectral norm **provably stays bounded** at fixed stepsize | the constraint view fits `rot_free`: SO(3)-ness by norm control instead of parameterization |
| **Gluon** (2505.13416, ICML 2026) | unifies Muon/Scion under a **layer-wise (L₀,L₁)-smoothness** model with adaptive layer-wise steps; theory-predicted stepsizes *match* the values Pethick et al. tuned by hand | the closest theoretical license for what LO-Muon does — read "layer-wise" as "application-context-wise" (§2) |
| **MuonClip** (Kimi K2, 2507.20534) | Muon + QK-Clip (rescale attention Q/K to stop logit explosion); trained 1T params crash-free | rejected as machinery (OPERA has no QK — I2), adopted as philosophy: fix the exploding *mechanism*, not the optimizer |
| **NorMuon** (2510.05491) | after NS, **row-wise (per-neuron) second-moment normalization** — Muon updates have highly non-uniform neuron norms; ~2× token efficiency claimed | OPERA variant: the same correction at **quaternion-block granularity** (nb blocks, not rows) — a natural future dose |
| **Dion / Dion3** (2504.05295 / 2608.11612) | **amortized** orthonormalization replaces per-step NS; Dion3 full-stack 6× optimizer-step speedup | porting concern only at our rung; matters if LO ever runs per-level NS ×10 on GPU |
| **Gram Newton–Schulz / GramMuon** (Tri Dao, 2026) | run NS on the Gram matrix GᵀG of the smaller dimension; 40–50% off the orthogonalization step | the cheap-NS primitive LO-Muon's implementation should use if cost shows up |
| **Mousse** (2603.09697) | curvature-aware rectification of Muon's geometry ("Muon as spectral steepest descent" framing) | parked: second-order-ish machinery beyond the rung's need |
| **AdaGrad-Meets-Muon** (OpenReview 280iEseqgx) | adaptive stepsizes layered on orthogonal updates | the minimal sibling of LO: adapt only a **per-level scalar step**, not the direction — the dose-response probe if LO passes |

**Theory line** (no new optimizer, but it tells us what NS *is*):
spectral descent convergence for non-smooth objectives (2605.26977);
Muon's implicit bias → max-margin under the **spectral norm** on
separable data (NeurIPS 2025); Muon as Lion with a nuclear-norm
geometry (OPT-ML 2025). Bernstein's derivation essay ("Deriving Muon")
is the load-bearing item and gets its own section.

**Riemannian toolbox** (for prereg stage 2): geoopt's RiemannianAdam
with retractions (Stiefel, SO(3), sphere); "Generalizing Adam to
Manifolds" (2305.16901) does all Adam steps in a **global tangent
space** — exactly the quaternion-tangent-at-identity trick `rot_free`
would need; Projective Manifold Gradient Layer (ICLR 2020) is the
rotation-estimation-by-Riemannian-gradient precedent; QGOpt ships
Riemannian Adam over unitaries/rotations.

## 2. The load-bearing transplant: Muon's own derivation assumes one
input distribution per layer — OPERA's compose operator has ⌈log₂T⌉ of
them

Bernstein derives Muon as **steepest descent under the RMS→RMS
operator norm**: a weight matrix maps between RMS-normed spaces; the
update ΔW that minimizes loss subject to ‖ΔW‖_{RMS→RMS} ≤ η keeps the
gradient's singular vectors and discards its magnitudes:
ΔW = −η·√(fan-out/fan-in)·UVᵀ. Two assumptions are explicit:

1. **one input space** per layer ("if inputs have RMS norm ≤ 1…"), and
2. a **single** gradient per weight.

OPERA's compose operator violates both at once, and we have already
measured it: dL/dW = Σ_ℓ G_ℓ with level-1 inputs at ‖h‖ ≈ 10.68 and
level-≥2 inputs at ‖h‖ ≈ 6.93 (2026-09-07 measurement). Under
Bernstein's own output-stability bound ‖Δy‖_RMS ≤ ‖ΔW‖_op·‖x‖_RMS, a
level consuming 1.54× larger inputs should get a **1.54× smaller ΔW
budget**. Combining the per-level constrained problems and projecting
onto the shared ball gives the composite steepest-descent step:

    ΔW = −η · NS( Σ_ℓ  (x̄_ℓ^ref / x̄_ℓ) · NS(G_ℓ) )

with x̄_ℓ the running mean input RMS at level ℓ. That is LO-Muon with
a **derived** weighting — not "equal direction votes" (the v1 guess)
but *per-application-context steepest descent*, which is what
Bernstein's argument becomes when its assumption is repaired rather
than ignored. Gluon's layer-wise (L₀,L₁)-smoothness is the same
correction at the layer granularity, and its headline result —
theory-predicted stepsizes matching hand-tuned practice — is evidence
the smoothness constants carry real signal, i.e. that the measured
10.68/6.93 gap is worth honoring rather than averaging away.

**Stage-1 consequence (registered as an amendment):** two treatment
arms — `lo_uniform` (w_ℓ = 1, the original registration) and
`lo_derived` (w_ℓ ∝ 1/x̄_ℓ, Bernstein+Gluon-derived). One mechanism,
one dose each; `lo_derived` is predicted-primary by theory,
`lo_uniform` is the theory-free ablation. The §4.1 kill-switch
(inter-level principal-angle cosine ≥ 0.95) still runs first and
still binds both arms.

**Stage-2 consequence:** Scion shows the alternative to retraction —
*constrain* the norm rather than project onto the manifold.
For `rot_free`'s 3×3 blocks that means either (a) exact polar
retraction via a geoopt-style retraction (the registered arm), or
(b) a Scion-style spectral-norm constraint on the blocks (the
amendment adds it as the pre-named alternative, not an extra arm —
whichever is cheaper to implement runs).

## 3. What is rejected, on the record

| candidate | why out (for now) |
|---|---|
| MuonClip's QK-Clip | attention-specific; OPERA has no attention (I2). The divergence guard already covers the stability role. |
| Mousse curvature rectification | beyond the rung's compute story; revisit if LO passes and quality-cost shows up |
| Dion amortized orthonormalization | solves a GPU-throughput problem we do not have at 3k steps on MPS |
| NorMuon's row-wise post-normalization | adopted **in spirit** as a named follow-up at quaternion-block granularity; not an arm until LO-Muon itself is decided — the fishing guard |

## 4. Novelty check for LO-Muon itself (the §2-of-the-prereg duty)

Gluon is per-**layer**; every other family member is per-layer; multi-
task gradient-combination literature (GradNorm etc.) combines
magnitudes of *different* parameters, not applications of *one*
parameter at different scales. Nothing found that orthogonalizes
**per application context of a shared weight**. The combination
operator Σ NS(G_ℓ) (sum in whitened space) appears absent from the
literature. This section becomes an amendment, not a defence, if a
reviewer finds a counterexample.

## Sources

- Scion: [arXiv 2502.07529](https://arxiv.org/pdf/2502.07529) ·
  [LIONS-EPFL/scion](https://github.com/LIONS-EPFL/scion) ·
  [ICML 2025](https://icml.cc/virtual/2025/poster/46586)
- Gluon: [arXiv 2505.13416](https://arxiv.org/abs/2505.13416) ·
  [ICML 2026](https://icml.cc/virtual/2026/poster/64921)
- Muon derivation: [Deriving Muon — Bernstein](https://jeremybernste.in/writing/deriving-muon)
- Moonlight / Muon-is-Scalable: [arXiv 2502.16982](https://arxiv.org/html/2502.16982v1) ·
  [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)
- MuonClip / Kimi K2: [arXiv 2507.20534](https://arxiv.org/html/2507.20534v1)
- NorMuon: [arXiv 2510.05491](https://arxiv.org/abs/2510.05491)
- Dion: [arXiv 2504.05295](https://arxiv.org/html/2504.05295v3) ·
  Dion3: [arXiv 2608.11612](https://www.alphaxiv.org/abs/2608.11612)
- Gram Newton–Schulz: [Tri Dao blog](https://tridao.me/blog/2026/gram-newton-schulz/) ·
  [Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)
- Mousse: [arXiv 2603.09697](https://arxiv.org/html/2603.09697v2)
- Spectral-descent convergence: [arXiv 2605.26977](https://arxiv.org/html/2605.26977v1)
- Implicit bias: [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117324) ·
  Muon-as-Lion: [OPT-ML 2025](https://opt-ml.org/papers/2025/paper137.pdf) ·
  AdaGrad-Meets-Muon: [OpenReview](https://openreview.net/pdf?id=280iEseqgx)
- Riemannian toolbox: [geoopt](https://github.com/geoopt/geoopt) ·
  [Generalizing Adam to Manifolds](https://arxiv.org/html/2305.16901v4) ·
  [Projective Manifold Gradient Layer](https://openreview.net/pdf?id=rLARdZ3FxCM) ·
  [QGOpt](https://scipost.org/SciPostPhys.10.3.079/pdf)
- Stochastic spectral descent view: [Silveti-Falls slides](https://elliit.se/wp-content/uploads/2026/05/SilvetiFalls.pdf) ·
  [steepest-descent survey](https://leloykun.github.io/ponder/steepest-descent-non-riemannian/)
