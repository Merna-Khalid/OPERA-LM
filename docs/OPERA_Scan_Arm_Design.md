# OPERA-Scan: Associative-Scan Arm — Design Note (v8.6, as implemented)
**Status:** IMPLEMENTED and selftested (`--fold scan`, 2026-07-21). Supersedes the
v1 design sections where marked. Motivation (profiling, 2026-07-20): the Fenwick
fold is kernel-launch-bound (46k op dispatches/step, 14x the transformer; fold =
62% of forward) and its O(T log T) per-position refolding is redundant.

## 1. From Fenwick fold to scan

`prefix_states` computed, for every position j, a left fold over j's <= log2(T)
Fenwick blocks — redundant shared sub-compositions, 28 serial steps/forward.
A parallel prefix scan computes ALL prefix states in O(log T) depth with no
per-position refolding. Requirement: an ASSOCIATIVE compose operator. The old
node (fusion gate -> LayerNorm -> tanh) is not associative, so per S4 -> Mamba:
associative operator inside the scan, learned nonlinearity in a per-position
readout. Martin & Cundy (arXiv:1709.04057) scanned the scalar version of the
same law; OPERA-Scan promotes it to a non-commutative group.

## 2. The in-scan operator (v8.6: conformal, with decay)

Per block, state is an affine pair (q, b): q a HOMOGENEOUS quaternion
(unit direction x learned magnitude s in (0,1)), b a paravector:

    (q1, b1) ⊕ (q2, b2) = (q1 ⊗ q2,  q1 ▷ b2 + b1)
    q ▷ (s, v) = |q| · (s, R(q/|q|) v)        # conformal: scale AND rotate

Prefixes expand as  B_j = b_j + q_j▷b_{j-1} + (q_j q_{j-1})▷b_{j-2} + ...
i.e. exactly  h_t = a_t·h_{t-1} + x_t  — the RG-LRU/GLA forgetting mechanic
riding inside the associative semidirect law (|q1⊗q2| = |q1||q2| exactly, so
associativity holds; selftest 9.5e-6). Leaf magnitude s_i = exp(-softplus(w·h_i)),
LRU-style (arXiv:2303.06349), init ~0.95.

**Why decay is mandatory (literature):** GORU (arXiv:1706.02761) showed pure
orthogonal transitions cannot forget; S4 -> Mamba's quality jump was
input-dependent decay; GLA/RG-LRU/GDN/RWKV-7 all put data-dependent
multiplicative decay INSIDE the scan. v8.5 (unit q, additive b) had every token
contributing to every prefix forever — "weaker than RetNet". v8.6 fixes this
and simultaneously makes long-T overflow impossible (|q| <= 1 by construction).

**Operand order matters:** the scan must expand with the later leaf on the left
(prefix_j = l_j ⊕ ... ⊕ l_0). The original order attenuated NEW content
(anti-forgetting); caught by the decay-direction selftest, fixed by flipping
to the opposite semigroup (still exactly associative, still causal).

## 3. Readout (per position, once per layer)

Normalize q (rotation is scale-invariant), learned gate vs identity baseline,
LayerNorm, tanh+0.1x, then the existing cross_mlp + blend gate. b is deliberately
NOT divided by accumulated |q| (that would undo decay); LN handles residual scale.

## 4. Implementation reality (deviations from v1 design)

- **Hillis-Steele scan, not Blelloch.** Faithful Blelloch measured 75,960
  dispatches/step (1.8x WORSE than the old fold): the down-sweep doubles level
  count and adds scatter/gather traffic on a launch-bound backend. Hillis-Steele
  has half the levels, no index tensors, no padding; O(T log T) element work is
  bandwidth MPS has. Same math up to float reassociation. (For a future fused
  Metal/CUDA kernel, Blelloch or CUB-style chunked scan returns — work-efficient
  wins when the GPU is saturated; chunking is required anyway by the 32KB
  Metal threadgroup limit. Also: the textbook down-sweep compose order assumes
  commutativity — wrong for us.)
- **Rodrigues-form rotation** in affine_compose (~15 aten ops vs ~60 for the
  double-quat_mul sandwich).
- torch's native `associative_scan` is no help on MPS (codegen is CUDA/XPU-only;
  generic fallback is the same algorithm; its autograd is O(T^2) memory).
  MLX `mx.fast.metal_kernel` + custom VJP is the proven Metal path
  (mlx-recurrence: 19-31x on this workload class) when we fuse.

## 5. Measured status (selftests ALL PASS)

- Associativity 9.5e-6; scan==naive 9.3e-6; causality 0.0; drift @T=4096 bounded
  (max |q|=0.986 by construction); decay-direction exact (x0.05 attenuation).
- Params @22M rung: 21,838,484 vs left-fold 22,254,488 (-1.87%).
- Op dispatches/step (fwd+bwd+opt, T=256, L=4): 24,148 vs left 42,230 = 0.572x.
- Known knob: init decay ~0.95 => ~14-token half-life; raise scan_wm bias from
  -3.0 if early training wants longer memory.

## 6. Are we still OPERA? (identity, post-scan)

- **O(peradic):** MORE true — v8.5's gated node enforced no axioms; v8.6 is a
  genuine algebra over the planar associative operad, selftested.
- **P(lanar):** intact and load-bearing — Hamilton product is non-commutative;
  order is how the model knows sequence direction.
- **E(quivariant):** unchanged partial status — SO(3) rotor transport is exact;
  readout LN/tanh break strict equivariance (as in the shipped model before).
  Paper phrasing: "geometrically constrained", not "equivariant".
- **R(ecursive):** the letter that changed — tree recursion became list
  recursion (h_t = a_t h_{t-1} + x_t); the tree survives as the parallel
  evaluation schedule, not as explicit forward-pass structure.
Thesis shift: "language is a parse tree composed geometrically" -> "language is
a sequence of geometric state transitions composed associatively". The old
`--fold left` arm remains in-tree; the paper narrative is that the tree arm
DISCOVERED the constraints (associativity for parallelization, decay for
forgetting) that the scan arm resolves.

## 7. Theory to disclose (paper honesty)

- Fixed-state scans are TC0-bounded (Illusion of State, arXiv:2404.08819):
  no exact state tracking at arbitrary length; no associative recall without a
  retrieval path (Wen et al., arXiv:2402.18510); Omega(sqrt L) width vs
  attention on reasoning (arXiv:2402.13934); copying/MQAR failures
  (Jelassi arXiv:2402.01032; Zoology arXiv:2312.04927).
- The SO(3) defense (cite vs "why not diagonal decay"): parity needs negative
  eigenvalues (arXiv:2411.12537); mod-3 counting needs non-triangular
  transitions; Householder products improve state tracking AND extrapolation
  (DeltaProduct, arXiv:2502.10297); diagonal selective SSMs length-generalize
  only on commutative automata (Terzic, arXiv:2412.19350) — quaternion compose
  is non-commutative by construction.
- Length-generalization claims must be narrowed to state-tracking/dynamics,
  not retrieval (Same Task More Tokens, arXiv:2402.14848).
- Position story: NoPE beats RoPE on length generalization (arXiv:2305.19466);
  RoPE's decay is not automatic (Barbero et al., arXiv:2410.06205) — ablate
  rotation vs free blocks in-scan (the `--rot free` analogue).

## 8. Prior art to cite and differentiate

Closest: RotRNN (arXiv:2407.07239 — scanned block-rotation transitions; we add
the affine carry + language modeling + operadic framing), LinOSS
(arXiv:2410.03943 — oscillatory SSM), RG-LRU/Griffin (arXiv:2402.19427 — our
gating template), GLA (arXiv:2312.06635), Martin & Cundy (arXiv:1709.04057 —
our law with scalars), quaternion RNNs (arXiv:1806.04418 — quaternion WEIGHTS,
not state transitions). Unclaimed-looking: an associative scan over quaternion
semidirect (rotation + conformal carry) pairs as an LM prefix mechanism.
Also must position against: Log-Linear Attention (arXiv:2506.04761 — keeps a
Fenwick-tree memory deliberately; our abandoned fold has a published champion),
and ParaRNN (arXiv:2510.21450 — Newton-parallelized nonlinear RNNs; the
evidence-backed Plan B if scan quality regresses).

## 9. The top-down arm ("magnifying glass", researched 2026-07-21)

Weakness to fix: fixed-state scans provably can't do associative recall.
Critical literature finding: additive/gated top-down flow NEVER fixed retrieval;
content-addressable paths did (Griffin, Based, Hymba meta tokens, Block-State
Transformer, CDSA). Ranked candidates:
- **Rank 2 (build first, cheap):** FiLM-style gated modulation of the readout
  from the scan prefix state — causal by construction, O(T·d). Biology:
  neuromodulatory gain (Servan-Schreiber 1990), apical amplification
  (Larkum 2013; Suzuki & Larkum 2020), amygdala salience (McGaugh 2004).
  Expect PPL gains, NOT recall closure. Also serves as ablation for Rank 1.
- **Rank 1 (the recall candidate):** prefix-causal latent bottleneck
  cross-attention (Perceiver-AR / BST style): K latents computed from scan
  states, positions query them. O(T·K), stays linear; latents MUST be
  prefix/chunk-causal (no Longformer-style symmetric global attention).
  Biology: Global Workspace broadcast (Baars; Dehaene & Changeux 2011;
  Goyal et al. arXiv:2103.01197; VanRullen & Kanai arXiv:2012.10390),
  pulvinar relay (Saalmann 2012). Evaluate with an MQAR-style probe.
- **Rank 3 (future/novel):** inside-outside second sweep over the scan's tree
  (IORNN arXiv:1611.06788; DIORA semantics on ONE tree). Causality forces an
  asymmetric "prefix-outside" variant — derive before building.
- **Rank 0 (control):** sliding-window attention hybrid (Griffin/Based) — the
  bar any top-down arm must justify itself against.


Key references: Blelloch CMU-CS-90-190; Martin & Cundy arXiv:1709.04057;
LRU arXiv:2303.06349; RG-LRU arXiv:2402.19427; GLA arXiv:2312.06635;
RotRNN arXiv:2407.07239; LinOSS arXiv:2410.03943; S5 arXiv:2208.04933;
Mamba arXiv:2312.00752; Mamba-2/SSD arXiv:2405.21060; DeltaProduct
arXiv:2502.10297; negative-eigenvalues arXiv:2411.12537; Terzic
arXiv:2412.19350; Illusion of State arXiv:2404.08819; Wen et al.
arXiv:2402.18510; Zoology arXiv:2312.04927; Based arXiv:2402.18668;
ParaRNN arXiv:2510.21450; Log-Linear Attention arXiv:2506.04761;
Sequential-Parallel Duality arXiv:2506.10918; Rao & Ballard 1999;
Bastos et al. 2012; Larkum 2013; Suzuki & Larkum 2020; McGaugh 2004;
Goyal et al. arXiv:2103.01197; Zhou et al. arXiv:1812.07035 (quaternion
double-cover continuity caveat for learned rotation outputs).
