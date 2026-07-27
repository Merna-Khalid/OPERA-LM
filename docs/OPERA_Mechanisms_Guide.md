# OPERA Mechanisms Guide

**The arms, the geometry, and the optimization knobs — what each mechanism
is, why it exists, and what happened to it in the research record.**
Companion to the README's configuration reference. Status labels:
**incumbent** (carries the headline results), **validated** (measured
win/parity), **falsified** (pre-registered, nulled, retired with the
evidence recorded), **exploratory** (selftested, not yet run at scale).

---

## 1. The state space

A token state is a vector in R^d organized as `nb` blocks of 4
(`d = 4·nb`). Each block is an element of the even subalgebra of Cl(3) —
isomorphic to the quaternions: one scalar channel s and one 3-vector
channel v. SO(3) acts block-diagonally, so each 3-vector rotates
independently. This is the "spinor" in the model's name: the state is a
collection of small rotors, not a flat vector.

## 2. The compose node

One internal node maps two children (h_L, h_R) to a parent, per block:

1. **Rotate:** the children's vector parts pass through per-block 3×3
   maps R_L, R_R (`rot_mode='so3'`: quaternion-parameterized rotations;
   `'free'`: unconstrained matrices — the paper's default after the
   rotation constraint was pre-registered and nulled twice).
2. **Geometric product:** the Clifford product of the two spinors. Its
   vector part contains the cross product v_L × v_R (the commutator —
   by Schur's lemma the *only* natural bilinear nonlinearity in 3D); its
   scalar part contains s_L·s_R − ⟨v_L, v_R⟩.
3. **Fusion gates:** three sigmoid gates, computed from both children,
   mix {rotated left, rotated right, geometric product}.
4. **Output rotation** R_O, then normalization and activation
   (see geometry knobs below).

Causal LM needs a state for *every prefix*, not just the root. The
Fenwick decomposition provides it exactly: every prefix of length L is
the disjoint union of at most ⌈log₂ L⌉+1 already-computed tree nodes,
which a **fold** composes into the prefix state. Total cost O(T log T)
compositions per layer. Because the decomposition depends on the binary
representation of L, every prefix length is computed by a structurally
different circuit — *position is structure*.

## 3. The fold variants (`fold_mode`)

The fold is where the prefix's ≤ log T blocks become one state. It is
the most-studied component of the architecture.

### `left` — the incumbent

Sequential gated fold over the blocks, largest (oldest, biggest) first,
using the full compose node at each step. All headline results are this
fold. `fold_rotors='separate'` gives the fold dedicated rotors instead
of sharing the tree's; `fold_scale` twists each incoming block by an
angle proportional to its tree level (scale, not position:
extrapolation-safe).

### `rack` — isometric transport

Each incoming block **conjugates** the accumulator by its own unit
spinor: acc ← q̂(next) · acc · q̂(next)⁻¹ + g ⊙ next. Conjugation is an
exact per-block isometry and satisfies the rack (self-distributivity)
axiom — verified numerically to 1e-6 in the test suite. Old content is
only ever *rotated*, never gated down, normed, or squashed.

- **`rack_exitnorm=True`** (the r18 variant): one LayerNorm at the fold
  *exit*, folded positions only. Motivation (the r17 diagnosis): the
  norm-free fold was stable but trained slowly — "normalization was
  doing optimization work as well as informational harm." Geometry
  inside, training machinery at the boundary.
- Status: **exploratory.** Isometry held; convergence at fixed budget
  was slower than `left` both at 20k steps (r17) and at the v9 toy rung.

### `spine` — the snapshot cache

Runs the compacted left fold but **keeps every running accumulator**
(the "spine" of progressively richer prefix states), then lets the final
state attend over its own spine with a deterministic sinusoidal step
encoding (defined for every depth: extrapolation-safe). Fenwick blocks
arrive largest-first, so spine step s is refinement at exponentially
descending scale — the tree-native form of an exponential-lag snapshot
cache. Status: **falsified at the toy rung** (+3.66 PPL vs the msup
baseline) — the fifth consecutive readout-side null, and the first under
fair conditions (objective paying for multi-scale content via msup).
"Availability ≠ use": access is not the bottleneck.

### `attend` — one-hop block attention

One attention round over each position's own ≤ log T Fenwick blocks
(keys carry the same deterministic level encoding), then a single
compose of the attended context with the most recent block. Every block
is one hop from the readout. Status: **validated for speed** (25%
faster per step at quality parity) — and historically important: it
falsified the structural bottleneck theory *in reverse* (one-hop access
*halved* early-context sensitivity), which is how the project learned
that forgetting is learned, not structural.

### `revolving` and `balanced`

Ablation arms. `revolving`: a fixed-depth gated fold where a learned
per-stage gate decides compose-vs-skip (`tree_drop` randomly forces
gates to identity during training). `balanced`: pairwise reduction over
the block slots. Both selftested; neither claimed.

### `oam` — orbital angular momentum (charged multi-channel fold)

In optics, a beam with a helical phase front carries OAM in integer
multiples — its *topological charge* m — and beams of different charge
are orthogonal modes: several information channels can be multiplexed
onto one beam, kept separate in flight, and demultiplexed at the
receiver. The OAM fold applies the same trick to the accumulator.

Instead of one accumulator, the fold runs **k parallel isometric
channels**. Channel c carries charge m_c = c − (k−1)/2 (k=4 → −1.5,
−0.5, +0.5, +1.5; custom via `oam_charges`):

- **Transport is charge-free and shared:** within every channel the
  accumulator is transported by rack-pure conjugation (exact isometry)
  or by the full node (`oam_transport='node'`). Channels never mix in
  flight; charge is their identity, not an interaction.
- **Charge acts only at injection** — the multiplexing step: an
  incoming Fenwick block from tree level l enters channel c twisted by
  θ_c(l) = m_c · φ · l about its z-axis. One learnable frequency φ per
  layer (init π/4, `oam_phi`; 0 = charge-off ablation, φ frozen). The
  twist depends on tree *level*, not absolute position:
  extrapolation-safe by construction.
- **Readout composes the channels** through the standard node, balanced
  pairwise in a fixed order — `oam_pair='seq'` (ascending charge) or
  `'conj'` (conjugate charges −m,+m meet first; the compose is
  non-commutative, so order is architecture) — or softmax-mixed
  (`oam_combine='sum'`).

The hypothesis was *phase multiplexing*: distinct charges let the fold
carry several rotationally-separated summaries of the same prefix at
zero extra norm cost. k=1 with charge 0 is bitwise the rack fold
(selftested), so the mechanism test is clean. Supporting flags:
`oam_shared_gate` (one gate for all channels), `oam_levelgate` (induced
charge: a per-level learned gate on the twist), `oam_chan_emb`
(per-channel identity embedding).

Status: **falsified — the charge line is closed.** Pre-registered under
the v8.1 spec; the v8.2 crash was diagnosed as the compose *readout*,
not the twist (v8.3, single-flag attribution); at maximum dose, across
two transports and two readouts, charge produced a null on its decision
metric. The machinery remains fully functional and selftested — a
falsified mechanism with verified infrastructure is a starting point for
experimentation, not dead code.

### `scan` — the associative-scan sibling

Replaces the tree+fold entirely with a Hillis–Steele inclusive scan that
produces all prefix states in O(log T) depth. The in-scan operator is
the conformal semidirect law on affine pairs (q, b) — q a homogeneous
quaternion (unit direction × learned magnitude in (0,1), an LRU-style
input-dependent decay), b a paravector — exactly associative by
construction, no gates or norms inside the scan. This is OPERA's answer
to the S4→Mamba lineage: the same associative-parallelization bargain,
promoted to a non-commutative group. Supporting arms:
`scan_decay_bias` (init half-life of the decay), `scan_salience`
(rank-64 FiLM gate on the readout), `workspace` (prefix-causal latent
cross-attention). Status: **validated for quality parity** with the
left fold at 0.57× the op dispatches; the decay-collapse diagnosis
(trained decay → ~1-token half-life regardless of init) motivated the
v9 training line.

## 4. Geometry knobs

| knob | values | meaning |
|---|---|---|
| `norm_mode` | `'layer'` (default) | full LayerNorm after the node. Cheap and effective, but its cross-dimension mean subtraction breaks strict equivariance |
| | `'rms'` | ONE global RMS over the d-vector (no mean subtraction) + per-block learned gains. Rotation-invariant, preserves relative block magnitudes — separates LayerNorm's useful part (energy pattern) from its geometry-breaking part |
| | `'blockrms'` | per-block RMS + learned gain. Fully SO(3)-equivariant — but measured +4 PPL (it forced every block to the same norm and erased the block-energy code); kept as a falsified option |
| `act_mode` | `'tanh'` (default) | tanh(x) + 0.1x after the node |
| | `'linear'` | no pointwise squash — with `'rms'`/`'blockrms'` the node is geometry-clean and all nonlinearity comes from the geometric product + gates |
| `node_residual` | `False` | learned gated residual: parent ← r·parent + (1−r)·½(h_L+h_R), r init ≈ 0.88 |
| `lock_mode` | `'none'` (default) | `'interference'` modulates the fusion gates by the children's geometric alignment (the relative-lock diagnostic lineage) |
| `rot_mode` | `'so3'` / `'free'` | quaternion rotations vs unconstrained 3×3. Pre-registered, nulled twice: the manifold constraint does not measurably contribute — `'free'` is the adopted default |

## 5. Training and optimization arms (v9)

The measured lesson of the project: the tree responds to being *trained*
like a tree. Four consecutive readout-side nulls plus the probe verdict
(forgetting is learned, not structural) moved the program here.

- **`fold_gate_bias=(b0, b1, b2)` — chrono fold init (arm A).** The
  fold's transport composes share the tree's gate weights but get a
  dedicated, learnable bias: accumulator init b0, new block b1,
  geometric product b2 (suggested 2.0, 0.0, −2.0). Chrono-init logic:
  start near identity transport, let the loss learn compression.
  Status: **falsified at the toy rung** — the init *held* (unlike the
  scan's decay collapse) and was a small uniform tax.
- **`msup_loss` — multi-scale node supervision (arm B).** Every internal
  tree node predicts the first token *after* its span, through the
  shared head (zero added parameters). The objective pays for what
  intermediate states carry, at every scale. Status: **validated at the
  toy rung: −9.87 PPL** — the first training-method win. Notable: the
  *unsupervised* tree already carries span-boundary signal (aux loss
  6.18 vs 9.21 chance).
- **`curriculum_len` — length curriculum (arm C).** Train at cur0
  tokens, double every `every` steps, capped at max_len; deterministic
  in step (exact resume), eval full-length. The tree's proven skill is
  depth extrapolation (boolean experiments: trained at 4, solved 32);
  the curriculum finally exercises it in the LM. Status: **validated**
  (quality parity while processing fewer tokens; stacks with msup
  without interference).
- **`grad_checkpoint`** — `'level'` recomputes each node's ~18
  intermediates in backward (~25–35% slower, several-fold activation
  memory reduction); `'layer'` checkpoints whole layers.
- **`use_metal`** — fused Metal (Apple Silicon) kernels for the compose
  node (rotation + geometric product + gated combine in one kernel;
  requires `rot_mode='so3'` for the verified adjoint).
- **`aux_frac`** (in `train_lm_loss`) — auxiliary-layer head+CE computed
  on a random fraction of positions, rescaled to stay unbiased; the
  final-layer term (reported PPL) is never subsampled.

## 6. Standing advice for new arms

1. One variable per run. 2. Same-session incumbent. 3. Pre-register the
gate before the run. 4. Zero-init or deterministic-init new parameters so
flags-off is bitwise the incumbent — and assert it in the selftest suite
(`python -m opera_lm.selftest`). Every arm above followed these rules,
including the falsified ones; that is why the record is trustworthy in
both directions.

