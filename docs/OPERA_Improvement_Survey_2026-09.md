# OPERA improvement survey — literature scan, 2026-09-07

**Scope.** Candidate improvements drawn from the 2025–26 literature, filtered
to those that preserve OPERA's four invariants:

- **I1** no positional encoding of any kind (position is the shape of the circuit)
- **I2** no pairwise token–token attention, no softmax over positions
- **I3** composition through the geometric node (rotate → geometric product → gates)
- **I4** exact prefix-causal readout via the Fenwick decomposition, O(T log T)

Every item below is checked against all four. Items that would break one are
listed in §6 with the reason, so the exclusion is on the record too.

---

## 1. Level-adaptive fold transport — the arm §8 names, with a proven recipe

**Status of the idea in the project:** already named in Paper v1.3 §8
("input-adaptive per-level transport weights in the fold"). Not yet specified
or implemented.

**External evidence.** *Adaptive Memory Decay for Log-Linear Attention*
(arXiv:2605.06946) runs on the **identical Fenwick decomposition**. Replacing
the static per-level weight with an input-adaptive one:

| metric | static λ | adaptive λ |
|---|---|---|
| MQAR, kv=32 | 2.9% | **57.9%** |
| length gen, train 128 → eval 256 | 3.3% | **33.2%** |
| selective copying, seq 1024 | 40.6% ± 19.8 | **54.8% ± 2.3** |
| WikiText-103 PPL (d=512) | 224.71 | **218.57** |

Note the variance collapse as much as the means — the static version was
seed-unstable.

**Their parameterization** (transplantable verbatim):

```
h_t = d_t W1                       # W1: [L, d_h],  d_t = per-level features
λ_t = softplus(h_t W2 + b)         # W2: [d_h, L],  W2 zero-init, b = 0.54
                                   # softplus(0.54) ≈ 1.0  → identity at init
```
Softplus, **not** softmax: independent per-level scaling avoids inter-level
competition, so several scales can fire at once.

**Where it lands in OPERA.** `model.py:1554` (`fold_mode == 'left'`). The fold
gate is computed inside `_compose` from `(acc, nxt)` plus an optional
**constant** `gate_bias` (`fold_gate_bias`, v9 arm A). It is content-dependent
but **level-blind** — `lvl` is computed at `model.py:1184` and used only by
`fold_scale` / `level_sin_enc`, never by the gate.

**This is why arm A failed, and why the adaptive version is a different bet.**
Arm A was a *static, level-blind* transport bias, and it was falsified as "a
small uniform tax." The adaptive-λ result says exactly that: static per-level
weighting is the thing that doesn't work. OPERA's own null and the external
result agree on the negative half; the positive half is untested here.

**The extrapolation trap — read before implementing.** Do **not** use a
one-hot over a fixed number of levels. At train length 512 the fold never sees
levels ≥ 10; a one-hot's rows 10–13 would arrive at eval length 8192
never-trained, and Path A's H1 is measured at exactly 16× training length.
Feed the level as a **smooth scalar function** — `lvl` itself plus its
`level_sin_enc` (already deterministic and defined at every depth, and already
declared extrapolation-safe for `fold_scale` and the attend fold). That keeps
I1 intact: level is *scale*, not position, and is a property of the circuit
shape, not an injected coordinate.

**Invariants:** I1 ✓ (level = circuit shape) · I2 ✓ (gates, no scores) ·
I3 ✓ (node unchanged) · I4 ✓.
**Flags-off bitwise:** yes, with `W2` zero-init.
**Caveat to state in the prereg:** this is a fold-side arm, and five
consecutive readout/fold-side arms have nulled. The reason to run it anyway is
that all five tested *availability* or *static* weighting; none tested
input-adaptive transport, and the sibling architecture on the same
decomposition reports a 20× swing on memory load.

---

## 2. Unexplored levels — the state-passing intervention, generalized to tree depth

**Status:** `opera-chat/prep_statepassing_data.py` exists, correctly motivated
by Buitrago Ruiz & Gu (arXiv:2507.02782). Grep shows it is referenced **only**
in the two Colab notebooks — it is not a flag in `opera_lm.train`, not in the
Kaggle Path A engine, and no result is recorded for it anywhere in `docs/`.

**Why this matters more for OPERA than for any SSM.** The unexplored-states
hypothesis says recurrent models fail at length because test-time states are
never visited in training. OPERA has the sharpest possible version of this:
the compose function is weight-tied across levels, and at training length 512
**tree levels 10–13 receive literally zero gradient**. Path A then evaluates
at 8192. The failure mode isn't "states drift out of distribution" — it's
"this circuit was never trained." Your own `_multistate_readout` docstring
already concedes the readout half of it ("slots beyond the training popcount
keep their never-trained small random gates").

The concat augmentation is the right instrument: chaining unrelated training
sequences synthesizes genuine deep-recursion node inputs from real tokens.

**RUN 2026-09-07 — `opera_lm/level_exposure.py`.** Step 1 below is done; the
measurement corrected two guesses in the first draft of this document and is
recorded in the Path A prereg amendments log.

| quantity (train 512 → eval 8192, Path A settings) | measured |
|---|---|
| fold entries per position, levels 0–8 | **0.50 each — uniform** |
| level 9 (span 512) exposure | 1 prefix per sequence, **256× deficit** |
| deepest *well*-trained level | **8** (span 256), not 9 |
| steps at full T=512 | **95%** (250 each at 64/128/256) |
| levels required at eval, never built | **10, 11, 12, 13** |
| eval fold entries at levels 10–13 | 4,096 each — **same frequency as trained levels** |
| eval positions reading ≥1 never-built block | **87.5%** (93.8% from level 8) |
| eval fold entries at never-built levels | **23.1%** (30.8% from level 8) |
| fold chain depth, trained → eval | 8 → 12 composes, but only **4.6%** of positions |

**Two corrections to this document's first draft.** (a) I claimed the
short→long curriculum "systematically underexposes deep levels." **False** at
Path A's settings — 95% of steps run at full length, and arm C is not
implicated at all. (b) I claimed levels 10–13 "receive literally zero
gradient." Too loose: the compose weights are *tied across levels*, and under
`fold_mode='left'` there are **no level-indexed parameters whatsoever**.
Nothing is randomly initialised at eval. The correct statement is that the
shared function is never *evaluated* on level-≥10 inputs during training —
an unvisited input regime, not an untrained module.

**Two findings worth the paper.** The uniform 0.50/position exposure across
levels 0–8 is a genuine positive result — the Fenwick decomposition starves no
scale, unlike a recurrence where early timesteps dominate. And the top-level
deficit is new: the root of the training tree is reachable by exactly one
prefix, so the deepest span the model ever learns to compose well is *half*
the training length.

**And it has no fix at fixed training length.** `msup` does not rescue it —
`msup_loss` breaks at `span >= T` (`losses.py:118`), so the top level is
explicitly skipped, and level 8's second node is dropped by the
`tgt_pos < T-1` mask too. That is not an oversight to patch: msup's target is
"the first token *after* the node's span," and for the root that token lies
outside the sequence **by definition**. No objective defined on a length-T
sequence can supervise the composition of a length-T span.

This is the useful result of the audit. The top-level deficit and the
never-built-levels gap are the *same problem* with the *same* and only
remedy — the model must see sequences longer than the span it needs to
compose. That is exactly what concatenation/state-passing provides, and it
makes §2.2 the load-bearing item rather than one option among several.

**Remaining steps:**

2. **Promote state-passing to a first-class training flag** in `opera_lm.train`
   (`--augment concat`, applied to a fraction of batches), so it can be gated
   like every other arm rather than living in a notebook.
3. **Short fine-tune phase, not a full retrain.** 2507.02782's finding is that
   a *brief* intervention suffices — which fits the Kaggle session budget and
   costs nothing against the pre-registered Path A schedules.

Both are **out of scope for Path A** (recorded as such in its amendments log)
and belong to a separately pre-registered follow-up study.

**Invariants:** all four untouched — this is purely data/training side.
**Priority:** highest expected-value-per-GPU-hour in this document. It is
objective/training-side, which is the one axis with a measured win in this
project (msup, −9.87 PPL), and it directly targets Path A's H1.

---

## 3. The readout-capacity ceiling — two dormant arms, and what the 2026 literature adds

Paper v1.3 §7 identifies a readout-capacity ceiling: OPERA reads each prefix
from one d-dimensional state. Two arms already exist, both zero-init,
selftested, and **never gated at scale**: `readout_mode='multistate'` (T0.4)
and `mem_mode='delta'` (T1.4).

The 2026 literature has converged hard on this being *the* gap between
linear-family and softmax models:

- ***Scaling Linear Attention Capacity*** (ICLR 2026): under typical configs
  linear attention's memory budget is roughly equivalent to softmax attention
  with a **64-token** context window. The gap on long-context retrieval is
  attributed to state size, not to the mixing rule.
- ***HOLA — A Hippocampus for Linear Attention*** (arXiv:2607.02303): pair the
  lossy recurrent state with a **bounded exact memory** (~64 entries).
  WikiText PPL 27.32 → 22.92 (below Transformer++); needle-in-haystack at 32k
  (16× training length) 0.14 → 0.58; ~5% peak memory overhead and ~12.5k extra
  parameters (0.004% of a 340M model).

HOLA's cache uses softmax over its entries, so **HOLA itself violates I2**.
But `mem_mode='delta'` is already the attention-free form of the same idea —
and two of HOLA's ablated details are missing from `_delta_memory`
(`model.py:1655`) and are each a few lines:

- **Surprise-gated writes.** HOLA selects what to retain by the delta-rule
  write magnitude `β·‖e‖`, where `e = v − Mk` is the prediction residual —
  tokens that were badly predicted *and* strongly committed. OPERA currently
  uses a plain `sigmoid(mem_beta(h))` (`model.py:1677`) with no residual term.
- **Decoupled query normalization.** HOLA reports that unit-L2 queries give
  near-uniform retrieval; a separate RMSNorm-γ on the *read* path sharpens it
  toward argmax while keys stay unit-norm for a stable write. OPERA normalizes
  `mem_k` but leaves `mem_q` unnormalized (`model.py:1675–1678`) — the
  opposite of HOLA's split.

**Recommendation:** gate `mem_mode='delta'` at the toy rung *with* these two
modifications, rather than as-is. Both are zero-risk to the invariants (rank-one
updates and matvecs; no score matrix, no softmax anywhere) and both have
published ablation support on the exact failure mode §7 describes.

**Invariants:** I1 ✓ · I2 ✓ (delta rule only) · I3 ✓ · I4 ✓.

---

## 4. Speed — the credibility problem, and the three levers

§7 currently concedes the wall-clock gap **widens** from ~3× at 22M to ~8× at
155M. That is the most quotable weakness in the draft. Three levers, cheapest
first:

### 4.1 Benchmark the Triton kernel you already have (hours)
`opera_lm/triton_kernel.py` is correctness-verified on a T4 (fwd/bwd ~1e-6)
but **wall-clock was never measured**. §8 admits this. Versor
(arXiv:2602.10195), whose bit-masked basis-contraction strategy the kernel
follows, reports that approach *solves* the Clifford-product bottleneck. This
is a done asset sitting unmeasured, and it gates the §7 crossover TODO.
Highest value per hour in the whole document.

### 4.2 Fix the Muon partition for 3×3 blocks
`opera_lm/muon.py` runs batched Newton–Schulz-5 over `rot_free`'s `nb`
per-block 3×3 maps. Two problems:

- NS-5's coefficients are tuned for the spectral profile of large matrices.
  For a 3×3, the orthogonal polar factor has a cheap closed form (batched SVD
  on 3×3, or the quaternion route). Exact polar is likely both faster and more
  accurate than 5 NS iterations here.
- *How Much Orthogonalization Does Muon Need?* (arXiv:2606.00371) and
  *Hierarchical Muon: Tiled Newton–Schulz* (arXiv:2606.27216) both bear
  directly on this shape.
- **Dion3** (arXiv:2608.11612) matches or beats Muon's loss at up to **6×**
  lower optimizer step time; its Gram Newton–Schulz reformulation cuts
  orthogonalization FLOPs 55%. Drop-in via the `dion` package.

**A free hypothesis worth testing while you're in there.** Muon orthogonalizes
each 3×3 block — which is *the SO(3) constraint reinstated through the update
rule* rather than the parameterization. `rot_mode='so3'` was pre-registered and
nulled twice; Muon on `rot_free` then gave +38.4 PPL. Those two facts sit
oddly together. A cheap three-way run — Muon on all matrix params (incumbent) /
Muon with the rot blocks moved to the AdamW side / AdamW everywhere — would
say whether the Muon win *is* the rotation manifold arriving by another door.
That would be a genuinely interesting result for the paper either way, and it
costs one toy-rung sweep.

### 4.3 Chunk the bottom of the tree
The structural cost is serial depth: ~2·log T stages × L layers of small
bandwidth-bound kernels (144 stages at T=512, L=8). The chunkwise-parallel
bargain that made GLA/Mamba-2/DeltaNet fast applies here: fuse the bottom
log₂C levels over a chunk of C tokens into one launch, cutting serial stages
to 2·log(T/C) + const. You already use the WY chunked form in `_chunk_delta`
(`model.py:1685`) — the same reasoning, applied to the tree rather than the
memory. **I4 is preserved exactly**: this changes the launch schedule, not the
decomposition or the values.

---

## 5. Where OPERA can actually win: state tracking

This is the strategic item. §4.8 established that OPERA does not beat a matched
transformer on raw-text pretraining at 155M, and Path A's H3 may confirm that
again. Chasing PPL parity is the axis where OPERA is structurally disadvantaged
(readout capacity, kernel maturity). State tracking is the axis where it is
structurally *advantaged*, and the literature has hardened enormously:

- *The Illusion of State in State-Space Models* (arXiv:2404.08819): the linear
  recurrence in parallelizable SSMs puts them in **TC⁰**; S₅ is in NC¹, so they
  cannot solve non-solvable word problems without depth scaling in T.
- *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*
  (arXiv:2411.12537, ICLR 2025): diagonal transitions restricted to [0,1] are
  why Mamba fails parity; extending to [−1,1] fixes it, and extends to DeltaNet.
- *DeltaProduct* (arXiv:2502.10297, NeurIPS 2025): products of n generalized
  Householders → diagonal-plus-rank-n transitions, a tunable expressivity dial,
  better state tracking **and significantly improved length extrapolation**.
- New permutation-composition benchmarks from Python REPL traces
  (arXiv:2602.14814) give a ready-made evaluation.

**OPERA's position on this axis is unusually strong and completely unstated in
the draft.** The whole DeltaProduct/negative-eigenvalue line is about escaping
diagonal, positive, commutative transitions. OPERA's transitions are per-block
**SO(3)** — non-abelian by construction — and the compose node is
**non-associative**, so the TC⁰ argument that bounds the whole scan-SSM family
does not apply to it in the same form. DeltaProduct's dial (n Householders) is
an approximation of what OPERA has natively.

Two consequences:

1. §8 already lists "a state-tracking evaluation track where the retired SO(3)
   constraint may show value on its home axis." That should be promoted from a
   future-work line to a **run**, using `rot_mode='so3'` with the isometric
   `rack` fold (exact group composition, `rack_exitnorm=True`). The rack fold's
   "optimization-limited at this budget" verdict was reached on LM perplexity;
   state tracking is the task it was designed for.
2. DeltaProduct's length-extrapolation finding is an external prior for Path A's
   H1 that the prereg doesn't currently cite.

**Invariants:** all four preserved; this is an evaluation track plus reviving a
shipped flag.

---

## 6. Rejected — would break an invariant

| candidate | why it's out |
|---|---|
| HOLA's KV cache as published (arXiv:2607.02303) | softmax over cached tokens → **I2**. Use `mem_mode='delta'` instead; keep HOLA's write-selection and query-norm details (§3). |
| Hierarchical sparse attention fixes (arXiv:2510.17196) — chunk encoder + CLS + bypassing residual | top-K retrieval over chunks is pairwise scoring → **I2**. The bypassing-residual idea alone (retrieved info gets dedicated modulation, not dilution in the main stream) is invariant-safe and is roughly what `mem_out`'s zero-init gate already does. |
| CARE / Clifford rotary embeddings (arXiv:2511.11665) | it is a *positional embedding*, even if a geometrically elegant one → **I1**. Tempting because it is quaternion-native; it is precisely the thing OPERA exists to not do. |
| Learned/dynamic chunk boundaries (H-Net lineage) | already excluded by the core-claim statement in §1 (learned-assignment pooling replacing the geometric node). |
| Sparse state expansion (arXiv:2507.16577) | row-sparse routing over an expanded state is a routing mechanism; would need a careful I2 argument before it's worth the trouble. Park it behind §3. |

---

## 7. Methodology note

*Most Transformer Modifications Still Do Not Transfer at 1–3B: A 2020–2026
Update to Narang et al.* (arXiv:2605.20798) re-runs the modification-transfer
question with downstream evals **and an explicit noise floor**. Worth citing in
§7's "Statistical power" paragraph: it is the strongest available external
support for the ~1.5 PPL noise band you already apply, and it makes the
single-seed caveat read as calibrated rather than as an apology. It also argues
for finishing the multi-seed TODO before adding arms — which cuts against
running everything in this document at once.

---

## 8. Suggested order

1. **Level-histogram diagnostic** (§2.1) — hours, no GPU, may reframe §4.3.
2. **Triton kernel benchmark** (§4.1) — hours, closes a §7/§8 TODO, and §7's
   widening-gap concession is the draft's weakest paragraph.
3. **State passing as a first-class flag + short fine-tune** (§2.2–2.3) —
   training-side, aligned with the project's only winning axis.
4. **Level-adaptive fold transport** (§1) — the one architecture arm with a
   strong external prior on the identical decomposition.
5. **`mem_mode='delta'` + HOLA's two fixes** (§3) — attacks the measured
   readout ceiling with an already-built, already-selftested arm.
6. **Muon partition experiment + Dion3** (§4.2) — one toy sweep, and the
   SO(3)-via-the-optimizer question is a paper-worthy result either way.
7. **State-tracking track** (§5) — the repositioning move, best done after
   Path A reports.

Items 1–3 change no architecture and risk no invariant. Items 4–5 are
zero-init flags, bitwise-identical when off, consistent with §7 of the
Mechanisms Guide ("standing advice for new arms").

---

## Addendum 2026-09-08: trajectory-dynamics program (executed)

The measurement-first program proposed after the curvature day has run
through Phase 1. Full records in three companion docs:
`trajectory_phase0_verdict.md` (the instrument and the three-architecture
baseline), `trajectory_phase1_arms.md` (homeo / quotient / fold_adapt at
the 5.9M rung), `trajectory_synthesis.md` (framing, positioning,
threats). Headline: integration is homeorhetic (constant-speed drift) in
OPERA, RoPE, and Mamba alike; content-adaptive stepping exists at scale
only as a brake (Mamba Δ); of the three new arms, fold_adapt — this
survey's §1 level-adaptive fold transport, made content-adaptive — is
the only one training uses, and it leans right on garden-path spillover
geometry at low power. New flags: `homeo_mode`, `node_paths=4`,
`fold_adapt` (all flags-off incumbent-identical; selftest arms in
`opera_lm/selftest.py`).
