# Redundancy, mutual-information scaling, and the Fenwick readout — research + plan (2026-09-24)

Prompted by two Quantum Darwinism papers Merna shared (Riedel & Zurek
2010; Zwolak, Riedel & Zurek 2014). Question: *does language carry a
similar redundancy, and can that improve OPERA?* This note collects the
language/ML literature that makes the analogy precise, maps it onto the
measured OPERA findings (F3–F7, G1–G4 in
`OPERA_Depth_Survival_diagnostic.md`), and lays out a coherent plan.
Standing constraint: **no attention in any form** (see
`OPERA_Literature_Scan_2026-09-24.md` §4); every proposal below is
attention-free.

---

## 1. The physics source: redundancy as accumulated evidence

**Riedel & Zurek 2010** ("Quantum Darwinism in an Everyday Environment:
Huge Redundancy in Scattered Photons", arXiv 1001.3419). Photons
scattered off a 1 μm dust grain in sunlight record its position ~10⁸
times in 1 μs. The signature is a **plateau**: the mutual information
between the system and an environment fragment jumps to the full
classical value H_S with a tiny fragment, then stays flat. Only the
*pointer* observable (position) is copied; everything else about the
object is recorded once or not at all.

**Zwolak, Riedel & Zurek 2014** ("Amplification, Redundancy, and the
Quantum Chernoff Information", arXiv 1312.5373). Makes it quantitative:
each environment subsystem carries a fixed rate of *distinguishing
evidence* (the quantum Chernoff information ξ), evidence accumulates
additively, and

    R_δ ≃ N · ξ / ln(1/δ)

— redundancy grows linearly with environment size N, and the fragment
needed for a (1−δ)-accurate record grows only like ln(1/δ)/ξ (errors fall
exponentially in the amount of environment observed).

The information-theoretic part (mutual-information plateau, R_δ,
additive evidence) is classical and applies to text. The quantum part
(discord; I rising to 2H_S only for the whole environment) has no
linguistic counterpart and is not used here.

## 2. The language-side literature

### 2.1 Language is redundant, and its redundancy has structure

- **Shannon redundancy.** English is ~70–75% redundant at the character
  level. House numbers: OPERA byte model 1.30 bits/byte vs the best exact
  n-gram (ctx-6) 1.92 (`runs_reprs/ngram_ce.json`).
- **Two regimes, as in Quantum Darwinism.** A few *pointer* variables —
  topic, language, register, speaker, grammatical state (number, gender,
  tense) — are stamped redundantly across a text (agreement morphology,
  coreference, recurring topic words); a few words almost anywhere
  identify them (plateau). Specific content — a rare name, a number, the
  exact spelling of a once-mentioned word — is recorded once.

### 2.2 Hierarchy produces power-law correlations — Lin & Tegmark (Entropy 2017, arXiv 1606.06737)

- **Theorem:** in any (irreducible, aperiodic) Markov process / probabilistic
  regular grammar, the mutual information between two symbols decays
  **exponentially** with separation (timescale set by the second
  eigenvalue).
- A probabilistic **context-free** (hierarchical, recursive) grammar can
  instead produce **power-law** decay.
- Measured: English Wikipedia, English and French text, the human genome
  and Bach all show roughly power-law MI decay (their Fig. 1). LSTMs
  reproduce critical behavior but *under-predict* long-range MI; tree-like
  or recurrent deep networks can emulate the recursive generative process.
- **Pestun & Vlassopoulos**, "Tensor Network Language Model" (arXiv
  1710.10248): cite a character-level fit I(l) ≈ c₁ l^(−0.37) + c₂ for
  25 < l < 1000 in English literature and propose isometric tensor
  networks as a renormalization-group view of language.

**For OPERA:** a dyadic tree is the right *family* of inductive bias for
a hierarchical source — supporting framing for the paper. Caveat: OPERA's
tree is a fixed dyadic hierarchy, not the syntactic one (see §2.4).

### 2.3 The state-size law — Hilberg → L²M

- **Hilberg's conjecture / relaxed Hilberg (Dębowski).** The mutual
  information between two adjacent text blocks of length L grows like a
  **power law**, I(L) ∝ L^β.
- **L²M: Mutual Information Scaling Law for Long-Context Language
  Modeling** — Chen, Mayné i Comas, Jin, Luo, Soljačić, NeurIPS 2025
  (arXiv 2503.04725):
  - verifies **bipartite** mutual information I(X_{1:ℓ}; Y_{ℓ+1:L}) ∝ L^β
    on PG19 and Wikipedia, using LLaMA-3.1-405B and DeepSeek-V3 as density
    estimators. Both of their estimators likely *underestimate* β. (The
    exact β was not in the text I extracted; ~0.4–0.5 is a reading off
    their figures, so treat it as an assumption.)
  - bipartite MI scales independently of the classic two-point MI and is
    the quantity long-context modeling needs.
  - **Theorem 5.2:** the bipartite MI a model can capture is bounded by
    its history state, I ≤ C·dim(z) + log M.
  - **Theorem 5.4 (L²M condition):** a single model is MI-capable at all
    lengths only if dim(z_{L/2}) ≳ L^β.
  - **Architectures:** transformers satisfy it (the KV cache grows ∝ L);
    SSMs, RNNs and linear attention do not (constant state); a
    logarithmically growing state does not either, for power-law data.
  - **Experiments:** on synthetic data, performance collapses onto a
    single curve in the ratio I/dim(z). On PG19, Mamba beats GPT-2 early
    in the sequence, but its NLL plateaus at later positions unless model
    size grows.

**Where OPERA sits.** Per layer, OPERA's history state — the Fenwick
roots that future positions read — is O(d · log T): between an SSM's
O(d) and a transformer's O(d · T). The *readout* each position hands to
the head is O(d). By L²M, a fixed-width OPERA falls behind L^β
asymptotically. Its log T growth helps, but only slowly.

**Sizing heuristic** (assumption-laden, for orientation only). Path A
trains at 512 and evaluates to 8192, a 16× ratio.
- At β ∈ [0.4, 0.5], I^BP grows 16^β ≈ 3.0–4.0×.
- The Fenwick root count grows only ~⌈log₂ 8193⌉/⌈log₂ 513⌉ = 14/10 = 1.4×.
- Holding the ratio I/dim(z) constant would therefore need per-root state
  ~2.1–2.9× wider. That is `state_mult` / `state_tie` k ≈ 2–3.

This assumes the model is MI-capable at the training length and that
the theory's *capability* bound governs *extrapolation*. Both are
untested; §4 Phase 1b measures the curve directly.

### 2.4 Fixed trees miss boundary correlations — MERA

Tree tensor networks capture hierarchical correlations, but two
neighbours on opposite sides of a block boundary only interact at their
lowest common ancestor. **MERA** (multiscale entanglement renormalization
ansatz) fixes this with **disentanglers**: local operators applied across
neighbouring blocks *before* each coarse-graining step. They remove
short-range correlation — i.e. redundancy — between adjacent blocks, so
the coarse-grained level carries only what is new. This is what makes
MERA capture critical (power-law) systems that plain trees cannot. (MERA
background: the 2025–26 tensor-network ML surveys, e.g. arXiv 2604.14287.)

**For OPERA:** tokens 7 and 8 first meet at the level-4 node of a
16-token block. Every tree node is built blind to its left context and
is only composed with it later, in the fold.

### 2.5 The right measurement language — Partial Information Decomposition

PID (Williams & Beer; reviews arXiv 2603.06678; applied to 26 vision-
language models in arXiv 2603.29676) splits the information two sources
A, B carry about a target Y into:
- **redundant** — carried by either source;
- **unique** — carried by only one;
- **synergistic** — carried only jointly.

With the simple minimum-mutual-information (MMI) redundancy, every term
follows from three mutual informations, each estimable by probe
cross-entropy:

    Red = min(I(Y;A), I(Y;B)),  Unq_A = I(Y;A) − Red,
    Syn = I(Y;A,B) − I(Y;A) − I(Y;B) + Red

This is the formal version of "which information is copied across
Fenwick blocks".

## 3. Synthesis — what this says about OPERA

1. **The fold is a fixed-budget compressor facing a two-regime source.**
   It keeps the redundant pointer variables and drops single-copy
   content. That is F3 (no addressable bytes beyond k≈4, yet 0.6 bits
   better than n-grams) and F4 (readout participation ratio 27 of 512):
   capacity concentrated on a few pointer directions.
2. **The two regimes want different operators.** Pointer variables are
   inferred from accumulated evidence, and evidence *adds* (Zwolak et
   al.: log-likelihood ratios sum; errors fall exponentially). The ideal
   aggregator for them is additive, associative and length-agnostic — a
   normalized running sum — not a non-associative, norm-squashed compose.
   Single-copy records need the opposite: routing without averaging.
3. **Redundancy between adjacent blocks wastes capacity twice.**
   Neighbouring blocks share pointer content, and the fold stores it in
   both the accumulator and the incoming block. MERA's disentanglers and
   predictive coding / DPCM remove exactly this.
4. **State size must track the data's MI growth** (L²M). No amount of
   allocation cleverness beats an undersized state asymptotically.
   `state_mult` / `state_tie` are the levers the theory names, and
   §2.3's heuristic gives a target k for Path A's length ratio.

## 4. Coherent plan

**Principles:** attention-free only; every arm single-variable against
the current best base; flags-off bitwise; same-session control.
Deltas under ~0.5% get a second seed before being believed (cross-
session spread of the incumbent recipe ≈ 0.2%).

### Phase 0 — running now: new baseline + first capacity readings

Queue `runs_reprs/queue_2026-09-24.log`: `st_off`, `st_head`, `x_sam`,
`x_nmix`, `st_auxw0`, `st_wide`, `x_tie2`. Output: the Phase-2 base
(stream head? SAMuon? aux weight?) and a first read on whether state
size binds (`x_tie2` vs `st_off`).

### Phase 1 — instruments (CPU, no training, ~hours)

- **1a. Fenwick-block PID probe.** Take gathered block states from a
  trained checkpoint. Target Y = next byte (and byte t−k). Estimate
  I(Y;A), I(Y;B), I(Y;A,B) by probe cross-entropy for (oldest, newest)
  block pairs and for individual levels, then compute MMI
  Red / Unq / Syn. Also a Darwinism curve: I(Y; newest m blocks) vs m —
  plateau or steady climb.
- **1b. Captured-MI curve (L²M instrument).** Apply L²M's direct
  estimator to OPERA itself: I^BP(ℓ) ≈ E[log q(Y|X) − log q(Y)] for a
  fixed-length Y after a prefix of length ℓ, across ℓ up to eval length,
  per arm. L²M's signature: the curve bends flat when state saturates,
  and lifts with state size.
- **Decision rules:**
  - old blocks carry large Unq that the fold's readout loses → Phase 2a/2b
    (innovation / disentangling);
  - Red dominates → 2c (consensus channel) frees the most capacity;
  - Syn large → the compose non-linearity (geometric product) is
    load-bearing: keep it, do not linearize;
  - 1b curve flattens and `x_tie2` lifts it → 2d (size the state by
    L²M) takes priority.

### Phase 2 — attention-free architecture arms (each vs the Phase-0 base)

- **2a. Innovation fold** (predictive coding, fold-level). Before each
  fold compose, `acc ← acc − (nxt·V)·Uᵀ`, U zero-init: bitwise incumbent
  at init, interior. Stores in the old-content channel only what the
  newer block cannot predict. Targets the fold, the measured destroyer
  (F3 span check). *Expect:* k=8 decode ↑, readout PR ↑.
- **2b. Causal disentangler** (MERA, tree-level). Before composing level
  ℓ+1, each node j gets a zero-init update from its *left* neighbour j−1
  at the same level: `node_j ← node_j + Δ(node_{j−1}, node_j)`.
  - Causal: any prefix that reads node j already covers node j−1's span.
  - Append-only: node j−1 exists when node j completes, so the
    incremental decoder stays exact.
  - Makes every node context-aware and innovation-coded at every scale;
    fixes boundary blindness.
  - Cost: one extra small op per tree node (the tree is O(T); the fold
    stays the dominant cost).
- **2c. Consensus (evidence) channel.** A normalized running sum of
  per-token evidence vectors, `c_t = (1/t) Σ_{i≤t} W x_i`, added to the
  readout through a zero-init projection.
  - Associative and exact; O(T) training, O(1) per token in the decoder.
  - No query, no key, no content-based weighting: attention-free.
  - Length-agnostic by construction (more evidence, same scale).
  - Lets the fold's capacity go to single-copy records.
  - Variant: a fixed multi-rate EMA bank instead of the plain mean.
- **2d. State sized by L²M.** `state_tie` with k from §2.3 (k≈2–3 for
  16× extrapolation), compared against the width-matched control, and
  checked with instrument 1b (does the captured-MI curve keep rising to
  eval length?).
- **2e. Fragment dropout** (length robustness). During training, randomly
  skip composing a Fenwick block into the fold, biased toward the
  oldest/largest block. Pointer variables live in every fragment, so the
  model learns not to depend on any one block — the regime of Path A's
  8192-token eval, where 87.5% of positions read a never-built-level
  block. Off at eval.
- **Priority order:** 2a → 2c → 2b → 2d → 2e, re-ordered by Phase 1's
  decision rules. 2a and 2b both remove adjacent-block redundancy (at
  fold vs tree level): run 2a first, and 2b only if 2a engages
  (k-decode rises) but plateaus.

### Phase 3 — combine and stress

- Stack the Phase-2 winners on the Phase-0 base; two seeds.
- L=8 (where `resid_mode='add'` should matter) and long-T evaluation (the
  unseen-level regime).
- Instrument 1b on the final stack: the captured-MI curve should keep
  rising with length where the incumbent's bends flat.

### Out of scope (by the no-attention rule)

Any query–key scoring over positions, chunks or layers (AttnRes/DAR,
Log-Linear query gates); content-query memories (`mem_mode='delta'`);
attention teachers (SMT); the attention instantiations of
Prefix-Scannable Models.

## 5. Phase 1 results (2026-09-24, control arm `st_off`, byte d512/L2)

Instruments: `experiments/block_pid.py` (1a), `experiments/mi_curve.py`
(1b), shared loader `experiments/arm_utils.py` (its layer loop matches
`forward()` logits exactly). JSON/logs in `runs_reprs/block_pid_st_off_*`
and `runs_reprs/mi_curve_st_off.*`.

### 5.1 Instrument 1b — the captured-MI curve saturates at ~100 bytes

Paired design: 64 held-out documents; a fixed 32-byte window; context
length l varied. I(l) is the bits about the window's last 31 bytes that
the model extracts from the l bytes before it.

| l | 1 | 4 | 16 | 48 | 96 | 256 | 1024 (train) | 2048 |
|---|---|---|---|---|---|---|---|---|
| I (bits) | 1.60 | 5.20 | 7.39 | 10.34 | 11.01 | 11.18 | 11.17 | 11.15 |
| window CE (b/byte) | 1.894 | 1.777 | 1.707 | 1.612 | 1.590 | 1.585 | 1.585 | 1.586 |

(± ≈ 0.6–0.9 bits sem.) Log-log slope over [16, 1024] = 0.079; growth
×1.006 from 512 → 1024 and ×0.998 from 1024 → 2048. The model extracts
**nothing further from context beyond ~100 bytes**, while natural
language's bipartite MI keeps growing as a power law (L²M). This is the
L²M signature of a saturated history state — measured, not assumed.

**Implication for the position-curve headline.** OPERA's flat
extrapolation (the README's "+0.07 nats at 2×") coincides, at this rung,
with a model that does not use context beyond ~100 bytes. Flatness is
then partly the absence of long-range use, not only robust long-range
use. The same logic as Path A's H4 (a flat curve does not discriminate
when the long context is unused) applies to OPERA's own curve. Checking
this at the 22M word-level rung and on Path A is now important.

### 5.2 Instrument 1a — the fold erases single-copy content; it creates next-byte synergy

Layer 0 (disjoint fragments), positions whose prefix has 5 blocks,
36,911 positions from 400 held-out sequences, MLP probe on 128-dim
per-source PCA. Values in bits (probe cross-entropy below the unigram
baseline, a lower bound).

| target | in its own block | all 5 blocks | fold readout |
|---|---|---|---|
| last byte of the oldest block (single copy) | **4.57** (of 4.93) | 4.33 | **0.00** |
| first byte of slot 1 | 0.93 | 0.42 | **0.00** |
| first byte of 2nd-newest block | 1.62 | 1.86 (newest-3: 2.24) | 0.22 |
| byte t−4 | 1.01 | 1.04 (newest-3: 1.42) | 0.70 |
| byte t−8 | 0.14 | 0.11 (newest-3: 0.13) | 0.06 |
| next byte | 1.07 (newest) | 1.01 (newest-2: 1.28) | **1.88** |

Layer 1 replicates: oldest block's last byte 4.23 in-block → 0.00 after
the fold; next byte 2.11 (newest block) vs 2.41 (fold).

- **Single-copy content present in old blocks is erased by the fold** —
  the information-theoretic form of F3, now with the input side
  measured. (Allocation, not necessarily harm: the objective may not pay
  for it. Phase 2 tests whether retaining it pays.)
- **The fold creates next-byte information** ~0.6–0.8 bits beyond an MLP
  read of the raw blocks: the compose non-linearity is load-bearing
  (decision rule: keep it, do not linearize).
- **MMI redundancy between oldest and newest blocks ≈ 0** for all
  byte-identity targets — expected (byte identity is single-copy by
  nature).
- **Pointer-variable test failed to measure.** A document-level style
  label (k=16 clusters of byte histograms) read 0 bits from every source
  including the fold. With ~320 training documents a document-level
  label is too coarse to probe; the redundancy (Darwinism) prediction is
  neither supported nor refuted. A position-level pointer target is
  needed before 2c (consensus channel) can be justified by data.
- Measurement notes: the oldest block's *first* byte is not a valid
  target (the oldest block always starts at position 0 → one value per
  document); lag-k targets under-read all sources (a lag target's offset
  inside its block varies with t, which no block encodes) — the
  block-anchored targets above avoid this.

### 5.3 Decisions (per §4 Phase 1 rules)

| rule | evidence | decision |
|---|---|---|
| old blocks carry Unq the fold loses | yes, strongly (4.6 → 0 bits) | 2a / 2b stay high |
| Red dominates | not measured validly | **2c on hold** until a valid pointer target |
| Syn large | yes (fold > block read by 0.6–0.8 bits) | keep the compose non-linearity |
| MI curve flattens; lifts with state | flattens at ~100 bytes; lift test = `x_tie2` (queued) | **2d first**, measured by 1b |

Revised Phase 2 order: **2d (state size, via `x_tie2`'s MI curve) → 2a
(innovation fold) → 2b (causal disentangler) → 2e (fragment dropout)**;
2c pending measurement. Primary Phase-2 metric alongside BPB: the
**plateau height and onset** of the 1b curve, and fold retention of the
single-copy targets (1a).

## 6. Sources

- Riedel & Zurek, *Quantum Darwinism in an Everyday Environment: Huge Redundancy in Scattered Photons* — arxiv.org/abs/1001.3419
- Zwolak, Riedel & Zurek, *Amplification, Redundancy, and the Quantum Chernoff Information* — arxiv.org/abs/1312.5373
- Lin & Tegmark, *Criticality in Formal Languages and Statistical Physics* (Entropy 19, 299, 2017) — arxiv.org/abs/1606.06737
- Pestun & Vlassopoulos, *Tensor Network Language Model* — arxiv.org/abs/1710.10248
- Chen, Mayné i Comas, Jin, Luo, Soljačić, *L²M: Mutual Information Scaling Law for Long-Context Language Modeling* (NeurIPS 2025) — arxiv.org/abs/2503.04725
- Dębowski, *The Relaxed Hilberg Conjecture: A Review and New Experimental Support* (J. Quant. Linguistics 22(4), 2015); Hilberg's hypothesis — en.wikipedia.org/wiki/Hilberg's_hypothesis
- *Quantum-inspired tensor networks in machine learning models* (TTN/MERA survey) — arxiv.org/abs/2604.14287
- *The mathematical landscape of partial information decomposition* — arxiv.org/abs/2603.06678; *A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models* — arxiv.org/abs/2603.29676
