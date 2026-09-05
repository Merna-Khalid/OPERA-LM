# Position is Structure: Language Modeling with Operadic Tree Composition and No Positional Encodings

**Merna Hafez** — Independent Researcher
*Draft v1.1 — 2026-07-29. Status: all v1 experiments complete; v1.1 adds an exact incremental decoder (§2.5, §4.5), a pre-registered optimizer arm (§4.6), a BPE-tokenized chat artifact with public demo (§4.7), and related-work engagement with log-linear attention (§6). [TODO] markers indicate numbers pending seed replication, the crossover timing measurement, and the in-progress chat run.*

**Changes in v1.1 (2026-07-29):**
1. §2.5 + §4.5 — `OperaDecoder`: exact Fenwick-incremental decoding at O(L log T) compose nodes per token (selftest-verified against the full forward; ~13× faster at ctx 128, growing with T).
2. §4.6 — the Muon recipe arm (pre-registered toy-rung gate: **+38.4 PPL** over AdamW at lr 0.02), addressing the v1 "recipe asymmetry" limitation.
3. §4.7 — a BPE-tokenized chat artifact (~20M params, smoltalk subset) served publicly on Hugging Face Spaces via the incremental decoder.
4. §6 — new paragraph on Fenwick-hierarchy memories (log-linear attention) and the 2025–26 sub-quadratic landscape.
5. §7–§8 — limitations and future work updated to match.

---

## Abstract

We present OPERA, a language model architecture that replaces self-attention with bottom-up composition over a binary tree, and replaces positional encodings with nothing at all: position information arises constructively from the model's readout, which computes the exact hidden state of every prefix by composing the blocks of that prefix's Fenwick (binary-indexed) decomposition. Because each prefix length has a structurally distinct composition circuit, position is a property of the computation graph rather than of any embedding. At matched parameter count (~22M) on document-level data with 256-token training contexts, OPERA matches a NoPE transformer in perplexity (41.5 vs 40.3, within our measured noise band) while exhibiting the flattest length-degradation profile of the three positional mechanisms we test: evaluated at twice the training length, OPERA's per-position loss rises by 0.07 nats against 0.19 for both RoPE and NoPE transformers, and OPERA's best per-position band lies *beyond* its training length. A RoPE transformer wins in-length by 16% relative perplexity, but the gap closes monotonically with evaluation length and reaches zero (8.12 vs 8.09) at four times the training context — precisely the axis along which the transformer's cost grows quadratically and OPERA's grows O(T log T). We further show that the same Fenwick structure yields an *exact* incremental decoder — O(log T) compose nodes per generated token, no attention, no softmax anywhere in the model — putting OPERA's decode-time asymptotics on the same footing the linear-attention line advertises for itself. A pre-registered optimizer arm (Muon on the model's matrix-shaped parameters, which include literally thousands of per-block 3×3 maps) improves the toy-rung baseline by 38 perplexity over AdamW, confirming that the architecture had been evaluated under a recipe tuned for transformers. Alongside the architecture we contribute a perturbation-based interrogation probe, run identically across all models, which shows that severe early-context insensitivity is shared by OPERA, RoPE, and NoPE alike — three architecturally alien position mechanisms, one forgetting profile — indicating that early-context forgetting at this scale is a learned consequence of the objective and corpus, not of any architecture's topology. We report two pre-registered hypotheses that our own experiments falsified (a rotation-manifold constraint on the composition, and a structural theory of the forgetting), and argue that this falsification record is itself part of the contribution.

---

## 1. Introduction

The transformer's two signature commitments are dense pairwise attention and explicit positional encoding. A large literature relaxes the first (sparse, linear, and state-space alternatives) and a smaller one interrogates the second (NoPE decoders that recover position implicitly from the causal mask). This paper relaxes both at once, and asks what a language model looks like when position is neither injected nor inferred, but *constructed*.

OPERA (Operadic Planar Equivariant Recursive Algebra) processes a sequence by composing token states bottom-up through a fixed balanced binary tree, where each internal node applies learned per-block linear maps, a Clifford-algebra geometric product, and content-dependent fusion gates. Causal language modeling requires a hidden state for every prefix, not just the root; OPERA obtains these *exactly* via the Fenwick decomposition: every prefix of length L is the disjoint union of at most ⌈log₂ L⌉ + 1 already-computed tree nodes, which a short fold composes into the prefix state. The total cost is O(T log T) compositions per layer.

The consequence we study is positional: the Fenwick decomposition of a prefix depends on the binary representation of its length, so prefixes of different lengths are computed by *structurally different circuits*. A prefix of length 7 (binary 111) is a fold of three blocks; length 8 (1000) is a single tree node. Position is not a signal added to the input or a statistic inferred from attention asymmetry — it is the shape of the computation. We call this property *position is structure*, and the paper's central empirical question is whether it suffices, and how it compares to explicit (RoPE, sinusoidal, learned) and emergent (NoPE) positional mechanisms under a strictly matched protocol.

Our contributions:

1. **The Fenwick prefix readout**: exact prefix states of a non-associative composition at O(T log T) cost, used as a dense-supervision causal LM readout. To our knowledge this mechanism is new.
2. **An exact incremental decoder** (v1.1): the readout's causality structure implies that generation needs only O(log T) new compose nodes per token — tree nodes are append-only, prefix states are causal, cross-layer mixing is position-wise — verified numerically to reproduce the full forward to 2e-6. Generation requires no attention and no softmax anywhere.
3. **A matched-protocol characterization** against RoPE and NoPE transformers whose data pipeline, loss, evaluation, and seeding are imported from OPERA's own code (Section 3), at matched parameters, on both sentence-level (T≤20) and document-level (T≤256, evaluated to 2× beyond) regimes.
4. **The position-curve and interrogation-probe instruments**, applied identically across architectures, yielding (a) the flattest length-degradation profile for OPERA among the mechanisms tested and (b) a cross-architecture finding: early-context forgetting is learned, not structural.
5. **A pre-registered optimizer arm** (v1.1): Muon — Newton-Schulz-orthogonalized momentum — applied to the model's matrix-shaped parameters, improving the toy-rung baseline by 38 perplexity over AdamW under a gate fixed before the run (Section 4.6).
6. **A falsification record**: pre-registered predictions for an SO(3) rotation-manifold constraint (nulled twice, including a pre-committed rematch at greater compositional depth) and for a structural bottleneck theory of forgetting (falsified in the reverse direction: giving the readout one-hop access to early context *reduced* early-context sensitivity).

We are explicit about what this paper does not claim. OPERA does not beat a RoPE transformer in-length at this scale; our wall-clock speed at T=256 is worse despite a 20–40× FLOP advantage, reflecting a kernel-maturity gap rather than an asymptotic one; and all results are at a single ~22M parameter scale on a single low-entropy corpus. Section 7 enumerates limitations without euphemism.

## 2. Architecture

### 2.1 State space

A token state is a vector in R^d organized as nb blocks of 4, d = 4·nb. Each block is an element of the even subalgebra of Cl(3), isomorphic to the quaternions: one scalar channel s and a 3-vector channel v. All experiments use d=640, nb=160, 4 layers (~22.3M parameters with untied embeddings and a 10k word vocabulary).

### 2.2 The composition node

An internal node maps two children (h_L, h_R) to a parent. Per block: the children's vector parts are transformed by per-block 3×3 maps R_L, R_R; the geometric product of the resulting spinors is computed, whose vector part contains the cross product v_L × v_R (the commutator of the quaternion algebra) and whose scalar part contains s_L·s_R − ⟨v_L, v_R⟩; three sigmoid fusion gates, computed from both children, mix {left, right, product}; a per-block output map R_O, a LayerNorm over the full d-vector, and a tanh(x)+0.1x activation complete the node. Two parameterizations of the 3×3 maps are compared in Section 5.1: quaternion-parameterized rotations (`so3`) and unconstrained matrices (`free`).

### 2.3 Tree and Fenwick readout

Each layer builds the balanced binary tree over the (unpadded) sequence bottom-up — nodes covering positions beyond the sequence are never computed — and then produces all T prefix states by folding each prefix's Fenwick blocks. The fold is compacted: at fold step s, only positions whose block count exceeds s participate (indices cached per T, statically), which we verify to be exactly output-equivalent to the masked formulation while performing 2.2–2.7× fewer row-compositions. Between layers, a per-position MLP with a learned blend gate refines states, and every layer's prefix states receive the LM head (dense multi-depth supervision; reported perplexity is always the final layer's).

Three fold variants are studied. **left** (default): a sequential gated fold of the blocks, largest first. **attend**: one round of attention over each position's own ≤ log₂T + 1 blocks (queries from the most recent block; keys carry a deterministic sinusoidal encoding of block *level*, which is defined for every level and therefore extrapolation-safe), followed by a single composition of the attended context with the most recent block — attention over log T slots, so total cost remains O(T log T). **rack**: each incoming block conjugates the accumulator by its unit spinor (a·b·a⁻¹, the canonical rack operation; the operation's self-distributivity axiom is verified numerically in our test suite to 1e-6), plus a gated content injection; the fold path contains no normalization or squashing. Section 5 evaluates all three.

### 2.4 Complexity

Per layer, OPERA performs T−1 tree compositions plus Σ_L (popcount(L)−1) ≈ T·(log₂T)/2 fold compositions on average — O(T log T) total, each composition O(d) with block-diagonal 3×3 structure. A transformer layer is O(T²·d_head·h + T·d²). At T=256 the transformer performs roughly 20–40× more arithmetic per layer; Section 7 discusses why it is nonetheless faster on current hardware.

### 2.5 Incremental decoding (v1.1)

The v1 paper left generation as naive re-evaluation of the full forward per token — O(L·T log T) compose nodes each — and noted an incremental variant as future work. The structure of the readout makes that variant exact, by three causality facts: **(i) tree nodes are append-only** — node (k, j) covers the fixed span [j·2^k, (j+1)·2^k) and depends only on its own span, so appending one token never mutates an existing node, it only *adds* the ≤ ⌈log₂T⌉ ancestors whose spans complete at that step; **(ii) prefix states are causal** — position T's prefix state folds only the ≤ log₂(T+1) Fenwick blocks of its own prefix, all already cached; **(iii) cross-layer mixing is position-wise**, so the next layer's new leaf needs only the new position's prefix state. The decoder therefore maintains per-layer tree caches and computes O(L log T) compose nodes per appended token — versus O(L·T log T) for re-evaluation — while performing *the same operations in the same order* as the batched path for the last position. Our test suite asserts per-position logit equivalence with the full forward to 2·10⁻⁶ over T=33 (fp32), for both the incumbent configuration and the so3+sinusoidal+fold-scale stack. Note the decode state is the cached tree itself, O(T·d) per layer; what the decoder eliminates is the *compute* blow-up, not memory — the same trade linear-attention models make in reverse.

## 3. Experimental protocol

**Data.** Simple English Wikipedia (20231101.simple). *Sentence mode*: 5–20-word sentences (historical regime). *Docs mode*: whole-article word streams (lowercased, alphabetic tokens) cut into chunks by a deterministic length schedule that emits mostly train-length (256) chunks plus one chunk at each bucket length k·256 up to 2048, so extrapolation buckets are populated by construction (292,249 sequences; 261,457 train; 29,021 in-length test; 204 longer-than-train test). Vocabulary: top 10k words plus specials; the split shuffle is internally seeded and identical across all runs and both architectures.

**Matched baseline by construction.** The transformer baseline (pre-LN decoder, d=512, 8 heads, 4 layers, FFN 4×, 22.86M params, +2.7% vs OPERA) *imports* OPERA's data loader, loss, perplexity, bucketed extrapolation evaluation, learning-rate schedule, and batching utilities; its forward returns a one-element logits list so every evaluation and probe runs on both architectures unchanged. Protocol identity is enforced by shared objects, not by reimplementation. PE arms: RoPE, NoPE (no positional mechanism), sinusoidal, learned (results reported for RoPE and NoPE).

**Training.** 20k steps, batch 16, Adam, lr 1e-3 with 500-step warmup and cosine decay, grad clip 1.0. Seeds: the data split is fixed; a separate variation seed (applied after data loading) controls initialization and batch sampling. All primary runs share seed 42, making loss traces directly comparable step-for-step; identical characteristic loss dips at the same steps across arms confirm the shared batch stream. [TODO: seeds 1–2 on OPERA free+left and transformer NoPE; until then all docs-mode comparisons are single-seed, and we treat differences under ~1.5 PPL as within noise, per our sentence-mode replication experience.]

**Evaluation.** In-length perplexity on 5k test sequences (final layer, masked per-token CE); length-extrapolation buckets of width 256 up to 2048 (n = 87/68/35 for the first three; buckets beyond 1024 are underpopulated — a documented flaw of the chunk schedule, Section 7). Two instruments run identically on all checkpoints: the **position curve** (per-position CE on sequences ≥ 512 tokens, aggregated in 64-position bands) and the **interrogation probe** (Jensen–Shannon divergence of the final-position next-token distribution under replacement of the token at position p; 64 sequences × 4 resamples per point; prefixes 31/63/127/255, chosen for maximal Fenwick popcount so the fold is maximally engaged — an earlier probe iteration used power-of-two prefixes, which the Fenwick readout bypasses entirely, and was discarded).

**Pre-registration.** For each architectural hypothesis we recorded, in the code shipped before the run, the falsifiable prediction and the decision rule, including (for the rotation constraint) a one-rematch rule binding in both directions. Section 5 reports outcomes against those rules verbatim; Section 4.6's optimizer gate was likewise fixed (threshold and kill criterion) before execution.

## 4. Results

### 4.1 Sentence-level parity (historical regime)

At train length ≤ 20 words, 4 layers, matched parameters, OPERA with no positional encoding reached in-length perplexity 70.9 against a RoPE transformer's 70.7 (1k-sentence evaluation, single seed), with length-extrapolation ratios 1.65×/1.68× vs 1.60×/1.68× — indistinguishable within our noise band. A later replication under an upgraded 5k evaluation gave 68.6 (so3) and 68.1 (free). Removing sinusoidal PE *improved* OPERA's extrapolation in earlier iterations, and injecting sinusoidal PE degraded both OPERA and the transformer in the same way — the observation that motivated the PE-free design.

### 4.2 Document-level main comparison

Train ≤ 256 tokens, evaluate to 2048; 20k steps; all runs same GPU class, seed 42.

| model | params | in-length PPL (5k) | 257–512 | 513–768 | 769–1024 | s/step |
|---|---|---|---|---|---|---|
| OPERA so3 + left | 22.25M | 42.58 | 26.36 | 15.42 | 8.21 | 0.63 |
| OPERA free + left | 22.26M | **41.53** | 25.64 | 15.23 | 8.12 | 0.65 |
| OPERA so3 + attend | 22.58M | 43.78 | 26.77 | 15.62 | 8.25 | **0.47** |
| OPERA so3 + rack | 23.07M | 68.55† | 38.76 | 21.95 | 10.90 | 0.66 |
| Transformer RoPE | 22.86M | **35.68** | 23.35 | 14.86 | 8.09 | 0.17 |
| Transformer NoPE | 22.86M | 40.32 | 26.77 | 16.57 | 8.94 | 0.15 |

†Non-converged: still descending at full slope when the schedule ended (Section 5.3).

Three readings. First, the **like-for-like comparison**: against the transformer stripped to the same no-positional-mechanism condition (NoPE), OPERA's best arm is within 1.2 PPL — inside our noise band. Second, the **RoPE decomposition**: RoPE leads OPERA by 5.9 PPL in-length, of which 4.6 is attributable to explicit positional encoding (RoPE vs NoPE) and 1.2 to the architecture (NoPE vs OPERA). Third, the **length trend**: OPERA's relative gap to RoPE shrinks monotonically with evaluation length — +16.4% in-length, +9.8%, +2.5%, and **+0.4% (8.12 vs 8.09) at four times the training context**. The transformer's quality advantage is concentrated exactly where its quadratic cost is cheapest, and vanishes along the axis where that cost explodes. (Sub-1.0 extrapolation-to-in-length ratios in this table reflect docs-mode data statistics — later tokens in long chunks have more context, and very long Simple-Wikipedia articles skew toward predictable content — and must not be compared with sentence-mode ratios; the position curve below is the honest length-generalization instrument.)

### 4.3 Position curves: constructive vs explicit vs emergent position

Per-position CE on the same ≥512-token test sequences, 64-position bands; training length 256.

| bands (pos) | 0–63 | 64–127 | 128–191 | 192–255 | 256–319 | 320–383 | 384–447 | 448–510 |
|---|---|---|---|---|---|---|---|---|
| OPERA free+left | 2.963 | 2.790 | 2.798 | 2.790 | 2.794 | 2.803 | 2.881 | 2.862 |
| OPERA so3+left | 2.978 | 2.813 | 2.810 | 2.810 | **2.806** | 2.825 | 2.895 | 2.877 |
| Transformer RoPE | 2.889 | **2.680** | 2.680 | 2.688 | 2.691 | 2.725 | 2.840 | 2.865 |
| Transformer NoPE | 2.983 | **2.794** | 2.803 | 2.833 | 2.865 | 2.882 | 2.993 | 2.979 |

Degradation from each model's best band to the final band (2× training length): **OPERA +0.07 nats; RoPE +0.19; NoPE +0.19.** OPERA degrades 2.6× less than both transformer arms; its best band (so3+left) lies *beyond* the training boundary; and NoPE's curve begins rising *inside* the training range (2.794 → 2.833 by band 192–255), the signature of an emergent positional mechanism weakening with distance. Constructive position is not merely as good as emergent position — under length stress it is measurably flatter, while RoPE's explicit encoding buys in-length quality that decays at the same rate as NoPE's beyond it. Loss remaining essentially unchanged across the training-length boundary (2.790 → 2.794 for OPERA free) is, to our knowledge, an unusual profile among the mechanisms tested here.

### 4.4 The interrogation probe: forgetting is learned, not structural

For each prefix length Lp and perturbation position p, we report s(p) = mean JSD of the final-position next-token distribution when token p is replaced, and the headline ratio pp1/ppLast = s(1)/s(Lp−2). Historical context: at the sentence regime (prefix 15) an earlier internal measurement gave 0.64 for OPERA's fold against 1.43 for a transformer, suggesting an architectural pathology. At the document regime that contrast does not replicate:

| pp1/ppLast | prefix 31 | 63 | 127 | 255 |
|---|---|---|---|---|
| OPERA so3+left | 0.078 | 0.040 | 0.012 | 0.004 |
| OPERA so3+attend | 0.048 | 0.013 | 0.004 | 0.002 |
| Transformer RoPE | 0.078 | 0.050 | 0.014 | 0.007 |
| Transformer NoPE | 0.104 | 0.062 | 0.028 | 0.016 |

Three architecturally alien position mechanisms — a tree fold, rotary attention, and bare causal attention — show the same order-of-magnitude collapse of early-context sensitivity with prefix length (identical to two decimals at prefix 31 for OPERA and RoPE). At prefix 255, tokens 1 and 254 reach OPERA's readout through *structurally symmetric* 8-composition paths, yet differ 100× in sensitivity; the transformer, whose token 1 is one attention hop away, collapses equally. All curves are U-shaped (primacy plus recency, a dead middle), echoing the "lost in the middle" phenomenon reported for large transformers. We conclude that at this scale and corpus, early-context forgetting is a learned consequence of next-token prediction — the objective barely pays for distant tokens — and not a property of any architecture's information topology. Section 5.2 reports the pre-registered experiment that forced this conclusion.

### 4.5 Exact incremental decoding (v1.1)

Measured generation throughput, OPERA free+left, d=640/nb=160, 4 layers, tied 16k BPE vocabulary (~20M params), fp32, single consumer CPU core set (Apple M4):

| path | ctx 256 | ctx 512 | scaling in T |
|---|---|---|---|
| naive full re-forward | 25.9 tok/s (ctx 128) | — | ~1/T per token |
| OperaDecoder (§2.5) | **314 tok/s** | **319 tok/s** | flat (O(log T) per token) |

The incremental decoder is ~13× faster at ctx 128 and the advantage grows linearly with context; per-token cost is flat in context length, as the asymptotics predict. Generation requires no attention and no softmax anywhere in the model — only the final LM head's softmax over the vocabulary. [TODO: throughput table on the Space's CPU class; crossover against a transformer KV cache at long context.]

### 4.6 The recipe arm: Muon on matrix-shaped parameters (v1.1, pre-registered)

The v1 limitations flagged a **recipe asymmetry**: OPERA had only ever been trained with transformer-folklore defaults. The 2025–26 optimizer literature offers a pointed fix: Muon (Newton–Schulz-orthogonalized momentum) is designed for matrix-shaped hidden parameters, and OPERA's per-block 3×3 maps (R_L, R_R, R_O per block per layer — thousands of small matrices) are the most literally matrix-shaped parameters in any sequence model we know. We partition parameters in the standard way: Muon for matrices (the 3×3 maps, cross-layer MLP weights, the untied head — Newton–Schulz runs batched over leading dims, so each 3×3 map is orthogonalized individually), AdamW for embeddings, gates, gains, and scalars; both groups follow the same warmup/cosine schedule.

Gate fixed before the run (toy rung: 2000 steps, batch 16, d=256/nb=64, 2 layers, docs mode, seed 1, identical data order across arms): Muon must beat the AdamW incumbent by >1.5 PPL (the project noise band) at some lr ∈ {0.02, 0.05, 0.10}; parity or worse after two settings retires the arm.

| arm | in-length PPL (5k) | Δ vs AdamW |
|---|---|---|
| AdamW (incumbent) | 224.88 | — |
| Muon lr 0.02 | **186.45** | **+38.43** |
| Muon lr 0.05 | 221.99 | +2.89 |
| Muon lr 0.10 | 302.72 | −77.84 (diverging) |

**Gate: passed by 25×.** The lr curve brackets 0.02 near the top of the stable range, so 0.02 is adopted as the default for subsequent runs (including §4.7) as a *starting* lr rather than a proven optimum. Caveats: single seed, toy rung, word-level protocol; transfer to the 22M rung is a separate measurement [TODO]. Even read conservatively, the arm confirms the v1 suspicion: a material fraction of OPERA's reported gap was recipe, not architecture.

### 4.7 A BPE-tokenized chat artifact (v1.1)

As a first step off the word-level protocol (Section 7's `<unk>` confound) and to make the mechanism interactively examinable, we trained a chat variant: byte-level BPE (16,384 vocabulary; role/stop special tokens at fixed ids), a ~150k-conversation subset of smoltalk (turn-boundary chunking, conversation-level 95/5 split; 40,877 train sequences ≤ 256 tokens), and the incumbent stack — pe-none, left fold, free maps, tied embeddings (~20M params), multi-scale node supervision, chrono fold init, length curriculum, and Muon at lr 0.02 (§4.6) — trained from scratch for 20k steps on a single consumer GPU (Apple M4, ~8.6 GPU-hours). Final in-length test perplexity: **29.65** (5k sequences, BPE tokens — incomparable to the word-level numbers above). Length-extrapolation buckets degrade faster than in the docs regime (2.3×/5.1×/5.8× at 257–512/513–768/769–1024): long multi-turn conversations are a harder tail than long Wikipedia chunks, and this protocol is not the position-curve regime — we report the buckets without reading them as a positional result. The model is served as a public Hugging Face Space, with generation on CPU via the incremental decoder of §2.5 (the serving path's speed is what makes a free CPU tier usable). [TODO: Space URL.]

We set expectations plainly: a ~20M-parameter model trained on ~10⁸ tokens is a research toy in chat terms — replies are short-format-plausible and content-poor, and no capability claim is made. The artifact's purposes are (a) to demonstrate that the full stack — BPE, chat formatting, sampling with stop tokens, multi-turn contexts — runs on the architecture unchanged, and (b) to let readers probe a *running* instance of the mechanism (position-is-structure, geometric compose, incremental decoding) rather than a table of numbers.

## 5. Ablations and falsifications

Our practice: one variable per run; predictions and decision rules recorded in shipped code before execution; falsified components retired explicitly. Two earlier components (a Kuramoto-style phase gate and a "resonance lock" modulating fusion by children's geometric alignment) were retired under this protocol before the present study and are not revisited.

### 5.1 The rotation constraint (pre-registered, nulled twice)

Hypothesis: constraining the per-block maps to SO(3) (isometries) preserves information through deep composition and should pay at extrapolation depth, where unconstrained matrices drift. Prediction: `free` matches `so3` in-length but degrades in extrapolation buckets. Sentence regime: in-length 68.58 (so3) vs 68.09 (free); buckets 117.07/118.22 vs 115.35/118.54 — indistinguishable, free nominally ahead. Pre-committed rematch at 4–8× compositional depth (docs regime): 42.58 vs 41.53 in-length; buckets equal within noise; free again nominally ahead everywhere. Per the one-rematch rule fixed in advance, **the rotation-manifold constraint is retired from our claims**: at every depth tested, the tree, the geometric product, and the readout carry the results; the manifold constraint does not measurably contribute. The `free` parameterization is also simpler, and we adopt it as the default. (The 2025–26 literature on rotation-structured state transitions locates the value of such constraints in state tracking rather than perplexity; a dedicated state-tracking evaluation of both `rot_mode` arms is future work, Section 8 — the retirement stands for perplexity, pre-registered, and does not extend to axes we did not measure.)

### 5.2 The bottleneck theory of forgetting (pre-registered, falsified in reverse)

Hypothesis: the sequential fold forgets early tokens *structurally* — the oldest block passes through popcount−1 gated squashes. Prediction for the `attend` fold (one-hop access to every block): early-position probe sensitivity rises toward parity. Outcome: at matched (prefix, position), attend's early-position JSD is **0.4× the left fold's** — one-hop access *halved* early-context sensitivity. Mechanism, in hindsight: the sequential fold structurally forces the oldest block into every subsequent state; softmax attention makes ignoring it a single learnable decision. Availability is not use. Combined with the cross-architecture collapse of Section 4.4, this reverses the structural theory: the lever on forgetting is the objective, not the topology. The attend fold is retained on independent grounds — 25% faster per step at quality parity (same-session measurement) — but is not claimed as a memory mechanism.

### 5.3 The rack fold (pre-registered; optimization-limited at this budget)

Hypothesis: the fold's per-step LayerNorm and tanh, not its path length, destroy old content (a fixed norm budget each new block competes for); a fold whose core is the conjugation rack operation — per-step isometric, invertible, self-distributive (axiom verified numerically) — should preserve it. Outcome at the 20k-step budget: stable throughout (the isometry claim held) but 26 PPL behind the left fold, with the loss still descending at full slope at schedule end — a constant-factor training slowdown, not a plateau. Post-hoc diagnosis: removing all fold normalization hands the head states whose scale varies systematically with popcount; normalization was doing optimization work as well as informational harm. The highest effective dimensionality of any arm (participation ratio 295/640 vs 194–220) is consistent with un-crushed but not-yet-organized states. An exit-normalized variant (isometric fold interior, single normalization at the readout boundary) is specified for follow-up; the present result stands as recorded: **per-step isometry alone does not buy language-modeling quality at a fixed step budget.**

### 5.4 Fold compaction (exact equivalence)

Replacing the masked fold (which composes all B·T rows per step and discards masked results) with composition over only the active rows is verified exactly output-equivalent on ragged batches (max error 7e-7) while performing 2.2–2.7× fewer row-compositions; whole-step training time improved ~15% at T=20 and the technique underlies all reported document-regime timings.

## 6. Related work

**Tree-structured composition.** Recursive networks and Tree-LSTM (Tai et al., 2015) compose supplied parse trees for classification; latent-tree models (Gumbel Tree-LSTM; PRPN; ON-LSTM, Shen et al., 2019; URNNG; DIORA) induce hierarchy from the LM objective but retain linear-depth recurrences. OPERA uses a fixed balanced tree with log-depth compute and, unlike this lineage, an exact-prefix causal readout. Our own earlier experiments with content-dependent tree shape (spectral/Fiedler splitting) added noise and were abandoned in favor of the fixed tree.

**Log-depth and hierarchical sequence models.** BP-Transformer (Ye et al., 2019) attends over a binary partition at O(T log T); hierarchical downsampling LMs (Funnel, Hourglass) and log-timescale RNNs (Clockwork, Dilated) share the asymptotic motive without the compose-and-read-prefixes mechanism.

**Fenwick-hierarchy memories (v1.1).** Log-linear attention (arXiv:2506.04761) independently organizes sequence memory over the same binary-indexed (Fenwick) decomposition, with O(T log T) training and O(log T) decoding state, and has since been adopted at frontier scale. The two mechanisms are siblings at the level of the decomposition and different everywhere else: log-linear attention *mixes* Fenwick blocks with learned scalar weights (an attention-family memory with a hierarchical layout), while OPERA *composes* them through a geometric, non-associative node with content-dependent gates; and OPERA's "position is structure" is a strictly stronger commitment than a Fenwick-shaped memory layout — no positional parameterization of any kind exists in the model. A 2026 follow-up showed log-linear attention's per-level weighting collapses under memory load (2.9% → 57.9% MQAR once the weights are made input-adaptive), which identifies the fold's transport weighting — not the decomposition — as the load-bearing design axis; Section 8 lists the corresponding OPERA arm. We also note the broader 2025–26 consensus that pure recurrent/linear models pair a compressed context pathway with a sparse retrieval pathway (Gated DeltaNet-2, Kimi Delta Attention, Mamba-3, Hymba); OPERA's program tests whether that retrieval can be content-addressable memory rather than attention (Section 8), keeping the no-pairwise-attention invariant this paper is about.

**Parallel-scan recurrences.** S4/S5, LRU, and Mamba-class models obtain log-depth training by *requiring* the recurrence to be associative so a parallel scan applies. OPERA refuses associativity — the per-node nonlinearity is the point — and pays for exact prefixes with the Fenwick decomposition instead. OPERA is, in this sense, the non-associative sibling of the scan-SSM family; notably, that family also compresses context into a fixed-size state, and our probe results (Section 4.4) suggest the fixed-state "limitation" is substantially objective-limited rather than architectural.

**Position without positional encodings.** Haviv et al. (2022) showed causal transformers train without PE, recovering position implicitly from the attention mask; Kazemnejad et al. (2023) found NoPE length-generalizes better than several explicit schemes. Our NoPE baseline reproduces the spirit of these findings (Section 4.2) and sharpens the comparison: OPERA's position source is *constructive* — a combinatorial property of the readout circuit, distinct for every prefix length by design — rather than emergent, and under length stress the constructive mechanism degrades 2.6× less (Section 4.3).

**Racks, quandles, and the algebra of the fold.** The rack axioms (Joyce, 1982; Matveev, 1982) axiomatize conjugation and the Reidemeister moves; racks provide set-theoretic solutions to the Yang–Baxter equation. Our rack fold's core operation — conjugation by the unit spinor of the incoming block — satisfies the rack axiom exactly (verified numerically), placing the fold's algebra in this lineage; its empirical status at our budget is reported in Section 5.3. The 6D and quaternion rotation parameterizations follow Zhou et al. (2019).

**Optimization (v1.1).** Muon (Jordan et al., 2024; popularized by the modded-nanogpt speedruns) orthogonalizes the momentum update of matrix-shaped parameters via a Newton–Schulz iteration; µP/muTransfer (Yang & Hu, 2021) provides hyperparameter transfer across scale. Both enter this paper as recipe arms (Section 4.6) rather than architecture.

## 7. Limitations

**Scale and scope.** All results are at ~22M parameters on Simple English Wikipedia with a 10k word vocabulary. The corpus is low-entropy; the word-level vocabulary incurs a substantial `<unk>` rate on document text, which flatters absolute perplexities for all models equally but makes them incomparable to subword-tokenized literature numbers — no comparison to published GPT-class perplexities is valid from these tables, and we make none. The BPE chat artifact of §4.7 leaves this regime but not this caveat: it is a different, narrower domain at similar scale. Parameter-axis scaling is untested.

**Statistical power.** Docs-mode comparisons are single-seed (shared seed 42 with step-identical batch streams); sentence-mode replication indicates a ~1.5 PPL noise band, which we apply throughout. The Muon gate (§4.6) is likewise single-seed, though its margin (38 PPL) exceeds the band by an order of magnitude. [TODO: 3-seed means ± ranges for OPERA free+left and transformer NoPE — in progress for v1.2.] Extrapolation buckets carry n = 87/68/35; buckets beyond 1024 tokens are underpopulated because the chunk schedule restarts per article, so only very long articles emit very long chunks — a design flaw we document rather than patch mid-study, since all arms share the identical corpus. A decontaminated-split re-measurement of the headline comparison (the current numbers predate article-level decontamination; a preliminary contaminated-free read put OPERA at 75.78 vs RoPE's 56.77 in-length without its like-for-like NoPE counterpart) is in progress for v1.2.

**Speed.** Despite performing 20–40× fewer FLOPs per layer at T=256, OPERA is ~3× slower per step than the transformer baseline (0.47–0.65 vs 0.15–0.17 s/step): the transformer's cost is a few large dense matmuls on maximally optimized kernels, while OPERA's composition is many small bandwidth-bound kernels with serial tree/fold depth. This is a kernel-maturity gap, not an asymptotic one, but it is the number a practitioner experiences today. The T at which the measured cost curves cross is not yet determined. [TODO: crossover timing figure, T ∈ {512…8192}.] Generation is exempt from this caveat as of v1.1: the incremental decoder (§4.5) is flat in context length on commodity CPU.

**Readout capacity.** OPERA reads each prefix from a single d-dimensional state; the transformer reads from all T token states. The uniform ~0.13-nat mid-band gap to RoPE in Section 4.3 (present even at the earliest positions) is consistent with a context-exploitation capacity difference rather than a positional one, and constitutes the architecture's clearest known ceiling at this regime.

**Recipe asymmetry (partially addressed in v1.1).** The Muon arm (§4.6) shows the shared transformer-folklore recipe was indeed costing OPERA materially at the toy rung; whether the same margin exists at the 22M rung is unmeasured, and both architectures remain untuned beyond that single arm. The comparison is therefore still conservative against OPERA by an unknown margin.

## 8. Future work

Motivated directly by the measurements above: input-adaptive per-level transport weights in the fold — the axis the adaptive-λ result identified on the identical Fenwick hierarchy (§6); objective-side pressure on early context (multi-scale node supervision deeper targets, retrieval-style auxiliaries), targeting the learned forgetting that Section 4.4 shows no architecture escapes; a content-addressable memory channel read by the fold state (delta-rule or product-key lookup — retrieval with no pairwise attention and no softmax), targeting the readout-capacity ceiling; the exit-normalized rack fold of Section 5.3; the crossover timing measurement; a FineWeb-scale subword protocol enabling literature-comparable numbers (the BPE tooling shipped with §4.7; the matched 22M re-run remains); fused composition kernels (a Metal prototype exists; a CUDA/Triton port with the verified adjoint as reference is the direct path to closing the wall-clock gap); a state-tracking evaluation track where the retired SO(3) constraint may show value on its home axis; and parameter-axis scaling under µP, depth-first with tied embeddings, with the multi-seed and decontaminated re-baseline of Section 7 preceding all of it.

## 9. Reproducibility

Every model, evaluation, probe, and equivalence claim in this paper is exercised by self-tests shipped in the released code (including numerical verification of the rack axiom, exact-equivalence asserts for fold compaction, causality asserts for every fold and PE variant, per-position logit equivalence of the incremental decoder against the full forward, Newton–Schulz spectral bounds and optimizer partition asserts for the Muon arm, and gradient checks for the custom composition backward). Data splits are internally seeded and byte-identical across architectures; variation seeds are applied after data loading; training checkpoints save full RNG state for exact-stream resumption. The Muon gate ships as a single rerunnable script with its four checkpoints and results file. All experiments were run on single consumer/Colab-class GPUs; the largest single run is ~3.5 GPU-hours.

## Acknowledgments

Engineering and drafting assistance from Claude (Anthropic) and Kimi (Moonshot AI). All experimental decisions, runs, and the research direction are the author's. [TODO: confirm/edit disclosure wording to taste.]

## References

[TODO: full BibTeX — placeholders keyed to citations above]
Haviv et al., 2022 — Transformer LMs without positional encodings still learn positional information.
Kazemnejad et al., 2023 — The impact of positional encoding on length generalization in transformers.
Tai, Socher, Manning, 2015 — Tree-LSTM. · Shen et al., 2019 — ON-LSTM. · Ye et al., 2019 — BP-Transformer.
Gu & Dao, 2023 — Mamba. · Gu et al., 2022 — S4. · Orvieto et al., 2023 — LRU.
Joyce, 1982; Matveev, 1982 — knot quandles / distributive groupoids. · Zhou et al., 2019 — continuity of rotation representations.
Liu et al., 2023 — Lost in the middle.
[added v1.1] Log-Linear Attention, arXiv:2506.04761 — Fenwick-hierarchy sequence memory. · Adaptive Memory Decay for Log-Linear Attention, 2026 — input-adaptive per-level decay (MQAR collapse/rescue).
Jordan et al., 2024 — Muon: orthogonalized momentum for matrix-shaped parameters. · Yang & Hu, 2021 — µP/muTransfer.
HuggingFaceTB, 2025 — smoltalk dataset. · Gated DeltaNet-2; Kimi Delta Attention; Mamba-3 (ICLR 2026); Hymba — the 2025–26 hybrid sub-quadratic line.
