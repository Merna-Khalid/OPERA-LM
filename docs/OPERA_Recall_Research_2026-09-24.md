# Why OPERA stops using context after ~100 bytes — research before the next experiment (2026-09-24)

Written after a day of experiments (`OPERA_Speed_and_Design_2026-09-24.md`
§4.5–4.11) in which ten interventions failed to move the ~128-byte
plateau of the captured-information curve. This note asks the question
that should have come first: **is the horizon specific to OPERA, and what
does the literature say actually breaks it?** No experiments were run for
it. Standing constraint: no attention.

---

## 1. The horizon is not OPERA-specific

- **Our own paper already showed it** (v1.3 §4.4, the interrogation
  probe): at 22M parameters, OPERA, a RoPE transformer and a NoPE
  transformer show "the same order-of-magnitude collapse of early-context
  sensitivity" (identical to two decimals at prefix 31 for OPERA and
  RoPE). Conclusion there: early-context forgetting is "a learned
  consequence of next-token prediction … not a property of any
  architecture's information topology." Today's MI-curve plateau is the
  same phenomenon measured with a different instrument.
- **Khandelwal et al., ACL 2018** — *Sharp Nearby, Fuzzy Far Away*
  (arXiv 1805.04623). LSTM LMs use about 200 tokens of context; word
  order matters only within roughly the last 50 words; beyond that the
  past acts as a topic ("fuzzy"). A neural cache model is what lets the
  LSTM copy words from distant context.
- **Sun et al., EMNLP 2021** — *Do Long-Range Language Models Actually
  Use Long-Range Context?* (arXiv 2109.09115). For long-range transformer
  LMs (up to 8K tokens), context beyond the previous 2K tokens improves
  predictions only on a small set of tokens — mainly those that can be
  copied from the distant context.

OPERA's ~100-byte onset is about 20 words: squarely in the "sharp
nearby" regime every small LM shows.

## 2. What breaks the regime: associative recall

- **Olsson et al., 2022** — *In-context Learning and Induction Heads*.
  Transformer LMs undergo a phase change early in training in which
  induction heads form and in-context learning — measured by the ICL
  score, loss at the 500th token minus loss at the 50th — improves
  dramatically. Induction heads implement copying: find an earlier
  occurrence of the current token, output what followed it. They need
  attention heads composed across ≥ 2 layers (they do not occur in
  1-layer models).
- **Arora et al., 2023** — *Zoology* (arXiv 2312.04927). Gated-
  convolution (attention-free) architectures underperform attention by
  up to 2.1 perplexity on the Pile, and **82% of that gap is explained by
  associative recall** — retrieving information mentioned earlier in
  context. A 70M attention model beats a 1.4B gated-convolution model on
  multi-query associative recall (MQAR).
- **Fu, Dao et al., ICLR 2023** — *Hungry Hungry Hippos* (arXiv
  2212.14052). Earlier SSMs could not do associative recall; H3 added it
  without attention using a shift SSM (to hold past tokens) and a
  multiplicative interaction between the SSM output and the current
  input "measuring similarity between tokens" — and came within 0.4
  perplexity of transformers on OpenWebText.

**Reading:** the function that lets a model use distant context is
recall, and every mechanism that provides it compares current content
with stored past content — softmax attention, linear attention / delta
rule, H3's multiplicative interaction, a cache, exact-match lookup. None
of the ten interventions tested on 2026-09-24 added a recall mechanism;
they changed capacity (state width, node mixing), allocation (innovation
fold, counterclockwise fold, disentangler), the objective (future
content, far-repeat weighting), the input (byte dropout), the optimizer
(SAMuon), or training length. That is why none moved the onset. It also
fits the measured facts: the tree blocks hold old single-copy content
(4+ bits) and the fold readout holds 0 — the readout never compares the
current context with what the blocks contain.

## 3. The consequence for the no-attention rule

Recall requires comparing the current context with stored content.
Under a rule that excludes every form of content comparison, OPERA
cannot do in-context recall by construction, and — per Zoology — most of
the raw-text gap to transformers remains. (A hypothesis consistent with
the paper's 155M results: OPERA won on chat-only data and lost on
raw-text pretraining; raw web text may reward recall more. Untested.)

So the real decision is not "which fold" but **which form of content
comparison, if any, is acceptable.** The forms, from most to least
attention-like:

| form | how it compares | attention? |
|---|---|---|
| softmax attention | scores every past position against a query | yes — excluded |
| linear attention / delta rule | outer-product memory read by a query vector | linear attention — excluded (2026-09-24 rule) |
| H3-style multiplicative interaction | elementwise product of a running past summary with the current input | borderline — formally a (kernelized) linear attention |
| exact-match lookup (LZ77 / PPM / cache) | equality of hashed n-grams | no scores; deferred as a possible "trick" |
| **geometric binding / unbinding (below)** | algebraic inverse in the model's own Clifford algebra | no scores, no softmax; content comparison through the algebra |

## 4. An OPERA-native candidate: geometric holographic memory

- **Plate, 1995** — *Holographic Reduced Representations* (IEEE Trans.
  Neural Networks): key–value pairs are *bound* (circular convolution),
  *superposed* (addition) into one fixed-size vector, and *unbound*
  (correlation with the key) to retrieve the value approximately. Many
  associations fit in one vector.
- **Aerts, Czachor & De Moor, 2009** — *Geometric Analogue of Holographic
  Reduced Representation* (J. Math. Psychology; arXiv 0710.2611):
  replace convolution with the **geometric product** — binding is the
  geometric product, superposition is addition, unbinding multiplies by
  the inverse.

This is OPERA's native algebra: the state is a set of quaternions
(even Cl(3)), the compose node already contains the geometric product,
and the repo already has the inverse (`quat_quotient`, h_L ⊗ h_R⁻¹, the
never-properly-tested `node_paths=4` path). And the tree already forms
bindings: every level-1 node composes a byte with the byte that follows
it — the (A → B) pair an induction head retrieves.

**Sketch (not built):** each position writes key ⊗ value (key = a
projection of byte t−1, value = a projection of byte t) into its tree
leaf's memory channel; tree nodes and the fold superpose (sum) these
channels, which is associative and so fits the downsweep; the readout
unbinds with the current key (memory ⊗ key⁻¹) and passes the result to
the head through a zero-init gain (the `lowrank_gain` pattern, to avoid
the Muon trap).
- Honest caveats: (1) it is still content comparison — through the
  algebra rather than through scores — so it needs your judgment under
  the no-attention rule; (2) superposition memories have finite capacity
  (interference grows with the number of stored pairs), so capacity per
  level must be sized; (3) the sum of bindings is structurally close to
  linear attention's outer-product memory — the difference is the
  quaternion algebra and the absence of any learned query–key score.

## 4a. The wider family: binding memories and where the no-attention boundary falls

All of these store key→value associations by **binding** (a product in
some algebra), **superpose** them by addition, and retrieve by
**unbinding** with the key. They differ in the algebra — and in whether
the memory is a fixed-size vector (compressed, noisy) or an outer-product
matrix (exact, but that is linear attention).

**Fixed-size vector memories (vector symbolic architectures / hyperdimensional computing):**

| scheme | binding | unbinding | note for OPERA |
|---|---|---|---|
| HRR (Plate 1995) | circular convolution (real vectors) | circular correlation | the original; commutative → order needs permutations |
| FHRR (Fourier HRR) | elementwise complex multiplication (unit phasors, i.e. phase addition) | multiply by the complex conjugate | the complex analogue of what OPERA would do with unit quaternions |
| MAP / BSC | elementwise multiply (bipolar) / XOR (binary) | the same op (self-inverse) | hardware-oriented; less relevant |
| MBAT (matrix binding) | invertible matrix product | inverse matrix | non-commutative → encodes order and hierarchy |
| **GAHRR** (Aerts, Czachor & De Moor 2009) | **geometric product** | multiply by the inverse | **OPERA's own algebra** |

Survey: Kleyko et al., *A Survey on Hyperdimensional Computing aka Vector
Symbolic Architectures* (ACM Computing Surveys, arXiv 2111.06077);
comparison: Schlegel et al., *A comparison of Vector Symbolic
Architectures* (arXiv 2001.11797).

**Outer-product (matrix) memories — the linear-attention side of the boundary:**
- Tensor Product Representations (Smolensky 1990): role ⊗ filler outer
  products; exact, but the memory is a matrix.
- Fast weights / linear attention: *Linear Transformers Are Secretly Fast
  Weight Programmers* (Schlag, Irie & Schmidhuber, ICML 2021) makes the
  TPR → linear-attention lineage explicit.
- Phase-Associative Memory (Vishwakarma & Agostino, 2026, arXiv
  2604.05030): a complex d×d state of outer products read by a conjugate
  inner product — linear attention in complex space.
These are excluded under the 2026-09-24 rule; the vector-memory schemes
above are the candidates.

**Holographic memory already used inside sequence models:**
- **Associative LSTM** (Danihelka, Wayne, Uria, Kalchbrenner & Graves,
  ICML 2016, arXiv 1602.03032): an LSTM with a complex-valued holographic
  memory, no extra parameters; HRR's capacity limit (interference grows
  with stored items) is handled by storing **redundant copies** (under
  different permutations) and averaging on retrieval — "faster learning
  on multiple memorization tasks". The closest precedent: a fixed-state
  recurrent model given a holographic recall memory.
- **Hrrformer** (Alam, Raff et al., ICML 2023, arXiv 2305.19534):
  recasts self-attention with HRR at O(n) time and space; 23× faster and
  24× less memory than a transformer on very long sequences (> 100k).
  Shows HRR retrieval can do the job at scale — though framed as an
  attention replacement.

**Decoding noisy retrievals:** resonator networks (Frady, Kent, Olshausen
& Sommer, Neural Computation 2020) factor bound products by searching in
superposition. In an LM the head can act as the cleanup memory (it maps a
noisy retrieved value to a byte distribution).

**Design implications for an OPERA holographic memory (not built):**
1. Unit quaternions as the binding elements (FHRR generalized to
   quaternions): the inverse is just the conjugate — cheap and stable —
   and the repo already normalizes quaternions (`quat_sandwich`, rack
   fold).
2. Non-commutative binding encodes order for free: key ⊗ value ≠ value ⊗
   key, so "B followed A" is distinguishable without HRR's permutations.
3. Capacity: shard the memory across Fenwick blocks (each block holds
   only its span's bindings; unbind per block and combine), so capacity
   grows with the number of blocks; plus Associative-LSTM-style redundant
   copies across quaternion slots to cut interference.
4. Key = the preceding byte(s), value = the byte — the induction-head
   pattern (level-1 tree nodes already pair each byte with its successor).

## 5. What to do next — one step at a time

1. **Measure the recall deficit directly, before building anything**
   (standard synthetic, minutes to train): multi-query associative recall
   (MQAR, Zoology's benchmark) on the current OPERA base. Prediction: it
   fails once the number of key–value pairs or their distance grows. If
   it passes, §2's explanation is wrong and nothing in §4 should be
   built.
2. **Only if (1) confirms the deficit and the form is acceptable to you:**
   build the geometric holographic memory and test it on MQAR first (the
   cheapest, most direct test of recall), and on language modeling only
   if it solves MQAR. Success on LM = the captured-information curve
   keeps rising past ~128 bytes, and the ICL score (loss at byte 500 −
   loss at byte 50) improves.
3. Independently of recall, the one measured quality lever is training
   length (4× steps = −10% BPB), now ~5× cheaper per step.

## 5a. Result of step 1 — OPERA cannot do in-context recall (2026-09-24)

`experiments/mqar.py`: MQAR as in Zoology — N key→value pairs at the
start of a 256-token sequence, each key re-queried once later among
random filler; loss and accuracy only at query positions; key, value and
filler vocabularies disjoint (512 / 512 / 64). OPERA base (stream head +
downsweep + Metal), d256, Muon recipe, 4,000 steps, batch 64, N sampled
from {4, 8, 16, 32}; 512 held-out sequences per N.

| model | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| OPERA L=2 | 0.002 | 0.003 | 0.002 | 0.002 |
| OPERA L=4 | 0.003 | 0.003 | 0.002 | 0.002 |
| 1/N ("knows the in-context value set") | 0.250 | 0.125 | 0.062 | 0.031 |

- Accuracy is at chance (1/512) at every N and every key→query distance,
  including distances under 32 tokens — well inside the ~100 bytes the
  model demonstrably uses. The query loss converges to 6.24 = ln 512: the
  model learns only that a value follows a key, not which values occurred
  in the sequence, let alone which one is bound to the queried key.
- **Pipeline control** (`--control`: same code, model and sparse
  query-only loss; target = the token just before the query, a purely
  local copy): **100% accuracy by step 500** at every N and distance. The
  task setup is learnable; the failure is specific to key→value binding.
- For reference (not run here — no attention): Zoology reports that small
  attention models solve MQAR at these sizes.

**Conclusion:** the deficit §2 predicted is confirmed, and it is not only
a long-range problem — OPERA's compose/fold has no way to compare the
current key with stored content even a few tokens back. This is the
missing function behind the ~100-byte plateau. Next (step 2): the
quaternion holographic memory (§4, §4a), tested on MQAR first.

## 5b. Result of step 2 — quaternion holographic memory gives OPERA recall (2026-09-25)

`hmem_nb` in `opera_lm/model.py` (decoder: `opera_lm/incremental.py`).
Write at every position: unit key of the previous token ⊗ value of the
current token (Hamilton product), times a learned write gate, summed
causally. Read: conj(key of the current token) ⊗ memory → LayerNorm →
projection → zero-init scalar gain into the fold readout. Selftests: a
single stored pair is recovered exactly (err 5e-7); retrieval cosine with
4 / 16 / 32 stored pairs over 256 slots = 0.50 / 0.26 / 0.17 (the
1/√pairs interference HRR theory predicts); bitwise incumbent at init;
causal; decoder exact.

MQAR, same setup as §5a (OPERA L=2, d256, 4,000 steps):

| model | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| fold only | 0.002 | 0.003 | 0.002 | 0.002 |
| **+ memory, 256 slots** | **1.000** | **1.000** | **1.000** | **0.973** |
| + memory, 64 slots | 0.998 | 0.988 | 0.789 | 0.399 |

- Recall is flat across key→query distance (e.g. 256 slots, N=32:
  0.98 / 0.97 / 0.97 / 0.97 across the distance buckets) — no horizon.
- Capacity behaves as the mechanism predicts: fewer slots degrade with
  more stored pairs (interference), more slots cancel it.
- Cost (first, unfused implementation): ~1 s/step at 256 slots vs
  ~0.35 s at 64 slots on this task — optimization needed before scale.

Next (step 3, proposed): language modeling on the byte corpus — does the
memory lift the captured-information curve past ~128 bytes, raise the
ICL score, and improve BPB? In a 2-layer model, layer-1 keys come from
the layer-0 stream (contextual — roughly the recent n-gram), which is
what byte-level induction needs; layer-0 keys are single bytes.

## 5c. Result of step 3 — the memory moves the horizon in language modeling (2026-09-25)

`b2m_hmem128`: the Metal base (stream head + downsweep) + holographic
memory, 128 slots per layer, byte corpus, standard 3,000-step recipe,
seed 42. 4.87M parameters (base 3.30M).

| measure | base (5 runs) | + memory | change |
|---|---|---|---|
| BPB | 2.0252 ± 0.0127 | **1.9263** | **−4.9% (≈ −7.8 sd)** |
| width control st_wide (5.06M params) | 2.0069 | 1.9263 | −4.0% — not parameter count |
| extrapolation 1025–2048 PPL | 4.053 (b2m_base) | 3.745 | −7.6% |
| MI plateau (bits about the next 31 bytes) | ~12.0 | ~16.3 | +4.3 bits |
| MI curve stops rising at | ~128 bytes | ~256 bytes | first onset shift in 2 days |
| ICL score (loss@500 − loss@50, bits) | +0.107 | +0.060 | −0.047 |

Per-position loss (bits), base → memory: 0–16: 2.275 → 2.290;
64–128: 1.852 → 1.782; 256–512: 1.925 → 1.820; 512–1023: 1.928 → 1.821.
The gain grows with the amount of context — the recall signature.

MI curve (memory): 5.2 / 8.6 / 13.9 / 16.1 / 16.6 / 16.3 / 15.9 / 15.2 bits
at 4 / 16 / 48 / 128 / 256 / 512 / 1024 / 2048 bytes.

Caveats: one seed; past ~512 bytes the curve droops (a plain sum memory
accumulates interference with every stored pair — no forgetting);
~1.6× training time in this unfused version.

Next candidates (not run): second seed; forgetting / decay or Fenwick
sharding of the memory against interference; fused/optimized memory
ops; combining with longer training (4× = −10% on the base).

## 5d. The droop past ~512 bytes — research before the fix (2026-09-25)

**Diagnosis (measured, no training).** The memory is a plain running sum,
so every write adds interference for all later reads. The learned layer-0
write gate is not selective: mean 0.66, median 0.61, 79% of positions
above 0.5, none below 0.1 — about **670 effective writes per 1024
bytes**. Holographic retrieval SNR scales like √(slots / stored items):
with 128 slots it falls from ~1.4 after 64 writes to ~0.44 by byte 1024,
and keeps falling past the training length, where no training sequence
ever had so many items (the Stuffed-Mamba lesson: a state trained on
short contexts never learns to forget).

**What the literature does:**
- **Associative LSTM** (Danihelka et al. 2016) puts the LSTM's gates on the
  holographic memory itself: c_t = g_f ⊙ c_{t−1} + g_i ⊙ (bound update) —
  a forget gate on the memory is the direct precedent.
- **Holographic memory models in cognitive science** (TODAM; Kelly et al.,
  *Holographic Declarative Memory*, Cognitive Science 2020) multiply the
  store by a forgetting coefficient α ∈ (0, 1) at every step, which
  privileges new over old information.
- **RetNet** (Sun et al. 2023, arXiv 2307.08621): fixed per-head decays
  spread geometrically, γ_h = 1 − 2^{−5−h}, so different heads keep
  different time horizons (recurrence S_t = γ S_{t−1} + k_tᵀ v_t).
- **Data-dependent forgetting** (GLA, arXiv 2312.06635; Mamba): the decay
  is a gate computed from the current input, S_t = G_t S_{t−1} + update;
  without any decay a linear-memory model struggles to forget, which has
  been linked to instability on long contexts. Data-dependent gates are
  generally preferred for retrieval, fixed decays for simplicity.
- **Erase-before-write** (delta rule; Gated DeltaNet-2, arXiv 2605.22791,
  gains on multi-key retrieval): remove the old value bound to a key
  before writing. Needs a content read at write time — possible later in
  holographic form, but a larger change.

**Proposed fix (two arms, one change each, on `b2m_hmem128`):**
1. **Multi-timescale decay per slot (RetNet / TODAM style):**
   M_t = λ_s ⊙ M_{t−1} + w_t (key_{t−1} ⊗ value_t), with λ_s learnable per
   slot (a 1-D parameter → AdamW), initialized geometrically from half-life
   ~8 bytes to ~4,000 bytes across the 128 slots. Content-independent,
   bounded effective item count, extrapolation-safe.
2. **Data-dependent forget gate (Associative LSTM / GLA style):**
   λ_{t,s} = σ(W_f x_t + b_s), b_s initialized to the same spread — the
   model decides per token what to forget.
Both are gates/decays, not comparisons — same no-attention status as the
memory itself. Computed with a parallel scan over (decay, update) pairs
(log₂ T elementwise steps); O(1) state per token in the decoder.

**Success criteria:** the MI curve stops drooping past 512 bytes (and past
the training length), extrapolation PPL improves, and BPB is at least as
good as `b2m_hmem128`; MQAR recall must stay near 100%.

## 5e. Results — second seed, and the droop fix (2026-09-25)

**Second seed confirms the memory.** b2m_hmem128_s43: 1.9569. Two-seed
mean 1.9416 vs the five-run base mean 2.0252 ± 0.0127 → **−4.1%,
t ≈ −7.9**; same-seed pairs −5.4% (seed 42) and −3.9% (seed 43).

**Decay arms (seed 42, on b2m_hmem128):**

| model | BPB | vs plain memory mean | extrap. 1025–2048 | MI peak → 2048 bytes |
|---|---|---|---|---|
| plain memory, seed 42 / 43 | 1.9263 / 1.9569 | — | 3.745 / 3.809 | −1.44 / −1.39 bits |
| **+ gated forget** | **1.8964** | **−2.3%** (−6.4% vs base) | **3.629** | **−0.02 bits (flat)** |
| + fixed decay | 1.9751 | +1.7% | 3.856 | −1.06 bits |

MI curves (bits, 16 / 48 / 128 / 256 / 512 / 1024 / 2048 bytes):
plain seed 42: 8.6 / 13.9 / 16.1 / 16.6 / 16.3 / 15.9 / 15.2;
gated: 9.0 / 13.3 / 14.8 / 15.1 / 15.2 / 15.2 / 15.2;
fixed: 7.8 / 11.9 / 13.9 / 14.5 / 14.4 / 14.1 / 13.4.

Per-position loss (bits), plain → gated: 0–16: 2.290 → 2.228;
16–64: 1.839 → 1.768; 256–512: 1.820 → 1.794; 512–1023: 1.821 → 1.794.

- The data-dependent forget gate removes the droop entirely (flat past
  the training length); a content-independent decay does not, and costs
  BPB — forgetting must depend on content (as in Associative LSTM / GLA).
- The gated model is better at every position, including very early
  ones, so its MI plateau (gain *from* context) sits below the plain
  memory's while its absolute loss is lower everywhere: across models of
  different local strength, per-position loss is the more reliable
  comparison; the MI curve's *shape* (droop or not) remains informative.
- One seed for the gated arm; its −2.3% over the plain memory exceeds the
  1.6% gap between the two plain seeds, but needs a second seed.

## 5f. Gated memory confirmed on a second seed; speed work (2026-09-25)

| model | seeds | BPB | mean | vs base (2.0252 ± 0.0127) |
|---|---|---|---|---|
| + holographic memory | 42 / 43 | 1.9263 / 1.9569 | 1.9416 | −4.1% |
| **+ gated forget** | 42 / 43 | 1.8964 / **1.8389** | **1.8676** | **−7.8%** |

Gated vs plain at the same seed: −1.6% (42), −6.0% (43); extrapolation
1025–2048 PPL 3.629 / 3.516 vs 3.745 / 3.809. The gated holographic memory
is the new best OPERA configuration.

**Speed (exact rewrites, verified to ~1e-6 on the trained gated
checkpoint; full selftest passes):** work-efficient (Blelloch-style)
decay scan, ~2.8× faster than the Hillis–Steele version; unbinding with
the conjugate folded into the product (~15% faster); one fused GEMM for
the key / value / write-gate projections. One memory layer 383 → 290 ms
fwd+bwd; the full step is still ~2.3× the base (902 vs 389 ms, both
measured alongside a training run). The memory is memory-bandwidth
bound (many elementwise kernels over B × T × 512 fp32 tensors, with
stored intermediates); the remaining fix is a fused Metal kernel (bind +
scan + unbind, recompute-in-backward), which leaves the math unchanged.

## 5g. Fused Metal kernel for the memory core (2026-09-25)

`opera_lm/hmem_kernel.py`: one GPU thread per (sequence, slot) carries the
memory in registers and runs bind → decayed superposition → unbind in one
forward pass (saving only M); the backward is one reverse pass with the
analytic quaternion gradients (L(p)ᵀ = L(p̄), R(p)ᵀ = R(p̄)). Used
automatically when `use_metal=True` on MPS; the PyTorch path elsewhere.
Correctness: forward and all four input gradients equal PyTorch autograd
of the reference to ~2e-7 relative (up to B=8, T=1024, 128 slots); the
model's Metal path equals its eager path (loss identical, grads 7.5e-7);
selftest `test_hmem_kernel`; full suite passes.

Idle-GPU timings (B=8, T=1024, d512, 2 layers, 128 slots, gated):

| | time |
|---|---|
| memory core, fwd+bwd: PyTorch → fused kernel | ~200 ms → **6.3 ms** |
| one memory layer, fwd+bwd | 383 (original) → 290 (PyTorch rewrite) → **55 ms** |
| full training step: base / memory (PyTorch) / memory (kernel) | 181 / 469 / **354 ms** (1.00× / 2.58× / 1.95×) |

The remaining memory cost is the surrounding PyTorch — the fused
key/value/write projection with key normalization (38.7 ms fwd+bwd per
layer) and the output norm + projection (10.3 ms). Next increment
(optional): move key normalization, the write/forget sigmoids and the
fp32 casts into the kernel (reading bf16 inputs directly).

## 5h. Fused kernel v2: the whole memory read in one kernel (2026-09-25)

`hmem_fused` (opera_lm/hmem_kernel.py): the keys / values / write gate /
forget gate come from **one GEMM**, and the kernel reads that raw
projection directly in bf16. Key normalization, the write and forget
sigmoids and the per-slot decay logit all run inside the kernel, next to
bind → decayed superposition → unbind. The backward pass goes straight
back to the raw projection, so there are no fp32 copies, no normalized-key
tensors and no gate tensors. Same math: all three decay modes
(none / fixed / gated), fp32 and bf16 inputs, agree with the PyTorch
reference to ~2e-7. The model's Metal path equals its eager path (loss
identical, every memory gradient ≤ 1e-6). Full selftest passes.

| full training step (B=8, T=1024, d512, 2 layers, 128 slots, gated) | time | vs base |
|---|---|---|
| PyTorch memory (§5f) | 469 ms | 2.58× |
| fused core kernel (§5g) | 354 ms | 1.95× |
| **fused v2 (this section)** | **~240 ms** | **~1.24×** |

(The base was 181–209 ms across sessions; each ratio is measured against
the base in the same session.)

## 7. The next quality lever — research before the next experiment (2026-09-25)

**Diagnosis from the code.** The memory reads the layer's input stream.
At layer 0 that stream is the raw byte embedding, so the layer-0 memory
has **only 256 possible keys**. Its write binds key(byte t−1) to
value(byte t), which makes it a document-local, forgetting **bigram
table**: "what came after an `e` in this document". Real copying needs
the key to identify the *context*: "what came after `the ` last time".
Layer 1's keys are contextual, but each still describes a single position.

**What the literature says:**
- *Anatomy of Associative Recall in Fixed-State Recurrences*
  (arXiv 2609.16183, Sept 2026) decomposes recall at a matched state
  budget into three levers:
  - A **short causal convolution** is the dominant one: +0.44–0.47
    accuracy, for both delta-rule and diagonal cells.
  - The transition structure is worth +0.19–0.32 without the
    convolution, but only **+0.03 once the convolution is present**.
  - Decay has no measurable cost.
  - Their advice: "depthwise convolution, width 4, causal, zero
    recurrent state" is non-negotiable for recall.
- **Canon layers** (Allen-Zhu, *Physics of Language Models 4.1*,
  NeurIPS 2025, arXiv 2512.17351):
  - The layer is h_t + conv1d([h_t, h_{t−1}, h_{t−2}, h_{t−3}]): width 4,
    causal, **residual essential**, no activation.
  - Placement "B", after the Q/K/V projections, is the analogue of our
    memory's projection.
  - It lifts NoPE models to RoPE quality and linear attention to
    Mamba2/GDN quality.
  - Most of Mamba2's advantage over GLA *is* its conv1d.
  - OPERA is a NoPE, linear-memory model, which is exactly the regime
    where Canon helps most.
- H3, Mamba and Based (Zoology) all pair their linear memory with a short
  convolution for the same reason.

**What we do not need, by an exact argument.** The delta rule
(erase-before-write, as in Gated DeltaNet) has nothing to add inside a
quaternion slot. With a unit key k, reading what is stored under k and
erasing it gives M − k⊗(k̄⊗M) = M − M. So erase-then-write with strength
β is **exactly (1−β)·M + β·k⊗v, the data-dependent forget gate the
gated memory already has**. The 2609.16183 result (transition type
barely matters once a convolution is present) agrees.

**Plan, one change at a time:**
1. **Short convolution on the memory projection** (`hmem_conv=4`, running
   now at 2 seeds):
   - y_t ← y_t + Σ_{j<4} c_j ⊙ y_{t−j}, over the raw projection (keys,
     values, write and forget logits).
   - Zero-init taps on AdamW, so it is identical to the gated memory at
     init.
   - Keys then describe the last 4 positions (4-byte contexts at layer 0).
   - Its own Metal kernel (fwd+bwd 6 ms per layer, exact); cost ~+10 ms
     per step.
   - Decoder: the last 4 projections, O(1) state.
   - **Success:** a 2-seed mean clearly below the gated mean 1.8676. The
     gated seeds differ by 3%, so a gain under ~1.5% cannot be claimed.
     The gain should show in per-position loss past ~64 bytes, where
     copying happens, and MQAR must stay at ~100%.
2. **An output gate on the memory read**, σ(W x) ⊙ LN(r) (GLA,
   Gated DeltaNet, RetNet's swish gate), only after step 1 is judged.
3. **The long run** with the best configuration (4× training gave −10%
   on the base).
4. Not now: a distance curriculum (2609.16183). It fixes sparse-supervision
   recall tasks; language modeling supervises every position.

## 7a. Result of step 1 — the short convolution is a null at this scale (2026-09-25)

| model | seed 42 | seed 43 | mean |
|---|---|---|---|
| gated memory | 1.8964 | 1.8389 | 1.8676 |
| + short conv (width 4) | 1.8566 (−2.1%) | 1.8983 (+3.2%) | 1.8775 (+0.5%) |

Per-position loss (bits; 0–16 / 16–64 / 64–128 / 256–512 / 512–1023):

| model | seed 42 | seed 43 |
|---|---|---|
| gated | 2.228 / 1.768 / 1.744 / 1.794 / 1.794 | 2.196 / 1.733 / 1.681 / 1.735 / 1.730 |
| + conv | 2.229 / 1.742 / 1.698 / 1.752 / 1.756 | 2.316 / 1.794 / 1.733 / 1.791 / 1.792 |

- **The same-seed differences have opposite signs, and the mean is within
  noise.** Each run moves every band by about the same amount, including
  the first 16 bytes, where no recall is possible. That is a whole-model
  seed shift, not a recall effect.
- **The model uses the convolution only lightly.** The learned taps are
  ~0.1 on the key channels at every lag, so keys stay mostly
  single-position. The value channels mainly rescale the current position
  (tap 0 ≈ 0.2–0.3). The write gate in layer 1 does use the previous
  position (taps 0.3–0.5).
- **Reading:** the recall papers measure the convolution on synthetic
  recall with small states. At our scale, the gated memory plus the tree
  already supply the local context the keys need. The flag stays in the
  code, off by default.
- **Measurement lesson:** with the memory, seed-to-seed spread is ~3%
  (vs 0.65% for the base), so a single-seed arm is uninformative, and
  even two seeds only resolve effects of ~2% or more.
- **Side effect of §5h:** a full 3000-step run now takes **16 min instead
  of ~33 min** (0.29–0.30 s/step with the memory).

## 7b. The long run — the memory's gain holds at 4× training (2026-09-25)

Both runs: seed 42, 12,000 steps (4× the standard), batch 8, T = 1024.

| model | params | BPB | extrap. PPL 1025–2048 | minutes |
|---|---|---|---|---|
| base, 12k steps (`b2_long`) | 3.30M | 1.8179 | 3.467 | 71 |
| **gated memory, 12k steps** (`b2m_hmem128_gated_long`) | 5.00M | **1.7043 (−6.2%)** | **3.18 (−8.3%)** | 61 |

Per-position loss (bits), base → memory:

| bytes | 0–16 | 16–64 | 64–128 | 128–256 | 256–512 | 512–1023 |
|---|---|---|---|---|---|---|
| base | 2.114 | 1.668 | 1.648 | 1.690 | 1.717 | 1.722 |
| memory | 2.077 | 1.588 | 1.558 | 1.585 | 1.610 | 1.611 |
| change | −1.8% | −4.8% | −5.5% | −6.2% | −6.2% | −6.4% |

- **New best OPERA: 1.7043 BPB**, −8.7% vs the gated memory at 3k steps
  (1.8676, 2-seed mean) and −16% vs the 3k base band (2.0252).
- **The gain grows with distance into the context:** −1.8% in the first
  16 bytes, where there is little to recall, and −6.4% past byte 512.
  That is the signature of recall, not of a generally stronger model. The
  extrapolation gain (−8.3%) is the largest of all.
- **The memory's relative gain is about the same as at 3k steps** (−6.2%
  vs −6.4% same-seed at 3k). The advantage is not a short-training
  artefact that washes out, and it has not grown either.
- **The memory run was faster** despite 1.5× the parameters (61 vs
  71 min), thanks to §5h. `b2_long` was trained before the Metal compose
  path.
- **Caveats:**
  - One seed. The effect is 2× the memory arms' seed spread, and the
    distance-graded band pattern supports it.
  - There is no parameter-matched long control. At 3k steps the
    width-matched base (`st_wide`, 5.06M, 2.0069) recovered under a
    quarter of the memory's gain, so parameter count does not explain it.
  - Both models' loss still rises from the 64–128 band to 512+ bytes.
    That is shared, so it is likely a property of the data position
    rather than of the memory.

## 8. Scaling up (2026-09-25)

**Data first.** Every earlier run trained on a 20,000-article slice of
Simple English Wikipedia, 37.5M training tokens. The cap existed only
because `build_corpus` stores chunks as Python int lists. The 12k-step
runs already made 2.6 passes over that slice, so a larger model would
mostly have learned to memorize it.
- `experiments/build_packed.py` packs the **full** corpus straight into
  the `opera_lm.packed` mmap format: 241,773 articles, **217M training
  tokens (5.8×)**, 11 s.
- It uses the same article stream, filter, seeded per-article split and
  chunk schedule as `build_corpus`. Its 20,000-article build is
  **identical** to the old training set (same 45,765 sequences as a
  multiset).
- So none of the old test articles are in the pool.
  `repr_study --packed` trains from the pool and **evaluates on the
  unchanged a20000 test sets**, which keeps every BPB comparable.
- Caveat: the later articles are shorter on average (~900 vs ~1,900
  bytes), so the added data is somewhat different text.

**Width versus depth at this size** (B=8, T=1024, memory on, one
training step):

| model | params | step |
|---|---|---|
| d512 × 2 layers (current) | 5.0M | 240 ms |
| d768 × 2 | 11.0M | 409 ms |
| d1024 × 2 | 19.4M | 624 ms |
| d512 × 4 (add + 1/√(2L) init) | 9.5M | 505 ms |
| d768 × 4 (add + 1/√(2L) init) | 21.1M | 803 ms |

- Width buys parameters more cheaply than depth here.
- Depth still carries the untested initialization fix (§4.11 of the
  Speed doc).
- Batch 16 costs 1.9× batch 8, so there is no efficiency gain from a
  larger batch.

**Runs** (12,000 steps = 98M tokens, seed 42, gated memory):
1. `b2m_hmem128_gated_full`: the 1.7043 model on the full pool, which
   measures the data effect.
2. `w768_hmem192_gated_full`: d768, 192 slots (memory width = model
   width, as at d512), which measures the width effect at equal tokens.

98M tokens is about the Chinchilla-optimal budget (~20 tokens per
parameter) for the 5M model, and under half of it for the 11M model. If
width wins even here, the next step is the 11M model trained for a full
pass over the pool (~27k steps, ~3.5 h).

## 8a. Results — width wins; the extra data does not help on this test set (2026-09-25)

Both runs: 12,000 steps, seed 42, full pool, evaluated on the a20000 test sets.

| arm | data | params | BPB | extrap. PPL (1025–2048) | ICL score |
|---|---|---|---|---|---|
| `b2_long` (no memory) | slice | 3.3M | 1.8179 | 3.47 | +0.121 |
| `b2m_hmem128_gated_long` | slice | 5.0M | 1.7043 | 3.18 | +0.103 |
| `b2m_hmem128_gated_full` | full pool | 5.0M | 1.7799 | 3.37 (0.98×) | +0.095 |
| **`w768_hmem192_gated_full`** | full pool | 11.0M | **1.6572** | **3.12** (0.99×) | **+0.044** |

Per-position loss (bits, `icl_score.py`, 400 test sequences):

| band | d512 slice | d512 full | d768 full | d768 vs d512 full |
|---|---|---|---|---|
| 0–16 | 2.077 | 2.155 | 2.119 | −1.7% |
| 16–64 | 1.588 | 1.686 | 1.622 | −3.8% |
| 64–128 | 1.558 | 1.657 | 1.556 | −6.1% |
| 128–256 | 1.585 | 1.685 | 1.566 | −7.1% |
| 256–512 | 1.610 | 1.711 | 1.589 | −7.1% |
| 512–1023 | 1.611 | 1.709 | 1.574 | −7.9% |

**Width is a real effect.**
- d768 is −6.9% BPB against d512 on the same data and token count.
  That is more than twice the ~3% seed spread.
- The gain grows with position, from −1.7% at bytes 0–16 to −7.9% past
  byte 512. A seed effect would shift every band by about the same
  amount, so this is better use of context.
- The ICL score halves (+0.095 → +0.044): loss rises less with depth into
  the document.
- It is the new best model: **1.6572 BPB**. That is −2.8% against the
  slice-trained d512 and −8.8% against the memory-free `b2_long`.
- It costs 1.7× the step time (0.51 s vs 0.30 s per step).

**More data made the same model worse (+4.4%) on this test set.**
- The loss rises in every band (+3.8% at 0–16, about +6% everywhere
  else).
- The pool is a different distribution from the test set:

  | | first 20k articles | rest of pool |
  |---|---|---|
  | mean sequence length | 820 bytes | 613 bytes |
  | full-length (1,024-byte) sequences | 62% | 32% |
  | share of pool tokens | 18% | 82% |

- The test set is drawn only from the first 20k articles, which are the
  older, longer ones. So the full-pool model trains mostly on shorter
  stubs, sees half as many long-context training positions, and is
  scored on text that is now under-represented.
- A single seed cannot fully separate this from noise: +4.4% sits just
  above the spread. The direction and the long-document deficit both
  point to the shift, not to "more data hurts".
- A repeat of 2.6 passes over 37.5M tokens was not yet overfitting a 5M
  model, so the old slice was not the bottleneck at 5M.

**Next step (not run).** The pool needs a fair measurement before the
11M full-pass run:
1. **Held-out set from the whole pool.** Build one from the test-split
   articles past 20k that `build_packed` currently drops, and report BPB
   on both test sets. It is evaluation-only and resolves whether the pool
   is worse or just different.
2. If long documents matter, either filter the pool to articles of at
   least one full chunk or weight sampling by length. Then run the 11M
   model for one pass (~27k steps, ~3.8 h at 0.51 s per step).

## 8b. The second test set — the full corpus is different, not worse (2026-09-26)

**Pool test set.** `build_packed.py` now also writes
`packed_bytes_T1024_all.test.pkl`:
- It takes a seeded uniform sample (seed 0) of 5,000 in-length chunks
  from the test-split chunks of **all** 241,773 articles (37,817 chunks
  in total), plus every longer chunk (1,035).
- Each chunk is tagged with its article index, so **new** (articles
  past the first 20,000; 4,338 of the 5,000) can be scored separately.
- The pool files it rewrites are byte-identical to before.
- `experiments/eval_pool.py` scores finished arms on every set. It
  recomputes the old a20000 numbers exactly (1.8179 / 1.7043 / 1.7799 /
  1.6572), which checks the loader.

| arm (BPB) | trained on | a20k | pool | new | long 1025–2048 |
|---|---|---|---|---|---|
| `b2_long` (no memory) | slice | 1.8179 | 1.9309 | 1.9540 | 2.3807 |
| `b2m_hmem128_gated_long` | slice | 1.7043 | 1.8389 | 1.8664 | 2.3490 |
| `b2m_hmem128_gated_full` | full pool | 1.7799 | **1.6203** | **1.5877** | 1.4253 |
| `w768_hmem192_gated_full` | full pool | 1.6572 | **1.4918** | **1.4584** | **1.2687** |

- **The full-pool model is better on the corpus it was trained on.**
  - d512 on the pool test: −11.9% vs the slice-trained model (−14.9% on
    new articles).
  - The +4.4% on a20k (§8a) was the test set's narrow slice of the
    corpus, not "more data hurts".
  - Some of the gain is templated text the slice model never saw: 3.9%
    of pool test chunks are "X is a commune in …" stubs, and 4.4% are
    wiki tables.
- **The long bucket is not a length result.**
  - 31% of the pool's 1025–2048-byte test chunks are minor-planet wiki
    tables (≥10 `||` separators), from 255 articles. A model trained on
    such tables predicts them easily.
  - So the slice-vs-pool gap there (2.35 → 1.43) says nothing about
    extrapolation.
- **Width holds on every set.** d768 vs d512, same data and tokens:
  −7.9% pool, −8.1% new, −11.0% long, −6.9% a20k.

## 8c. Plan for the large model — Colab A100, FineWeb-Edu bytes (2026-09-26)

**Data.** 25+ A100 hours will far outrun Simple Wikipedia (217M bytes
supports ~11M params at one pass, or ~40M at four; more repeats stop
helping). So `experiments/build_fineweb_bytes.py` builds a byte pool
from FineWeb-Edu (`sample-10BT`, streamed in dataset order):
- **Held-out split:** per document, `Random(42)` holds out 0.5% of
  documents, up to 20,000 test documents.
- **Training chunks:** plain consecutive 1,024-byte chunks. No byte is
  dropped. The Simple-Wikipedia schedule sends every chunk after the
  eighth to evaluation, which would discard about half of any document
  over 8 KB.
- **Test sets:** a 5,000-chunk in-length set cut the same way, and an
  extrapolation set of each held-out document's opening ≤4,096 bytes.
- **Storage:** uint16, streamed to disk (fixed 128-byte `.npy` header
  patched with the final length), so RAM stays flat.
- **Evaluation:** primary = FineWeb-Edu held-out. The Simple-Wikipedia
  a20k / pool / new sets are kept for continuity (cross-domain for a
  FineWeb model).

**CUDA path.**
- The arms name the Metal kernels. `repr_study.device_kernels` maps them
  to the Triton compose kernel on CUDA, with identical math.
- The memory runs as the PyTorch reference (`_decay_scan`) under
  `torch.compile`.
- Checked on this machine by emulating the CUDA path on CPU:
  - Triton fallback vs eager: loss equal, max gradient difference
    1.4e-6.
  - The compiled model gives the same loss.
- `experiments/device_check.py` repeats this on the GPU (device + compile
  vs CPU fp32, loss and every gradient) and must pass before training.

**Runbook:** `colab/OPERA_Scale.ipynb`.
1. Setup and the correctness tests.
2. Rebuild the Simple-Wikipedia test sets from the fixed snapshot,
   asserted against the local counts.
3. Build the FineWeb pool (cached on Drive).
4. Throughput benchmark over width, depth and batch
   (`experiments/bench_scale.py`: forward + backward + Muon, bf16,
   compile).
5. Train with `--save-every 1000 --resume` into a Drive run directory
   (`OPERA_RUNS`), so a disconnected runtime resumes the exact batch
   stream (tested: kill mid-run → `RESUMED`), and score on every test
   set at the end.

Arms are named `fw_d{width}_L{layers}` (memory slots = width/4; L > 2
uses the additive residual with 1/√(2L) init).

**Choosing the size.** Wait for the benchmark: the fold is not a plain
matrix-multiply workload, so A100 throughput cannot be read off from the
Mac. Rule of thumb: tokens ≈ 20 × params.
- Depth > 2 is still unvalidated at this scale.
- So the first GPU hours should go to a short ladder: two or three
  sizes, plus a 2- vs 4-layer check at equal parameters, at a fixed
  token budget.
- Then the single long run.

## 6. Sources

- Khandelwal, He, Qi, Jurafsky — *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context*, ACL 2018 — arxiv.org/abs/1805.04623
- Sun, Krishna, Mattarella-Micke, Iyyer — *Do Long-Range Language Models Actually Use Long-Range Context?*, EMNLP 2021 — arxiv.org/abs/2109.09115
- Olsson et al. — *In-context Learning and Induction Heads*, 2022 — transformer-circuits.pub/2022/in-context-learning-and-induction-heads
- Arora, Eyuboglu et al. — *Zoology: Measuring and Improving Recall in Efficient Language Models*, 2023 — arxiv.org/abs/2312.04927
- Fu, Dao et al. — *Hungry Hungry Hippos: Towards Language Modeling with State Space Models*, ICLR 2023 — arxiv.org/abs/2212.14052
- Plate — *Holographic Reduced Representations*, IEEE Trans. Neural Networks 6(3), 1995
- Aerts, Czachor, De Moor — *Geometric Analogue of Holographic Reduced Representation*, J. Math. Psychology, 2009 — arxiv.org/abs/0710.2611
- Kleyko et al. — *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures*, ACM CSUR — arxiv.org/abs/2111.06077
- Schlegel, Neubert, Protzel — *A comparison of Vector Symbolic Architectures* — arxiv.org/abs/2001.11797
- Smolensky — *Tensor product variable binding*, Artificial Intelligence 46, 1990
- Schlag, Irie, Schmidhuber — *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021 — arxiv.org/abs/2102.11174
- Danihelka, Wayne, Uria, Kalchbrenner, Graves — *Associative Long Short-Term Memory*, ICML 2016 — arxiv.org/abs/1602.03032
- Alam, Raff et al. — *Recasting Self-Attention with Holographic Reduced Representations*, ICML 2023 — arxiv.org/abs/2305.19534
- Frady, Kent, Olshausen, Sommer — *Resonator Networks* 1 & 2, Neural Computation 32(12), 2020
- Vishwakarma, Agostino — *Phase-Associative Memory: Sequence Modeling in Complex Hilbert Space*, 2026 — arxiv.org/abs/2604.05030
- Sun et al. — *Retentive Network: A Successor to Transformer for Large Language Models*, 2023 — arxiv.org/abs/2307.08621
- Yang et al. — *Gated Linear Attention Transformers with Hardware-Efficient Training*, ICML 2024 — arxiv.org/abs/2312.06635
- Kelly, Arora, West, Reitter — *Holographic Declarative Memory*, Cognitive Science 44, 2020
- *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, 2026 — arxiv.org/abs/2605.22791
- OPERA paper draft v1.3 §4.4 (interrogation probe), this repo
- *Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It*, 2026 — arxiv.org/abs/2609.16183
- Allen-Zhu — *Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers*, NeurIPS 2025 — arxiv.org/abs/2512.17351
- Arora et al. — *Simple linear attention language models balance the recall-throughput tradeoff* (Based), 2024 — arxiv.org/abs/2402.18668
- Yang, Kautz, Hatamizadeh — *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025 — arxiv.org/abs/2412.06464
- Hoffmann et al. — *Training Compute-Optimal Large Language Models* (Chinchilla), 2022 — arxiv.org/abs/2203.15556
- Muennighoff et al. — *Scaling Data-Constrained Language Models*, NeurIPS 2023 — arxiv.org/abs/2305.16264
- Penedo et al. — *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024 — arxiv.org/abs/2406.17557
