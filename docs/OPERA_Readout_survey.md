# OPERA readout/addressing survey — 2026-09-13

Companion to `OPERA_Depth_Survival_diagnostic.md` (F1–F7, G1–G4). The
diagnostic found: identity survives, but the Fenwick readout is a
content-blind compressor — no addressable old bytes beyond ~4 (F3),
rank collapse through depth (F4), position absent (F5), a small
deterministic texture at zero-fold positions (F6). This survey maps the
2024–2026 literature onto those findings and extracts concrete,
invariant-respecting transplants. Same rule as the optimizer survey:
nothing here is adopted; every candidate needs its own prereg.

## 1. The field-level picture: our F3 is a known theorem, and OPERA is NOT in the doomed class

- **Fixed-size-state models cannot retrieve.** *Repeat After Me*
  (arXiv:2402.01032, ICML 2024): GSSMs (Mamba-class) are
  information-theoretically limited at copying/retrieval by their
  constant-size state; transformers copy strings of exponential length.
  *Zoology/Based* (arXiv:2402.18668) traces the same failure to MQAR
  (multi-query associative recall). *DeltaNet* line
  (arXiv:2406.06484): vanilla linear attention is an append-only memory
  that cannot overwrite — the delta rule fixes part of it.
- **The field's fixes, in order of adoption:** (a) hybrid layers —
  interleave full attention (Jamba, Samba, Hymba; Qwen3-Next at 3:1
  GatedDeltaNet:attention; Kimi Linear; hybrid Mamba-3 beats even
  transformers on real-world retrieval, ICLR 2026 oral); (b) local
  windows — Based's sliding-window attention is *the* recall restorer,
  and BLT/Megabyte make local-bytes + global-latent the standard at our
  exact rung; (c) more expressive recurrence — selectivity/delta-rule
  (Mamba 1→3, Gated DeltaNet, RWKV-7); (d) log-growing state —
  Log-Linear Attention.
- **Why OPERA is not in the doomed class:** OPERA's per-position state
  is not constant-size — it is the O(log T) Fenwick block path. The
  retrieval weakness we measured (F3) is therefore *not* forced by an
  information-theoretic bound: the span-flatness check (composition
  preserves addressability) shows the old bytes' information is
  reachable in principle; the readout just never selects it. Our F3 is
  an *allocation* failure, and the literature says allocation failures
  are exactly the fixable ones.

## 2. The closest external architecture: Log-Linear Attention (arXiv:2506.04761)

Read closely — it shares OPERA's data structure and complexity class:

- **Fenwick tree organizes the state.** One matrix-valued state per
  level of the prefix decomposition (outer-product sums over each
  dyadic bucket); decode-time state O(log T); training O(T log T)
  (chunkwise scan); custom Triton beats FlashAttention-2 past 8K.
- **No positional encoding** — position enters via the Fenwick
  bucketing, same claim as our I1.
- **The readout is where they spent their content-dependence budget:**
  each level's contribution is gated by a query-dependent scalar
  λ_ℓ(x_t) ≥ 0 (a linear projection of the input) — `o_t = Σ_ℓ
  λ_t^(ℓ) · q_t^T S_t^(ℓ)`. Not attention over positions; per-level
  gates. With λ equal at all levels it collapses to plain linear
  attention.
- **Results:** MQAR +9–16 points over the base models (Mamba-2 dim16:
  46.9→55.9; GDN dim32: 79.0→84.4); RULER needle-in-haystack passkey
  at 16K: **21.6→72.4**; better WikiText/LMB PPL at ~800M/50B tokens
  for <0.4–3% extra params; still a gap to parameter-matched
  transformers.

**Reading for OPERA:** the minimal intervention that moved recall in
this class is *query-dependent per-level gates at the readout* — a
tiny mechanism sitting exactly on the axis between our content-blind
fold and full block-routing attention. Our fold has no query term at
all. That is the single best-evidenced gap in the OPERA stack.

## 3. Transplants (candidates for prereg; NOT adopted)

**T1 — Per-level query gates on the readout (from Log-Linear
Attention). New candidate, jumps the queue.** Gate each gathered
block's entry into the fold by λ_ℓ(x_t) (scalar or per-nb), computed
from the layer input, zero-λ ⇔ bitwise incumbent at init. Cost O(log T)
per position, ~zero params. Respects I1–I4 (no softmax over positions,
no pairwise attention; levels are structure, not positions). External
evidence: the MQAR/NIAH jumps above. In-house tension to disclose: the
*bistable* arms (content-gated fold accumulator) lost by ~4% — but
bistable gated the accumulator *update*, not the readout's per-level
contributions, and had no query term; Log-Linear's gains came
specifically from readout-side gates. Kill-switch: k=8 decodability
must rise; else the mechanism did not engage and any loss wiggle is
noise.

**T2 — Local-guarantee decomposition (= R2, now field-backed).**
Always split the last m≈4–8 leaves into their own blocks so every
position's readout contains a sharp local window (and no zero-fold
readout ever occurs). Field analogues: Based's sliding-window component
(recall restorer), BLT/Megabyte (local bytes + global latent — the
standard at the byte rung), Longformer's local+global. Content-blind,
O(T), no invariant issues.

**T3 — Selective/delta-style fold (deferred).** Query-dependent
overwrite in the accumulator (Mamba-3/GDN direction). Highest risk:
nearest in-house neighbour (bistable) falsified; revisit only if T1
shows the readout side is where content-dependence pays.

**T4 — Rank as instrument only (from Dong et al., arXiv 2105.02750,
and ICML 2025 "Mind the Gap").** Transformer rank collapse is
doubly-exponential in depth *without* residual paths; residuals+MLP+LN
are the known preventers. Our P-stream declines with depth *despite*
the gated blend (36→22 at L8) — keep PR in the Path A instruments
(R3), no surgery.

**T5 — NoPE calibration (no transplant).** Causal masking alone can
induce position-dependent attention patterns even with zero parameters
(arXiv:2509.21042); NoPE transformers are production-viable (SmolLM3,
Kimi K3 reported). Two implications: (a) do not add PE to OPERA (I1;
position absence is not currently loss-binding at our rung — F5); (b)
for Path A, expect the TF-NoPE arm to be competitive with RoPE at
moderate lengths, which sharpens H-gate predictions. Caveat: NoPE
models still have content-addressable attention through which position
leaks; OPERA's fold leaks far less — if long-context H1/H2 ever hinge
on position, the fix must come from tree-native structure (e.g.,
span-dependent twists — `fold_theta` exists), not PE.

## 4. Rejections

- **Full-attention hybrid layers** (Qwen3-Next/Jamba style): violates
  I2 (no pairwise attention). Would be a charter amendment, not an arm.
- **KV-cache / external test-time memory** (Titans-class): linearly
  growing state; abandons the O(log T) state claim — different
  architecture class.
- **Positional encodings of any kind:** I1; and T5 says unnecessary.
- **More depth / bigger head / wider highway:** measured dead ends
  (G2, D5, D7 linear saturation, G1).

## 5. Novelty check (post-survey)

- Fenwick + O(T log T) + no-PE **exists** (Log-Linear Attention) —
  cite as the closest external architecture in any writeup; OPERA's
  remaining distinct claims are the geometric/quaternion merge (vs
  additive outer-product states), absence of any query-key inner
  product (I-invariants), node-level (not level-level) prefix states,
  and the probe-level mechanistic diagnostics.
- Per-level query gates over a *geometric* fold: not in the literature
  (Log-Linear gates linear-attention states). T1 is a genuine
  combination with external evidence.
- Content-addressable routing over the O(log T) path *blocks*
  (original R1/attend-family): still unclaimed at log-T granularity;
  now second in line behind T1.

## 6. Steer addendum (Merna, 2026-09-13) — geometry-first

Decision after this survey: **no attention-family mechanisms** — the v8.0
attend fold already lost in-house, and content-dependent routing is off
the program by charter preference. T1 (query gates) and the original R1
(block-routing) move to the rejected shelf; note the house's own
Improvement Survey 2026-09 already contains an adaptive-λ design
(softplus per-level gates, zero-init, level features via lvl +
level_sin_enc, I1–I4 ✓) if content-*gating* (not scoring) is ever
wanted — its own analysis says the *static* half is falsified (fold
gate bias arm) and the adaptive half untested.

The promoted direction is **geometry-native: scale-graded readout** —
see the diagnostic doc's follow-up section. The mechanism in one line:
quaternion composition preserves factor addressability (F3 span check)
but the fold multiplies all scales into a single rotor, and one rotor
per quaternion block has 3 dof — it cannot address ⌈log T⌉ scales.
Keep per-scale rotors in level-assigned subspaces instead of
multiplying them together; the dyadic structure of the tree becomes
the geometry of the state. Content-blind, invariant-clean, and
probe-falsifiable (k≥8 decodability must rise in level slots, else
kill).

## 7. Sources

- Log-Linear Attention — arxiv.org/abs/2506.04761 (+ OpenReview
  mOJgZWkXKW)
- Repeat After Me (transformers vs SSMs at copying) —
  arxiv.org/abs/2402.01032
- Zoology / Based (recall-throughput tradeoff, MQAR, sliding window) —
  arxiv.org/abs/2402.18668
- DeltaNet parallelization / Gated DeltaNet — arxiv.org/abs/2406.06484
- Qwen3-Next hybrid 3:1 — qwen.ai blog; vLLM blog 2025-09-11
- Mamba-3 — arxiv.org/html/2603.15569v1; Princeton PLI blog; ICLR 2026
  oral (hybrid outperforms transformers on retrieval)
- BLT (Byte Latent Transformer) — ACL 2025 (Pagnoni et al.); Fast BLT
  — arxiv.org/html/2605.08044v1; Megabyte (prior local/global)
- Attention rank collapse — Dong et al., PMLR v139 (2105.02750);
  Mind the Gap, ICML 2025
- NoPE lineage — Kazemnejad et al. 2305.19466; Behind RoPE (causal
  mask induces position) — arxiv.org/html/2509.21042v1; Raschka
  architecture gallery (SmolLM3/Kimi K3 NoPE)

## 8. DeepSeek addendum (2026-09-21) — memory as a separate axis, and a constrained residual stream

Surveyed at Merna's request. Three things from DeepSeek's 2025–26
output matter to us: one is a direct external answer to our F3/G2
deficit pair, one corroborates the registered scale-graded readout,
one speaks to G1/F4. Nothing registered changes.

### 8.1 Engram / DSE — conditional memory via O(1) n-gram lookup (arXiv:2601.07372, Jan 2026, ACL 2026)

Mechanism (from the HTML v2 full text):

- Suffix n-grams over NFKC-compressed token ids (surjective V→V′
  projection, 23% vocab reduction; max order 3 — adding 4-grams was
  *slightly suboptimal* at fixed table budget, diluting frequent
  patterns). K multi-head multiplicative-XOR hashes per order into
  prime-sized tables; heads concatenated → e_t (8 heads, d_mem 1280
  at 27B).
- Injection: hidden state h_t is the Query, the retrieved e_t supplies
  K/V; depthwise causal conv (kernel 4, dilation = max order);
  Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ; **additive residual before
  attention/MoE**, at layer 2 (single-layer best) and layer 15 of the
  30-block 27B.
- Gate α_t = σ(RMSNorm(h_t)·RMSNorm(k_t)/√d) — retrieved memory that
  contradicts the context gates toward zero (handles hash collisions,
  polysemy). This is nearly isomorphic to our blend_gate vocabulary
  (scalar sigmoid alignment gate) — external validation of the gating
  genre we already use.
- Results: 3B-MoE/100B-token ablation val loss 1.808→1.768 (Δ0.04);
  27B iso-parameter iso-FLOPs vs pure MoE: MMLU +3.4, BBH +5.0,
  MATH +2.4; **RULER multi-query NIAH 84.2→97.0**; LongPPL-32k
  4.38→4.14. A U-shaped sparsity-allocation law between MoE compute
  and static memory, optimum ≈75–80% MoE share. Lookup is O(1) per
  token (deterministic addressing → prefetch; a 100B-param table fully
  offloaded to host DRAM costs 2.8% throughput).
- Mechanistic claim relevant to us: memory "relieves the backbone's
  early layers from static reconstruction," effectively deepening the
  network for reasoning.

**Why it matters here.** This is the attention-world's answer to
exactly the deficit pair we measured: F3 (no addressable bytes beyond
k≈4) + G2 (cliff identical at every layer — every layer re-encodes
local statistics instead of getting them for free). Our own n-gram
ladder (`runs_reprs/ngram_ce.json`: ctx-6 stupid-backoff 1.92 bits vs
model 1.30) already prices the static share of our CE; Engram is
evidence that the module which captures it is a *lookup*, not a better
mixer. And it is NOT attention-family: deterministic hash addressing,
no query-key inner products over positions, no content-scored routing
— charter-compatible as a mechanism class, with one honest tension:
it is memory, not geometry. If ever preregistered, two OPERA-native
instantiations keep the geometric vocabulary: (a) hash the last k
bytes → table of quaternion slot-corrections, gate-blended into the
readout (retrieved value is a rotor, not a vector);
(b) hash → per-level scalar priors feeding the scale-graded readout
(memory modulates allocation). Both keep routing content-blind (the
hash IS the address).

**Honesty box.** Gains are demonstrated at 3B–27B on 100B+ tokens;
the small-model datum is Δ0.04 val at 3B. Our 155M/250M regime is
more knowledge-starved (arguably larger relative gain) but unproven.
Orthogonal to scale-graded readout (allocation vs static recall) —
sequenced after it under the single-dose rule, never combined in one
arm.

### 8.2 DeepSeek-V4 (arXiv:2606.19348, Apr 2026) — two-scale sparse ladder + manifold-constrained residual + Muon

- **CSA/HCA hybrid (long-context):** CSA compresses every m=4 tokens'
  KV into one entry; a FP4 lightning indexer scores compressed blocks
  and attends sparsely to top-k (512 Flash / 1024 Pro), plus a
  128-token sliding-window branch for local detail. HCA compresses at
  m′=128 with *dense* attention over the compressed entries, plus the
  same window. Layers after the first two interleave CSA and HCA.
  Result at 1M context: 27% of V3.2's per-token FLOPs, 10% of its KV
  cache. DSA lineage (V3.2-Exp, 2025-09-29): the indexer uses a
  Euclidean-distance "decaying light" prior computed via FFT — a
  content-blind geometric distance prior.
- **Read-across:** a coarse-to-fine two-scale ladder (fine m=4 near
  detail, coarse m′=128 far context) + a local window is, in the
  attention world, a scale-graded readout. This *corroborates* the
  registered design (level-assigned slots, near levels fine / far
  levels coarse) externally; SG-H1..H4 are untouched and the prereg
  stays frozen. The decaying-light prior is the same philosophical
  family as our no-PE stance: distance structure as explicit geometry.
- **mHC (Manifold-Constrained Hyper-Connections;** standalone paper
  Xie et al. 2025, deployed as V4 §2.2): residual stream expanded to
  n=4 streams; X_{l+1} = B_l·X_l + C_l·F_l(A_l·X_l); **B_l projected
  onto the Birkhoff polytope** (doubly stochastic) via 20
  Sinkhorn–Knopp iterations → ‖B‖₂ ≤ 1, closed under multiplication,
  stable in deep stacks; A_l/C_l sigmoid-bounded; all three
  dynamically parameterized per token (RMSNorm-conditioned + learned
  static bias). Overhead 6.7% of the pipeline stage. Motivation:
  unconstrained hyper-connections amplify signal at depth
  (secondaries report ~3000× at 27B) and were blocking HC scaling.
- **Read-across:** our blend_gate update — `current = g·F(P) +
  (1−g)·current` — is the n=1 scalar special case of this scheme. Our
  G1 measurement (∏(1−g) = 0.021: the multiplicative highway is
  closed; identity survives by re-encoding, not transport) is
  precisely the failure class the Birkhoff constraint forecloses:
  doubly-stochastic mixing is non-expansive *and* non-contractive
  (the scalar doubly-stochastic matrix is exactly 1 — guaranteed
  pass-through). And n parallel streams mixed over the convex hull of
  permutations is an anti-rank-collapse device (our F4: PR 47.7→27 of
  512). Candidate for the depth-scaling phase: an n=2
  Birkhoff-constrained stream mixer replacing the scalar gate —
  geometry-native by construction, and *dynamic* per token, which is
  the dimension the falsified static gb arm lacked. Not now:
  single-dose discipline — the scale-graded readout is the registered
  next dose, and mHC-style mixing wants L≥8 to show its effect (our
  L=8 probe already exhibits the staircase, so the failure mode is
  present at reachable depths).
- **Muon** at 32T tokens in V4: further external validation of the
  house optimizer choice.

### 8.3 Not applicable / rejected for us

- MoE routing, FP8, pipeline/All-to-All sharding, QK-Clip
  interactions — hardware/scale specific, off-program.
- Sparse attention per se: attention-family, rejected by steer
  (2026-09-13). We take the design facts (two-scale ladder,
  content-blind distance prior), not the mechanism.
- NSA (arXiv:2502.11089, three-branch compressed/selected/window):
  genre already covered in §1–§2; noted as one more vote that
  multi-scale allocation beats any single scale.

### 8.4 What this changes

Nothing registered. Priority order stands: **scale-graded readout
first** (registered, awaiting implementation). Then, if F3-class
deficits persist — SG-H2 failing (k-decodability did not rise) is the
pre-stated signal — an Engram-style conditional memory is the
literature-backed next prereg: the one external mechanism that
directly supplies byte-level associative recall at O(1), with
`ngram_ce.py` already in hand as its ceiling calibration. mHC-style
constrained stream mixing is the depth-phase candidate for G1/F4.

### 8.5 Sources (DeepSeek addendum)

- Engram / DSE — arxiv.org/abs/2601.07372; code:
  github.com/deepseek-ai/Engram; ACL 2026 anthology entry
- DeepSeek-V4 — arxiv.org/abs/2606.19348 (mHC §2.2, CSA/HCA §2
  long-context); tech-report PDF on HF deepseek-ai/DeepSeek-V4-Pro;
  release note api-docs.deepseek.com/news/news260424
- mHC standalone — Xie et al., "mHC: Manifold-Constrained
  Hyper-Connections" (2025)
- DSA deployment — DeepSeek-V3.2-Exp announcement 2025-09-29
  (api-docs.deepseek.com); vLLM day-0 blog 2025-09-29; V3.2 report
  (Dec 2025)
- NSA — arxiv.org/abs/2502.11089
