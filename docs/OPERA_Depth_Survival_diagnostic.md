# OPERA depth-survivability diagnostic — 2026-09-09 (exploratory)

**Status: EXPLORATORY. No gates, no adoption.** Nothing in this document
changes the Path A arm set. Any intervention motivated by it requires its
own pre-registration first (house rule). This is measurement on the
existing incumbent, prompted by Merna's question: *is there something we
missed — data survivability from the first layer to the last, language
modelling, etc.?*

## Setup

- Model: `rx_fgate_wd` incumbent (bytes, vocab 259, d=512, nb=128,
  **L=2**, T=1024, pe=none, fold=left, rot=free, Muon recipe
  lr 0.02 / include fusion_gate / wd 0.01; BPB 2.037).
- Probe corpus: the study's own eval split, 128 sequences, the model's
  own CE mask (t < len−1). Dataflow instrumented per layer:
  `S_in[l]` (leaf states entering layer l), `P[l]` (Fenwick prefix
  readout — what the LM head consumes), `S_out[l] = g·MLP(P) + (1−g)·S_in`
  (states handed to the next layer), `g[l]` (scalar blend gate).
- Probes: closed-form ridge (linear), batch-level 80/20 split; a
  2-layer MLP probe (256 hidden, AdamW, 4 epochs) for the absence
  claims; an untrained seed-42 control of identical shape; an exact
  stupid-backoff n-gram ladder on the same positions.
- Scripts: `experiments/depth_survival.py`, `experiments/mlp_probe.py`,
  `experiments/ngram_ce.py`; JSONs in `runs_reprs/`.

## Findings

**F1 — Byte identity survives the whole stack.** Current-byte linear
decodability: S_in 0.998 / P_0 0.986 / P_1 0.955 / S_out 0.997. There is
no identity-loss problem. The "data survivability" question is answered
*negatively for identity, positively for addressability* (F3).

**F2 — The carry-forward gate is healthy at L=2.** Mean gates 0.39 / 0.47
(no saturation), identity attenuation ∏(1−g) = 0.32. The scalar highway
is not the bottleneck at this depth. (L=8 extension below.)

**F3 — The recency cliff (the main finding).** Linear decodability of the
byte at t−k from the final readout P_1: k=1 0.68, k=2 0.41, k=3 0.28,
k=4 0.21, k=8 0.11, k≥16 ~0.07 (flat). The untrained control already
leaks k=1 at 0.37 — training lifts k≤4 substantially but leaves k≥8 at
the untrained level: **training never extends the recency horizon.**
MLP probes on the same tensors (demonstrated power 0.90–0.98 on the
identity task; next-byte 0.523 vs the head's 0.605) recover only
0.17–0.18 at k=8–16: the mid-range byte information is **absent**, not
merely nonlinearly folded. English words are 4–8 bytes; mid-word
conditioning sits in the dead zone.
*Mechanism check:* grouping lag-k accuracy by the span of the Fenwick
block containing byte t−k shows **no span effect** (k=8: 0.104 in
span-8 blocks, 0.101 in span-512). The cliff is not "older bytes get
averaged with more siblings" — composition depth per se does not destroy
identity; the trained readout simply does not allocate addressable
capacity beyond ~4 bytes back. It is a learned allocation outcome, i.e.
exactly what a content-addressable readout (R1) would have to change.

**F4 — Training collapses the readout rank.** Participation ratio of the
prefix covariance: untrained P 223–260 → trained P_0 47.7 → P_1 **27**
(of 512). The final readout concentrates into a narrow subspace, and
depth narrows it further. The LM head is close to linearly saturated on
it (next-byte ridge 0.544 vs argmax 0.605), so this is not a head
bottleneck — it is the representation the composer chooses to emit.
Connects to the closed scale-tied-operator observation (trained level
deltas grow with depth): depth currently *contracts* the operator's
output, not just reuses it.

**F5 — Position information is absent, period.** 32-bin position decode:
embeddings 0.031 (chance, by construction — pe=none), final readout
0.057 linear / **0.059 with a 2-layer MLP probe** (probe class shown to
reach 0.9+ on identity from the same tensors), states 0.047. The
untrained control scores 0.058–0.065: the tree's intrinsic position
leakage is tiny and training adds nothing. This does not violate I1 (it
is the design), but it quantifies *how little* usable position signal
the structure alone provides — even nonlinearly.

**F6 — Deterministic loss texture from the Fenwick readout shape
(smaller than first reported).** Per-position CE vs prefix node count
count(t): Spearman −0.31 trained, −0.45 untrained. By count: count=1 →
CE 1.654, count=10 → 0.952 (count(t)=popcount(t+1), the ruler
sequence). Matched-position check (context length controlled): at
power-of-two positions (t+1 = 2^a — the prefix is a single block, zero
fold steps) CE is **+0.18…+0.28 bits above the ±30-position
neighborhood**. *Correction (2026-09-13):* the initial "smoothing
bound" of 4.2% of CE (every position lifted to its ±30-neighborhood
mean) conflates natural content-difficulty fluctuation with the Fenwick
effect. The Fenwick-attributable part — lifting the count≤3 rows
(~17% of positions, of which count=1 is ~1%) to the count≥4 level — is
≈ **0.5–1% of CE**. Mechanism check: grouping by the *last block's*
span shows CE flat (1.129–1.179 for spans 1–32) and k=1 decodability
nearly flat (0.61–0.69) — the anomaly is **not** recent bytes being
buried in a large block (composition preserves addressability,
F3-span-check); it is specific to the zero-fold-step single-node
readout distribution the head almost never sees in training. The
texture and the recency cliff (F3) are **separate** phenomena.

**F7 — The head consumes the blurriest readout.** At L=2, layer 0's
readout has sharper recency than the final layer's (k=1: 0.83 → 0.68;
k=2: 0.51 → 0.41) while per-layer CE improves 1.617 → 1.304. The L=8
extension shows the full shape: recency *rises* through mid-depth
(P_2..P_4 peak, k=1 ≈ 0.80–0.82, k=2 ≈ 0.49–0.50) and then falls at
the last layer (P_7: 0.66 / 0.41 — the worst since P_0); identity
(0.966 mid → 0.877 final) and readout rank (36 mid → 22 final) decline
the same way. Meanwhile per-layer CE improves monotonically at every
depth (L0 2.378 → L7 1.473, still ~−0.08/layer at the end — no
saturation). The last layer systematically trades addressable detail
for predictive compression.

**Calibration — the model is far better than windowed statistics.**
Exact stupid-backoff n-gram ladder on the same 128 sequences (16.4M-id
train stream): ctx-1 3.64, ctx-2 3.02, ctx-3 2.42, ctx-4 2.05,
ctx-6 1.92, ctx-8 2.25, ctx-12 3.25 bits/unit (best 1.92) vs the
model's **1.30**. So the fold retains strong *distributional* content of
the history (0.6 bits beyond the best n-gram) while losing *addressable*
old-byte identity (F3). The readout behaves as a compressor, not a
memory.

## Interpretation (post-hoc, labelled as such)

The missed thing is not survivability of the input, and not the optimizer
(closed 2026-09-09) — it is **what the readout is**: a fixed-shape
bottom-up summary that compresses well but addresses poorly. Attention
keeps every old token addressable at readout (KV cache); the Fenwick
fold exposes one folded sum per position, and the fold shape itself
stamps a ruler-sequence texture on capacity (F6). The 0.62-bit margin
over n-grams says compression is working; the cliff (F3), the rank
collapse (F4) and the texture (F6) say addressing is the weak axis.
Depth does not create any of these — the cliff is identical at every
layer (G2) — but it *chooses* the same trade again at every layer: the
final readout is the most compressed of all (F7/G3), and the pass-through
highway amplitude is gone by L8 (∏(1−g)=0.02) with identity surviving
only by re-encoding (G1).

## Registered follow-ups (status after the 2026-09-13 steer)

- **REGISTERED — scale-graded readout** (`OPERA_ScaleGraded_prereg.md`):
  level-assigned subspaces in the fold; the tree's dyadic geometry
  becomes state geometry. Content-blind, parameter-free, gates SG-H1..H4.
  This supersedes R1/R2 below under the geometry-first steer (no
  attention-family mechanisms).
- R1 (content-addressable fold, attend family) — off-program (steer;
  also falsified in v8.0). The survey's T1 query-gates likewise
  off-program; the house's own adaptive-λ design stays on the shelf,
  untested.
- R2 (local-guarantee decomposition) — subsumed by scale grading (group
  0 owns the finest scales at every position) if SG-H1/H2 pass;
  standalone revival only via new registration.
- **R3 — rank through depth as a standing instrument** in Path A logs
  (monitor-only; zero risk to the arm set).

None of these may enter Path A without passing their own gates.

## L=8 extension (Path-A depth): `runs_reprs/l8_probe`

Probe model: same recipe (Muon 0.02, include fusion_gate, wd 0.01), bytes
T=1024, d=512/nb=128, **L=8**, 1500 steps (half the incumbent's budget —
absolute levels undertrained, per-layer trends internally valid; final
eval PPL 4.88, extrapolation 1025–2048 PPL 4.85 — still flat).

**G1 — Gate staircase, no saturation, highway closed anyway.** Mean
blend gates by layer: 0.16, 0.27, 0.33, 0.37, 0.46, 0.40, 0.48, 0.54 —
monotone-ish rise, zero positions >0.9 anywhere. The model learns
"early layers pass through, late layers rewrite." Yet the identity
attenuation ∏(1−g) = **0.021**: only ~2% of the embedding's amplitude
reaches the last layer through the carry channel. Byte identity still
decodes at 0.88–0.96 everywhere — it survives by *re-encoding through
the readout each layer*, not by the highway. Direct answer to the
survivability question: **identity survives; the pass-through channel
does not.**

**G2 — The recency cliff is a layer signature, not a depth effect.**
k=8 decodability is 0.10–0.11 at *every* layer P_0..P_7 (k=16: 0.07–0.08);
no depth ever allocates addressability beyond ~4 bytes. Whatever depth
adds (CE 2.38 → 1.47), it never buys old-byte addressability.

**G3 — Mid-depth is sharpest; the final readout is the most compressed.**
Recency k=1 by layer: P_0 0.64, then 0.79/0.82/0.80/0.80/0.76/0.75,
P_7 **0.66**; identity 0.966 mid → 0.877 final; readout PR 36 mid →
**22** final (top-1 0.16). The stream feeding the LM head is the
narrowest, most context-mixed representation in the stack.

**G4 — Texture and position-blindness persist at depth.** D6 Spearman
−0.333 (L=2: −0.308); position 32-bin decode 0.045 (chance 0.031).

Artifacts: `runs_reprs/depth_survival_{rx_fgate_wd,control_init,l8_final,
l8_step500}.json`, `runs_reprs/mlp_probe_rx_fgate_wd.json`,
`runs_reprs/ngram_ce.json`, `runs_reprs/l8_probe/` (checkpoint +
results). Step-500 tables (`l8_step500`) are superseded by the final
ones; the transient mid-training dip (P_2 k=1 0.35) and the extreme
early rank collapse (S_in PR → 9) wash out by step 1500 — middle-layer
representations are the last thing training stabilizes.
