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

**F4 — Training collapses the readout rank.** Participation ratio of the
prefix covariance: untrained P 223–260 → trained P_0 47.7 → P_1 **27**
(of 512). The final readout concentrates into a narrow subspace, and
depth narrows it further. The LM head is close to linearly saturated on
it (next-byte ridge 0.544 vs argmax 0.605), so this is not a head
bottleneck — it is the representation the composer chooses to emit.
Connects to the closed scale-tied-operator observation (trained level
deltas grow with depth): depth currently *contracts* the operator's
output, not just reuses it.

**F5 — Position information is nearly absent, linearly.** 32-bin position
decode: embeddings 0.031 (chance, by construction — pe=none), final
readout 0.057, states 0.047. The untrained control scores 0.058–0.065:
the tree's intrinsic position leakage is tiny and training adds nothing.
This does not violate I1 (it is the design), but it quantifies *how
little* usable position signal the structure alone provides.

**F6 — Deterministic loss texture from the Fenwick readout shape.**
Per-position CE vs prefix node count count(t): Spearman −0.31 trained,
−0.45 untrained. By count: count=1 → CE 1.654, count=10 → 0.952
(0.70 bits/unit span; count(t)=popcount(t+1), the ruler sequence).
Matched-position check (context length controlled): at power-of-two
positions (t+1 = 2^a — the fold degenerates to a single block) CE is
**+0.18…+0.28 bits above the ±30-position neighborhood**. This is a
real structural handicap on ~17% of positions (count ≤ 3), not a
context-length artifact.

**F7 — Depth trades recency for prediction.** Layer 0's readout has
*sharper* recency than the final layer's (k=1: 0.83 → 0.68; k=2:
0.51 → 0.41) while per-layer CE improves 1.617 → 1.304. Deepening
washes out local detail the first layer already had. (L=8 extension
below tests whether this continues monotonically at Path-A depth.)

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
collapse (F4) and the texture (F6) say addressing is the weak axis. At
L=2 all three show up already; they are properties of the layer, not of
depth — depth only amplifies (F7, F4).

## Registered follow-ups this motivates (NOT run; each needs a prereg)

- **R1 — Content-addressable / recency-weighted fold** (the attend
  family, v8.0-era losers, now with a mechanistic motivation and a new
  rung: bytes). Prereg gate: BPB ≥1% better than incumbent with no
  extrapolation regression; kill-switch on probe horizon (k=8 decodability
  must rise, else the mechanism isn't what moved the loss).
- **R2 — Readout-shape debias**: compensate count(t) in the fold mixing
  so capacity is position-uniform (targets F6 directly; small, isolated).
- **R3 — Rank through depth as a standing instrument** in Path A logs
  (monitor-only; zero risk to the arm set).

None of these may enter Path A without passing their own gates.

## L=8 extension (pending)

`runs_reprs/l8_probe` (same recipe, 1500 steps, grad-checkpoint layer)
trains as this is written; per-layer tables (identity/recency k≤16/rank/
CE/gates for P_0..P_7) will be appended when it completes.
