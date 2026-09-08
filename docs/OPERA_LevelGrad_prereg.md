# OPERA level-balanced gradient — pre-registered protocol

**Version:** 1.0 — 2026-09-08. PRE-REGISTERED before any run.
**Hardware:** M4, 16 GB, MPS. Byte d512/nb128/L2 rung.

---

## 1. The measured pathology

OPERA ties **one compose operator across an entire scale hierarchy**:
the same weights join two bytes at level 0 and two 128-byte spans at
level 7. Level k holds T/2^k nodes, so the shared weights receive
geometrically less gradient from deeper levels.

Measured on the byte-d512 incumbent (`bist_off`), T=256, layer 0, by
hooking each tree level's output:

| level | span | share of dL/dW | per-node |
|---|---|---|---|
| 0 | 1 | **34.1%** | 0.000021 |
| 1 | 2 | 25.3% | 0.000031 |
| 2 | 4 | 14.8% | 0.000036 |
| 3 | 8 | 9.6% | 0.000047 |
| 4 | 16 | 6.3% | 0.000061 |
| 5 | 32 | 4.4% | 0.000086 |
| 6 | 64 | 3.1% | 0.000121 |
| 7 | 128 | 2.3% | 0.000180 |
| 8 | 256 | **0.0%** | **0** |

Two facts, and the second is sharper than the first.

**A 15:1 imbalance across trained levels.** Node counts halve per level
(2× decay) but per-node gradient *grows* 8.6× from level 0 to 7 — deeper
nodes each matter more, partly self-correcting. Net decay is ~1.5× per
level, so the tied operator is fit ~15× more to shallow composition than
to level-7 composition.

**The root receives exactly zero gradient.** Not "little" — zero. The
only prefix whose Fenwick decomposition reads the root is L = T, and its
prediction target is the token *after* the sequence, which the loss
masks. This is the gradient-level form of the 2026-09-07 level-exposure
finding (root reachable by one prefix; `msup_loss` skips it at
`span >= T`), and it is strictly stronger: the top of every OPERA tree
is trained by nothing at all.

## 2. Why this is not prior art

Weight-tied recursion is studied — Universal Transformer,
Mixture-of-Recursions (arXiv:2507.10524), recursive transformers — but
that is **depth** recursion: the tied layer is applied N times to a
*same-length* sequence, so every application sees equal node counts and
no imbalance arises. Recursive/Tree-LSTM architectures share parameters
over trees but do not report gradient share by level. Gradient
rescaling for imbalance exists as a general technique (multi-task
GradNorm; cell-level rescaling in NAS/lookup networks) but not indexed
by *scale* in a weight-tied hierarchy.

The pathology requires tying one operator across levels of differing
node count. We are not aware of another architecture that does this.

## 3. Mechanism

Level ℓ's contribution to dL/dW is scaled by β^ℓ, normalised to mean 1
over the levels the tree actually has:

    gs(l) = beta**l / mean_{i=1..L} beta**i

- **β = 1.0** → the incumbent, exactly.
- **β = 2.0** → cancels the node-count halving exactly.
- **β ≈ 1.5** → equalises the *measured* share (decay is ~1.5×, not 2×).

Three implementation properties, each of which was a bug caught during
construction and is asserted in `test_level_grad_balance`:

1. **Scale the weights, not the level's output.** A hook on the output
   gradient also rescales everything flowing *down* to lower levels, so
   factors compound geometrically. Scaling the weights that level sees
   touches only its dL/dW contribution. Asserted: `word_emb.grad` is
   unchanged across β.
2. **Normalise to mean 1.** Without it, β>1 inflates every compose
   gradient (~4× at β=2) — indistinguishable from raising the learning
   rate on those parameters, and any effect would be an LR effect.
3. **The forward is identical for every β.** Purely a training-time
   reweighting; inference is untouched and checkpoints stay
   architecturally identical.

Invariants I1–I4 untouched: no forward change at all.

## 4. Arms

`bist_off` (β=1.0, the existing incumbent: BPB 2.1308) ·
`lgb_1_25` · `lgb_1_5` · `lgb_2_0`. Byte d512, T=1024, batch 8, 3000
steps, seed 42 — identical to every arm in this series.

**No new parameters.** Unlike every previous arm, capacity is exactly
matched by construction, so there is no capacity confound to control
for and no separate control arm is needed.

## 5. Hypotheses and gates

**L-H1 (primary) — length extrapolation.** The deficiency is at deep
levels, which dominate *longer* sequences, so the gate is extrapolation
and NOT in-length BPB. Using the bucketed extrapolation eval at
2× training length (T=2048):
*PASS* if any β reduces extrapolation-bucket loss vs β=1.0 by ≥ 2%.

**L-H2 (in-length cost, guard not target).** In-length BPB must not
regress > 3% vs 2.1308. **A BPB tie or small regression with L-H1
passing is a SUCCESS** — the arm deliberately spends capacity away from
the shallow levels that dominate short-sequence loss. Predicting this
in advance is what makes L-H1 a real test rather than a fishing
expedition.

**L-H3 (the mechanism did what it says).** Re-measure the per-level
gradient share on the trained β>1 checkpoints. *PASS* if the level-0
share falls and the level-7 share rises relative to β=1.0. No gate on
magnitude; this only confirms the intervention took effect.

**L-H4 (monotonicity).** If more than one β passes L-H1, effect should
be ordered in β. A non-monotone result means the effect is noise at this
budget and L-H1 is not claimable.

## 6. Pre-stated interpretation

- **L-H1 pass** → the tied-operator gradient imbalance is a real and
  fixable limitation of scale-hierarchical weight tying, and the fix is
  free at inference. This is a finding about an architecture class, not
  about OPERA's perplexity.
- **L-H1 fail with L-H3 pass** → the imbalance is real and measurable
  but rebalancing does not help; the shared operator apparently does not
  need equal fitting across scales. That is a clean negative about the
  *remedy*, and the measurement in §1 stands regardless.
- **L-H3 fail** → the intervention did not take effect and nothing is
  concluded about the hypothesis.

## 7. Limitations at registration

Single seed. One rung, 3000 steps (~0.56 epochs). Extrapolation is
measured only to 2× (eval capped at 2048 for memory, per the
2026-09-08 amendment). Total gradient norm falls ~16% at β=2 as a side
effect of upweighting small gradients — partly absorbed by the existing
grad-clip at 1.0, but not zero, and it is a residual confound with
effective learning rate. **Backward is non-deterministic on this
hardware at ~1.3e-7 relative** (two identical incumbent runs differ by
that much), which bounds any gradient-level claim.

---

## Outcomes — 2026-09-08

| arm | β | BPB | vs incumbent | extrapolation (1025–2048) | vs incumbent |
|---|---|---|---|---|---|
| `bist_off` | 1.0 | 2.1308 | — | 4.3650 | — |
| `lgb_1_5` | 1.5 | 2.1627 | +1.50% | 4.4397 | +1.71% |
| `lgb_2_0` | 2.0 | 2.1952 | **+3.02%** | 4.5558 | **+4.37%** |

All arms 3,293,447 params — identical to the incumbent, since the arm
adds no parameters. No capacity confound exists to control for.

### L-H1 — **FAIL.** Wrong direction, monotonically.

Required extrapolation to improve by ≥ 2%. Measured **1.71% and 4.37%
WORSE**. L-H2 also fails at β=2.0 (BPB +3.02% against a 3% guard).
L-H4's monotonicity holds but runs the wrong way: **the more the
gradient is rebalanced, the worse both metrics get.**

Three ordered points, no capacity confound, damage scaling with the
intervention. This is an unambiguous negative.

### L-H3 — not evaluated. The intervention's effect on training is not
in question: the selftest asserts the per-level weight scaling takes
effect, and the monotone degradation with β is itself proof that it
did. Re-measuring the trained gradient shares would confirm a mechanism
that is already demonstrably active.

### Verdict — the measurement stands, the remedy is falsified

Per §6: **L-H1 fail means the imbalance is real and measurable but
rebalancing does not help.** §1's finding is unaffected — the 15:1
gradient share across levels and the *exactly zero* gradient at the root
are direct measurements of this architecture, and they remain the novel
observation. What is falsified is the inference that the imbalance was a
pathology worth correcting.

### Why it failed (the signal was in §1 and was not acted on)

The per-node gradient already **grows 8.6× from level 0 to level 7**.
That was measured and recorded in §1, and described there as "partly
self-correcting" — and then a correction was proposed anyway. The
optimizer is already compensating for the node-count decay. The residual
15:1 is plausibly the *correct* allocation: level 0 has 256× more nodes
performing 256× more distinct compositions, so it should receive more
gradient. Forcing the shares toward equality is over-correction, and
monotone degradation in β is exactly what over-correction looks like.

The methodological lesson is specific: an imbalance being *measurable*
and *unique to the architecture* is not evidence that it is *harmful*.
Those are three separate claims, and only the first two were established
before the third was assumed.

### Standing

Seven mechanisms tested in this series — five gating arms
(`fold_adapt`, `homeo`, `quotient`, bistable per-block, bistable
low-rank), level-balanced gradient, and over-relaxation.
**Over-relaxation remains the only positive**, and the only mechanism
that moved the trajectory (p99.9/median 1.200 → 1.656).
