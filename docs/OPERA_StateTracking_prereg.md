# OPERA state tracking — pre-registered protocol: group word problems and the SO(3) rematch

**Version:** 1.0 — 2026-09-07
**Status:** PRE-REGISTERED for the arms in §3. Honest scope of that claim
is stated in §0: a parity harness check and one A5 pilot ran *before*
this document was written. Everything in §3 is registered before its run.
**Hardware:** MacBook M4, 16 GB, CPU. No Kaggle quota is consumed, so
this study does not compete with Path A.

---

## 0. What already ran before registration (disclosed)

1. **Harness validation, parity (C_2), OPERA so3, d=64/nb=16/2 layers,
   800 steps.** In-length accuracy 1.0000; length generalization
   1.0000 / 0.9970 / 0.9009 / 0.7003 at T = 32/64/128/256. Purpose was
   to prove the harness trains and evaluates at all. Reported as a
   harness check, never as a result.
2. **A5, OPERA so3, LM-default node (`norm_mode='layer'`,
   `act_mode='tanh'`), d=128/nb=32/2 layers, 4,000 steps.** Result:
   in-length accuracy **0.0778** (chance 0.0167 — 4.7x chance), loss
   plateaued at 3.82 against chance ln(60)=4.094; length generalization
   0.0778/0.0462/0.0321/0.0243/0.0202 at T=32..512. **It learned
   something and did not learn the group.**

   *Diagnosis, recorded before the follow-up ran:* this is a defect in
   the PILOT'S CONFIGURATION, not evidence about OPERA. The exact A_5
   construction requires the node to be quaternion multiplication, and
   two LM defaults destroy that representation — `norm_mode='layer'`
   subtracts a mean across all d dimensions (mixing separate rotor
   blocks) and `act_mode='tanh'` squashes components off the unit
   sphere. The geometry-clean pair is `norm_mode='blockrms'` (per-block
   RMS: exactly SO(3)-equivariant, and a no-op on unit quaternions) with
   `act_mode='linear'`. Both are documented in the Mechanisms Guide §4
   as the equivariance-preserving options, and both were measured as
   *losses* on language modeling (blockrms at +4 PPL) — which is why
   they are not defaults, and why this task is their home axis.

   **Protocol amendment made BEFORE the follow-up run:** §3's arms are
   defined at geometry-clean settings, and the pilot's configuration is
   promoted to a registered ablation (arm A0) so the node-geometry
   question is answered rather than assumed. This changes what "arm A"
   means and is therefore recorded here, timestamped ahead of results.

Group construction is verified independently of any model: closure,
identity, inverses, associativity on a random sample, and the
Latin-square property, for all five groups; running-product targets are
cross-checked against an independent scan (`opera_lm.state_tracking`).

## 1. Motivation

The project's five falsified readout arms and the §4.8 data-regime
result all say the same thing: OPERA does not win on language-model
perplexity. State tracking is the axis where it has a *structural*
advantage that no amount of transformer kernel maturity erases, and it
has never been measured.

**The complexity-class argument.** The word problem for a group G is
NC^1-complete when G is non-solvable (Barrington, 1989) and in TC^0
when G is solvable. A fixed-depth transformer is a constant-depth
threshold circuit (TC^0) and cannot solve A_5 for growing T (Merrill &
Sabharwal, arXiv:2404.08819). A parallel-scan SSM with diagonal,
positive transitions is abelian in flight and collapses to TC^0 too —
the reason Mamba fails parity, and the reason the negative-eigenvalue
(arXiv:2411.12537) and Householder-product (DeltaProduct,
arXiv:2502.10297) lines exist.

**Why OPERA is not in that box.** (a) Its compute graph is *log-depth
in T* — the tree over a length-T prefix has ceil(log2 T) composition
levels, so circuit depth grows with the sequence. Log-depth is the home
of NC^1. (b) **A_5 is the rotation group of the icosahedron: a finite
subgroup of SO(3), order 60.** An OPERA block state is a rotor in
SO(3); the compose node's geometric product is quaternion
multiplication — the group operation itself. So an exact construction
exists: embed the 60 icosahedral rotations as token rotors and the node
composes them by definition.

This is the only claim in the project where "OPERA can represent the
solution" is a theorem rather than a hope. The empirical question is
strictly whether SGD finds it.

## 2. Instrument

`opera_lm/state_tracking.py`. Sequences of group elements; target at
position t is the running product g_1...g_{t+1}, aligned with no shift
to OPERA's logits[:, t] (which reads prefix 0..t). Accuracy over ALL
positions, not just the last. Chance = 1/|G|.

Train at T=32; evaluate at T = 32, 64, 128, 256, 512 (1x to 16x).
Length generalization is the whole measurement: any model can memorize
a fixed-length map, so in-length accuracy alone proves nothing.

**Depth-exposure confound, quantified in advance.** Training at T=32
builds tree levels 0–5 only; T=256 requires levels 6–8, and 75.4% of
its positions read at least one never-built-level block
(`opera_lm.level_exposure`). The pre-stated null for "no depth
generalization at all" is therefore

    floor = p_clean * 1.0 + (1 - p_clean) * (1/|G|)

with p_clean the fraction of positions reading only trained levels.
Accuracy is reported against BOTH chance and this floor. For parity at
T=256 the floor is 0.623 and the harness check scored 0.7003 — i.e. it
recovered ~20% of the floor-to-ceiling gap. This confound applies to
every OPERA arm equally and does not apply to the transformer arm;
that asymmetry is a limitation, recorded here, not a result.

## 3. Registered arms and gates

All arms: d=128, nb=32, 2 layers, AdamW + OneCycle, lr 3e-3, batch 64,
seed 42, budget set by the §0 pilot. One variable per run.

| arm | architecture |
|---|---|
| A | OPERA, `rot_mode='so3'`, **`norm_mode='blockrms'`, `act_mode='linear'`**, `pe_mode='none'`, fold left |
| B | OPERA, `rot_mode='free'`, otherwise identical to A |
| A0 | OPERA, `rot_mode='so3'`, LM-default node (`layer`/`tanh`) — the §0 pilot, promoted to a node-geometry ablation |
| C | causal transformer, matched d/layers, **given** learned absolute PE |

Arm C is deliberately given a positional encoding: this benchmark tests
the complexity class, not the position mechanism, and withholding PE
would weaken the baseline for the wrong reason.

**H1 (non-solvable separation) — the headline.** On **A5**, OPERA
(better of A/B) exceeds the transformer at T >= 128.
*Decision:* PASS if `acc_opera(128) - acc_tf(128) >= 0.10` AND the same
sign holds at 256. Report all lengths regardless.

**H2 (the SO(3) rematch) — the retired constraint on its home axis.**
On **A5**, `rot_mode='so3'` beats `rot_mode='free'`.
*Decision:* PASS if `acc_A - acc_B >= 0.05` at T=32 (in-length, where
the depth confound is absent). This is the pre-committed third test of
a constraint nulled twice on language modeling; a PASS is the first
positive result for it and is reported as such. A NULL closes the SO(3)
line permanently — stated in advance, binding either way.

**H3 (solvable control).** On **A4/S4** (solvable, in TC^0) the
transformer should NOT be separated from OPERA by the H1 margin.
*Decision:* the H1 separation is credited to non-solvability only if
H3's margin is smaller than H1's on the same protocol. Without this
control an H1 pass could be any capacity difference; with it, the
separation tracks the complexity class.

**H2b (node geometry) — registered with H2, decided the same way.**
On **A5**, arm A (geometry-clean node) beats arm A0 (LM-default node).
*Decision:* PASS if `acc_A - acc_A0 >= 0.05` at T=32. A0's measured
in-length accuracy is 0.0778, so the comparison point is fixed in
advance. A PASS says the equivariance-preserving options nulled on
language modeling are load-bearing here — the same "home axis" claim
H2 makes for SO(3), on a second flag. A NULL says the pilot's
configuration was not the limiting factor and the §0 diagnosis was
wrong, which must then be stated plainly in the outcomes.

**H4 (depth honesty).** OPERA's A5 accuracy at T=256 is compared to
the §2 floor. If it does not exceed the floor, OPERA has learned the
group *at trained depths only*, and no length-generalization claim is
made — the result is reported as in-length state tracking plus a
depth-extrapolation failure, with §2's remedy (state passing) named as
future work rather than run here.

## 4. Pre-stated interpretation

An H1+H3 pass is a genuine architectural result for OPERA that does not
depend on beating a transformer on perplexity, and it is the strongest
available reframing of the paper (see
`docs/OPERA_Improvement_Survey_2026-09.md` §5).

An H1 null means the log-depth argument does not survive contact with
SGD at this budget, and the correct conclusion is that OPERA's
advantage is theoretical rather than trainable — which is a publishable
negative result on the project's own falsification standard, and should
be written up as one.

## 5. Limitations acknowledged at registration

Single seed (project convention; the noise band on these accuracies is
NOT yet measured and multi-seed is the first follow-up). Single model
scale. CPU-only, so budgets are small and a null may be
budget-limited — any null is reported with the training curve so
under-training is visible. Arm C is a compact transformer written for
this benchmark, not the project's tuned `TransformerBaseline`. A5 is
tested; S5 (order 120, also non-solvable) is left to follow-up.

---

## Outcomes — session 1, 2026-09-07 (CPU, M4)

### Verdict: INCONCLUSIVE — budget-limited, not architecture-limited.

**No hypothesis is decided.** The registered gates are not evaluable
because *neither* architecture learned A5 at this budget, so there is
nothing to discriminate. Recorded in full anyway.

**The observation that matters — a cliff at exactly one composition.**
A5, T=8, 2,500 steps, accuracy by position (position t needs t
compositions; chance 0.0167):

| arch | params | pos0 | pos1 | pos2 | pos3 | pos4 | pos7 |
|---|---|---|---|---|---|---|---|
| OPERA so3, geometry-clean | 205,760 | 1.0000 | 0.9725 | 0.0670 | 0.0192 | 0.0155 | 0.0178 |
| transformer (learned PE) | 936,508 | 1.0000 | 0.9915 | 0.0760 | 0.0178 | 0.0207 | 0.0155 |

Both solve one group product almost perfectly and both collapse to
chance at two — the transformer with **4.5x the parameters**. The wall
is at the same place for both, so at this budget A5 discriminates
nothing. This matches the literature's setup requirements (these tasks
are trained far longer, usually with a length curriculum); it is a
statement about 2,500 CPU steps, not about either architecture.

**Supporting runs.**

- *Parity (C_2), OPERA so3, 800 steps:* in-length **1.0000**;
  1.0000 / 0.9970 / 0.9009 / 0.7003 at T=32/64/128/256. The model does
  learn a solvable-group tracking task perfectly in-length. The decay
  is quantitatively consistent with the depth-exposure floor of §2
  (floor 0.623 at T=256, observed 0.7003 — recovering ~20% of the
  floor-to-ceiling gap), which is the first direct cross-validation of
  `opera_lm.level_exposure` against a trained model.
- *H2b (node geometry), A5, T=32:* arm A (blockrms/linear) **0.0505**
  vs arm A0 (layer/tanh) **0.0778**. Directionally NULL — in fact
  reversed — but both sit near chance, so the comparison carries no
  information and H2b is recorded as not evaluable, not as a null.
  **The §0 diagnosis was wrong**: the pilot's node configuration was
  not the limiting factor.
- *msup (node-level span supervision), A5, T=8:* supervising every tree
  node against its span's exact group element helps but does not
  rescue — pos2 0.0670 -> 0.1523, pos3 0.0192 -> 0.0432. Consistent
  with msup's LM win being real and with the failure being upstream of
  supervision.
- *Manifold-matching (falsified hypothesis, recorded):* level
  statistics show level 0 is a different distribution from levels >=1
  (mean |h| 10.68 vs 6.93; per-block |h| std 0.629 vs 0.050) while
  levels 1-3 are near-identical (the node IS close to a fixed point
  once inside). Hypothesis: a weight-tied node fails at depth >= 2
  because it only ever trained on token-embedding inputs. **Tested and
  falsified** — projecting embeddings onto the node's manifold
  (per-block RMS) changed nothing (pos2 0.0920 -> 0.0777). Recorded
  because the hypothesis is natural and someone will re-propose it.

### What this does and does not license

It does NOT license "OPERA cannot do state tracking," and it does not
license the reverse. The theoretical argument of §1 is untouched — it
was never a claim about trainability at 2,500 CPU steps.

### Required before any claim (session 2)

1. **Length curriculum** (T=2 -> 4 -> 8 -> ...), the standard setup in
   this literature and absent here; the pos0/pos1-only success suggests
   the model needs to be walked up in depth.
2. **10-50x the step budget**, on GPU rather than CPU.
3. **A4/S4 solvable controls** (H3) — currently unrun, and without them
   any A5 separation is uninterpretable.
4. **rot_mode='free' arm** (H2) — unrun; the SO(3) rematch is still
   open, neither passed nor nulled.
5. **Multi-seed**, since single-seed accuracies near chance are noise.

Until those run, the correct public statement is: "the instrument is
built and verified; the benchmark is budget-limited at CPU scale; the
SO(3) rematch has not yet been decided."
