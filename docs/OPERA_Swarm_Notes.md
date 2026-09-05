# OPERA Swarm Notes: Division-of-Labor Arms for a Geometric LM

*2026-08-04 — literature scan + arm proposals. Status: exploratory notes,
not a pre-registration. Trigger: the question "monolith or division?" —
the brain's answer is specialized subnetworks with dense integration;
OPERA's compose node is a candidate integration mechanism.*

## 1. What the literature says

**The "ensemble geometric model" intersection is empty.** Geometric
algebra architectures exist and are producing results in adjacent
modalities — [Geometric Clifford Algebra Networks](https://brandstetter-johannes.github.io/publication/ruhe-2023-cgans/)
(Ruhe et al., ICML 2023), CliffordNet in vision — but none combine with
ensembles/modularity. Meanwhile the modularity/ensemble literature is
entirely built on flat vector spaces. **The intersection is white
space**: nobody has used a geometric composition law as the fusion
mechanism for an ensemble of specialist models. OPERA's compose node is
exactly such a law.

**The swarm idea has a real, scored track record:**

- **Mixture-of-Agents** ([Wang et al. 2024](https://arxiv.org/abs/2406.04692)):
  layered LLMs propose-and-refine; beats single models at matched
  params. Inference-time only. But see the honest counterweight:
  [Rethinking MoA](https://arxiv.org/abs/2502.00674) (2025) finds gains
  shrink under careful controls — the aggregator is where swarms die.
- **Branch-Train-Merge** (Li et al.; see the
  [MoE survey](https://arxiv.org/pdf/2407.06204v1)): train domain
  specialist LMs *independently* (no shared parameters, no
  communication), ensemble or merge at inference. **This is the
  no-funds-friendly design**: specialists train separately on one
  laptop, sequentially, with resume. BTX (Sukhbaatar et al. 2024)
  merges the specialists into one MoE afterward.
- **Model soups / model merging** ([survey](https://arxiv.org/abs/2603.09938)):
  weight-averaging of same-basin models. Not applicable across
  differently-seeded OPERA trees (different basins), but applicable to
  same-run checkpoints (free).
- **[Modular deep learning survey](https://arxiv.org/abs/2302.11529)**
  (Pfeiffer et al. 2023): the taxonomy — computation vs parameter vs
  data modularity. OPERA swarm arms below span all three.
- **[LLM ensemble survey list](https://github.com/junchenzhi/Awesome-LLM-Ensemble)**:
  token-level / span-level / response-level ensembling. Token-level
  (probability fusion) is the one compatible with OPERA serving.

**Standing warning (project's own record + the MoA rethink):** the
aggregator is the hard part. Every arm below names its aggregation
mechanism and its kill criterion before any run.

## 2. Why OPERA is an unusually good swarm substrate

1. **The compose node already fuses two states geometrically** —
   learned per-block maps + geometric product + gates. Nothing in the
   mechanism requires both children to come from the same tree.
2. **The Fenwick readout gives per-position states from any tree** — so
   specialists and composer share a common interface (per-position
   prefix states, d-dimensional spinor blocks).
3. **Position-is-structure is per-tree**: specialists can have
   different depths/widths and still expose the same state interface.
4. The role-split diagnostic already proved the monolith's weak point
   (retrieval: assistant-gap 0.79 nats); a retrieval specialist +
   generalist is a concrete division of labor to test.

## 3. Candidate arms (cheapest first)

### S1 — Logit-ensemble of specialists (product of experts; zero new training code)
Train K small OPERA specialists on data slices (e.g. smoltalk subsets:
everyday dialogue / math-code / summarize-rewrite), then ensemble at
the logit level: geometric mean of the specialists' next-token
distributions (literally the *geometric* mean — on-brand, and the
standard product-of-experts fusion). No new model code; serving is
K incremental decoders + a log-add.
- **Cost**: 3 × ~7M-param runs (~2-3h each on M4 at reduced steps) +
  eval. Overnight-friendly, fully resumable.
- **Gate**: the ensemble beats ONE 21M monolith trained on the same
  token budget (matched total params, matched total tokens) on
  in-length PPL of the mixed test set — and per-slice PPL must show
  specialist wins on their own slices (otherwise "division" bought
  nothing). **Kill**: ensemble ≤ monolith on mixed PPL → record as
  "division of labor does not pay at 20M scale on chat data".
- **Reading for the paper**: this is the BTM question at OPERA scale,
  with a cleaner specialist definition (data-sliced trees).

### S2 — Geometric composer over frozen specialists (the novel one)
K frozen specialists produce per-position prefix states; a small
OPERA composer tree (its leaves = concatenated/projected specialist
states, or one composer per layer reading specialist layer states)
fuses them via the compose node. Train ONLY the composer (~2-5M
params) on the mixed data — hours, not days.
- **The claim**: the geometric product is a *model-fusion* law, not
  just a within-model law. If it works, "ensemble geometric model"
  exists for the first time, and OPERA is it.
- **Gate**: composer-over-2-specialists beats (a) each specialist alone
  and (b) the S1 logit-ensemble of the same pair, at <10% of the
  monolith's training cost. **Kill**: fails to beat the logit ensemble
  → the geometric composer adds nothing over probability fusion;
  record, retire.
- **Risk** (be honest): state-space mismatch — specialists' spinor
  blocks live in different learned bases; the composer's first map
  must learn the alignment. This is exactly what the per-block 3×3
  maps are for, but it might not be enough.

### S3 — Shared-trunk OPERA-MoE (division inside one tree)
One tree, but the cross-layer MLP (or the fold's injection path)
becomes a sparse MoE: E small experts, top-1/2 gating per position.
Standard MoE transplanted onto OPERA's position-wise layers.
- **Gate**: match monolith PPL at ~40% active params (the classic MoE
  win), or beat it at matched params. Kill: no win at matched
  *active* params after one gating-variant retry.
- **Note**: least novel (MoE is mapped territory), but the cheapest
  credible "OPERA scales like the field expects" evidence, and it
  de-risks S2's routing question in isolation.

### S4 — Retrieval specialist pair (targeted at the measured gap)
Two specialists sharing one embedding: a gist tree (the incumbent
fold) and a retrieval-heavy tree (fold + T1.4 delta memory), fused by
S1 or S2. Directly attacks the assistant-gap with division of labor:
gist vs retrieval is the brain-like split the data already motivates.
- Depends on T1.4 existing (implemented, untested); natural sequenced
  follow-up, not a first move.

## 4. Recommendation

**S1 first** (pure measurement, overnight, answers "does division pay
at all at our scale"), **S2 second** (the scientifically interesting
one — geometric fusion as a first), S3 only if S1/S2 suggest routing
is the blocker. S4 after T1.4's gate resolves. All gates evaluated on
the chat protocol against the standing references (monolith 29.65
in-length; assistant-gap 0.791 nats).

## 4.5 Pre-registration (2026-08-04, chat protocol): T1.4 and S4

User direction: S4 first. One training run serves both gates — the
gist specialist already exists (incumbent chat checkpoint, in-length
PPL 29.65, assistant-gap 0.791 nats); the retrieval specialist IS the
T1.4 arm (`--mem delta`, mem_dim 128) on the identical chat protocol
(20k steps, seed 42, Muon 0.02, msup+curriculum+chrono stack).

**T1.4 gate (adapted from the roadmap to the chat protocol).**
Metric: the role-split assistant-gap (role_split.py, n=2000,
test_short). PASS: assistant-gap shrinks >= 50% (0.791 -> <= 0.40
nats) AND in-length PPL regression < 1.5 (stays <= 31.2) AND
position-curve boundary degradation stays <= +0.10 nats (position_curve.py).
KILL: assistant-gap unchanged or PPL regresses materially -> the
retrieval hypothesis for the ceiling is wrong at this scale; record,
redirect to T0.4/S3.

**S4 gate (new).** Fused pair (geometric-mean logit fusion,
ensemble_chat.py) vs its two members. PASS: fused in-length PPL
(5k test) beats the BETTER member by >= 5% relative AND fused
assistant-gap <= 0.9x the better member's gap. KILL: fused
underperforms its better member -> "probability fusion of
complementary OPERA specialists adds nothing at 20M scale" — record,
and the S2 composer question becomes the live one.

Both evals run on identical held-out data (test_short[:5000] for PPL;
test_short[:2000] for the role split; same 500 long conversations for
the position curve). Single seed caveat applies to all of it.

## 5. References

- Wang et al. 2024, Mixture-of-Agents, arXiv:2406.04692
- Li et al., Branch-Train-Merge (in MoE survey arXiv:2407.06204)
- Sukhbaatar et al. 2024, Branch-Train-Mix (BTX)
- Wortsman et al. 2022, Model Soups
- Pfeiffer et al. 2023, Modular Deep Learning survey, arXiv:2302.11529
- Ruhe et al. 2023, Geometric Clifford Algebra Networks (ICML)
- Rethinking MoA 2025, arXiv:2502.00674 (counterweight)
- Awesome-LLM-Ensemble (survey list), github.com/junchenzhi
