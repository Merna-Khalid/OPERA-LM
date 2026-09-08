# OPERA input representation — pre-registered protocol: bytes vs BPE

**Version:** 1.0 — 2026-09-08
**Status:** PRE-REGISTERED. Written after the corpora were built and
throughput measured, but **before any training run completed** and
before any BPB number existed. Deviations are appended to the
Amendments log, never rewritten.
**Hardware:** MacBook M4, 16 GB, MPS. No Kaggle quota; does not compete
with Path A.

---

## 1. Motivation

Two problems with one cause.

**Measurement.** `gp_benchmark.py` skips 120 of 144 SAP ClassicGP
stimuli as OOV, leaving 12 cells, and NP/Z is structurally unmeasurable
because `load_data`'s word path applies
`[w for w in text.lower().split() if w.isalpha()]`, deleting the
disambiguating comma. The paired re-analysis of the phase-1 arms
(2026-09-08) showed fold_adapt's only reliable movement is on surprisal
(Δ +0.224, t=+2.26); at n=12 nothing else is resolvable. **Input
representation gates every downstream result in the trajectory
program.**

**Research.** OPERA removes positional encoding. Tokenization is the
other imposed discretization. Byte-level models — BLT, H-Net, ByteFlow
(arXiv:2603.03583, ICLR 2026) — all *downsample*: they must choose K
chunk boundaries so an expensive backbone runs on a shorter sequence,
and ByteFlow's ablation shows the choice is worth ~9 points
(coding-rate 50.89% > word boundaries 49.38% > random 41.34%). **OPERA
does not downsample.** It composes every dyadic span at every level and
reads every prefix exactly, so it never chooses boundaries at all.

## 2. The crux

OPERA's tree covers only intervals `[i·2^k, (i+1)·2^k)`. A word at
bytes 5..11 is no single node. The no-downsampling argument says this
should not matter — the prefix readout is exact and the model never
commits to a segmentation — but that is reasoning, not measurement.
**This study is the test of that argument.** If it fails, §1's core
claim forbids the H-Net fix (learned boundaries), so byte-level is a
genuine bet, not a hedge.

## 3. Protocol (frozen)

### 3.1 Matching
Corpora built by `experiments/build_reprs.py` from the same Simple
English Wikipedia articles, same `random.Random(42)` article-level 90/10
split, same `doc_chunks` schedule, same 20,000-article cap. Verified:
**both arms cover an identical 43,907,160 raw training bytes.**

Measured compression on this corpus: bytes 1.0000 units/byte, BPE
0.2855 (3.53 bytes/token), word 0.1252 (7.99 bytes/word).

| arm | repr | T | batch | context | bytes/step |
|---|---|---|---|---|---|
| bytes_d256 | bytes | 1024 | 8 | 1024 B | 8192 |
| bpe_d256 | bpe | 290 | 8 | 1016 B | 8195 |

Steps, text seen, and context window all match to <1%.

### 3.2 Parameters — the comparison is compute-matched
| arm | total | emb+head | non-embedding |
|---|---|---|---|
| bytes_d256 | 893,191 | 132,608 (14.8%) | **760,583** |
| bpe_d256 | 9,165,316 | 8,388,608 (91.5%) | **776,708** |

91.5% of the BPE model is a lookup table. Non-embedding parameters — the
rotors, gates, norms and MLPs that actually compose — differ by 2%. The
primary contrast therefore needs no correction. Total parameters are
reported alongside because a reader will ask, and because the 10x gap
*is* the argument for byte-level at small scale.

### 3.3 Shared
d=256, nb=64, 2 layers, pe none, fold left, rot free, seed 42, 3000
steps, AdamW. Identical to the phase-1 arm rung except batch (8 vs 16)
and T, both forced by the byte context budget.

### 3.4 Metric
**Bits per byte**, the only quantity comparable across tokenizations:
`bpb = (nats/unit) × (units/byte) / ln 2`. PPL-per-unit is reported but
is **not** comparable between arms and no conclusion may rest on it.

## 4. Hypotheses and gates

**H1 (byte-level is viable at matched compute) — primary.**
*PASS* if `bpb(bytes_d256) ≤ bpb(bpe_d256) × 1.10`. Rationale: within
10% at a tenth of the total parameters makes byte-level the better
deployment, and validates the no-downsampling argument of §2. *FAIL* if
bytes is >10% worse — the dyadic/morpheme misalignment is real and the
answer is BPE as the permanent representation (§6).

**H2 (the saved parameters buy something).** Byte width ladder
`bytes_d384` (1.71M non-emb) and `bytes_d512` (3.03M non-emb, 3.9× the
BPE arm's compute, still a third of its total size). *PASS* if BPB
decreases monotonically across d=256→384→512. A non-monotone ladder
means the byte arm is data-limited, not capacity-limited, at this
budget, and the 3000-step rung cannot rank representations.

**H3 (measurement unblock) — the reason this study exists.**
`gp_benchmark.py` on the byte and BPE checkpoints must score
**all 144 cells** with NP/Z present and non-zero. This is a coverage
assertion, not a quality claim, and is checked independently of H1/H2.
If H1 fails, H3 still stands for the BPE arm.

**H4 (invariance — the Kirby-calculus idea, made testable).**
Byte-level assigns one canonical representation to a string; BPE's
segmentation of digit runs is arbitrary. On held-out digit strings and
repeated characters, compare per-character CE variance under
semantically-null perturbations. *Reported, no gate* — this is a first
measurement, not a decision.

## 5. Limitations acknowledged at registration

Single seed (project convention). One scale (~0.9M non-embedding). 3000
steps ≈ 0.56 epochs of a 43.9M-byte pool — under-trained by any modern
standard, and byte-level models are known to need more steps for equal
quality, so **a byte-side null at this budget is weak evidence**; the
training curve is reported so under-training is visible. Simple English
Wikipedia only. The BPE tokenizer (16,384, Path A's) was trained on
smoltalk, not on Wikipedia — a mild mismatch that slightly disfavors the
BPE arm and is not corrected, since retraining it would break
comparability with §4.8 and Path A.

## 6. Pre-stated decision

- **H1 pass** → byte-level becomes the default representation for the
  trajectory program; rerun the phase-1 arms on it at full GP coverage.
- **H1 fail, H3 pass** → BPE becomes the default; the coverage problem
  is solved either way and the trajectory program proceeds. Byte-level
  is recorded as measured-and-rejected for OPERA, with the
  dyadic/morpheme misalignment as the stated cause.

Either outcome unblocks the measurement. That is why this study is worth
running before any further architecture arm.

---

## Outcomes — 2026-09-08

### Headline table

| arm | total params | non-emb | ctx bytes | PPL/unit | **BPB** | min |
|---|---|---|---|---|---|---|
| bytes_d256 | 893,191 | 760,583 | 1024 | 5.15 | **2.3642** | 33 |
| bpe_d256 | 9,165,316 | 776,708 | 1016 | 183.31 | **2.1461** | 18 |

PPL/unit is *not* comparable across arms (different units); BPB is.

### H1 — **FAIL**, by 0.163 percentage points

`bpb(bytes)/bpb(bpe) = 1.10163`; the pre-registered gate was `≤ 1.10`
(threshold 2.3607, measured 2.3642). **The gate fails and is recorded as
failed.**

It must equally be recorded that **the margin is far below what this
study can resolve**: single seed, one scale, 3000 steps ≈ 0.56 epochs.
A 0.16 pp difference is not distinguishable from seed noise, and no
noise band for BPB was established in advance (a gap in the
registration, noted for next time). The honest reading is *"byte-level
lands within measurement error of BPE at matched composing compute, on
the wrong side of an arbitrary threshold"* — not *"byte-level is worse."*
It is also not *"byte-level is fine"*: the pre-registered decision in §6
is triggered by the gate as written, not by the margin.

Context that does not change the verdict but bears on cost: the byte arm
reached this with **10.3× fewer total parameters** (0.89M vs 9.17M) and
took **1.8× longer** (33 vs 18 min) for the same bytes/step — the
sequence-length tax is real and is the byte side's true cost here, not
quality.

### H2 — **PASS**, and it overturns the H1 reading (run 2026-09-08, after
the H1 verdict was recorded)

| arm | total | non-emb | BPB | vs BPE | min |
|---|---|---|---|---|---|
| bpe_d256 | 9,165,316 | 776,708 | 2.1461 | 1.0000 | 18 |
| bytes_d256 | 893,191 | 760,583 | 2.3642 | 1.1016 | 33 |
| bytes_d384 | 1,904,903 | 1,705,991 | 2.2110 | 1.0302 | 45 |
| **bytes_d512** | **3,293,447** | 3,028,231 | **2.1317** | **0.9933** | 60 |

Strictly monotone, as H2 required. **`bytes_d512` beats the BPE arm by
0.67% BPB with 2.78× fewer total parameters.**

**This explains H1's failure.** The d=256 byte arm was capacity-starved,
not representation-limited: 760K composing parameters were doing
character-level composition that BPE's merge table supplies for free.
Spending the ~8.4M parameters BPE burns on a lookup table *on
composition instead* clears the gap and crosses it at a third of BPE's
total size.

Symmetry of standards, stated explicitly: the H1 failure margin
(0.163 pp) was called below this study's resolution. The H2 crossing
margin is 0.67 pp — 4× larger, still single-seed, still without a
pre-registered noise band, and therefore **also not decisive as a single
comparison**. What carries the conclusion is the three-point monotone
ladder plus the crossing, which is far harder to produce from seed noise
than either endpoint alone, and which matches ByteFlow's reported
widening of the byte advantage with scale.

Against over-reading it: the slope is **shallowing** — −0.190 BPB per
e-fold of composing parameters from d256→d384, then −0.138 from
d384→d512. The curve is flattening as it crosses, not accelerating
through. An extrapolation from the first two points predicted 2.102; the
measured value is 2.1317, i.e. the prediction was optimistic.

### H3 — **PASS**, decisively. The reason the study existed.

| | word-level (incumbent) | bytes | bpe |
|---|---|---|---|
| sentences scored | 24 | **144** | **144** |
| skipped (OOV/alignment) | 120 | **0** | **0** |
| amb/unamb cells | 12 | **72** | **72** |
| NP/Z measurable | no (0/48) | **yes** | **yes** |

6× the cells and NP/Z measurable for the first time, under both
representations. Every future arm is now decidable at 72 cells instead
of 12.

### H4 — reported, no gate. Byte-level is more invariant.

Tokenizer-only (no model): byte encoding assigns exactly N units to an
N-digit string (SD **0.00** at 4, 7 and 10 digits); BPE assigns the same
10-digit string 4–7 units (SD 0.54) on merge-table luck alone.

With the models, on 200 random 7-digit strings in a fixed carrier:

| arm | bits/char | CV | units/span |
|---|---|---|---|
| bytes | 7.039 ± 0.599 | **0.0851** | 7.00 ± 0.00 |
| bpe | 7.505 ± 1.037 | 0.1382 | 4.05 ± 0.45 |

The byte arm's cost spread is **38% lower**. Every string is drawn from
the same distribution and is equally hard in principle, so the residual
spread is sensitivity to how the string happened to be cut. First
measurement; no competence claim is made — neither model can do
arithmetic at this budget.

### Unregistered observation: GP effects are near-null for BOTH arms

Standardized GP effect at the critical word: bytes surp d = −0.019, bpe
surp d = +0.092, against the word-level phase-1 incumbent's +0.356 (on
12 cells). Coverage went **up** and signal went **down**.

Cause, and it is a protocol artifact rather than a finding about
representation: these arms saw **~4× less text** than the phase-1 word
arm. Word: batch 16 × 256 words × 3000 = 12.3M words ≈ 98M raw bytes.
Here: batch 8 × 1024 bytes × 3000 = 24.6M raw bytes. The batch halving
and the byte context budget were both forced by memory, and the
consequence on total text was not accounted for at registration. **This
does not affect H1** (the two arms are matched to each other), but it
means no GP comparison against the phase-1 record is valid, and the
72-cell instrument has not yet been exercised on a model strong enough
to show garden-path effects.

### Decision — superseded, and why

§6's rule (H1 fail + H3 pass → BPE default) was **applied and then
withdrawn within the same session**, on evidence rather than preference.
The sequence is recorded in full because the reasoning matters more than
the outcome:

1. H1 failed at d=256 by 0.163 pp. §6's rule was applied: BPE default.
2. That rule was written against a threshold (1.10) chosen without a
   noise band, and it fired on a margin the study cannot resolve.
3. **The decisive experiment — H2 — had not been run when the rule
   fired.** Recommending a default before running the study's own
   capacity test was the error.
4. H2 then passed monotonically and crossed BPE at 2.78× fewer
   parameters.

**Adopted: byte-level (`bytes_d512` configuration) is the default
representation for the trajectory program.**

The H1 record stands unaltered — it failed, at d=256, and that entry is
permanent. What changed is the interpretation the ladder licenses: H1
measured a capacity ceiling, not a representation ceiling. Pre-registered
gates bind what gets *recorded*; they do not bind a research decision
against an experiment the registration itself specified and that had not
yet been run.

### What would still strengthen it

1. **Multi-seed BPB with a measured noise band.** Still missing, and it
   is the reason neither the 0.16 pp loss nor the 0.67 pp win is
   decisive alone. Highest-value follow-up.
2. **One more rung (`bytes_d640`/`d768`).** The slope is shallowing;
   a fourth point establishes whether the crossing widens or saturates.
3. **Text-matched budgets, not step-matched.** These arms are matched to
   each other but saw ~4× less text than the phase-1 word arm, which is
   why GP effects are near-null for every arm here.
4. **Wall clock.** The byte win costs 60 min vs BPE's 18 at this rung.
   Quality per parameter favours bytes; quality per second does not, and
   the sequence-length tax is the real engineering cost.

## Amendments log (append-only)

**2026-09-08 — `eval_max_len` capped at 2048 (after the byte arm's first
run, before its rerun completed).** The byte arm's first run finished
training (3000/3000, final loss 1.954) and was then killed by the OS
during final evaluation at `eval_max_len = T*4 = 4096`; measured free
memory at the time was 62 MB. The checkpoint had not yet been written
and the run was lost. `eval_max_len` drives only the extrapolation
buckets and never the in-length PPL from which BPB and H1 are computed,
so the cap cannot affect any gate. The rerun used identical seed, data,
steps and batch. A mid-training periodic eval from the lost run (step
2000, PPL 4.80 → BPB 2.263) was discarded rather than reported, since it
was neither final nor backed by a surviving checkpoint.

**2026-09-08 — per-arm checkpoint directories.** `train()` builds its
checkpoint filename from the config tag, which encodes pe/fold/rot/opt
but **not** `vocab_size` or `max_len` — the only two fields that differ
between these arms. A shared `out_dir` therefore had the byte run
silently overwriting the completed BPE checkpoint; caught before loss.
Arms now write to `runs_reprs/<arm>/`.
