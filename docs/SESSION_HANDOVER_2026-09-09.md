# Session handover — 2026-09-07 → 09

Everything below is on disk and **uncommitted**. `git status` shows the
working tree. Selftest is at **40/40 ALL PASS** (`python -m opera_lm.selftest`).

---

## 0. TL;DR — where things actually stand

**The single largest result of the session was a recipe change, not an
architecture change.** Switching the byte rung from AdamW to Muon —
already validated and shipped in this repo, never passed to any byte
run — gave **BPB 2.1308 → 2.0508 (−3.75%)** and extrapolation **−6.73%**.
That is larger than all seven mechanism arms combined.

**Consequence:** every comparison made on 2026-09-08 — the
representation study and all seven mechanism arms — was measured against
a baseline 3.75% off its achievable quality. Details in §4.

**Current best model:** `runs_reprs/rx_muon/`, byte-level d512/nb128/L2,
BPB 2.0508.

---

## 1. New instruments (all runnable, all validated)

| module | what it measures |
|---|---|
| `opera_lm/level_exposure.py` | per-level tree/fold exposure; pure combinatorics, no GPU |
| `opera_lm/curvature.py` | finite-difference curvature proxy + h-linearity validity gate |
| `opera_lm/garden_path.py` | per-token curvature, within-position design, confound controls |
| `opera_lm/step_nulls.py` | isotropic / shuffled-token null models for step size |
| `opera_lm/reprs.py` | matched word/BPE/byte encoders sharing `data.py`'s split |
| `opera_lm/state_tracking.py` | group word problems (parity, A4, S4, A5, S5) |
| `experiments/repr_study.py` | the arm driver everything ran through |
| `experiments/bistable_gates.py` | pre-registered gate evaluation |
| `experiments/invariance_probe.py` | tokenization-invariance on digit strings |

`opera_lm/gp_benchmark.py` gained `--arch opera-sub` (byte/BPE scoring
via char-offset alignment, sharing the Mamba path). The word-level path
that produced phase-0/1 is untouched.

---

## 2. Findings that survive

**Level exposure (2026-09-07).** Fenwick gives *uniform* 0.50 fold
entries per position across levels 0–8 — no scale is starved. But the
**root of the training tree is reachable by exactly one prefix**, and at
the gradient level receives **exactly zero gradient** (its only reader
is L=T, whose prediction target is masked). `msup` cannot fix this:
it breaks at `span >= T`, and its target for the root lies outside the
sequence by definition. At train 512 / eval 8192, **87.5% of eval
positions read at least one never-built-level block.**

**Trajectory phase 2 — the null models (2026-09-08).** Constant-speed
integration is **not** geometrically forced: radius free (CV 0.052–0.059),
**41–58% of the available step range unused**, observed variance 1.75×
the isotropic null in two independent models. And step size is nearly
**input-independent** — shuffling tokens moves it 1.8%.

**Representation (2026-09-08).** Byte-level width ladder crossed BPE:
2.3642 → 2.2110 → **2.1317** vs BPE 2.1461, at **2.78× fewer parameters**.
H1 failed at d=256 by 0.163pp (capacity starvation), H2 passed
monotonically. GP coverage went **24/144 → 144/144 stimuli, 12 → 72
cells, NP/Z measurable for the first time.** *Caveat: all AdamW numbers.*

**Over-relaxation — the one positive mechanism (2026-09-08).**
`acc ← (1−g)·acc + g·composed`, g>1 makes the coefficient on `acc`
negative. Tail p99.9/median **1.200 → 1.656** (gate was ≥1.60), control
arm unmoved at 1.204, **0 divergence skips**. Five gating arms failed
because a convex blend is pinned between its endpoints and cannot
increase displacement; this is the one operation that escapes that.

**Scale-tied gradient imbalance (2026-09-09).** 15:1 gradient share
across levels into the shared compose weights, zero at the root. See
`docs/OPEN_scale_tied_operator.md`.

---

## 3. Falsified, with controls

| mechanism | outcome |
|---|---|
| bistable fold, per-block | used heavily (66% of updates, median gain 1.65), **no excursions** |
| bistable, low-rank r1/r4 | used *more* (83%/86%), tail got **smaller** — falsified the post-mortem of the above |
| level-balanced gradient β=1.25/1.5/2.0 | monotonically **worse** on both BPB and extrapolation |
| `msup` at byte rung | −0.25% (vs −9.87 PPL at word toy rung) — did not transfer |
| curvature-vs-perplexity account | 2/4 knobs unmeasurable; isometry ≠ flatness |
| state tracking pilot | inconclusive — transformer failed identically, budget-limited |

---

## 4. What needs re-measuring under Muon

- **Byte-vs-BPE margin is unknown in both directions.** Bytes crossed BPE
  by 0.67%, both AdamW. Bytes now 2.0508; **BPE has not been re-run under
  Muon.** Do not quote the representation margin until it is.
- **Over-relaxation positive and level-balance falsification** were both
  AdamW-baseline; neither reproduced under Muon.
- Gating nulls are probably safe — the convexity argument is structural,
  not optimizer-dependent.

---

## 5. Next actions, in order

**1. `fusion_gate` into the Muon partition.** `EXCLUDE_SUBSTR` in
`opera_lm/muon.py:130` contains `'gate'`, which catches `fusion_gate` —
**917,760 of 3,287,040 two-dimensional params (28%) routed to AdamW by
accidental substring match.** Narrow to `'blend_gate'`. Largest known
misallocation, ~1 hour.

**2. Muon weight decay.** `muon.py:126` sets `weight_decay=0` "by
design" for the original clean A/B. Moonshot (arXiv:2502.16982) names it
as one of two techniques crucial for scaling Muon; the other (the
`max(1,m/n)^0.5` rescale) is already at `muon.py:102`.

**3. Re-run BPE under Muon** (§4).

**4. `mem_mode='delta'` — the readout-capacity axis.** Implemented,
selftested, zero-init, **never run once.** `mem_dim=128` gives each
prefix a 128×128 = 16,384-number state alongside the 512-dim fold state
(**32× read capacity**) for +36% params. This is the only axis that
attacks the structural reason transformers win raw-text PPL — no
optimizer can add capacity. Add the two HOLA details the current
implementation lacks: **surprise-gated writes** (`β·‖v − Mk‖` instead of
the plain sigmoid at `model.py:1677`) and a **decoupled query norm**
(`mem_k` is normalized but `mem_q` is not — the opposite of HOLA's
split). Needs a **params-matched control** (a wider tied model at
~4.5M) so a win isn't just parameter count. Run on the Muon baseline.

**5. Per-level Muon (optional, novel).** Muon orthogonalizes the
*summed* gradient; for a scale-tied operator that sum mixes applications
with different input statistics. `Σ NS(g_ℓ)` ≠ `NS(Σ g_ℓ)`. Tap
per-level gradients (the `_GradScale` machinery in `model.py` is the
starting point), orthogonalize each, then sum. Distinct from the
falsified level-balance arm: that applied a *fixed* rescale which the
optimizer's own normalization washes out; this is scale-invariant per
level. Applies to `fusion_gate` and `rot_free` only. **Do §1 first** —
`fusion_gate` isn't in the partition yet, so there'd be nothing to test.

---

## 6. Standing constraints

- **Invariants:** I1 no positional encoding · I2 no pairwise attention
  or softmax over positions · I3 geometric compose node · I4 exact
  prefix-causal Fenwick readout. Every arm this session preserved all four.
- **House rule that keeps failing:** zero-init behind a *multiplicative
  sigmoid* gate produces cold-gate nulls (bias −6 → gradient suppressed
  ~400×; killed `homeo` and `quotient`). Prefer designs where the
  incumbent is **interior** to the parameter space — over-relaxation's
  g=1 is the model to copy: bitwise incumbent *and* gradients flowing
  both directions.
- **Hardware:** 16 GB. Five d512 models back-to-back caused a swap storm
  (24 GB swap, one run lost). Serialize runs; `--eval-cap 2048`.
  **Backward is non-deterministic at ~1.3e-7 relative** — bounds any
  gradient-level claim.
- **Path A is untouched** and still consumes no GPU quota.

---

## 7. Prereg documents

`OPERA_PathA_prereg.md` (amended) · `OPERA_StateTracking_prereg.md` ·
`OPERA_Repr_prereg.md` · `OPERA_Bistable_prereg.md` (+ low-rank
addendum) · `OPERA_Relax_prereg.md` · `OPERA_LevelGrad_prereg.md` ·
`OPERA_Recipe_results.md` · `OPEN_scale_tied_operator.md` ·
`OPERA_Improvement_Survey_2026-09.md` · `trajectory_phase2_nulls.md`

Every one carries its outcomes, including the ones that went against the
hypothesis, and the amendments log where the protocol moved.
