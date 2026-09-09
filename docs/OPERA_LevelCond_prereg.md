# OPERA level-conditioned compose weights — pre-registered protocol

**Version:** 1.0 — 2026-09-09
**Status:** PRE-REGISTERED before any implementation run. This is
OPEN-problem §5b (`OPEN_scale_tied_operator.md`), promoted to a
registered arm now that both optimizer-side rescues of scale-tying are
falsified (β^ℓ under AdamW, 2026-09-08; LO-Muon under Muon,
2026-09-09). If deep levels need different treatment, it must live in
the forward pass.
**Hardware:** MacBook M4, 16 GB, MPS. No Kaggle quota.

---

## 1. The mechanism

The compose operator is one weight set W applied at ⌈log₂T⌉ levels
whose input distributions differ (measured: level-1 RMS 1.020 vs
0.63–0.70 deeper). Give each level its own effective operator while
staying *defined* at unseen depths:

    W_ℓ = W + U · diag(f(ℓ)) · V

- **f(ℓ) = fixed sinusoidal level features** (the `level_sin_enc`
  basis already used by `fold_scale` and the attend fold, already
  declared extrapolation-safe): smooth in ℓ by construction, defined
  at every depth, NO learned level embedding and NO one-hot — levels
  ≥ 10 arriving at eval get well-defined features, not never-trained
  rows. This is the extrapolation trap the survey's §1 warning names,
  avoided structurally.
- **Zero-init U** (the up factor); V keeps its random init — the LoRA
  convention from the bistable rank study. `U = 0` is bitwise the
  incumbent at init while dL/dU ≠ 0 (the delta is linear in U), so
  there is no cold gate and no warm-gate compromise.
- **Rank r = 8** at the validation rung: parameter cost r·(3nb + 2d)
  ≈ 11K/layer at d512 — 0.3% of the model. Capacity is NOT the
  hypothesis; scale-specialization is.
- Applied to `fusion_gate.{l}.weight` only (the level-tied tensor
  whose gradient structure is measured). `rot_free` deferred, same
  scope note as LO-Muon v1. The FOLD's applications are scale-mixed
  per call and use the base W (pre-stated; conditioning them would
  need a per-position level mixture, which is a different mechanism).

Invariants: I1 ✓ (ℓ is circuit shape — scale, not position; the same
status `level_sin_enc` already has) · I2 ✓ · I3 ✓ (the node's form is
unchanged; its weights become scale-conditioned) · I4 ✓ (same Fenwick
blocks, same causality).

## 2. Why this is the right next arm, on the record

- Both optimizer-side equalizations of the shared operator are
  falsified; the one untested rescue is making the operator
  non-shared *in the forward*, which is also what the input-
  distribution measurement (10.68 vs 6.93 at the word rung; 1.020 vs
  0.63–0.70 at bytes) says the levels actually differ in.
- Path A's own amendment (2026-09-07) documents the depth-extrapolation
  confound: 87.5% of eval positions at 8192 read never-built-level
  blocks. A mechanism that gives unseen depths a *smooth continuation*
  of the operator rather than its exact copy is the only registered
  idea that targets that regime directly.
- The falsified `homeo`/`quotient` arms were additive modules with
  cold gates; this is an interior, zero-init, low-rank reparameter-
  ization — the over-relaxation design pattern, which is the one that
  produced the series' only positive.

## 3. Protocol

Rung: byte d512/nb128/L2, T=1024, batch 8, 3000 steps, seed 42, recipe
`rx_fgate_wd` (the current incumbent, BPB 2.0371) — same as the
RM-series so all arms share a baseline. Arms: `lc_off` (the incumbent
recipe re-run in-family) and `lc_r8` (level-conditioned, r=8).

## 4. Hypotheses and gates

**LC-H1 (primary) — length extrapolation.** The deficiency the
mechanism targets lives at deep levels, which dominate longer
sequences. *PASS* if the 1025–2048 bucket improves ≥ 2.0% vs `lc_off`
(mirroring the level-grad gate structure, for the same reason).
**This is the adopt-into-Path-A gate.**

**LC-H2 (in-length guard).** BPB regression ≤ 1.5%. A small
regression with LC-H1 passing is a SUCCESS — declared in advance, as
in the level-grad registration (the arm deliberately spends capacity
away from the shallow levels that dominate short-sequence loss).

**LC-H3 (usage — no gate, the house rule).** Report ‖U·diag(f(ℓ))·V‖
relative to ‖W‖ per level at convergence, and the per-level effective
operator distance cos(W_ℓ, W). An arm whose deltas stay ≈ 0 nulled by
parameters, and that must be reported as such.

**LC-H4 (extrapolation-safety check).** Forward at T = 4096 (4×
training length, levels 11–12 never trained) must be finite and its
per-level energy profile continuous in ℓ — the sinusoidal basis
guarantees this structurally; the check exists so a NaN or a
discontinuity at unseen depths is caught before it can eat a GPU
session.

## 5. Pre-stated interpretation

- **LC-H1 pass** → the scale-tied operator was the binding constraint
  after all, and the fix is forward-side; propose as the Path A OPERA
  arm (amendment before training, replacing the plain arm — Path A
  cannot afford two OPERA pretrains).
- **LC-H1 fail with LC-H3 showing real per-level deltas** → the shared
  operator is what the objective prefers in BOTH currencies now
  (optimizer-side and weights-side); the OPEN problem closes with a
  complete answer: scale-tying is not a pathology at this rung.
  That is a publishable negative with three falsifications behind it.
- **LC-H3 ≈ 0** → cold-parameter null; the init/gradient design failed
  and nothing is concluded about the hypothesis (distinguishable by
  construction — the point of zero-init-U with gradient flow).

## 6. Limitations at registration

Single seed; one rung; 3000 steps ≈ 0.56 epochs; extrapolation
measured to 2× only (memory cap amendment, Repr study) while Path A
evaluates at 16× — adoption is a stated transfer bet whose final
arbiter is Path A's own instrument. r=8 is a single dose; no ladder.
The fold uses base W (§1).

## Amendments log (append-only)
