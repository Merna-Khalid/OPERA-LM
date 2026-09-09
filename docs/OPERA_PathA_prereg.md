# OPERA Path A — Pre-registered protocol: length generalization of pretrained chat models (train @ 512, evaluate to 8192)

**Version:** 1.0 — 2026-09-05
**Status:** PRE-REGISTERED. Committed before the first GPU session of this
study. No post-hoc edits: deviations discovered mid-study are appended in
the *Amendments log*, never rewritten.
**Hardware note:** all runs on Kaggle free tier (2×T4, DDP where stated),
fp16 + GradScaler (Turing capability gate), interrupt-safe multi-session
execution with exact-stream resume. Multi-session resume restores model,
optimizer, step, and all RNG states, so a resumed run's batch stream is
identical to an uninterrupted one on the same hardware/software stack.

---

## 1. Motivation and lineage

This study extends the 155M Colab study (§4.8 of the paper draft v1.3):

- Chat-only from scratch: **OPERA won** in-length (11.78 vs RoPE 18.60)
  and every extrapolation bucket to 2048.
- FineWeb-Edu pretrain (82M tokens) from scratch: transformer won (83.55
  vs 105.89).
- Two-stage (pretrain → smoltalk SFT): **transformer won** (11.24 vs
  13.04), reversing run 1.

Two questions are open and both are pre-registered here:
(a) does OPERA's flat position curve — measured only on the 20M pilot —
survive pretraining + SFT at 155M and extend to 16× training length?
(b) does the two-stage flip persist at 3× the pretraining budget?

## 2. Protocol (frozen)

### 2.1 Data

- **Tokenizer:** byte-level BPE. Default = the existing 16,384-vocab
  `tokenizer.json` (the one used by §4.8). A fresh tokenizer trained on
  2,000,000 smoltalk message lines (the Mac-segfault fix, run on cloud
  RAM) is evaluated under the adoption rule in §5.1 before any training.
- **SFT corpus:** full smoltalk `all` split (1,043,917 conversations),
  `prepare_data.py --n-convs 5000000 --max-len 512 --eval-max-len 8192`,
  conversation-level 95/5 split (seed 42, as baked into the script),
  train = chunks ≤ 512, test_long = (512, 8192].
- **Pretrain corpus:** FineWeb-Edu `sample-10BT` streamed to a 250M-token
  pool (`prepare_fineweb.py`), chunked at 512.
- **Packed loading:** pools are packed to flat int32 arrays + offsets and
  memory-mapped (`--batch-source packed`); batch streams are verified
  identical to the incumbent `GpuBatchSource` (selftest) so sampling
  semantics change nothing.

### 2.2 Arms (matched protocol)

| arm | architecture | params | position mechanism |
|---|---|---|---|
| OPERA | d=1664, nb=416, 8 layers, fold left, rot free, msup, tie | 154,815,249 | constructive (Fenwick), pe none |
| TF RoPE | d=1264, 8 heads, 7 layers, tied | 155,049,777 | rotary (vanilla extension, `max_pe_len=8192`, no NTK/YaRN — limitation, §7) |
| TF NoPE | same as RoPE arm | same | none (causal-mask emergent) |

Identical: seed 42, Muon(lr 0.02) + AdamW(lr 1e-3) partition, warmup 500,
WSD schedule (decay over final 20%), grad clip 1.0, fp16 autocast +
GradScaler (T4), curriculum (64, 250) → 512, effective batch **32**
(16/rank × 2 ranks DDP for OPERA on 2×T4; batch 32 single-T4 for each
transformer arm; the two TF arms run concurrently, one per GPU).

Known deviation from §4.8 (logged): §4.8 was batch 16 × T=256 single A100
with cosine LR; this study is batch 32 × T=512 with WSD. All three arms
share the new protocol; no arm is favored. LR was not retuned at the new
batch size (recipe-asymmetry limitation carries over).

### 2.3 Schedules

- **Stage 1 — pretrain:** 15,000 steps × 16,384 tokens/step ≈ **246M
  tokens** (target "≈250M"; 3.0× the §4.8 attempt). Decision points at
  step 6,100 (≈100M) and 9,150 (≈150M) per §5.3.
- **Stage 2 — SFT:** 2,500 steps ≈ 41M tokens ≈ 1.2 epochs of the smoltalk
  pool — token-matched to §4.8's SFT (10k × 4,096). Initialized via
  `--init-weights-from` (weights only).
- **Stage 3 — instruments:** (a) in-length PPL, test_short[:5000] @ 512;
  (b) extrapolation buckets (width 512) to 8192 from test_long, counts
  reported per bucket; (c) position curves to 8192 (below); (d) sampling
  demo at contexts 1k/2k/4k/8k from held-out long conversations, temp 0.8
  top-p 0.9, `max_ctx` 8192, incremental decoder.

### 2.4 Position-curve instrument (`opera-chat/position_curve.py`)

Final-layer per-position next-token CE, fp32, no autocast, no grad;
sequences = first 1,000 of test_long with ≥ 1024 tokens (pool order —
the v1.2 convention); bands of 128 positions to 8192; per-band token
counts always reported. Headline quantities per arm: best-band CE,
degradation best → final 2× band (last band ending ≤ 1024), degradation
best → final horizon band (the band containing 8191), across-boundary
delta (mean CE [512,1024) minus [0,512)), mean CE beyond 4096.

## 3. Hypotheses and decision rules

**H1 (flatness survives pretraining).** OPERA's position-curve
degradation (best band → final horizon band, §2.4) is ≤ 50% of TF
RoPE's on the same sequences. *Decision:* PASS if
`deg_opera ≤ 0.5 × deg_rope` AND OPERA's across-boundary delta
≤ +0.10 nats. FAIL otherwise; report both numbers regardless.

**H2 (absolute long-context quality).** OPERA's mean CE over positions
≥ 4096 (aggregated bands) is lower than TF RoPE's. *Decision:* PASS if
strictly lower. Report NoPE alongside (no gate).

**H3 (the two-stage flip at 3× budget).** In-length PPL (§2.3a) ranks
the three arms after SFT. *Decision:* report the ranking verbatim; the
flip PERSISTS if RoPE < OPERA; CLOSES/REVERSES otherwise. Pre-stated
interpretation: persistence supports §4.8's "transformer raw-text head
start" reading; reversal is the headline outcome for geometric models.
Either way this is a result, not a failure.

**H4 (honest flatness null).** If TF RoPE's degradation (best → final
horizon band) is < 0.15 nats, RoPE is "flat" on this protocol and H1's
flatness claim does not discriminate; the long-context claim then rests
on H2 + O(T log T) cost only. *Decision:* recorded automatically from
the curve; no discretion.

All gates are computed by a script from the three `position_curve.py`
JSON outputs + results.jsonl rows; numbers are copied verbatim into the
outcomes section, appended below when the study completes.

## 4. Session-robustness policy (part of the protocol)

Kaggle sessions are governor-limited (clean stop before the 9h T4×2
cap), checkpoint every 1,000 steps (≈25–40 min), checkpoints pushed to
the persistent Kaggle Dataset in-session. A preempted session resumes
from the last pushed checkpoint with exact-stream resume. Consequence
for validity: none intended — resume is state-exact; any deviation
(e.g. a checkpoint lost to a double fault, forcing a re-run from an
earlier step) is logged in the Amendments log with the step range
affected.

## 5. Data rules (pre-committed)

### 5.1 Tokenizer adoption rule
Compare old (500k-line) vs new (2M-line) tokenizer on 1,000 conversations
sampled from the smoltalk stream (in-distribution sample; streaming a
strictly held-out range past the tokenizers' training lines costs ~an
hour of session time for no decision value — the rule measures
compression of the training distribution): median tokens/conversation
and total tokens. Adopt the new one ONLY if median compression improves
by ≥ 1%; otherwise keep the §4.8 tokenizer for comparability. Either way
both artifacts are kept and the measurement is reported.

### 5.2 Eval-pool population rule
Extrapolation buckets and position bands are reported with their token
counts; thin buckets (< 50 sequences for PPL buckets, < 5,000 tokens for
bands) are marked and excluded from gate arithmetic (H2 uses aggregated
bands ≥ 4096 only if they clear this floor). Population is a property of
smoltalk's long tail; it is reported, not patched.

### 5.3 Pretraining budget decision rule
Default target 246M tokens. If Kaggle quota/throughput makes the full
budget impractical, the pre-stated floors are: **150M** (usable, report
as "floor-150") and **100M** (minimum viable, "floor-100"); the SFT
stage may start from any floor, and the achieved budget is reported with
every result. No arm gets a different budget than another.

## 6. Honesty box: what "chatable" means at this budget

A 155M model pretrained on ~250M tokens (0.01% of SmolLM2-135M's 2T) and
SFT'd on smoltalk. **Will:** correct chat format (turn-taking, stopping),
fluent grammatical multi-turn replies that reference context, simple
instruction-following behaviors, recall of very common facts, and — the
study's claim — measurably better length behavior at 4–8k context than
the matched RoPE baseline if H1/H2 pass. **Will not:** reliable factual
knowledge, arithmetic, complex multi-constraint instructions, or
assistant-level helpfulness. This text ships with the demo.

## 7. Limitations acknowledged at registration

Single seed (project convention); single model scale (155M); vanilla
RoPE extension only (no NTK/YaRN — a stronger RoPE arm is future work);
WSD + batch-32 protocol differs from §4.8's cosine + batch-16 (shared by
all arms, but cross-study comparisons carry this caveat); position
mechanisms compared at one training length (512); FineWeb-Edu only
(no code/multilingual mix); Kaggle T4 fp16 numerics (GradScaler; the
NaN-skip divergence guard is active and its skip counts are reported).

---

## Outcomes (appended when complete; empty until then)

*(nothing yet)*

## Amendments log (append-only)

**2026-09-06 — FineWeb pool semantics corrected; stream target raised
(before any GPU training ran).** The data session's uploads revealed
that `prepare_fineweb.py --max-tokens` caps *streamed* tokens, and
`doc_chunks(long_first=True)` routes only ~9% of FineWeb-Edu's streamed
tokens into ≤512 train chunks (measured: 250M streamed → 22.1M-token
pool; §4.8 had the same shape: 500M → 19.9M). §2.1's "250M-token pool"
wording was wrong about pool size; §2.3's "246M trained tokens, 3× §4.8"
stands as *trained*-token accounting, but unique-token exposure would
have been ~1.1× §4.8 with ~11 epochs — not the intended budget increase.
Amendment: stream cap raised to 1.0B (target pool ≥ 80M unique train
tokens; ~3 epochs at 15,000 steps). Engine knobs:
`PATHA_FINWEB_TOKENS` / `PATHA_FINWEB_POOL_MIN`. The smoltalk SFT pool
measured 114M train tokens (T=512 admits far more whole conversations
than T=256 did); SFT remains 2,500 steps ≈ 41M trained tokens —
token-matched to §4.8's SFT. H1–H4 are unchanged; H3's "3× budget"
reading now covers both trained tokens AND ~4× unique pretraining
tokens versus §4.8.

**2026-09-07 — Level-exposure measurement; pre-stated interpretation of
H1/H2 (no protocol change, recorded BEFORE any GPU training ran).**
A combinatorial audit of which parts of the circuit this protocol's
training actually visits (`opera_lm/level_exposure.py`; no model, no
data, no GPU — pure counting over the shipped `fenwick_blocks`) is
recorded here so that H1/H2 are read against it rather than after it.
Measured at this study's exact settings (train T=512, curriculum
(64, 250), 15,000 steps, batch 32, eval T=8192):

- **Training exposure is uniform across scales, levels 0–8.** Each level
  0–8 receives exactly 0.50 fold entries per position (~half of all
  prefixes have any given bit set). This is a property of the Fenwick
  decomposition and is reported as a positive architectural finding: no
  scale is starved relative to another.
- **The top level of the training tree is not.** Level 9 (span 512 = the
  full training sequence) is reachable by exactly one prefix (L=512), so
  it sees ~256× less signal than every level below it. The deepest
  *well*-trained level is therefore **8 (span 256)**, not 9. `msup` does
  not compensate: `msup_loss` breaks at `span >= T` (`losses.py:118`),
  and its target — the first token *after* the node's span — lies outside
  the sequence for the root by definition. No objective defined on a
  length-T sequence can supervise the composition of a length-T span.
- **The curriculum is not implicated.** 95% of steps run at the full
  T=512 (250 steps each at 64/128/256). Any level-exposure effect here
  is a property of training length, not of arm C.
- **Levels 10–13 are never built during training** and are required at
  eval T=8192, where levels 10/11/12 each receive 4,096 fold entries —
  the *same* frequency as every trained level. **87.5%** of eval
  positions read at least one never-built-level block (**93.8%**
  counting from the deepest well-trained level 8); **23.1%** of all eval
  fold entries sit at never-built levels (**30.8%** from level 8).
- **Fold chain depth is a minor axis.** Deepest accumulator chain is 8
  composes trained vs 12 at eval, but only 4.6% of eval positions fold
  deeper than training ever produced.

*Scope, stated precisely.* This counts EXPOSURE, not learning. OPERA
ties the compose weights across all levels and, under this study's
`fold_mode='left'` configuration, has **no level-indexed parameters at
all** — nothing is randomly initialised at eval, and a level-12 node is
a trained function applied at an unseen recursion depth, not an
untrained module. The audit establishes only that the input regime at
8192 was never visited during training.

*Pre-stated interpretation (binding on how the outcomes are written
up).* H1 and H2 at 8192 measure the positional mechanism **confounded
with** depth extrapolation of the tied compose function into a
never-visited input regime; the two cannot be separated by this
protocol. The confound is not unique to OPERA — the RoPE arm meets
unseen rotation angles and the NoPE arm unseen mask lengths — so the
three-arm comparison stands as pre-registered and **no gate, hypothesis,
or decision rule is modified**. What changes is the write-up: an OPERA
H1 failure may not be evidenced against "position is structure," and an
H1 pass is the stronger result for having been obtained under this
handicap. Either outcome is reported with these counts alongside.

*Remedy, explicitly OUT of scope for Path A.* The state-passing
intervention (concatenation augmentation, Buitrago Ruiz & Gu,
arXiv:2507.02782; `opera-chat/prep_statepassing_data.py`) targets this
exact regime and is **not** part of this protocol. It belongs to a
separate, separately pre-registered study, and is named here only so
that the record shows it was identified before the outcomes, not after
them.

**2026-09-09 — Recipe adoption + Session-2 recipe sweep (logged
BEFORE any GPU training; only CPU data sessions have run).** The
byte-rung program of 2026-09-08/09 changed what is known about the
optimizer recipe this protocol froze on 2026-09-06:

1. **`fusion_gate` was misrouted to AdamW by the `'gate'` substring**
   in `muon.py`'s exclusion list — 28% of 2-D parameters at d512, and
   ~21% of ALL parameters (33.2M/154.8M) at this study's OPERA config.
   This was an accident, not a design choice; §4.8's OPERA arm carried
   the same misrouting (its TF arm was unaffected — its matrices were
   always in Muon). OPERA adopts the include-list routing
   (`--muon-include fusion_gate`).
2. **Muon-side weight decay 0.01** (Moonshot's scaling recipe; byte
   rung: −0.23% alone, additive with the routing fix). Both arms adopt.
3. **`muon_lr=0.02` was gated at the word-level toy rung (T0.1) and
   carried unmodified into §4.8 and this protocol.** It has never been
   measured on BPE chat/FineWeb data, at 155M, or on T4s.

Amendment: the recipe is **measured on this study's own data,
tokenization, and hardware** inside Session 2, before any long run:
a 20M sweep (5 OPERA probes — lr {0.01, 0.02, 0.04} at wd 0.01 with
the routing fix, a wd-0 control, a routing-off control; 3 TF probes —
lr {0.01, 0.02, 0.04} at wd 0.01; 2,000 steps each on the FineWeb
pool, one probe per GPU) followed by a 155M × 250-step confirmation
of the top-2 lr. **Adoption rules, pre-stated:** keep lr 0.02 unless
another value wins by >1%; keep wd 0.01 unless 0 wins by >1%; keep
the fusion_gate routing (it is a correctness fix) unless off wins by
>1.5%; an at-scale (155M) ranking flip overrides the 20M lr ranking.
The adopted recipe is written to state.json and used by every later
stage. Session order changes accordingly: the free CPU FineWeb
extension (old session 3) runs BEFORE the GPU smoke so the sweep
measures on the final 80M+ pool.

*Fairness principle, stated:* matched means matched protocol — same
schedule, data, steps, seeds, and measurement. Each arm additionally
runs its best-known recipe; a known misrouting is not part of either
architecture. The TF arms have no fusion_gate analogue, so the
include is OPERA-only by construction, not an asymmetry. H1–H4 and
all other protocol elements are unchanged; H3's two-stage reading
compares THIS study's arms (§4.8 remains context, not a control).

*Not adopted, on the record:* LO-Muon (falsified 2026-09-09: +3.6%
BPB, +4.9% extrapolation at the byte rung with routing confirmed);
over-relaxation (BPB-negative at the byte rung, unvalidated under
Muon, and its benefit is a trajectory metric this study does not
gate on); the bistable arms (falsified); byte-level representation
(§5.1's adoption rule settled it; token-matched §4.8 comparability);
`mem_mode='delta'` (implemented and selftested but never validated —
a post-study arm, not something to smuggle into a 3-week commit).
