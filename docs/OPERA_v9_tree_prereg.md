# PRE-REGISTRATION: OPERA v9 Tree-Training Line — Toy Rung

**Rung:** 5.89M params (d=256, nb=64, L=2, vocab 10k), `--data docs --pe none
--rot free --seed 1`, train max_len 256, eval 512, 2000 steps, batch 16.
Same machine, same session, sequential execution (no GPU contention).
**Incumbent:** `--fold left`, no v9 flags, run FIRST — it sets every baseline.

## Arms (single variable each)

1. `incumbent` — baseline re-run.
2. `gb` — `--gate-bias 2.0,0.0,-2.0` (arm A: chrono fold transport init).
3. `msup` — `--msup` (arm B: multi-scale node supervision, write-side).
4. `exitnorm` — `--fold rack --rack-exitnorm` (arm D: isometric interior,
   boundary norm).
5. `cur` — `--curriculum 64:250` (arm C: 64→128→256, full length from
   step 500).

`--fold spine` (read-side) is DEFERRED: it runs only if arm B passes P1
(access is a fair test only when the objective pays for use).

## Gates (binding both ways, vs the same-session incumbent)

- **P1 (decision, arms gb/msup/exitnorm):** in-length PPL improves by
  > 1.5 absolute (the established noise band at this rung).
- **P1-cur (arm C, adjusted):** arm C spends 25% of its budget below full
  length, so in-length comparison is disfavored by construction. C's
  decision metric is the **extrapolation buckets + sec/step**; an
  in-length result within 1.5 abs of the incumbent counts as PASS.
- **P2 (guard, all arms):** no extrapolation bucket worsens by > 4%
  relative; any bucket improving by > 4% relative is reportable.
- **G1 (arm A guard):** report final `fold_gate_bias` drift per row
  (g0/g1/g2). If the trained bias returns to ~0 everywhere, the init was
  decoration — logged as the fold-transport extension of the db-5.0
  lesson ("init-only nudges get undone when the objective doesn't pay").
- **G2 (arm B guard):** report the msup aux loss trajectory; if P1 fails
  with the aux loss low, nodes learned their targets without helping the
  readout — the supervision/readout split is the next question, not more
  supervision weight.
- **FUTILITY:** divergence (NaN) or final in-length PPL > 2× incumbent →
  arm killed, recorded as failed.

## Interpretation (pre-committed)

- A passes → transport init matters at this rung; sweep b0 ∈ {1, 2, 4}
  as one follow-up, then carry the winner to the 22M decontaminated rung.
- B passes → objective-side pressure works; spine becomes a fair test and
  runs next.
- D passes → the r17 diagnosis confirmed (norm at the boundary, geometry
  inside); exitnorm becomes the default fold for the 22M rung.
- C passes → curriculum is the training default for all later rungs
  (cheap + extrapolation-safe).
- All fail → the toy rung is uninformative for training-method arms;
  escalate ONE arm (the prior-strongest) to the 22M rung before
  concluding anything (toy→rung inversion has precedent: levy).

*Edits to thresholds after results are visible are protocol violations
and void the arm.*

---

## OUTCOMES (recorded 2026-07-26, after the runs; thresholds unchanged)

Same-session numbers (incumbent: in-length 224.54, bucket 257–512 104.10):

| arm | in-length | Δ | bucket | verdict |
|---|---|---|---|---|
| gb (A) | 227.45 | +2.91 | 105.52 (+1.4%) | **P1 FAIL** |
| msup (B) | 214.67 | **−9.87** | 100.29 (−3.7%) | **P1 PASS** (6.6× noise band) |
| exitnorm (D) | 266.26 | +41.72 | 118.46 (+13.8%) | **P1 FAIL** |
| cur (C) | 225.48 | +0.94 | 104.35 (+0.2%) | **P1-cur PASS** |

Sec/step: incumbent 0.29, gb 0.30, msup 0.46 (msup path skips the
aux-frac optimization — engineering debt, not math), exitnorm 0.27,
cur 0.28.

**G1 (arm A):** the chrono init HELD — g0 +1.79/+1.78 (from +2.0), g1
≈ 0.0, g2 −1.94/−1.97 (from −2.0); max per-block drift 0.35. Unlike
db-5.0, training did NOT undo the init. Verdict: strong pass-through
transport at init is not decoration but a small uniform tax at this
rung. A falsified with mechanism intact; A does not escalate.

**G2 (arm B):** msup-trained nodes 5.95 vs incumbent 6.18 (chance 9.21).
Two readings: (i) the supervision is learned by the nodes; (ii) the
UNSUPERVISED incumbent tree already carries substantial span-boundary
signal — the structure holds multi-scale content for free, and paying
the objective for it improves the main readout. B's win is not "nodes
finally learned spans"; it is the objective lever working as diagnosed.

**Interpretation per the pre-committed table:**
- B passes → objective-side pressure confirmed; spine unlocked, runs as
  `msup+spine` vs the msup baseline (single variable on top of B).
- A fails with init intact → closed at the toy rung.
- D fails → recorded; consistent with the rack family's known slow
  convergence (r17: −26 PPL at 20k steps, still descending). Not
  escalated from a 2k-step budget.
- C passes → candidate training default for later rungs; stack test
  (`msup+cur`) permitted after the spine result.
- B's bucket −3.7% relative: noted, NOT claimed (below the 4%
  reportable threshold).

## FOLLOW-UP: spine (read-side), unlocked by B's pass

`msup+spine` vs the msup baseline (214.67 / 100.29), single variable on
top of B, run 2026-07-26: **in-length 218.33 (+3.66), bucket 102.22
(+1.9% rel, inside guard), 0.52 s/step → P1 FAIL.**

This is the FIFTH consecutive readout-side null, and the first under
fair-test conditions (objective paying for multi-scale content via
msup). One-hop access to the fold's own accumulator states does not
help even when the objective pays for use. "Availability ≠ use" is
thereby promoted from repeated observation to a stated principle of the
tree line at this scale: the readout bottleneck is not access.
The read-side line CLOSES at the toy rung (one rematch maximum, at the
22M rung, only if the 22M msup arm passes its gate).

## FOLLOW-UP: the permitted stack (B+C)

`msup+cur64x250` vs the msup baseline (214.67 / 100.29), run 2026-07-26:
**in-length 214.42 (−0.25, inside noise), bucket 99.93 (−0.4% rel,
inside guard), 0.42 s/step (vs 0.46 msup, 0.28 cur-only).**

Verdict: NO INTERFERENCE — C stacks on B at quality parity while
processing fewer tokens (the average hides the early-stage savings:
steps 0–500 run at T=64/128). The B+C stack is the tree line's training
default going forward, and the arm that escalates to the 22M
decontaminated multi-seed rung. Prerequisite (engineering debt): the
msup path bypasses the aux-frac head subsampling — at 22M the msup
overhead is material and must be fixed before the rung.
