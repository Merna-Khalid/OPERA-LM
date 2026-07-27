# PRE-REGISTRATION: The Geometric Timeline Line (v8.10 / v8.11)
.
**Written:** 2026-07-25, before any implementation or results.
**Incumbent:** `--fold left --rack-exitnorm`-era LEFT fold, same-session re-run required.
Historical seed-1 references (22M rung, docs-en, T=1024/eval 4096):
in-length 45.11; extrapolation buckets 27.38 / 15.98 / 8.41. These are
REFERENCES ONLY — all gates are judged against the SAME-SESSION incumbent.

---

## 0. The shared hypothesis and how the two arms split it

The scaling campaign's diagnosis: the prefix state is a write-time-compressed
endpoint; the loss bills its lossiness in-length everywhere, and read-time
access to more history does not help at this scale (attend-fold null).

The "geometric timeline" claim is that the *trajectory* h_1..h_t through state
space carries information the endpoint h_t discards. The two arms test two
non-equivalent versions of that claim:

- **r19 / v8.10 TRAJECTORY (write-side):** enrich what each position's
  representation *contains* — causal path features (swept-area / Lévy
  bivectors) of the prefix-state trajectory, injected into the readout.
  No change to access. Tests: "the endpoint is the bottleneck."
- **r20 / v8.11 SNAPSHOT CACHE (read-side):** enrich what each position can
  *read* — one-hop access to the full prefix states at exponential lags
  t−1, t−2, t−4, ..., t−2^⌊log t⌋. No change to state content.
  Tests: "access to past states (not span-block summaries) is the bottleneck."

**Discrimination logic (pre-committed interpretation):**
- r19 wins, r20 nulls → bottleneck is written content; availability≠use
  extends from Fenwick blocks to lagged states. Timeline line continues on
  the write side (higher-order signature terms are the follow-up arm).
- r20 wins, r19 nulls → the attend-fold null was about *block summaries*,
  not access per se; lagged full states are the useful reads. Continue
  read side.
- Both null → the timeline hypothesis is falsified at this rung in both
  its write and read forms; the line CLOSES (one rematch max, per SO(3)
  precedent, and only with a mechanism not reducible to r19/r20).
- Both win → stack in a single follow-up arm (single variable: the stack
  vs the better singleton).

**Prior on record (stated for honesty, not a gate):** the attend-fold result
predicts r20 nulls. r20 runs anyway because its reads differ in kind
(composed prefix states vs disjoint span blocks) and because a clean second
null is what upgrades "availability ≠ use" from one result to a principle.

---

## 1. r19 / v8.10 — TRAJECTORY READOUT (`--traj levy`)

### Mechanism (exact)
The left fold already produces the per-position prefix state h_t for every t.
Per 3-D geometric block b, define displacements δ_t = h_t − h_{t−1} (δ_1 = 0)
and the causal cumulative **Lévy area** (level-2 antisymmetric path
signature, Chen):

    A_t^(b) = ½ · Σ_{s≤t} ( h_{s−1}^(b) × δ_s^(b) )        ∈ R³ per block

Computation: one elementwise cross product + one `cumsum` over T — no scan
operator change, no new recurrence, exactly causal by construction
(selftest), no positional parameters (extrapolation-safe by construction).
Blocks whose native width ≠ 3 use the first 3 scalars per block slot
(spinor states: vector part); exact slicing fixed at implementation and
asserted in the param/shape selftest.

Injection: `traj_proj: Linear(3·n_traj_blocks → d)`, **zero-init**, added to
the position's pre-readout state, then the unchanged
exitnorm → cross_mlp → blend path. Per-layer or final-layer-only is NOT a
free choice: **final layer only** (one variable; per-layer is a follow-up).

### Anchors (selftests, all must pass before training)
- (a) `traj_proj` zero-init ⇒ step-0 forward is BITWISE `--fold left`.
- (b) causality: perturbing token t+1 changes no A_s, s ≤ t.
- (c) A_t matches a reference O(T²) loop on random inputs.
- (d) param delta = 3·n_traj_blocks·d + d exactly; ≤ 5% of 22M rung.
- (e) bounded values at T=4096 (no drift; cumsum of bounded cross products
  over normalized states — assert max |A| growth ≤ O(T), gate normed).

### Budgets (binding)
- Params: ≤ +5% at the 22M rung.
- **Step time: ≤ +10% sec/step vs same-session left, same GPU, measured
  at step 50–100 median.** (Mechanism is one cumsum + one matmul; if it
  costs more than 10% the implementation is wrong, not the idea.)

### Gates (binding both ways, vs SAME-SESSION left)
- **P1 (decision):** in-length PPL improves by > 1.5 absolute
  (outside the established ±1.5 noise band).
- **P2 (secondary, independent claim):** extrapolation bucket PPLs — no
  bucket worsens by > 4% relative (guard), and any bucket improving by
  > 4% relative is reportable.
- **G1 (use-guard):** if P1 fails, report ‖traj_proj‖ and the gradient
  norm through A. Near-zero learned norm ⇒ "path features available but
  unused" — logged as the write-side extension of availability≠use.
- **FUTILITY:** eval PPL > 2× incumbent at matched step 2000 → kill.
  Sec/step > 1.25× incumbent after the perf pass → kill regardless of loss.

---

## 2. r20 / v8.11 — SNAPSHOT CACHE (`--snap log`)

### Mechanism (exact)
For position t, gather the already-computed prefix states at exponential
lags: S_t = { h_{t−2^k} : k = 0..⌊log₂ t⌋, t−2^k ≥ 1 } (≤ 10 slots at
T=1024; ≤ 12 at eval 4096 — defined for every depth, extrapolates by
construction). One strided `gather` — states already exist; no recomputation.

Read: single attention round, query from h_t, keys = snapshot states + the
deterministic sinusoidal LEVEL encoding of k (reused verbatim from the
attend fold: smooth in k, zero params, defined for unseen k). dk = 64.
Output added through a **zero-init per-channel gate** post-exitnorm
(workspace-style), then the unchanged downstream path. Final layer only
(one variable).

### Anchors
- (a) gate zero-init ⇒ step-0 forward BITWISE left.
- (b) causality: gather indices strictly < t (leak-selftest, workspace-style).
- (c) k=0-only ablation flag (reads only h_{t−1}) reserved as the trivial-
  recency null if r20 passes P1.
- (d) param delta = 2·dk·d + d (+ gate d) exactly; ≤ 5% at 22M rung.

### Budgets (binding)
- Params: ≤ +5%.
- **Step time: ≤ +20% sec/step vs same-session left** (gather + attention
  over ≤ 10 slots; if it exceeds 20% the launches must be batched before
  the run counts).

### Gates (binding both ways, vs SAME-SESSION left)
- **P1 (decision):** in-length PPL improves by > 1.5 absolute.
- **P2 (secondary):** probe pp1/ppLast@256 rises off the floor to ≥ 0.05
  AND at least one extrapolation bucket improves > 4% relative — access
  arms must show the access being *used*, or a P1 win is attributed to
  incidental capacity and the k=0 null (anchor c) runs before any claim.
- **G1 (guard):** no extrapolation bucket worsens > 4% relative.
- **FUTILITY:** as r19 (2× PPL at step 2000; 1.25× sec/step hard cap).

---

## 3. Standing efficiency criterion (the transformer clock)

Carried on every arm in this line, unchanged from the v8.4 scale gate:
**tokens/sec at eval T=4096 vs the t3 NoPE baseline on the same GPU —
OPERA must be faster at T ≥ 2048, or the efficiency claim is not made.**
Sec/step at the 22M rung is recorded in the results jsonl for every arm;
an arm that wins its PPL gate but breaches its step-time budget does NOT
advance to a scale rung until the perf debt is paid and re-measured.

## 4. Run plan (same session, same GPU, same seed, same flags)

1. Same-session left incumbent re-run (sets all gate baselines + sec/step).
2. r19 (cheaper, more novel, no adverse prior).
3. r20.
4. Diagnostics on every arm: position curve, interrogation probe, fold
   diagnostic, extrapolation buckets, sec/step, ‖gate‖ trajectories.
5. Interpretation per §0 discrimination table — written before results,
   applied without renegotiation.

**Version banners:** v8.10 (r19), v8.11 (r20). Strict flag parsing.
Zip bundle per session. Any k>1-style bundling of the two mechanisms in
one arm is prohibited (single-variable discipline).

---
*Gate numbers marked binding become binding when Merna confirms them in
writing. Edits to thresholds after results are visible are protocol
violations and void the arm.*
