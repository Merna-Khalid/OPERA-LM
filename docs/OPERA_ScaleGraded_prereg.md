# OPERA scale-graded readout — pre-registration (2026-09-13)

**Status: REGISTERED before implementation. No graded arm has been
built or run.** Edits to gates after results are visible are protocol
violations and void the arm. Exploratory ancestor:
`OPERA_Depth_Survival_diagnostic.md` (F1–F7, G1–G4). Survey context and
steer: `OPERA_Readout_survey.md` §6 (no attention-family mechanisms).

## 1. Mechanism

One line: **stop multiplying the Fenwick path's scales into a single
rotor — give each scale its own subspace, so the tree's dyadic geometry
becomes the geometry of the state.**

Measured premise (the before-construction kill-switch, already run —
the diagnostic): quaternion composition preserves factor addressability
at any span (F3 span-flatness), yet byte t−k is undecodable from the
folded readout for k≥8 (F3) — the destruction happens at the fold,
where ⌈log T⌉ blocks are composed into one d-vector. A single rotor
per quaternion slot has 3 dof; it cannot address ten scales. v9-G2
corroborates the substrate: levels already carry distinct multi-scale
content for free.

Specification (partition variant; the twist variant stays unregistered):

- Slot layout over the nb quaternion slots: slots [0, R) are **global**
  and receive the classical sequential fold over ALL path blocks —
  incumbent behavior, unchanged. Slots [R, nb) are **graded**, divided
  into G groups of equal width.
- Level routing: `group(ℓ) = min(ℓ, G−1)` — clamped, identical mapping
  at every T (levels 0…G−2 own groups; all deeper levels share the top
  group). Extrapolation-safe by construction: no slot sees a new
  level→group pairing at eval length; deeper levels pile into a group
  that already means "far past".
- Within each group, its path blocks (in path order, oldest→newest as
  today) compose sequentially with the SAME scale-tied operator. No new
  weights anywhere: routing only. Value-flow stays block-diagonal per
  quaternion slot (masked composition); the fusion gate may keep
  conditioning on the full 2d input — gates global, values never mixed
  across slots.
- Invariants: I1 ✓ (level is scale, circuit shape — same argument as
  `level_sin_enc`, not an injected coordinate). I2 ✓ (no scores, no
  routing by content — routing is by structure). I3 ✓ (compose node
  untouched). I4 ✓ (same Fenwick prefix set, same causality). Params:
  byte-identical to incumbent — matched by construction, zero new
  tensors. NOT bitwise-incumbent at init: the information topology
  differs from step 0 by design; the control arm is the incumbent
  architecture.

## 2. Arms (single dose; no sweeps without amendment)

| arm | R (global slots) | G | graded slots/group | note |
|---|---|---|---|---|
| `grade_off` | — | — | — | incumbent config, fresh same-session control |
| `grade_r24` | 24 | 8 | 13 | hybrid: keeps ¾ of the incumbent compressor + graded scale memory |
| `grade_full` | 0 | 8 | 16 | radical: pure scale-graded state |

Protocol (identical to this week's validation family): repr_study
pipeline, bytes T=1024, d=512, nb=128, L=2, 3000 steps, batch 8, seed
42, Muon 0.02 / include fusion_gate / wd 0.01, eval to 2048. ~75–95
min/arm on MPS.

## 3. Gates (pre-stated)

- **SG-H1 (adopt):** a graded arm with BPB ≤ 0.99 × `grade_off`
  (same-session) AND 1025–2048 extrapolation PPL not worse than
  `grade_off` by >1%. Adoption ⇒ Path A amendment + full probe battery
  rerun on the adopted arm.
- **SG-H2 (mechanism kill-switch):** the adopting arm must show k=8
  byte decodability from the final readout ≥ 0.20 (incumbent ≈ 0.105;
  ≈2×). If SG-H1 passes but SG-H2 fails: NO adoption — the loss moved
  without the mechanism engaging; record as anomaly, treat as noise.
- **SG-H3 (mechanism expectations, non-binding):** k=1 decode ≥
  incumbent; position-bin decode > 0.08 (incumbent 0.057); per-group
  participation ratios reported (graded slots must not all collapse to
  one cone).
- **SG-H4 (guards):** param counts byte-identical across arms; the
  routing unit-test (a level-ℓ block perturbation must not move other
  groups' slots) and the causality selftest pass; single dose as
  registered.

## 4. Pre-stated interpretations (binding)

- **H1+H2 PASS** → adopt into Path A via amendment; geometry claim
  upgraded: the readout is a scale-graded multivector.
- **H1 PASS, H2 FAIL** → no adoption (SG-H2 protects against
  noise-fishing); one retry of the same arm allowed only if BPB margin
  is inside the same-session noise band.
- **H1 FAIL, H2 PASS** (mechanism engages, no BPB gain) → the geometry
  is expressible but does not pay at the 20M/3.3M rung; NOT adopted;
  revival requires a registered scale argument (e.g., as a Path A
  instrument or at longer T), not a re-roll.
- **BOTH FAIL** → closed; joins gb on the ruled-out shelf as "fold
  topology changes at fixed params do not move this rung."

## 5. Relationship to the in-house record (prior results, disclosed)

- `fold_gate_bias` (v9 arm A/gb): FALSIFIED — static transport bias is
  a uniform tax. Different mechanism: gb reweighted a mixing fold;
  grading changes the mixing topology itself and adds no weights.
- `fold_scale` (span twist): implemented, NEVER run as an arm —
  untested, not falsified. Not registered here; the partition variant
  is the cleaner bet (structural separation vs phase separation).
- fold attend (v8.0): FALSIFIED, and off-program by the 2026-09-13
  steer (no attention-family mechanisms).
- Improvement-Survey adaptive-λ (content-gated per-level softplus):
  house-designed, untested, OFF-program under the same steer
  (content-gating); the registered arms above are strictly
  content-blind.
- msup (v9 B, PASS): levels carry multi-scale span signal — supporting
  evidence for the substrate, orthogonal to routing.

## 6. Risks (honest)

- Capacity split: 13–16 slots/group is narrow; if the global compressor
  needs more than R=24, `grade_r24` will show it and `grade_full`
  tests the extreme. That is the dose, not a sweep.
- The gate may learn to route around graded slots (fusion-gate
  conditioning sees everything); SG-H2 detects this — if k=8 stays at
  incumbent level, the routing died in training and the arm fails
  cleanly.
- Same-session MPS noise ≈ 0.04% (within-family control handles it);
  cross-session drift ~0.2% is why `grade_off` reruns fresh.
- L=2 only: depth interaction untested here (the diagnostic says the
  fold is the destroyer at every depth; if adopted, Path A probes
  cover L=8).
