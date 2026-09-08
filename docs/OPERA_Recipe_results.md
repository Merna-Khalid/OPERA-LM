# Recipe results — byte d512 rung, 2026-09-09

**Status:** RESULT, not a pre-registered study. These arms were run
after noticing that every byte run in the representation and mechanism
series used `train()`'s defaults, i.e. **none** of the improvements this
project had already validated. Recorded as a recipe measurement with no
gates; the numbers are what they are.

Rung: byte d512/nb128/L2, T=1024, batch 8, 3000 steps, seed 42 —
identical to every arm in the 2026-09-08 series. Incumbent is
`bist_off` (BPB 2.1308), which reproduces `bytes_d512` to ΔBPB 0.00088.

---

## Results

| arm | recipe | BPB | vs incumbent | extrapolation (1025–2048) | vs incumbent |
|---|---|---|---|---|---|
| `bist_off` | AdamW lr 1e-3, cosine | 2.1308 | — | 4.3650 | — |
| **`rx_muon`** | **Muon lr 0.02** | **2.0508** | **−3.75%** | **4.0711** | **−6.73%** |
| `rx_msup` | msup weight 0.1 | 2.1255 | −0.25% | 4.3462 | −0.43% |
| `rx_full` | Muon + msup + curriculum(128,250) + WSD | 2.0687 | −2.92% | 4.1029 | −6.00% |

Muon partition on this model: 2,236,672 params under Muon /
1,056,775 under AdamW.

## Findings

**1. Muon alone is the largest single improvement measured in this
project's byte series** — larger than all seven mechanism arms combined,
in the right direction on both metrics, from a change that required no
new code. `opera_lm/muon.py` was already written, selftested, exported,
and validated at the toy rung (186.45 vs AdamW 224.88, +38.4 PPL, under
a pre-registered gate). It had simply never been passed to the byte
runs, because `experiments/repr_study.py` never set `optimizer=`.

**2. Stacking HURTS.** `rx_full` is 0.83% worse than `rx_muon` on BPB
and 0.78% worse on extrapolation. Adding msup + curriculum + WSD on top
of Muon costs quality rather than adding to it.

**3. msup does not transfer to the byte rung.** −0.25% BPB here against
−9.87 PPL at the word-level toy rung where it was validated. Since msup
is the only *objective-side* win in the project's record, and its
targets are "the first token after each node's span," the natural
reading is that span-boundary supervision is much weaker signal at byte
granularity than at word granularity — a byte span boundary is usually
mid-word. Not tested; stated as the most likely explanation.

Given (2) and (3), the remaining cost in `rx_full` is attributable to
curriculum, WSD, or their interaction with Muon. Not isolated.

## Consequences for the rest of the record

**The new byte incumbent is BPB 2.0508** (`rx_muon`).

**Every comparison in the 2026-09-08 series was measured on an
AdamW-undertrained baseline.** That covers the representation study
(bytes vs BPE) and all seven mechanism arms — bistable ×3, level-balance
×3, over-relaxation ×2. Specifically:

- **The representation result may have moved.** The byte ladder crossed
  BPE at 2.1317 vs 2.1461, a 0.67% margin. Both were AdamW numbers.
  Byte-level is now at 2.0508, which is 4.4% below BPE's AdamW number —
  but BPE has not been re-run under Muon, so **the margin is unknown in
  both directions** until it is. This should be re-measured before the
  representation claim is used anywhere.
- **Mechanism nulls may be optimizer artifacts.** Seven arms were judged
  against a baseline that was 3.75% off its achievable quality. The
  gating nulls are probably safe (the *convexity* argument that explains
  them is structural, not optimizer-dependent), but the level-balance
  falsification and the over-relaxation positive were both measured on
  AdamW and neither has been reproduced under Muon.

## Open, unfixed

**`fusion_gate` is excluded from the Muon partition by accident.**
`EXCLUDE_SUBSTR = ('emb', 'gate', 'quat', 'theta', 'beta')` in
`opera_lm/muon.py:130` was written to keep embeddings, quaternions,
gains and 1-D parameters out. The substring `'gate'` also catches
`fusion_gate` — two (384, 1024) matrices that *are* the compose node's
fusion, and ordinary well-conditioned 2-D weights. Measured at d=512:
**917,760 of 3,287,040 two-dimensional parameters (28%) routed to AdamW
by name match.**

Composition of the current Muon partition, for context on where the
−3.75% is coming from:

| tensor | params | share |
|---|---|---|
| `cross_mlp` (×4) | 2,097,152 | 93.8% |
| `head` | 132,608 | 5.9% |
| `rot_free` | 6,912 | **0.3%** |

Muon's gain here is almost entirely from optimizing a plain 2-layer MLP.
The geometric parameters are 0.3% of what it touches and the fusion
gates are not touched at all.

**Also unfixed:** `muon.py:126` sets `weight_decay=0` "by design" to
keep the original A/B clean. Moonshot (arXiv:2502.16982) names adding
weight decay as one of exactly two techniques crucial for scaling Muon;
the other — the `max(1, m/n)^0.5` update rescale — is already present at
`muon.py:102`.

**Not pursued:** exact polar retraction for the 3×3 rotors. Newton–Schulz
is measurably worse at 3×3 (orthogonality error 0.5594 vs 0.15–0.21 at
the large shapes it was tuned for; singular values spanning 0.0022–1.2049
where exact polar gives 1.0000). But "Muon is Not That Special: Random
or Inverted Spectra Work Just as Well" (arXiv:2605.11181) argues exact
orthogonalization is not the active ingredient, which undercuts the
motivation. Recorded because the measurement is real even though the
inference is doubtful — and because `rot_free` is 0.3% of the partition,
so the ceiling on this fix is low regardless.

## Next

1. `fusion_gate` into the Muon partition (narrow `'gate'` to
   `'blend_gate'`). Largest measured misallocation.
2. Muon + weight decay.
3. Re-run BPE under Muon before quoting the representation margin.
4. Optionally isolate which of curriculum / WSD costs `rx_full` its
   0.83%.
