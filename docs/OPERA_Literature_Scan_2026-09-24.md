# OPERA literature scan — 2026-09-24

Companion to `OPERA_Readout_survey.md` (2026-09-13, + DeepSeek addendum
2026-09-21) and `OPERA_Improvement_Survey_2026-09.md`. Scope: work from
roughly March–September 2026 not already covered there (Log-Linear
Attention, Adaptive Memory Decay, Engram, DeepSeek-V4/mHC, Mamba-3, BLT,
NSA, DeltaNet are covered in those documents).

**Standing constraint (Merna, 2026-09-24): no attention, ever.** Not over
positions, not over chunks, not over layers, not as a teacher, and not
its linear/kernel forms (a content query read against stored keys). Every
item below is classified against that rule in §4; the recommendations in
§5 use only attention-free items.

Verification note: abstracts of SAMuon, Dual Attention Residuals, SMT,
MLP-LDRU, Attention Residuals and Multi-Byte Prediction were read
directly; PR-LSTM, Prefix-Scannable Models and M²RNN details come from
their full-text HTML via summarization (specific and mutually
consistent, but not line-checked).

---

## 1. Novelty: the closest prior work (cite in the paper)

### 1.1 Prefix-Scannable Models — Yau et al., ICLR 2026 (arXiv 2506.10918)

Formalizes the model class OPERA belongs to.

- **Training:** a static (Blelloch-style) tree scan over the sequence.
- **Online inference:** a *binary-counter scan*. The binary expansion of
  t+1 says which block sizes 2^k are present; at most one mini-tree root
  per block size is stored (≤ ⌈log₂(t+1)⌉ roots); each trailing 1-bit
  triggers a merge and a carry; the prefix value aggregates the occupied
  roots from most- to least-significant bit. This is OPERA's Fenwick fold
  and `OperaDecoder`, exactly.
- **Non-associative operators are allowed.** Each prefix's output is then
  well-defined but depends on the tree's parenthesization — OPERA's
  "position is structure". Training and online inference evaluate the
  same tree, so they agree (their Theorem 4.2; OPERA's decoder selftest).
- Their instantiation (Transformer-PSM) is attention-based: chunked
  softmax attention as the aggregator plus an attention "inference
  module" over the current chunk. Results: WikiText-103 PPL 22.45 at
  chunk 256 vs GPT-2 22.28; S5 state tracking trained to length 18,
  generalizes beyond 160; perfect MQAR at chunk 64.

**Consequence for OPERA's claims.** The Fenwick/binary-counter structure,
O(log T) online state, and non-associative tree composition are no
longer OPERA-specific. What remains distinct: the geometric (Cl(3)-even)
compose node; no attention anywhere in the model; token-level (not
chunk-level) tree leaves; language-modeling results and position curves
for a fully attention-free tree model. `OPERA_Readout_survey.md` §5
(novelty check) predates this and should be updated.

### 1.2 The 2026 non-associative tree cluster

- **Parallel Recursive LSTM (PR-LSTM)** — arXiv 2605.17108. Tokens are
  mapped independently to (h, c) states, then merged over a balanced
  tree by an LSTM-style block: binary composition
  `c = i⊙u + f₁⊙c₁ + f₂⊙c₂`, `h = o⊙tanh(c)`, followed by a unary
  refinement step. O(log T) depth, O(T) work. Formal languages only
  (train length ≤ 40, test to 500): 6/15 tasks solved vs LSTM 5/15, RNN
  4/15, Transformer 2/15. Explicitly no language-model evaluation; no
  per-prefix readout (sequence classification).
- **MLP-LDRU (Log-Depth Recurrent Unit)** — Pert, Alrajeh, Russo, arXiv
  2605.26035. Operator: `[g_i; g_j] = MLP([h_i; h_j])` (hidden 2d→4d→2d),
  `f_i = V_i (g_i ∘ h_i) + b_i` with V_i **identity-initialized**, output
  `W_out (f_i + f_j) + b_out`; shared residual FFN + LayerNorm between
  tree steps; zero-padding for odd lengths. 21 regular-language tasks:
  100% OOD accuracy on 18, ≥ 99.9% on the other 3. Operator ablation on
  Modular Arithmetic: element-wise sum 32.6%, linear 60.8%, **gated sum
  67.1%**, GRC 82.3%, MLP-LDRU 100%. For ListOps an explicit
  **associativity regularizer** (cosine distance between (a∘b)∘c and
  a∘(b∘c)) improved generalization. Sequence classification only.

Neither PR-LSTM nor MLP-LDRU does language modeling — OPERA is the LM
evidence this line lacks.

---

## 2. Mechanism evidence relevant to open OPERA findings

### 2.1 Value mixing inside the compose node (review item 4; F3/F4)

OPERA's node value path is block-diagonal per quaternion slot (measured
2026-09-24 on `rx_fgate_wd`: 54–77% of the ∂parent/∂child Jacobian
energy in the diagonal 4×4 blocks; cross-block influence is only
multiplicative, via gates). MLP-LDRU's ablation places "gated sum" — the
class OPERA's per-slot gated combine belongs to (plus the geometric
product) — at 67.1% vs 100% for an operator with **full, identity-
initialized value projections**. External support for adding cross-slot
value mixing to the node.

### 2.2 State size is the capacity knob (review item 4; `state_mult`)

**M²RNN** — Mishra, Tan, Stoica, Gonzalez, Dao, arXiv 2603.14360.
Matrix-valued non-linear RNN: `Z_t = tanh(H_{t-1} W + k_t v_tᵀ)`,
`H_t = f_t H_{t-1} + (1−f_t) Z_t`, `y_t = H_tᵀ q_t + w_r ⊙ v_t`.
Ablations: **state size, not non-linearity, drives the gap** between
non-linear and linear RNNs; the forget gate is essential for gradient
stability. S3 permutation tracking generalizes perfectly beyond
training length; in a 7B-MoE hybrid, −0.4 to −0.5 WikiText PPL vs Gated
DeltaNet hybrids at 3× smaller recurrent state. (The hybrids interleave
attention; the ablation finding about state size is the transferable
part. Its readout `H_tᵀ q_t` is a content query — see §4.)

### 2.3 An additive path through the tree (F3)

Every OPERA node ends in LayerNorm + tanh, so there is no unsquashed
channel from leaves to high tree levels. PR-LSTM's cell `c` is exactly
such a channel (forget-gated additive merge of both children's cells,
exposed through an output gate).

### 2.4 Associativity as a knob (F6)

MLP-LDRU's regularizer makes the operator approximately associative,
i.e. the prefix readout becomes nearly independent of the Fenwick
decomposition shape. For OPERA this would likely shrink the
power-of-two CE texture (F6) — and would equally erase the structural
position signal the "position is structure" thesis rests on (F5 already
measures that signal as small). A deliberate trade-off, not a free win.

---

## 3. Recipe / optimizer

- **SAMuon (Spectral-Aware Muon)** — Wu, Yu, Zhang, Woodland, arXiv
  2608.25990 (Aug 2026). Out-of-sample spectral probing of the momentum
  buffer: a "volatile head" (dominant singular direction, at the edge of
  stability) needs a small step; the "tolerant bulk" permits several
  times larger steps. Muon's uniform whitening underuses the bulk.
  SAMuon holds the head at Muon's scale and amplifies the bulk by γ
  (γ = 1 recovers Muon). SAMuon-lite: two-level profile via rank-one
  power iteration, near-zero overhead, no extra optimizer state.
  modded-nanogpt 124M–1B: **13.3–24.0% fewer tokens** than Muon to the
  same validation loss; beats tuned AdamW and Muon in every tested
  scale/batch configuration.
  *OPERA caveat:* `rot_free`'s 3×3 blocks have almost no spectrum to
  reallocate; the effect would come through `fusion_gate` (1024×384 per
  layer at d512) and `cross_mlp`.
- **NorMuon** — arXiv 2510.05491 (ICML 2026). Normalizes per-neuron
  update norms after orthogonalization. 124M: Muon needs 6% more
  iterations than NorMuon; 1.1B: 11.3% better than Muon.

---

## 4. Attention filter (Merna's rule: no attention in any form)

| item | uses attention? | usable? |
|---|---|---|
| Prefix-Scannable Models (Transformer-PSM) | yes (softmax aggregator + chunk module) | cite only; the binary-counter scan itself is attention-free and already OPERA |
| PR-LSTM cell channel | no | **yes** |
| MLP-LDRU identity-init value mixing | no | **yes** |
| MLP-LDRU associativity regularizer | no | yes, with the §2.4 trade-off |
| M²RNN state-size finding | no | **yes** (the finding; not its `Hᵀq` query readout) |
| SAMuon / NorMuon | no | **yes** |
| Attention Residuals (Kimi, 2603.15031), Dual Attention Residuals (2607.18730) | yes (softmax over earlier layers) | **no** |
| DenseFormer-style depth averaging (learned *static* scalar weights over earlier layers' outputs; cited in DAR) | no | yes — the attention-free way to let the head/next layer reach mid-stack readouts (F7/G3) |
| mHC (Birkhoff/Sinkhorn-constrained stream mixing; DeepSeek addendum §8.2) | no (doubly-stochastic mixing, no query–key scores) | yes, depth phase |
| Supervised Memory Training (Kumar & Isola, 2606.06479) | yes (Transformer teacher) | **no** |
| Multi-Byte Prediction (2608.15454) | yes (hierarchical Transformer with attention masking) | no; plain multi-token heads on OPERA would be attention-free but low priority (msup did not transfer to bytes: −0.25%) |
| Log-Linear-style per-level query gates (survey T1) | content-query gating | no (already rejected 2026-09-13) |
| Engram | hash lookup is attention-free; its gate σ(RMSNorm(h)·RMSNorm(k)) is a query–key score | only with the gate replaced by a plain learned gate |

**In-repo arms that are attention and therefore off the table:**
`fold_mode='attend'`, `fold_mode='spine'`, `workspace=True` (scan), and
`mem_mode='delta'` (a content query `M q` against keys written by the
delta rule — linear attention in the DeltaNet sense). The code can stay
(selftested infrastructure) but none of them should be run.

---

## 5. Recommended improvements (attention-free only)

Ordered by expected value per unit of effort. Implementation status
(2026-09-24): items 1–7 and 9 are implemented as flags and selftested
(`samuon_gamma`, `node_mix_rank`, `state_tie` added after this scan);
item 8 is not. Exploratory runs queued as repr_study arms `st_off`,
`st_head`, `x_sam` (γ=3.54), `x_nmix` (r=64), `st_auxw0`, `st_wide`,
`x_tie2`; results to be appended below.

**Tier 1 — cheap, high expected value**

1. **`head_mode='stream'`** (implemented). The head reads LN(stream);
   reclaims the 34% of L=2 parameters that currently receive no
   gradient. Compare against `st_wide` (d640) so a capacity win isn't
   credited to topology.
2. **`aux_weight`** (implemented, train-side). 0.5 on every non-final
   layer leaves the final layer only 22% of the loss at L=8; try 0 / 0.1.
3. **SAMuon-lite** on `fusion_gate` + `cross_mlp` (not implemented;
   optimizer-only, ~1 hour). Muon was the largest single gain in the
   project (−3.75% BPB); this is its direct successor. Sweep γ lightly.

**Tier 2 — architecture, directly aimed at F3/F4**

4. **Identity-initialized cross-slot value mixing in the node**
   (MLP-LDRU evidence, §2.1). Add a mixing map on each child's values
   (or on the node output) initialized to identity, so step 0 is
   bitwise the incumbent with gradients flowing both ways. Full d×d
   costs ~2d² per node vs the gate's 1.5d²; use low-rank (I + UVᵀ, U
   zero-init) or a banded/block form.
5. **Tied-gate state expansion** (M²RNN finding, §2.2). Each quaternion
   slot carries m quaternions that share the slot's gates and rotations;
   leaves are projected d → m·d and the readout back to d. State grows
   m× while gate cost stays flat — the cheap variant of the implemented
   `state_mult` (whose fusion gate grows ~4× at k=2).
6. **`state_mult`** (implemented) as the expensive reference point for 5,
   with the width-matched control (d920 / d800).

**Tier 3 — targeted fixes and depth**

7. **Fold-side:** `fold_h0` (F6 zero-fold readouts) and
   `fold_gate='separate'` (tree vs fold roles) — implemented.
8. **Additive cell channel in the node** (PR-LSTM, §2.3): a forget-gated
   additive merge `c = f_L⊙c_L + f_R⊙c_R + i⊙u` carried alongside the
   spinor state, exposed through an output gate — an unsquashed path
   from leaves to high tree levels.
9. **Depth (when moving to L=8):** `resid_mode='add'` (implemented),
   then either DenseFormer-style static depth weights for the head input
   (F7/G3) or mHC-style constrained stream mixing (G1/F4). Both are
   attention-free.

**Deprioritized:** associativity regularizer (conflicts with "position is
structure"; only if F6 texture becomes loss-binding); multi-token heads
(byte-rung evidence is weak).

**Paper to-do:** cite Prefix-Scannable Models, PR-LSTM and MLP-LDRU in
related work and update the novelty statement (§1.1).

---

## 6. Sources

- Sequential-Parallel Duality in Prefix Scannable Models — arxiv.org/abs/2506.10918 (HTML v1; ICLR 2026, mlanthology.org/iclr/2026/yau2026iclr-sequential)
- Parallel Recursive LSTM — arxiv.org/abs/2605.17108
- Length Generalization with Log-Depth Recurrent Units — arxiv.org/abs/2605.26035
- M²RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling — arxiv.org/abs/2603.14360
- Spectral Allocation: Why Muon Outperforms Adam, and How to Improve Muon — arxiv.org/abs/2608.25990
- NorMuon: Making Muon more efficient and scalable — arxiv.org/abs/2510.05491
- Attention Residuals (Kimi Team) — arxiv.org/abs/2603.15031 (listed for the filter; not usable)
- Dual Attention Residuals — arxiv.org/abs/2607.18730 (listed for the filter; not usable)
- Pretraining Recurrent Networks without Recurrence — arxiv.org/abs/2606.06479 (listed for the filter; not usable)
- Dynamic Multi-Byte Prediction With Hierarchical Language Models — arxiv.org/abs/2608.15454
- Log-Linear Attention — arxiv.org/abs/2506.04761; Adaptive Memory Decay for Log-Linear Attention — arxiv.org/abs/2605.06946 (already covered in earlier surveys)
