# OPERA-LM

**Operadic Planar Equivariant Recursive Algebra** — a language model that
replaces self-attention with bottom-up composition over a binary tree, and
replaces positional encodings with nothing at all.

Token states live in the even subalgebra of Cl(3) (block-diagonal
quaternions: one scalar + one 3-vector channel per block). Composition is
geometric: per-block 3×3 maps, the Clifford geometric product (whose vector
part is the cross product), and learned fusion gates. Causal LM readout is
**exact**: every prefix's hidden state is composed from the blocks of its
Fenwick (binary-indexed) decomposition, at O(T log T) total cost per layer.

Because the Fenwick decomposition of a prefix depends on the binary
representation of its length, prefixes of different lengths are computed by
*structurally different circuits*. Position is not injected and not
inferred — it is the shape of the computation. We call this **position is
structure**.

```python
import torch
from opera_lm import OperaSpinorFenwickTree

model = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160,
                               num_layers=4, pe_mode='none',
                               fold_mode='left', rot_mode='free')
ids = torch.randint(1, 10000, (2, 128))
lengths = torch.tensor([128, 97])
logits = model(ids, lengths)[-1]        # final-layer logits [2, 128, 10000]
```

## Install

```bash
pip install -e .            # core (torch, numpy)
pip install -e ".[data]"    # + HuggingFace datasets, for opera_lm.train / load_data
# or without installing the package:
pip install -r requirements.txt
```

Verify the installation — the full shipped self-test suite (equivalence,
causality, fold variants, param budgets, gradient checks):

```bash
python -m opera_lm.selftest     # ends with: ALL PASS
python examples/train_toy.py    # trains on random tokens, no download
```

## What's in the package

- **`opera_lm.model`** — `OperaSpinorFenwickTree`, the complete model (all
  arms below), plus the functional helpers (`fenwick_blocks`, quaternion
  ops, `geometric_product`, `associative_scan`, `count_params`).
- **`opera_lm.incremental`** — `OperaDecoder`: Fenwick-incremental
  decoding at O(L log T) compose nodes per token (vs O(L T log T) for a
  full re-forward; ~13× faster at T=128, growing with T). Exact vs the
  full forward (selftested, atol 1e-5); `fold_mode='left'` only.
- **`opera_lm.muon`** — `Muon` (Newton-Schulz orthogonalized momentum,
  batched per 3×3 map for `rot_free`) + `split_muon_params`: the T0.1
  optimizer arm, toy-gate validated (+38.4 PPL over AdamW at lr 0.02).
- **`opera_lm.losses`** — `lm_loss`, `train_lm_loss` (with unbiased
  auxiliary-position subsampling), `msup_loss`.
- **`opera_lm.train`** — the full research training loop (`train()`),
  with GPU-resident data, exact batch-stream checkpoint resume, and
  bucketed length-extrapolation evaluation.
- **`opera_lm.data`** — the Wikipedia loaders (`load_data`), needs the
  `[data]` extra.
- **`opera_lm.metal_kernel`** — optional fused Metal (Apple Silicon)
  kernels for the compose node; used only when `use_metal=True`.

## Configuration reference

Every arm is a constructor flag on `OperaSpinorFenwickTree`. Flags left at
defaults reproduce the validated incumbent (the paper's `--fold left`
model) exactly.

### Core

| parameter | default | meaning |
|---|---|---|
| `vocab_size` | — | vocabulary size |
| `d`, `nb` | 768, 192 | state width; must satisfy `d = 4·nb` (nb spinor blocks of scalar+3-vector) |
| `num_layers` | 2 | tree+fold layers; each layer's prefix states get LM-head supervision |
| `pe_mode` | `'sin'` | `'none'` = position-is-structure (the paper configuration), `'sin'`, `'rotor'` |
| `rot_mode` | `'so3'` | per-block maps: `'so3'` (quaternion-parameterized rotations) or `'free'` (unconstrained 3×3; the paper's default after the pre-registered ablation) |
| `tie` | `False` | tie embedding and head (with the logit-scale init fix) |
| `dropout` | `0.0` | dropout on embeddings and cross-layer MLP |

### Fold variants (`fold_mode`) — how each prefix's Fenwick blocks compose

*(For the full mechanism guide — the state space, the compose node, and
every fold explained including the OAM fold — see
[`docs/OPERA_Mechanisms_Guide.md`](docs/OPERA_Mechanisms_Guide.md).)*

| fold | idea | extra flags | status in the research record |
|---|---|---|---|
| `'left'` (default) | sequential gated fold, largest block first | `fold_rotors='separate'`, `fold_scale` | the incumbent; all headline results |
| `'rack'` | each block *conjugates* the accumulator by its unit spinor (exact isometry) + gated injection | `rack_exitnorm=True` (one LayerNorm at the fold exit — the r18 variant) | isometry verified; slower convergence at fixed budget |
| `'spine'` | attention over the fold's running accumulators (the tree-native snapshot cache) | — | null at toy rung, even with msup (5th readout null) |
| `'attend'` | one attention round over each position's own ≤ log T blocks, then one compose | — | quality parity, 25% faster; reverse-falsified the bottleneck theory |
| `'revolving'` | fixed-depth gated fold, one gate per stage | `tree_drop` | ablation arm |
| `'balanced'` | pairwise reduction over the blocks | — | ablation arm |
| `'oam'` | k charged isometric channels twisted at injection | `oam_k`, `oam_charges`, `oam_phi`, `oam_shared_gate`, `oam_combine`, `oam_pair`, `oam_transport`, `oam_levelgate`, `oam_chan_emb` | charge line closed (null at maximum dose) |
| `'scan'` | replaces tree+fold with an associative scan over quaternion semidirect pairs (the SSM-style sibling) | `scan_salience`, `scan_decay_bias`, `workspace` | quality parity with left at 0.57× dispatches |

```python
# The r18 isometric fold: geometry inside, one norm at the boundary
model = OperaSpinorFenwickTree(vocab_size=10000, d=640, nb=160, num_layers=4,
                               pe_mode='none', fold_mode='rack',
                               rack_exitnorm=True, rot_mode='free')
```

### v9 training arms — train the tree as a tree

```python
from opera_lm import OperaSpinorFenwickTree, lm_loss, msup_loss, curriculum_len

# A. Chrono fold init: fold transport starts near pass-through
#    (accumulator +2.0, new block 0.0, geometric product -2.0); learnable.
model = OperaSpinorFenwickTree(..., fold_mode='left',
                               fold_gate_bias=(2.0, 0.0, -2.0))

# B. Multi-scale node supervision: every internal tree node predicts the
#    first token AFTER its span. Adds no parameters (shares the head).
#    The arm that won the toy rung (-9.87 PPL).
all_logits, levels = model(ids, lengths, return_levels=True)
loss, _, _ = lm_loss(all_logits, ids, lengths)
loss = loss + 0.1 * msup_loss(model, levels, ids, lengths)

# C. Length curriculum: T0 tokens, doubling every `every` steps, capped.
#    Deterministic in step (exact resume); eval stays full-length.
T = curriculum_len(step, cur0=64, every=2000, max_len=1024)
ids, lengths = ids[:, :T], lengths.clamp(max=T)
```

### Geometry and optimization knobs

| parameter | default | meaning |
|---|---|---|
| `norm_mode` | `'layer'` | node normalization: `'layer'`, `'rms'` (geometry-preserving global RMS + per-block gains), `'blockrms'` (SO(3)-equivariant per-block RMS) |
| `act_mode` | `'tanh'` | node activation: `'tanh'` (tanh+0.1x) or `'linear'` (equivariance-clean node) |
| `node_residual` | `False` | gated residual across the compose node |
| `lock_mode` | `'none'` | `'interference'` modulates fusion gates by children's geometric alignment (diagnostic lineage) |
| `tree_drop` | `0.0` | randomly force fold gates to identity during training (revolving fold) |
| `grad_checkpoint` | `''` | `'level'` (recompute tree nodes) or `'layer'` (whole layers) for activation memory |
| `use_metal` | `False` | fused Metal kernels on Apple Silicon (requires `rot_mode='so3'`) |

### The full research pipeline

`opera_lm.train.train()` is the exact loop the results were produced with —
GPU-resident data, AMP, torch.compile, checkpoint/resume with exact batch
streams, bucketed extrapolation eval, and the RMT spectral diagnostic:

```python
from opera_lm.train import train

train(steps=20000, batch=16, max_len=256, vocab_size=10000,
      d=640, nb=160, num_layers=4, eval_max_len=2048,
      device='cuda', pe_mode='none', fold_mode='left', rot_mode='free',
      data_mode='docs',          # 'sentences' | 'docs' | 'docs-en'
      msup=True, msup_weight=0.1,                 # arm B
      fold_gate_bias=(2.0, 0.0, -2.0),            # arm A
      curriculum=(64, 2000),                      # arm C
      optimizer='muon', muon_lr=0.02,             # T0.1: Muon on matrix
                                                  # params (rot_free's
                                                  # per-block 3x3 maps,
                                                  # cross_mlp, head),
                                                  # AdamW on the rest
      seed=42, out_dir='runs')
```

`train()` also accepts `data=(train, test_short, test_long, vocab_size)`
(pre-tokenized sequences, bypassing `load_data`) and `idx2word` —
the entry point for custom tokenizers/corpora (see `opera-chat/`).
`opera_lm.Muon` / `split_muon_params` are exported for standalone use,
and `opera_lm.OperaDecoder` gives O(log T)-per-token incremental
decoding for `fold_mode='left'` models (exact vs full forward,
selftested).

### Recommended starting points

- **The Original model:** `pe_mode='none', fold_mode='left', rot_mode='free'`
- **The v9 stack (toy-validated):** the above + `msup=True` +
  `curriculum=(64, 250)` at toy scale
- **To experiment:** every arm above composes by constructor flag; the
  self-test suite (`python -m opera_lm.selftest`) anchors equivalence,
  causality, and param budgets for all of them — extend it when you add
  your own.

## Measured results (22M params, Simple English Wikipedia, word-level 10k vocab)

Original matched protocol (shared data pipeline/loss/eval with the
transformer baseline; see `docs/opera_paper_draft_v1.md`):

| model | in-length PPL | 257–512 | 513–768 | 769–1024 |
|---|---|---|---|---|
| OPERA free + left | 41.53 | 25.64 | 15.23 | 8.12 |
| Transformer NoPE | 40.32 | 26.77 | 16.57 | 8.94 |
| Transformer RoPE | 35.68 | 23.35 | 14.86 | 8.09 |

The distinctive result is the **position curve**: evaluated at 2× training
length, OPERA's per-position loss rises **+0.07 nats** vs **+0.19** for both
RoPE and NoPE transformers, and OPERA's best band lies *beyond* its training
length. The relative gap to RoPE shrinks monotonically with evaluation
length and reaches +0.4% at 4× the training context — the axis along which
attention's cost grows quadratically and OPERA's grows O(T log T).

**Honesty box.** (1) After article-level decontamination, OPERA stands at
75.78 in-length vs RoPE's 56.77 at the 22M rung; the like-for-like
decontaminated NoPE comparison and multi-seed replication are on the
roadmap below, and numbers from the original split should be read with
that caveat. (2) All results are ~22M params, single corpus, word-level
vocab — not comparable to subword-tokenized literature PPLs. (3) OPERA is
currently ~3× slower per step than the transformer despite 20–40× fewer
FLOPs: a kernel-maturity gap, not an asymptotic one. (4) OPERA does not
beat transformers; this package exists so the mechanism, the instruments,
and the results can be examined and built on.

## The falsification record

This project pre-registers hypotheses and retires falsified components in
writing (`docs/` contains the pre-registrations):

- The SO(3) rotation-manifold constraint: nulled twice (including a
  pre-committed rematch at greater compositional depth); retired.
- The structural bottleneck theory of forgetting: falsified *in reverse* —
  one-hop access to early context halved early-context sensitivity.
  Early-context forgetting is a learned property of the objective, shared
  by OPERA, RoPE, and NoPE alike.
- Five consecutive readout-side arms (attend fold, salience, decay-bias,
  trajectory features, snapshot spine) nullified. The lever that works is
  objective-side: **multi-scale node supervision** gave −9.87 PPL at the
  toy rung — tree nodes already carry span-boundary signal unsupervised
  (aux loss 6.18 vs 9.21 chance), and paying the objective for it
  improves the main readout.

## Research record

- `docs/OPERA_Mechanisms_Guide.md` — **start here**: every arm, fold,
  geometry knob, and optimization knob explained, with its research
  status (incumbent / validated / falsified / exploratory)
- `docs/opera_paper_draft_v1.md` — the draft (v1)
- `docs/opera_paper_draft_v1_1.md` — **current draft**: v1.1 adds exact
  incremental decoding (§2.5/§4.5), the Muon recipe arm (§4.6), the BPE
  chat artifact (§4.7), and log-linear-attention related work (§6)
- `docs/OPERA_Technical_Reference.md` — the mathematical foundation
  (operads, SO(3), Schur's lemma fusion)
- `docs/OPERA_Resonance_Framework.md` — phase-locking formulation
- `docs/OPERA_Scan_Arm_Design.md` — the associative-scan arm
- `docs/OPERA_v9_tree_prereg.md` — the tree-training line (with outcomes)
- `docs/OPERA_v8.10_timeline_prereg.md` — the timeline arms (falsified)

## Roadmap

- Decontaminated-split re-measurement of the headline comparisons
  (OPERA vs NoPE), with position curves
- Multi-seed replication of the main results
- `msup` + curriculum at the 22M rung (toy-rung win: −9.87 PPL)
- Fused composition kernels (CUDA/Triton); crossover timing vs attention

## Citation

```bibtex
@software{hafez2026opera,
  author = {Hafez, Merna},
  title = {OPERA-LM: Spinor Fenwick-Tree Language Modeling with No
           Positional Encodings},
  year = {2026},
  url = {https://github.com/Merna-Khalid/OPERA-LM}
}
```

## License

MIT — see `LICENSE`.
