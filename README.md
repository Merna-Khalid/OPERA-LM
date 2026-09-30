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

**No attention, by design.** OPERA uses no softmax attention, no linear
attention, and no attention over layers. In-context recall comes instead
from a **gated quaternion holographic memory**:
- **Write:** each position binds the key of the previous token to the
  value of the current one with the Hamilton product, and adds the result
  to a per-slot superposition. A learned, data-dependent forget gate
  controls how fast each slot decays.
- **Read:** unbinding with the conjugate of the current key retrieves the
  value that followed it.

The memory is a fixed-size vector state: O(1) per token, with no cache
that grows with length. It took the model from chance to 97–100% on
associative recall (MQAR) and improves byte-level language modeling
(see [Current results](#current-results-byte-level-simple-english-wikipedia)).

```python
import torch
from opera_lm import OperaSpinorFenwickTree

# The current model: byte-level (259 ids), stream head, gated memory
model = OperaSpinorFenwickTree(vocab_size=259, d=512, nb=128, num_layers=2,
                               pe_mode='none', fold_mode='left',
                               rot_mode='free', head_mode='stream',
                               fold_impl='downsweep',
                               hmem_nb=128, hmem_decay='gated')
ids = torch.randint(3, 259, (2, 1024))
lengths = torch.tensor([1024, 700])
logits = model(ids, lengths).logits[-1] # final-layer logits [2, 1024, 259]
```

Add `use_metal=True` on Apple Silicon or `use_triton=True` on CUDA for the
fused kernels. The math is identical (selftested).

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
- **`opera_lm.triton_kernel`** — the same compose-node kernel ported to
  Triton for CUDA (`use_triton=True`); `python -m opera_lm.triton_kernel
  --test` checks it against the reference on your GPU.
- **`opera_lm.hmem_kernel`** — fused Metal kernels for the holographic
  memory: the whole read (key normalization, gates, bind, decayed
  superposition, unbind) in one kernel with an analytic backward, plus
  the causal short convolution. PyTorch reference path everywhere else.
- **`opera_lm.packed`** — memory-mapped sequence pools
  (`PackedBatchSource`) for corpora too large to hold in RAM, with exact
  batch-stream resume.
- **`opera_lm.reprs`** — byte / BPE corpus builders and the BPB
  conversion.

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
| `'spine'` | attention over the fold's running accumulators (the tree-native snapshot cache) | — | **retired** — uses attention; null at toy rung (5th readout null); kept only for the record |
| `'attend'` | one attention round over each position's own ≤ log T blocks, then one compose | — | **retired** — uses attention; kept only for the record (it reverse-falsified the bottleneck theory) |
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

### Holographic memory (in-context recall, attention-free)

Full design, derivation and results:
[`docs/OPERA_Recall_Research_2026-09-24.md`](docs/OPERA_Recall_Research_2026-09-24.md).

| parameter | default | meaning |
|---|---|---|
| `hmem_nb` | `0` | `n>0`: n quaternion memory slots per layer (use `d/4`). Write = unit key(t−1) ⊗ value(t) × write gate; read = conj(key(t)) ⊗ memory → LayerNorm → projection → zero-init gain (exactly the base model at init) |
| `hmem_decay` | `'none'` | `'gated'`: a data-dependent forget gate per slot (the recommended setting — it fixes the long-range droop of the plain sum); `'fixed'`: learned per-slot decay, content-independent |
| `hmem_conv` | `0` | `K>0`: causal depthwise short convolution (zero-init taps) over the memory's projections; null at the current scale |

Memory cost after kernel fusion: 1.24× the base model's training step at
d512 (it was 1.95× before the fused kernels).

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
out = model(ids, lengths, return_levels=True)
loss, _, _ = lm_loss(out.logits, ids, lengths)
loss = loss + 0.1 * msup_loss(model, out.levels, ids, lengths)

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
| `use_metal` | `False` | fused Metal compose kernel on Apple Silicon (works with `rot_mode='free'` too); ~2× faster training steps, same model (selftested) |
| `use_triton` | `False` | the same compose kernel in Triton for CUDA; correctness-verified on a T4, speed not yet measured |
| `fold_impl` | `'compact'` | `'downsweep'`: the same left fold for every prefix, computed top-down over the tree so prefixes share their composes (~T per layer instead of ~(T/2) log T; `fold_mode='left'` only; equal up to float reassociation) |

### Stream, tree-width and optimizer knobs (exploratory; `docs/OPERA_Literature_Scan_2026-09-24.md`)

| parameter | default | meaning |
|---|---|---|
| `head_mode` | `'fold'` | `'stream'`: the head reads `LN(current)` after each layer's update. With `'fold'` the last layer's `cross_mlp`/`blend_gate` receive no gradient (34% of a d512/L2 model) |
| `resid_mode` | `'blend'` | `'add'`: additive pre-norm residual instead of the convex scalar blend (needs `head_mode='stream'`) |
| `fold_gate` | `'shared'` | `'separate'`: the left fold gets its own fusion gate, copy-initialized (bitwise incumbent at init) |
| `fold_h0` | `False` | every prefix folds from a learned per-layer start state (no raw-tree-node readouts) |
| `state_mult` | `1` | tree/fold width = `state_mult·d` behind per-layer in/out projections (left fold) |
| `state_tie` | `False` | with `state_mult=k`: the k copies of each slot share its gates and rotors (state grows k×, gate output width unchanged) |
| `node_mix_rank` | `0` | `r>0`: identity-initialized rank-r cross-slot value mixing inside the compose node (bitwise incumbent at init) |
| `aux_weight` (in `train()`) | `0.5` | loss weight of each non-final layer's head |
| `samuon_gamma` (in `train()`) | `1.0` | SAMuon-lite tail boost on Muon's update (arXiv 2608.25990); `1.0` is plain Muon, byte-identical; cosine warmup over `samuon_warmup_frac` (0.3) of training |

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

- **The current model (byte-level, with memory):** the quickstart above —
  `vocab_size=259, pe_mode='none', fold_mode='left', rot_mode='free',
  head_mode='stream', fold_impl='downsweep', hmem_nb=d/4,
  hmem_decay='gated'`, trained with Muon (`optimizer='muon',
  muon_lr=0.02, muon_include='fusion_gate', muon_wd=0.01`)
- **The Original model:** `pe_mode='none', fold_mode='left', rot_mode='free'`
- **The v9 stack (toy-validated):** the above + `msup=True` +
  `curriculum=(64, 250)` at toy scale
- **To experiment:** every arm above composes by constructor flag; the
  self-test suite (`python -m opera_lm.selftest`) anchors equivalence,
  causality, and param budgets for all of them — extend it when you add
  your own.

## Current results (byte-level, Simple English Wikipedia)

Raw UTF-8 bytes (no tokenizer), 1,024-byte training context, bits per
byte (BPB, lower is better). Full record, including the null results:
[`docs/OPERA_Recall_Research_2026-09-24.md`](docs/OPERA_Recall_Research_2026-09-24.md).

**Recall (MQAR).** N key→value pairs, each re-queried later in a 256-token
sequence; accuracy at the query positions. d256, 2 layers.

| model | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| OPERA, fold only | 0.002 | 0.003 | 0.002 | 0.002 |
| **+ holographic memory, 256 slots** | **1.000** | **1.000** | **1.000** | **0.973** |
| + holographic memory, 64 slots | 0.998 | 0.988 | 0.789 | 0.399 |

- Without the memory, OPERA is at chance (1/512) even a few tokens back.
  A local-copy control on the same pipeline reaches 100%, so the
  failure is specific to key→value binding.
- With the memory, recall is flat across distance.
- Capacity follows holographic-memory theory: fewer slots, more
  interference.

**Language modeling.** 12,000 steps × 8,192 bytes per step. Two test
sets:
- **a20k:** the original test set, from the first 20,000 articles.
- **full:** a seeded sample from every test article in the corpus.

| model | params | trained on | BPB a20k | BPB full |
|---|---|---|---|---|
| OPERA, no memory | 3.3M | 20k-article slice | 1.8179 | 1.9309 |
| + gated memory | 5.0M | 20k-article slice | 1.7043 | 1.8389 |
| + gated memory | 5.0M | full corpus (217M bytes) | 1.7799 | 1.6203 |
| **+ gated memory, d768** | **11.0M** | full corpus | **1.6572** | **1.4918** |

- **Memory:** −6.2% BPB at equal training (a20k). The gain grows with
  position in the document, from −1.8% on bytes 0–16 to −6.4% past
  byte 512. That pattern means better use of context, not seed noise.
- **Width at fixed data:** −6.9% to −11% BPB on every test set.
- **Length:** at 2× the training length (1,025–2,048 bytes) the loss
  does not rise (PPL 0.99× in-length for the d768 model), with no
  positional encoding.
- **More data:** the full-corpus model is worse on the a20k test set but
  much better on the full one, because the first 20,000 articles are
  longer and unrepresentative. Both are reported for that reason.

**Honesty box.**
1. **Scale is small** (≤11M params, one corpus). The scale runs are
   single-seed; memory models vary ~3% BPB across seeds, so effects
   under ~3% need more seeds.
2. **No transformer baseline in this line, by design** (no attention);
   comparisons are within OPERA.
3. **Held-out text only:** test articles never enter training. About 4%
   of the full test set is templated stubs or tables, which flatters the
   full-corpus model a little.

**Reproduce** (Apple Silicon; add `--device cuda` elsewhere):

```bash
python experiments/mqar.py --hmem-nb 256 --tag hmem256          # recall
python experiments/build_reprs.py --max-articles 20000 --modes bytes
python experiments/build_packed.py                              # full-corpus pool + test set
python experiments/repr_study.py --arms b2m_hmem128_gated_long --steps 12000
python experiments/repr_study.py --arms w768_hmem192_gated_full --steps 12000 \
    --packed assets/packed_bytes_T1024_all
python experiments/eval_pool.py --arms w768_hmem192_gated_full  # every test set
```

**Scaling next:** [`colab/OPERA_Scale.ipynb`](colab/OPERA_Scale.ipynb)
runs a compute-optimal ladder on FineWeb-Edu bytes on a Colab A100.
The notebook covers GPU correctness checks, data build, throughput
benchmark, and resumable training.

## Earlier results — 22M params, word-level 10k vocab (the original matched protocol)

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
- The ~100-byte context plateau was a **missing function, not a missing
  signal**: on MQAR the fold alone is at chance even a few tokens back,
  while a local-copy control solves it. That diagnosis led to the
  holographic memory.
- A causal short convolution in front of the memory (the H3 / Mamba /
  Canon-layer idea) was **null** at this scale: −2.1% on one seed,
  +3.2% on another.

## Research record

- `docs/OPERA_Mechanisms_Guide.md` — **start here**: every arm, fold,
  geometry knob, and optimization knob explained, with its research
  status (incumbent / validated / falsified / exploratory)
- `docs/OPERA_Recall_Research_2026-09-24.md` — **the current line**:
  the recall diagnosis (MQAR), the quaternion holographic memory, its
  gated decay, the fused kernels, the long run and the scale-up, with
  every result and null
- `docs/OPERA_Speed_and_Design_2026-09-24.md` — kernel and fold speed
  work (downsweep fold, fused Metal kernels, width vs depth cost)
- `docs/OPERA_Literature_Scan_2026-09-24.md`,
  `docs/OPERA_Redundancy_Research_2026-09-24.md`,
  `docs/OPERA_Readout_survey.md` — the surveys behind the stream head,
  the memory and the null arms
- `docs/OPERA_Paper_Draft_v1.3.md` — **current paper draft** (the 22M
  word-level results; v1.3 expands the architecture section);
  `docs/opera_paper_draft_v1.md` … `v1_2.md` — earlier drafts
- `docs/OPERA_Technical_Reference.md` — the mathematical foundation
  (operads, SO(3), Schur's lemma fusion)
- `docs/OPERA_Resonance_Framework.md` — phase-locking formulation
- `docs/OPERA_Scan_Arm_Design.md` — the associative-scan arm
- `docs/OPERA_v9_tree_prereg.md` — the tree-training line (with outcomes)
- `docs/OPERA_v8.10_timeline_prereg.md` — the timeline arms (falsified)

## Roadmap

- **Compute-optimal scaling ladder** on FineWeb-Edu bytes (4–5 sizes,
  ~10M–150M params, Colab A100): fit loss against compute and compare
  the slope with published byte-level and attention-free models
- Depth > 2 at scale (additive residual with 1/√(2L) init; untested so
  far)
- A Triton kernel for the holographic memory (it runs as the PyTorch
  reference under `torch.compile` on CUDA today)
- Multi-seed replication of the scale results
- Decontaminated-split re-measurement of the 22M word-level comparisons,
  with position curves
- `msup` + curriculum at the 22M rung (toy-rung win: −9.87 PPL)

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
