# opera-chat

BPE-tokenized chat pipeline for OPERA-LM: tokenizer training, data prep,
training (smoltalk chat, and FineWeb-Edu raw-text pretraining), sampling,
local Gradio checks, and Hugging Face Spaces deployment. Everything runs
against the `opera-lm/` library via `sys.path` (no install needed).

**Current recommended path: `OPERA_Colab_Demo.ipynb` on a Colab GPU**
(A100 tested). It runs the full pipeline below end-to-end — data prep,
~155M-parameter training (both OPERA and a matched transformer baseline),
the raw-text-pretrain → smoltalk-SFT two-stage variant, local sanity
checks, and publishing to a Hugging Face Space. The scripts also run
locally (see [Local / CPU-MPS pipeline](#local--cpu-mps-pipeline) below
and `docs/OPERA_Paper_Draft_v1.3.md` §4.7 for the original ~20M pilot),
but the measured results below are all from the Colab/CUDA runs.

## The 155M-parameter Colab study (current)

Full writeup: `docs/OPERA_Paper_Draft_v1.3.md` §4.8. Summary:

**Setup.** OPERA (d=1664, nb=416, 8 layers, 154,815,249 params) vs a
matched RoPE transformer (d=1264, 8 heads, 7 layers, 155,049,777 params,
+0.15%) — same data, same Muon(lr 0.02)/AdamW partition, same length
curriculum, seed 42, `torch.compile` + bf16 on a single A100-SXM4-40GB.
16,384-token BPE vocab (`tokenizer.json`, byte-level, specials at ids
0-4 — see [Chat token format](#chat-token-format)).

**1. Chat-only, from scratch** — full smoltalk `all` split (1,043,917
conversations, 223,987 train sequences ≤256 tokens), 20k steps.

| model | in-length PPL (5k) | 257-512 | 513-768 | 1793-2048 | s/step |
|---|---|---|---|---|---|
| OPERA free+left | **11.78** | **25.48** (2.16×) | **79.32** (6.73×) | **162.19** (13.77×) | 0.65 |
| Transformer RoPE | 18.60 | 39.49 (2.12×) | 129.82 (6.98×) | 323.87 (17.41×) | 0.08 |

OPERA wins in-length by 36.7% relative and every extrapolation bucket
(up to 50% relative at the farthest), despite training 8.1× slower per
step.

**2. Raw-text pretraining, from scratch** — FineWeb-Edu `sample-10BT`,
capped at 500M streamed tokens (`prepare_fineweb.py`), 20k steps (≈82M
tokens trained, ≈4 epochs over the 19.9M-token train pool — a single-
session budget, not full-corpus single-epoch pretraining).

| model | in-length PPL (5k) | extrapolation ratio range |
|---|---|---|
| OPERA free+left | 105.89 | 1.03×-1.09× (flat) |
| Transformer RoPE | **83.55** | 1.00×-1.07× (also flat) |

Here the transformer wins (21% lower PPL). Both architectures show
near-flat length-extrapolation on this corpus — **this falsifies the
hypothesis that OPERA's flatness on raw text is architecture-specific**;
it's a property of FineWeb-Edu's single-topic documents, observed under
both architectures. See the paper §5.5 for this recorded as a nulled
pre-registered-in-spirit hypothesis.

**3. Two-stage: FineWeb pretrain → smoltalk SFT** — each architecture's
own run-2 checkpoint continued for 10k more steps on the same full
smoltalk data via `--init-weights-from` (weight-only load, independent
of `--resume`).

| model | in-length PPL (5k) | 257-512 | 1793-2048 |
|---|---|---|---|
| OPERA free+left | 13.04 | 24.83 (1.90×) | 136.97 (10.51×) |
| Transformer RoPE | **11.24** | **19.67** (1.75×) | **106.09** (9.44×) |

The transformer wins here too, reversing run 1's ranking — pretraining
helps both architectures converge faster on the eventual chat objective,
but the transformer's raw-text head start is larger and flips the final
result. **Takeaway: OPERA's demonstrated advantage at this scale is
specific to end-to-end training on structured, multi-turn dialogue, not
general-purpose.** See the paper §4.8's synthesis paragraph and §7's
"data-regime dependence" limitation for the honest read on this.

**Compute:** ≈7.9 GPU-hours total across the six main runs above on a
single A100; the largest single run (OPERA chat-only) is ≈3.6 GPU-hours.

## Pipeline (Colab / CUDA)

```bash
# from the repo root, inside OPERA_Colab_Demo.ipynb (or equivalent shell):

# 1. data
python opera-chat/prepare_data.py --tokenizer opera-chat/tokenizer.json \
    --out data_chat_full.pkl --n-convs 5000000 --max-len 256 --eval-max-len 2048
python opera-chat/prepare_fineweb.py --tokenizer opera-chat/tokenizer.json \
    --out data_fineweb.pkl --max-tokens 500000000   # Stage 2 only

# 2. train OPERA (~155M: d=1664 nb=416 L=8, pe none, fold left, rot free)
python opera-chat/train_chat.py --data data_chat_full.pkl --steps 20000 \
    --batch 16 --max-len 256 --eval-max-len 2048 --d 1664 --nb 416 \
    --num-layers 8 --device cuda --optimizer muon --muon-lr 0.02 \
    --compile default --out-dir runs_chat_full

# 2b. Stage 2: pretrain on FineWeb, then continue into smoltalk SFT
python opera-chat/train_chat.py --data data_fineweb.pkl --steps 20000 \
    ... --out-dir runs_fineweb
python opera-chat/train_chat.py --data data_chat_full.pkl --steps 10000 \
    --init-weights-from <runs_fineweb checkpoint>.pt --out-dir runs_fineweb_smoltalk_sft

# 3. matched transformer baseline (same protocol, --init-weights-from also supported)
python opera-chat/train_tf_chat.py --pe rope --data data_chat_full.pkl \
    --steps 20000 --batch 16 --max-len 256 --eval-max-len 2048 \
    --d 1264 --num-layers 7 --device cuda --optimizer muon --muon-lr 0.02 \
    --out-dir runs_chat_tf

# 4. sample / local Gradio check (see gotchas below)
python opera-chat/generate_chat.py --ckpt runs_chat_full/opera_v8_0_....pt \
    --config runs_chat_full/model_config.json --prompt "hello there"

# 5. deploy (needs --token or HF_TOKEN)
python opera-chat/push_to_hub.py --ckpt runs_chat_full/opera_v8_0_....pt \
    --config runs_chat_full/model_config.json \
    --model-repo USER/opera-lm-chat --space-repo USER/opera-lm-chat-space
```

### Local Gradio check gotchas

`app.py` requires `tokenizer.json`, `model_config.json`, and the
checkpoint all inside `$MODEL_DIR` — but `train_chat.py`/`train_tf_chat.py`
only write the checkpoint + `model_config.json` there, not the tokenizer.
Copy it in first (use an **absolute** path — `demo.launch(share=True)`
blocks, so if you interrupt it your notebook's CWD is left wherever the
launch cell last `%cd`'d to, not necessarily the repo root):

```python
import shutil, os
REPO_ROOT = '/content/repo'   # or wherever the repo was cloned
shutil.copy(f'{REPO_ROOT}/opera-chat/tokenizer.json', f'{RUN_DIR}/tokenizer.json')
os.environ['MODEL_DIR'] = RUN_DIR
# optional: override the demo's title/description per checkpoint
os.environ['MODEL_DESC'] = "..."
```

`app.py` reads `MODEL_TITLE` / `MODEL_DESC` env vars (falling back to a
default description) so the same script can serve differently-described
checkpoints (e.g. the chat-only vs the two-stage model) without editing
the file. Pass them explicitly on the `!python -c "..."` command line
too (`MODEL_DIR=$RUN_DIR MODEL_DESC="$MODEL_DESC" python -c ...`) — a
notebook `os.environ[...]` assignment doesn't automatically propagate
into the subshell a `!`-prefixed command spawns.

### CUDA-specific fixes baked into the training loop

Both required to train reliably at this scale on a single Colab GPU;
covered by `opera_lm.selftest`'s CUDA-gated assertions:

- **OOM-retry backstop** (`opera_lm.train._oom_backstop`): halves the
  eval batch size and retries on a CUDA OOM. Wired into every perplexity
  call site (previously only `extrapolation_eval` had it — an unguarded
  periodic-eval call crashed a full 150M-scale run mid-schedule before
  this fix).
- **`torch.compile` dynamo-guard fix**: the compose node's per-call
  dense-rotation cache was keyed by Python `id()`, which dynamo can't
  guard on — this produced a recompilation storm (~210s/step, vs the
  ~0.4-0.65s/step reported above) until fixed to detect
  `torch.compiler.is_compiling()` and fall back to the compile-safe
  `einsum` path.

## T0.1 gate result (2026-07-29): Muon adopted at lr 0.02

Pre-registered toy-rung gate (`muon_gate.py`, runs in `runs_muon_gate/`):
2000 steps, seed 1, d=256/nb=64/L=2, Simple Wiki docs, identical
data order across arms. Final in-length PPL (5k test):

| arm | PPL | delta |
|---|---|---|
| AdamW (incumbent) | 224.88 | — |
| Muon lr=0.02 | **186.45** | **+38.43** |
| Muon lr=0.05 | 221.99 | +2.89 |
| Muon lr=0.10 | 302.72 | −77.84 (diverging) |

Gate: PASS (>1.5 PPL) → Muon lr=0.02 is the default in `run_full.sh` and
every Colab run above. Caveats: single seed, toy rung, word-level
protocol; the 0.05/0.10 arms bracket 0.02 near the top of the stable
range, so 0.02 transfers to the BPE chat run — and later the 155M Colab
runs — as the *starting* lr, not a proven optimum (it was reused
unmodified at 155M without independently re-gating it at that scale;
see the paper §4.6 and §7's "recipe asymmetry" limitation).

## Chat token format

Special tokens, fixed ids: `<|pad|>`=0 (the library zero-pads batches),
`<|user|>`=1, `<|assistant|>`=2, `<|end|>`=3, `<|system|>`=4. Per message:
role token + BPE(text.strip()); assistant messages get a trailing
`<|end|>`. A generation prompt is the formatted history plus a bare
`<|assistant|>`; generation stops at `<|end|>`; `<|pad|>` is never
emitted. See `chat_common.py`. (`format_conversation` accepts `content`
as either a plain string or a list of parts — some `gradio>=6`
`ChatInterface` versions hand message content through as parts even
outside `multimodal=True`.)

## Deploy artifacts

`push_to_hub.py` uploads a model repo with:

- `model_config.json` — vocab_size, d, nb, num_layers, tie, pe_mode,
  fold_mode, rot_mode, fold_gate_bias, and `ckpt` (final .pt path)
- the final checkpoint (raw `model.state_dict()`)
- `tokenizer.json`

and a Space repo with `app.py`, `generate_chat.py`, `chat_common.py`,
`requirements.txt`, `README.md` (from `space_README.md`), and a
self-contained `opera_lm/` package copy. The uploaded `app.py` has its
`HF_REPO` default patched to the model repo id, so the Space needs no
configuration. Both the model repo and a CPU Basic Space are free on
Hugging Face regardless of model size; only an explicit GPU/always-on
hardware upgrade is billed, and nothing here requires one — the whole
point of the incremental `OperaDecoder` is fast-enough CPU serving.

## Honesty box: what to expect from the chat demo

~150M parameters trained on ~82M tokens total is roughly five orders of
magnitude less data than a real small chat model (SmolLM2-135M: ~11T
tokens). Expect grammatically fluent, often semantically incoherent
replies — the model has learned local sentence statistics, not world
knowledge. The lower perplexity vs. the matched transformer (table above)
is a real, controlled statement about which architecture uses the same
tiny data budget better; it is not a capability claim. See
`docs/OPERA_Paper_Draft_v1.3.md` for the full, unhedged results and
limitations.

## Local / CPU-MPS pipeline

The scripts run locally too (no Colab/CUDA required) for smoke tests or
small reruns — same commands as above with `--device mps` (Apple
Silicon) or `--device cpu`, smaller `--steps`/`--batch`/`--d`/`--nb`.
`bash opera-chat/run_full.sh [--device cuda] [--steps N]` runs the
tokenizer → data → train → sample sequence in one shot (interrupt-safe;
rerun `train_chat.py` with `--resume` to continue).

### History: the original ~20M / 150k-subset pilot (superseded)

The first working chat run (2026-07-31, Apple M4, 150k-conversation
smoltalk subset, `d=640 nb=160 L=4`, ~20M params, no matched-transformer
baseline at first) reached in-length PPL 29.65 (5k, BPE). Extrapolation
buckets 2.3×/5.1×/5.8× — the long-conversation tail is hard; not the
paper's position-curve regime. Sample behavior: short factual prompts
often landed ("capital of France" → "Paris, ..." before looping);
jokes/multi-turn were grammatical nonsense — the research-toy quality
the honesty box above still promises, just at a higher param/data
budget now.

A same-protocol matched-transformer comparison followed (2026-08-04,
`train_tf_chat.py` / `run_tf_baselines.sh`, tied 21.0M vs OPERA 19.9M,
same data/steps/schedule/curriculum/seed/Muon recipe):

| model | in-length PPL | 257-512 | 513-768 | 769-1024 |
|---|---|---|---|---|
| TF RoPE | **16.96** | 40.9 (2.41×) | 90.2 (5.32×) | 102.3 (6.03×) |
| TF NoPE | 24.33 | 72.6 (2.98×) | 188.7 (7.76×) | 239.8 (9.86×) |
| OPERA chat | 29.65 | 68.3 (2.31×) | 151.6 (5.11×) | 171.9 (5.80×) |

Position curves (500 held-out conversations, 512-1024 tokens):
boundary-crossing delta RoPE +0.031 / NoPE +0.335 / OPERA +0.055 nats;
best→final-2×-band RoPE +0.176 / NoPE +0.688 / OPERA +0.259. Reading:
(1) in-length, transformers won this protocol by more than in the
word-level regime; (2) NoPE's emergent-position decay reproduced here
(monotone rise, +0.688) and OPERA avoided it (flat past the boundary
hump); (3) OPERA had the best extrapolation ratio at every bucket and
beat NoPE in absolute PPL beyond ~768 tokens (171.9 vs 239.8) despite
losing in-length; (4) RoPE was barely stressed at only 4× training
length on short-range chat turns. Single seed.

This pilot is documented in the paper as §4.7 and superseded by the
155M/full-split Colab study above (§4.8) — an order of magnitude more
data, a proper matched-parameter transformer baseline from the start,
and (new) the raw-text-pretrain comparison. Kept here for the record,
not deleted, per this project's practice of not erasing superseded
results. The T0.1 Muon gate above is unchanged and carries through
unmodified to both pilots and the Colab study.
