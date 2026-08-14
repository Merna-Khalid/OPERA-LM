# opera-chat

BPE-tokenized chat pipeline for OPERA-LM: tokenizer training, data prep,
training, sampling, and Hugging Face Spaces deployment. Everything runs
against the `opera-lm/` library via `sys.path` (no install needed).

Env: `/opt/anaconda3/envs/CWUW/bin/python` (torch 2.10 + MPS, tokenizers,
datasets, gradio 6 for local app testing).

**One command, full pipeline:** `bash opera-chat/run_full.sh`
(optional `--device cuda`, `--steps N`; ~8-14h for 20k steps on an M4,
interrupt-safe — rerun `train_chat.py` with `--resume` to continue).

## Pipeline

```bash
cd opera-chat

# 1. tokenizer (byte-level BPE, specials at ids 0..4)
python train_tokenizer.py --smoke                 # 30k lines, quick check
python train_tokenizer.py --out tokenizer.json    # full: 500k lines (2M
                                                  # segfaults the Rust
                                                  # trainer on 16GB)

# 2. data (smoltalk conversations -> token-id pools)
python prepare_data.py --smoke                    # 2k convs, streaming
python prepare_data.py --tokenizer tokenizer.json --out data_chat.pkl

# 3. train (~20M params: d=640 nb=160 L=4, pe none, fold left, rot free)
python train_chat.py --data data_chat.pkl --steps 200     # smoke
python train_chat.py --data data_chat.pkl --device mps --steps 20000 \
    --optimizer muon        # roadmap T0.1 (Muon on matrix params; default
                            # in run_full.sh; --muon-lr 0.02, sweep
                            # {0.02, 0.05, 0.1} per the roadmap gate)

# 4. sample
python generate_chat.py --ckpt runs_chat/opera_v8_0_....pt \
    --config runs_chat/model_config.json --prompt "hello there"
python generate_chat.py --selftest                # no files needed

# 5. deploy (needs --token or HF_TOKEN)
python push_to_hub.py --ckpt runs_chat/opera_v8_0_....pt \
    --config runs_chat/model_config.json \
    --model-repo USER/opera-lm-chat --space-repo USER/opera-lm-chat-space
```

Measured local runtimes (M4, smoke rung): tokenizer (30k lines) ~25s,
data prep (2k convs) ~60s, training ~0.7s/step at curriculum T=64 rising
toward ~1.5-2.5s/step at T=256 → budget ~8-14h for 20k steps.
Generation (incremental OperaDecoder): ~315 tok/s on CPU at ctx 256-512,
flat in context length; ~26 tok/s with `--naive` full re-forward.

## Chat run result (2026-07-31): the deployable model

`runs_chat/`: 150k smoltalk conversations (40,877 train seqs), 20k steps
with Muon lr 0.02 + msup + chrono init + curriculum (~8.6h on M4, with
two exact-stream resumes mid-run). **In-length test PPL 29.65** (5k,
BPE). Extrapolation buckets 2.3×/5.1×/5.8× — the long-conversation tail
is hard; not the paper's position-curve regime. Sample behavior: short
factual prompts often land ("capital of France" → "Paris, ..." before
looping); jokes/multi-turn are grammatical nonsense. Exactly the
research-toy quality the Space's honesty box promises.

Position curve on the chat checkpoint (`position_curve.py`, 500 held-out
conversations 512-1024 tokens, eval-only): +0.055 nats across the
training-length boundary (0-255 vs 256-511 mean CE), curve bends back
DOWN past position 320, and at 3-4x training length CE is at/below the
in-length mean — the paper's flatness signature, on BPE chat data. The
paper-style best->final-2x-band metric reads +0.259 nats, but the best
band (0-63) is flatteringly easy chat openings; report the boundary
crossing, not the headline metric. Caveats: single model, no matched
transformer on this protocol, long-form high-entropy population
(CE ~5.1 != the 29.65 short-chat test PPL), shrinking band counts
beyond 512.

## Matched transformer baselines (2026-08-04): the v1.2 comparison

`train_tf_chat.py` / `run_tf_baselines.sh`: the paper's matched
TransformerBaseline (tied, 21.0M vs OPERA 19.9M) on the SAME data_chat
protocol — same data, steps (20k), batch, schedule, curriculum, seed,
Muon recipe. Position curves: `pc_rope.log`, `pc_nope.log`,
`position_curve.log` (OPERA). In-length PPL (5k) and extrapolation:

| model | in-length PPL | 257-512 | 513-768 | 769-1024 |
|---|---|---|---|---|
| TF RoPE | **16.96** | 40.9 (2.41x) | 90.2 (5.32x) | 102.3 (6.03x) |
| TF NoPE | 24.33 | 72.6 (2.98x) | 188.7 (7.76x) | 239.8 (9.86x) |
| OPERA chat | 29.65 | 68.3 (2.31x) | 151.6 (5.11x) | 171.9 (5.80x) |

Position curves (same 500 conversations): boundary-crossing delta
RoPE +0.031 / NoPE +0.335 / OPERA +0.055 nats; best->final-2x-band
RoPE +0.176 / NoPE +0.688 / OPERA +0.259. Reading:
(1) in-length, transformers win this protocol by more than in the
word-level regime — attention's context exploitation matters more on
chat data; (2) NoPE's emergent-position decay reproduces here
(monotone rise, +0.688) and OPERA avoids it (flat past the boundary
hump) — the paper's asset, replicated on a subword protocol;
(3) OPERA has the best extrapolation RATIO at every bucket and BEATS
NoPE in absolute PPL beyond ~768 tokens (171.9 vs 239.8) despite
losing in-length; (4) RoPE is barely stressed at only 4x training
length on short-range chat turns — the honest limit of this dataset
as a positional stress test. Single seed; v1.2 material.

Cloud note: pass `--device cuda` to `train_chat.py`. To continue an
interrupted run, copy the `*_train_ckpt.pt` alongside and rerun with
`--resume` (exact batch-stream resume).

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

Gate: PASS (>1.5 PPL) → Muon lr=0.02 is the default in `run_full.sh`.
Caveats: single seed, toy rung, word-level protocol; the 0.05/0.10 arms
bracket 0.02 near the top of the stable range, so 0.02 transfers to the
BPE chat run as the *starting* lr, not a proven optimum.

## Chat token format

Special tokens, fixed ids: `<|pad|>`=0 (the library zero-pads batches),
`<|user|>`=1, `<|assistant|>`=2, `<|end|>`=3, `<|system|>`=4. Per message:
role token + BPE(text.strip()); assistant messages get a trailing
`<|end|>`. A generation prompt is the formatted history plus a bare
`<|assistant|>`; generation stops at `<|end|>`; `<|pad|>` is never
emitted. See `chat_common.py`.

## Deploy artifacts

`push_to_hub.py` uploads a model repo with:

- `model_config.json` — vocab_size, d, nb, num_layers, tie, pe_mode,
  fold_mode, rot_mode, fold_gate_bias, and `ckpt` (final .pt path)
- the final checkpoint (raw `model.state_dict()`)
- `tokenizer.json`

and a Space repo with `app.py`, `generate_chat.py`, `chat_common.py`,
`requirements.txt`, `README.md`, and a self-contained `opera_lm/`
package copy. The uploaded `app.py` has its `HF_REPO` default patched to
the model repo id, so the Space needs no configuration.
