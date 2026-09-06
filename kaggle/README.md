# Path A on Kaggle — runbook

The two-stage pretrained long-context study: OPERA vs TF-RoPE vs
TF-NoPE at 155M params, FineWeb-Edu pretraining (~246M tokens) →
smoltalk SFT, trained at T=512, evaluated to 8192. Full protocol and
pre-registered gates: `docs/OPERA_PathA_prereg.md`. Everything runs on
Kaggle's free tier (2×T4); cost $0, calendar ~3–4 weeks of weekly
quota, ~6–9 sessions.

## One-time setup (~10 min, Kaggle web UI)

1. **Create the notebook**: Kaggle → Code → New Notebook → File →
   Import Notebook → upload `kaggle/OPERA_PathA.ipynb` from this repo.
2. **Settings → Accelerator:** start with **none (CPU)** for the first
   (data-build) session — it's free and uses no GPU quota — then switch
   to **GPU T4 x2** for every session after. (P100 is confirmed broken —
   Kaggle's torch wheel has no sm_60 kernels; the engine also checks.)
   Switching accelerator restarts the session; that's expected and safe
   (see Session mechanics below).
3. **Settings → Internet → On.**
4. **Add-ons → Secrets** → add two secrets:
   - `KAGGLE_USERNAME` — your kaggle username
   - `KAGGLE_KEY` — from kaggle.com → Settings → API →
     "Create New Token" (the `key` field of the downloaded kaggle.json)
5. **Pin the commit**: in the notebook's second cell, replace
   `PATHA_COMMIT` with the commit hash you pushed this kit under (or
   leave `main`).

## Session flow (each one: open notebook → **Save & Run All** → walk away)

**Session mechanics to know:** switching the accelerator (CPU ↔ T4 x2)
restarts the session and wipes `/kaggle/working`. Only pushed datasets
and saved notebook versions survive. The engine pushes after **every
artifact and stage**, so a killed session loses at most the artifact in
flight; **Save & Run All** (not interactive Run All) additionally
persists `/kaggle/working` as a notebook version. Attached inputs (Add
Input) are notebook configuration — they persist across restarts and
remount each session.

The engine (`kaggle/patha_session.py`) auto-picks the next stage, trains
inside a wall-clock governor (default 10.5h + 30min hard margin, under
Kaggle's ~12h cap), checkpoints every 1000 steps, and pushes:

- `opera-lm-patha-ckpt` (private dataset): runs/, state.json, logs/,
  artifacts/
- `opera-lm-patha-data` (private dataset): tokenizer(s), pools, packed
  npy pairs, bucket report — written once by the data stage

**After the first data-session push** (and after any later session):
edit the notebook → Add Input → select your `opera-lm-patha-data` and
`opera-lm-patha-ckpt` datasets, so the next session resumes from the
pushed checkpoints and reads the pools from the read-only mount
(memory-mapped; no re-download, no re-tokenization — session start is
~2 min).

Recommended session order (the engine does this automatically with
`--stages auto`):

| # | accelerator | what happens | ~time |
|---|---|---|---|
| 1 | **none (CPU)** | env selftests; tokenizer 2M lines (RAM fix, peak RSS logged); saturation check; smoltalk + FineWeb prep at 512/8192; pack to mmap pools; bucket report; data-dataset push | 3–5 h (free, no GPU quota) |
| 2 | T4 x2, `--stages smoke` ONLY | 20M/500-step end-to-end smoke (packed source, 8k eval, position curve) + 155M throughput measurements → s/step recorded. NOT `auto`: don't start pretraining on a small FineWeb pool | ~1 h |
| 3 | **none (CPU)**, `--stages data` | FineWeb re-stream at the 1B cap (the 250M stream lands only ~22M unique ≤512 tokens — see the prereg's 2026-09-06 amendment); repack + push | ~4–7 h (free) |
| 4+ | T4 x2, `--stages auto` | OPERA pretrain 15,000 steps (DDP), governor-limited resumes | ~3 weekly quotas |
| — | T4 x2 | TF RoPE + TF NoPE pretrains concurrently (one per GPU) | ~1–2 sessions |
| — | T4 x2 | SFT ×3 (opera 2,500 steps; then the TF pair) | ~1 session |
| — | T4 x2 | position curves ×3 to 8192, gates H1–H4, long-context samples | <1 h |

After session 3's push: re-attach the updated `opera-lm-patha-data`
dataset (the mount is a snapshot; re-adding picks up the bigger pool).

## What to watch between sessions

- The engine prints a status block; or run
  `python kaggle/patha_session.py --stages status` manually.
- `runs/*/results.jsonl` rows appear at each run's completion;
  `logs/*.log` tails are printed automatically on any failure.
- Preemption between pushes loses at most one 1000-step checkpoint
  interval; the exact-stream resume makes the loss invisible to the
  training stream.

## After the study

Pull artifacts to the Mac:

```bash
kaggle datasets download -d <user>/opera-lm-patha-ckpt -p pathA_artifacts
```

Then: fill the prereg's *Outcomes* section from
`artifacts/gates.json` (PASS/FAIL verbatim), update
`opera-chat/README.md`, and optionally draft paper §4.9 from
`artifacts/summary.md`.

## Manual overrides

```bash
python kaggle/patha_session.py --stages data            # force a stage
python kaggle/patha_session.py --stages pretrain_opera
python kaggle/patha_session.py --stages curves
GOVERNOR_HOURS=8 python kaggle/patha_session.py         # shorter day
SKIP_SELFTEST=1 python kaggle/patha_session.py          # skip full selftest
PATHA_FINWEB_TOKENS=250000000 PATHA_FINWEB_POOL_MIN=0 \
  python kaggle/patha_session.py --stages data          # skip the 1B extension
```
