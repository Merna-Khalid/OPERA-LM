"""train_tf_chat.py -- matched transformer baseline for the BPE chat run.

Trains TransformerBaseline (opera_transformer_baseline_v2.py, the paper's
matched baseline class) on data_chat.pkl with the SAME protocol as the
OPERA chat run: same data, batch 16, max_len 256, 20k steps, lr 1e-3
warmup-500 cosine, bf16 autocast, grad clip 1.0, curriculum (64,250),
seed 42, Muon with the same matrix/AdamW partition. Embeddings are TIED
(head.weight = word_emb.weight) to keep params matched to the tied OPERA
chat model (~21.0M vs 19.9M, +5%). Architecture-specific OPERA arms
(msup, chrono fold bias) have no transformer counterpart -- same
convention as the paper (matched protocol, architecture-native training
methods).

Usage:
  python train_tf_chat.py --pe rope --steps 20000 --device mps
  python train_tf_chat.py --pe nope --steps 20000 --device mps --resume
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch

from chat_common import ensure_opera_lm

ensure_opera_lm()
# opera_transformer_baseline_v2.py lives alongside this script, so Python's
# implicit "script dir on sys.path[0]" import already finds it -- no repo-
# root sys.path hack needed (that assumed the old sibling-directory layout).

from opera_transformer_baseline_v2 import TransformerBaseline   # noqa: E402
from opera_lm.losses import lm_loss                              # noqa: E402
from opera_lm.muon import Muon, split_muon_params                # noqa: E402
from opera_lm.train import (GpuBatchSource, get_lr, curriculum_len,
                            compute_perplexity, extrapolation_eval,
                            count_params, _eval_batch,
                            _oom_backstop)                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pe", required=True,
                   choices=["rope", "nope", "sin", "learned"])
    p.add_argument("--data", default="data_chat.pkl")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-max-len", type=int, default=1024)
    p.add_argument("--device", default="mps")
    p.add_argument("--out-dir", default="runs_chat_tf")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="muon",
                   choices=["adamw", "muon"])
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--curriculum-t0", type=int, default=64)
    p.add_argument("--curriculum-every", type=int, default=250)
    p.add_argument("--lr-schedule", default="cosine",
                   choices=["cosine", "wsd"],
                   help="wsd = warmup-stable-decay (same option as "
                        "opera_lm.train; keep arms matched)")
    p.add_argument("--wsd-decay-frac", type=float, default=0.2)
    p.add_argument("--packed-data", default=None,
                   help="path prefix of a packed train pool (mmap; "
                        "stream-identical to the default source). With "
                        "this flag the --data pkl may carry an empty "
                        "'train' list (eval pools only)")
    p.add_argument("--init-weights-from", default=None,
                   help="load an existing checkpoint's weights before "
                        "training (e.g. a pretrained-on-raw-text "
                        "checkpoint, for a fine-tune phase); independent "
                        "of --resume")
    a = p.parse_args()

    with open(a.data, "rb") as f:
        d = pickle.load(f)
    train_data = d["train"]
    test_short, test_long = d["test_short"], d["test_long"]
    vocab_size = d["vocab_size"]
    assert test_short and test_long, \
        "test_short/test_long must be non-empty"
    if not a.packed_data:
        assert train_data, "train list empty and no --packed-data given"
    print(f"data: train={len(train_data)} short={len(test_short)} "
          f"long={len(test_long)} vocab={vocab_size}", flush=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    model = TransformerBaseline(vocab_size, d=a.d, nheads=8,
                                num_layers=a.num_layers, pe_mode=a.pe,
                                max_pe_len=a.eval_max_len,
                                tie=True).to(a.device)
    # tie=True (OPERA chat runs tie=True): ~21.0M vs OPERA's 19.9M
    if a.init_weights_from is not None:
        sd = torch.load(a.init_weights_from, map_location=a.device,
                        weights_only=True)
        model.load_state_dict(sd)
        print(f"  Initialized weights from {a.init_weights_from}", flush=True)
    npar = count_params(model)
    print(f"=== transformer baseline [{a.pe}] on chat protocol ===",
          flush=True)
    print(f"  params: {npar:,} (tied)  steps={a.steps} batch={a.batch} "
          f"max_len={a.max_len} device={a.device}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"tf_{a.pe}_opt-{a.optimizer}"
    final_path = os.path.join(a.out_dir, f"{tag}.pt")
    train_ckpt = os.path.join(a.out_dir, f"{tag}_train_ckpt.pt")

    if a.packed_data:
        from opera_lm.packed import PackedBatchSource
        source = PackedBatchSource(a.packed_data, a.max_len, a.device,
                                   a.seed)
        print(f"  packed batch source: {source.N:,} seqs (mmap)", flush=True)
    else:
        source = GpuBatchSource(train_data, a.max_len, a.device, a.seed)
    if a.optimizer == "muon":
        muon_p, adam_p = split_muon_params(model)
        opt = Muon([
            {"params": muon_p, "use_muon": True, "lr": a.muon_lr},
            {"params": adam_p, "use_muon": False, "lr": a.max_lr},
        ], lr=a.max_lr)
        print(f"  Muon: {sum(p.numel() for p in muon_p):,} matrix params "
              f"(lr {a.muon_lr}) + AdamW: "
              f"{sum(p.numel() for p in adam_p):,} (lr {a.max_lr})",
              flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.max_lr,
                                weight_decay=0.0, foreach=True)

    start_step = 0
    if a.resume and os.path.exists(train_ckpt):
        st = torch.load(train_ckpt, map_location=a.device,
                        weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if st.get("gpu_rng") is not None:
            source.load_state_dict(st["gpu_rng"])
        print(f"  RESUMED from {train_ckpt} at step {start_step}",
              flush=True)

    # AMP dtype gate, mirroring opera_lm.train: bf16 only where NATIVE
    # (Ampere+ sm_80; MPS). Turing (T4, Kaggle) emulates bf16 slowly --
    # use fp16 + GradScaler there instead.
    if a.device == 'cuda':
        cap = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler(a.device)
    print(f"  AMP dtype: {amp_dtype}", flush=True)

    init_ppl = _oom_backstop(
        lambda b: compute_perplexity(model, test_short[:200], a.max_len, b, a.device),
        32, a.device)
    print(f"  Initial per-token perplexity: {init_ppl:.2f} "
          f"(chance ~ {vocab_size})", flush=True)

    t0 = time.time()
    nan_skips = 0
    last_loss_finite = True
    for step in range(start_step, a.steps):
        lr = get_lr(step, a.warmup_steps, a.steps, a.max_lr,
                    schedule=a.lr_schedule, wsd_decay_frac=a.wsd_decay_frac)
        for g in opt.param_groups:
            if a.optimizer == "muon" and g.get("use_muon"):
                g["lr"] = lr * (a.muon_lr / a.max_lr)
            else:
                g["lr"] = lr

        token_ids, lengths = source.sample(a.batch)
        t_cur = curriculum_len(step, a.curriculum_t0, a.curriculum_every,
                               a.max_len)
        if t_cur < token_ids.shape[1]:
            token_ids = token_ids[:, :t_cur]
            lengths = lengths.clamp(max=t_cur)

        with torch.autocast(device_type=a.device, dtype=amp_dtype):
            all_logits = model(token_ids, lengths)
            loss, _, _ = lm_loss(all_logits, token_ids, lengths)

        # Divergence guard, same policy as opera_lm.train: a non-finite
        # batch is skipped whole (no backward, no update); checkpoint
        # writes are gated on a finite loss so a poisoned run never
        # overwrites its last healthy checkpoint.
        if not bool(torch.isfinite(loss).all().item()):
            nan_skips += 1
            if nan_skips == 1 or nan_skips % 50 == 0:
                print(f"    WARNING: non-finite loss at step {step} "
                      f"(batch skipped; {nan_skips} skips so far)",
                      flush=True)
            last_loss_finite = False
            continue
        last_loss_finite = True

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % 200 == 0 or step == a.steps - 1:
            el = time.time() - t0
            done = step - start_step + 1
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"lr {lr:.5f}  ({el:.1f}s, {el / done:.2f}s/step)  "
                  f"T_cur {t_cur}", flush=True)
        if (a.save_every and step > 0 and last_loss_finite
                and step % a.save_every == 0):
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step,
                        "py_rng": random.getstate(),
                        "np_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "gpu_rng": source.state_dict()}, train_ckpt)
            print(f"    checkpoint -> {train_ckpt}", flush=True)
        if step % 1000 == 0 and step > 0:
            ppl = _oom_backstop(
                lambda b: compute_perplexity(model, test_short[:200], a.max_len, b, a.device),
                _eval_batch(32, a.max_len), a.device)
            print(f"    per-token perplexity: {ppl:.2f}", flush=True)

    ppl = _oom_backstop(
        lambda b: compute_perplexity(model, test_short[:5000], a.max_len, b, a.device),
        _eval_batch(32, a.max_len), a.device)
    print(f"\n=== Final: [{a.pe}] in-length PPL {ppl:.2f} (5k) ===",
          flush=True)
    extrap = extrapolation_eval(model, test_long, a.max_len,
                                a.eval_max_len, 16, a.device)
    for bucket, (bppl, n) in extrap.items():
        if bppl is not None:
            print(f"  len {bucket}: PPL {bppl:.2f} (n={n}, "
                  f"{bppl / ppl:.2f}x in-length)", flush=True)

    torch.save(model.state_dict(), final_path)
    with open(os.path.join(a.out_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps({
            "model": "transformer_baseline", "pe": a.pe,
            "optimizer": a.optimizer, "muon_lr": (a.muon_lr
                            if a.optimizer == "muon" else None),
            "params": npar, "d": a.d, "num_layers": a.num_layers,
            "vocab_size": vocab_size, "steps": a.steps, "seed": a.seed,
            "lr_schedule": a.lr_schedule, "nan_skips": nan_skips,
            "test_perplexity_in_length": ppl,
            "extrapolation": {k: v[0] for k, v in extrap.items()},
        }) + "\n")
    print(f"ckpt -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
