"""train_kagriculture.py -- behavior-cloning training loop for the OPERA
Kaggriculture agent (Phase 4 of the plan). A bespoke loop, NOT
opera_lm.train.train(): that function's curriculum/msup/aux_frac/DDP/
Muon machinery exists to extract marginal supervision from expensive,
roughly-fixed FineWeb-scale pretraining data on a large model -- none of
that applies to this tiny custom vocab with freely-regenerable self-play
data. Reuses only the low-level building blocks: OperaSpinorFenwickTree
directly, and opera_lm.train.get_lr as a pure LR-schedule function.

Usage:
  python train_kagriculture.py --data kagri_data.pkl --steps 3000
  python train_kagriculture.py --data kagri_data_nogeom.pkl --steps 3000 \
      --out-dir runs_kagri_nogeom
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch

from kagri_common import ensure_opera_lm  # same dir; script dir is on sys.path[0]

ensure_opera_lm()

from opera_lm.model import OperaSpinorFenwickTree                 # noqa: E402
from opera_lm.train import get_lr                                  # noqa: E402
from kagri_losses import masked_lm_loss, action_exact_match        # noqa: E402


def make_batch(chunks, device):
    """chunks: list of dicts with tokens/geom/geom_mask/action_mask
    (variable length, <= the dataset's max_len). Pads to the batch's own
    max length -- not the dataset's global max_len -- since most
    windows are shorter than the cap."""
    B = len(chunks)
    T = max(len(c["tokens"]) for c in chunks)
    token_ids = torch.zeros(B, T, dtype=torch.long)
    geom = torch.zeros(B, T, 3, dtype=torch.float32)
    geom_mask = torch.zeros(B, T, dtype=torch.bool)
    action_mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(chunks):
        t = len(c["tokens"])
        token_ids[i, :t] = torch.from_numpy(c["tokens"])
        geom[i, :t] = torch.from_numpy(c["geom"])
        geom_mask[i, :t] = torch.from_numpy(c["geom_mask"])
        action_mask[i, :t] = torch.from_numpy(c["action_mask"])
        lengths[i] = t
    return (token_ids.to(device), geom.to(device), geom_mask.to(device),
           action_mask.to(device), lengths.to(device))


def evaluate(model, chunks, device, n=256):
    model.eval()
    sample = chunks if len(chunks) <= n else random.sample(chunks, n)
    total_loss, total_count, total_acc_num, total_acc_den = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(sample), 32):
            batch = sample[i:i + 32]
            token_ids, geom, geom_mask, action_mask, lengths = make_batch(
                batch, device)
            out = model(token_ids, lengths, geom=geom, geom_mask=geom_mask,
                       head_last_only=True)
            logits = out.logits[-1]
            loss, count = masked_lm_loss(logits, token_ids, lengths, action_mask)
            acc = action_exact_match(logits, token_ids, lengths, action_mask)
            total_loss += loss.item() * int(count.item())
            total_count += int(count.item())
            total_acc_num += acc.item() * int(count.item())
            total_acc_den += int(count.item())
    model.train()
    denom = max(1, total_count)
    return total_loss / denom, total_acc_num / max(1, total_acc_den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="kagri_data.pkl")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--nb", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="runs_kagri")
    p.add_argument("--save-every", type=int, default=500)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.data, "rb") as f:
        bundle = pickle.load(f)
    vocab_size = bundle["vocab_size"]
    geom_block = bundle["geom_block"]
    use_geom = bundle.get("use_geom", True)
    max_len = bundle["max_len"]
    train_chunks = bundle["train"]
    test_short = bundle["test_short"]
    print(f"data: train={len(train_chunks)} test_short={len(test_short)} "
          f"vocab={vocab_size} use_geom={use_geom}", flush=True)

    model = OperaSpinorFenwickTree(
        vocab_size=vocab_size, d=a.d, nb=a.nb, num_layers=a.num_layers,
        tie=True, pe_mode='none', fold_mode='left', rot_mode='free')
    model.to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.max_lr)

    t0 = time.time()
    for step in range(a.steps):
        lr = get_lr(step, a.warmup_steps, a.steps, a.max_lr)
        for g in opt.param_groups:
            g['lr'] = lr

        batch = random.sample(train_chunks, min(a.batch, len(train_chunks)))
        token_ids, geom, geom_mask, action_mask, lengths = make_batch(
            batch, a.device)
        out = model(token_ids, lengths, geom=geom, geom_mask=geom_mask,
                   head_last_only=True)
        loss, count = masked_lm_loss(out.logits[-1], token_ids, lengths,
                                     action_mask)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == a.steps - 1:
            elapsed = time.time() - t0
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.5f}  "
                  f"({elapsed:.1f}s, {elapsed / (step + 1):.2f}s/step)",
                  flush=True)

        if a.save_every and step > 0 and step % a.save_every == 0:
            torch.save(model.state_dict(),
                      os.path.join(a.out_dir, "train_ckpt.pt"))

        if step > 0 and step % 500 == 0:
            eval_loss, eval_acc = evaluate(model, test_short, a.device)
            print(f"    held-out: loss {eval_loss:.4f}  "
                  f"exact-match {eval_acc:.3f}", flush=True)

    eval_loss, eval_acc = evaluate(model, test_short, a.device, n=len(test_short))
    print(f"\n=== Final Evaluation ===\n  held-out loss {eval_loss:.4f}  "
          f"exact-match {eval_acc:.3f}", flush=True)

    ckpt_path = os.path.join(a.out_dir, "kagri_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    cfg = {"vocab_size": vocab_size, "d": a.d, "nb": a.nb,
          "num_layers": a.num_layers, "tie": True, "pe_mode": "none",
          "fold_mode": "left", "rot_mode": "free", "geom_block": geom_block,
          "use_geom": use_geom, "ckpt": ckpt_path, "max_len": max_len,
          "final_eval_loss": eval_loss, "final_eval_exact_match": eval_acc}
    cfg_path = os.path.join(a.out_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"checkpoint -> {ckpt_path}\nconfig -> {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
