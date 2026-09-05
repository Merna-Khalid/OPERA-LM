"""train_transformer_kagriculture.py -- the matched-baseline counterpart
to train_kagriculture.py: same behavior-cloning recipe, same data, same
loss, but a parameter-matched plain transformer (TransformerBaseline
from opera-chat/opera_transformer_baseline_v2.py, the SAME class used
for every other OPERA-vs-transformer comparison in this project) instead
of OperaSpinorFenwickTree.

pe_mode='nope' (no positional encoding at all -- position is implicit/
emergent from the causal mask) is the deliberate, fair comparison arm
against OPERA's pe_mode='none' (position is explicit/constructive from
the Fenwick decomposition): same claim ("no injected position"),
different mechanism. This is what actually answers "does OPERA's
architecture do something useful here, or would any similarly-sized
sequence model do just as well/poorly" -- the comparison this project's
own paper insists on everywhere else, and the one thing missing from
the OPERA-only Kaggriculture results so far.

Trained on kagri_data_nogeom.pkl specifically: a plain transformer has
no equivalent of geometric injection, so the fair data condition is the
one where positions are ordinary discrete tokens like everything else.

Params matched to OPERA's Kaggriculture config (d=128, nb=32,
num_layers=2, tie=True -> 223,109 params) via d=96, nheads=4,
num_layers=2, ffn_mult=3.3, tie=True -> 221,390 params (99.2%).

Usage:
  python train_transformer_kagriculture.py --data kagri_data_nogeom.pkl --steps 3000
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

from kagri_common import ensure_opera_lm

ensure_opera_lm()
_CHAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "opera-chat")
if _CHAT_DIR not in sys.path:
    sys.path.insert(0, _CHAT_DIR)

from opera_lm.train import get_lr                                  # noqa: E402
from kagri_losses import masked_lm_loss, action_exact_match        # noqa: E402
from opera_transformer_baseline_v2 import TransformerBaseline      # noqa: E402

D, NHEADS, NUM_LAYERS, FFN_MULT = 96, 4, 2, 3.3   # -> 221,390 params, 99.2% of OPERA's


def make_batch(chunks, device):
    B = len(chunks)
    T = max(len(c["tokens"]) for c in chunks)
    token_ids = torch.zeros(B, T, dtype=torch.long)
    action_mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(chunks):
        t = len(c["tokens"])
        token_ids[i, :t] = torch.from_numpy(c["tokens"])
        action_mask[i, :t] = torch.from_numpy(c["action_mask"])
        lengths[i] = t
    return token_ids.to(device), action_mask.to(device), lengths.to(device)


def evaluate(model, chunks, device, n=256):
    model.eval()
    sample = chunks if len(chunks) <= n else random.sample(chunks, n)
    total_loss, total_count, total_acc_num, total_acc_den = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(sample), 32):
            batch = sample[i:i + 32]
            token_ids, action_mask, lengths = make_batch(batch, device)
            logits = model(token_ids, lengths)[0]
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
    p.add_argument("--data", default="kagri_data_nogeom.pkl")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="runs_kagri_transformer")
    p.add_argument("--save-every", type=int, default=500)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.data, "rb") as f:
        bundle = pickle.load(f)
    vocab_size = bundle["vocab_size"]
    use_geom = bundle.get("use_geom", True)
    assert not use_geom, (
        "train_transformer_kagriculture.py is the plain-transformer "
        "baseline -- it has no equivalent of geometric injection, so it "
        "must train on the --no-geom dataset (kagri_data_nogeom.pkl), "
        "not the geometry-injected one.")
    max_len = bundle["max_len"]
    train_chunks = bundle["train"]
    test_short = bundle["test_short"]
    print(f"data: train={len(train_chunks)} test_short={len(test_short)} "
          f"vocab={vocab_size}", flush=True)

    model = TransformerBaseline(vocab_size=vocab_size, d=D, nheads=NHEADS,
                                num_layers=NUM_LAYERS, pe_mode='nope',
                                ffn_mult=FFN_MULT, tie=True)
    model.to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} "
          f"(OPERA Kaggriculture config: 223,109 -> {n_params / 223109 * 100:.1f}%)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.max_lr)

    t0 = time.time()
    for step in range(a.steps):
        lr = get_lr(step, a.warmup_steps, a.steps, a.max_lr)
        for g in opt.param_groups:
            g['lr'] = lr

        batch = random.sample(train_chunks, min(a.batch, len(train_chunks)))
        token_ids, action_mask, lengths = make_batch(batch, a.device)
        logits = model(token_ids, lengths)[0]
        loss, count = masked_lm_loss(logits, token_ids, lengths, action_mask)

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
    cfg = {"vocab_size": vocab_size, "d": D, "nheads": NHEADS,
          "num_layers": NUM_LAYERS, "ffn_mult": FFN_MULT, "tie": True,
          "pe_mode": "nope", "use_geom": False, "ckpt": ckpt_path,
          "max_len": max_len, "final_eval_loss": eval_loss,
          "final_eval_exact_match": eval_acc, "n_params": n_params}
    cfg_path = os.path.join(a.out_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"checkpoint -> {ckpt_path}\nconfig -> {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
