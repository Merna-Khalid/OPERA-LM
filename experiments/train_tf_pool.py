"""Matched transformer-baseline trainer for a packed pool (the ladder's
tf_byte / tf_tok rungs -- experiments/ladder.py --model tf_byte|tf_tok).

Generalizes opera-chat/train_tf_chat.py (which trains TransformerBaseline
on the small chat pkl) to any packed byte or token pool
(build_fineweb_bytes.py / build_fineweb_tokens.py), with the same
checkpoint/resume/Muon-split machinery, so a rung here is directly
comparable to an OPERA rung (experiments/repr_study.py) trained via
experiments/ladder.py --model opera: same pool, same T, same batch, same
ratio-derived step count, same seed, same cosine schedule.

NOT reusing opera_lm.train.compute_perplexity/extrapolation_eval directly:
those call `model(..., head_last_only=True).logits`, which assumes
OPERA's OperaOutput NamedTuple. TransformerBaseline.forward returns a
plain one-element list (opera-chat/opera_transformer_baseline_v2.py), so
this file defines its own compute_perplexity_tf / extrapolation_eval_tf
that index [0] instead -- everything else (lm_loss, get_lr, make_batch_full,
count_params, _eval_batch, _oom_backstop, split_muon_params, Muon,
PackedBatchSource) is reused unchanged from opera_lm.

  python experiments/train_tf_pool.py --arm tfb_d512_L2_r20 --pe nope \
      --vocab-size 259 --d 512 --num-layers 2 --steps 9537 \
      --packed assets/fineweb_bytes_T1024_4GB --test-pkl \
      assets/fineweb_bytes_T1024_4GB.test.pkl --units-per-byte 1.0 \
      --summary runs_reprs/repr_summary.json --device cuda
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'opera-chat'))

from opera_transformer_baseline_v2 import TransformerBaseline   # noqa: E402
from opera_lm.losses import lm_loss                              # noqa: E402
from opera_lm.muon import Muon, split_muon_params                # noqa: E402
from opera_lm.packed import PackedBatchSource                    # noqa: E402
from opera_lm.reprs import nats_to_bpb                            # noqa: E402
from opera_lm.train import (get_lr, make_batch_full, count_params,
                            _eval_batch, _oom_backstop)           # noqa: E402


@torch.no_grad()
def compute_perplexity_tf(model, test_data, max_len, batch_size, device):
    """compute_perplexity (opera_lm.train), but for a list-returning model
    (no head_last_only / .logits -- see module docstring)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        if not batch:
            continue
        token_ids, lengths = make_batch_full(batch, max_len)
        token_ids, lengths = token_ids.to(device), lengths.to(device)
        all_logits = model(token_ids, lengths)
        _, final_sum, vcount = lm_loss(all_logits, token_ids, lengths)
        total_loss += final_sum.item()
        total_tokens += vcount.item()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


@torch.no_grad()
def extrapolation_eval_tf(model, test_long, train_max_len, eval_max_len,
                          batch_size, device):
    model.eval()
    buckets = []
    lo = train_max_len + 1
    while lo <= eval_max_len:
        hi = min(lo + train_max_len - 1, eval_max_len)
        buckets.append((lo, hi))
        lo = hi + 1
    results = {}
    _oom = (getattr(torch, 'OutOfMemoryError', RuntimeError), RuntimeError)
    for (lo, hi) in buckets:
        sents = [s for s in test_long if lo <= len(s) <= hi]
        if len(sents) < 10:
            results[f"{lo}-{hi}"] = (None, len(sents))
            continue
        eff = max(1, (batch_size * train_max_len) // hi)
        while True:
            try:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                ppl = compute_perplexity_tf(model, sents, hi, eff, device)
                break
            except _oom as e:
                if 'out of memory' not in str(e).lower():
                    raise
                if device == 'cuda':
                    torch.cuda.empty_cache()
                if eff == 1:
                    raise
                eff = max(1, eff // 2)
        results[f"{lo}-{hi}"] = (ppl, len(sents))
    model.train()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, help='tag for checkpoints/summary row')
    p.add_argument('--pe', required=True, choices=['rope', 'nope', 'sin', 'learned'])
    p.add_argument('--vocab-size', type=int, required=True)
    p.add_argument('--units-per-byte', type=float, required=True,
                   help='1.0 for a byte pool; tokens/raw_bytes for a BPE pool '
                        '(build_fineweb_tokens.py .meta.json)')
    p.add_argument('--d', type=int, default=512)
    p.add_argument('--nheads', type=int, default=None, help='default d // 64')
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--max-len', type=int, default=1024)
    p.add_argument('--eval-cap', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    p.add_argument('--packed', required=True)
    p.add_argument('--test-pkl', required=True)
    p.add_argument('--out-dir', default=os.environ.get('OPERA_RUNS',
                   os.path.join(ROOT, 'runs_reprs')))
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-lr', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--lr-schedule', default='cosine', choices=['cosine', 'wsd'])
    p.add_argument('--wsd-decay-frac', type=float, default=0.2)
    p.add_argument('--optimizer', default='muon', choices=['adamw', 'muon'])
    p.add_argument('--muon-lr', type=float, default=0.02)
    p.add_argument('--muon-wd', type=float, default=0.0)
    p.add_argument('--muon-lr-scale', type=float, default=1.0,
                   help='resume-safe multiplier on the Muon group lr, same '
                        'role as ladder.py/repr_study.py --muon-lr-scale')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--summary', default=None,
                   help='repr_summary.json-shaped file to append this arm '
                        'into (ladder.py fit reads it, filtered by arm prefix)')
    p.add_argument('--pool-test', default=None,
                   help='also score on this Simple-Wikipedia pool '
                        '(experiments/eval_pool.py-shaped .test.pkl); skipped '
                        'for tf_tok since eval_pool assumes the byte vocab')
    a = p.parse_args()

    args_summary = a.summary or os.path.join(a.out_dir, 'repr_summary.json')
    if os.path.exists(args_summary):
        with open(args_summary) as f:
            prior = json.load(f)
        if a.arm in prior:
            print(f"[skip] {a.arm} already in {args_summary}")
            return

    nheads = a.nheads or max(1, a.d // 64)
    assert a.d % nheads == 0, f"d={a.d} not divisible by nheads={nheads}"

    arm_dir = os.path.join(a.out_dir, a.arm)
    os.makedirs(arm_dir, exist_ok=True)

    with open(a.test_pkl, 'rb') as f:
        tp = pickle.load(f)
    test_short = [c for _, c in tp['test_short']]
    test_long = [c for _, c in tp['test_long']]

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    model = TransformerBaseline(a.vocab_size, d=a.d, nheads=nheads,
                                num_layers=a.num_layers, pe_mode=a.pe,
                                max_pe_len=min(a.max_len * 4, a.eval_cap),
                                tie=False).to(a.device)
    npar = count_params(model)
    print(f"\n{'=' * 70}\n[{a.arm}] transformer pe={a.pe} d={a.d} "
          f"nheads={nheads} L={a.num_layers} V={a.vocab_size}: "
          f"{npar:,} params", flush=True)

    source = PackedBatchSource(a.packed, a.max_len, a.device, a.seed)

    muon_np, adam_np = split_muon_params(model)
    muon_p = [p for _, p in muon_np]
    adam_p = [p for _, p in adam_np]
    if a.optimizer == 'muon':
        opt = Muon([
            {'params': muon_p, 'use_muon': True, 'lr': a.muon_lr,
             'weight_decay': a.muon_wd},
            {'params': adam_p, 'use_muon': False, 'lr': a.max_lr},
        ], lr=a.max_lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.max_lr,
                                weight_decay=0.0, foreach=True)

    train_ckpt = os.path.join(arm_dir, 'train_ckpt.pt')
    final_path = os.path.join(arm_dir, 'final.pt')
    start_step = 0
    if a.resume and os.path.exists(train_ckpt):
        st = torch.load(train_ckpt, map_location=a.device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        start_step = st['step'] + 1
        random.setstate(st['py_rng'])
        np.random.set_state(st['np_rng'])
        torch.set_rng_state(st['torch_rng'].cpu())
        source.load_state_dict(st['gpu_rng'])
        print(f"  RESUMED from {train_ckpt} at step {start_step}", flush=True)

    if a.device == 'cuda':
        cap = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler(a.device) if amp_dtype == torch.float16 else None

    t0 = time.time()
    nan_skips = 0
    last_loss_finite = True
    for step in range(start_step, a.steps):
        lr = get_lr(step, a.warmup_steps, a.steps, a.max_lr,
                   schedule=a.lr_schedule, wsd_decay_frac=a.wsd_decay_frac)
        for g in opt.param_groups:
            if a.optimizer == 'muon' and g.get('use_muon'):
                g['lr'] = lr * (a.muon_lr / a.max_lr) * a.muon_lr_scale
            else:
                g['lr'] = lr

        token_ids, lengths = source.sample(a.batch)
        with torch.autocast(device_type=a.device, dtype=amp_dtype):
            all_logits = model(token_ids, lengths)
            loss, _, _ = lm_loss(all_logits, token_ids, lengths)

        if not bool(torch.isfinite(loss).all().item()):
            nan_skips += 1
            if nan_skips == 1 or nan_skips % 50 == 0:
                print(f"    WARNING: non-finite loss at step {step} "
                      f"(batch skipped; {nan_skips} skips so far)", flush=True)
            last_loss_finite = False
            continue
        last_loss_finite = True

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()

        if step % 200 == 0 or step == a.steps - 1:
            el = time.time() - t0
            done = step - start_step + 1
            print(f"  step {step:6d}  loss {loss.item():.4f}  lr {lr:.5f}  "
                  f"({el:.1f}s, {el / done:.2f}s/step)", flush=True)
        if (a.save_every and step > 0 and last_loss_finite
                and step % a.save_every == 0):
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                       'step': step, 'py_rng': random.getstate(),
                       'np_rng': np.random.get_state(),
                       'torch_rng': torch.get_rng_state(),
                       'gpu_rng': source.state_dict()}, train_ckpt)

    mins = (time.time() - t0) / 60
    ppl = _oom_backstop(
        lambda b: compute_perplexity_tf(model, test_short, a.max_len, b, a.device),
        _eval_batch(32, a.max_len), a.device)
    nats = math.log(ppl)
    bpb = nats_to_bpb(nats, a.units_per_byte)
    print(f"\n=== [{a.arm}] final in-length PPL {ppl:.2f}  BPB {bpb:.4f} "
          f"({mins:.1f} min, {nan_skips} nan-skips) ===", flush=True)
    eml = min(a.max_len * 4, a.eval_cap)
    extrap = extrapolation_eval_tf(model, test_long, a.max_len, eml, 16, a.device)
    for bucket, (bppl, n) in extrap.items():
        if bppl is not None:
            print(f"  len {bucket}: PPL {bppl:.2f} (n={n})", flush=True)

    torch.save(model.state_dict(), final_path)

    if args_summary:
        summ = {}
        if os.path.exists(args_summary):
            with open(args_summary) as f:
                summ = json.load(f)
        summ[a.arm] = {
            'repr': 'bytes' if abs(a.units_per_byte - 1.0) < 1e-9 else 'bpe',
            'max_len': a.max_len, 'd': a.d, 'nb': nheads,
            'vocab_size': a.vocab_size, 'params': npar,
            'steps': a.steps, 'batch': a.batch,
            'units_per_byte': a.units_per_byte,
            'ppl_in_units': ppl, 'nats_per_unit': nats, 'bpb': bpb,
            'minutes': mins, 'nan_skips': nan_skips,
            'ckpt_dir': arm_dir, 'packed': a.packed, 'layers': a.num_layers,
            'test_pkl': a.test_pkl, 'model': 'transformer', 'pe': a.pe,
            'extrapolation': {k: v[0] for k, v in extrap.items()},
        }
        os.makedirs(os.path.dirname(args_summary) or '.', exist_ok=True)
        with open(args_summary, 'w') as f:
            json.dump(summ, f, indent=2)

    if a.pool_test:
        print("  (--pool-test skipped: eval_pool.py assumes the OPERA byte "
              "arm registry, not a standalone transformer checkpoint)",
              flush=True)

    print(f"ckpt -> {final_path}", flush=True)


if __name__ == '__main__':
    main()
