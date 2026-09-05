"""position_curve.py -- the paper's position-curve instrument (eval-only).

Method (mirrors opera_paper_draft §4.3): per-position next-token CE
(final layer, fp32, no autocast, no grad) on held-out long sequences,
aggregated into equal-width position bands. With train length T and eval
horizon H, bands cover [0, H); the band starting at T marks the training
boundary and [T, 2T) is the 2x-training-length zone.

Headline metrics: degradation from the model's best band to its final
2x band (paper: OPERA +0.07, RoPE/NoPE +0.19), the across-boundary delta
(mean CE [T,2T) minus [0,T)), and -- at H > 2T -- the far-band
degradation (best band to the final band of the whole horizon), the
Path A long-context instrument.

Port of the original 20M-pilot script (2026-08-04) with three changes:
(1) the OPERA forward now returns an OperaOutput named tuple -- the old
    `[-1]` indexing grabbed `.energy`; we index `.logits[-1]`;
(2) everything is argparse (device, lengths, TF architecture) instead of
    hardcoded pilot constants (mps / 256 / 1024 / d=512);
(3) results are also written to a JSON file for later table-making.

Usage (Path A protocol, train 512, eval 8192):
  python position_curve.py --ckpt runs_pathA/opera_....pt \
      --config runs_pathA/model_config.json --data data_chat_512.pkl \
      --device cuda --train-len 512 --max-len 8192 --out pc_opera.json
  python position_curve.py --tf --pe rope --d 1264 --num-layers 7 \
      --ckpt runs_pathA/tf_rope.pt --data data_chat_512.pkl \
      --device cuda --train-len 512 --max-len 8192 --out pc_rope.json
"""
import argparse
import json
import os
import pickle

import torch
import torch.nn.functional as F

from chat_common import ensure_opera_lm

ensure_opera_lm()
from generate_chat import load_model                       # noqa: E402
from opera_transformer_baseline_v2 import TransformerBaseline  # noqa: E402


@torch.no_grad()
def _batch_ce(model, ids, lens):
    """Per-position next-token CE sum/count contribution of one batch
    (fp32 forward, no autocast -- the instrument's convention)."""
    out = model(ids, lens, head_last_only=True)
    logits = (out.logits if hasattr(out, "logits") else out)[-1]
    lg = logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                         reduction='none').reshape(ids.shape[0],
                                                   ids.shape[1] - 1)
    pos = torch.arange(ids.shape[1] - 1, device=ids.device)
    valid = (pos[None, :] + 1) < lens[:, None]
    return ce * valid.float(), valid


@torch.no_grad()
def per_position_ce(model, data, max_len, device, batch):
    """Sum of next-token CE and token count per position (fp64 accum).
    A batch that OOMs (long horizons materialize [B, T, V] logits) is
    retried element-by-element before giving up."""
    S = torch.zeros(max_len - 1, dtype=torch.float64)
    C = torch.zeros(max_len - 1, dtype=torch.float64)
    model.eval()
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        for b, s in enumerate(chunk):
            assert len(s) >= 2, "sequence shorter than 2 tokens"
        done = False
        for size in (len(chunk), 1):
            if done or size > len(chunk):
                continue
            try:
                sub = chunk[:size]
                ids = torch.zeros(len(sub), max_len, dtype=torch.long)
                lens = torch.zeros(len(sub), dtype=torch.long)
                for b, s in enumerate(sub):
                    ids[b, :len(s)] = torch.tensor(s[:max_len])
                    lens[b] = min(len(s), max_len)
                ids, lens = ids.to(device), lens.to(device)
                ce_m, val_m = _batch_ce(model, ids, lens)
                for b in range(len(sub)):
                    S += (ce_m[b] * val_m[b].float()).cpu().double()
                C += val_m.sum(dim=0).cpu().double()
                done = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    (OOM at sub-batch {size}; retrying smaller)",
                      flush=True)
        if not done:
            raise RuntimeError("position_curve: OOM even at batch 1")
        if (i // batch) % 25 == 0:
            print(f"  {i}/{len(data)}", flush=True)
    return S, C


def band_table(S, C, train_len, max_len, band):
    """(lo, hi, mean_ce, n_tokens) for every populated band to max_len."""
    rows = []
    for lo in range(0, max_len, band):
        hi = min(lo + band, max_len)
        n_tok = int(C[lo:hi].sum())
        if n_tok == 0:
            continue
        ce = (S[lo:hi].sum() / max(C[lo:hi].sum(), 1)).item()
        rows.append({"lo": lo, "hi": hi, "ce": ce, "tokens": n_tok})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data_chat.pkl",
                   help="pkl with test_long pool (keys as prepare_data.py)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None,
                   help="model_config.json (OPERA arm; required unless --tf)")
    p.add_argument("--tf", action="store_true",
                   help="load a TransformerBaseline checkpoint instead")
    p.add_argument("--pe", default="rope", choices=["rope", "nope",
                                                    "sin", "learned"])
    p.add_argument("--d", type=int, default=1264,
                   help="TF width (155M-study default 1264)")
    p.add_argument("--num-layers", type=int, default=7,
                   help="TF depth (155M-study default 7)")
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--device", default="mps")
    p.add_argument("--train-len", type=int, default=512)
    p.add_argument("--max-len", type=int, default=8192,
                   help="eval horizon (logits are [B, T, V] fp32; OOM "
                        "auto-retries smaller batches)")
    p.add_argument("--min-len", type=int, default=None,
                   help="keep test sequences >= this (default 2*train-len)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--band", type=int, default=128)
    p.add_argument("--n", type=int, default=1000,
                   help="max sequences (first-n in pool order, the "
                        "v1.2 convention)")
    p.add_argument("--out", default=None, help="write results JSON here")
    a = p.parse_args()
    min_len = a.min_len if a.min_len is not None else 2 * a.train_len

    with open(a.data, "rb") as f:
        d = pickle.load(f)
    pool = d["test_long"]
    seqs = [s for s in pool if len(s) >= min_len][:a.n]
    assert seqs, f"no test_long sequences >= {min_len} tokens"
    n_long = sum(1 for s in seqs if len(s) >= a.max_len)
    print(f"eval set: {len(seqs)} held-out sequences (>= {min_len} tokens, "
          f"capped {a.max_len}; {n_long} reach the horizon; train length "
          f"{a.train_len})", flush=True)

    if a.tf:
        model = TransformerBaseline(d["vocab_size"], d=a.d,
                                    nheads=a.nheads,
                                    num_layers=a.num_layers,
                                    pe_mode=a.pe,
                                    max_pe_len=a.max_len, tie=True)
        st = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        if "model" in st:                  # train_ckpt vs final state_dict
            st = st["model"]
        model.load_state_dict(st)
        model = model.to(a.device).eval()
        arm = f"transformer-{a.pe}"
    else:
        assert a.config, "--config (model_config.json) required for OPERA"
        model = load_model(a.config, a.ckpt, a.device)
        arm = "opera"
    print(f"model: {arm} <- {a.ckpt}", flush=True)

    S, C = per_position_ce(model, seqs, a.max_len, a.device, a.batch)
    bands = band_table(S, C, a.train_len, a.max_len, a.band)

    print(f"\n{'band':>12s} | {'CE':>7s} | {'tokens':>8s}")
    print("-" * 35)
    for b in bands:
        mark = ""
        if b["lo"] == a.train_len - (a.train_len % a.band or a.band):
            mark = "  <- train boundary zone"
        if b["lo"] == a.train_len:
            mark = "  <- 2x zone starts"
        print(f"{b['lo']:>6d}-{b['hi'] - 1:<5d} | {b['ce']:7.4f} | "
              f"{b['tokens']:>8d}{mark}", flush=True)

    in_bands = [b for b in bands if b["hi"] <= a.train_len]
    two_x = [b for b in bands if a.train_len <= b["lo"] < 2 * a.train_len]
    best = min(bands, key=lambda b: b["ce"])
    final_2x = max(two_x, key=lambda b: b["hi"]) if two_x else None
    final_all = bands[-1]

    def mean(lo, hi):
        tot = C[lo:hi].sum()
        return (S[lo:hi].sum() / max(tot, 1)).item() if tot > 0 else None

    b_ce = mean(0, a.train_len)
    a_ce = mean(a.train_len, 2 * a.train_len)
    res = {
        "arm": arm, "ckpt": a.ckpt, "data": a.data,
        "train_len": a.train_len, "max_len": a.max_len,
        "band": a.band, "n_seqs": len(seqs),
        "n_reach_horizon": n_long,
        "bands": bands,
        "mean_ce_in_length": b_ce,
        "mean_ce_2x": a_ce,
        "boundary_delta": (a_ce - b_ce) if (a_ce is not None
                                            and b_ce is not None) else None,
        "best_band": best,
        "degradation_best_to_final_2x": (final_2x["ce"] - best["ce"])
        if final_2x else None,
        "degradation_best_to_final_horizon": final_all["ce"] - best["ce"],
    }
    print(f"\nbest band {best['lo']}-{best['hi'] - 1}: {best['ce']:.4f}")
    if final_2x:
        print(f"final 2x band {final_2x['lo']}-{final_2x['hi'] - 1}: "
              f"{final_2x['ce']:.4f}  "
              f"(degradation {res['degradation_best_to_final_2x']:+.3f} nats;"
              f"  paper 20M pilot: OPERA +0.259 / RoPE +0.176 / NoPE +0.688)")
    print(f"final horizon band {final_all['lo']}-{final_all['hi'] - 1}: "
          f"{final_all['ce']:.4f}  (degradation "
          f"{res['degradation_best_to_final_horizon']:+.3f} nats)")
    if b_ce is not None and a_ce is not None:
        print(f"mean CE in-length (0-{a.train_len - 1}): {b_ce:.4f} | "
              f"2x zone ({a.train_len}-{2 * a.train_len - 1}): {a_ce:.4f} | "
              f"across-boundary delta: {a_ce - b_ce:+.4f}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"results -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
