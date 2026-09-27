"""Training-step throughput of the gated-memory OPERA model across sizes
(Recall research §8c): width d, layers L, batch B at T=1024, one full
step = forward + backward + Muon/AdamW update, bf16 autocast, the same
kernel backend and torch.compile policy as train() on that device.

L > 2 uses the additive residual with GPT-2 init scaling
(resid_mode='add', resid_init_scale='auto'); L = 2 is the incumbent
blend residual. Memory slots = d / 4 (as in every memory arm).

  python experiments/bench_scale.py                       # default grid
  python experiments/bench_scale.py --grid 1024x4x32 1536x4x32
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from opera_lm import OperaSpinorFenwickTree                 # noqa: E402
from opera_lm.losses import lm_loss                         # noqa: E402
from opera_lm.muon import Muon, split_muon_params           # noqa: E402

DEFAULT_GRID = ['768x2x8', '768x2x32', '1024x2x32', '1024x4x32',
                '1536x2x32', '1536x4x32', '2048x4x16', '1024x8x32']


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def bench(d, L, B, device, compile_model, reps, T=1024):
    kw = dict(vocab_size=259, d=d, nb=d // 4, num_layers=L, pe_mode='none',
              fold_mode='left', rot_mode='free', head_mode='stream',
              fold_impl='downsweep', hmem_nb=d // 4, hmem_decay='gated',
              use_metal=(device == 'mps'), use_triton=(device == 'cuda'))
    if L > 2:
        kw.update(resid_mode='add', resid_init_scale='auto')
    torch.manual_seed(0)
    model = OperaSpinorFenwickTree(**kw).to(device)
    params = sum(p.numel() for p in model.parameters())
    muon_np, adam_np = split_muon_params(model, ('fusion_gate',))
    opt = Muon([{'params': [p for _, p in muon_np], 'use_muon': True,
                 'lr': 0.02, 'weight_decay': 0.01},
                {'params': [p for _, p in adam_np], 'use_muon': False,
                 'lr': 1e-3}])
    fwd = torch.compile(model) if compile_model else model
    tok = torch.randint(3, 259, (B, T), device=device)
    lens = torch.full((B,), T, device=device)
    amp = torch.bfloat16

    def step():
        with torch.autocast(device, dtype=amp, enabled=device != 'cpu'):
            out = fwd(tok, lens)
        loss, _, _ = lm_loss(out.logits, tok, lens)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    t0 = time.time()
    for _ in range(3):
        step()
    sync(device)
    warm = time.time() - t0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t = time.time()
        step()
        sync(device)
        ts.append(time.time() - t)
    ts.sort()
    ms = 1000 * ts[len(ts) // 2]
    peak = (torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda'
            else torch.mps.driver_allocated_memory() / 2**30 if device == 'mps'
            else float('nan'))
    del model, fwd, opt
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    return dict(d=d, L=L, B=B, params=params, ms=ms, tok_s=B * T / ms * 1000,
                peak_gib=peak, warmup_s=warm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', nargs='+', default=DEFAULT_GRID,
                   help='entries dxLxB')
    p.add_argument('--device', default=None)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--out', default=None, help='append results as JSON lines')
    args = p.parse_args()
    dev = args.device or ('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    compile_model = dev == 'cuda' and not args.no_compile
    if compile_model:
        import torch._dynamo as _dynamo
        _dynamo.config.recompile_limit = 64     # as train(): per-layer graphs
    if dev == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"device {dev}, compile {compile_model}", flush=True)
    for g in args.grid:
        d, L, B = (int(x) for x in g.split('x'))
        try:
            r = bench(d, L, B, dev, compile_model, args.reps)
        except torch.OutOfMemoryError:
            print(f"d{d} L{L} B{B}: OOM", flush=True)
            torch.cuda.empty_cache()
            continue
        print(f"d{d:5d} L{L} B{B:3d}  params {r['params'] / 1e6:6.1f}M  "
              f"step {r['ms']:7.0f} ms  {r['tok_s'] / 1e3:7.1f}k tok/s  "
              f"peak {r['peak_gib']:5.1f} GiB  (warmup {r['warmup_s']:.0f}s)",
              flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
