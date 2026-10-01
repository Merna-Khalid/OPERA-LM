"""Equivalence check for a new accelerator (run first on any new GPU):
the gated-memory OPERA model on `--device` with that device's kernel
backend (Triton on CUDA, Metal on MPS) and, on CUDA, torch.compile --
against the eager fp32 model on CPU with the same weights and batch.
Compares the loss and every parameter gradient.

  python experiments/device_check.py [--device cuda]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from opera_lm import OperaSpinorFenwickTree                 # noqa: E402
from opera_lm.losses import lm_loss                         # noqa: E402


def loss_and_grads(model, tok, lens):
    model.zero_grad(set_to_none=True)
    loss, _, _ = lm_loss(model(tok, lens).logits, tok, lens)
    loss.backward()
    # a torch.compile'd model prefixes its parameter names with _orig_mod.
    return loss.item(), {n.removeprefix('_orig_mod.'): p.grad.float().cpu()
                         for n, p in model.named_parameters() if p.grad is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    p.add_argument('--layers', type=int, default=2)
    args = p.parse_args()
    dev = args.device
    # compare in full fp32: opera_lm.model enables TF32 on CUDA (~1e-3
    # relative per matmul on Ampere+), which would be measured as error here
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    kw = dict(vocab_size=259, d=256, nb=64, num_layers=args.layers, pe_mode='none',
              fold_mode='left', rot_mode='free', head_mode='stream',
              fold_impl='downsweep', hmem_nb=64, hmem_decay='gated')
    if args.layers > 2:
        kw.update(resid_mode='add', resid_init_scale='auto')
    torch.manual_seed(0)
    ref = OperaSpinorFenwickTree(**kw)
    fast = OperaSpinorFenwickTree(use_metal=(dev == 'mps'), use_triton=(dev == 'cuda'), **kw)
    fast.load_state_dict(ref.state_dict())
    fast.to(dev)
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(3, 259, (4, 1024), generator=g)
    lens = torch.tensor([1024, 1000, 700, 64])
    l_ref, g_ref = loss_and_grads(ref, tok, lens)
    runs = [('eager', fast)]
    if dev == 'cuda':
        runs.append(('compiled', torch.compile(fast)))
    ok = True
    for label, m in runs:
        l, gr = loss_and_grads(m, tok.to(dev), lens.to(dev))
        rel = max(((gr[n] - g_ref[n]).norm() / g_ref[n].norm().clamp_min(1e-12)).item()
                  for n in g_ref)
        worst = max(g_ref, key=lambda n: ((gr[n] - g_ref[n]).norm()
                                          / g_ref[n].norm().clamp_min(1e-12)).item())
        good = abs(l - l_ref) < 1e-3 and rel < 1e-2
        ok &= good
        print(f"{dev} {label:8s} loss {l:.6f} vs cpu {l_ref:.6f}  "
              f"max rel grad err {rel:.2e} ({worst})  {'OK' if good else 'FAIL'}",
              flush=True)
    print('DEVICE CHECK ' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
