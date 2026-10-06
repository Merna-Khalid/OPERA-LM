"""GPU verification for the rot_free-under-compile fix + Muon fresh-group
warmup (Recall research §8f). NOT YET RUN ON A GPU as of 2026-10-01 --
this must PASS before OPERA_TRITON_CUSTOM_OP=1 / muon_fresh_substr is used
on any run that matters. Cheap: a gradient check, then a short smoke train
at a small width, both in a couple of minutes on an A100. Uses synthetic
random token data (no download, no pool needed).

  OPERA_TRITON_CUSTOM_OP=1 python experiments/verify_rotfree_fix.py --device cuda

Background: compiled on CUDA, the plain autograd.Function compose kernel
gives `rot_free` an exactly-zero gradient (experiments/device_check.py,
rel err 1.00) -- every ladder rung and the first ~6k steps of the 18h
demo run trained with it structurally frozen. opera_lm.triton_kernel's
torch.library custom-op path (OPERA_TRITON_CUSTOM_OP=1) fixes that, but
its first (only) GPU use diverged at step 147 -- plausibly because Muon's
orthogonalized update hit a never-before-updated rot_free at full
strength (and/or the fp32 tanh overflow that 8g root-caused afterwards --
that one is fixed in triton_kernel.py and checked by this suite's --test
case 4b, not by this script). train()'s muon_fresh_substr/muon_fresh_warmup
ramps a named group's Muon lr from 0 instead. This script checks both
pieces before either is trusted again.
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def check_gradients(device, d, nb, layers):
    print(f"\n--- (1) gradient check: compiled CUDA vs CPU fp32 reference, "
          f"d{d} L{layers} ---", flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import device_check as dc
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from opera_lm import OperaSpinorFenwickTree
    kw = dict(vocab_size=259, d=d, nb=nb, num_layers=layers, pe_mode='none',
              fold_mode='left', rot_mode='free', head_mode='stream',
              fold_impl='downsweep', hmem_nb=nb, hmem_decay='gated')
    if layers > 2:
        kw.update(resid_mode='add', resid_init_scale='auto')
    torch.manual_seed(0)
    ref = OperaSpinorFenwickTree(**kw)
    fast = OperaSpinorFenwickTree(use_triton=True, **kw)
    fast.load_state_dict(ref.state_dict())
    fast.to(device)
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(3, 259, (4, 1024), generator=g)
    lens = torch.tensor([1024, 1000, 700, 64])
    l_ref, g_ref = dc.loss_and_grads(ref, tok, lens)
    ok = True
    for label, m in (('eager', fast), ('compiled', torch.compile(fast))):
        l, gr = dc.loss_and_grads(m, tok.to(device), lens.to(device))
        missing = [n for n in g_ref if n not in gr]
        bad = missing + [n for n in g_ref if n in gr and
                         ((gr[n] - g_ref[n]).norm()
                          / g_ref[n].norm().clamp_min(1e-12)).item() > 1e-2]
        good = abs(l - l_ref) < 1e-3 and not bad
        ok &= good
        rot_err = (((gr.get('rot_free', torch.zeros(1)) - g_ref['rot_free']).norm()
                   / g_ref['rot_free'].norm().clamp_min(1e-12)).item()
                   if 'rot_free' in g_ref else float('nan'))
        print(f"  {label:8s} loss {l:.6f} vs {l_ref:.6f}  rot_free rel err "
              f"{rot_err:.2e}  {'OK' if good else 'FAIL ' + str(bad[:5])}",
              flush=True)
    return ok


def check_training(device, d, nb, layers, steps, fresh_warmup, batch=16, T=512):
    print(f"\n--- (2) smoke train: d{d} L{layers}, {steps} steps, "
          f"fresh_warmup={fresh_warmup} ---", flush=True)
    import importlib
    train_mod = importlib.import_module('opera_lm.train')
    r = random.Random(0)
    seqs = [[r.randrange(3, 259) for _ in range(T)] for _ in range(2000)]
    data = (seqs, seqs[:200], [], 259)
    kw = dict(steps=steps, batch=batch, max_len=T, vocab_size=259, d=d, nb=nb,
              num_layers=layers, eval_max_len=T, device=device, pe_mode='none',
              fold_mode='left', rot_mode='free', data=data, seed=0,
              out_dir='/tmp/verify_rotfree_fix', optimizer='muon',
              head_mode='stream', fold_impl='downsweep', hmem_nb=nb,
              hmem_decay='gated', use_triton=(device == 'cuda'),
              muon_fresh_substr='rot_free', muon_fresh_warmup=fresh_warmup,
              muon_lr=0.02)
    try:
        res = train_mod.train(**kw)
    except RuntimeError as e:
        print(f"  FAIL: training diverged -- {e}", flush=True)
        return False
    skips = res.get('nan_skips', 0)
    frac = skips / steps
    ok = frac < 0.05
    print(f"  final loss {res['final_loss']:.4f}  nan_skips {skips}/{steps} "
          f"({100 * frac:.1f}%)  {'OK' if ok else 'FAIL (too many skips)'}",
          flush=True)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda')
    p.add_argument('--d', type=int, default=1024, help='width for both checks')
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--steps', type=int, default=500, help='smoke-train steps')
    p.add_argument('--fresh-warmup', type=int, default=100)
    args = p.parse_args()
    assert os.environ.get('OPERA_TRITON_CUSTOM_OP') == '1', (
        "set OPERA_TRITON_CUSTOM_OP=1 -- this script verifies that path, "
        "not the current default (autograd.Function, frozen rot_free)")
    ok1 = check_gradients(args.device, args.d, args.d // 4, args.layers)
    ok2 = check_training(args.device, args.d, args.d // 4, args.layers,
                         args.steps, args.fresh_warmup)
    print(f"\nVERIFY_ROTFREE_FIX: {'PASS' if ok1 and ok2 else 'FAIL'}")
    if ok1 and ok2:
        print("Both checks passed. Before using this on a real budget, also "
              "run a short smoke test (a few hundred steps) at the ACTUAL "
              "target width -- this script used d%d for speed." % args.d)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
