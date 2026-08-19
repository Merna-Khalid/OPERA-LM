"""
Fused Triton kernel for OPERA's compose node -- a CUDA port of
metal_kernel.py's V2 kernel (`fused_node`): rotate both children +
geometric product + gated combine + output rotation, all in one kernel
launch (forward), with a matching backward kernel that recomputes those
intermediates from the saved inputs instead of loading them. Also ports
the V3 fused activation (`fused_act`, tanh(x) + 0.1x).

This file does not re-derive the compose node's math: it transliterates
the already-verified MSL kernel in metal_kernel.py (NODE_FWD/NODE_BWD)
line-for-line into Triton, and reuses metal_kernel.py's own einsum-based
reference implementations (_node_fwd_reference / _node_bwd_reference) as
both the CPU/no-CUDA fallback and the correctness oracle below -- so the
risk surface here is "did the transliteration preserve the math," not
"is the math right" (that was already settled by metal_kernel.py's own
test suite).

Strategy credit: Truong, M. H., Hirst, E. (2026), "Versor: A Geometric
Sequence Architecture Enhanced Scale Generalization and Interpretability
via Conformal Algebra" (arXiv:2602.10195), report a 78x kernel speedup
for their Clifford-algebra sequence model from custom Triton/MLX kernels
that exploit the algebra's FIXED, sparse basis-multiplication table
(bit-masked basis contraction) instead of a dense GEMM per geometric
product. OPERA's compose node has the same shape of opportunity: `gs`/
`gv` below are a small fixed bilinear form (7 multiplies + a cross
product per block, not a general matmul), which is exactly why the
eager path's dense-GEMM block-rotation trick (`_dense_rot`/M_L/M_R in
model.py) is memory-hungry relative to the actual arithmetic intensity
-- materializing an [nb*3, nb*3]-shaped dense rotation matrix to do a
handful of FMAs per block is the mismatch this kernel removes, mirroring
Versor's bit-masked-contraction fix for the same underlying mismatch.

Requirements: a CUDA device and `triton` installed. Falls back to pure
PyTorch (identical math, delegated to metal_kernel.py's own reference
functions) everywhere else -- the fallback is also the reference for the
correctness tests below.

Run the test suite ON YOUR GPU MACHINE before training with it:
    python3 -m opera_lm.triton_kernel --test
It checks (1) the fallback reproduces metal_kernel.py's own verified
reference math, (2) analytic backward matches autograd, and -- if a CUDA
GPU + triton are available -- (3) the Triton kernel's forward/backward
match the fallback bitwise-tolerably, including a non-orthogonal R_O case
(rot_mode='free').
NOTE: written and math-verified WITHOUT a CUDA GPU available in this
session -- ported mechanically from the already-hardware-verified Metal
kernel rather than re-derived, but the Triton execution path itself is
UNTESTED until --test passes on real hardware. Please run it (Kaggle's
T4s, e.g.) before trusting this in a real training run, and report back
anything that fails.
"""
import torch

from .metal_kernel import (_node_fwd_reference, _node_bwd_reference,
                           _fwd_reference)

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except ImportError as e:                              # pragma: no cover
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e


def triton_available():
    return triton is not None and torch.cuda.is_available()


DEFAULT_BLOCK_SIZE = 256


if triton is not None:

    @triton.jit
    def _node_fwd_kernel(
        h_l_ptr, h_r_ptr, RL_ptr, RR_ptr, RO_ptr, g_ptr, out_ptr,
        total, nb,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = t < total
        n = t // nb
        b = t % nb
        base = t * 4
        rb = b * 9

        sl = tl.load(h_l_ptr + base + 0, mask=mask, other=0.0)
        vlx = tl.load(h_l_ptr + base + 1, mask=mask, other=0.0)
        vly = tl.load(h_l_ptr + base + 2, mask=mask, other=0.0)
        vlz = tl.load(h_l_ptr + base + 3, mask=mask, other=0.0)
        sr = tl.load(h_r_ptr + base + 0, mask=mask, other=0.0)
        vrx = tl.load(h_r_ptr + base + 1, mask=mask, other=0.0)
        vry = tl.load(h_r_ptr + base + 2, mask=mask, other=0.0)
        vrz = tl.load(h_r_ptr + base + 3, mask=mask, other=0.0)

        RL0 = tl.load(RL_ptr + rb + 0, mask=mask, other=0.0)
        RL1 = tl.load(RL_ptr + rb + 1, mask=mask, other=0.0)
        RL2 = tl.load(RL_ptr + rb + 2, mask=mask, other=0.0)
        RL3 = tl.load(RL_ptr + rb + 3, mask=mask, other=0.0)
        RL4 = tl.load(RL_ptr + rb + 4, mask=mask, other=0.0)
        RL5 = tl.load(RL_ptr + rb + 5, mask=mask, other=0.0)
        RL6 = tl.load(RL_ptr + rb + 6, mask=mask, other=0.0)
        RL7 = tl.load(RL_ptr + rb + 7, mask=mask, other=0.0)
        RL8 = tl.load(RL_ptr + rb + 8, mask=mask, other=0.0)
        RR0 = tl.load(RR_ptr + rb + 0, mask=mask, other=0.0)
        RR1 = tl.load(RR_ptr + rb + 1, mask=mask, other=0.0)
        RR2 = tl.load(RR_ptr + rb + 2, mask=mask, other=0.0)
        RR3 = tl.load(RR_ptr + rb + 3, mask=mask, other=0.0)
        RR4 = tl.load(RR_ptr + rb + 4, mask=mask, other=0.0)
        RR5 = tl.load(RR_ptr + rb + 5, mask=mask, other=0.0)
        RR6 = tl.load(RR_ptr + rb + 6, mask=mask, other=0.0)
        RR7 = tl.load(RR_ptr + rb + 7, mask=mask, other=0.0)
        RR8 = tl.load(RR_ptr + rb + 8, mask=mask, other=0.0)

        v0x = RL0 * vlx + RL1 * vly + RL2 * vlz
        v0y = RL3 * vlx + RL4 * vly + RL5 * vlz
        v0z = RL6 * vlx + RL7 * vly + RL8 * vlz
        v1x = RR0 * vrx + RR1 * vry + RR2 * vrz
        v1y = RR3 * vrx + RR4 * vry + RR5 * vrz
        v1z = RR6 * vrx + RR7 * vry + RR8 * vrz

        gs = sl * sr - (v0x * v1x + v0y * v1y + v0z * v1z)
        gvx = sl * v1x + sr * v0x + (v0y * v1z - v0z * v1y)
        gvy = sl * v1y + sr * v0y + (v0z * v1x - v0x * v1z)
        gvz = sl * v1z + sr * v0z + (v0x * v1y - v0y * v1x)

        g0 = tl.load(g_ptr + (n * 3 + 0) * nb + b, mask=mask, other=0.0)
        g1 = tl.load(g_ptr + (n * 3 + 1) * nb + b, mask=mask, other=0.0)
        g2 = tl.load(g_ptr + (n * 3 + 2) * nb + b, mask=mask, other=0.0)

        fs = g0 * sl + g1 * sr + g2 * gs
        fvx = g0 * v0x + g1 * v1x + g2 * gvx
        fvy = g0 * v0y + g1 * v1y + g2 * gvy
        fvz = g0 * v0z + g1 * v1z + g2 * gvz

        RO0 = tl.load(RO_ptr + rb + 0, mask=mask, other=0.0)
        RO1 = tl.load(RO_ptr + rb + 1, mask=mask, other=0.0)
        RO2 = tl.load(RO_ptr + rb + 2, mask=mask, other=0.0)
        RO3 = tl.load(RO_ptr + rb + 3, mask=mask, other=0.0)
        RO4 = tl.load(RO_ptr + rb + 4, mask=mask, other=0.0)
        RO5 = tl.load(RO_ptr + rb + 5, mask=mask, other=0.0)
        RO6 = tl.load(RO_ptr + rb + 6, mask=mask, other=0.0)
        RO7 = tl.load(RO_ptr + rb + 7, mask=mask, other=0.0)
        RO8 = tl.load(RO_ptr + rb + 8, mask=mask, other=0.0)

        ovx = RO0 * fvx + RO1 * fvy + RO2 * fvz
        ovy = RO3 * fvx + RO4 * fvy + RO5 * fvz
        ovz = RO6 * fvx + RO7 * fvy + RO8 * fvz

        tl.store(out_ptr + base + 0, fs, mask=mask)
        tl.store(out_ptr + base + 1, ovx, mask=mask)
        tl.store(out_ptr + base + 2, ovy, mask=mask)
        tl.store(out_ptr + base + 3, ovz, mask=mask)

    @triton.jit
    def _node_bwd_kernel(
        h_l_ptr, h_r_ptr, RL_ptr, RR_ptr, RO_ptr, g_ptr, gout_ptr,
        dh_l_ptr, dh_r_ptr, dg_ptr, dv0_ptr, dv1_ptr, fv_ptr,
        total, nb,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = t < total
        n = t // nb
        b = t % nb
        base = t * 4
        b3 = t * 3
        rb = b * 9

        sl = tl.load(h_l_ptr + base + 0, mask=mask, other=0.0)
        vlx = tl.load(h_l_ptr + base + 1, mask=mask, other=0.0)
        vly = tl.load(h_l_ptr + base + 2, mask=mask, other=0.0)
        vlz = tl.load(h_l_ptr + base + 3, mask=mask, other=0.0)
        sr = tl.load(h_r_ptr + base + 0, mask=mask, other=0.0)
        vrx = tl.load(h_r_ptr + base + 1, mask=mask, other=0.0)
        vry = tl.load(h_r_ptr + base + 2, mask=mask, other=0.0)
        vrz = tl.load(h_r_ptr + base + 3, mask=mask, other=0.0)

        RL0 = tl.load(RL_ptr + rb + 0, mask=mask, other=0.0)
        RL1 = tl.load(RL_ptr + rb + 1, mask=mask, other=0.0)
        RL2 = tl.load(RL_ptr + rb + 2, mask=mask, other=0.0)
        RL3 = tl.load(RL_ptr + rb + 3, mask=mask, other=0.0)
        RL4 = tl.load(RL_ptr + rb + 4, mask=mask, other=0.0)
        RL5 = tl.load(RL_ptr + rb + 5, mask=mask, other=0.0)
        RL6 = tl.load(RL_ptr + rb + 6, mask=mask, other=0.0)
        RL7 = tl.load(RL_ptr + rb + 7, mask=mask, other=0.0)
        RL8 = tl.load(RL_ptr + rb + 8, mask=mask, other=0.0)
        RR0 = tl.load(RR_ptr + rb + 0, mask=mask, other=0.0)
        RR1 = tl.load(RR_ptr + rb + 1, mask=mask, other=0.0)
        RR2 = tl.load(RR_ptr + rb + 2, mask=mask, other=0.0)
        RR3 = tl.load(RR_ptr + rb + 3, mask=mask, other=0.0)
        RR4 = tl.load(RR_ptr + rb + 4, mask=mask, other=0.0)
        RR5 = tl.load(RR_ptr + rb + 5, mask=mask, other=0.0)
        RR6 = tl.load(RR_ptr + rb + 6, mask=mask, other=0.0)
        RR7 = tl.load(RR_ptr + rb + 7, mask=mask, other=0.0)
        RR8 = tl.load(RR_ptr + rb + 8, mask=mask, other=0.0)
        RO0 = tl.load(RO_ptr + rb + 0, mask=mask, other=0.0)
        RO1 = tl.load(RO_ptr + rb + 1, mask=mask, other=0.0)
        RO2 = tl.load(RO_ptr + rb + 2, mask=mask, other=0.0)
        RO3 = tl.load(RO_ptr + rb + 3, mask=mask, other=0.0)
        RO4 = tl.load(RO_ptr + rb + 4, mask=mask, other=0.0)
        RO5 = tl.load(RO_ptr + rb + 5, mask=mask, other=0.0)
        RO6 = tl.load(RO_ptr + rb + 6, mask=mask, other=0.0)
        RO7 = tl.load(RO_ptr + rb + 7, mask=mask, other=0.0)
        RO8 = tl.load(RO_ptr + rb + 8, mask=mask, other=0.0)

        # recompute v0, v1, gs, gv, fv -- nothing was saved
        v0x = RL0 * vlx + RL1 * vly + RL2 * vlz
        v0y = RL3 * vlx + RL4 * vly + RL5 * vlz
        v0z = RL6 * vlx + RL7 * vly + RL8 * vlz
        v1x = RR0 * vrx + RR1 * vry + RR2 * vrz
        v1y = RR3 * vrx + RR4 * vry + RR5 * vrz
        v1z = RR6 * vrx + RR7 * vry + RR8 * vrz
        gs = sl * sr - (v0x * v1x + v0y * v1y + v0z * v1z)
        gvx = sl * v1x + sr * v0x + (v0y * v1z - v0z * v1y)
        gvy = sl * v1y + sr * v0y + (v0z * v1x - v0x * v1z)
        gvz = sl * v1z + sr * v0z + (v0x * v1y - v0y * v1x)

        g0 = tl.load(g_ptr + (n * 3 + 0) * nb + b, mask=mask, other=0.0)
        g1 = tl.load(g_ptr + (n * 3 + 1) * nb + b, mask=mask, other=0.0)
        g2 = tl.load(g_ptr + (n * 3 + 2) * nb + b, mask=mask, other=0.0)
        fvx = g0 * v0x + g1 * v1x + g2 * gvx
        fvy = g0 * v0y + g1 * v1y + g2 * gvy
        fvz = g0 * v0z + g1 * v1z + g2 * gvz

        ds_out = tl.load(gout_ptr + base + 0, mask=mask, other=0.0)
        dovx = tl.load(gout_ptr + base + 1, mask=mask, other=0.0)
        dovy = tl.load(gout_ptr + base + 2, mask=mask, other=0.0)
        dovz = tl.load(gout_ptr + base + 3, mask=mask, other=0.0)
        # dfv = R_O^T @ dov
        dfvx = RO0 * dovx + RO3 * dovy + RO6 * dovz
        dfvy = RO1 * dovx + RO4 * dovy + RO7 * dovz
        dfvz = RO2 * dovx + RO5 * dovy + RO8 * dovz

        dg0 = ds_out * sl + (dfvx * v0x + dfvy * v0y + dfvz * v0z)
        dg1 = ds_out * sr + (dfvx * v1x + dfvy * v1y + dfvz * v1z)
        dg2 = ds_out * gs + (dfvx * gvx + dfvy * gvy + dfvz * gvz)
        dgs = g2 * ds_out
        dgvx = g2 * dfvx
        dgvy = g2 * dfvy
        dgvz = g2 * dfvz

        dsl = g0 * ds_out + dgs * sr + (dgvx * v1x + dgvy * v1y + dgvz * v1z)
        dsr = g1 * ds_out + dgs * sl + (dgvx * v0x + dgvy * v0y + dgvz * v0z)
        dv0x = g0 * dfvx + sr * dgvx + (v1y * dgvz - v1z * dgvy) - dgs * v1x
        dv0y = g0 * dfvy + sr * dgvy + (v1z * dgvx - v1x * dgvz) - dgs * v1y
        dv0z = g0 * dfvz + sr * dgvz + (v1x * dgvy - v1y * dgvx) - dgs * v1z
        dv1x = g1 * dfvx + sl * dgvx + (dgvy * v0z - dgvz * v0y) - dgs * v0x
        dv1y = g1 * dfvy + sl * dgvy + (dgvz * v0x - dgvx * v0z) - dgs * v0y
        dv1z = g1 * dfvz + sl * dgvz + (dgvx * v0y - dgvy * v0x) - dgs * v0z

        # rotate gradients back: dv_l = R_L^T dv0 ; dv_r = R_R^T dv1
        dvlx = RL0 * dv0x + RL3 * dv0y + RL6 * dv0z
        dvly = RL1 * dv0x + RL4 * dv0y + RL7 * dv0z
        dvlz = RL2 * dv0x + RL5 * dv0y + RL8 * dv0z
        dvrx = RR0 * dv1x + RR3 * dv1y + RR6 * dv1z
        dvry = RR1 * dv1x + RR4 * dv1y + RR7 * dv1z
        dvrz = RR2 * dv1x + RR5 * dv1y + RR8 * dv1z

        tl.store(dh_l_ptr + base + 0, dsl, mask=mask)
        tl.store(dh_l_ptr + base + 1, dvlx, mask=mask)
        tl.store(dh_l_ptr + base + 2, dvly, mask=mask)
        tl.store(dh_l_ptr + base + 3, dvlz, mask=mask)
        tl.store(dh_r_ptr + base + 0, dsr, mask=mask)
        tl.store(dh_r_ptr + base + 1, dvrx, mask=mask)
        tl.store(dh_r_ptr + base + 2, dvry, mask=mask)
        tl.store(dh_r_ptr + base + 3, dvrz, mask=mask)
        tl.store(dg_ptr + (n * 3 + 0) * nb + b, dg0, mask=mask)
        tl.store(dg_ptr + (n * 3 + 1) * nb + b, dg1, mask=mask)
        tl.store(dg_ptr + (n * 3 + 2) * nb + b, dg2, mask=mask)
        tl.store(dv0_ptr + b3 + 0, dv0x, mask=mask)
        tl.store(dv0_ptr + b3 + 1, dv0y, mask=mask)
        tl.store(dv0_ptr + b3 + 2, dv0z, mask=mask)
        tl.store(dv1_ptr + b3 + 0, dv1x, mask=mask)
        tl.store(dv1_ptr + b3 + 1, dv1y, mask=mask)
        tl.store(dv1_ptr + b3 + 2, dv1z, mask=mask)
        tl.store(fv_ptr + b3 + 0, fvx, mask=mask)
        tl.store(fv_ptr + b3 + 1, fvy, mask=mask)
        tl.store(fv_ptr + b3 + 2, fvz, mask=mask)

    @triton.jit
    def _act_fwd_kernel(x_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # tl.tanh is available in recent triton; exp2-based fallback keeps
        # this portable across triton versions without a version probe.
        e = tl.exp(2.0 * x)
        th = (e - 1.0) / (e + 1.0)
        tl.store(y_ptr + offs, th + 0.1 * x, mask=mask)

    @triton.jit
    def _act_bwd_kernel(x_ptr, gy_ptr, gx_ptr, total, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        gy = tl.load(gy_ptr + offs, mask=mask, other=0.0)
        e = tl.exp(2.0 * x)
        th = (e - 1.0) / (e + 1.0)
        tl.store(gx_ptr + offs, gy * (1.0 - th * th + 0.1), mask=mask)


class FusedNodeTriton(torch.autograd.Function):
    """Same contract as metal_kernel.FusedNode: saves only the six
    inputs, backward recomputes everything else from them."""

    @staticmethod
    def forward(ctx, h_l, h_r, R_L, R_R, R_O, g):
        if g.dtype != h_l.dtype:
            g = g.to(h_l.dtype)
        if triton_available() and h_l.device.type == 'cuda':
            N, nb, _ = h_l.shape
            h_lc = h_l.contiguous().float()
            h_rc = h_r.contiguous().float()
            Rl = R_L.reshape(nb, 9).float().contiguous()
            Rr = R_R.reshape(nb, 9).float().contiguous()
            Ro = R_O.reshape(nb, 9).float().contiguous()
            gc = g.contiguous().float()
            out = torch.empty_like(h_lc)
            total = N * nb
            grid = (triton.cdiv(total, DEFAULT_BLOCK_SIZE),)
            _node_fwd_kernel[grid](h_lc, h_rc, Rl, Rr, Ro, gc, out,
                                   total, nb, BLOCK_SIZE=DEFAULT_BLOCK_SIZE)
            out = out.to(h_l.dtype)
        else:
            out = _node_fwd_reference(h_l, h_r, R_L, R_R, R_O, g)
        ctx.save_for_backward(h_l, h_r, R_L, R_R, R_O, g)
        return out

    @staticmethod
    def backward(ctx, gout):
        h_l, h_r, R_L, R_R, R_O, g = ctx.saved_tensors
        if triton_available() and h_l.device.type == 'cuda':
            N, nb, _ = h_l.shape
            h_lc = h_l.contiguous().float()
            h_rc = h_r.contiguous().float()
            Rl = R_L.reshape(nb, 9).float().contiguous()
            Rr = R_R.reshape(nb, 9).float().contiguous()
            Ro = R_O.reshape(nb, 9).float().contiguous()
            gc = g.contiguous().float()
            goutc = gout.contiguous().to(h_l.dtype).float()
            dh_l = torch.empty_like(h_lc)
            dh_r = torch.empty_like(h_rc)
            dg = torch.empty_like(gc)
            dv0 = torch.empty(N, nb, 3, device=h_l.device, dtype=torch.float32)
            dv1 = torch.empty(N, nb, 3, device=h_l.device, dtype=torch.float32)
            fv = torch.empty(N, nb, 3, device=h_l.device, dtype=torch.float32)
            total = N * nb
            grid = (triton.cdiv(total, DEFAULT_BLOCK_SIZE),)
            _node_bwd_kernel[grid](h_lc, h_rc, Rl, Rr, Ro, gc, goutc,
                                   dh_l, dh_r, dg, dv0, dv1, fv,
                                   total, nb, BLOCK_SIZE=DEFAULT_BLOCK_SIZE)
            dh_l = dh_l.to(h_l.dtype)
            dh_r = dh_r.to(h_r.dtype)
            dg = dg.to(g.dtype)
            # Same reasoning as metal_kernel.FusedNode.backward: dv0/dv1/fv
            # come free from the kernel's in-register recompute, so dR_L/
            # dR_R/dR_O reduce to a plain einsum over N here rather than
            # an R_O^{-1}-based recovery (which would require R_O
            # orthogonal, i.e. rot_mode='so3' only -- this path supports
            # rot_mode='free' too).
            dR_L = torch.einsum('nki,nkj->kij', dv0, h_l[..., 1:].float())
            dR_R = torch.einsum('nki,nkj->kij', dv1, h_r[..., 1:].float())
            dR_O = torch.einsum('nki,nkj->kij', gout[..., 1:].float(), fv)
            return dh_l, dh_r, dR_L, dR_R, dR_O, dg
        return _node_bwd_reference(h_l, h_r, R_L, R_R, R_O, g, gout)


class FusedActTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if triton_available() and x.device.type == 'cuda':
            xc = x.contiguous().float()
            y = torch.empty_like(xc)
            total = x.numel()
            grid = (triton.cdiv(total, DEFAULT_BLOCK_SIZE),)
            _act_fwd_kernel[grid](xc, y, total, BLOCK_SIZE=DEFAULT_BLOCK_SIZE)
            return y.to(x.dtype)
        return torch.tanh(x) + 0.1 * x

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        if triton_available() and x.device.type == 'cuda':
            xc = x.contiguous().float()
            gyc = gy.contiguous().to(x.dtype).float()
            gx = torch.empty_like(xc)
            total = x.numel()
            grid = (triton.cdiv(total, DEFAULT_BLOCK_SIZE),)
            _act_bwd_kernel[grid](xc, gyc, gx, total, BLOCK_SIZE=DEFAULT_BLOCK_SIZE)
            return gx.to(x.dtype)
        th = torch.tanh(x)
        return gy * (1.0 - th * th + 0.1)


def fused_node_triton(h_l, h_r, R_L, R_R, R_O, g):
    """h_*: [N, nb, 4]; R_*: [nb, 3, 3]; g: [N, 3, nb] (post-sigmoid
    gates). Returns [N, nb, 4]. CUDA+triton port of
    metal_kernel.fused_node; same contract, same fallback math."""
    return FusedNodeTriton.apply(h_l, h_r, R_L, R_R, R_O, g)


def fused_act_triton(x):
    return FusedActTriton.apply(x)


# ---------------------------------------------------------------------------

def _test():
    torch.manual_seed(0)
    N, nb = 64, 16
    dev = 'cuda' if triton_available() else 'cpu'
    print(f"Testing on {dev} (triton_available={triton_available()}, "
          f"triton_import_error={_TRITON_IMPORT_ERROR})")

    h_l = torch.randn(N, nb, 4, dtype=torch.float64)
    h_r = torch.randn(N, nb, 4, dtype=torch.float64)
    q = torch.randn(2, nb, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    def rot(w, x, y, z):
        return torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)

    R_L, R_R = rot(w[0], x[0], y[0], z[0]), rot(w[1], x[1], y[1], z[1])

    # (1) fallback forward matches metal_kernel.py's own verified reference
    g = torch.rand(N, 3, nb, dtype=torch.float64)
    out = _node_fwd_reference(h_l, h_r, R_L, R_R, R_L, g)
    p0, p1, geo = _fwd_reference(h_l, h_r, R_L, R_R)
    fs = g[:, 0] * p0[..., 0] + g[:, 1] * p1[..., 0] + g[:, 2] * geo[..., 0]
    fv = (g[:, 0].unsqueeze(-1) * p0[..., 1:] + g[:, 1].unsqueeze(-1) * p1[..., 1:]
          + g[:, 2].unsqueeze(-1) * geo[..., 1:])
    ov = torch.einsum('kij,nkj->nki', R_L, fv)
    e1 = max((out[..., 0] - fs).abs().max().item(), (out[..., 1:] - ov).abs().max().item())
    print(f"  (1) fallback vs metal_kernel reference: err {e1:.2e}")
    assert e1 < 1e-12

    # (2) analytic backward (fallback path) vs autograd
    args = [h_l.detach().clone().requires_grad_(True),
            h_r.detach().clone().requires_grad_(True),
            R_L.clone().requires_grad_(True),
            R_R.clone().requires_grad_(True),
            R_L.clone().requires_grad_(True),
            g.clone().requires_grad_(True)]
    out = fused_node_triton(*args)
    out.sin().sum().backward()
    ga = [a.grad.clone() for a in args]
    for a in args:
        a.grad = None
    out2 = _node_fwd_reference(*args)
    out2.sin().sum().backward()
    e2 = max((gr - a.grad).abs().max().item() for gr, a in zip(ga, args))
    print(f"  (2) analytic backward (fallback) vs autograd: err {e2:.2e}")
    assert e2 < 1e-9

    # (3) Triton vs fallback (CUDA-only)
    if triton_available():
        h32l = torch.randn(N, nb, 4, device='cuda', requires_grad=True)
        h32r = torch.randn(N, nb, 4, device='cuda', requires_grad=True)
        g32 = torch.rand(N, 3, nb, device='cuda', requires_grad=True)
        RL32 = R_L.float().to('cuda')
        RR32 = R_R.float().to('cuda')
        out_t = fused_node_triton(h32l, h32r, RL32, RR32, RL32, g32)
        (out_t * 1.7).sum().backward()
        gm = (h32l.grad.cpu().clone(), h32r.grad.cpu().clone(), g32.grad.cpu().clone())
        hl_c = h32l.detach().cpu().requires_grad_(True)
        hr_c = h32r.detach().cpu().requires_grad_(True)
        g_c = g32.detach().cpu().requires_grad_(True)
        out_f = _node_fwd_reference(hl_c, hr_c, RL32.cpu(), RR32.cpu(), RL32.cpu(), g_c)
        (out_f * 1.7).sum().backward()
        e3 = (out_t.detach().cpu() - out_f.detach()).abs().max().item()
        e4 = max((a - b.grad).abs().max().item() for a, b in zip(gm, (hl_c, hr_c, g_c)))
        print(f"  (3) Triton fwd vs fallback: err {e3:.2e}; bwd err {e4:.2e}")
        assert e3 < 1e-4 and e4 < 1e-3

        # non-orthogonal R_O (rot_mode='free'): the case an R_O^{-1}-based
        # fv-recovery trick would get wrong -- this path recomputes fv
        # in-kernel instead, so it must hold here too.
        R_O_free = (R_L + 0.7 * torch.randn(2, nb, 3, 3, dtype=torch.float64)[0]).float().to('cuda')
        ortho_err = (R_O_free.cpu().double().mT @ R_O_free.cpu().double()
                     - torch.eye(3, dtype=torch.float64)).abs().max().item()
        assert ortho_err > 0.1, "R_O_free accidentally close to orthogonal"
        h32l = torch.randn(N, nb, 4, device='cuda', requires_grad=True)
        h32r = torch.randn(N, nb, 4, device='cuda', requires_grad=True)
        g32 = torch.rand(N, 3, nb, device='cuda', requires_grad=True)
        R_O_g = R_O_free.clone().requires_grad_(True)
        out_t = fused_node_triton(h32l, h32r, RL32, RR32, R_O_g, g32)
        (out_t * 1.3).sum().backward()
        gm = (h32l.grad.cpu().clone(), h32r.grad.cpu().clone(),
              g32.grad.cpu().clone(), R_O_g.grad.cpu().clone())
        hl_c = h32l.detach().cpu().requires_grad_(True)
        hr_c = h32r.detach().cpu().requires_grad_(True)
        g_c = g32.detach().cpu().requires_grad_(True)
        R_O_c = R_O_free.detach().cpu().requires_grad_(True)
        out_f = _node_fwd_reference(hl_c, hr_c, RL32.cpu(), RR32.cpu(), R_O_c, g_c)
        (out_f * 1.3).sum().backward()
        e5 = (out_t.detach().cpu() - out_f.detach()).abs().max().item()
        e6 = max((a - b.grad).abs().max().item()
                 for a, b in zip(gm, (hl_c, hr_c, g_c, R_O_c)))
        print(f"  (3b) Triton fwd vs fallback, NON-ORTHOGONAL R_O: "
              f"fwd err {e5:.2e}; bwd err {e6:.2e}")
        assert e5 < 1e-4 and e6 < 1e-3

        # activation
        xc = torch.randn(4096, device='cuda', requires_grad=True)
        yc = fused_act_triton(xc)
        yc.sum().backward()
        xr = xc.detach().cpu().requires_grad_(True)
        yr = torch.tanh(xr) + 0.1 * xr
        yr.sum().backward()
        e7 = max((yc.detach().cpu() - yr.detach()).abs().max().item(),
                 (xc.grad.cpu() - xr.grad).abs().max().item())
        print(f"  (4) fused_act_triton vs eager: err {e7:.2e}")
        assert e7 < 1e-4
    else:
        print("  (3)/(4) SKIPPED: no CUDA+triton on this machine -- "
              "run on your GPU box (e.g. Kaggle)")
    print("  ALL PASS")


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        _test()
    else:
        print(__doc__)
