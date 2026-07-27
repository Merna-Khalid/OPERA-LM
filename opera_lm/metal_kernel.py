"""
Fused Metal kernel for OPERA's compose core: rotate both children +
geometric product in ONE kernel (forward), with a matching backward
kernel that RECOMPUTES intermediates instead of loading them (memory win
on both bytes-moved and bytes-saved).

What is fused (per block b, per row n):
    v0 = R_L[b] @ v_l          s0 = s_l
    v1 = R_R[b] @ v_r          s1 = s_r
    gs = s0*s1 - dot(v0, v1)
    gv = s0*v1 + s1*v0 + cross(v0, v1)
Eager PyTorch performs this as ~8 separate elementwise/einsum passes over
[N, nb, *] tensors (reading+writing each intermediate to memory, and
saving them for backward). The kernel does one read of the inputs and one
write of the outputs; backward reads inputs + output-grads and recomputes
v0, v1 in registers.

Requirements: torch >= 2.7 with torch.mps.compile_shader (prototype API),
Apple Silicon. Falls back to pure PyTorch (identical math) elsewhere --
the fallback is also the reference for the correctness tests.

Run the test suite ON YOUR MAC before training with it:
    python3 opera_metal_kernel.py --test
It checks (1) fallback matches the original v7.7 composition math,
(2) analytic backward matches autograd, and -- if Metal is available --
(3) Metal forward/backward match the fallback bitwise-tolerably.
NOTE: written and math-verified in a CPU container; the Metal execution
path itself is UNTESTED until --test passes on your machine.
"""
import torch

MSL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Layout: h_l, h_r: [N, nb, 4] contiguous; R_L, R_R: [nb, 9] row-major;
// outputs p0, p1, geo: [N, nb, 4]. One thread per (n, b).

kernel void fused_compose_fwd(
    device const float* h_l   [[buffer(0)]],
    device const float* h_r   [[buffer(1)]],
    device const float* R_L   [[buffer(2)]],
    device const float* R_R   [[buffer(3)]],
    device float*       p0    [[buffer(4)]],
    device float*       p1    [[buffer(5)]],
    device float*       geo   [[buffer(6)]],
    constant uint&      total [[buffer(7)]],
    constant uint&      nb    [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total) return;
    uint b = tid % nb;
    uint base = tid * 4;
    uint rb = b * 9;

    float sl = h_l[base + 0];
    float3 vl = float3(h_l[base + 1], h_l[base + 2], h_l[base + 3]);
    float sr = h_r[base + 0];
    float3 vr = float3(h_r[base + 1], h_r[base + 2], h_r[base + 3]);

    float3 v0 = float3(
        R_L[rb+0]*vl.x + R_L[rb+1]*vl.y + R_L[rb+2]*vl.z,
        R_L[rb+3]*vl.x + R_L[rb+4]*vl.y + R_L[rb+5]*vl.z,
        R_L[rb+6]*vl.x + R_L[rb+7]*vl.y + R_L[rb+8]*vl.z);
    float3 v1 = float3(
        R_R[rb+0]*vr.x + R_R[rb+1]*vr.y + R_R[rb+2]*vr.z,
        R_R[rb+3]*vr.x + R_R[rb+4]*vr.y + R_R[rb+5]*vr.z,
        R_R[rb+6]*vr.x + R_R[rb+7]*vr.y + R_R[rb+8]*vr.z);

    float gs = sl * sr - dot(v0, v1);
    float3 gv = sl * v1 + sr * v0 + cross(v0, v1);

    p0[base+0] = sl;  p0[base+1] = v0.x; p0[base+2] = v0.y; p0[base+3] = v0.z;
    p1[base+0] = sr;  p1[base+1] = v1.x; p1[base+2] = v1.y; p1[base+3] = v1.z;
    geo[base+0] = gs; geo[base+1] = gv.x; geo[base+2] = gv.y; geo[base+3] = gv.z;
}

kernel void fused_compose_bwd(
    device const float* h_l   [[buffer(0)]],
    device const float* h_r   [[buffer(1)]],
    device const float* R_L   [[buffer(2)]],
    device const float* R_R   [[buffer(3)]],
    device const float* gp0   [[buffer(4)]],
    device const float* gp1   [[buffer(5)]],
    device const float* ggeo  [[buffer(6)]],
    device float*       dh_l  [[buffer(7)]],
    device float*       dh_r  [[buffer(8)]],
    device float*       dv0o  [[buffer(9)]],   // dL/d(v0) pre-rotation-transpose, for dR
    device float*       dv1o  [[buffer(10)]],
    constant uint&      total [[buffer(11)]],
    constant uint&      nb    [[buffer(12)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total) return;
    uint b = tid % nb;
    uint base = tid * 4;
    uint b3 = tid * 3;
    uint rb = b * 9;

    float sl = h_l[base + 0];
    float3 vl = float3(h_l[base + 1], h_l[base + 2], h_l[base + 3]);
    float sr = h_r[base + 0];
    float3 vr = float3(h_r[base + 1], h_r[base + 2], h_r[base + 3]);

    // recompute rotated children in registers (nothing was saved)
    float3 v0 = float3(
        R_L[rb+0]*vl.x + R_L[rb+1]*vl.y + R_L[rb+2]*vl.z,
        R_L[rb+3]*vl.x + R_L[rb+4]*vl.y + R_L[rb+5]*vl.z,
        R_L[rb+6]*vl.x + R_L[rb+7]*vl.y + R_L[rb+8]*vl.z);
    float3 v1 = float3(
        R_R[rb+0]*vr.x + R_R[rb+1]*vr.y + R_R[rb+2]*vr.z,
        R_R[rb+3]*vr.x + R_R[rb+4]*vr.y + R_R[rb+5]*vr.z,
        R_R[rb+6]*vr.x + R_R[rb+7]*vr.y + R_R[rb+8]*vr.z);

    float gs0 = gp0[base+0];
    float3 gv0d = float3(gp0[base+1], gp0[base+2], gp0[base+3]);
    float gs1 = gp1[base+0];
    float3 gv1d = float3(gp1[base+1], gp1[base+2], gp1[base+3]);
    float ggs = ggeo[base+0];
    float3 ggv = float3(ggeo[base+1], ggeo[base+2], ggeo[base+3]);

    // d gs: gs = sl*sr - dot(v0,v1)
    // d gv: gv = sl*v1 + sr*v0 + v0 x v1
    float dsl = gs0 + ggs * sr + dot(ggv, v1);
    float dsr = gs1 + ggs * sl + dot(ggv, v0);
    float3 dv0 = gv0d + sr * ggv + cross(v1, ggv) - ggs * v1;
    float3 dv1 = gv1d + sl * ggv + cross(ggv, v0) - ggs * v0;

    // rotate gradients back: dv_l = R_L^T dv0 ; dv_r = R_R^T dv1
    float3 dvl = float3(
        R_L[rb+0]*dv0.x + R_L[rb+3]*dv0.y + R_L[rb+6]*dv0.z,
        R_L[rb+1]*dv0.x + R_L[rb+4]*dv0.y + R_L[rb+7]*dv0.z,
        R_L[rb+2]*dv0.x + R_L[rb+5]*dv0.y + R_L[rb+8]*dv0.z);
    float3 dvr = float3(
        R_R[rb+0]*dv1.x + R_R[rb+3]*dv1.y + R_R[rb+6]*dv1.z,
        R_R[rb+1]*dv1.x + R_R[rb+4]*dv1.y + R_R[rb+7]*dv1.z,
        R_R[rb+2]*dv1.x + R_R[rb+5]*dv1.y + R_R[rb+8]*dv1.z);

    dh_l[base+0] = dsl; dh_l[base+1] = dvl.x; dh_l[base+2] = dvl.y; dh_l[base+3] = dvl.z;
    dh_r[base+0] = dsr; dh_r[base+1] = dvr.x; dh_r[base+2] = dvr.y; dh_r[base+3] = dvr.z;
    dv0o[b3+0] = dv0.x; dv0o[b3+1] = dv0.y; dv0o[b3+2] = dv0.z;
    dv1o[b3+0] = dv1.x; dv1o[b3+1] = dv1.y; dv1o[b3+2] = dv1.z;
}
"""

_lib = None


def metal_available():
    return (torch.backends.mps.is_available()
            and hasattr(torch.mps, 'compile_shader'))


def _get_lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(MSL_SOURCE)
    return _lib


def _fwd_reference(h_l, h_r, R_L, R_R):
    """Pure-PyTorch fallback: identical math, used off-Metal and as the
    correctness reference. h_*: [N, nb, 4], R_*: [nb, 3, 3]."""
    s_l, v_l = h_l[..., 0], h_l[..., 1:]
    s_r, v_r = h_r[..., 0], h_r[..., 1:]
    v0 = torch.einsum('kij,nkj->nki', R_L, v_l)
    v1 = torch.einsum('kij,nkj->nki', R_R, v_r)
    dot = (v0 * v1).sum(-1)
    cross = torch.cross(v0, v1, dim=-1)
    gs = s_l * s_r - dot
    gv = s_l.unsqueeze(-1) * v1 + s_r.unsqueeze(-1) * v0 + cross
    p0 = torch.cat([s_l.unsqueeze(-1), v0], dim=-1)
    p1 = torch.cat([s_r.unsqueeze(-1), v1], dim=-1)
    geo = torch.cat([gs.unsqueeze(-1), gv], dim=-1)
    return p0, p1, geo


def _bwd_reference(h_l, h_r, R_L, R_R, gp0, gp1, ggeo):
    s_l, v_l = h_l[..., 0], h_l[..., 1:]
    s_r, v_r = h_r[..., 0], h_r[..., 1:]
    v0 = torch.einsum('kij,nkj->nki', R_L, v_l)
    v1 = torch.einsum('kij,nkj->nki', R_R, v_r)
    ggs, ggv = ggeo[..., 0], ggeo[..., 1:]
    dsl = gp0[..., 0] + ggs * s_r + (ggv * v1).sum(-1)
    dsr = gp1[..., 0] + ggs * s_l + (ggv * v0).sum(-1)
    dv0 = (gp0[..., 1:] + s_r.unsqueeze(-1) * ggv
           + torch.cross(v1, ggv, dim=-1) - ggs.unsqueeze(-1) * v1)
    dv1 = (gp1[..., 1:] + s_l.unsqueeze(-1) * ggv
           + torch.cross(ggv, v0, dim=-1) - ggs.unsqueeze(-1) * v0)
    dvl = torch.einsum('kji,nkj->nki', R_L, dv0)   # R^T
    dvr = torch.einsum('kji,nkj->nki', R_R, dv1)
    dh_l = torch.cat([dsl.unsqueeze(-1), dvl], dim=-1)
    dh_r = torch.cat([dsr.unsqueeze(-1), dvr], dim=-1)
    dR_L = torch.einsum('nki,nkj->kij', dv0, v_l)
    dR_R = torch.einsum('nki,nkj->kij', dv1, v_r)
    return dh_l, dh_r, dR_L, dR_R


class FusedCompose(torch.autograd.Function):
    """Saves ONLY the four inputs; backward recomputes v0/v1 in-kernel."""

    @staticmethod
    def forward(ctx, h_l, h_r, R_L, R_R):
        ctx.save_for_backward(h_l, h_r, R_L, R_R)
        if metal_available() and h_l.device.type == 'mps':
            N, nb, _ = h_l.shape
            lib = _get_lib()
            p0 = torch.empty_like(h_l)
            p1 = torch.empty_like(h_l)
            geo = torch.empty_like(h_l)
            Rl = R_L.reshape(nb, 9).contiguous()
            Rr = R_R.reshape(nb, 9).contiguous()
            lib.fused_compose_fwd(h_l.contiguous(), h_r.contiguous(),
                                  Rl, Rr, p0, p1, geo,
                                  N * nb, nb,
                                  threads=N * nb)
            return p0, p1, geo
        return _fwd_reference(h_l, h_r, R_L, R_R)

    @staticmethod
    def backward(ctx, gp0, gp1, ggeo):
        h_l, h_r, R_L, R_R = ctx.saved_tensors
        if metal_available() and h_l.device.type == 'mps':
            N, nb, _ = h_l.shape
            lib = _get_lib()
            dh_l = torch.empty_like(h_l)
            dh_r = torch.empty_like(h_r)
            dv0 = torch.empty(N, nb, 3, device=h_l.device, dtype=h_l.dtype)
            dv1 = torch.empty(N, nb, 3, device=h_l.device, dtype=h_l.dtype)
            Rl = R_L.reshape(nb, 9).contiguous()
            Rr = R_R.reshape(nb, 9).contiguous()
            lib.fused_compose_bwd(h_l.contiguous(), h_r.contiguous(), Rl, Rr,
                                  gp0.contiguous(), gp1.contiguous(),
                                  ggeo.contiguous(),
                                  dh_l, dh_r, dv0, dv1,
                                  N * nb, nb, threads=N * nb)
            s_lv = h_l[..., 1:]
            s_rv = h_r[..., 1:]
            dR_L = torch.einsum('nki,nkj->kij', dv0, s_lv)
            dR_R = torch.einsum('nki,nkj->kij', dv1, s_rv)
            return dh_l, dh_r, dR_L, dR_R
        return _bwd_reference(h_l, h_r, R_L, R_R, gp0, gp1, ggeo)


def fused_compose(h_l, h_r, R_L, R_R):
    """h_l, h_r: [N, nb, 4]; R_L, R_R: [nb, 3, 3] ->
    p0, p1, geo: [N, nb, 4] each (scalar in channel 0)."""
    return FusedCompose.apply(h_l, h_r, R_L, R_R)



# ===========================================================================
# KERNEL V2: full compose node (pre-norm) in one kernel.
# Fuses: rotate children + geometric product + gated combine + output
# rotation. Eager keeps only the gate matmul (dense, MPS-native) and the
# norm (native fused op). Backward recomputes ALL intermediates in
# registers; the only saved tensors are the kernel's inputs.
# ===========================================================================

MSL_SOURCE_V2 = r"""
#include <metal_stdlib>
using namespace metal;

// Grid-stride pattern: fixed-size grid, each thread loops over its share.
// FTYPE variants: float (fp32) and half I/O with float accumulation.

#define NODE_FWD(NAME, FTYPE)                                              \
kernel void NAME(                                                          \
    device const FTYPE* h_l  [[buffer(0)]],                                \
    device const FTYPE* h_r  [[buffer(1)]],                                \
    device const float* R_L  [[buffer(2)]],                                \
    device const float* R_R  [[buffer(3)]],                                \
    device const float* R_O  [[buffer(4)]],                                \
    device const FTYPE* g    [[buffer(5)]],                                \
    device FTYPE*       out  [[buffer(6)]],                                \
    constant uint&      total [[buffer(7)]],                               \
    constant uint&      nb   [[buffer(8)]],                                \
    uint tid [[thread_position_in_grid]],                                  \
    uint gsz [[threads_per_grid]])                                         \
{                                                                          \
  for (uint t = tid; t < total; t += gsz) {                                \
    uint n = t / nb;                                                       \
    uint b = t % nb;                                                       \
    uint base = t * 4;                                                     \
    uint rb = b * 9;                                                       \
    float sl = float(h_l[base+0]);                                         \
    float3 vl = float3(h_l[base+1], h_l[base+2], h_l[base+3]);             \
    float sr = float(h_r[base+0]);                                         \
    float3 vr = float3(h_r[base+1], h_r[base+2], h_r[base+3]);             \
    float3 v0 = float3(R_L[rb+0]*vl.x + R_L[rb+1]*vl.y + R_L[rb+2]*vl.z,   \
                       R_L[rb+3]*vl.x + R_L[rb+4]*vl.y + R_L[rb+5]*vl.z,   \
                       R_L[rb+6]*vl.x + R_L[rb+7]*vl.y + R_L[rb+8]*vl.z);  \
    float3 v1 = float3(R_R[rb+0]*vr.x + R_R[rb+1]*vr.y + R_R[rb+2]*vr.z,   \
                       R_R[rb+3]*vr.x + R_R[rb+4]*vr.y + R_R[rb+5]*vr.z,   \
                       R_R[rb+6]*vr.x + R_R[rb+7]*vr.y + R_R[rb+8]*vr.z);  \
    float gs = sl * sr - dot(v0, v1);                                      \
    float3 gv = sl * v1 + sr * v0 + cross(v0, v1);                         \
    float g0 = float(g[(n*3 + 0)*nb + b]);                                 \
    float g1 = float(g[(n*3 + 1)*nb + b]);                                 \
    float g2 = float(g[(n*3 + 2)*nb + b]);                                 \
    float fs = g0*sl + g1*sr + g2*gs;                                      \
    float3 fv = g0*v0 + g1*v1 + g2*gv;                                     \
    float3 ov = float3(R_O[rb+0]*fv.x + R_O[rb+1]*fv.y + R_O[rb+2]*fv.z,   \
                       R_O[rb+3]*fv.x + R_O[rb+4]*fv.y + R_O[rb+5]*fv.z,   \
                       R_O[rb+6]*fv.x + R_O[rb+7]*fv.y + R_O[rb+8]*fv.z);  \
    out[base+0] = FTYPE(fs); out[base+1] = FTYPE(ov.x);                    \
    out[base+2] = FTYPE(ov.y); out[base+3] = FTYPE(ov.z);                  \
  }                                                                        \
}

NODE_FWD(fused_node_fwd, float)
NODE_FWD(fused_node_fwd_h, half)

#define NODE_BWD(NAME, FTYPE)                                              \
kernel void NAME(                                                          \
    device const FTYPE* h_l  [[buffer(0)]],                                \
    device const FTYPE* h_r  [[buffer(1)]],                                \
    device const float* R_L  [[buffer(2)]],                                \
    device const float* R_R  [[buffer(3)]],                                \
    device const float* R_O  [[buffer(4)]],                                \
    device const FTYPE* g    [[buffer(5)]],                                \
    device const FTYPE* gout [[buffer(6)]],                                \
    device FTYPE*       dh_l [[buffer(7)]],                                \
    device FTYPE*       dh_r [[buffer(8)]],                                \
    device FTYPE*       dg   [[buffer(9)]],                                \
    device float*       dv0o [[buffer(10)]],                               \
    device float*       dv1o [[buffer(11)]],                               \
    constant uint&      total [[buffer(12)]],                              \
    constant uint&      nb   [[buffer(13)]],                               \
    uint tid [[thread_position_in_grid]],                                  \
    uint gsz [[threads_per_grid]])                                         \
{                                                                          \
  for (uint t = tid; t < total; t += gsz) {                                \
    uint n = t / nb;                                                       \
    uint b = t % nb;                                                       \
    uint base = t * 4;                                                     \
    uint rb = b * 9;                                                       \
    float sl = float(h_l[base+0]);                                         \
    float3 vl = float3(h_l[base+1], h_l[base+2], h_l[base+3]);             \
    float sr = float(h_r[base+0]);                                         \
    float3 vr = float3(h_r[base+1], h_r[base+2], h_r[base+3]);             \
    float3 v0 = float3(R_L[rb+0]*vl.x + R_L[rb+1]*vl.y + R_L[rb+2]*vl.z,   \
                       R_L[rb+3]*vl.x + R_L[rb+4]*vl.y + R_L[rb+5]*vl.z,   \
                       R_L[rb+6]*vl.x + R_L[rb+7]*vl.y + R_L[rb+8]*vl.z);  \
    float3 v1 = float3(R_R[rb+0]*vr.x + R_R[rb+1]*vr.y + R_R[rb+2]*vr.z,   \
                       R_R[rb+3]*vr.x + R_R[rb+4]*vr.y + R_R[rb+5]*vr.z,   \
                       R_R[rb+6]*vr.x + R_R[rb+7]*vr.y + R_R[rb+8]*vr.z);  \
    float gs = sl * sr - dot(v0, v1);                                      \
    float3 gv = sl * v1 + sr * v0 + cross(v0, v1);                         \
    float g0 = float(g[(n*3 + 0)*nb + b]);                                 \
    float g1 = float(g[(n*3 + 1)*nb + b]);                                 \
    float g2 = float(g[(n*3 + 2)*nb + b]);                                 \
    float ds_out = float(gout[base+0]);                                    \
    float3 dov = float3(gout[base+1], gout[base+2], gout[base+3]);         \
    float3 dfv = float3(R_O[rb+0]*dov.x + R_O[rb+3]*dov.y + R_O[rb+6]*dov.z,\
                        R_O[rb+1]*dov.x + R_O[rb+4]*dov.y + R_O[rb+7]*dov.z,\
                        R_O[rb+2]*dov.x + R_O[rb+5]*dov.y + R_O[rb+8]*dov.z);\
    float dg0 = ds_out*sl + dot(dfv, v0);                                  \
    float dg1 = ds_out*sr + dot(dfv, v1);                                  \
    float dg2 = ds_out*gs + dot(dfv, gv);                                  \
    float dgs = g2 * ds_out;                                               \
    float3 dgv = g2 * dfv;                                                 \
    float dsl = g0*ds_out + dgs*sr + dot(dgv, v1);                         \
    float dsr = g1*ds_out + dgs*sl + dot(dgv, v0);                         \
    float3 dv0 = g0*dfv + sr*dgv + cross(v1, dgv) - dgs*v1;                \
    float3 dv1 = g1*dfv + sl*dgv + cross(dgv, v0) - dgs*v0;                \
    float3 dvl = float3(R_L[rb+0]*dv0.x + R_L[rb+3]*dv0.y + R_L[rb+6]*dv0.z,\
                        R_L[rb+1]*dv0.x + R_L[rb+4]*dv0.y + R_L[rb+7]*dv0.z,\
                        R_L[rb+2]*dv0.x + R_L[rb+5]*dv0.y + R_L[rb+8]*dv0.z);\
    float3 dvr = float3(R_R[rb+0]*dv1.x + R_R[rb+3]*dv1.y + R_R[rb+6]*dv1.z,\
                        R_R[rb+1]*dv1.x + R_R[rb+4]*dv1.y + R_R[rb+7]*dv1.z,\
                        R_R[rb+2]*dv1.x + R_R[rb+5]*dv1.y + R_R[rb+8]*dv1.z);\
    dh_l[base+0] = FTYPE(dsl); dh_l[base+1] = FTYPE(dvl.x);                \
    dh_l[base+2] = FTYPE(dvl.y); dh_l[base+3] = FTYPE(dvl.z);              \
    dh_r[base+0] = FTYPE(dsr); dh_r[base+1] = FTYPE(dvr.x);                \
    dh_r[base+2] = FTYPE(dvr.y); dh_r[base+3] = FTYPE(dvr.z);              \
    dg[(n*3 + 0)*nb + b] = FTYPE(dg0);                                     \
    dg[(n*3 + 1)*nb + b] = FTYPE(dg1);                                     \
    dg[(n*3 + 2)*nb + b] = FTYPE(dg2);                                     \
    uint b3 = t * 3;                                                       \
    dv0o[b3+0] = dv0.x; dv0o[b3+1] = dv0.y; dv0o[b3+2] = dv0.z;            \
    dv1o[b3+0] = dv1.x; dv1o[b3+1] = dv1.y; dv1o[b3+2] = dv1.z;            \
  }                                                                        \
}

NODE_BWD(fused_node_bwd, float)
NODE_BWD(fused_node_bwd_h, half)

// fused activation: y = tanh(x) + 0.1x ; backward dy*(1 - tanh(x)^2 + 0.1)
#define ACT(NAME, FTYPE)                                                   \
kernel void NAME(                                                          \
    device const FTYPE* x   [[buffer(0)]],                                 \
    device FTYPE*       y   [[buffer(1)]],                                 \
    constant uint&      total [[buffer(2)]],                               \
    uint tid [[thread_position_in_grid]],                                  \
    uint gsz [[threads_per_grid]])                                         \
{                                                                          \
  for (uint t = tid; t < total; t += gsz) {                                \
    float v = float(x[t]);                                                 \
    y[t] = FTYPE(tanh(v) + 0.1f * v);                                      \
  }                                                                        \
}
ACT(fused_act_fwd, float)
ACT(fused_act_fwd_h, half)

#define ACTB(NAME, FTYPE)                                                  \
kernel void NAME(                                                          \
    device const FTYPE* x   [[buffer(0)]],                                 \
    device const FTYPE* gy  [[buffer(1)]],                                 \
    device FTYPE*       gx  [[buffer(2)]],                                 \
    constant uint&      total [[buffer(3)]],                               \
    uint tid [[thread_position_in_grid]],                                  \
    uint gsz [[threads_per_grid]])                                         \
{                                                                          \
  for (uint t = tid; t < total; t += gsz) {                                \
    float th = tanh(float(x[t]));                                          \
    gx[t] = FTYPE(float(gy[t]) * (1.0f - th*th + 0.1f));                   \
  }                                                                        \
}
ACTB(fused_act_bwd, float)
ACTB(fused_act_bwd_h, half)
"""

_lib_v2 = None


def _get_lib_v2():
    global _lib_v2
    if _lib_v2 is None:
        _lib_v2 = torch.mps.compile_shader(MSL_SOURCE_V2)
    return _lib_v2


def _node_fwd_reference(h_l, h_r, R_L, R_R, R_O, g):
    s_l, v_l = h_l[..., 0], h_l[..., 1:]
    s_r, v_r = h_r[..., 0], h_r[..., 1:]
    v0 = torch.einsum('kij,nkj->nki', R_L, v_l)
    v1 = torch.einsum('kij,nkj->nki', R_R, v_r)
    gs = s_l * s_r - (v0 * v1).sum(-1)
    gv = (s_l.unsqueeze(-1) * v1 + s_r.unsqueeze(-1) * v0
          + torch.cross(v0, v1, dim=-1))
    g0, g1, g2 = g[:, 0, :], g[:, 1, :], g[:, 2, :]
    fs = g0 * s_l + g1 * s_r + g2 * gs
    fv = (g0.unsqueeze(-1) * v0 + g1.unsqueeze(-1) * v1
          + g2.unsqueeze(-1) * gv)
    ov = torch.einsum('kij,nkj->nki', R_O, fv)
    return torch.cat([fs.unsqueeze(-1), ov], dim=-1)


def _node_bwd_reference(h_l, h_r, R_L, R_R, R_O, g, gout):
    s_l, v_l = h_l[..., 0], h_l[..., 1:]
    s_r, v_r = h_r[..., 0], h_r[..., 1:]
    v0 = torch.einsum('kij,nkj->nki', R_L, v_l)
    v1 = torch.einsum('kij,nkj->nki', R_R, v_r)
    gs = s_l * s_r - (v0 * v1).sum(-1)
    gv = (s_l.unsqueeze(-1) * v1 + s_r.unsqueeze(-1) * v0
          + torch.cross(v0, v1, dim=-1))
    g0, g1, g2 = g[:, 0, :], g[:, 1, :], g[:, 2, :]
    fv = (g0.unsqueeze(-1) * v0 + g1.unsqueeze(-1) * v1
          + g2.unsqueeze(-1) * gv)
    ds_out = gout[..., 0]
    dov = gout[..., 1:]
    dfv = torch.einsum('kji,nkj->nki', R_O, dov)
    dg0 = ds_out * s_l + (dfv * v0).sum(-1)
    dg1 = ds_out * s_r + (dfv * v1).sum(-1)
    dg2 = ds_out * gs + (dfv * gv).sum(-1)
    dgs = g2 * ds_out
    dgv = g2.unsqueeze(-1) * dfv
    dsl = g0 * ds_out + dgs * s_r + (dgv * v1).sum(-1)
    dsr = g1 * ds_out + dgs * s_l + (dgv * v0).sum(-1)
    dv0 = (g0.unsqueeze(-1) * dfv + s_r.unsqueeze(-1) * dgv
           + torch.cross(v1, dgv, dim=-1) - dgs.unsqueeze(-1) * v1)
    dv1 = (g1.unsqueeze(-1) * dfv + s_l.unsqueeze(-1) * dgv
           + torch.cross(dgv, v0, dim=-1) - dgs.unsqueeze(-1) * v0)
    dvl = torch.einsum('kji,nkj->nki', R_L, dv0)
    dvr = torch.einsum('kji,nkj->nki', R_R, dv1)
    dh_l = torch.cat([dsl.unsqueeze(-1), dvl], dim=-1)
    dh_r = torch.cat([dsr.unsqueeze(-1), dvr], dim=-1)
    dg = torch.stack([dg0, dg1, dg2], dim=1)
    dR_L = torch.einsum('nki,nkj->kij', dv0, v_l)
    dR_R = torch.einsum('nki,nkj->kij', dv1, v_r)
    dR_O = torch.einsum('nki,nkj->kij', dov, fv)
    return dh_l, dh_r, dR_L, dR_R, dR_O, dg


GRID_CAP = 1 << 20  # grid-stride: dispatch at most ~1M threads


class FusedNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_l, h_r, R_L, R_R, R_O, g):
        # dtype unification: under autocast the gate matmul emits fp16
        # while LN keeps states fp32; a mixed call would make the kernel
        # read the wrong element width (garbage + OOB). Promote to the
        # states' dtype.
        if g.dtype != h_l.dtype:
            g = g.to(h_l.dtype)
        out = None
        if metal_available() and h_l.device.type == 'mps':
            N, nb, _ = h_l.shape
            lib = _get_lib_v2()
            out = torch.empty_like(h_l)
            fwd = lib.fused_node_fwd_h if h_l.dtype == torch.float16 else lib.fused_node_fwd
            fwd(h_l.contiguous(), h_r.contiguous(),
                R_L.reshape(nb, 9).float().contiguous(),
                R_R.reshape(nb, 9).float().contiguous(),
                R_O.reshape(nb, 9).float().contiguous(),
                g.contiguous(), out, N * nb, nb,
                threads=min(N * nb, GRID_CAP))
        else:
            out = _node_fwd_reference(h_l, h_r, R_L, R_R, R_O, g)
        # save output too: fv is recovered as R_O^T @ out.v in backward
        # (rotations are orthogonal), dropping the fv/dv0/dv1 aux buffers.
        ctx.save_for_backward(h_l, h_r, R_L, R_R, R_O, g, out)
        return out

    @staticmethod
    def backward(ctx, gout):
        h_l, h_r, R_L, R_R, R_O, g, out = ctx.saved_tensors
        if metal_available() and h_l.device.type == 'mps':
            N, nb, _ = h_l.shape
            lib = _get_lib_v2()
            dh_l = torch.empty_like(h_l)
            dh_r = torch.empty_like(h_r)
            dg = torch.empty_like(g)
            bwd = lib.fused_node_bwd_h if h_l.dtype == torch.float16 else lib.fused_node_bwd
            if gout.dtype != h_l.dtype:
                gout = gout.to(h_l.dtype)
            dv0 = torch.empty(N, nb, 3, device=h_l.device, dtype=torch.float32)
            dv1 = torch.empty(N, nb, 3, device=h_l.device, dtype=torch.float32)
            bwd(h_l.contiguous(), h_r.contiguous(),
                R_L.reshape(nb, 9).float().contiguous(),
                R_R.reshape(nb, 9).float().contiguous(),
                R_O.reshape(nb, 9).float().contiguous(),
                g.contiguous(), gout.contiguous(),
                dh_l, dh_r, dg, dv0, dv1, N * nb, nb,
                threads=min(N * nb, GRID_CAP))
            # dv0/dv1 written from registers in-kernel (free); only fv is
            # recovered eagerly, from the SAVED FORWARD OUTPUT (retained
            # by the norm's autograd anyway): fv = R_O^T @ out.v
            fv = torch.einsum('kji,nkj->nki', R_O, out[..., 1:].float())
            dR_L = torch.einsum('nki,nkj->kij', dv0, h_l[..., 1:].float())
            dR_R = torch.einsum('nki,nkj->kij', dv1, h_r[..., 1:].float())
            dR_O = torch.einsum('nki,nkj->kij', gout[..., 1:].float(), fv)
            return dh_l, dh_r, dR_L, dR_R, dR_O, dg
        return _node_bwd_reference(h_l, h_r, R_L, R_R, R_O, g, gout)


class FusedAct(torch.autograd.Function):
    """y = tanh(x) + 0.1x with analytic backward; saves only x."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if metal_available() and x.device.type == 'mps':
            lib = _get_lib_v2()
            y = torch.empty_like(x)
            fwd = lib.fused_act_fwd_h if x.dtype == torch.float16 else lib.fused_act_fwd
            fwd(x.contiguous(), y, x.numel(), threads=min(x.numel(), GRID_CAP))
            return y
        return torch.tanh(x) + 0.1 * x

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        if metal_available() and x.device.type == 'mps':
            lib = _get_lib_v2()
            gx = torch.empty_like(x)
            bwd = lib.fused_act_bwd_h if x.dtype == torch.float16 else lib.fused_act_bwd
            bwd(x.contiguous(), gy.contiguous(), gx, x.numel(),
                threads=min(x.numel(), GRID_CAP))
            return gx
        th = torch.tanh(x)
        return gy * (1.0 - th * th + 0.1)


def fused_act(x):
    return FusedAct.apply(x)


def fused_node(h_l, h_r, R_L, R_R, R_O, g):
    """Full compose node pre-norm. h_*: [N, nb, 4]; R_*: [nb, 3, 3];
    g: [N, 3, nb] (post-sigmoid gates). Returns [N, nb, 4]."""
    return FusedNode.apply(h_l, h_r, R_L, R_R, R_O, g)

# ---------------------------------------------------------------------------

def _test():
    import sys
    torch.manual_seed(0)
    N, nb = 64, 16
    dev = 'mps' if metal_available() else 'cpu'
    print(f"Testing on {dev} (metal_available={metal_available()})")

    # (1) fallback forward matches original v7.7 math
    h_l = torch.randn(N, nb, 4, dtype=torch.float64)
    h_r = torch.randn(N, nb, 4, dtype=torch.float64)
    q = torch.randn(2, nb, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    def rot(w, x, y, z):
        return torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
    R_L, R_R = rot(w[0], x[0], y[0], z[0]), rot(w[1], x[1], y[1], z[1])
    p0, p1, geo = _fwd_reference(h_l, h_r, R_L, R_R)
    # original path
    v0 = torch.einsum('kij,nkj->nki', R_L, h_l[..., 1:])
    v1 = torch.einsum('kij,nkj->nki', R_R, h_r[..., 1:])
    gs = h_l[..., 0]*h_r[..., 0] - (v0*v1).sum(-1)
    gv = (h_l[..., 0].unsqueeze(-1)*v1 + h_r[..., 0].unsqueeze(-1)*v0
          + torch.cross(v0, v1, dim=-1))
    e1 = max((p0[..., 1:]-v0).abs().max().item(),
             (geo[..., 0]-gs).abs().max().item(),
             (geo[..., 1:]-gv).abs().max().item())
    print(f"  (1) fallback vs original math: err {e1:.2e}")
    assert e1 < 1e-12

    # (2) analytic backward vs autograd (float64 gradcheck)
    h_l.requires_grad_(True); h_r.requires_grad_(True)
    R_Lg = R_L.clone().requires_grad_(True)
    R_Rg = R_R.clone().requires_grad_(True)
    p0, p1, geo = fused_compose(h_l, h_r, R_Lg, R_Rg)
    loss = (p0.sin() + p1.cos() + geo.tanh()).sum()
    loss.backward()
    g_analytic = [t.grad.clone() for t in (h_l, h_r, R_Lg, R_Rg)]
    for t in (h_l, h_r, R_Lg, R_Rg):
        t.grad = None
    p0r, p1r, geor = _fwd_reference(h_l, h_r, R_Lg, R_Rg)
    (p0r.sin() + p1r.cos() + geor.tanh()).sum().backward()
    e2 = max((ga - t.grad).abs().max().item()
             for ga, t in zip(g_analytic, (h_l, h_r, R_Lg, R_Rg)))
    print(f"  (2) analytic backward vs autograd: err {e2:.2e}")
    assert e2 < 1e-9

    # (3) Metal vs fallback (only on Mac)
    if metal_available():
        h_l32 = torch.randn(N, nb, 4, device='mps')
        h_r32 = torch.randn(N, nb, 4, device='mps')
        RL32 = R_L.float().to('mps'); RR32 = R_R.float().to('mps')
        outs_m = fused_compose(h_l32, h_r32, RL32, RR32)
        outs_f = _fwd_reference(h_l32.cpu(), h_r32.cpu(), RL32.cpu(), RR32.cpu())
        e3 = max((m.cpu() - f).abs().max().item() for m, f in zip(outs_m, outs_f))
        print(f"  (3) Metal forward vs fallback: err {e3:.2e}")
        assert e3 < 1e-4
        # backward
        h_l32.requires_grad_(True); h_r32.requires_grad_(True)
        p0, p1, geo = fused_compose(h_l32, h_r32, RL32, RR32)
        (p0.sum() + 2*p1.sum() + 3*geo.sum()).backward()
        gm = (h_l32.grad.cpu().clone(), h_r32.grad.cpu().clone())
        h_lc = h_l32.detach().cpu().requires_grad_(True)
        h_rc = h_r32.detach().cpu().requires_grad_(True)
        p0, p1, geo = _fwd_reference(h_lc, h_rc, RL32.cpu(), RR32.cpu())
        (p0.sum() + 2*p1.sum() + 3*geo.sum()).backward()
        e4 = max((gm[0]-h_lc.grad).abs().max().item(),
                 (gm[1]-h_rc.grad).abs().max().item())
        print(f"      Metal backward vs fallback: err {e4:.2e}")
        assert e4 < 1e-4
    else:
        print("  (3) SKIPPED: no Metal on this machine -- run on your Mac")

    # (4) V2 full-node: fallback matches composed reference ops
    g = torch.rand(N, 3, nb, dtype=torch.float64)
    out = _node_fwd_reference(h_l.detach(), h_r.detach(), R_L, R_R, R_L, g)
    # independent construction
    p0, p1, geo = _fwd_reference(h_l.detach(), h_r.detach(), R_L, R_R)
    fs = g[:,0]*p0[...,0] + g[:,1]*p1[...,0] + g[:,2]*geo[...,0]
    fv = (g[:,0].unsqueeze(-1)*p0[...,1:] + g[:,1].unsqueeze(-1)*p1[...,1:]
          + g[:,2].unsqueeze(-1)*geo[...,1:])
    ov = torch.einsum('kij,nkj->nki', R_L, fv)
    e5 = max((out[...,0]-fs).abs().max().item(), (out[...,1:]-ov).abs().max().item())
    print(f"  (4) V2 forward vs composed ops: err {e5:.2e}")
    assert e5 < 1e-12

    # (5) V2 analytic backward vs autograd (all six inputs)
    args = [h_l.detach().clone().requires_grad_(True),
            h_r.detach().clone().requires_grad_(True),
            R_L.clone().requires_grad_(True),
            R_R.clone().requires_grad_(True),
            R_L.clone().requires_grad_(True),
            g.clone().requires_grad_(True)]
    out = fused_node(*args)
    (out.sin()).sum().backward()
    ga = [a.grad.clone() for a in args]
    for a in args: a.grad = None
    out2 = _node_fwd_reference(*args)
    (out2.sin()).sum().backward()
    e6 = max((x - a.grad).abs().max().item() for x, a in zip(ga, args))
    print(f"  (5) V2 analytic backward vs autograd: err {e6:.2e}")
    assert e6 < 1e-9

    # (6) V2 Metal vs fallback (Mac only)
    if metal_available():
        h32l = torch.randn(N, nb, 4, device='mps', requires_grad=True)
        h32r = torch.randn(N, nb, 4, device='mps', requires_grad=True)
        g32 = torch.rand(N, 3, nb, device='mps', requires_grad=True)
        RL32 = R_L.float().to('mps'); RR32 = R_R.float().to('mps')
        out_m = fused_node(h32l, h32r, RL32, RR32, RL32, g32)
        (out_m * 1.7).sum().backward()
        gm = (h32l.grad.cpu().clone(), h32r.grad.cpu().clone(), g32.grad.cpu().clone())
        hl_c = h32l.detach().cpu().requires_grad_(True)
        hr_c = h32r.detach().cpu().requires_grad_(True)
        g_c = g32.detach().cpu().requires_grad_(True)
        out_f = _node_fwd_reference(hl_c, hr_c, RL32.cpu(), RR32.cpu(), RL32.cpu(), g_c)
        (out_f * 1.7).sum().backward()
        e7 = (out_m.detach().cpu() - out_f.detach()).abs().max().item()
        e8 = max((a - b.grad).abs().max().item()
                 for a, b in zip(gm, (hl_c, hr_c, g_c)))
        print(f"  (6) V2 Metal fwd vs fallback: err {e7:.2e}; bwd err {e8:.2e}")
        assert e7 < 1e-4 and e8 < 1e-3
    else:
        print("  (6) SKIPPED: V2 Metal check runs on your Mac")

    # (7) V3: fused act fwd/bwd vs eager
    x = torch.randn(500, dtype=torch.float64, requires_grad=True)
    y = fused_act(x); y.sin().sum().backward()
    ga = x.grad.clone(); x.grad = None
    y2 = torch.tanh(x) + 0.1 * x; y2.sin().sum().backward()
    e9 = max((y - y2).abs().max().item(), (ga - x.grad).abs().max().item())
    print(f"  (7) fused act fwd+bwd vs eager: err {e9:.2e}")
    assert e9 < 1e-12

    # (8) V3 Metal: fp32 and fp16 kernels vs fallback (Mac only)
    if metal_available():
        for dt, tol_f, tol_b in [(torch.float32, 1e-4, 1e-3),
                                 (torch.float16, 3e-2, 6e-2)]:
            hl = torch.randn(N, nb, 4, device='mps', dtype=dt, requires_grad=True)
            hr = torch.randn(N, nb, 4, device='mps', dtype=dt, requires_grad=True)
            gg = torch.rand(N, 3, nb, device='mps', dtype=dt, requires_grad=True)
            RL32 = R_L.float().to('mps'); RR32 = R_R.float().to('mps')
            out_m = fused_node(hl, hr, RL32, RR32, RL32, gg)
            out_m.float().sum().backward()
            gm = (hl.grad.float().cpu().clone(), hr.grad.float().cpu().clone(),
                  gg.grad.float().cpu().clone())
            hl_c = hl.detach().float().cpu().requires_grad_(True)
            hr_c = hr.detach().float().cpu().requires_grad_(True)
            gg_c = gg.detach().float().cpu().requires_grad_(True)
            out_f = _node_fwd_reference(hl_c, hr_c, RL32.cpu(), RR32.cpu(),
                                        RL32.cpu(), gg_c)
            out_f.sum().backward()
            ef = (out_m.detach().float().cpu() - out_f.detach()).abs().max().item()
            eb = max((a - b.grad).abs().max().item()
                     for a, b in zip(gm, (hl_c, hr_c, gg_c)))
            print(f"  (8) V3 Metal {dt}: fwd err {ef:.2e} (tol {tol_f}), "
                  f"bwd err {eb:.2e} (tol {tol_b})")
            assert ef < tol_f and eb < tol_b
        # act on metal
        xm = torch.randn(4096, device='mps', requires_grad=True)
        ym = fused_act(xm); ym.sum().backward()
        xc = xm.detach().cpu().requires_grad_(True)
        yc = torch.tanh(xc) + 0.1 * xc; yc.sum().backward()
        e10 = max((ym.cpu() - yc).abs().max().item(),
                  (xm.grad.cpu() - xc.grad).abs().max().item())
        print(f"      fused act Metal vs eager: err {e10:.2e}")
        assert e10 < 1e-4
    else:
        print("  (8) SKIPPED: V3 Metal checks run on your Mac")
    print("  ALL PASS")


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        _test()
    else:
        print(__doc__)
