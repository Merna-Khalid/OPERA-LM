"""Fused Metal kernel for the quaternion holographic memory core.

Replaces the bind -> decay scan -> unbind chain of
OperaSpinorFenwickTree._hmem_read (docs/OPERA_Recall_Research_2026-09-24.md
§5f: that chain is memory-bandwidth bound -- dozens of elementwise kernels
over B x T x n x 4 fp32 tensors with stored intermediates).

Per (batch b, slot s), sequentially over t (M_{-1} = 0, k_{-1} = 0):
    u_t = w_t * (k_{t-1} (x) v_t)          bind (Hamilton product)
    M_t = a_t * M_{t-1} + u_t              decayed superposition
    r_t = conj(k_t) (x) M_t                unbind
One GPU thread per (b, s) carries M in registers; forward writes r and M
(M is the only saved activation). Backward is one reverse pass with the
analytic gradients (quaternion left/right-multiplication matrices satisfy
L(p)^T = L(conj p), R(p)^T = R(conj p)):
    G_t   = k_t (x) g_t + a_{t+1} G_{t+1}                 (dL/dM_t)
    dk_t  = M_t (x) conj(g_t) + w_{t+1} (G_{t+1} (x) conj(v_{t+1}))
    dv_t  = w_t (conj(k_{t-1}) (x) G_t)
    dw_t  = sum_s <G_t, k_{t-1} (x) v_t>
    da_t  = <G_t, M_{t-1}>
Verified against PyTorch autograd of the reference implementation
(selftest test_hmem_kernel). Off-MPS, the reference (identical math) runs.
Quaternion component order (w, x, y, z), matching model.quat_mul.
"""
import torch

MSL = r"""
#include <metal_stdlib>
using namespace metal;

inline float4 qmul(float4 a, float4 b) {
    return float4(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
                  a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                  a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
                  a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}
inline float4 qconj(float4 a) { return float4(a.x, -a.y, -a.z, -a.w); }
inline float4 ld4(device const float* p, uint i) {
    return float4(p[i], p[i + 1], p[i + 2], p[i + 3]);
}
inline void st4(device float* p, uint i, float4 x) {
    p[i] = x.x; p[i + 1] = x.y; p[i + 2] = x.z; p[i + 3] = x.w;
}

// k, v, r, M: [B, T, n, 4]; w: [B, T]; a: [B, T, n]. One thread per (b, s).
kernel void hmem_fwd(
    device const float* k   [[buffer(0)]],
    device const float* v   [[buffer(1)]],
    device const float* w   [[buffer(2)]],
    device const float* a   [[buffer(3)]],
    device float*       r   [[buffer(4)]],
    device float*       Mo  [[buffer(5)]],
    constant uint&      B   [[buffer(6)]],
    constant uint&      T   [[buffer(7)]],
    constant uint&      n   [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= B * n) return;
    uint b = tid / n, s = tid % n;
    float4 M = float4(0.0f);
    float4 kp = float4(0.0f);
    for (uint t = 0; t < T; ++t) {
        uint bt = b * T + t;
        uint i4 = (bt * n + s) * 4;
        float4 kt = ld4(k, i4);
        float4 vt = ld4(v, i4);
        M = a[bt * n + s] * M + w[bt] * qmul(kp, vt);
        st4(Mo, i4, M);
        st4(r, i4, qmul(qconj(kt), M));
        kp = kt;
    }
}

kernel void hmem_bwd(
    device const float* k   [[buffer(0)]],
    device const float* v   [[buffer(1)]],
    device const float* w   [[buffer(2)]],
    device const float* a   [[buffer(3)]],
    device const float* Mo  [[buffer(4)]],
    device const float* g   [[buffer(5)]],
    device float*       dk  [[buffer(6)]],
    device float*       dv  [[buffer(7)]],
    device float*       dwn [[buffer(8)]],
    device float*       da  [[buffer(9)]],
    constant uint&      B   [[buffer(10)]],
    constant uint&      T   [[buffer(11)]],
    constant uint&      n   [[buffer(12)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= B * n) return;
    uint b = tid / n, s = tid % n;
    float4 Gn = float4(0.0f);      // G_{t+1}
    float4 vn = float4(0.0f);      // v_{t+1}
    float wn = 0.0f, an = 0.0f;    // w_{t+1}, a_{t+1}
    for (int t = int(T) - 1; t >= 0; --t) {
        uint bt = b * T + uint(t);
        uint i4 = (bt * n + s) * 4;
        float4 kt = ld4(k, i4);
        float4 gt = ld4(g, i4);
        float4 Mt = ld4(Mo, i4);
        float4 vt = ld4(v, i4);
        float wt = w[bt];
        float at = a[bt * n + s];
        float4 G = qmul(kt, gt) + an * Gn;
        float4 dkt = qmul(Mt, qconj(gt)) + wn * qmul(Gn, qconj(vn));
        float4 Mp = float4(0.0f), kp = float4(0.0f);
        if (t > 0) {
            uint j4 = ((bt - 1) * n + s) * 4;
            Mp = ld4(Mo, j4);
            kp = ld4(k, j4);
        }
        st4(dk, i4, dkt);
        st4(dv, i4, wt * qmul(qconj(kp), G));
        dwn[bt * n + s] = dot(G, qmul(kp, vt));
        da[bt * n + s] = dot(G, Mp);
        Gn = G; vn = vt; wn = wt; an = at;
    }
}
"""

_lib = None


def kernel_available():
    return (torch.backends.mps.is_available()
            and hasattr(torch.mps, 'compile_shader'))


def _get_lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(MSL)
    return _lib


def hmem_core_reference(k, v, w, a):
    """PyTorch reference (identical math): k, v [B,T,n,4] fp32, w [B,T],
    a [B,T,n]. Returns r [B,T,n,4]."""
    from .model import quat_mul, quat_mul_conj_a, _decay_scan
    kprev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], 1)
    bind = quat_mul(kprev, v) * w[..., None, None]
    M = _decay_scan(a.unsqueeze(-1), bind)
    return quat_mul_conj_a(k, M)


class HmemCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, w, a):
        B, T, n, _ = k.shape
        k, v, w, a = (x.float().contiguous() for x in (k, v, w, a))
        lib = _get_lib()
        r = torch.empty_like(k)
        M = torch.empty_like(k)
        lib.hmem_fwd(k, v, w, a, r, M, B, T, n, threads=B * n)
        ctx.save_for_backward(k, v, w, a, M)
        return r

    @staticmethod
    def backward(ctx, g):
        k, v, w, a, M = ctx.saved_tensors
        B, T, n, _ = k.shape
        lib = _get_lib()
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        dwn, da = torch.empty_like(a), torch.empty_like(a)
        lib.hmem_bwd(k, v, w, a, M, g.float().contiguous(), dk, dv, dwn, da,
                     B, T, n, threads=B * n)
        return dk, dv, dwn.sum(-1), da


def hmem_core(k, v, w, a):
    """Fused holographic memory core on MPS; reference elsewhere."""
    if k.device.type == 'mps' and kernel_available():
        return HmemCore.apply(k, v, w, a)
    return hmem_core_reference(k, v, w, a)


# ---------------------------------------------------------------------------
# Fused v2: the kernel reads the RAW projection y = x W^T + b (bf16 or fp32,
# no casts, no intermediates) and does key normalization, the write/forget
# sigmoids and the decay itself. Channel layout of y [B, T, C]:
#     [0, 4n) raw keys | [4n, 8n) values | 8n write logit | [8n+1, 9n+1)
#     forget logits (mode 'gated' only; the per-slot decay logit L_s is
#     added inside the kernel)
# mode 0 'none' (a = 1), 1 'fixed' (a = sigmoid(L_s)), 2 'gated'
# (a = sigmoid(y_f + L_s)). Same math as model._hmem_parts + hmem_core:
# k = kr / max(|kr|, 1e-12) (F.normalize), w = sigmoid(y_w).
# Backward returns dy (in y's dtype) and dL; the write-logit channel is the
# slot-sum of per-slot partials, reduced in PyTorch.
# ---------------------------------------------------------------------------
MSL2_TMPL = r"""
#include <metal_stdlib>
using namespace metal;
typedef TIN tin;

inline float4 qmul(float4 a, float4 b) {
    return float4(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
                  a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                  a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
                  a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}
inline float4 qconj(float4 a) { return float4(a.x, -a.y, -a.z, -a.w); }
inline float4 ldy(device const tin* p, uint i) {
    return float4(float(p[i]), float(p[i + 1]), float(p[i + 2]), float(p[i + 3]));
}
inline float4 ld4(device const float* p, uint i) {
    return float4(p[i], p[i + 1], p[i + 2], p[i + 3]);
}
inline void st4(device float* p, uint i, float4 x) {
    p[i] = x.x; p[i + 1] = x.y; p[i + 2] = x.z; p[i + 3] = x.w;
}
inline float sigm(float z) { return 1.0f / (1.0f + exp(-z)); }

kernel void hmem2_fwd(
    device const tin*   y   [[buffer(0)]],
    device const float* L   [[buffer(1)]],
    device float*       r   [[buffer(2)]],
    device float*       Mo  [[buffer(3)]],
    constant uint&      B   [[buffer(4)]],
    constant uint&      T   [[buffer(5)]],
    constant uint&      n   [[buffer(6)]],
    constant uint&      C   [[buffer(7)]],
    constant uint&      mode [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= B * n) return;
    uint b = tid / n, s = tid % n;
    float Ls = (mode > 0) ? L[s] : 0.0f;
    float afix = sigm(Ls);
    float4 M = float4(0.0f), kp = float4(0.0f);
    for (uint t = 0; t < T; ++t) {
        uint bt = b * T + t;
        uint yb = bt * C;
        float4 kr = ldy(y, yb + 4 * s);
        float4 kt = kr / max(length(kr), 1e-12f);
        float4 vt = ldy(y, yb + 4 * n + 4 * s);
        float wt = sigm(float(y[yb + 8 * n]));
        float at = (mode == 2) ? sigm(float(y[yb + 8 * n + 1 + s]) + Ls)
                               : ((mode == 1) ? afix : 1.0f);
        M = at * M + wt * qmul(kp, vt);
        uint i4 = (bt * n + s) * 4;
        st4(Mo, i4, M);
        st4(r, i4, qmul(qconj(kt), M));
        kp = kt;
    }
}

kernel void hmem2_bwd(
    device const tin*   y   [[buffer(0)]],
    device const float* L   [[buffer(1)]],
    device const float* Mo  [[buffer(2)]],
    device const float* g   [[buffer(3)]],
    device float*       dy  [[buffer(4)]],
    device float*       dwp [[buffer(5)]],
    device float*       dap [[buffer(6)]],
    constant uint&      B   [[buffer(7)]],
    constant uint&      T   [[buffer(8)]],
    constant uint&      n   [[buffer(9)]],
    constant uint&      C   [[buffer(10)]],
    constant uint&      mode [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= B * n) return;
    uint b = tid / n, s = tid % n;
    float Ls = (mode > 0) ? L[s] : 0.0f;
    float afix = sigm(Ls);
    float4 Gn = float4(0.0f), vn = float4(0.0f);
    float wn = 0.0f, an = 0.0f;
    // k_t for the current t (computed at the previous, later iteration
    // as its k_{t-1}); start: compute k_{T-1} directly
    uint yl = (b * T + (T - 1)) * C;
    float4 krc = ldy(y, yl + 4 * s);
    float nc = length(krc);
    float4 kc = krc / max(nc, 1e-12f);
    for (int t = int(T) - 1; t >= 0; --t) {
        uint bt = b * T + uint(t);
        uint yb = bt * C;
        uint i4 = (bt * n + s) * 4;
        float4 kt = kc; float nt = nc;
        float4 vt = ldy(y, yb + 4 * n + 4 * s);
        float wt = sigm(float(y[yb + 8 * n]));
        float at = (mode == 2) ? sigm(float(y[yb + 8 * n + 1 + s]) + Ls)
                               : ((mode == 1) ? afix : 1.0f);
        float4 gt = ld4(g, i4);
        float4 Mt = ld4(Mo, i4);
        float4 G = qmul(kt, gt) + an * Gn;
        float4 dkt = qmul(Mt, qconj(gt)) + wn * qmul(Gn, qconj(vn));
        float4 Mp = float4(0.0f), kp = float4(0.0f);
        float np_ = 0.0f;
        if (t > 0) {
            uint j4 = ((bt - 1) * n + s) * 4;
            Mp = ld4(Mo, j4);
            float4 krp = ldy(y, (bt - 1) * C + 4 * s);
            np_ = length(krp);
            kp = krp / max(np_, 1e-12f);
        }
        // through the normalization: d kr = (dk - k <k, dk>) / |kr|
        float4 dkr = (nt > 1e-12f) ? (dkt - kt * dot(kt, dkt)) / nt
                                   : dkt / 1e-12f;
        st4(dy, yb + 4 * s, dkr);
        st4(dy, yb + 4 * n + 4 * s, wt * qmul(qconj(kp), G));
        dwp[bt * n + s] = dot(G, qmul(kp, vt)) * wt * (1.0f - wt);
        float dat = dot(G, Mp) * at * (1.0f - at);
        dap[bt * n + s] = dat;
        if (mode == 2) dy[yb + 8 * n + 1 + s] = dat;
        Gn = G; vn = vt; wn = wt; an = at;
        kc = kp; nc = np_;
    }
}
"""

_lib2 = {}


def _get_lib2(dtype):
    if dtype not in _lib2:
        tin = {torch.float32: 'float', torch.bfloat16: 'bfloat',
               torch.float16: 'half'}[dtype]
        _lib2[dtype] = torch.mps.compile_shader(MSL2_TMPL.replace('TIN', tin))
    return _lib2[dtype]


_MODES = {'none': 0, 'fixed': 1, 'gated': 2}


def hmem_fused_reference(y, L, n, mode):
    """PyTorch reference of the fused v2 op (same math, fp32 inside)."""
    yf = y.float()
    k = torch.nn.functional.normalize(
        yf[..., :4 * n].reshape(*y.shape[:-1], n, 4), dim=-1)
    v = yf[..., 4 * n:8 * n].reshape(*y.shape[:-1], n, 4)
    w = torch.sigmoid(yf[..., 8 * n])
    if mode == 'none':
        a = torch.ones_like(k[..., 0])
    elif mode == 'fixed':
        a = torch.sigmoid(L.float()).expand(k.shape[:3])
    else:
        a = torch.sigmoid(yf[..., 8 * n + 1:9 * n + 1] + L.float())
    return hmem_core_reference(k, v, w, a)


class HmemFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, L, n, mode):
        B, T, C = y.shape
        y = y.contiguous()
        Lf = L.float().contiguous()
        lib = _get_lib2(y.dtype)
        r = torch.empty(B, T, n, 4, device=y.device, dtype=torch.float32)
        M = torch.empty_like(r)
        md = _MODES[mode]
        lib.hmem2_fwd(y, Lf, r, M, B, T, n, C, md, threads=B * n)
        ctx.save_for_backward(y, Lf, M)
        ctx.n, ctx.md, ctx.L_dtype = n, md, L.dtype
        return r

    @staticmethod
    def backward(ctx, g):
        y, Lf, M = ctx.saved_tensors
        n, md = ctx.n, ctx.md
        B, T, C = y.shape
        lib = _get_lib2(y.dtype)
        dy = torch.zeros(B, T, C, device=y.device, dtype=torch.float32)
        dwp = torch.empty(B, T, n, device=y.device, dtype=torch.float32)
        dap = torch.empty_like(dwp)
        lib.hmem2_bwd(y, Lf, M, g.float().contiguous(), dy, dwp, dap,
                      B, T, n, C, md, threads=B * n)
        dy[..., 8 * n] = dwp.sum(-1)
        dL = dap.sum((0, 1)).to(ctx.L_dtype) if md > 0 else None
        return dy.to(y.dtype), dL, None, None


def hmem_fused(y, L, n, mode):
    """Holographic memory straight from the raw projection y (see MSL2_TMPL
    for the channel layout). Returns r [B, T, n, 4] fp32. L: per-slot decay
    logits [n] (ignored for mode 'none')."""
    if y.device.type == 'mps' and kernel_available():
        if L is None:
            L = y.new_zeros(n, dtype=torch.float32)
        return HmemFused.apply(y, L, n, mode)
    return hmem_fused_reference(y, L, n, mode)


# ---------------------------------------------------------------------------
# Causal depthwise short convolution with residual (model.hmem_conv):
#     out[b,t,c] = y[b,t,c] + sum_{j<K} w[c,j] * y[b,t-j,c]   (y_{<0} = 0)
# fp32 accumulation, output in y's dtype. Backward:
#     dy[b,t,c] = g[b,t,c] + sum_j w[c,j] * g[b,t+j,c]
#     dw[c,j]   = sum_{b,t} g[b,t,c] * y[b,t-j,c]   (per-chunk partials)
# ---------------------------------------------------------------------------
MSL_CONV_TMPL = r"""
#include <metal_stdlib>
using namespace metal;
typedef TIN tin;

kernel void dwconv_fwd(
    device const tin*   y   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device tin*         o   [[buffer(2)]],
    constant uint&      T   [[buffer(3)]],
    constant uint&      C   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      N   [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint c = i % C, t = (i / C) % T;
    float acc = float(y[i]);
    for (uint j = 0; j < K && j <= t; ++j)
        acc += w[c * K + j] * float(y[i - j * C]);
    o[i] = tin(acc);
}

kernel void dwconv_bwd_x(
    device const float* g   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       dy  [[buffer(2)]],
    constant uint&      T   [[buffer(3)]],
    constant uint&      C   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      N   [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint c = i % C, t = (i / C) % T;
    float acc = g[i];
    for (uint j = 0; j < K && t + j < T; ++j)
        acc += w[c * K + j] * g[i + j * C];
    dy[i] = acc;
}

// one thread per (chunk p, channel c): rows [p*R, (p+1)*R) of the B*T rows
kernel void dwconv_bwd_w(
    device const tin*   y   [[buffer(0)]],
    device const float* g   [[buffer(1)]],
    device float*       dwp [[buffer(2)]],
    constant uint&      T   [[buffer(3)]],
    constant uint&      C   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      NR  [[buffer(6)]],
    constant uint&      R   [[buffer(7)]],
    constant uint&      P   [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P * C) return;
    uint c = i % C, p = i / C;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint r1 = min(NR, (p + 1) * R);
    for (uint row = p * R; row < r1; ++row) {
        uint t = row % T;
        float gv = g[row * C + c];
        for (uint j = 0; j < K && j <= t; ++j)
            acc[j] += gv * float(y[(row - j) * C + c]);
    }
    for (uint j = 0; j < K; ++j) dwp[(p * C + c) * K + j] = acc[j];
}
"""

_libc = {}


def _get_libc(dtype):
    if dtype not in _libc:
        tin = {torch.float32: 'float', torch.bfloat16: 'bfloat',
               torch.float16: 'half'}[dtype]
        _libc[dtype] = torch.mps.compile_shader(MSL_CONV_TMPL.replace('TIN', tin))
    return _libc[dtype]


def causal_dwconv_reference(y, w):
    """y [B, T, C], w [C, K] -> y + causal depthwise conv (fp32 inside)."""
    import torch.nn.functional as F
    yf = y.float()
    out = yf + w[:, 0].float() * yf
    for j in range(1, w.shape[1]):
        out = out + w[:, j].float() * F.pad(yf[:, :-j], (0, 0, j, 0))
    return out.to(y.dtype)


class CausalDWConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, w):
        B, T, C = y.shape
        K = w.shape[1]
        assert K <= 8
        y = y.contiguous()
        wf = w.float().contiguous()
        o = torch.empty_like(y)
        _get_libc(y.dtype).dwconv_fwd(y, wf, o, T, C, K, y.numel(),
                                      threads=y.numel())
        ctx.save_for_backward(y, wf)
        ctx.w_dtype = w.dtype
        return o

    @staticmethod
    def backward(ctx, g):
        y, wf = ctx.saved_tensors
        B, T, C = y.shape
        K = wf.shape[1]
        lib = _get_libc(y.dtype)
        gf = g.float().contiguous()
        dy = torch.empty_like(gf)
        lib.dwconv_bwd_x(gf, wf, dy, T, C, K, gf.numel(), threads=gf.numel())
        NR = B * T
        R = 128
        P = (NR + R - 1) // R
        dwp = torch.empty(P, C, K, device=y.device, dtype=torch.float32)
        lib.dwconv_bwd_w(y, gf, dwp, T, C, K, NR, R, P, threads=P * C)
        return dy.to(y.dtype), dwp.sum(0).to(ctx.w_dtype)


def causal_dwconv(y, w):
    if y.device.type == 'mps' and kernel_available():
        return CausalDWConv.apply(y, w)
    return causal_dwconv_reference(y, w)
