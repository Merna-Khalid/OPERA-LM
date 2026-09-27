"""OperaSpinorFenwickTree: spinor (Cl(3)) Fenwick-tree language model."""
import math
from typing import NamedTuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    # OPT: TF32 for any fp32 matmul outside autocast (norms, probes).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# ============================================================================
# UTILITIES
# ============================================================================

# Spinor-homeostasis anchor table depth (homeo_mode='on'): covers tree
# spans to 2^16 tokens; deeper parents reuse the deepest anchor.
HOMEO_MAX_LEVELS = 16

def sinusoidal_pos_enc(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: (d + 1) // 2][: pe[:, 1::2].shape[1]])
    return pe


def rotor_pos_tables(T, nb, device, base=10000.0):
    """cos/sin tables [T, nb] for rotor PE: block b at position t rotates
    its vector part by angle omega_b * t about the block z-axis, with
    omega_b = base^(-b/nb) (RoPE's frequency schedule). Zero parameters,
    defined for all t, an isometry per block."""
    freqs = base ** (-torch.arange(nb, device=device, dtype=torch.float32) / nb)
    t = torch.arange(T, device=device, dtype=torch.float32)
    ang = torch.outer(t, freqs)                                   # [T, nb]
    return ang.cos(), ang.sin()


def apply_rotor_pe(states, cos, sin, nb):
    """states [B, T, d=4*nb] -> rotate (v_x, v_y) of each block by the
    position angle; v_z and the scalar channel are untouched."""
    B, T, d = states.shape
    h = states.reshape(B, T, nb, 4)
    s = h[..., 0]
    vx, vy, vz = h[..., 1], h[..., 2], h[..., 3]
    c = cos[None, :, :]                                           # [1, T, nb]
    sn = sin[None, :, :]
    vx2 = vx * c - vy * sn
    vy2 = vx * sn + vy * c
    return torch.stack([s, vx2, vy2, vz], dim=-1).reshape(B, T, d)


def fenwick_blocks(T):
    table = []
    max_blocks = 1
    for L in range(1, T + 1):
        blocks = []
        start = 0
        rem = L
        k = rem.bit_length() - 1
        while rem > 0:
            size = 1 << k
            if size <= rem:
                blocks.append((k, start >> k))
                start += size
                rem -= size
            k -= 1
        table.append(blocks)
        max_blocks = max(max_blocks, len(blocks))
    return table, max_blocks


def level_sin_enc(lvl, dk):
    """Deterministic sinusoidal encoding of the Fenwick block LEVEL
    (block span = 2^level). Smooth in level and defined for EVERY level:
    extrapolates to depths never seen in training, zero parameters --
    the same property that made rotor PE and fold-scale extrapolation-
    safe, applied to the attend readout's keys. Base 100 (levels are
    small integers, <= ~30)."""
    pos = lvl.to(torch.float32).unsqueeze(-1)                      # [T,S,1]
    i = torch.arange(0, dk, 2, device=lvl.device, dtype=torch.float32)
    div = torch.exp(-math.log(100.0) * i / dk)
    ang = pos * div                                                # [T,S,dk/2]
    out = torch.zeros(*lvl.shape, dk, device=lvl.device)
    out[..., 0::2] = torch.sin(ang)
    out[..., 1::2] = torch.cos(ang)
    return out


def fold_work_counts(T):
    """Row-compositions in the fold per layer: masked vs compacted.
    Host-side accounting used by the selftest printout."""
    table, max_blocks = fenwick_blocks(T)
    masked = (max_blocks - 1) * T
    compacted = sum(len(blocks) - 1 for blocks in table)
    return masked, compacted, max_blocks


# ============================================================================
# QUATERNION / Cl(3)-EVEN PRIMITIVES (all-real, MPS-safe)
# ============================================================================

def quat_to_rotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two = 2.0
    r00 = 1 - two * (y * y + z * z)
    r01 = two * (x * y - w * z)
    r02 = two * (x * z + w * y)
    r10 = two * (x * y + w * z)
    r11 = 1 - two * (x * x + z * z)
    r12 = two * (y * z - w * x)
    r20 = two * (x * z - w * y)
    r21 = two * (y * z + w * x)
    r22 = 1 - two * (x * x + y * y)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_mul(a, b):
    """Hamilton product per block. a, b: [..., 4] as (w, x, y, z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_sandwich(q_unit, h):
    """Conjugation q h q^-1 for unit q: THE rack operation. Exact isometry
    (|q h q̄| = |h|); fixes the scalar channel; rotates the vector part.
    With x ▷ y := ŷ x ŷ⁻¹ (ŷ = y normalized), self-distributivity
    (x▷y)▷z = (x▷z)▷(y▷z) holds exactly -- verified in the selftest."""
    return quat_mul(quat_mul(q_unit, h), quat_conj(q_unit))


def geometric_product(s0, v0, s1, v1):
    dot = (v0 * v1).sum(dim=-1)
    cross = torch.cross(v0, v1, dim=-1)
    s_out = s0 * s1 - dot
    v_out = s0.unsqueeze(-1) * v1 + s1.unsqueeze(-1) * v0 + cross
    return s_out, v_out


def quat_quotient(s0, v0, s1, v1):
    """q_L (x) q_R^{-1} per block: the relative transform between
    siblings -- the node's only operation that BACKS OUT rather than
    accumulates (quotient arm, node_paths=4). q^{-1} = conj(q)/|q|^2,
    eps-guarded. Note: for near-unit blocks this degenerates toward a
    sign flip on the vector part -- the expected effect is modest; that
    is the bet, stated."""
    n2 = (s1 * s1 + (v1 * v1).sum(dim=-1)).clamp(min=1e-8)
    return geometric_product(s0, v0, s1 / n2,
                             -v1 / n2.unsqueeze(-1))


def relative_lock(s0, v0, s1, v1, eps=1e-8):
    n0 = torch.sqrt(s0 * s0 + (v0 * v0).sum(-1) + eps)
    n1 = torch.sqrt(s1 * s1 + (v1 * v1).sum(-1) + eps)
    inner = (s0 * s1 + (v0 * v1).sum(-1)) / (n0 * n1)
    return (1.0 - inner * inner).clamp(0.0, 1.0)


def inject_geometry(states, geom, geom_mask, nb, geom_block=0):
    """Overwrite the VECTOR part (dims 1:4) of block `geom_block` of
    `states` ([..., d], d=4*nb) with `geom` ([..., 3]) wherever
    `geom_mask` ([...], bool) is True; the scalar channel and every
    other block are untouched. Used by callers (e.g. a downstream
    spatial-reasoning task) that want a leaf's position to be a literal
    rotor-composable vector instead of a token embedding -- text/LM
    callers never pass geom, so this function does not otherwise exist
    in their compute graph.

    Built via cat/where rather than an in-place write into the reshaped
    view (`h[..., geom_block, 1:] = ...`): `states` is typically
    word_emb's output, which requires grad, and in-place-writing into a
    view of it trips autograd's version-counter check the first time
    this runs inside a backward pass. cat/where has no such hazard.

    geom/geom_mask may be plain numpy arrays (the natural output of a
    non-torch encoding pipeline, e.g. a game-observation tokenizer) --
    coerced to tensors here so every caller gets this for free instead
    of each one remembering to convert."""
    if not torch.is_tensor(geom):
        geom = torch.as_tensor(geom, dtype=states.dtype, device=states.device)
    if not torch.is_tensor(geom_mask):
        geom_mask = torch.as_tensor(geom_mask, dtype=torch.bool,
                                    device=states.device)
    shape = states.shape
    h = states.reshape(*shape[:-1], nb, 4)
    s, v = h[..., geom_block, :1], h[..., geom_block, 1:]
    new_block = torch.cat(
        [s, torch.where(geom_mask.unsqueeze(-1), geom, v)], dim=-1
    ).unsqueeze(-2)
    h = torch.cat([h[..., :geom_block, :], new_block,
                   h[..., geom_block + 1:, :]], dim=-2)
    return h.reshape(*shape)


# ============================================================================
# OPERA-SCAN (v8.5) -- associative affine scan (OPERA_Scan_Arm_Design.md)
# ============================================================================

def affine_compose(q1, b1, q2, b2):
    """The in-scan operator, per block: (q1,b1) ⊕ (q2,b2) =
    (q1 ⊗ q2,  q1 ▷ b2 + b1).  ⊗ = Hamilton product; ▷ = CONFORMAL action
    of the homogeneous quaternion q1 on the paravector b: vector part
    rotated AND scaled by |q1|, scalar part scaled by |q1| too:
        q ▷ (s, v) = |q| · (s, R(q/|q|) v).
    This is the affine/conformal semidirect product (scale-rotation ⋉
    translation): EXACTLY associative (selftested ~1e-6), since
    A(q)(b) = |q|·(b.s, R(q/|q|) b.v) is a group action --
    |q1⊗q2| = |q1||q2| and R is a homomorphism H* -> SO(3).
    v8.6 DECAY (GORU/RG-LRU/GLA/LRU mechanic): leaf magnitudes are
    parameterized in (0,1) (see scan_prefix), so |q| acts as a learned,
    input-dependent multiplicative decay on whatever is composed to the
    leaf's RIGHT -- in the scan's convention (later leaf on the LEFT):
    everything OLDER. A leaf of magnitude s < 1 attenuates all earlier
    contributions by s. q is carried UNNORMALIZED through the scan
    (design §6: in-scan normalization would break float associativity).
    The rotation uses the Rodrigues form v' = v + 2w(u×v) + 2u×(u×v)
    (~15 aten ops) instead of the two-quat_mul sandwich (~60): the scan
    is kernel-launch-bound, so the lean form nearly halves step
    dispatches."""
    q = quat_mul(q1, q2)
    n1 = q1.norm(dim=-1, keepdim=True) + 1e-8       # |q1| = decay factor
    qn = q1 / n1
    w, u = qn[..., :1], qn[..., 1:]
    c = torch.cross(u, b2[..., 1:], dim=-1)
    v = n1 * (b2[..., 1:] + 2.0 * (w * c + torch.cross(u, c, dim=-1)))
    b = torch.cat([b1[..., :1] + n1 * b2[..., :1],  # scalar scaled, not rotated
                   b1[..., 1:] + v], dim=-1)
    return q, b


def associative_scan(q, b):
    """INCLUSIVE prefix scan under affine_compose at ALL positions
    (q, b: [B, T, n, 4]), Hillis-Steele: at level s (s = 1, 2, 4, ...),
    x[i] <- x[i] ⊕ x[i-s] (both operands read from the SAME previous
    generation). O(log T) depth, causal by construction: position k's
    output composes leaves 0..k only. No padding, no index tensors --
    any T (odd / non-power-of-2) natively.
    OPERAND ORDER (v8.6): the LATER leaf is the LEFT operand, so
    prefix_j = l_j ⊕ l_{j-1} ⊕ ... ⊕ l_0. Expanding the law gives
    B_j = b_j + q_j▷b_{j-1} + (q_j q_{j-1})▷b_{j-2} + ... -- token i's
    contribution is transported (rotated AND scaled by the decay
    product Π_{l>i} |q_l|) by all NEWER leaves. This is the RG-LRU/GLA
    direction (h_t = a_t·h_{t-1} + x_t: a leaf's magnitude decays
    everything OLDER); the opposite order would freeze old content and
    attenuate new (caught by the decay-direction selftest). The opposite
    semigroup is still exactly associative.
    DEVIATION from the design note (which specifies Blelloch): Blelloch
    does O(T) compose WORK vs Hillis-Steele's O(T log T), but its
    down-sweep doubles the level count and needs scatter/gather
    (index_copy) traffic per level. The fold is kernel-LAUNCH-bound
    (design §6's own caveat), and measured eager dispatches/step come
    out ~2x lower with Hillis-Steele. Same math up to float
    reassociation; selftested against the naive loop."""
    T = q.shape[1]
    shift = 1
    while shift < T:
        qc, bc = affine_compose(q[:, shift:], b[:, shift:],
                                q[:, :-shift], b[:, :-shift])
        q = torch.cat([q[:, :shift], qc], dim=1)
        b = torch.cat([b[:, :shift], bc], dim=1)
        shift *= 2
    return q, b


class _GradScale(torch.autograd.Function):
    """Identity forward; scales the gradient backward.

    Used to reweight how much each TREE LEVEL contributes to the SHARED
    compose weights. Scaling a level's output gradient directly would be
    wrong -- that gradient also flows down to every level beneath it, so
    the factors would compound geometrically. Scaling the WEIGHTS the
    level sees touches only that level's contribution to dL/dW and
    leaves the input path untouched.
    """

    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


def grad_scale(x, s):
    return x if s == 1.0 else _GradScale.apply(x, s)


# ---------------------------------------------------------------------------
# LEVEL CAPTURE (docs/OPERA_Optimizer_prereg.md). Pass-through autograd
# wrapper that decomposes the gradient of a SCALE-TIED parameter (one
# weight set consumed at every tree level) into its per-level
# contributions. Off by default and completely inert: the wrapper is
# only applied when a capture registry is active, so training without
# it is byte-identical. Two consumers:
#   1. the kill-switch instrument (experiments/level_cosine.py) -- if
#      the levels' whitened gradients already agree, LO-Muon is a no-op
#      by construction and is not built;
#   2. the LO-Muon router -- the optimizer reads per-level gradients
#      from the registry instead of the summed .grad.
# The registry also carries a per-level input-RMS EMA ('rms'), the
# statistic the DERIVED weighting (w_l ~ 1/x_l) normalizes by.
_LEVEL_CAPTURE = None


class _LevelCapture(torch.autograd.Function):
    """Identity forward; backward ACCUMULATES a copy of this call's
    gradient into the registry and passes it through unchanged (the
    parameter's total .grad is therefore untouched)."""

    @staticmethod
    def forward(ctx, x, key):
        ctx.key = key
        return x

    @staticmethod
    def backward(ctx, g):
        if _LEVEL_CAPTURE is not None:
            grads = _LEVEL_CAPTURE['grads']
            buf = grads.get(ctx.key)
            if buf is None:
                grads[ctx.key] = g.detach().clone()
            else:
                buf.add_(g.detach())
        return g, None


def enable_level_capture():
    """Turn per-level gradient capture ON (registry returned for tests)."""
    global _LEVEL_CAPTURE
    _LEVEL_CAPTURE = {'grads': {}, 'rms': {}}
    return _LEVEL_CAPTURE


def disable_level_capture():
    global _LEVEL_CAPTURE
    _LEVEL_CAPTURE = None


def level_capture_dict():
    return _LEVEL_CAPTURE


def _maybe_capture_w(W, layer_idx, level_idx, name='fusion_gate'):
    """Wrap the compose node's fusion weight for the current level."""
    if level_idx is not None and _LEVEL_CAPTURE is not None:
        return _LevelCapture.apply(
            W, (f'{name}.{layer_idx}.weight', level_idx))
    return W


def _decay_scan(a, b):
    """Inclusive linear recurrence M_t = a_t * M_{t-1} + b_t over dim 1,
    WORK-EFFICIENT (Blelloch-style up-sweep / down-sweep over a binary
    tree -- the same pattern as the downsweep fold): ~2T elementwise work
    instead of Hillis-Steele's T log T. Pairs combine as (A2 A1,
    A2 B1 + B2) (earlier, later). a is kept narrow and broadcast against b
    (e.g. a [B,T,n,1], b [B,T,n,4]). T is padded to a power of two with
    identity elements (a=1, b=0). Exact up to float reassociation
    (selftest: equals the naive recurrence)."""
    T = b.shape[1]
    P = 1 << max(T - 1, 1).bit_length()
    if P != T:
        a = torch.cat([a, a.new_ones(a.shape[0], P - T, *a.shape[2:])], 1)
        b = torch.cat([b, b.new_zeros(b.shape[0], P - T, *b.shape[2:])], 1)
    As, Bs = [a], [b]
    while As[-1].shape[1] > 1:
        A, Bv = As[-1], Bs[-1]
        A1, A2, B1, B2 = A[:, 0::2], A[:, 1::2], Bv[:, 0::2], Bv[:, 1::2]
        As.append(A2 * A1)
        Bs.append(A2 * B1 + B2)
    PA, PB = As[-1], Bs[-1]
    for k in range(len(As) - 2, -1, -1):
        A, Bv = As[k], Bs[k]
        evB = torch.cat([Bv[:, :1], A[:, 2::2] * PB[:, :-1] + Bv[:, 2::2]], 1)
        PB_new = torch.stack([evB, PB], 2).reshape(Bv.shape)
        if k > 0:
            evA = torch.cat([A[:, :1], A[:, 2::2] * PA[:, :-1]], 1)
            PA = torch.stack([evA, PA], 2).reshape(A.shape)
        PB = PB_new
    return PB[:, :T]


def quat_mul_conj_a(a, b):
    """conj(a) (x) b per block without materializing conj(a) -- the
    holographic unbinding (one fewer full-size copy than
    quat_mul(quat_conj(a), b))."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([aw * bw + ax * bx + ay * by + az * bz,
                        aw * bx - ax * bw - ay * bz + az * by,
                        aw * by + ax * bz - ay * bw - az * bx,
                        aw * bz - ax * by + ay * bx - az * bw], dim=-1)


def level_sin_features(level, r, device, dtype):
    """f(l) for the level-conditioned weights: the fixed sinusoidal
    basis, evaluated at one integer level (levels are small ints; the
    basis is defined at EVERY depth, seen or not)."""
    return level_sin_enc(torch.tensor([level], device=device), r)[0].to(dtype)


class OperaOutput(NamedTuple):
    """forward()'s return value: fields absent from a given call (the
    corresponding return_* flag was False) are None rather than the field
    itself missing, so callers never need to know which other flags were
    passed to find their own field's position. A NamedTuple (not a plain
    dataclass) on purpose: DDP's find_unused_parameters gradient sync walks
    the forward output via isinstance(obj, (list, tuple)) to locate
    tensors (torch/nn/parallel/distributed.py's _find_tensors) and does not
    recurse into arbitrary objects, so a plain dataclass would silently
    break DDP training (opera_lm.train.train's ddp=True path)."""
    logits: list
    tree: Optional[list] = None
    levels: Optional[list] = None
    states: Optional[list] = None
    energy: Optional[list] = None


# ============================================================================
# OPERA v7.9 — Spinor Tree
# ============================================================================

class OperaSpinorFenwickTree(nn.Module):
    def __init__(self, vocab_size, d=768, nb=192, num_layers=2, lock_mode='none',
                 tie=False, dropout=0.0, pe_mode='sin',
                 fold_mode='left', fold_rotors='shared', fold_scale=False,
                 norm_mode='layer', act_mode='tanh', node_residual=False,
                 tree_drop=0.0, grad_checkpoint='', use_metal=False,
                 use_triton=False,
                 rot_mode='so3', oam_k=4, oam_charges='auto',
                 oam_phi=0.7853981633974483, oam_shared_gate=False,
                 oam_combine='compose', oam_pair='seq', rack_exitnorm=False,
                 oam_transport='rack', oam_levelgate=False,
                 oam_chan_emb=False, scan_salience=False, scan_decay_bias=-3.0,
                 workspace=False, fold_gate_bias=None,
                 readout_mode='none', readout_max_slots=16,
                 mem_mode='none', mem_dim=128, homeo_mode='off',
                 node_paths=3, fold_adapt='off', fold_bistable='off',
                 bist_rank=0, fold_relax='off', fold_grade=None,
                 level_grad_balance=1.0, level_cond_rank=0,
                 head_mode='fold', resid_mode='blend', fold_gate='shared',
                 fold_h0=False, state_mult=1, state_tie=False,
                 node_mix_rank=0, fold_innov_rank=0, tree_disent_rank=0,
                 fold_impl='compact', lowrank_gain=False, fold_dir='left',
                 future_bag=None, resid_init_scale=None, hmem_nb=0,
                 hmem_decay='none', hmem_conv=0):
        super().__init__()
        assert d == 4 * nb, f"d must equal 4*nb (got d={d}, nb={nb})"
        # STREAM / TREE-WIDTH ARMS (architecture review 2026-09-24; draft
        # prereg docs/OPERA_Stream_prereg.md). All defaults reproduce the
        # incumbent bitwise.
        #   head_mode='stream': the LM head reads LN(current) -- the
        #     residual stream AFTER each layer's blend -- instead of the
        #     fold output. Flags-off, the last layer's cross_mlp and
        #     blend_gate feed nothing (34% of the L=2 incumbent's params
        #     receive no gradient); 'stream' makes them live.
        #   resid_mode='add': current <- current + cross_mlp(prefix), with
        #     a pre-norm on each layer's tree input, instead of the convex
        #     scalar blend (G1: prod(1-g) = 0.02 at L=8). Requires 'stream'
        #     (otherwise the last layer's MLP is dead again).
        #   fold_gate='separate': the left fold gets its own fusion gate,
        #     INITIALIZED AS A COPY of the tree gate (bitwise incumbent at
        #     init, no RNG consumed). The tree composes equal-span siblings
        #     at one level; the fold composes a large accumulator with a
        #     small block, scales mixed within a call.
        #   fold_h0=True: every position's left fold starts from a learned
        #     per-layer accumulator h0 (zero-init), so no readout is a raw
        #     tree node (F6's zero-fold positions) and graded groups get a
        #     trained neutral start instead of zeros. +T compositions/layer.
        #   state_mult=k>1: the tree/fold run at width k*d (k*nb quaternion
        #     slots) behind per-layer in/out projections; embedding, head,
        #     cross_mlp and the stream stay at d. The per-prefix state is
        #     k*d numbers (review item 4: the value path never moves content
        #     between slots, so state width is the capacity knob).
        assert head_mode in ('fold', 'stream')
        assert resid_mode in ('blend', 'add')
        assert resid_mode == 'blend' or head_mode == 'stream', \
            "resid_mode='add' needs head_mode='stream' (else the last " \
            "layer's cross_mlp feeds nothing)"
        assert fold_gate in ('shared', 'separate')
        assert int(state_mult) == state_mult and state_mult >= 1
        state_mult = int(state_mult)
        for _flag, _on in (('fold_gate', fold_gate != 'shared'),
                           ('fold_h0', bool(fold_h0)),
                           ('state_mult', state_mult > 1)):
            if _on:
                assert fold_mode == 'left', \
                    f"{_flag} is implemented for the left fold"
        if state_mult > 1:
            assert (readout_mode == 'none' and mem_mode == 'none'
                    and homeo_mode == 'off' and node_paths == 3
                    and fold_adapt == 'off' and fold_bistable == 'off'
                    and fold_relax == 'off' and level_cond_rank == 0
                    and not fold_scale and fold_rotors == 'shared'), \
                "state_mult > 1 supports the plain left fold (+ fold_grade, " \
                "fold_gate_bias, fold_gate, fold_h0) only"
        # state_tie=True (needs state_mult=k>1): the k copies of each
        #   quaternion slot SHARE the slot's gates and rotations -- the
        #   fusion gate emits 3*nb (not 3*k*nb) channels, tiled k times,
        #   and the rotors are tiled likewise. State grows k-fold while
        #   the gate's output width stays the incumbent's (its input is
        #   the k*d tree state, so its cost grows k-fold, not k^2).
        #   Motivation: M2RNN (arXiv 2603.14360) -- state size, not
        #   non-linearity, is the capacity knob.
        # node_mix_rank=r>0: identity-initialized low-rank cross-slot
        #   value mixing inside the compose node, parent <- parent +
        #   (parent @ V) @ U^T before the node norm, U zero-init. The
        #   incumbent node never moves values between slots (Jacobian:
        #   54-77% block-diagonal energy); MLP-LDRU (arXiv 2605.26035)
        #   shows gated-sum operators at 67.1% vs 100% with full
        #   identity-init value projections. U = 0 -> bitwise incumbent
        #   at init, dL/dU != 0 (interior, no cold gate).
        assert not state_tie or state_mult > 1, \
            "state_tie shares gates across state_mult copies; needs " \
            "state_mult > 1"
        # fold_innov_rank=r>0 (INNOVATION FOLD, predictive coding at the
        #   fold; docs/OPERA_Redundancy_Research_2026-09-24.md §4 2a):
        #   before each left-fold compose, acc <- acc - (nxt @ V) @ U^T --
        #   remove from the old-content accumulator the part the newer
        #   block already predicts, so the parent does not store it twice
        #   and single-copy old content (measured: 4.6 bits in its block,
        #   0 after the fold) has room. U zero-init: bitwise incumbent at
        #   init, interior (dL/dU != 0).
        # tree_disent_rank=r>0 (CAUSAL DISENTANGLER, MERA; §4 2b): at every
        #   tree level, node_j <- node_j + (node_{j-1} @ V) @ U^T using the
        #   RAW left neighbour (no chain). Causal: every prefix that reads
        #   node j already covers node j-1's span; append-only: node j-1
        #   exists when node j completes. U zero-init: bitwise at init.
        # fold_impl='downsweep': the SAME left fold for every prefix,
        #   computed top-down over the tree (Blelloch-style downsweep, cf.
        #   Prefix-Scannable Models, arXiv 2506.10918): E_k[2j] = E_{k+1}[j],
        #   E_k[2j+1] = compose(E_{k+1}[j], node_k[2j]). Every prefix's chain
        #   of composes is identical to the compacted fold's, but shared
        #   across prefixes: ~T composes per layer instead of
        #   sum(popcount-1) (~(T/2) log T; 4097 -> ~1013 at T=1024).
        #   Equal to 'compact' up to float reassociation in batched GEMMs.
        assert fold_impl in ('compact', 'downsweep')
        if fold_impl == 'downsweep':
            assert fold_mode == 'left', "downsweep computes the left fold"
            assert (fold_grade in (None, 'off', '') and not fold_h0
                    and fold_bistable == 'off' and fold_relax == 'off'
                    and readout_mode == 'none' and not fold_scale
                    and fold_adapt == 'off' and fold_rotors == 'shared'), \
                "fold_impl='downsweep' supports the plain left fold " \
                "(+ fold_gate_bias, fold_gate, fold_innov_rank) for now"
        self.fold_impl = fold_impl
        assert fold_innov_rank >= 0 and tree_disent_rank >= 0
        if fold_innov_rank:
            assert fold_mode == 'left', \
                "fold_innov_rank is implemented for the left fold"
        if tree_disent_rank:
            assert fold_mode != 'scan', \
                "tree_disent_rank acts on tree levels; scan builds none"
        # lowrank_gain=True: the three low-rank corrections (node_mix,
        #   fold_innov, tree_disent) become gain * (x V) U^T / (|U| |V|):
        #   U, V random (their DIRECTION is learned, Muon-friendly), a
        #   per-layer scalar gain starts at 0 (AdamW) -> bitwise incumbent
        #   at init; the map is normalized so that `gain` is the typical
        #   relative size of the correction (see _lowrank_corr).
        #   Motivation (measured 2026-09-24): with U zero-init routed to
        #   Muon (fixed-size orthogonalized steps, x2.8 tall-matrix scale),
        #   the corrections grew to 1.3x (innov), 6.6x (disent) and 30x
        #   (node_mix) the size of their inputs and all three arms lost
        #   1.3-7.7% BPB -- a learning-rate artifact, not a mechanism test.
        self.lowrank_gain = bool(lowrank_gain)
        # fold_dir='both' (COUNTERCLOCKWISE FOLD; path splitting after
        #   Chakaravarthy et al., arXiv 1602.04478: traverse the path both
        #   ways and join at the boundary). The left fold
        #   ((B1 o B2) o B3) o B4 composes the oldest block first and
        #   squashes it through every later compose (measured: its
        #   single-copy content 4.6 bits in-block -> 0 after the fold). The
        #   right fold B1 o (B2 o (B3 o B4)) composes the newest blocks
        #   first and joins the oldest LAST. Readout: P = P_left + gain (.)
        #   P_right, gain a per-slot vector, zero-init, AdamW (1-D) ->
        #   bitwise incumbent at init; every block gets a short path to
        #   the readout from one side. Cost: the right fold has no shared
        #   structure across prefixes (sum(popcount-1) composes, like the
        #   compacted left fold).
        assert fold_dir in ('left', 'both')
        if fold_dir == 'both':
            assert fold_mode == 'left' and fold_grade in (None, 'off', '') \
                and not fold_h0 and readout_mode == 'none', \
                "fold_dir='both' is implemented for the plain left fold"
        self.fold_dir = fold_dir
        # future_bag=(W, K): TRAINING-ONLY auxiliary head (never used by
        #   forward(); inference and the decoder are unchanged). From each
        #   position's final head input it predicts the distribution of
        #   hashed byte-trigrams (K buckets) in the next W bytes
        #   (opera_lm.losses.future_bag_loss). Motivation: next-byte loss
        #   barely rewards far context (MI curve flat beyond ~100 bytes),
        #   while 11% of 8-byte continuations recur only >100 bytes back;
        #   the past-future MI grows with the future window (L2M), so a
        #   window-level target pays the state to keep content that will
        #   recur.
        self.future_bag = tuple(future_bag) if future_bag else None
        assert node_mix_rank >= 0
        if node_mix_rank:
            assert fold_mode != 'scan', \
                "node_mix_rank mixes the compose node; scan builds none"
        self.head_mode = head_mode
        self.resid_mode = resid_mode
        self.state_mult = state_mult
        self.state_tie = bool(state_tie)
        self.node_mix_rank = int(node_mix_rank)
        # d_model/nb_model: the stream width (embedding, head, cross_mlp).
        # d/nb from here on: the TREE width, which is what every compose /
        # fold / rotation code path reads as self.d / self.nb.
        d_model, nb_model = d, nb
        d, nb = d * state_mult, nb * state_mult
        self.d_model, self.nb_model = d_model, nb_model
        # nb_gate: slots with their OWN gates/rotors (== nb unless
        # state_tie, where the state_mult copies of a slot share them).
        nb_gate = nb_model if state_tie else nb
        self.nb_gate = nb_gate
        assert lock_mode in ('none', 'interference')
        assert pe_mode in ('sin', 'none', 'rotor')
        assert rot_mode in ('so3', 'free')
        self.rot_mode = rot_mode
        assert fold_mode in ('left', 'left-masked', 'balanced', 'revolving',
                             'attend', 'rack', 'spine', 'oam', 'scan')
        assert not scan_salience or fold_mode == 'scan', \
            "--salience is a --fold scan flag (readout FiLM gate)"
        assert not workspace or fold_mode == 'scan', \
            "--workspace is a --fold scan flag (latent read path)"
        if fold_mode == 'scan':
            # OPERA-SCAN (v8.5): the tree+node machinery is REPLACED by the
            # associative scan (see OPERA_Scan_Arm_Design.md), so flags that
            # tune nodes/Fenwick-fold internals would be silently dead --
            # reject them (strict-flag convention). The readout nonlinearity
            # is fixed (LN + tanh(+0.1x), the node nonlinearities relocated).
            assert nb % 2 == 0, \
                "--fold scan needs even nb (scan blocks are paravector pairs)"
            assert rot_mode == 'so3', "--fold scan learns its own rotations"
            assert lock_mode == 'none', "--fold scan builds no tree nodes"
            assert not use_metal, \
                "--fold scan is eager-only (no MSL kernel); drop --metal"
            assert not use_triton, \
                "--fold scan is eager-only (no Triton kernel); drop --triton"
            assert fold_rotors == 'shared' and not fold_scale, \
                "--fold-rotors/--fold-scale tune the Fenwick fold, not the scan"
            assert not node_residual and tree_drop == 0.0, \
                "--node-residual/--tree-drop tune the compose node, not the scan"
            assert norm_mode == 'layer' and act_mode == 'tanh', \
                "--fold scan readout is fixed LN + tanh(+0.1x)"
            assert grad_checkpoint != 'level', \
                "--checkpoint level wraps compose nodes; use 'layer' with scan"
        assert homeo_mode in ('off', 'on')
        if homeo_mode == 'on':
            assert fold_mode != 'scan', \
                "--homeo anchors tree levels; --fold scan builds no tree"
        assert node_paths in (3, 4)
        if node_paths == 4:
            # QUOTIENT PATH (trajectory-dynamics arm, 2026-09): a fourth
            # gated path h_L (x) h_R^{-1}. The V2 whole-node kernels
            # (metal/triton) hard-code three gate channels, so this arm
            # is eager-only in v1.
            assert not (use_metal or use_triton), \
                "--node-paths 4 is eager-only (the fused metal/triton " \
                "node kernels hard-code 3 gate channels)"
            assert fold_mode != 'scan', \
                "--node-paths tunes the compose node; --fold scan builds none"
        # LEVEL-BALANCED GRADIENT (docs/OPERA_LevelGrad_prereg.md).
        # OPERA ties one compose operator across the whole scale
        # hierarchy: the same weights join two bytes at level 0 and two
        # 128-byte spans at level 7. Level k has T/2^k nodes, so the
        # shared weights receive geometrically less gradient from deeper
        # levels. MEASURED on the byte-d512 incumbent (T=256, layer 0):
        # gradient share 34.1 / 25.3 / 14.8 / 9.6 / 6.3 / 4.4 / 3.1 /
        # 2.3 / 0.0 % for levels 0..8 -- a 15:1 imbalance across trained
        # levels, and EXACTLY ZERO at the root (the only prefix reading
        # it is L=T, whose prediction target is masked).
        #
        # beta = 1.0 is the incumbent exactly (grad_scale is identity).
        # Level l's contribution to dL/dW is scaled by beta**l:
        # beta=2 cancels the node-count halving exactly; the measured
        # decay is ~1.5x per level (per-node gradient GROWS with depth,
        # partly self-correcting), so beta~1.5 equalises.
        #
        # This is not depth-recursion weight tying (Universal
        # Transformer, Mixture-of-Recursions), where the tied layer is
        # applied N times to a same-length sequence and node counts are
        # equal. The imbalance is specific to tying across SCALE.
        assert level_grad_balance > 0
        self.level_grad_balance = float(level_grad_balance)
        assert fold_relax in ('off', 'under', 'over')
        if fold_relax != 'off':
            assert fold_mode == 'left', \
                "fold_relax is implemented for the left fold"
            assert fold_bistable == 'off', \
                "fold_relax and fold_bistable both rewrite the fold " \
                "accumulator; enabling both makes attribution impossible"
        # SCALE-GRADED READOUT (docs/OPERA_ScaleGraded_prereg.md, REGISTERED
        # 2026-09-13): slot-partitioned fold accumulator. Slots [0,R) keep
        # the classical fold over ALL path blocks; slots [R,nb) are divided
        # into G equal groups and a level-l block composes only into
        # group min(l, G-1). Routing only -- zero new parameters, compose
        # node untouched (the fusion gate still conditions on the full 2d
        # input; only the accumulator WRITE is masked per slot). NOT
        # bitwise-incumbent at init: the information topology differs from
        # step 0 by design; the control is the incumbent architecture.
        if fold_grade not in (None, 'off', ''):
            if isinstance(fold_grade, str):
                parts = fold_grade.split(',')
                assert len(parts) == 2, \
                    "--fold-grade expects 'R,G' (e.g. '24,8')"
                R_g, G_g = int(parts[0]), int(parts[1])
            else:
                R_g, G_g = fold_grade
            assert fold_mode == 'left', \
                "fold_grade is implemented for the left fold"
            assert 0 <= R_g < nb and G_g >= 1 and (nb - R_g) % G_g == 0, \
                f"fold_grade needs 0 <= R < nb and (nb-R) % G == 0 " \
                f"(got R={R_g}, G={G_g}, nb={nb})"
            self.fold_grade = (R_g, G_g)
        else:
            self.fold_grade = None
        assert fold_bistable in ('off', 'mono', 'bi')
        if fold_bistable != 'off':
            assert fold_mode == 'left', \
                "fold_bistable is implemented for the left fold"
        assert fold_adapt in ('off', 'on')
        if fold_adapt == 'on':
            assert fold_mode != 'scan', \
                "--fold-adapt modulates the Fenwick fold; --fold scan " \
                "replaces it"
        assert oam_combine in ('compose', 'sum')
        assert oam_pair in ('seq', 'conj')
        assert oam_transport in ('rack', 'node')
        assert oam_k >= 1
        assert fold_rotors in ('shared', 'separate')
        assert norm_mode in ('layer', 'blockrms', 'rms')
        assert act_mode in ('tanh', 'linear')
        assert not (use_metal and use_triton), \
            "--metal and --triton are alternative kernel backends (Apple " \
            "Silicon vs CUDA); pass at most one"
        self.norm_mode = norm_mode
        self.act_mode = act_mode
        self.node_paths = node_paths
        self.fold_adapt = fold_adapt
        self.node_residual = node_residual
        self.tree_drop = tree_drop
        self.grad_checkpoint = grad_checkpoint
        self.use_metal = use_metal
        self.use_triton = use_triton
        if use_metal:
            # FusedNode.backward recomputes fv from the saved inputs
            # (metal_kernel.py) rather than recovering it via R_O^{-1},
            # so it no longer requires R_O to be orthogonal -- both
            # 'so3' and 'free' rotations are supported.
            from .metal_kernel import metal_available
            if not metal_available():
                print("WARNING: --metal requested but torch.mps.compile_shader"
                      " unavailable; using the (identical-math) fallback.",
                      flush=True)
        if use_triton:
            # Same no-inversion backward strategy as FusedNode, ported to
            # Triton for CUDA -- see triton_kernel.py's module docstring
            # for the Versor (arXiv:2602.10195) kernel-strategy credit.
            from .triton_kernel import triton_available
            if not triton_available():
                print("WARNING: --triton requested but no CUDA device / "
                      "triton install found; using the (identical-math) "
                      "fallback.", flush=True)
        self.pe_mode = pe_mode
        self.fold_mode = fold_mode
        self.fold_rotors = fold_rotors
        self.fold_scale = fold_scale
        self._rotor_cache = {}
        self.vocab_size = vocab_size
        self.d = d
        self.nb = nb
        self.num_layers = num_layers
        self.lock_mode = lock_mode
        self.tie = tie
        self.dropout = dropout

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        if tie:
            # Tied head + THE INIT FIX. Tying a N(0,1) embedding straight
            # into the head gives initial logits of magnitude ~sqrt(d)
            # (v6.7 confound; v7.2 init PPL 3.9e42). A learned scalar
            # logit_scale init 1/sqrt(d) restores initial loss ~ ln(V).
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.word_emb.weight
            self.logit_scale = nn.Parameter(torch.tensor(d_model ** -0.5))
        else:
            self.head = nn.Linear(d_model, vocab_size)
            self.logit_scale = None

        if fold_mode == 'scan':
            # --fold scan (v8.5): the tree's per-block rotors are replaced
            # by learned relative rotations carried IN the scan (q channel).
            self.quat = None
            self.rot_free = None
        elif rot_mode == 'so3':
            q_init = torch.zeros(num_layers, 3, nb_gate, 4)
            q_init[..., 0] = 1.0
            q_init += torch.randn_like(q_init) * 0.1
            self.quat = nn.Parameter(q_init)
            self.rot_free = None
        else:
            # --rot free (r15): unconstrained 3x3 per block per role per
            # layer. Init identity + 0.1 noise -- near-identity like the
            # so3 init, so the two arms start in comparable regimes.
            m_init = torch.eye(3).expand(num_layers, 3, nb_gate, 3, 3).clone()
            m_init += torch.randn_like(m_init) * 0.1
            self.rot_free = nn.Parameter(m_init)
            self.quat = None

        # Fold-specific parameters, created ONLY when their flag is on so
        # flags-off remains parameter-identical to v7.4/v7.0.
        if fold_rotors == 'separate':
            if rot_mode == 'so3':
                qf = torch.zeros(num_layers, 3, nb, 4)
                qf[..., 0] = 1.0
                qf += torch.randn_like(qf) * 0.1
                self.quat_fold = nn.Parameter(qf)
                self.rot_free_fold = None
            else:
                mf = torch.eye(3).expand(num_layers, 3, nb, 3, 3).clone()
                mf += torch.randn_like(mf) * 0.1
                self.rot_free_fold = nn.Parameter(mf)
                self.quat_fold = None
        else:
            self.quat_fold = None
            self.rot_free_fold = None
        if fold_scale:
            self.fold_theta = nn.Parameter(torch.full((num_layers, nb), 0.1))
        else:
            self.fold_theta = None
        if fold_mode == 'revolving':
            # REVOLVING DOORS: fixed-depth gated fold. Every position runs
            # every stage; a learned content gate decides compose vs
            # identity passthrough:
            #     acc <- g * compose(acc, slot) + (1-g) * acc
            # (a) static graph, no data-dependent control flow (compiler-
            #     native formulation of the fold);
            # (b) an identity HIGHWAY through the readout path -- the exact
            #     location where the interrogation probe convicted the
            #     forgetting (pp1/pp15 = 0.64 vs transformer 1.43);
            # (c) with --tree-drop, gates are randomly forced to identity
            #     during training (tree-shaped stochastic depth), training
            #     robustness to composition depths never seen in data --
            #     the direct attack on the under-training hypothesis.
            # Gate bias init +2.0 -> g ~ 0.88: starts near the left fold.
            self.fold_gate = nn.ModuleList(
                [nn.Linear(2 * d, 1) for _ in range(num_layers)])
            for g in self.fold_gate:
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, 2.0)
        else:
            self.fold_gate = None

        if fold_mode == 'scan':
            # --fold scan (v8.5): no compose nodes -> no fusion gate, node
            # norm, or lock depth. Replaced by the scan modules below.
            self.fusion_gate = None
            self.comp_norm = None
            self.block_gain = None
        else:
            self.fusion_gate = nn.ModuleList(
                [nn.Linear(2 * d, 3 * nb_gate)
                 for _ in range(num_layers)])
            if norm_mode == 'layer':
                self.comp_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
                self.block_gain = None
            elif norm_mode == 'rms':
                # --norm rms: ONE global RMS over the full d-vector, no mean
                # subtraction, per-block learned gains. Still exactly
                # SO(3)-equivariant (the global norm is rotation-invariant),
                # but -- unlike blockrms, which forced every block to the
                # same norm and erased the block-energy code (the likely
                # cause of blockrms's +4 PPL failure) -- this PRESERVES
                # relative block magnitudes. Separates LayerNorm's useful
                # part (energy pattern) from its geometry-breaking part
                # (cross-dimension mean subtraction).
                self.comp_norm = None
                self.block_gain = nn.Parameter(torch.ones(num_layers, nb))
            else:
                # --norm blockrms: per-block RMS normalization with a learned
                # per-block gain. No mean subtraction, no cross-block coupling.
                # Block norm is rotation-invariant (rotation acts on v, |v|
                # preserved, scalar untouched), so dividing by it -- and
                # scaling by a per-block scalar -- is SO(3)-EQUIVARIANT. This
                # is the first normalization in the project that preserves
                # the geometry the architecture is built on.
                self.comp_norm = None
                self.block_gain = nn.Parameter(torch.ones(num_layers, nb))
        if node_residual:
            # --node-residual: identity highway through the node.
            # out = r * composed + (1-r) * midpoint(children),
            # r = sigmoid(res_logit), init 2.0 -> r ~ 0.88 (mostly
            # composed at start; the highway is learnable per layer).
            self.res_logit = nn.Parameter(torch.full((num_layers,), 2.0))
        else:
            self.res_logit = None
        self.mod_depth = (None if fold_mode == 'scan' else
                          nn.Parameter(torch.full((num_layers,), -0.85)))

        # --fold scan (v8.5): OPERA-SCAN modules. Leaves: h_i -> (q_i, b_i)
        # via two small learned maps (per layer). The scan state is 8
        # scalars per block (q quaternion + b paravector) over nb/2 blocks
        # -- paravector PAIRS of the d-wide state (design §4: d == 4*nb
        # untouched) -- so the scan state itself is exactly d-wide.
        # Readout: per-block gate mixing the prefix state with the identity
        # baseline (bias -2.0, rack convention: near-neutral start), then
        # LayerNorm + tanh(+0.1x) -- the node nonlinearities, relocated
        # (design §3). q is normalized HERE, never in-scan (design §6).
        # v8.6: scan_wm parameterizes the per-leaf MAGNITUDE in (0,1),
        # LRU-style (s = exp(-softplus(x)), bias -3 -> s ~ 0.95: weak
        # decay at init). Under the conformal compose this is a learned,
        # input-dependent multiplicative decay INSIDE the scan
        # (RG-LRU/GLA/GDN mechanic): a leaf of magnitude s attenuates all
        # older content by s. Params: +d*nbs+nbs per layer.
        if fold_mode == 'scan':
            nbs = nb // 2
            self.scan_nbs = nbs
            self.scan_wq = nn.ModuleList(
                [nn.Linear(d, 4 * nbs) for _ in range(num_layers)])
            self.scan_wb = nn.ModuleList(
                [nn.Linear(d, 4 * nbs) for _ in range(num_layers)])
            with torch.no_grad():
                for wq in self.scan_wq:
                    # near-identity quaternion leaves at init: slow |q|
                    # drift over long scans (design §6 risk)
                    wq.weight.mul_(0.1)
                    wq.bias.zero_()
                    wq.bias.view(nbs, 4)[:, 0] = 1.0
            self.scan_wm = nn.ModuleList(
                [nn.Linear(d, nbs) for _ in range(num_layers)])
            with torch.no_grad():
                for wm in self.scan_wm:
                    wm.weight.mul_(0.1)            # near-uniform decay at init
                    # init decay s = e^-softplus(bias): -3.0 ~ 0.95 (hl ~14 tok),
                    # -5.0 ~ 0.993 (hl ~100 tok). v8.9: decay-audit showed the
                    # -3.0 init collapses to ~1.3-token median memory in
                    # training (long-range grads attenuated s^k); --scan-decay-bias.
                    wm.bias.fill_(scan_decay_bias)
            self.scan_gate = nn.ModuleList(
                [nn.Linear(d, nbs) for _ in range(num_layers)])
            for gm in self.scan_gate:
                nn.init.constant_(gm.bias, -2.0)
            self.scan_norm = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(num_layers)])
            base = torch.zeros(nbs, 8)
            base[:, 0] = 1.0                       # identity (q,b) per block
            self.register_buffer('scan_base', base, persistent=False)
        else:
            self.scan_nbs = 0
            self.scan_wq = None
            self.scan_wb = None
            self.scan_wm = None
            self.scan_gate = None
            self.scan_norm = None

        self.cross_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                          nn.Linear(d_model * 2, d_model))
            for _ in range(num_layers)
        ])
        # resid_init_scale: multiply each cross_mlp output projection by
        #   this factor at init ('auto' = 1/sqrt(2L), GPT-2's residual
        #   scaling) and zero its bias. Measured at L=8 (float64 perturbation
        #   growth per layer): blend/default 4.84x, add/default 1.38x,
        #   add + 1/sqrt(2L) 1.05x -- the deep stack is expansive without
        #   it (d8 runs: loss spike at peak LR, +38..49% BPB). In-place
        #   multiply: no RNG consumed; None = incumbent bitwise.
        if resid_init_scale is not None:
            sc = ((2.0 * num_layers) ** -0.5 if resid_init_scale == 'auto'
                  else float(resid_init_scale))
            with torch.no_grad():
                for mlp in self.cross_mlp:
                    mlp[-1].weight.mul_(sc)
                    mlp[-1].bias.zero_()
        self.resid_init_scale = resid_init_scale
        # hmem_nb=n>0: QUATERNION HOLOGRAPHIC MEMORY (in-context recall;
        #   docs/OPERA_Recall_Research_2026-09-24.md §4/§4a/§5a). Per layer,
        #   n quaternion slots. Write at every position i: bind the unit
        #   key of the PREVIOUS token with the value of the current token,
        #   key_{i-1} (x) value_i (Hamilton product -- non-commutative, so
        #   "B followed A" != "A followed B"), scaled by a learned write
        #   gate, and superpose by a running sum (exact, O(T); a sum needs
        #   no tree). Read at position t: unbind with the conjugate of the
        #   current token's unit key, conj(key_t) (x) M_t -- a value bound
        #   to a matching key returns exactly, others as rotated noise
        #   (geometric analogue of holographic reduced representations,
        #   Aerts/Czachor/De Moor 2009; redundancy across slots as in
        #   Associative LSTM, Danihelka et al. 2016). No scores, no softmax.
        #   Output: LayerNorm -> projection -> scalar gain (zero-init,
        #   AdamW) added to the fold readout: bitwise incumbent at init.
        #   Motivation: MQAR at chance (1/512) for the fold, pipeline
        #   control 100%.
        assert hmem_nb >= 0
        self.hmem_nb = int(hmem_nb)
        # hmem_decay (docs/OPERA_Recall_Research_2026-09-24.md §5d): the
        #   plain running sum accumulates interference (measured ~670
        #   effective writes per 1024 bytes; SNR ~ sqrt(slots/items)).
        #   'fixed': M_t = lambda_s M_{t-1} + update, lambda_s learnable per
        #     slot (1-D -> AdamW), half-lives spread geometrically 8..4096
        #     bytes (RetNet multi-scale decay; TODAM forgetting coefficient).
        #   'gated': lambda_{t,s} = sigmoid(W_f x_t + b_s) with W_f zero-init
        #     and b_s = the same spread -> identical to 'fixed' at init;
        #     forgetting can then depend on content (Associative LSTM's
        #     forget gate on the holographic memory; GLA/Mamba).
        assert hmem_decay in ('none', 'fixed', 'gated')
        assert hmem_decay == 'none' or hmem_nb > 0
        self.hmem_decay = hmem_decay
        # hmem_conv=K>0 (docs/OPERA_Recall_Research_2026-09-24.md §7): a
        #   causal depthwise convolution of width K with residual over the
        #   memory's raw projection (keys, values, write/forget logits):
        #   y_t <- y_t + sum_j c_j * y_{t-j}, j = 0..K-1 (Canon layer,
        #   Allen-Zhu 2025, position B; the short conv of H3 / Mamba /
        #   Based). Keys then describe the last K positions instead of one:
        #   at layer 0 a single byte has only 256 possible keys. Zero-init
        #   taps (AdamW) -> identical to the memory without it at init.
        assert hmem_conv >= 0 and (hmem_conv == 0 or hmem_nb > 0)
        self.hmem_conv = int(hmem_conv)
        if resid_mode == 'blend':
            self.blend_gate = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(),
                              nn.Linear(d_model // 4, 1), nn.Sigmoid())
                for _ in range(num_layers)
            ])
        else:
            # resid_mode='add': no blend gate (it would be a dead module).
            # Not RNG-identical to the incumbent from here on -- 'add' is
            # a different arm, never claimed bitwise.
            self.blend_gate = None

        self._fenwick_cache = {}
        self._rot_dense_cache = {}
        self._grade_mask_cache = {}

        # --fold attend (v8.0): q/k projections for the block-attention
        # readout. Created LAST so every shared module above consumes the
        # identical RNG stream as a non-attend model with the same seed
        # (enables exact cross-mode equality tests at single-block
        # positions). dk=64. Values are the raw blocks (attention as
        # ROUTING, not transformation); a single OPERA composition then
        # merges attended context with the most recent block.
        self.attn_dk = 64
        if fold_mode == 'attend':
            dk = self.attn_dk
            self.fold_attn_q = nn.Parameter(
                torch.randn(num_layers, dk, d) * (d ** -0.5))
            self.fold_attn_k = nn.Parameter(
                torch.randn(num_layers, dk, d) * (d ** -0.5))
        else:
            self.fold_attn_q = None
            self.fold_attn_k = None

        # --fold rack (r17): per-block injection gate for the rack fold.
        # Created last (same RNG-stream rule as attend). Bias init -2.0:
        # g ~ 0.12, so the fold starts near PURE conjugation (isometric)
        # and learns how much content to inject. Params: 2*d*nb + nb per
        # layer (~820k at d=640/nb=160/4L; documented, report both counts).
        if fold_mode == 'rack':
            self.rack_gate = nn.ModuleList(
                [nn.Linear(2 * d, nb) for _ in range(num_layers)])
            for gm in self.rack_gate:
                nn.init.constant_(gm.bias, -2.0)
        else:
            self.rack_gate = None

        # --rack-exitnorm (r18): a dedicated LayerNorm applied ONCE at the
        # rack fold's EXIT, multi-block positions only (single-block
        # positions bypass the fold entirely -- keeps the single-block ==
        # left equivalence exact). The fold INTERIOR stays norm-free
        # (per-step isometry untouched: old content is only ever rotated);
        # the norm addresses r17's training-speed deficit hypothesis --
        # the un-normalized fold exit fighting downstream statistics --
        # without giving up the rack's core claim. +2d params per layer.
        # LayerNorm init is deterministic (ones/zeros): the RNG stream and
        # therefore all other params are IDENTICAL to --fold rack.
        if fold_mode == 'rack' and rack_exitnorm:
            self.rack_exit_norm = nn.ModuleList(
                [nn.LayerNorm(d) for _ in range(num_layers)])
        else:
            self.rack_exit_norm = None

        # --fold spine (r19): q/k projections for attention over the
        # RUNNING FOLD ACCUMULATORS (the "spine" -- cumulative prefix
        # states after each fold step), queried from the final state.
        # Values are the raw accumulators; keys carry the deterministic
        # sinusoidal STEP encoding (small integers, defined for every
        # depth: extrapolation-safe). Created last (RNG-stream rule).
        # Params: 2*dk*d per layer, dk=64 -- same budget as attend.
        if fold_mode == 'spine':
            dk = self.attn_dk
            self.spine_q = nn.Parameter(
                torch.randn(num_layers, dk, d) * (d ** -0.5))
            self.spine_k = nn.Parameter(
                torch.randn(num_layers, dk, d) * (d ** -0.5))
        else:
            self.spine_q = None
            self.spine_k = None

        # --fold oam (v8.1): k CHARGED ISOMETRIC FOLD CHANNELS. One
        # injection gate per channel per layer (bias -2.0: near-pure
        # conjugation at init, rack convention); ONE learnable base
        # frequency phi per layer. Created LAST (RNG-stream rule); gates
        # are created layer-major, channel-minor, so at k=1 the draw
        # sequence is IDENTICAL to rack_gate's (bitwise rack equivalence,
        # asserted in selftest).
        if fold_mode == 'oam':
            if oam_charges == 'auto':
                charges = [c - (oam_k - 1) / 2.0 for c in range(oam_k)]
            else:
                charges = [float(x) for x in str(oam_charges).split(',')]
                assert len(charges) == oam_k,                     f"--oam-charges must list {oam_k} values, got {charges}"
            self.oam_k = oam_k
            self.oam_combine = oam_combine
            self.oam_shared_gate = oam_shared_gate
            self.oam_pair = oam_pair
            self.oam_transport = oam_transport
            self.register_buffer('oam_charge_vec',
                                 torch.tensor(charges, dtype=torch.float32))
            # --oam-pair conj: reorder channel slots BEFORE the compose
            # readout so conjugate charges (-m, +m) meet FIRST in the
            # pairwise tree (smallest |m| pair leftmost). The compose is
            # non-commutative -- order is architecture -- and 'seq'
            # (ascending charge, the v8.1 default) pairs (-1.5,-0.5) and
            # (+0.5,+1.5) at k=4 auto: NOT conjugates. If the physics
            # intuition is +-m interference, 'conj' is the principled
            # order. Stable sort by (|m|, m): auto k=4 -> perm [1,2,0,3]
            # i.e. (-0.5,+0.5)(-1.5,+1.5). Buffer is non-persistent: the
            # state_dict is unchanged, old checkpoints load as-is.
            if oam_pair == 'conj':
                order = sorted(range(oam_k),
                               key=lambda c: (abs(charges[c]), charges[c]))
            else:
                order = list(range(oam_k))
            self.register_buffer('oam_pair_perm',
                                 torch.tensor(order, dtype=torch.long),
                                 persistent=False)
            # charge active? (phi init 0 or all-zero charges -> skip the
            # rotation ENTIRELY: exact rack path, no fp noise)
            self._oam_charge_on = (float(oam_phi) != 0.0
                                   and any(abs(c) > 0 for c in charges))
            if oam_transport == 'node':
                # v8.2 NODE TRANSPORT: channels run the FULL compose node
                # with SHARED layer parameters (rotors, fusion gate, norm,
                # act -- everything). No injection gates exist: the node's
                # own fusion gate is the gate. The ONLY per-channel
                # difference is the charge phase, so the mechanism test is
                # zero-param up to phi (num_layers floats). If node-OAM
                # beats left, the phase multiplexing did it -- there is no
                # capacity to credit.
                assert not oam_shared_gate, \
                    "--oam-shared-gate is a rack-transport flag"
                self.oam_gate = None
                self.oam_gate_shared = None
                self.oam_bias = None
                self.oam_scale = None
            elif oam_shared_gate:
                self.oam_gate = None
                self.oam_gate_shared = nn.ModuleList(
                    [nn.Linear(2 * d, nb, bias=False)
                     for _ in range(num_layers)])
                self.oam_bias = nn.Parameter(
                    torch.full((num_layers, oam_k, nb), -2.0))
                self.oam_scale = nn.Parameter(
                    torch.ones(num_layers, oam_k))
            else:
                self.oam_gate = nn.ModuleList([
                    nn.ModuleList([nn.Linear(2 * d, nb)
                                   for _ in range(oam_k)])
                    for _ in range(num_layers)])
                for gm in self.oam_gate:
                    for g in gm:
                        nn.init.constant_(g.bias, -2.0)
                self.oam_gate_shared = None
                self.oam_bias = None
                self.oam_scale = None
            self.oam_phi = nn.Parameter(
                torch.full((num_layers,), float(oam_phi)))
            if float(oam_phi) == 0.0:
                self.oam_phi.requires_grad_(False)      # charge-off ablation
            if oam_combine == 'sum':
                self.oam_alpha = nn.Parameter(
                    torch.zeros(num_layers, oam_k))
            else:
                self.oam_alpha = None
            # v8.3 INDUCED OAM. Both deterministic-init (no RNG draws):
            # streams and non-oam arms are unaffected.
            # --oam-levelgate: sigma(a_l) gates the twist per level,
            # a_l init -3 -> twist ~5% at start; training OPENS it.
            self.oam_level_gate = None
            if oam_levelgate:
                self.oam_level_gate = nn.Parameter(
                    torch.full((num_layers, 16), -3.0))
            # --oam-chan-emb: per-channel identity embedding added to the
            # injected block's scalar parts (PAM identity for the shared
            # node). Zeros init: identity emerges only if useful.
            self.oam_chan_emb = None
            if oam_chan_emb:
                self.oam_chan_emb = nn.Parameter(
                    torch.zeros(num_layers, oam_k, nb))
        else:
            self.oam_k = 0
            self.oam_combine = 'compose'
            self.oam_shared_gate = False
            self.oam_pair = 'seq'
            self.oam_pair_perm = None
            self.oam_transport = 'rack'
            self._oam_charge_on = False
            self.oam_level_gate = None
            self.oam_chan_emb = None
            self.oam_gate = None
            self.oam_gate_shared = None
            self.oam_bias = None
            self.oam_scale = None
            self.oam_phi = None
            self.oam_alpha = None

        # --salience (v8.7, scan arm only): RANK-r FiLM SALIENCE GATE on the
        # scan readout ("amygdala/apical amplification" arm). A bottleneck
        # d -> r -> 2d map reads the position's OWN accumulated prefix
        # context (the pre-LN gated scan state `mixed`) and emits
        # per-feature gamma/beta applied after LayerNorm, before tanh.
        # Created LAST (RNG-stream rule, same as attend/rack/spine/oam):
        # salience-ON shares every draw with salience-OFF at the same
        # seed, and the final projection is ZERO-INIT (gamma = 1 + 0,
        # beta = 0) so the flag is an EXACT functional no-op at init
        # (selftested). Params per layer: d*r + r + r*2d + 2d.
        self.salience_rank = 64
        if fold_mode == 'scan' and scan_salience:
            r = self.salience_rank
            self.salience = nn.ModuleList([
                nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Linear(r, 2 * d))
                for _ in range(num_layers)])
            for sm in self.salience:
                nn.init.zeros_(sm[-1].weight)
                nn.init.zeros_(sm[-1].bias)
        else:
            self.salience = None

        # --workspace (v8.9, scan arm only): RANK-1 WORKSPACE --
        # prefix-causal latent-bottleneck cross-attention (Perceiver-AR /
        # Block-State-Transformer style). Motivation (measured): the scan's
        # CE curve is context-FLAT and trained decay collapses to ~1-token
        # half-life from both init biases -- long-range content is not
        # decodable from the prefix state. The workspace gives each
        # position a CONTENT-ADDRESSABLE read path over the past:
        # chunks of C=32 positions pool K=4 latents (learned queries,
        # routing attention, dk=64); positions query latents of STRICTLY
        # earlier chunks only (causal by construction, selftested);
        # a ZERO-INIT per-channel gate adds the readout post-LayerNorm
        # (exact identity at init). Cost O(T*K*n_chunks): linear in T.
        # Created LAST (RNG-stream rule, after salience). Params per
        # layer: K*d + 6*(d*dk) + d (latent queries + 6 projections +
        # gate) = 248,960 at d=640/dk=64/K=4.
        self.ws_chunk = 32
        self.ws_k = 4
        self.ws_dk = 64
        if fold_mode == 'scan' and workspace:
            sd = d ** -0.5
            self.ws_latq = nn.Parameter(
                torch.randn(num_layers, self.ws_k, d) * sd)
            for nm in ('ws_qp', 'ws_kp', 'ws_qr', 'ws_kr', 'ws_v'):
                setattr(self, nm, nn.Parameter(
                    torch.randn(num_layers, self.ws_dk, d) * sd))
            self.ws_o = nn.Parameter(
                torch.randn(num_layers, d, self.ws_dk) * sd)
            self.ws_gate = nn.Parameter(torch.zeros(num_layers, d))
        else:
            for nm in ('ws_latq', 'ws_qp', 'ws_kp', 'ws_qr', 'ws_kr',
                       'ws_v', 'ws_o', 'ws_gate'):
                setattr(self, nm, None)
        self._ws_cache = {}

        # --gate-bias (v9, arm A): CHRONO-STYLE FOLD GATE INIT. The
        # left/spine fold's transport composes use the tree's SHARED
        # fusion-gate WEIGHTS but a dedicated per-layer gate BIAS [3, nb]:
        # g0 (accumulator pass-through) init b0, g1 (new block) init b1,
        # g2 (geometric product) init b2. Deterministic init (no RNG
        # consumed) -> every other parameter is bit-identical to the
        # incumbent at the same seed; the TREE path never reads it
        # (children are symmetric, the fold is not). The bias stays
        # LEARNABLE -- the init is the intervention; whether training
        # keeps or undoes it is the measurement (db-5.0 lesson). Created
        # LAST (RNG-stream rule).
        if fold_gate_bias is not None:
            assert fold_mode in ('left', 'spine'), \
                "--gate-bias tunes the left/spine fold transport"
            assert len(fold_gate_bias) == 3, \
                "--gate-bias takes 3 values: b0,b1,b2"
            b0, b1, b2 = (float(x) for x in fold_gate_bias)
            gb_init = torch.empty(num_layers, 3, nb_gate)
            gb_init[:, 0, :] = b0
            gb_init[:, 1, :] = b1
            gb_init[:, 2, :] = b2
            self.fold_gate_bias = nn.Parameter(gb_init)
        else:
            self.fold_gate_bias = None

        # T0.4 (roadmap): MULTI-STATE GEOMETRIC READOUT. The head reads
        # the prefix's raw Fenwick block states (plus the attend fold's
        # deterministic level encoding) through a FIXED learned reduction
        # (per-slot gates + zero-init output projection, added residually
        # to the fold state). No routing, no softmax, no pairwise
        # interaction. Zero-init output => flags-on is the incumbent at
        # init (standing rule). Slot gates use small random init: zero
        # would dead-lock against the zero-init projection (no gradient
        # to either). Created after every other parameter (RNG-stream
        # rule), so shared params are bitwise the incumbent's.
        self.readout_mode = readout_mode
        if readout_mode == 'multistate':
            assert fold_mode == 'left', \
                "readout_mode='multistate' is implemented for the left fold"
            self.readout_max_slots = readout_max_slots
            self.readout_gate = nn.Parameter(
                0.02 * torch.randn(num_layers, readout_max_slots, d))
            self.readout_out = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(num_layers)])
            for ro in self.readout_out:
                nn.init.zeros_(ro.weight)
        else:
            assert readout_mode == 'none', readout_mode
            self.readout_gate = None
            self.readout_out = None

        # T1.4 (roadmap): DELTA-RULE MEMORY CHANNEL. A parallel matrix
        # memory alongside the fold, written recurrently with the
        # error-correcting delta rule and read by content query from the
        # fold state -- content-addressable retrieval with no token-token
        # score matrix and no softmax. mem_out is zero-init => flags-on
        # is the incumbent at init. Created last (RNG-stream rule).
        self.mem_mode = mem_mode
        if mem_mode == 'delta':
            assert fold_mode != 'scan', \
                "mem_mode='delta' reads tree/fold states; scan builds none"
            self.mem_dim = mem_dim
            self.mem_k = nn.ModuleList(
                [nn.Linear(d, mem_dim, bias=False)
                 for _ in range(num_layers)])
            self.mem_v = nn.ModuleList(
                [nn.Linear(d, mem_dim, bias=False)
                 for _ in range(num_layers)])
            self.mem_q = nn.ModuleList(
                [nn.Linear(d, mem_dim, bias=False)
                 for _ in range(num_layers)])
            self.mem_beta = nn.ModuleList(
                [nn.Linear(d, 1) for _ in range(num_layers)])
            self.mem_gate = nn.ModuleList(
                [nn.Linear(d + mem_dim, d) for _ in range(num_layers)])
            self.mem_out = nn.ModuleList(
                [nn.Linear(mem_dim, d, bias=False)
                 for _ in range(num_layers)])
            for mo in self.mem_out:
                nn.init.zeros_(mo.weight)
        else:
            assert mem_mode == 'none', mem_mode
            self.mem_dim = 0

        # SPINOR HOMEOSTASIS (trajectory-dynamics arm, 2026-09): per
        # (layer, tree level) anchor spinor q* in the model's own state
        # space; after each tree composition the parent block is rotated
        # a gated fraction of the way toward the anchor along the
        # quaternion geodesic (slerp), magnitude preserved. Semantics:
        # each level has a resting orientation -- predictable content
        # stays near it (small excursion), reinterpretive content lands
        # far from it (large excursion, then relaxation). Gate logits
        # init -6.0 (sigmoid ~= 0.0025) so flags-on is ~incumbent at
        # init; flags-off skips the code path entirely and stays
        # bitwise/parameter identical to the incumbent. 16 anchor levels
        # cover spans to 2^16 tokens; deeper parents (eval-time
        # extrapolation) reuse the deepest anchor. Created last
        # (RNG-stream rule).
        self.homeo_mode = homeo_mode
        if homeo_mode == 'on':
            anchors = torch.randn(num_layers, HOMEO_MAX_LEVELS, nb, 4)
            anchors = anchors / anchors.norm(dim=-1, keepdim=True)
            self.homeo_anchor = nn.Parameter(anchors)
            self.homeo_gate = nn.Parameter(torch.full(
                (num_layers, HOMEO_MAX_LEVELS, nb), -6.0))
        else:
            self.homeo_anchor = None
            self.homeo_gate = None

        # CONTENT-ADAPTIVE FOLD TRANSPORT (trajectory-dynamics arm,
        # 2026-09): the fold's transport twist per block is READ FROM
        # THE BLOCK'S OWN STATE -- ang = <h_block, w> -- instead of the
        # static per-level angle of --fold-scale. The tree decides per
        # node how far to rotate based on what is being composed:
        # predictable content -> angle ~0 (the state coasts); marked
        # content -> a real turn. w is ZERO-INIT: flags-on is bitwise
        # the incumbent at init AND consumes no RNG (zeros), so the
        # RNG-stream rule holds trivially. Created last.
        if fold_adapt == 'on':
            self.fold_adapt_w = nn.Parameter(
                torch.zeros(num_layers, nb, 4))
        else:
            self.fold_adapt_w = None

        # BISTABLE FOLD (docs/OPERA_Bistable_prereg.md). The fold's
        # accumulator update becomes the BRC recurrence (Vecoven, Ernst
        # & Drion, arXiv:2006.05252):
        #
        #     acc <- (1 - c) * composed + c * tanh(a (*) acc)
        #
        # per block. `a` is the feedback GAIN: the map x -> tanh(a x)
        # is monotonic (one stable fixed point) for a <= 1 and acquires
        # a negative-slope region with TWO stable fixed points for
        # a > 1. So bistability is reachable iff the gain range admits
        # a > 1 -- which is exactly what separates the two arms:
        #
        #   'mono' : a = sigmoid(z)     in (0, 1)  -- CONTROL, never bistable
        #   'bi'   : a = 1 + tanh(z)    in (0, 2)  -- BRC's range
        #
        # Identical parameter count and identical init; the ONLY
        # difference is whether a can exceed 1. The contrast therefore
        # isolates bistability rather than "a new module".
        #
        # WARM INIT, on purpose. b_c = 0 (c ~ 0.5) and the gain head is
        # zero-init so a ~ 1.0 ('bi') / 0.5 ('mono') at step 0. Phase 1's
        # homeo and quotient arms both nulled with gates stuck at
        # sigmoid(-6) = 0.0025 -- a multiplicative gate that small has
        # its own gradient suppressed by g(1-g), so it cannot open: the
        # null was a property of the init, not of the mechanism. Flags-on
        # is therefore NOT bitwise the incumbent here; 'off' is the
        # incumbent and 'mono' is the capacity-matched control.
        # bist_rank: 0 = one independent gain per block (the registered
        # 2026-09-08 arm, FALSIFIED); r > 0 = the nb gains are forced
        # through an r-dimensional bottleneck, so they can only move
        # together.
        #
        # WHY: the per-block arm used bistability heavily (66% of updates
        # at median gain 1.65) and produced no trajectory excursions. The
        # diagnosis was that nb=128 INDEPENDENT commitments average out
        # in the norm -- hypercube corners are all at similar distance,
        # so switching one changes little. Metastable-attractor models
        # (Recanatesi et al., Neuron 2021) produce lingering-then-abrupt
        # dynamics by coupling a high-dimensional state to a
        # LOW-DIMENSIONAL modulator, which is exactly the correlation
        # this bottleneck imposes. rank=1 is the extreme: a single
        # scalar drives all nb gains through one fixed profile.
        self.fold_bistable = fold_bistable
        self.bist_rank = bist_rank
        if fold_bistable != 'off':
            def _gain_head():
                if bist_rank > 0:
                    return nn.Sequential(
                        nn.Linear(2 * d, bist_rank, bias=False),
                        nn.Linear(bist_rank, nb))
                return nn.Linear(2 * d, nb)
            self.bist_a = nn.ModuleList(
                [_gain_head() for _ in range(num_layers)])
            self.bist_c = nn.ModuleList(
                [nn.Linear(2 * d, nb) for _ in range(num_layers)])
            for la, lc in zip(self.bist_a, self.bist_c):
                # Output zero at init -> a is exactly 1.0 ('bi') / 0.5
                # ('mono'), i.e. exactly ON the bifurcation point, so any
                # bistability is attributable to training.
                #
                # LoRA convention for the factored head: the DOWN
                # projection keeps its random init and only the UP
                # projection is zeroed. Zeroing both would make
                # dL/dW_down proportional to W_up = 0 and the factor
                # would never receive gradient at all -- the same
                # cold-start failure that produced phase 1's two
                # uninformative nulls. With this init only the first
                # step is blind.
                last = la[-1] if isinstance(la, nn.Sequential) else la
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                nn.init.zeros_(lc.bias)
        else:
            self.bist_a = None
            self.bist_c = None

        # OVER-RELAXATION on the fold accumulator.
        #
        #     acc <- acc + gamma (*) (composed - acc)
        #
        # gamma = 1 is EXACTLY the incumbent (acc <- composed); gamma < 1
        # under-relaxes (lands short of composed); gamma > 1 OVERSHOOTS
        # past it. This is successive over-relaxation, per block.
        #
        # WHY THIS CLASS AND NOT ANOTHER GATE. Four mechanisms have now
        # failed to move the step distribution -- fold_adapt (sigmoid),
        # Mamba's Delta (softplus), bistable per-block, bistable
        # low-rank. They differ in smoothness, granularity and stability
        # structure, and share exactly one property: every one is a gate
        # on a CONVEX blend, acc <- (1-c) X + c Y, which lands strictly
        # BETWEEN X and Y. A gate chooses where in that interval to
        # land; it can never pass either endpoint. No amount of gating
        # can therefore increase displacement -- that is arithmetic, not
        # a training failure. Over-relaxation is the one operation class
        # that escapes it, and phase 2 measured 41-58% of the
        # geometrically available step range going unused.
        #
        # THE INCUMBENT IS INTERIOR. gamma = 1 + tanh(z) with z zero-init
        # gives gamma = 1 exactly -> bitwise the incumbent, while
        # gradients flow in BOTH directions. Neither the cold-gate
        # failure (phase 1's homeo/quotient) nor the warm-init/not-
        # bitwise compromise (the bistable arms) applies here.
        #
        # CONTROL. 'under' clamps the same expression at 1:
        # gamma = 1 + min(tanh(z), 0) in (0, 1]. Identical parameters,
        # identical init, identical dynamics at step 0 -- the ONLY
        # difference is whether gamma can exceed 1. The contrast
        # therefore isolates OVERSHOOT, not capacity.
        self.fold_relax = fold_relax
        if fold_relax != 'off':
            self.relax_g = nn.ModuleList(
                [nn.Linear(2 * d, nb) for _ in range(num_layers)])
            for lg in self.relax_g:
                nn.init.zeros_(lg.weight)
                nn.init.zeros_(lg.bias)
        else:
            self.relax_g = None

        # LEVEL-CONDITIONED COMPOSE WEIGHTS (docs/OPERA_LevelCond_prereg.md,
        # OPEN 5b): W_l = W + U diag(f(l)) V per tree level, with f the
        # FIXED sinusoidal level basis (level_sin_enc) -- smooth in l and
        # defined at every depth, so levels beyond training get a smooth
        # continuation of the operator, never never-trained rows. Zero-init
        # U: bitwise the incumbent at init while dL/dU != 0 (the delta is
        # linear in U) -- no cold gate, no warm-gate compromise. The fold's
        # applications are scale-mixed per call and use the base W
        # (pre-stated in the prereg).
        self.level_cond_rank = int(level_cond_rank)
        if self.level_cond_rank > 0:
            r = self.level_cond_rank
            self.lc_U = nn.ParameterList(
                [nn.Parameter(torch.zeros(3 * nb, r)) for _ in range(num_layers)])
            self.lc_V = nn.ParameterList(
                [nn.Parameter(torch.randn(r, 2 * d) * (2 * d) ** -0.5)
                 for _ in range(num_layers)])
        else:
            self.lc_U = None
            self.lc_V = None

        # QUOTIENT PATH gate (node_paths=4): separate module created
        # LAST so the incumbent fusion_gate and RNG stream are exactly
        # the incumbent's. Bias -6 -> sigmoid ~= 0.0025: flags-on is
        # ~incumbent at init.
        if node_paths == 4:
            self.quotient_gate = nn.ModuleList(
                [nn.Linear(2 * d, nb) for _ in range(num_layers)])
            with torch.no_grad():
                for qg in self.quotient_gate:
                    qg.bias.fill_(-6.0)
        else:
            self.quotient_gate = None

        # STREAM / TREE-WIDTH ARMS (see the constructor head). Created
        # LAST (RNG-stream rule); the deterministic ones first so that
        # combining them never shifts the random draws of state_mult's
        # projections.
        # head_mode='stream': one LayerNorm per layer on the post-blend
        # stream (per-layer so aux heads read their own layer's stream).
        self.stream_norm = (nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)])
            if head_mode == 'stream' else None)
        # resid_mode='add': pre-norm on each layer's tree input (the
        # additive stream's norm grows with depth; the tree's leaves
        # should not).
        self.tree_in_norm = (nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)])
            if resid_mode == 'add' else None)
        # fold_gate='separate': a copy of the tree gate (no RNG consumed;
        # bitwise incumbent at init, then free to specialize).
        if fold_gate == 'separate':
            self.fusion_gate_fold = nn.ModuleList()
            for fg in self.fusion_gate:
                g = nn.Linear(2 * d, 3 * nb_gate)
                with torch.no_grad():
                    g.weight.copy_(fg.weight)
                    g.bias.copy_(fg.bias)
                self.fusion_gate_fold.append(g)
        else:
            self.fusion_gate_fold = None
        # fold_h0: per-layer learned initial fold accumulator, zero-init.
        # A ParameterList of 1-D tensors on purpose: split_muon_params
        # routes by ndim, and a [L, d] matrix would be orthogonalized.
        self.fold_h0 = (nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(num_layers)])
            if fold_h0 else None)
        # state_mult: leaves = tree_in(x) [d_model -> d], readout =
        # tree_out(prefix) [d -> d_model]. Variance-preserving init
        # (std fan_in^-1/2): leaves start at the incumbent's unit scale.
        if state_mult > 1:
            self.tree_in = nn.ModuleList(
                [nn.Linear(d_model, d, bias=False) for _ in range(num_layers)])
            self.tree_out = nn.ModuleList(
                [nn.Linear(d, d_model, bias=False) for _ in range(num_layers)])
            with torch.no_grad():
                for ti, to in zip(self.tree_in, self.tree_out):
                    ti.weight.normal_(0.0, d_model ** -0.5)
                    to.weight.normal_(0.0, d ** -0.5)
        else:
            self.tree_in = None
            self.tree_out = None
        # node_mix_rank: parent += (parent @ V) @ U^T, U zero-init (bitwise
        # incumbent at init), V variance-preserving. Created after every
        # other parameter (RNG-stream rule).
        def _lowrank(r):
            down = nn.ParameterList(
                [nn.Parameter(torch.randn(d, r) * d ** -0.5)
                 for _ in range(num_layers)])
            if lowrank_gain:
                up = nn.ParameterList(
                    [nn.Parameter(torch.randn(d, r) * d ** -0.5)
                     for _ in range(num_layers)])
                gain = nn.ParameterList(
                    [nn.Parameter(torch.zeros(())) for _ in range(num_layers)])
            else:
                up = nn.ParameterList(
                    [nn.Parameter(torch.zeros(d, r))
                     for _ in range(num_layers)])
                gain = None
            return down, up, gain
        # fold_dir='both': per-slot join gain for the right fold (zeros ->
        # bitwise incumbent; deterministic, no RNG consumed).
        self.fold_join_gain = (nn.ParameterList(
            [nn.Parameter(torch.zeros(nb)) for _ in range(num_layers)])
            if fold_dir == 'both' else None)
        # future_bag head (training-only); after the RNG-consuming modules
        # above so every forward-path parameter is the incumbent's.
        self.future_head = None
        if node_mix_rank:
            self.node_mix_down, self.node_mix_up, self.node_mix_gain = \
                _lowrank(node_mix_rank)
        else:
            self.node_mix_down = self.node_mix_up = self.node_mix_gain = None

        # INNOVATION FOLD / CAUSAL DISENTANGLER low-rank maps (U zero-init
        # -> bitwise incumbent; V variance-preserving; or the lowrank_gain
        # form). Created LAST.
        self.fold_innov_down, self.fold_innov_up, self.fold_innov_gain = (
            _lowrank(fold_innov_rank) if fold_innov_rank else (None,) * 3)
        self.tree_disent_down, self.tree_disent_up, self.tree_disent_gain = (
            _lowrank(tree_disent_rank) if tree_disent_rank else (None,) * 3)
        if self.future_bag is not None:
            self.future_head = nn.Linear(d_model, self.future_bag[1])
        # hmem: created LAST (RNG-stream rule) -- every other parameter is
        # the incumbent's at the same seed. Names: '*_gate' and '*_gain'
        # route to AdamW; the three projections are ordinary matrices.
        if self.hmem_nb:
            D = 4 * self.hmem_nb
            self.hmem_k = nn.ModuleList(
                [nn.Linear(d_model, D, bias=False) for _ in range(num_layers)])
            self.hmem_v = nn.ModuleList(
                [nn.Linear(d_model, D, bias=False) for _ in range(num_layers)])
            self.hmem_o = nn.ModuleList(
                [nn.Linear(D, d_model, bias=False) for _ in range(num_layers)])
            self.hmem_write_gate = nn.ModuleList(
                [nn.Linear(d_model, 1) for _ in range(num_layers)])
            with torch.no_grad():
                for wg in self.hmem_write_gate:
                    wg.bias.fill_(2.0)            # write ~0.88 at init
            self.hmem_norm = nn.ModuleList(
                [nn.LayerNorm(D) for _ in range(num_layers)])
            self.hmem_gain = nn.ParameterList(
                [nn.Parameter(torch.zeros(())) for _ in range(num_layers)])
            # decay: deterministic init (no RNG): half-lives 8..4096 bytes
            n = self.hmem_nb
            hl = 8.0 * (512.0 ** (torch.arange(n, dtype=torch.float32)
                                   / max(n - 1, 1)))
            lam = 0.5 ** (1.0 / hl)
            logit = torch.log(lam) - torch.log1p(-lam)
            self.hmem_decay_logit = (nn.ParameterList(
                [nn.Parameter(logit.clone()) for _ in range(num_layers)])
                if self.hmem_decay != 'none' else None)
            if self.hmem_decay == 'gated':
                self.hmem_forget_gate = nn.ModuleList(
                    [nn.Linear(d_model, n, bias=False)
                     for _ in range(num_layers)])
                with torch.no_grad():
                    for fg in self.hmem_forget_gate:
                        fg.weight.zero_()
            else:
                self.hmem_forget_gate = None
        else:
            self.hmem_decay_logit = self.hmem_forget_gate = None
            self.hmem_k = self.hmem_v = self.hmem_o = None
            self.hmem_write_gate = self.hmem_norm = self.hmem_gain = None
        # hmem_conv taps [C, K] per layer, zero-init (deterministic, no RNG),
        # created after everything else; C = the raw projection's channels.
        if self.hmem_conv:
            C = 8 * self.hmem_nb + 1 + (self.hmem_nb if self.hmem_decay == 'gated' else 0)
            self.hmem_conv_w = nn.ParameterList(
                [nn.Parameter(torch.zeros(C, self.hmem_conv))
                 for _ in range(num_layers)])
        else:
            self.hmem_conv_w = None

    def apply_head(self, h):
        logits = self.head(h)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        return logits

    def node_readout(self, h, layer_idx):
        """Map tree-width node states [..., d] to the head's input width
        (msup reads tree nodes through the head). Identity unless
        state_mult > 1, where it applies the layer's out-projection."""
        if self.tree_out is None:
            return h
        return self.tree_out[layer_idx](h)

    def get_rotations(self, layer_idx, fold=False):
        if self.rot_mode == 'free':
            src = (self.rot_free_fold
                   if (fold and self.rot_free_fold is not None)
                   else self.rot_free)
            M = src[layer_idx]
        else:
            src = (self.quat_fold if (fold and self.quat_fold is not None)
                   else self.quat)
            q = src[layer_idx]
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            M = quat_to_rotmat(q)
        if self.state_tie:
            # state_tie: the state_mult copies of each slot share its
            # rotor (copy-major slot layout: tree slot c*nb_gate + j).
            M = M.repeat(1, self.state_mult, 1, 1)
        return M[0], M[1], M[2]

    def _dense_rot(self, R):
        """Block-diagonal dense form M of the per-block 3x3 rotation R
        [nb,3,3]: M[(k,j),(k',i)] = R[k,i,j] * delta_kk', so the per-block
        rotation einsum('kij,nkj->nki', R, v) becomes ONE GEMM
        v.reshape(N, 3*nb) @ M. Same math up to float-op reordering (MPS
        GEMM fma: measured <=1e-6 fwd on unit-scale inputs); the MPS bmm
        it replaces costs ~2x per node because its encoding scales with
        the nb=160 batch. The zero-mask multiply builds M EXACTLY (each
        nonzero is a single R element). Built once per get_rotations()
        call: every compose node of a layer shares the same R objects, so
        the cache is keyed on tensor identity (strong refs keep id()
        unique) and cleared at each forward. Returns None when the cache
        is unsafe: --checkpoint level recomputes compose_pair_batch in
        backward with the SAME R objects, and a cached M from the
        original graph would disconnect the recompute from R.

        Also unsafe under torch.compile: id() is a Python object identity,
        not a traceable value, so dynamo can only make the cache lookup
        safe by guarding on the tensor's exact memory address -- which
        changes every step, forcing a full recompile every call (observed
        on CUDA: ~210s/step, hitting recompile_limit within a handful of
        steps). Skip the cache during tracing; the caller's einsum
        fallback (see compose_pair_batch) is fully compile-safe and is
        the same math, just without the single-GEMM memoization."""
        if self.grad_checkpoint == 'level':
            return None
        if torch.compiler.is_compiling():
            return None
        key = id(R)
        ent = self._rot_dense_cache.get(key)
        if ent is None:
            nb = self.nb
            eye = torch.eye(nb, device=R.device, dtype=R.dtype)
            M = (R.transpose(1, 2)[:, :, None, :]
                 * eye[:, None, :, None]).reshape(3 * nb, 3 * nb)
            if len(self._rot_dense_cache) > 32:
                self._rot_dense_cache.clear()
            self._rot_dense_cache[key] = (R, M)
            return M
        return ent[1]

    def _level_delta(self, layer_idx, level, W):
        """U diag(f(l)) V for this level's effective compose weights
        (level-conditioned arm). Cached per (layer, level, step) is
        unnecessary: the matmul is ~3M flops at the d512 rung."""
        f = level_sin_features(level, self.level_cond_rank,
                               W.device, W.dtype)
        return (self.lc_U[layer_idx] * f[None, :]) @ self.lc_V[layer_idx]

    def _compose(self, h_left, h_right, layer_idx, R_L, R_R, R_O,
                 gate_scale=1.0,
                 gate_bias=None,
                 level_idx=None,
                 fold=False):
        """compose_pair_batch, optionally under activation checkpointing:
        backward recomputes the node's ~18 intermediates from its two
        inputs instead of storing them (~25-35% slower steps for a
        several-fold activation-memory reduction). gate_bias (v9 arm A):
        optional [3, nb] replacement for the fusion-gate bias -- the
        fold's chrono init; None reproduces the incumbent node exactly.
        level_idx: the tree level this call composes (1 = span-2 nodes);
        consumed only by the level-capture registry (LO-Muon router /
        kill-switch instrument), never by the math.
        fold: this call is a fold transport compose -- selects the fold's
        own fusion gate when fold_gate='separate' (else identical)."""
        if self.grad_checkpoint == 'level' and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.compose_pair_batch, h_left, h_right,
                              layer_idx, R_L, R_R, R_O, gate_bias,
                              gate_scale, level_idx, fold,
                              use_reentrant=False)
        return self.compose_pair_batch(h_left, h_right, layer_idx,
                                       R_L, R_R, R_O, gate_bias,
                                       gate_scale, level_idx, fold)

    def compose_pair_batch(self, h_left, h_right, layer_idx, R_L, R_R, R_O,
                           gate_bias=None, gate_scale=1.0, level_idx=None,
                           fold=False):
        N = h_left.shape[0]
        nb = self.nb
        if fold and self.fusion_gate_fold is not None:
            gmod, gname = self.fusion_gate_fold[layer_idx], 'fusion_gate_fold'
        else:
            gmod, gname = self.fusion_gate[layer_idx], 'fusion_gate'
        gb = gmod.bias if gate_bias is None else gate_bias.reshape(-1)

        hl = h_left.reshape(N, nb, 4)
        hr = h_right.reshape(N, nb, 4)
        if (self.use_metal or self.use_triton) and self.lock_mode == 'none':
            # KERNEL V2: entire node pre-norm in ONE kernel (rotations +
            # geometric product + gated combine + output rotation).
            # Eager keeps only the gate matmul and the norm. The
            # diagnostic lock is SKIPPED here (~8 elementwise passes of
            # pure overhead per node) except during tree inspection.
            # use_triton (CUDA) and use_metal (Apple Silicon) share this
            # branch: same math, same autograd.Function contract, just a
            # different fused kernel underneath -- see triton_kernel.py.
            if self.use_triton:
                from .triton_kernel import fused_node_triton as fused_node
            else:
                from .metal_kernel import fused_node
            W = gmod.weight
            if self.lc_U is not None and isinstance(level_idx, int):
                W = W + self._level_delta(layer_idx, level_idx, W)
            W = _maybe_capture_w(W, layer_idx, level_idx, gname)
            b = gb
            g = (F.linear(h_left, W[:, :self.d]) +
                 F.linear(h_right, W[:, self.d:]) + b)
            g = torch.sigmoid(g.reshape(N, 3, self.nb_gate))
            if self.state_tie:
                g = g.repeat(1, 1, self.state_mult)
            parent = fused_node(hl, hr, R_L, R_R, R_O, g).reshape(N, -1)
            parent = self._node_mix(parent, layer_idx)
            if getattr(self, '_need_locks', False):
                with torch.no_grad():
                    v_l = hl[..., 1:]; v_r = hr[..., 1:]
                    v0d = torch.einsum('kij,nkj->nki', R_L, v_l)
                    v1d = torch.einsum('kij,nkj->nki', R_R, v_r)
                    lock_scalar = relative_lock(
                        hl[..., 0], v0d, hr[..., 0], v1d).mean(-1, keepdim=True)
            else:
                lock_scalar = torch.zeros(N, 1, device=h_left.device)
            if getattr(self, '_need_energy', False):
                with torch.no_grad():
                    energy_scalar = parent.reshape(N, nb, 4).norm(
                        dim=-1).mean(-1, keepdim=True)
            else:
                energy_scalar = torch.zeros(N, 1, device=h_left.device)
            if self.norm_mode == 'layer':
                parent = self.comp_norm[layer_idx](parent)
            elif self.norm_mode == 'rms':
                rms = torch.sqrt((parent * parent).mean(-1, keepdim=True) + 1e-6)
                pb = (parent / rms).reshape(N, nb, 4)
                parent = (pb * self.block_gain[layer_idx][None, :, None]).reshape(N, -1)
            else:
                pb = parent.reshape(N, nb, 4)
                rms = torch.sqrt((pb * pb).mean(-1, keepdim=True) + 1e-6)
                parent = (pb / rms * self.block_gain[layer_idx][None, :, None]).reshape(N, -1)
            if self.act_mode == 'tanh':
                if self.use_triton:
                    from .triton_kernel import fused_act_triton as fused_act
                else:
                    from .metal_kernel import fused_act
                parent = fused_act(parent)
            if self.res_logit is not None:
                r = torch.sigmoid(self.res_logit[layer_idx])
                parent = r * parent + (1.0 - r) * 0.5 * (h_left + h_right)
            return parent, lock_scalar, energy_scalar
        if self.use_metal:
            # lock_mode='interference' needs lock-modulated gates: use the
            # v1 kernel (rotate+geo) and keep the rest eager.
            from .metal_kernel import fused_compose
            p0f, p1f, geof = fused_compose(hl, hr, R_L, R_R)
            s0, v0 = p0f[..., 0], p0f[..., 1:]
            s1, v1 = p1f[..., 0], p1f[..., 1:]
            gs, gv = geof[..., 0], geof[..., 1:]
        else:
            s_l, v_l = hl[..., 0], hl[..., 1:]
            s_r, v_r = hr[..., 0], hr[..., 1:]
            M_L, M_R = self._dense_rot(R_L), self._dense_rot(R_R)
            if M_L is not None and M_R is not None:
                # single-GEMM block rotations (see _dense_rot): ~2x
                # faster than the per-block bmm on MPS.
                v0 = (v_l.reshape(N, -1) @ M_L).reshape(N, nb, 3)
                v1 = (v_r.reshape(N, -1) @ M_R).reshape(N, nb, 3)
            else:
                v0 = torch.einsum('kij,nkj->nki', R_L, v_l)
                v1 = torch.einsum('kij,nkj->nki', R_R, v_r)
            s0, s1 = s_l, s_r
            gs, gv = geometric_product(s0, v0, s1, v1)

        # gate without materializing cat([h_left, h_right]): two half-
        # matmuls on the same weight -- identical math, one less [N, 2d]
        # tensor written and saved for backward at EVERY node. Chained
        # addmm: (b + h_left@W1^T) + h_right@W2^T in 2 kernel launches
        # (float-op reordering only).
        W = gmod.weight
        if self.lc_U is not None and isinstance(level_idx, int):
            W = W + self._level_delta(layer_idx, level_idx, W)
        W = grad_scale(_maybe_capture_w(W, layer_idx, level_idx, gname),
                       gate_scale)
        b = grad_scale(gb, gate_scale) if gate_scale != 1.0 else gb
        g = torch.addmm(torch.addmm(b, h_left, W[:, :self.d].t()),
                        h_right, W[:, self.d:].t())
        g = torch.sigmoid(g.reshape(N, 3, self.nb_gate))
        if self.state_tie:
            g = g.repeat(1, 1, self.state_mult)
        g0, g1, g2 = g[:, 0, :], g[:, 1, :], g[:, 2, :]

        if self.lock_mode == 'interference':
            lock = relative_lock(s0, v0, s1, v1)
            depth = torch.sigmoid(self.mod_depth[layer_idx])
            passthrough = 1.0 - depth * lock
            g0 = g0 * passthrough
            g1 = g1 * passthrough
            g2 = g2 * lock
            lock_scalar = lock.mean(dim=-1, keepdim=True)
        elif getattr(self, '_need_locks', False):
            with torch.no_grad():
                lock_scalar = relative_lock(s0, v0, s1, v1).mean(dim=-1, keepdim=True)
        else:
            # lock_mode='none': the lock is diagnostic-only (read via
            # return_tree tree inspection). Skip it when no caller reads
            # it -- same gate as the fused-metal path above.
            lock_scalar = torch.zeros(N, 1, device=h_left.device)

        fs = torch.addcmul(torch.addcmul(g0 * s0, g1, s1), g2, gs)
        fv = torch.addcmul(torch.addcmul(g0.unsqueeze(-1) * v0,
                                         g1.unsqueeze(-1), v1),
                           g2.unsqueeze(-1), gv)
        if self.node_paths == 4:
            # quotient path: g3 gates h_L (x) h_R^{-1}, the relative
            # transform between siblings. Own gate module (created last
            # in __init__, bias -6 -> ~incumbent at init) so the
            # incumbent fusion_gate and the RNG stream are untouched.
            Wq = self.quotient_gate[layer_idx].weight
            g3 = torch.sigmoid(torch.addmm(
                torch.addmm(self.quotient_gate[layer_idx].bias,
                            h_left, Wq[:, :self.d].t()),
                h_right, Wq[:, self.d:].t()))
            qs, qv = quat_quotient(s0, v0, s1, v1)
            fs = fs + g3 * qs
            fv = fv + g3.unsqueeze(-1) * qv
        M_O = self._dense_rot(R_O)
        if M_O is not None:
            fv = (fv.reshape(N, -1) @ M_O).reshape(N, nb, 3)
        else:
            fv = torch.einsum('kij,nkj->nki', R_O, fv)

        parent = torch.cat([fs.unsqueeze(-1), fv], dim=-1).reshape(N, -1)
        parent = self._node_mix(parent, layer_idx)
        if getattr(self, '_need_energy', False):
            with torch.no_grad():
                energy_scalar = parent.reshape(N, nb, 4).norm(
                    dim=-1).mean(-1, keepdim=True)
        else:
            energy_scalar = torch.zeros(N, 1, device=h_left.device)
        if self.norm_mode == 'layer':
            parent = self.comp_norm[layer_idx](parent)
        elif self.norm_mode == 'rms':
            rms = torch.sqrt((parent * parent).mean(dim=-1, keepdim=True) + 1e-6)
            pb = (parent / rms).reshape(N, nb, 4)
            pb = pb * self.block_gain[layer_idx][None, :, None]
            parent = pb.reshape(N, -1)
        else:
            pb = parent.reshape(N, nb, 4)
            rms = torch.sqrt((pb * pb).mean(dim=-1, keepdim=True) + 1e-6)
            pb = pb / rms * self.block_gain[layer_idx][None, :, None]
            parent = pb.reshape(N, -1)
        if self.act_mode == 'tanh':
            parent = torch.tanh(parent) + 0.1 * parent
        # act_mode == 'linear': no pointwise squash (tanh is not
        # equivariant; with blockrms+linear the node is geometry-clean
        # and all nonlinearity comes from the geometric product + gates)
        if self.res_logit is not None:
            r = torch.sigmoid(self.res_logit[layer_idx])
            parent = r * parent + (1.0 - r) * 0.5 * (h_left + h_right)
        return parent, lock_scalar, energy_scalar

    def _hmem_parts(self, x, layer_idx):
        """Unit keys [.., n, 4], values [.., n, 4] and write gate [.., 1]
        for the holographic memory, from the layer's input stream x."""
        n = self.hmem_nb
        # one runtime-concatenated GEMM for k, v and the write gate (the
        # three modules and the state_dict are unchanged)
        W = torch.cat([self.hmem_k[layer_idx].weight,
                       self.hmem_v[layer_idx].weight,
                       self.hmem_write_gate[layer_idx].weight], 0)
        bias = torch.cat([W.new_zeros(8 * n),
                          self.hmem_write_gate[layer_idx].bias], 0)
        y = F.linear(x, W, bias).float()
        k = F.normalize(y[..., :4 * n].reshape(*x.shape[:-1], n, 4), dim=-1)
        v = y[..., 4 * n:8 * n].reshape(*x.shape[:-1], n, 4)
        w = torch.sigmoid(y[..., 8 * n:])
        return k, v, w

    def _hmem_proj(self, x, layer_idx, conv=True):
        """Raw memory projection y [.., C]: keys | values | write logit |
        forget logits ('gated'), one GEMM; plus the hmem_conv short
        convolution over the sequence (x: [B, T, d])."""
        n = self.hmem_nb
        Ws = [self.hmem_k[layer_idx].weight, self.hmem_v[layer_idx].weight,
              self.hmem_write_gate[layer_idx].weight]
        bs = [Ws[0].new_zeros(8 * n), self.hmem_write_gate[layer_idx].bias]
        if self.hmem_forget_gate is not None:
            Ws.append(self.hmem_forget_gate[layer_idx].weight)
            bs.append(Ws[0].new_zeros(n))
        y = F.linear(x, torch.cat(Ws, 0), torch.cat(bs, 0))
        if self.hmem_conv and conv:
            # Metal kernel on MPS, the same op in PyTorch elsewhere
            from .hmem_kernel import causal_dwconv
            y = causal_dwconv(y, self.hmem_conv_w[layer_idx])      # taps [C, K]
        return y

    def _hmem_conv_row(self, y_hist, layer_idx):
        """Decoder: the convolved projection of the newest position from
        y_hist [K, C] (row 0 = newest raw projection, zeros before t=0)."""
        c = self.hmem_conv_w[layer_idx]
        yf = y_hist.float()
        return (yf[0] + (c.t() * yf).sum(0)).to(y_hist.dtype)

    def _hmem_out(self, r, layer_idx, dtype):
        """Retrieved quaternions r [.., n, 4] -> readout contribution."""
        r = self.hmem_norm[layer_idx](r.reshape(*r.shape[:-2], -1))
        return (self.hmem_gain[layer_idx]
                * self.hmem_o[layer_idx](r.to(dtype))).to(dtype)

    def _hmem_read(self, x, layer_idx):
        """Quaternion holographic memory over the whole sequence, causal.
        x: [B, T, d_model]. M_t = sum_{i<=t} w_i key_{i-1} (x) value_i;
        read r_t = conj(key_t) (x) M_t. fp32 inside (running sums)."""
        if (self.use_metal and x.device.type == 'mps') or self.hmem_conv:
            # fused Metal kernel (opera_lm/hmem_kernel.py, hmem_fused): one
            # GEMM for keys / values / write gate / forget gate, then the
            # kernel reads the raw (bf16) projection and does key
            # normalization, the sigmoids, bind + decayed superposition +
            # unbind in one pass, analytic backward -- identical math
            # (selftest test_hmem_kernel). Off MPS, the same op in PyTorch.
            from .hmem_kernel import hmem_fused
            y = self._hmem_proj(x, layer_idx)
            L = (self.hmem_decay_logit[layer_idx]
                 if self.hmem_decay_logit is not None else None)
            r = hmem_fused(y, L, self.hmem_nb, self.hmem_decay)
            return self._hmem_out(r, layer_idx, x.dtype)
        k, v, w = self._hmem_parts(x, layer_idx)                 # [B,T,n,4]
        kprev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], 1)
        bind = quat_mul(kprev, v) * w.unsqueeze(-1)
        if self.hmem_decay == 'none':
            M = bind.cumsum(1)
        else:
            a = self._hmem_lambda(x, layer_idx)                   # [B,T,n] or [n]
            if a.dim() == 1:
                a = a.expand(bind.shape[:3])
            M = _decay_scan(a.unsqueeze(-1), bind)
        r = quat_mul_conj_a(k, M)
        return self._hmem_out(r, layer_idx, x.dtype)

    def _hmem_lambda(self, x, layer_idx):
        """Per-slot decay lambda in (0,1): [n] ('fixed') or [.., n]
        ('gated', from the layer's input stream x)."""
        z = self.hmem_decay_logit[layer_idx].float()
        if self.hmem_forget_gate is not None:
            z = z + self.hmem_forget_gate[layer_idx](x).float()
        return torch.sigmoid(z)

    def _lowrank_corr(self, x, name, layer_idx):
        """(x V) U^T for the low-rank correction `name` ('node_mix',
        'fold_innov', 'tree_disent'). With lowrank_gain the map M = V U^T
        is rescaled by gain * sqrt(d) / |M|_F: for isotropic x,
        E|x M|^2 = |x|^2 |M|_F^2 / d, so `gain` is the TYPICAL relative
        size of the correction -- its direction is learned (Muon on U, V;
        their norms cancel), its magnitude only through the AdamW scalar.
        |M|_F^2 = sum((U^T U) * (V^T V)) (r x r, cheap)."""
        V = getattr(self, name + '_down')[layer_idx].to(x.dtype)
        U = getattr(self, name + '_up')[layer_idx].to(x.dtype)
        c = (x @ V) @ U.t()
        g = getattr(self, name + '_gain')
        if g is not None:
            fro = ((U.t() @ U) * (V.t() @ V)).sum().clamp_min(1e-12).sqrt()
            c = c * (g[layer_idx].to(x.dtype) * (U.shape[0] ** 0.5) / fro)
        return c

    def _node_mix(self, parent, layer_idx):
        """node_mix_rank: identity + low-rank cross-slot value mixing on
        the pre-norm node output. Identity (returns `parent` itself) when
        off; exactly `parent` at init when on (U = 0, or gain = 0)."""
        if self.node_mix_up is None:
            return parent
        return parent + self._lowrank_corr(parent, 'node_mix', layer_idx)

    def _fold_innov(self, acc, nxt, layer_idx):
        """Innovation fold: acc - (nxt V) U^T (identity when off; exact
        `acc` at init since U = 0)."""
        if self.fold_innov_up is None:
            return acc
        return acc - self._lowrank_corr(nxt, 'fold_innov', layer_idx)

    def _disentangle(self, nodes, layer_idx):
        """Causal disentangler on one tree level, nodes [B, n, d]: node j
        += (raw node j-1) V U^T; node 0 has no left neighbour (+0).
        Identity when off; exact at init (U = 0)."""
        if self.tree_disent_up is None:
            return nodes
        left = torch.cat([torch.zeros_like(nodes[:, :1]), nodes[:, :-1]], 1)
        return nodes + self._lowrank_corr(left, 'tree_disent', layer_idx)

    def _homeo_relax(self, parent, layer_idx, level):
        """homeo_mode='on': rotate each parent block a gated fraction of
        the way toward the level's anchor spinor along the quaternion
        geodesic (slerp), preserving block magnitude. parent: [N, d].
        level: tree level of these parents (1 = span-2 nodes); clamped
        to the anchor table (eval-time extrapolation reuses the deepest
        anchor)."""
        nb = self.nb
        level = min(level, HOMEO_MAX_LEVELS - 1)
        g = torch.sigmoid(self.homeo_gate[layer_idx, level])   # [nb]
        a = self.homeo_anchor[layer_idx, level]                # [nb, 4]
        pb = parent.reshape(-1, nb, 4)
        pn = pb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        qn = pb / pn                                           # unit dir
        an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dot = (qn * an).sum(dim=-1, keepdim=True)
        an = torch.where(dot < 0, -an, an)                     # shortest arc
        dot = dot.abs().clamp(max=1.0 - 1e-6)
        theta = torch.acos(dot)
        sin_th = torch.sin(theta)
        g4 = g.unsqueeze(-1)                                   # [nb, 1]
        # slerp weights; near-parallel anchors fall back to lerp
        w_q = torch.where(sin_th > 1e-4, torch.sin((1 - g4) * theta)
                          / sin_th.clamp(min=1e-8), 1 - g4)
        w_a = torch.where(sin_th > 1e-4, torch.sin(g4 * theta)
                          / sin_th.clamp(min=1e-8), g4)
        out = w_q * qn + w_a * an
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (out * pn).reshape(-1, 4 * nb)

    def build_tree(self, states, layer_idx, R_L, R_R, R_O):
        """ON-FLY INDEXING: no padding anywhere. A Fenwick block (k, j)
        is referenced only when (j+1)*2^k <= T, so 'partial' parents that
        would cover padding are NEVER read -- computing them (as the old
        padded version did) was pure waste: up to ~50% of tree work when
        T sits just above a power of two. Level k now holds exactly
        floor(prev/2) nodes; the referenced set (and therefore every
        output) is IDENTICAL to the padded version."""
        B, n0, d = states.shape
        states = self._disentangle(states, layer_idx)   # level 0 (identity if off)
        levels = [states]
        locks = []
        energies = []
        current = states
        lvl_i = 0
        beta = self.level_grad_balance
        # NORMALISE the per-level weights to mean 1 over the levels this
        # tree actually has. Without this, beta>1 would simply inflate
        # every compose gradient -- indistinguishable from raising the
        # learning rate on those parameters, and any effect would be an
        # LR effect rather than a BALANCE effect. With it, beta purely
        # redistributes a fixed total between shallow and deep levels.
        _n_lv = max(1, int(math.floor(math.log2(max(states.shape[1], 2)))))
        _z = (sum(beta ** i for i in range(1, _n_lv + 1)) / _n_lv
              if beta != 1.0 else 1.0)
        while current.shape[1] >= 2:
            n = current.shape[1]
            m = n // 2
            left = current[:, 0:2 * m:2, :]
            right = current[:, 1:2 * m:2, :]
            N = B * m
            lvl_i += 1
            # Level lvl_i's share of dL/dW is scaled by beta**lvl_i.
            gs = (beta ** lvl_i) / _z
            if _LEVEL_CAPTURE is not None:
                # input-RMS EMA per level: the statistic the DERIVED
                # LO-Muon weighting (w_l ~ 1/x_l) normalizes by. The
                # compose inputs are the (left, right) halves.
                r = (left.detach().square().mean().sqrt()
                     + right.detach().square().mean().sqrt()).item() / 2
                prev = _LEVEL_CAPTURE['rms'].get(lvl_i)
                _LEVEL_CAPTURE['rms'][lvl_i] = (
                    r if prev is None else 0.9 * prev + 0.1 * r)
            rl, rr, ro = (grad_scale(R_L, gs), grad_scale(R_R, gs),
                          grad_scale(R_O, gs))
            parent, lock, energy = self._compose(
                left.reshape(N, d), right.reshape(N, d), layer_idx,
                rl, rr, ro, gate_scale=gs, level_idx=lvl_i)
            if self.homeo_mode == 'on':
                # spinor homeostasis: relax the new parents toward the
                # level's anchor (level 1 = span-2 nodes; leaves untouched)
                parent = self._homeo_relax(parent, layer_idx, len(levels))
            current = self._disentangle(parent.reshape(B, m, d), layer_idx)
            levels.append(current)
            locks.append(lock.reshape(B, m))
            energies.append(energy.reshape(B, m))
        return levels, locks, energies

    def _fenwick_indices(self, T, num_levels, level_offsets, device):
        key = (T, num_levels, str(device))
        if key in self._fenwick_cache:
            return self._fenwick_cache[key]
        table, max_blocks = fenwick_blocks(T)
        idx = torch.zeros(T, max_blocks, dtype=torch.long)
        lvl = torch.zeros(T, max_blocks, dtype=torch.long)
        count = torch.zeros(T, dtype=torch.long)
        for j, blocks in enumerate(table):
            count[j] = len(blocks)
            for s, (level, node) in enumerate(blocks):
                idx[j, s] = level_offsets[level] + node
                lvl[j, s] = level
        # FOLD COMPACTION (v7.9): active positions per fold step,
        # host-computed once per T and cached. Static tensors -- no
        # data-dependent control flow at run time (compile-safe).
        active = []
        for s in range(max_blocks):
            act = torch.nonzero(count > s, as_tuple=False).squeeze(-1)
            active.append(act.to(device))
        out = (idx.to(device), count.to(device), max_blocks, lvl.to(device),
               active)
        self._fenwick_cache[key] = out
        return out

    def _grade_fold_masks(self, T, num_levels, level_offsets, device):
        """Per-fold-step accumulator write masks for the scale-graded
        readout (docs/OPERA_ScaleGraded_prereg.md §1). masks[s-1] is
        [m_s, nb] bool over the ACTIVE rows of fold step s: True where a
        slot takes the composed value (global slots [0,R) always; graded
        slot j only when the step's block level routes to j's group,
        group(l) = min(l, G-1)). seed_mask is [T, nb] for the
        initialization: the path's slot-0 block (the oldest, level
        lvl[t, 0]) seeds only global slots + its own group; every other
        group starts at ZERO -- a group's fold sees only ITS blocks
        (compose has no neutral element, so the content-free empty-fold
        value is zero). Static per (T, R, G, nb) -- computed once from
        the cached Fenwick decomposition, never from data."""
        R, G = self.fold_grade
        nb = self.nb
        key = (T, num_levels, R, G, nb, str(device))
        if key in self._grade_mask_cache:
            return self._grade_mask_cache[key]
        _, count, max_blocks, lvl, active = self._fenwick_indices(
            T, num_levels, level_offsets, device)
        w = (nb - R) // G
        slot_group = torch.full((nb,), -1, dtype=torch.long,
                                device=lvl.device)
        slot_group[R:] = torch.arange(nb - R, dtype=torch.long,
                                      device=lvl.device) // w
        g_seed = lvl[:, 0].clamp(max=G - 1)                       # [T]
        seed_mask = ((slot_group[None, :] < 0) |
                     (slot_group[None, :] == g_seed[:, None]))     # [T, nb]
        masks = []
        for s in range(1, max_blocks):
            act = active[s]
            if act.numel() == 0:
                masks.append(None)
                continue
            g_pos = lvl[act, s].clamp(max=G - 1)                # [m]
            keep = ((slot_group[None, :] < 0) |
                    (slot_group[None, :] == g_pos[:, None]))    # [m, nb]
            masks.append(keep)
        out = (seed_mask, masks)
        self._grade_mask_cache[key] = out
        return out

    def prefix_states(self, levels, T, layer_idx, R_L, R_R, R_O):
        B = levels[0].shape[0]
        d = self.d
        nb = self.nb
        device = levels[0].device

        # Dedicated fold rotors if enabled; otherwise the tree's.
        if self.quat_fold is not None:
            R_L, R_R, R_O = self.get_rotations(layer_idx, fold=True)

        level_offsets = []
        off = 0
        for lv in levels:
            level_offsets.append(off)
            off += lv.shape[1]
        flat = torch.cat(levels, dim=1)

        idx, count, max_blocks, lvl, active = self._fenwick_indices(
            T, len(levels), level_offsets, device)
        gathered = flat[:, idx.reshape(-1), :].reshape(B, T, max_blocks, d)

        # --fold-scale: block of span 2^k enters the fold twisted by angle
        # k * theta_b about its z-axis (vector parts only). Scale, not
        # absolute position: defined for every level, smooth at unseen depths.
        if self.fold_theta is not None:
            theta = self.fold_theta[layer_idx]                    # [nb]
            ang = lvl.to(theta.dtype)[:, :, None] * theta[None, None, :]  # [T,S,nb]
            c = torch.cos(ang)[None]                              # [1,T,S,nb]
            s = torch.sin(ang)[None]
            h = gathered.reshape(B, T, max_blocks, nb, 4)
            sc, vx, vy, vz = h[..., 0], h[..., 1], h[..., 2], h[..., 3]
            vx2 = vx * c - vy * s
            vy2 = vx * s + vy * c
            gathered = torch.stack([sc, vx2, vy2, vz], dim=-1).reshape(
                B, T, max_blocks, d)

        # --fold-adapt: content-adaptive transport twist, read from the
        # block's own state (zero-init w -> ang 0 -> bitwise incumbent).
        if self.fold_adapt_w is not None:
            w = self.fold_adapt_w[layer_idx]                  # [nb, 4]
            h = gathered.reshape(B, T, max_blocks, nb, 4)
            ang = (h * w[None, None, None]).sum(-1)           # [B,T,S,nb]
            c, s = torch.cos(ang), torch.sin(ang)
            sc, vx, vy, vz = h[..., 0], h[..., 1], h[..., 2], h[..., 3]
            vx2 = vx * c - vy * s
            vy2 = vx * s + vy * c
            gathered = torch.stack([sc, vx2, vy2, vz], dim=-1).reshape(
                B, T, max_blocks, d)

        if self.fold_mode == 'revolving':
            acc = gathered[:, :, 0, :]
            validf = (torch.arange(max_blocks, device=device)[None, :]
                      < count[:, None]).float()               # [T, S]
            for s_idx in range(1, max_blocks):
                nxt = gathered[:, :, s_idx, :]
                composed, _, _ = self._compose(
                    acc.reshape(B * T, d), nxt.reshape(B * T, d),
                    layer_idx, R_L, R_R, R_O)
                composed = composed.reshape(B, T, d)
                Wg = self.fold_gate[layer_idx].weight
                bg = self.fold_gate[layer_idx].bias
                gate = torch.sigmoid(
                    F.linear(acc, Wg[:, :d]) +
                    F.linear(nxt, Wg[:, d:]) + bg)            # [B, T, 1]
                g = gate * validf[None, :, s_idx, None]
                if self.training and self.tree_drop > 0:
                    keep = (torch.rand(B, T, 1, device=device)
                            > self.tree_drop).float()
                    g = g * keep
                acc = g * composed + (1.0 - g) * acc
            return acc

        if self.fold_mode == 'spine':
            # SPINE READOUT (r19): run the compacted left fold, KEEP the
            # running accumulator after every step (the spine: cumulative
            # prefix states, each having absorbed one more block), then
            # let the final state attend over its own spine and compose
            # with the attended context. Motivation is the MEASURED
            # mid-band capacity gap to RoPE (position-curve CE 2.79 vs
            # 2.68): the head currently reads d floats per prefix; the
            # spine offers <= log2(T)+1 progressively-richer states,
            # including early accumulators whose content predates the
            # final squashes. PRE-REGISTERED (r19): (1) prediction is
            # tie-to-small-gain in-length (attend's multi-state tie
            # bounds expectations); (2) the decision metric is the
            # MID-BAND POSITION CURVE, not in-length PPL; (3) no change
            # predicted on probe early-sensitivity (forgetting is
            # learned). If spine ties, the capacity theory's next
            # instrument is k parallel folds (wider channel), not more
            # readout variants.
            acc = gathered[:, :, 0, :]
            snaps = [acc]
            fgb = (self.fold_gate_bias[layer_idx]
                   if self.fold_gate_bias is not None else None)
            for s_idx in range(1, max_blocks):
                act = active[s_idx]
                m = act.numel()
                if m == 0:
                    break
                a = acc.index_select(1, act)
                nxt = gathered[:, act, s_idx, :]
                # v9 arm A: chrono bias on the TRANSPORT composes only;
                # the final ctx compose below is readout, not transport.
                composed, _, _ = self._compose(
                    a.reshape(B * m, d), nxt.reshape(B * m, d),
                    layer_idx, R_L, R_R, R_O, gate_bias=fgb)
                acc = acc.index_copy(1, act, composed.reshape(B, m, d))
                snaps.append(acc)
            S_eff = len(snaps)
            if S_eff == 1:
                return acc
            sp = torch.stack(snaps, dim=2)                   # [B,T,S_eff,d]
            steps = torch.arange(S_eff, device=device)
            validf = (steps[None, :]
                      < count.clamp(max=S_eff)[:, None])     # [T,S_eff]
            dk = self.attn_dk
            q = F.linear(acc, self.spine_q[layer_idx])       # [B,T,dk]
            k = F.linear(sp, self.spine_k[layer_idx])        # [B,T,S,dk]
            step_ids = steps[None, :].expand(count.shape[0], -1)
            k = k + level_sin_enc(step_ids, dk).unsqueeze(0)
            scores = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(dk)
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~validf[None, :, :], neg)
            w = torch.softmax(scores, dim=-1)                # [B,T,S_eff]
            ctx = (w.unsqueeze(-1) * sp).sum(dim=2)          # [B,T,d]
            composed, _, _ = self._compose(
                ctx.reshape(B * T, d), acc.reshape(B * T, d),
                layer_idx, R_L, R_R, R_O)
            out = composed.reshape(B, T, d)
            single = (count <= 1)[None, :, None]
            return torch.where(single, gathered[:, :, 0, :], out)

        if self.fold_mode == 'rack':
            # RACK FOLD (r17): each incoming Fenwick block CONJUGATES the
            # accumulator -- acc ← q̂(next) · acc · q̂(next)⁻¹ + g ⊙ next.
            # The conjugation core is a genuine rack (self-distributive,
            # per-step invertible, exact per-block isometry: old content
            # is ROTATED, never gated down, normed, or squashed -- no
            # LayerNorm, no tanh anywhere in the fold path). The gated
            # injection admits new content and is NOT part of the rack
            # (documented: rack-core, not rack-pure). Quandle lineage:
            # conjugation is the canonical rack; the operation the Cl(3)
            # even algebra was already carrying.
            # PRE-REGISTERED (r17): (1) in-length parity with left;
            # (2) probe early-position sensitivity UNCHANGED (forgetting
            # is learned, per the p2 probe verdict) -- if rack IMPROVES
            # it, the bottleneck theory revives in modified form: norm
            # crushing, not path length, was the mechanism; (3) rack's
            # own claim is fold-path norm stability at depth (>= 4x).
            acc = gathered[:, :, 0, :]
            for s_idx in range(1, max_blocks):
                act = active[s_idx]
                m = act.numel()
                if m == 0:
                    break
                a_flat = acc.index_select(1, act).reshape(B * m, d)
                n_flat = gathered[:, act, s_idx, :].reshape(B * m, d)
                a_blk = a_flat.reshape(B * m, nb, 4)
                n_blk = n_flat.reshape(B * m, nb, 4)
                q = n_blk / (n_blk.norm(dim=-1, keepdim=True) + 1e-8)
                rot = quat_sandwich(q, a_blk)                    # isometry
                Wg = self.rack_gate[layer_idx]
                g = torch.sigmoid(
                    F.linear(a_flat, Wg.weight[:, :d]) +
                    F.linear(n_flat, Wg.weight[:, d:]) + Wg.bias)
                new = rot + g.unsqueeze(-1) * n_blk
                acc = acc.index_copy(1, act, new.reshape(B, m, d))
            if self.rack_exit_norm is not None:
                # r18: normalize ONCE at exit, folded positions only.
                # Single-block positions (count<=1) never entered the
                # fold and must stay bit-identical to --fold left.
                normed = self.rack_exit_norm[layer_idx](acc)
                single = (count <= 1)[None, :, None]
                acc = torch.where(single, gathered[:, :, 0, :], normed)
            return acc

        if self.fold_mode == 'oam':
            # CHARGED MULTI-CHANNEL FOLD (v8.1, "OAM fold"). k parallel
            # accumulators; transport = rack-pure conjugation (exact
            # isometry, charge-free ON PURPOSE -- OAM charge is carried
            # unchanged through propagation and acts at INTERACTION, i.e.
            # at injection). Injection into channel c is twisted by
            # theta_c(l) = m_c * phi * l (level-only dependence:
            # extrapolation-safe like --fold-scale). Readout composes the
            # channels (ascending-charge order, fixed: the compose is not
            # commutative, order is architecture) or softmax-mixes them.
            # k=1 with charge off is BITWISE --fold rack (selftested).
            k = self.oam_k
            charges = self.oam_charge_vec                        # [k]
            phi = self.oam_phi[layer_idx]
            acc = gathered[:, :, 0, :].unsqueeze(2).expand(B, T, k, d).contiguous()
            if self.oam_transport == 'node':
                # v8.2 NODE TRANSPORT: acc_c <- compose(acc_c, twist_c(next))
                # -- the FULL OPERA node per fold step (the transport the
                # docs-scale rack falsification showed is load-bearing),
                # all node params SHARED across channels. Channel identity
                # comes ONLY from the injection twist theta_c(l) =
                # m_c*phi*l (level-only: extrapolation-safe). k=1 with
                # charge 0 is BITWISE --fold left; k>1 with phi=0 makes
                # all channels identical (readout-artifact null),
                # selftested.
                for s_idx in range(1, max_blocks):
                    act = active[s_idx]
                    m = act.numel()
                    if m == 0:
                        break
                    nxt = gathered[:, act, s_idx, :]             # [B, m, d]
                    a = acc.index_select(1, act)                 # [B, m, k, d]
                    n_exp = nxt.unsqueeze(2).expand(B, m, k, d)
                    if self._oam_charge_on:
                        tw = phi * charges[None, :]
                        if self.oam_level_gate is not None:
                            # v8.3: induced charge -- per-level gate
                            # sigma(a_l), init ~0.047, learnable per level
                            lg = torch.sigmoid(self.oam_level_gate[layer_idx][
                                lvl[act, s_idx].clamp(max=15)])
                            tw = lg[:, None].to(tw.dtype) * tw
                        ang = (lvl[act, s_idx].to(phi.dtype)[:, None] * tw)         # [m, k]
                        co = torch.cos(ang)[None, :, :, None]    # [1,m,k,1]
                        si = torch.sin(ang)[None, :, :, None]
                        hb = n_exp.reshape(B, m, k, nb, 4)
                        s0_, vx, vy, vz = (hb[..., 0], hb[..., 1],
                                           hb[..., 2], hb[..., 3])
                        vx2 = vx * co - vy * si
                        vy2 = vx * si + vy * co
                        n_in = torch.stack(
                            [s0_, vx2, vy2, vz], dim=-1).reshape(B * m * k, d)
                    else:
                        n_in = n_exp.reshape(B * m * k, d)
                    if self.oam_chan_emb is not None:
                        v = n_in.reshape(B, m, k, nb, 4)
                        v = torch.stack(
                            [v[..., 0] + self.oam_chan_emb[layer_idx][None, None],
                             v[..., 1], v[..., 2], v[..., 3]], dim=-1)
                        n_in = v.reshape(B * m * k, d)
                    composed, _, _ = self._compose(
                        a.reshape(B * m * k, d), n_in,
                        layer_idx, R_L, R_R, R_O)
                    acc = acc.index_copy(1, act,
                                         composed.reshape(B, m, k, d))
            else:
                if k > 1 and not self.oam_shared_gate:
                    # HOIST (opt2): the per-channel gate stack is constant
                    # across fold steps; stacking it inside the loop was
                    # max_blocks-1 redundant stacks per layer per forward
                    # (compile hoists it, eager MPS/CPU does not). Bitwise
                    # identical output.
                    Wc = torch.stack([self.oam_gate[layer_idx][c].weight
                                      for c in range(k)])            # [k, nb, 2d]
                    bc = torch.stack([self.oam_gate[layer_idx][c].bias
                                      for c in range(k)])            # [k, nb]
                for s_idx in range(1, max_blocks):
                    act = active[s_idx]
                    m = act.numel()
                    if m == 0:
                        break
                    nxt = gathered[:, act, s_idx, :]                 # [B, m, d]
                    a = acc.index_select(1, act)                     # [B, m, k, d]
                    a_flat = a.reshape(B * m * k, d)
                    n_exp = nxt.unsqueeze(2).expand(B, m, k, d).reshape(B * m * k, d)
                    a_blk = a_flat.reshape(B * m * k, nb, 4)
                    n_blk = n_exp.reshape(B * m * k, nb, 4)
                    q = n_blk / (n_blk.norm(dim=-1, keepdim=True) + 1e-8)
                    rot = quat_sandwich(q, a_blk)                    # isometry
                    if self._oam_charge_on:
                        tw = phi * charges[None, :]
                        if self.oam_level_gate is not None:
                            # v8.3: induced charge -- per-level gate
                            # sigma(a_l), init ~0.047, learnable per level
                            lg = torch.sigmoid(self.oam_level_gate[layer_idx][
                                lvl[act, s_idx].clamp(max=15)])
                            tw = lg[:, None].to(tw.dtype) * tw
                        ang = (lvl[act, s_idx].to(phi.dtype)[:, None] * tw)             # [m, k]
                        co = torch.cos(ang)[None, :, :, None]        # [1,m,k,1]
                        si = torch.sin(ang)[None, :, :, None]
                        hb = n_exp.reshape(B, m, k, nb, 4)
                        s0_, vx, vy, vz = hb[..., 0], hb[..., 1], hb[..., 2], hb[..., 3]
                        vx2 = vx * co - vy * si
                        vy2 = vx * si + vy * co
                        n_in = torch.stack([s0_, vx2, vy2, vz], dim=-1).reshape(B * m * k, nb, 4)
                    else:
                        n_in = n_blk
                    if self.oam_shared_gate:
                        Ws = self.oam_gate_shared[layer_idx]
                        base = (F.linear(a_flat, Ws.weight[:, :d])
                                + F.linear(n_exp, Ws.weight[:, d:]))
                        g = torch.sigmoid(
                            base.reshape(B, m, k, nb)
                            * self.oam_scale[layer_idx][None, None, :, None]
                            + self.oam_bias[layer_idx][None, None])  # [B,m,k,nb]
                    elif k == 1:
                        # EXACT rack formulation (bitwise equivalence path).
                        Wg = self.oam_gate[layer_idx][0]
                        g = torch.sigmoid(
                            F.linear(a_flat, Wg.weight[:, :d]) +
                            F.linear(n_exp, Wg.weight[:, d:]) + Wg.bias)
                        g = g.reshape(B, m, k, nb)
                    else:
                        # BUG FIX (opt2): the v8.1 einsum was 'bmki,cni->bmcn'
                        # -- k absent from the output means einsum SUMS over
                        # it: every channel's gate read the k-SUM of all
                        # channels' inputs (pooled, ~k-fold inflated pre-
                        # sigmoid), not its own channel. 'bmki,kni->bmkn'
                        # matches channel to gate: g[...,c,:] is a function of
                        # channel c's accumulator only. Selftest (g) asserts
                        # per-channel semantics against a reference loop.
                        # ANY k>1 non-shared-gate result produced before this
                        # fix used pooled gates and must be re-run.
                        inp = torch.cat([a, n_exp.reshape(B, m, k, d)], dim=-1)
                        g = torch.sigmoid(
                            torch.einsum('bmki,kni->bmkn', inp, Wc)
                            + bc[None, None])                        # [B,m,k,nb]
                    if self.oam_chan_emb is not None:
                        v = n_in.reshape(B, m, k, nb, 4)
                        v = torch.stack(
                            [v[..., 0] + self.oam_chan_emb[layer_idx][None, None],
                             v[..., 1], v[..., 2], v[..., 3]], dim=-1)
                        n_in = v.reshape(B * m * k, nb, 4)
                    new = rot + g.reshape(B * m * k, nb, 1) * n_in
                    acc = acc.index_copy(1, act, new.reshape(B, m, k, d))
            if k == 1:
                return acc[:, :, 0, :]
            if self.oam_combine == 'sum':
                w = torch.softmax(self.oam_alpha[layer_idx], dim=-1)
                out = (acc * w[None, None, :, None]).sum(dim=2)  # [B,T,d]
            else:
                slots = acc                                      # [B,T,k,d]
                if self.oam_pair == 'conj':
                    # (-m,+m) meet first in the pairwise tree; see init.
                    slots = slots.index_select(2, self.oam_pair_perm)
                kk = k
                while kk > 1:
                    npairs = kk // 2
                    cl = slots[:, :, 0:2 * npairs:2, :]
                    cr = slots[:, :, 1:2 * npairs:2, :]
                    comp, _, _ = self._compose(
                        cl.reshape(B * T * npairs, d),
                        cr.reshape(B * T * npairs, d),
                        layer_idx, R_L, R_R, R_O)
                    comp = comp.reshape(B, T, npairs, d)
                    if kk % 2 == 1:
                        comp = torch.cat([comp, slots[:, :, -1:, :]], dim=2)
                    slots = comp
                    kk = slots.shape[2]
                out = slots[:, :, 0, :]
            single = (count <= 1)[None, :, None]
            return torch.where(single, gathered[:, :, 0, :], out)

        if self.fold_mode == 'attend':
            # GATHER-THEN-COMPOSE (v8.0): one attention round over each
            # position's own Fenwick blocks (log T slots -- O(T log T)
            # total, NOT token attention), then ONE composition merging
            # attended context with the most recent block. Every block is
            # one hop from the readout: no serial accumulator, no order
            # asymmetry, no zero-sum norm budget across fold steps.
            # Compacted like the left fold: single-block positions
            # (count==1) bypass and return their block unchanged.
            act = active[1]                          # positions with count > 1
            base = gathered[:, :, 0, :]
            if act.numel() == 0:
                return base
            m = act.numel()
            g_act = gathered[:, act]                             # [B,m,S,d]
            count_a = count[act]                                 # [m]
            lvl_a = lvl[act]                                     # [m,S]
            S = max_blocks
            ar = torch.arange(m, device=device)
            last = g_act[:, ar, (count_a - 1)]                   # [B,m,d]

            Wq = self.fold_attn_q[layer_idx]                     # [dk,d]
            Wk = self.fold_attn_k[layer_idx]
            dk = self.attn_dk
            q = F.linear(last, Wq)                               # [B,m,dk]
            k = F.linear(g_act, Wk)                              # [B,m,S,dk]
            k = k + level_sin_enc(lvl_a, dk).unsqueeze(0)        # level info
            scores = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(dk)  # [B,m,S]
            validf = (torch.arange(S, device=device)[None, :]
                      < count_a[:, None])                        # [m,S]
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~validf[None, :, :], neg)
            w = torch.softmax(scores, dim=-1)                    # [B,m,S]
            ctx = (w.unsqueeze(-1) * g_act).sum(dim=2)           # [B,m,d]

            composed, _, _ = self._compose(
                ctx.reshape(B * m, d), last.reshape(B * m, d),
                layer_idx, R_L, R_R, R_O)
            return base.index_copy(1, act, composed.reshape(B, m, d))

        if (self.fold_mode == 'left' and self.fold_impl == 'downsweep'
                and _LEVEL_CAPTURE is None):
            return self._join_right(
                self._downsweep_fold(levels, T, layer_idx, R_L, R_R, R_O),
                gathered, count, max_blocks, layer_idx, R_L, R_R, R_O)

        if self.fold_mode == 'left':
            # FOLD COMPACTION (v7.9): compose ONLY the active rows per
            # step. Position j participates in step s iff count[j] > s;
            # its inputs (its own acc, its own slot s) and composition
            # order are IDENTICAL to the masked fold, so outputs are
            # exactly equivalent (selftest asserts vs left-masked).
            # active[s] is cached per T (static: compile-safe); the
            # write-back is an out-of-place, differentiable index_copy.
            # Work drops from (max_blocks-1)*T row-compositions to
            # sum_L (popcount(L)-1) -- ~2.3-2.7x less fold work, all of
            # it bandwidth (the measured MPS bottleneck).
            # SCALE-GRADED READOUT: per-step write masks (global slots
            # always update; graded slots only on their group's steps).
            # The compose call itself is unchanged -- the node still sees
            # the full (acc, block) pair; only the accumulator write is
            # masked per quaternion slot (values never mix across slots;
            # the fusion gate may keep conditioning on the full 2d input).
            # The seed (slot-0 block) likewise writes only global + its
            # own group; unrouted groups start at zero -- a group's fold
            # sees only ITS blocks.
            # FOLD_H0: the fold starts from the learned accumulator h0 and
            # composes EVERY path block (step 0 included: active[0] is all
            # positions), so the readout is never a raw tree node. With
            # grading, step 0's write uses the seed mask -- unrouted
            # groups keep h0 (a trained neutral start) instead of zero.
            h0 = (self.fold_h0[layer_idx] if self.fold_h0 is not None
                  else None)
            if self.fold_grade is not None:
                seed_mask, grade_masks = self._grade_fold_masks(
                    T, len(levels), level_offsets, device)
                step_masks = [seed_mask] + grade_masks    # index = s_idx
            else:
                step_masks = None
            if h0 is not None:
                acc = h0.to(gathered.dtype).expand(B, T, d)
                first_step = 0
            elif step_masks is not None:
                seed0 = gathered[:, :, 0, :].reshape(B, T, nb, 4)
                acc = torch.where(seed_mask[None, :, :, None], seed0,
                                  torch.zeros_like(seed0)).reshape(B, T, d)
                first_step = 1
            else:
                acc = gathered[:, :, 0, :]
                first_step = 1
            fgb = (self.fold_gate_bias[layer_idx]
                   if self.fold_gate_bias is not None else None)
            for s_idx in range(first_step, max_blocks):
                act = active[s_idx]                       # [m] static per (T,s)
                m = act.numel()                           # python int at trace
                if m == 0:
                    break
                a = acc.index_select(1, act)              # [B, m, d]
                nxt = gathered[:, act, s_idx, :]          # [B, m, d]
                if _LEVEL_CAPTURE is not None:
                    # fold bucket: slot s_idx's block level varies PER
                    # POSITION (each prefix's Fenwick decomposition),
                    # so the fold's applications of the shared weight
                    # are scale-MIXED within one call. They get their
                    # own capture bucket rather than a fake level; the
                    # kill-switch reports it separately and LO-Muon
                    # gives it its own momentum/whitening.
                    r = (a.detach().square().mean().sqrt()
                         + nxt.detach().square().mean().sqrt()).item() / 2
                    prev = _LEVEL_CAPTURE['rms'].get('fold')
                    _LEVEL_CAPTURE['rms']['fold'] = (
                        r if prev is None else 0.9 * prev + 0.1 * r)
                # v9 arm A: chrono-biased gate on the fold transport
                # (shared weights; only the bias differs from the tree).
                composed, _, _ = self._compose(
                    self._fold_innov(a, nxt, layer_idx).reshape(B * m, d),
                    nxt.reshape(B * m, d),
                    layer_idx, R_L, R_R, R_O, gate_bias=fgb,
                    level_idx=('fold' if _LEVEL_CAPTURE is not None
                               else None), fold=True)
                if self.bist_a is not None:
                    composed = self._bistable_update(
                        a.reshape(B * m, d), nxt.reshape(B * m, d),
                        composed, layer_idx)
                if self.relax_g is not None:
                    composed = self._relax_update(
                        a.reshape(B * m, d), nxt.reshape(B * m, d),
                        composed, layer_idx)
                if step_masks is not None:
                    # masked write: non-routed graded slots keep their
                    # pre-step accumulator values verbatim.
                    keep = step_masks[s_idx]                     # [m, nb]
                    newv = composed.reshape(B, m, nb, 4)
                    oldv = a.reshape(B, m, nb, 4)
                    composed = torch.where(keep[None, :, :, None], newv,
                                           oldv).reshape(B, m, d)
                acc = acc.index_copy(1, act, composed.reshape(B, m, d))
            if self.readout_mode == 'multistate':
                acc = acc + self._multistate_readout(gathered, lvl, count,
                                                     layer_idx)
            return self._join_right(acc, gathered, count, max_blocks,
                                    layer_idx, R_L, R_R, R_O)

        if self.fold_mode == 'left-masked':
            # v7.7 reference implementation, kept verbatim for the
            # equivalence selftest. Composes ALL rows each step and
            # discards masked results with torch.where.
            acc = gathered[:, :, 0, :]
            for s_idx in range(1, max_blocks):
                # NOTE: no data-dependent break here. `bool(tensor.any())`
                # forces a device sync and makes control flow depend on
                # tensor VALUES, which breaks torch.compile (dynamo
                # recompile-limit failure observed in benchmarks). The
                # loop trip count is deterministic per T; iterations past
                # a position's block count are masked no-ops.
                has_slot = (count > s_idx)
                nxt = gathered[:, :, s_idx, :]
                composed, _, _ = self._compose(
                    acc.reshape(B * T, d), nxt.reshape(B * T, d),
                    layer_idx, R_L, R_R, R_O)
                composed = composed.reshape(B, T, d)
                mask = has_slot[None, :, None]
                acc = torch.where(mask, composed, acc)
            return acc

        # --fold balanced: pairwise reduction rounds. Validity per slot is
        # prefix-contiguous (slot s valid iff s < count), and pairing
        # (0,1),(2,3),... preserves contiguity, so after each round the
        # surviving slots are again a contiguous prefix.
        slots = gathered                                          # [B,T,S,d]
        valid = (torch.arange(max_blocks, device=device)[None, :]
                 < count[:, None])                                # [T,S]
        S = max_blocks
        while S > 1:
            n_pairs = S // 2
            left = slots[:, :, 0:2 * n_pairs:2, :]                # [B,T,n_pairs,d]
            right = slots[:, :, 1:2 * n_pairs:2, :]
            composed, _, _ = self._compose(
                left.reshape(B * T * n_pairs, d),
                right.reshape(B * T * n_pairs, d),
                layer_idx, R_L, R_R, R_O)
            composed = composed.reshape(B, T, n_pairs, d)
            right_valid = valid[:, 1:2 * n_pairs:2]               # [T,n_pairs]
            merged = torch.where(right_valid[None, :, :, None], composed, left)
            if S % 2 == 1:
                merged = torch.cat([merged, slots[:, :, -1:, :]], dim=2)
                new_valid = torch.cat(
                    [valid[:, 0:2 * n_pairs:2], valid[:, -1:]], dim=1)
            else:
                new_valid = valid[:, 0:2 * n_pairs:2]
            slots, valid = merged, new_valid
            S = slots.shape[2]
        return slots[:, :, 0, :]

    def _join_right(self, acc, gathered, count, max_blocks, layer_idx,
                    R_L, R_R, R_O):
        """fold_dir='both': acc + gain (.) right_fold. Identity when off."""
        if self.fold_join_gain is None:
            return acc
        right = self._right_fold(gathered, count, max_blocks, layer_idx,
                                 R_L, R_R, R_O)
        B, T, d = acc.shape
        g = self.fold_join_gain[layer_idx].to(acc.dtype)
        return (acc.reshape(B, T, self.nb, 4)
                + g[None, None, :, None]
                * right.reshape(B, T, self.nb, 4)).reshape(B, T, d)

    def _right_fold(self, gathered, count, max_blocks, layer_idx,
                    R_L, R_R, R_O):
        """Counterclockwise (right-nested) fold of each prefix's Fenwick
        blocks: B1 o (B2 o (... o B_S)). Start from the newest block
        (slot count-1); then for s = S-2 .. 0, acc <- compose(block_s, acc)
        -- the older block is the LEFT child (time order preserved).
        Compacted: only positions whose slot s exists and is not their
        last take part in step s."""
        B, T, S, d = gathered.shape
        dev = gathered.device
        tpos = torch.arange(T, device=dev)
        acc = gathered[:, tpos, count - 1, :]                     # newest block
        for s_idx in range(max_blocks - 2, -1, -1):
            act = torch.nonzero(count > s_idx + 1, as_tuple=False).squeeze(-1)
            m = act.numel()
            if m == 0:
                continue
            blk = gathered[:, act, s_idx, :]
            a = acc.index_select(1, act)
            c, _, _ = self._compose(blk.reshape(B * m, d), a.reshape(B * m, d),
                                    layer_idx, R_L, R_R, R_O, fold=True)
            acc = acc.index_copy(1, act, c.reshape(B, m, d))
        return acc

    def _downsweep_fold(self, levels, T, layer_idx, R_L, R_R, R_O):
        """All-prefix left fold, top-down (fold_impl='downsweep').

        E_k[j] = fold of the Fenwick blocks of prefix length j * 2^k
        (largest block first, left-nested) -- stored for j = 1..(T >> k)
        (j = 0 is the empty fold). Coarse to fine:
            E_k[2j]   = E_{k+1}[j]                      (same prefix)
            E_k[2j+1] = compose(E_{k+1}[j], node_k[2j]) (one more, smallest
                                                        block), j >= 1
            E_k[1]    = node_k[0]                       (empty acc: no compose)
        Prefix state of position t = E_0[t + 1]. Same compose chain per
        prefix as the compacted fold (same operands, same order, same
        fold gate/bias/innovation), shared across prefixes."""
        B, d = levels[0].shape[0], levels[0].shape[-1]
        dev = levels[0].device
        fgb = (self.fold_gate_bias[layer_idx]
               if self.fold_gate_bias is not None else None)
        K = len(levels) - 1
        E = levels[K][:, :1]                       # E_K[1..1]
        for k in range(K - 1, -1, -1):
            n = T >> k
            n_odd = (n - 1) // 2 + 1               # indices 2j+1 <= n, j = 0..
            blocks = levels[k][:, 0:2 * n_odd:2]   # node_k[2j]
            parts = [blocks[:, :1]]                # E_k[1] = node_k[0]
            if n_odd > 1:
                acc = E[:, :n_odd - 1]             # E_{k+1}[j], j = 1..n_odd-1
                nxt = blocks[:, 1:]
                c, _, _ = self._compose(
                    self._fold_innov(acc, nxt, layer_idx).reshape(-1, d),
                    nxt.reshape(-1, d), layer_idx, R_L, R_R, R_O,
                    gate_bias=fgb, fold=True)
                odd = torch.cat([parts[0], c.reshape(B, -1, d)], 1)
            else:
                odd = parts[0]
            # interleave: slot 2i (index 2i+1) <- odd[i]; slot 2i+1 (index
            # 2i+2) <- E_{k+1}[i+1]
            n_even = n // 2
            out = torch.empty(B, n, d, device=dev, dtype=odd.dtype)
            out[:, 0::2] = odd[:, :(n + 1) // 2]
            if n_even:
                out[:, 1::2] = E[:, :n_even].to(odd.dtype)
            E = out
        return E

    def _bistable_update(self, acc, nxt, composed, layer_idx):
        """BRC recurrence on the fold accumulator (arXiv:2006.05252).

            acc <- (1 - c) * composed + c * tanh(a (*) acc)

        acc/nxt/composed: [N, d]. `a` and `c` are per BLOCK ([N, nb]),
        broadcast over each block's 4 spinor components, so a block
        commits or does not commit as a unit -- the same granularity the
        fusion gate already uses.

        WHY THE GAIN GOES HERE AND NOT INSIDE THE NODE: `composed` is
        already past comp_norm, and every norm_mode rescales the node's
        output. A gain applied to the node's INPUT would be renormalized
        away before it could bend anything. The fold's accumulator
        update is the one point on this path where a > 1 survives.

        WHY IT CANNOT DIVERGE: the positive feedback sits inside a tanh,
        so |tanh(a*acc)| < 1 regardless of a. The bound is structural,
        not a clamp -- no gradient is discarded.

        Bistability: x -> tanh(a x) has one stable fixed point for
        a <= 1 and two for a > 1 (the negative-slope region opens at
        a = 1). 'mono' caps a below 1 by construction and is the
        capacity-matched control; 'bi' uses BRC's (0, 2).
        """
        N = acc.shape[0]
        h = torch.cat([acc, nxt], dim=-1)
        z = self.bist_a[layer_idx](h)                        # [N, nb]
        if self.fold_bistable == 'bi':
            a = 1.0 + torch.tanh(z)                          # (0, 2)
        else:
            a = torch.sigmoid(z)                             # (0, 1)
        c = torch.sigmoid(self.bist_c[layer_idx](h))         # [N, nb]
        ab = a.unsqueeze(-1)                                 # [N, nb, 1]
        cb = c.unsqueeze(-1)
        accb = acc.reshape(N, self.nb, 4)
        fb = composed.reshape(N, self.nb, 4)
        out = (1.0 - cb) * fb + cb * torch.tanh(ab * accb)
        return out.reshape(N, -1)

    def _relax_update(self, acc, nxt, composed, layer_idx):
        """acc <- acc + gamma (*) (composed - acc), per block.

        gamma = 1 reproduces the incumbent exactly. gamma > 1 overshoots
        PAST composed -- the only tested operation that can place the new
        state outside the interval [acc, composed], and therefore the
        only one that can increase displacement rather than redistribute
        it."""
        N = acc.shape[0]
        z = torch.tanh(self.relax_g[layer_idx](
            torch.cat([acc, nxt], dim=-1)))               # [N, nb]
        if self.fold_relax == 'under':
            z = torch.clamp(z, max=0.0)                   # gamma <= 1
        g = (1.0 + z).unsqueeze(-1)                       # [N, nb, 1]
        ab = acc.reshape(N, self.nb, 4)
        fb = composed.reshape(N, self.nb, 4)
        # Written as (1-g)*acc + g*composed rather than the algebraically
        # identical acc + g*(composed-acc): at g=1 the first form yields
        # `composed` EXACTLY (0*acc + 1*composed), while the second
        # rounds (a + (b-a)) != b in floating point. That exactness is
        # what makes flags-on bitwise the incumbent at init, which is the
        # property the whole design rests on.
        #
        # This is NOT the convex blend that the four falsified arms used.
        # There c was a sigmoid in (0,1), so both coefficients were
        # positive and the result was pinned between the endpoints. Here
        # g > 1 makes (1-g) NEGATIVE -- the state is pushed AWAY from
        # acc, past composed. A negative coefficient is exactly what
        # extrapolation means and what no gate can produce.
        return ((1.0 - g) * ab + g * fb).reshape(N, -1)

    @torch.no_grad()
    def relax_stats(self, acc, nxt, layer_idx):
        """Usage report: is gamma actually driven past 1? An 'over' arm
        that never overshoots is a cold-parameter null, not a failed
        mechanism -- the same distinction that decided the bistable
        study."""
        z = torch.tanh(self.relax_g[layer_idx](
            torch.cat([acc, nxt], dim=-1)))
        if self.fold_relax == 'under':
            z = torch.clamp(z, max=0.0)
        g = 1.0 + z
        return {'gamma_mean': g.mean().item(), 'gamma_max': g.max().item(),
                'frac_overshoot': (g > 1.0).float().mean().item(),
                'frac_deep': (g > 1.5).float().mean().item()}

    @torch.no_grad()
    def bistable_stats(self, acc, nxt, layer_idx):
        """Usage report for H3: what fraction of blocks are actually in
        the bistable regime (a > 1), and the gain/retention means. The
        arm reporting its own usage is what made the phase-1 nulls
        interpretable -- 'bi' that never leaves a <= 1 is a null by cold
        parameters, not a failed mechanism, and the two must not be
        confused."""
        h = torch.cat([acc, nxt], dim=-1)
        z = self.bist_a[layer_idx](h)
        a = (1.0 + torch.tanh(z) if self.fold_bistable == 'bi'
             else torch.sigmoid(z))
        c = torch.sigmoid(self.bist_c[layer_idx](h))
        return {'a_mean': a.mean().item(), 'a_max': a.max().item(),
                'frac_bistable': (a > 1.0).float().mean().item(),
                'c_mean': c.mean().item()}

    def _multistate_readout(self, gathered, lvl, count, layer_idx):
        """T0.4 (roadmap): the head ALSO reads the prefix's <= log T + 1
        raw Fenwick block states, each carrying the attend fold's
        deterministic, extrapolation-safe level encoding, reduced by a
        FIXED learned reduction (static per-slot gates + a zero-init
        output projection). No routing, no softmax, no pairwise
        interaction -- a pure test of whether the mid-band ceiling is a
        width-of-readout problem. Slots beyond the training popcount keep
        their never-trained small random gates; their contribution at
        extrapolated lengths is O(0.02)-scale by construction."""
        B, T, S, d = gathered.shape
        le = level_sin_enc(lvl, d).to(gathered.dtype)            # [T,S,d]
        g = self.readout_gate[layer_idx, :S]                     # [S,d]
        valid = (torch.arange(S, device=gathered.device)[None, :]
                 < count[:, None]).to(gathered.dtype)            # [T,S]
        red = (g * (gathered + le) * valid[None, :, :, None]).sum(dim=2)
        return self.readout_out[layer_idx](red)                  # zero-init

    def _delta_memory(self, h, prefix, layer_idx):
        """T1.4 (roadmap): delta-rule matrix memory alongside the fold.
        M_t = M_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T -- the
        error-correcting write (the old association is removed before the
        new one is written), with keys normalized as in DeltaNet. The
        fold's prefix state queries by content (y_t = M_t q_t); a learned
        gate mixes the zero-init-projected retrieval into the head state.
        Writes/reads are rank-one updates and matvecs: no token-token
        score matrix, no softmax anywhere. The scan runs in fp32 for
        recurrence stability. Padded positions only affect their own
        row's later pad positions, which the loss masks -- masking and
        causality are both safe. Keys/values/beta come from the layer's
        token states h; the query comes from the fold's prefix state.

        TRAINING path: chunkwise WY form (DeltaNet, Yang et al. 2024,
        arXiv:2406.06484): S_t = sum_i u_i k_i^T with pseudo-values
        U = (I+A)^{-1} (beta*V), A[t,i] = beta_t (k_i . k_t) strictly
        lower -- one triangular solve per chunk instead of T sequential
        rank-one updates. Numerically equivalent to the recurrent form
        (_delta_memory_naive; selftest cross-checks the two)."""
        k = F.normalize(self.mem_k[layer_idx](h), dim=-1).float()
        v = self.mem_v[layer_idx](h).float()
        beta = torch.sigmoid(self.mem_beta[layer_idx](h)).float()
        q = self.mem_q[layer_idx](prefix).float()
        y = self._chunk_delta(k, v, beta, q)
        y = y.to(prefix.dtype)
        g = torch.sigmoid(self.mem_gate[layer_idx](
            torch.cat([prefix, y], dim=-1)))
        return g * self.mem_out[layer_idx](y)     # zero-init projection

    def _chunk_delta(self, k, v, beta, q, chunk_size=64):
        """Chunkwise delta rule. k,q: [B,T,dk]; v: [B,T,dv]; beta: [B,T,1].
        Returns y: [B,T,dv] with y_t = S_t q_t (post-write state).
        Inter-chunk state S0 [B,dv,dk] is additive: S0 += U^T K."""
        B, T, dk = k.shape
        S0 = k.new_zeros(B, v.shape[-1], dk)
        ys = []
        for s in range(0, T, chunk_size):
            K, V = k[:, s:s + chunk_size], v[:, s:s + chunk_size]
            Q, Bt = q[:, s:s + chunk_size], beta[:, s:s + chunk_size]
            C = K.shape[1]
            # A[t,i] = beta_t (k_i . k_t), strictly lower (i < t)
            A = (K @ K.transpose(-1, -2)) * Bt
            A = A.tril(-1)
            # U = (I + A)^{-1} (beta * V): one triangular solve
            eye = torch.eye(C, device=k.device, dtype=k.dtype).expand(
                B, C, C)
            U = torch.linalg.solve_triangular(
                eye + A, Bt * V, upper=False, unitriangular=True)
            # y = Q S0^T + (tril(Q K^T)) U
            causal = torch.tril(
                torch.ones(C, C, device=k.device, dtype=torch.bool))
            y = Q @ S0.transpose(-1, -2) + (Q @ K.transpose(-1, -2)
                                            * causal) @ U
            S0 = S0 + U.transpose(-1, -2) @ K
            ys.append(y)
        return torch.cat(ys, dim=1)

    def _delta_memory_naive(self, h, prefix, layer_idx):
        """Reference recurrent form of _delta_memory (the original
        sequential scan, 2026-08-04). Kept for the chunked-form
        equivalence selftest; inference (OperaDecoder) uses this form
        too, where it is optimal -- one rank-one update per token."""
        B, T, _ = h.shape
        dm = self.mem_dim
        k = F.normalize(self.mem_k[layer_idx](h), dim=-1).float()
        v = self.mem_v[layer_idx](h).float()
        beta = torch.sigmoid(self.mem_beta[layer_idx](h)).float()
        q = self.mem_q[layer_idx](prefix).float()
        M = torch.zeros(B, dm, dm, device=h.device, dtype=torch.float32)
        ys = []
        for t in range(T):
            kt = k[:, t].unsqueeze(-1)                           # [B,dm,1]
            bt = beta[:, t].unsqueeze(-1)                        # [B,1,1]
            # M + beta * (v - M k) k^T  ==  M(I - beta k k^T) + beta v k^T
            M = M + bt * ((v[:, t].unsqueeze(-1) - M @ kt)
                          @ kt.transpose(-1, -2))
            ys.append((M @ q[:, t].unsqueeze(-1)).squeeze(-1))   # [B,dm]
        y = torch.stack(ys, dim=1).to(prefix.dtype)              # [B,T,dm]
        g = torch.sigmoid(self.mem_gate[layer_idx](
            torch.cat([prefix, y], dim=-1)))
        return g * self.mem_out[layer_idx](y)     # zero-init projection

    def _ws_masks(self, T, device):
        """Workspace chunk masks, host-computed once per (T, device) and
        cached (static: compile-safe, same convention as _fenwick_indices).
        rm [T, nc*K]: position i may read latents of chunks STRICTLY
        before chunk(i) -- earlier chunks are fully in the past; the
        position's own chunk is excluded (intra-chunk context already
        flows through the scan, and including it would leak future
        tokens within the chunk). pv [nc, C]: pool-valid positions
        (T not a multiple of C -> tail padding is masked out)."""
        key = (T, str(device))
        ent = self._ws_cache.get(key)
        if ent is None:
            C, K = self.ws_chunk, self.ws_k
            nc = (T + C - 1) // C
            chunk_of = torch.arange(T, device=device) // C
            rm = (chunk_of[:, None]
                  > torch.arange(nc, device=device)[None, :])   # [T,nc]
            rm = rm[:, :, None].expand(T, nc, K).reshape(T, nc * K)
            pv = (torch.arange(nc * C, device=device) < T).reshape(nc, C)
            ent = (rm, pv)
            self._ws_cache[key] = ent
        return ent

    def _workspace_read(self, x, layer_idx):
        """RANK-1 WORKSPACE read path (v8.9). x: [B,T,d] pre-readout scan
        states. POOL: per chunk of C positions, K learned latent queries
        attend to the chunk's states (routing attention -- values are the
        raw states) -> latents [B,nc,K,d]. READ: each position queries
        latents of strictly earlier chunks -> [B,T,d]. O(T*K*nc): linear
        in T. The -inf mask uses finfo.min (bf16/fp16-safe); rows with no
        valid latents (chunk 0) softmax to uniform and are zeroed by the
        mask multiply -- no NaN, exact zeros."""
        B, T, d = x.shape
        C, K, dk = self.ws_chunk, self.ws_k, self.ws_dk
        nc = (T + C - 1) // C
        pad = nc * C - T
        rm, pv = self._ws_masks(T, x.device)
        xp = (torch.cat([x, x.new_zeros(B, pad, d)], dim=1)
              if pad else x)
        xc = xp.reshape(B, nc, C, d)
        # POOL latents
        qp = F.linear(self.ws_latq[layer_idx],
                      self.ws_qp[layer_idx])                  # [K,dk]
        kp = torch.matmul(xc, self.ws_kp[layer_idx].t())      # [B,nc,C,dk]
        sp = torch.matmul(kp, qp.t()) / math.sqrt(dk)         # [B,nc,C,K]
        sp = sp.masked_fill(~pv[None, :, :, None],
                            torch.finfo(sp.dtype).min)
        wp = torch.softmax(sp, dim=2)
        lat = torch.einsum('bnck,bncd->bnkd', wp, xc)         # [B,nc,K,d]
        # READ (strictly-earlier-chunk mask -> causal by construction)
        qr = torch.matmul(x, self.ws_qr[layer_idx].t())       # [B,T,dk]
        kr = torch.matmul(lat, self.ws_kr[layer_idx].t())     # [B,nc,K,dk]
        vr = torch.matmul(lat, self.ws_v[layer_idx].t())      # [B,nc,K,dk]
        sr = torch.einsum('bte,bnke->btnk', qr, kr) / math.sqrt(dk)
        sr = sr.reshape(B, T, nc * K).masked_fill(
            ~rm[None], torch.finfo(sr.dtype).min)
        wr = (torch.softmax(sr, dim=-1)
              * rm[None].to(sr.dtype)).reshape(B, T, nc, K)
        out = torch.einsum('btnk,bnke->bte', wr, vr)
        return torch.matmul(out, self.ws_o[layer_idx].t())    # [B,T,d]

    def scan_prefix(self, h, layer_idx):
        """OPERA-SCAN fold (v8.6): replaces build_tree + prefix_states.
        Leaves h_i -> (q_i, b_i): q_i homogeneous -- unit direction from
        scan_wq (normalized ONCE at the leaf, not in-scan) times a learned
        input-dependent magnitude s_i in (0,1) from scan_wm (LRU-style
        exp(-softplus)); Hillis-Steele scan under the associative
        CONFORMAL compose (decay rides inside the law: older content is
        multiplied by the magnitudes of all newer leaves) -> ALL prefix
        states in O(log T) depth, then ONE readout per position:
        gate-vs-identity -> LayerNorm -> tanh(+0.1x). No gate/norm/tanh
        inside the scan (design §2/§3)."""
        B, T, d = h.shape
        nbs = self.scan_nbs
        # FUSED LEAF GEMM (v8.8 speed): scan_wq/scan_wb/scan_wm/scan_gate
        # all read h -- one runtime-concatenated GEMM instead of four
        # (state_dict keys, RNG stream, param count UNCHANGED: the cat is
        # a runtime op; backward splits the grads back to the four
        # modules. Same math up to float-op reordering in the GEMM).
        W = torch.cat([self.scan_wq[layer_idx].weight,
                       self.scan_wb[layer_idx].weight,
                       self.scan_wm[layer_idx].weight,
                       self.scan_gate[layer_idx].weight], dim=0)
        bias = torch.cat([self.scan_wq[layer_idx].bias,
                          self.scan_wb[layer_idx].bias,
                          self.scan_wm[layer_idx].bias,
                          self.scan_gate[layer_idx].bias], dim=0)
        leaf = F.linear(h, W, bias)                    # [B,T,10*nbs]
        q = leaf[..., :4 * nbs].reshape(B, T, nbs, 4)
        b = leaf[..., 4 * nbs:8 * nbs].reshape(B, T, nbs, 4)
        # homogeneous leaf: direction x decay magnitude (LRU-style, (0,1))
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        mag = torch.exp(-F.softplus(leaf[..., 8 * nbs:9 * nbs]))  # [B,T,nbs]
        q = q * mag.unsqueeze(-1)
        q, b = associative_scan(q, b)
        # READOUT: q normalized HERE (scale-invariant: the decay product
        # |q| is discarded -- the accumulated decay already acts on b,
        # and LayerNorm handles the residual scale; deliberately NOT
        # dividing b by |q|, which would UNDO the decay). Gate mixes the
        # prefix state with the identity baseline. At long T |q| may
        # underflow toward 0 (decay < 1): qn -> 0, no NaN, the gate
        # baseline keeps the readout well-defined.
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        feat = torch.cat([qn, b], dim=-1)              # [B,T,nbs,8]
        g = torch.sigmoid(leaf[..., 9 * nbs:])            # [B,T,nbs]
        mixed = (g.unsqueeze(-1) * feat
                 + (1.0 - g.unsqueeze(-1)) * self.scan_base)
        prefix = self.scan_norm[layer_idx](mixed.reshape(B, T, d))
        if self.ws_gate is not None:
            # v8.9 WORKSPACE: content-addressable read over past chunks,
            # added post-LayerNorm through a ZERO-INIT per-channel gate
            # (exact identity at init, same trick as --salience).
            prefix = (prefix + self.ws_gate[layer_idx]
                      * self._workspace_read(mixed.reshape(B, T, d),
                                             layer_idx))
        if self.salience is not None:
            # v8.7 SALIENCE FiLM: per-feature gamma/beta driven by the
            # position's OWN accumulated context `mixed` (the pre-LN
            # gated scan state -- chosen over h because h is token-local;
            # the arm's claim is top-down modulation by ACCUMULATED
            # context). Causal by construction (mixed is a prefix state).
            # Placement: AFTER LayerNorm (gamma acts on normalized
            # features -- a true per-feature gain -- and beta a true
            # bias), BEFORE tanh (modulation passes through the bounded
            # nonlinearity: no activation blow-up, beta shifts the
            # operating point). Zero-init final projection -> identity.
            film = self.salience[layer_idx](mixed.reshape(B, T, d))
            prefix = (1.0 + film[..., :d]) * prefix + film[..., d:]
        return torch.tanh(prefix) + 0.1 * prefix

    def forward(self, token_ids, lengths, return_tree=False, return_levels=False,
                return_states=False, head_last_only=False, return_energy=False,
                geom=None, geom_mask=None, geom_block=0):
        B, T = token_ids.shape
        device = token_ids.device
        d = self.d_model             # stream width (== self.d unless state_mult)

        states = self.word_emb(token_ids)
        if self.pe_mode == 'sin':
            states = states + sinusoidal_pos_enc(T, d, device).unsqueeze(0)
        elif self.pe_mode == 'rotor':
            key = (T, str(device))
            if key not in self._rotor_cache:
                self._rotor_cache[key] = rotor_pos_tables(T, self.nb_model,
                                                          device)
            cos, sin = self._rotor_cache[key]
            states = apply_rotor_pe(states, cos, sin, self.nb_model)
        # pe_mode == 'none': tree/Fenwick structure is the only position source
        if self.dropout > 0:
            states = F.dropout(states, p=self.dropout, training=self.training)
        if geom is not None:
            # Non-LM callers only (e.g. a spatial-reasoning task): splice
            # literal geometric vectors into specific leaves' block
            # `geom_block` BEFORE the tree ever sees them, so the tree's
            # existing rotation/geometric-product composition operates on
            # real coordinates for those leaves, not learned embeddings.
            states = inject_geometry(states, geom, geom_mask, self.nb_model,
                                     geom_block)

        pad = None  # on-fly indexing: tree built without padding

        # Instance state read by _compose/compose_pair_batch further down
        # this same forward call -- not thread-safe if forward() is ever
        # invoked concurrently on one shared module instance (DDP is
        # multi-process, so this doesn't currently apply, but torch.compile
        # or a future multi-threaded caller would race on it).
        self._need_locks = return_tree
        self._need_energy = return_energy
        self._rot_dense_cache.clear()
        per_layer_prefix = []
        tree_info = []
        per_layer_levels = []
        per_layer_energies = []

        def _layer_body(current, layer_idx, T):
            # Tree input: the stream itself (incumbent), pre-normed
            # (resid_mode='add'), then lifted to tree width (state_mult).
            x = current
            if self.tree_in_norm is not None:
                x = self.tree_in_norm[layer_idx](x)
            if self.tree_in is not None:
                x = self.tree_in[layer_idx](x)
            if self.fold_mode == 'scan':
                # no tree: levels/locks/energies are empty (msup/tree-
                # inspection/energy-diagnostic N/A)
                return self.scan_prefix(x, layer_idx), [], [], []
            R_L, R_R, R_O = self.get_rotations(layer_idx)
            levels, locks, energies = self.build_tree(
                x, layer_idx, R_L, R_R, R_O)
            prefix = self.prefix_states(levels, T, layer_idx, R_L, R_R, R_O)
            if self.mem_mode == 'delta':
                # T1.4: the memory reads the layer's token states, the
                # fold's prefix state queries it by content.
                prefix = prefix + self._delta_memory(x, prefix, layer_idx)
            if self.tree_out is not None:
                prefix = self.tree_out[layer_idx](prefix)
            if self.hmem_nb:
                prefix = prefix + self._hmem_read(current, layer_idx)
            return prefix, levels, locks, energies

        current = states
        for layer_idx in range(self.num_layers):
            if self.grad_checkpoint == 'layer' and self.training:
                from torch.utils.checkpoint import checkpoint
                assert not (return_tree or return_levels or return_energy), \
                    "--checkpoint layer is incompatible with tree/level/energy outputs"
                prefix = checkpoint(
                    lambda c, li=layer_idx: _layer_body(c, li, T)[0],
                    current, use_reentrant=False)
                levels = locks = energies = None
            else:
                prefix, levels, locks, energies = _layer_body(current, layer_idx, T)
            if return_tree:
                tree_info.append((levels, locks))
            if return_levels:
                per_layer_levels.append(levels)
            if return_energy:
                per_layer_energies.append(energies)

            if (self.head_mode == 'fold'
                    and layer_idx == self.num_layers - 1
                    and not (self.dropout > 0 and self.training)):
                # The head reads the fold output; the last layer's stream
                # update would feed nothing (its cross_mlp/blend_gate get
                # no gradient either way) -- skip computing it. Not under
                # training dropout: its F.dropout draw advances the global
                # RNG, and skipping it would shift the batch stream.
                per_layer_prefix.append(prefix)
                break
            mixed = self.cross_mlp[layer_idx](prefix)
            if self.dropout > 0:
                mixed = F.dropout(mixed, p=self.dropout, training=self.training)
            if self.resid_mode == 'add':
                current = current + mixed
            else:
                gate = self.blend_gate[layer_idx](prefix)
                current = gate * mixed + (1 - gate) * current
            # per_layer_prefix holds each layer's HEAD INPUT (what `states`
            # returns and train_lm_loss applies the head to): the fold
            # output (incumbent) or the normalized post-update stream.
            per_layer_prefix.append(
                self.stream_norm[layer_idx](current)
                if self.head_mode == 'stream' else prefix)

        if head_last_only:
            # OPT: eval/probe only read [-1]; skip aux-layer head GEMMs.
            # lm_loss with L=1 reduces to the final-layer term, so reported
            # PPL is IDENTICAL to the 4-logit path.
            all_logits = [self.apply_head(per_layer_prefix[-1])]
        else:
            all_logits = [self.apply_head(p) for p in per_layer_prefix]

        return OperaOutput(
            logits=all_logits,
            tree=tree_info if return_tree else None,
            levels=per_layer_levels if return_levels else None,
            states=per_layer_prefix if return_states else None,
            energy=per_layer_energies if return_energy else None,
        )



def count_params(model):
    return sum(p.numel() for p in model.parameters())
