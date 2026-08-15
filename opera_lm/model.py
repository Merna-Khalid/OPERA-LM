"""OperaSpinorFenwickTree: spinor (Cl(3)) Fenwick-tree language model."""
import math
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


def relative_lock(s0, v0, s1, v1, eps=1e-8):
    n0 = torch.sqrt(s0 * s0 + (v0 * v0).sum(-1) + eps)
    n1 = torch.sqrt(s1 * s1 + (v1 * v1).sum(-1) + eps)
    inner = (s0 * s1 + (v0 * v1).sum(-1)) / (n0 * n1)
    return (1.0 - inner * inner).clamp(0.0, 1.0)


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


# ============================================================================
# OPERA v7.9 — Spinor Tree
# ============================================================================

class OperaSpinorFenwickTree(nn.Module):
    def __init__(self, vocab_size, d=768, nb=192, num_layers=2, lock_mode='none',
                 tie=False, dropout=0.0, pe_mode='sin',
                 fold_mode='left', fold_rotors='shared', fold_scale=False,
                 norm_mode='layer', act_mode='tanh', node_residual=False,
                 tree_drop=0.0, grad_checkpoint='', use_metal=False,
                 rot_mode='so3', oam_k=4, oam_charges='auto',
                 oam_phi=0.7853981633974483, oam_shared_gate=False,
                 oam_combine='compose', oam_pair='seq', rack_exitnorm=False,
                 oam_transport='rack', oam_levelgate=False,
                 oam_chan_emb=False, scan_salience=False, scan_decay_bias=-3.0,
                 workspace=False, fold_gate_bias=None,
                 readout_mode='none', readout_max_slots=16,
                 mem_mode='none', mem_dim=128):
        super().__init__()
        assert d == 4 * nb, f"d must equal 4*nb (got d={d}, nb={nb})"
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
            assert fold_rotors == 'shared' and not fold_scale, \
                "--fold-rotors/--fold-scale tune the Fenwick fold, not the scan"
            assert not node_residual and tree_drop == 0.0, \
                "--node-residual/--tree-drop tune the compose node, not the scan"
            assert norm_mode == 'layer' and act_mode == 'tanh', \
                "--fold scan readout is fixed LN + tanh(+0.1x)"
            assert grad_checkpoint != 'level', \
                "--checkpoint level wraps compose nodes; use 'layer' with scan"
        assert oam_combine in ('compose', 'sum')
        assert oam_pair in ('seq', 'conj')
        assert oam_transport in ('rack', 'node')
        assert oam_k >= 1
        assert fold_rotors in ('shared', 'separate')
        assert norm_mode in ('layer', 'blockrms', 'rms')
        assert act_mode in ('tanh', 'linear')
        self.norm_mode = norm_mode
        self.act_mode = act_mode
        self.node_residual = node_residual
        self.tree_drop = tree_drop
        self.grad_checkpoint = grad_checkpoint
        self.use_metal = use_metal
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

        self.word_emb = nn.Embedding(vocab_size, d, padding_idx=0)

        if tie:
            # Tied head + THE INIT FIX. Tying a N(0,1) embedding straight
            # into the head gives initial logits of magnitude ~sqrt(d)
            # (v6.7 confound; v7.2 init PPL 3.9e42). A learned scalar
            # logit_scale init 1/sqrt(d) restores initial loss ~ ln(V).
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.head.weight = self.word_emb.weight
            self.logit_scale = nn.Parameter(torch.tensor(d ** -0.5))
        else:
            self.head = nn.Linear(d, vocab_size)
            self.logit_scale = None

        if fold_mode == 'scan':
            # --fold scan (v8.5): the tree's per-block rotors are replaced
            # by learned relative rotations carried IN the scan (q channel).
            self.quat = None
            self.rot_free = None
        elif rot_mode == 'so3':
            q_init = torch.zeros(num_layers, 3, nb, 4)
            q_init[..., 0] = 1.0
            q_init += torch.randn_like(q_init) * 0.1
            self.quat = nn.Parameter(q_init)
            self.rot_free = None
        else:
            # --rot free (r15): unconstrained 3x3 per block per role per
            # layer. Init identity + 0.1 noise -- near-identity like the
            # so3 init, so the two arms start in comparable regimes.
            m_init = torch.eye(3).expand(num_layers, 3, nb, 3, 3).clone()
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
            self.fusion_gate = nn.ModuleList([nn.Linear(2 * d, 3 * nb) for _ in range(num_layers)])
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
            nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
            for _ in range(num_layers)
        ])
        self.blend_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d // 4), nn.GELU(), nn.Linear(d // 4, 1), nn.Sigmoid())
            for _ in range(num_layers)
        ])

        self._fenwick_cache = {}
        self._rot_dense_cache = {}

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
            gb_init = torch.empty(num_layers, 3, nb)
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

    def apply_head(self, h):
        logits = self.head(h)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        return logits

    def get_rotations(self, layer_idx, fold=False):
        if self.rot_mode == 'free':
            src = (self.rot_free_fold
                   if (fold and self.rot_free_fold is not None)
                   else self.rot_free)
            M = src[layer_idx]
            return M[0], M[1], M[2]
        src = self.quat_fold if (fold and self.quat_fold is not None) else self.quat
        q = src[layer_idx]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        R = quat_to_rotmat(q)
        return R[0], R[1], R[2]

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

    def _compose(self, h_left, h_right, layer_idx, R_L, R_R, R_O,
                 gate_bias=None):
        """compose_pair_batch, optionally under activation checkpointing:
        backward recomputes the node's ~18 intermediates from its two
        inputs instead of storing them (~25-35% slower steps for a
        several-fold activation-memory reduction). gate_bias (v9 arm A):
        optional [3, nb] replacement for the fusion-gate bias -- the
        fold's chrono init; None reproduces the incumbent node exactly."""
        if self.grad_checkpoint == 'level' and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.compose_pair_batch, h_left, h_right,
                              layer_idx, R_L, R_R, R_O, gate_bias,
                              use_reentrant=False)
        return self.compose_pair_batch(h_left, h_right, layer_idx,
                                       R_L, R_R, R_O, gate_bias)

    def compose_pair_batch(self, h_left, h_right, layer_idx, R_L, R_R, R_O,
                           gate_bias=None):
        N = h_left.shape[0]
        nb = self.nb
        gb = (self.fusion_gate[layer_idx].bias if gate_bias is None
              else gate_bias.reshape(-1))

        hl = h_left.reshape(N, nb, 4)
        hr = h_right.reshape(N, nb, 4)
        if self.use_metal and self.lock_mode == 'none':
            # KERNEL V2: entire node pre-norm in ONE kernel (rotations +
            # geometric product + gated combine + output rotation).
            # Eager keeps only the gate matmul and the norm. The
            # diagnostic lock is SKIPPED here (~8 elementwise passes of
            # pure overhead per node) except during tree inspection.
            from .metal_kernel import fused_node
            W = self.fusion_gate[layer_idx].weight
            b = gb
            g = (F.linear(h_left, W[:, :self.d]) +
                 F.linear(h_right, W[:, self.d:]) + b)
            g = torch.sigmoid(g.reshape(N, 3, nb))
            parent = fused_node(hl, hr, R_L, R_R, R_O, g).reshape(N, -1)
            if getattr(self, '_need_locks', False):
                with torch.no_grad():
                    v_l = hl[..., 1:]; v_r = hr[..., 1:]
                    v0d = torch.einsum('kij,nkj->nki', R_L, v_l)
                    v1d = torch.einsum('kij,nkj->nki', R_R, v_r)
                    lock_scalar = relative_lock(
                        hl[..., 0], v0d, hr[..., 0], v1d).mean(-1, keepdim=True)
            else:
                lock_scalar = torch.zeros(N, 1, device=h_left.device)
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
                from .metal_kernel import fused_act
                parent = fused_act(parent)
            if self.res_logit is not None:
                r = torch.sigmoid(self.res_logit[layer_idx])
                parent = r * parent + (1.0 - r) * 0.5 * (h_left + h_right)
            return parent, lock_scalar
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
        W = self.fusion_gate[layer_idx].weight
        b = gb
        g = torch.addmm(torch.addmm(b, h_left, W[:, :self.d].t()),
                        h_right, W[:, self.d:].t())
        g = torch.sigmoid(g.reshape(N, 3, nb))
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
        M_O = self._dense_rot(R_O)
        if M_O is not None:
            fv = (fv.reshape(N, -1) @ M_O).reshape(N, nb, 3)
        else:
            fv = torch.einsum('kij,nkj->nki', R_O, fv)

        parent = torch.cat([fs.unsqueeze(-1), fv], dim=-1).reshape(N, -1)
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
        return parent, lock_scalar

    def build_tree(self, states, layer_idx, R_L, R_R, R_O):
        """ON-FLY INDEXING: no padding anywhere. A Fenwick block (k, j)
        is referenced only when (j+1)*2^k <= T, so 'partial' parents that
        would cover padding are NEVER read -- computing them (as the old
        padded version did) was pure waste: up to ~50% of tree work when
        T sits just above a power of two. Level k now holds exactly
        floor(prev/2) nodes; the referenced set (and therefore every
        output) is IDENTICAL to the padded version."""
        B, n0, d = states.shape
        levels = [states]
        locks = []
        current = states
        while current.shape[1] >= 2:
            n = current.shape[1]
            m = n // 2
            left = current[:, 0:2 * m:2, :]
            right = current[:, 1:2 * m:2, :]
            N = B * m
            parent, lock = self._compose(
                left.reshape(N, d), right.reshape(N, d), layer_idx, R_L, R_R, R_O)
            current = parent.reshape(B, m, d)
            levels.append(current)
            locks.append(lock.reshape(B, m))
        return levels, locks

    def _fenwick_indices(self, T, num_levels, level_offsets, device):
        key = (T, num_levels)
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

        if self.fold_mode == 'revolving':
            acc = gathered[:, :, 0, :]
            validf = (torch.arange(max_blocks, device=device)[None, :]
                      < count[:, None]).float()               # [T, S]
            for s_idx in range(1, max_blocks):
                nxt = gathered[:, :, s_idx, :]
                composed, _ = self._compose(
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
                composed, _ = self._compose(
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
            composed, _ = self._compose(
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
                    composed, _ = self._compose(
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
                    comp, _ = self._compose(
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

            composed, _ = self._compose(
                ctx.reshape(B * m, d), last.reshape(B * m, d),
                layer_idx, R_L, R_R, R_O)
            return base.index_copy(1, act, composed.reshape(B, m, d))

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
            acc = gathered[:, :, 0, :]
            fgb = (self.fold_gate_bias[layer_idx]
                   if self.fold_gate_bias is not None else None)
            for s_idx in range(1, max_blocks):
                act = active[s_idx]                       # [m] static per (T,s)
                m = act.numel()                           # python int at trace
                if m == 0:
                    break
                a = acc.index_select(1, act)              # [B, m, d]
                nxt = gathered[:, act, s_idx, :]          # [B, m, d]
                # v9 arm A: chrono-biased gate on the fold transport
                # (shared weights; only the bias differs from the tree).
                composed, _ = self._compose(
                    a.reshape(B * m, d), nxt.reshape(B * m, d),
                    layer_idx, R_L, R_R, R_O, gate_bias=fgb)
                acc = acc.index_copy(1, act, composed.reshape(B, m, d))
            if self.readout_mode == 'multistate':
                acc = acc + self._multistate_readout(gathered, lvl, count,
                                                     layer_idx)
            return acc

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
                composed, _ = self._compose(
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
            composed, _ = self._compose(
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
                return_states=False, head_last_only=False):
        B, T = token_ids.shape
        device = token_ids.device
        d = self.d

        states = self.word_emb(token_ids)
        if self.pe_mode == 'sin':
            states = states + sinusoidal_pos_enc(T, d, device).unsqueeze(0)
        elif self.pe_mode == 'rotor':
            key = (T, str(device))
            if key not in self._rotor_cache:
                self._rotor_cache[key] = rotor_pos_tables(T, self.nb, device)
            cos, sin = self._rotor_cache[key]
            states = apply_rotor_pe(states, cos, sin, self.nb)
        # pe_mode == 'none': tree/Fenwick structure is the only position source
        if self.dropout > 0:
            states = F.dropout(states, p=self.dropout, training=self.training)

        pad = None  # on-fly indexing: tree built without padding

        self._need_locks = return_tree
        self._rot_dense_cache.clear()
        per_layer_prefix = []
        tree_info = []
        per_layer_levels = []

        def _layer_body(current, layer_idx, T):
            if self.fold_mode == 'scan':
                # no tree: levels/locks are empty (msup/tree-inspection N/A)
                return self.scan_prefix(current, layer_idx), [], []
            R_L, R_R, R_O = self.get_rotations(layer_idx)
            levels, locks = self.build_tree(current, layer_idx, R_L, R_R, R_O)
            prefix = self.prefix_states(levels, T, layer_idx, R_L, R_R, R_O)
            if self.mem_mode == 'delta':
                # T1.4: the memory reads the layer's token states, the
                # fold's prefix state queries it by content.
                prefix = prefix + self._delta_memory(current, prefix,
                                                     layer_idx)
            return prefix, levels, locks

        current = states
        for layer_idx in range(self.num_layers):
            if self.grad_checkpoint == 'layer' and self.training:
                from torch.utils.checkpoint import checkpoint
                assert not (return_tree or return_levels), \
                    "--checkpoint layer is incompatible with tree/level outputs"
                prefix = checkpoint(
                    lambda c, li=layer_idx: _layer_body(c, li, T)[0],
                    current, use_reentrant=False)
                levels = locks = None
            else:
                prefix, levels, locks = _layer_body(current, layer_idx, T)
            per_layer_prefix.append(prefix)
            if return_tree:
                tree_info.append((levels, locks))
            if return_levels:
                per_layer_levels.append(levels)

            mixed = self.cross_mlp[layer_idx](prefix)
            if self.dropout > 0:
                mixed = F.dropout(mixed, p=self.dropout, training=self.training)
            gate = self.blend_gate[layer_idx](prefix)
            current = gate * mixed + (1 - gate) * current

        if head_last_only:
            # OPT: eval/probe only read [-1]; skip aux-layer head GEMMs.
            # lm_loss with L=1 reduces to the final-layer term, so reported
            # PPL is IDENTICAL to the 4-logit path.
            all_logits = [self.apply_head(per_layer_prefix[-1])]
        else:
            all_logits = [self.apply_head(p) for p in per_layer_prefix]

        extras = []
        if return_tree:
            extras.append(tree_info)
        if return_levels:
            extras.append(per_layer_levels)
        if return_states:
            extras.append(per_layer_prefix)
        if extras:
            return (all_logits, *extras)
        return all_logits



def count_params(model):
    return sum(p.numel() for p in model.parameters())
