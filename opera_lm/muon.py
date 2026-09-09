"""Muon optimizer (Newton-Schulz orthogonalized momentum) for OPERA-LM.

Implements the roadmap's T0.1 arm: Muon for matrix-shaped hidden
parameters, AdamW for the rest, in ONE optimizer (one state_dict, so the
train()/resume machinery is untouched). Groups carry a `use_muon` flag;
non-Muon groups get plain AdamW (weight_decay=0, matching the incumbent
recipe's Adam-equivalent math).

Why this fits OPERA specifically: the per-block 3x3 maps (rot_free:
[L, 3, nb, 3, 3]) are the most literally matrix-shaped parameters in any
sequence model. Newton-Schulz runs BATCHED over the leading dims, so each
block's 3x3 map is orthogonalized individually -- the update respects the
block-diagonal structure rather than flattening through it.

Partition rule (split_muon_params): a parameter goes to Muon iff it is
a matrix (ndim >= 2) and its name contains none of {'emb', 'gate',
'quat', 'theta', 'beta'} -- i.e. embeddings, gates, quaternion rotors,
and gain vectors stay on AdamW, per the standard Muon partitioning
(embeddings and scalar/gain parameters are not orthogonalizable in any
meaningful basis; quaternion rotors are not matrices). 'beta' covers
the T1.4 delta-memory write-strength gate (mem_beta, shape [1, d]):
it is a gate in function, just not named mem_gate, and a (1, d)
"matrix" has nothing to orthogonalize against -- Newton-Schulz on it
degenerates to a unit-norm rescale of the gradient row at the matrix
LR instead of the gate LR.

Reference: Keller Jordan's modded-nanogpt Muon; the 350M linear-
attention bake-off protocol that found Muon dominant across four
recurrent architectures (roadmap Section 3.1).
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Batched Newton-Schulz iteration to orthogonalize G: (..., m, n).

    Returns a matrix with (approximately) orthonormal rows (m <= n) or
    columns (m > n), same shape and dtype as G. Quintic coefficients from
    Keller Jordan; the iteration runs in bfloat16 for speed (fp32 matmul
    precision is unnecessary for a 5-step fixed-point polish)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for `use_muon` groups, AdamW for the rest.

    Muon update (per parameter, matrices batched over leading dims):
        buf <- momentum * buf + grad
        d   <- grad + momentum * buf   (nesterov) else buf
        p   <- p - lr * NS5(d) * max(1, m/n)**0.5
    """

    def __init__(self, param_groups, lr=1e-3, momentum=0.95, nesterov=True,
                 ns_steps=5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps,
                        use_muon=False, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group['use_muon']:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr = group['lr']
        momentum = group['momentum']
        names = group.get('names') or [''] * len(group['params'])
        for name, p in zip(names, group['params']):
            g = p.grad
            if g is None:
                continue
            self._muon_update_one(group, name, p, g, lr, momentum)

    def _muon_update_one(self, group, name, p, g, lr, momentum):
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(g)
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(g)
        d = g.add(buf, alpha=momentum) if group['nesterov'] else buf
        o = zeropower_via_newtonschulz5(d, steps=group['ns_steps'])
        # shape adjustment (modded-nanogpt): a [m, n] update's RMS
        # matches Adam-scale when scaled by max(1, m/n)**0.5.
        scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        p.add_(o, alpha=-lr * scale)
        # decoupled weight decay (stage 0b: Moonshot's scaling recipe;
        # default 0.0 keeps the update byte-identical to the incumbent).
        wd = group.get('weight_decay', 0.0)
        if wd:
            p.mul_(1 - lr * wd)

    def _adamw_step(self, group):
        lr = group['lr']
        b1, b2 = group['betas']
        eps = group['eps']
        for p in group['params']:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(g)
                state['exp_avg_sq'] = torch.zeros_like(g)
                state['step'] = 0
            m, v = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            t = state['step']
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            p.addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)
        # weight_decay=0 by design: keeps the math equal to the incumbent
        # recipe's Adam-equivalent AdamW(weight_decay=0.0).


EXCLUDE_SUBSTR = ('emb', 'gate', 'quat', 'theta', 'beta')


def split_muon_params(model, include=()):
    """Partition model.named_parameters() into (muon, adamw) lists of
    (name, param) pairs.

    Muon: matrix-shaped (ndim >= 2) parameters that either match the
    INCLUDE list (checked first -- the stage-0a fix routes the compose
    node's fusion matrices in by explicit name, single-variable, with
    every other name's routing unchanged) or contain none of
    EXCLUDE_SUBSTR (rot_free's per-block 3x3 maps, cross_mlp weights,
    the untied head). AdamW: embeddings, gates, quaternions, gains,
    biases, and every 1-dim parameter."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and (any(s in name for s in include)
                            or not any(s in name for s in EXCLUDE_SUBSTR)):
            muon.append((name, p))
        else:
            adamw.append((name, p))
    return muon, adamw


class LOMuon(Muon):
    """Level-orthogonalized Muon for SCALE-TIED parameters
    (docs/OPERA_Optimizer_prereg.md stage 1).

    Muon orthogonalizes the SUMMED gradient of the compose node's fusion
    matrix -- a sum over log2(T) tree levels whose applications differ
    in count and input distribution. Newton-Schulz is approximately
    scale-invariant, so it already discards the 15:1 magnitude
    imbalance; what it cannot discard is which level's geometry
    dominates the sum's directions. LOMuon changes one thing:

        NS( sum_l G_l )   ->   NS( sum_l  w_l * NS(m_l) )

    -- the sum happens in WHITENED space, so every level contributes a
    direction vote rather than a magnitude vote, and one final NS
    restores Muon's unit-RMS update discipline.

    Per-level gradients arrive through the model's level-capture
    registry (opera_lm.model._LevelCapture router); this optimizer
    reads them instead of the summed .grad for the parameters named in
    lo_names. Uniform w_l = 1 (theory-free ablation) or derived
    w_l ~ 1/x_l (per-application-context steepest descent: Bernstein's
    RMS-operator-norm bound says a level consuming larger inputs should
    get a smaller dW budget; x_l is the registry's per-level input-RMS
    EMA).

    Zero new parameters; forward untouched; params not in lo_names get
    exactly Muon's update.
    """

    def __init__(self, param_groups, lo_names=(), lo_mode='uniform', **kw):
        super().__init__(param_groups, **kw)
        assert lo_mode in ('uniform', 'derived')
        self.lo_names = set(lo_names)
        self.lo_mode = lo_mode

    def _muon_update_one(self, group, name, p, g, lr, momentum):
        if name not in self.lo_names:
            super()._muon_update_one(group, name, p, g, lr, momentum)
            return
        from .model import level_capture_dict
        cap = level_capture_dict()
        if cap is None:
            super()._muon_update_one(group, name, p, g, lr, momentum)
            return
        grads, rms = cap['grads'], cap['rms']
        state = self.state[p]
        moms = state.setdefault('lo_mom', {})
        # uniform cadence: EVERY level's momentum decays each step;
        # levels present this step (curriculum / short sequences) get
        # the new gradient added on top.
        for m in moms.values():
            m.mul_(momentum)
        for (n, l), gl in grads.items():
            if n != name:
                continue
            if l in moms:
                moms[l].add_(gl)
            else:
                moms[l] = gl.detach().clone()
        if not moms:
            return
        if self.lo_mode == 'derived':
            base = max([r for r in rms.values() if r] or [1.0])
            w = {l: (base / rms[l]) if rms.get(l) else 1.0
                 for l in moms}
        else:
            w = {l: 1.0 for l in moms}
        u = None
        for l, m in moms.items():
            ul = zeropower_via_newtonschulz5(m, steps=group['ns_steps'])
            ul = ul * w[l]
            u = ul if u is None else u + ul
        u = zeropower_via_newtonschulz5(u, steps=group['ns_steps'])
        scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        p.add_(u, alpha=-lr * scale)
        wd = group.get('weight_decay', 0.0)
        if wd:
            p.mul_(1 - lr * wd)

    def zero_grad(self, set_to_none=True):
        """Standard zero_grad, plus: the level-capture registry's
        gradient buffers are per-BACKWARD (they must not accumulate
        across steps). The per-level input-RMS EMA PERSISTS -- it is a
        running statistic, not a gradient."""
        super().zero_grad(set_to_none=set_to_none)
        from .model import level_capture_dict
        cap = level_capture_dict()
        if cap is not None:
            cap['grads'] = {}
