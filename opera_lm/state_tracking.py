"""State-tracking benchmark: group word problems, with length
generalization.

WHY THIS IS OPERA'S HOME AXIS
-----------------------------
The word problem for a group G: read a sequence of elements
g_1, g_2, ..., g_T and emit the running product at every prefix. Its
difficulty is set by the algebra of G, not by its size:

  * G solvable (C_2 = parity, A_4, S_4)  -- the prefix products can be
    computed by a constant-depth threshold circuit, i.e. in TC^0.
  * G NON-solvable (A_5, S_5)            -- the word problem is
    NC^1-complete (Barrington, 1989). It is NOT in TC^0 unless the
    classes collapse.

That distinction is what makes it the standard probe for sequence
models (Merrill & Sabharwal, arXiv:2404.08819; Grazzi et al.,
arXiv:2411.12537; Siems et al., DeltaProduct, arXiv:2502.10297):

  * A fixed-depth transformer is a constant-depth threshold circuit,
    so it is in TC^0 and cannot solve A_5 for growing T.
  * A parallel-scan SSM with DIAGONAL, POSITIVE state transitions is
    abelian in flight and collapses to TC^0 as well -- this is exactly
    why Mamba fails parity. The whole negative-eigenvalue /
    Householder-product line is the effort to escape that box.

OPERA is not in that box, for two independent reasons:

  1. **Its compute graph is log-depth in T.** The tree over a length-T
     prefix has ceil(log2 T) levels of composition, so the circuit
     depth GROWS with the sequence. Log-depth is the natural home of
     NC^1 -- the class that contains A_5. Transformers and scan-SSMs
     are constant-depth by construction; OPERA is not.
  2. **Its state space already contains the answer.** A_5 is the
     rotation group of the icosahedron: a finite subgroup of SO(3),
     order 60. An OPERA block state IS a rotor in SO(3), and the
     compose node's geometric product IS quaternion multiplication --
     the group operation itself. So there is an exact construction:
     embed the 60 icosahedral rotations as the token rotors and the
     node composes them correctly by definition.

Consequence: unlike every other claim in this project, "OPERA can
represent the solution" is a theorem here, not a hope. The open
question is purely whether SGD FINDS the construction. That is a clean
empirical question with an unambiguous answer, and it is the reason
`rot_mode='so3'` -- pre-registered and nulled twice on language
modeling -- is the arm to run FIRST on this task rather than the one
that was retired.

PROTOCOL
--------
Train at a short length, evaluate at increasing lengths, report
per-length accuracy. Length generalization is the whole point: any
model can memorize a fixed-length lookup, so in-length accuracy alone
proves nothing. The headline quantity is accuracy at 8-16x the training
length on a non-solvable group.

Chance is 1/|G|. A model that has learned the group is at ~1.0; a model
that has memorized short patterns falls off a cliff as T grows.

    python -m opera_lm.state_tracking --group A5
    python -m opera_lm.state_tracking --group parity --rot-mode free
    python -m opera_lm.state_tracking --group A5 --arch transformer
"""
import argparse
import itertools
import json
import time

import torch
import torch.nn.functional as F

from .model import OperaSpinorFenwickTree, count_params

# Solvability drives the complexity class, so it is recorded per group
# rather than left implicit: only the non-solvable ones are outside TC^0.
GROUPS = {
    'parity': dict(n=2, even_only=False, solvable=True),   # C_2
    'A4':     dict(n=4, even_only=True,  solvable=True),   # order 12
    'S4':     dict(n=4, even_only=False, solvable=True),   # order 24
    'A5':     dict(n=5, even_only=True,  solvable=False),  # order 60, NC^1
    'S5':     dict(n=5, even_only=False, solvable=False),  # order 120, NC^1
}


def _parity_of(perm):
    """Parity of a permutation, by counting inversions."""
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return inv % 2


def build_group(name):
    """Elements as permutation tuples plus the Cayley table.

    Composition convention is 'a THEN b', i.e. compose(a, b)[i] =
    b[a[i]], so a left-to-right scan of the sequence accumulates the
    running product in reading order -- matching the order in which
    OPERA's fold consumes Fenwick blocks (largest/leftmost span first).
    """
    spec = GROUPS[name]
    n = spec['n']
    elems = [p for p in itertools.permutations(range(n))
             if not spec['even_only'] or _parity_of(p) == 0]
    elems.sort()
    index = {e: i for i, e in enumerate(elems)}
    size = len(elems)
    table = torch.zeros(size, size, dtype=torch.long)
    for i, a in enumerate(elems):
        for j, b in enumerate(elems):
            table[i, j] = index[tuple(b[a[k]] for k in range(n))]
    identity = index[tuple(range(n))]
    return elems, table, identity, spec['solvable']


def make_data(table, identity, n_seq, T, seed):
    """Random element sequences and their running products.

    inputs[b, t]  -- index of g_{t+1}
    targets[b, t] -- index of g_1 * g_2 * ... * g_{t+1}

    targets[:, t] is the product over the prefix 0..t, which is exactly
    the prefix OPERA's logits[:, t] is computed from (losses.py:63 --
    logits[:, t] predicts token t+1, i.e. it reads tokens 0..t). No
    shift is applied anywhere; the alignment is direct.
    """
    g = torch.Generator().manual_seed(seed)
    size = table.shape[0]
    inputs = torch.randint(0, size, (n_seq, T), generator=g)
    targets = torch.empty_like(inputs)
    acc = torch.full((n_seq,), identity, dtype=torch.long)
    for t in range(T):
        acc = table[acc, inputs[:, t]]
        targets[:, t] = acc
    return inputs, targets


def build_model(arch, size, d, nb, layers, rot_mode, device,
                norm_mode='layer', act_mode='tanh'):
    """The node's geometry flags matter more here than anywhere else in
    the project, so they are first-class arguments rather than defaults.

    The exact A_5 construction needs the node to BE quaternion
    multiplication: gates selecting the geometric-product path, rotors
    at identity. Two defaults break that representation --
    `norm_mode='layer'` subtracts a mean across all d dimensions
    (destroying per-block rotor structure) and `act_mode='tanh'`
    squashes the components off the unit sphere. The geometry-clean
    pair is `norm_mode='blockrms'` (per-block RMS: exactly SO(3)-
    equivariant, and a NO-OP on unit quaternions) with
    `act_mode='linear'`. Both were measured as losses on language
    modeling -- blockrms at +4 PPL -- which is why they are not the
    defaults, and both are on their home axis here.
    """
    if arch == 'opera':
        m = OperaSpinorFenwickTree(
            vocab_size=size, d=d, nb=nb, num_layers=layers,
            pe_mode='none',            # position is structure: never injected
            fold_mode='left', rot_mode=rot_mode, tie=False,
            norm_mode=norm_mode, act_mode=act_mode)
    elif arch == 'transformer':
        import math
        import torch.nn as nn

        class TinyTF(nn.Module):
            """Matched-budget causal transformer with learned absolute
            PE. Deliberately GIVEN a positional encoding -- the point of
            this benchmark is the complexity class, not the position
            mechanism, and withholding PE would weaken the baseline for
            the wrong reason."""

            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(size, d)
                self.pos = nn.Embedding(4096, d)
                layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=max(1, d // 64), dim_feedforward=4 * d,
                    dropout=0.0, batch_first=True, norm_first=True,
                    activation='gelu')
                self.enc = nn.TransformerEncoder(layer, layers)
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, size)

            def forward(self, ids, lengths):
                T = ids.shape[1]
                x = self.emb(ids) + self.pos(
                    torch.arange(T, device=ids.device))[None]
                mask = torch.triu(torch.ones(T, T, device=ids.device,
                                             dtype=torch.bool), 1)
                h = self.enc(x, mask=mask, is_causal=True)
                return [self.head(self.norm(h))]

        m = TinyTF()
    else:
        raise ValueError(arch)
    return m.to(device)


def _logits(model, ids, lengths):
    out = model(ids, lengths)
    return out.logits if hasattr(out, 'logits') else out


@torch.no_grad()
def evaluate(model, table, identity, T, n_seq, batch, device, seed):
    """Per-position accuracy at length T (all positions, not just the
    last -- a model that only gets late positions right is not tracking
    state)."""
    model.eval()
    inputs, targets = make_data(table, identity, n_seq, T, seed)
    correct = total = 0
    for s in range(0, n_seq, batch):
        ids = inputs[s:s + batch].to(device)
        tgt = targets[s:s + batch].to(device)
        lengths = torch.full((ids.shape[0],), T, dtype=torch.long,
                             device=device)
        pred = _logits(model, ids, lengths)[-1].argmax(-1)
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    model.train()
    return correct / total


def run(args):
    device = torch.device(args.device)
    elems, table, identity, solvable = build_group(args.group)
    size = len(elems)
    print('=' * 70)
    print(f"STATE TRACKING — group {args.group} (order {size}), "
          f"{'SOLVABLE (in TC^0)' if solvable else 'NON-SOLVABLE (NC^1)'}")
    print(f"  arch={args.arch} rot_mode={args.rot_mode} "
          f"norm={args.norm_mode} act={args.act_mode} d={args.d} "
          f"nb={args.nb} layers={args.layers}")
    print(f"  train T={args.train_len}  eval T="
          f"{','.join(str(t) for t in args.eval_lens)}  "
          f"steps={args.steps} device={device}")
    print(f"  chance = {1.0 / size:.4f}")
    print('=' * 70)

    model = build_model(args.arch, size, args.d, args.nb, args.layers,
                        args.rot_mode, device, args.norm_mode,
                        args.act_mode)
    print(f"  params: {count_params(model):,}")

    inputs, targets = make_data(table, identity, args.n_train,
                                args.train_len, args.seed)
    inputs, targets = inputs.to(device), targets.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, args.n_train, (args.batch,), generator=g)
        ids, tgt = inputs[idx], targets[idx]
        lengths = torch.full((args.batch,), args.train_len,
                             dtype=torch.long, device=device)
        all_lg = _logits(model, ids, lengths)
        # Final layer carries the reported metric; auxiliary layers get
        # the same 0.5 weight lm_loss uses, so the objective matches the
        # project's convention rather than inventing a new one.
        loss = 0.0
        L = len(all_lg)
        for li, lg in enumerate(all_lg):
            w = 1.0 if li == L - 1 else 0.5
            loss = loss + w * F.cross_entropy(
                lg.reshape(-1, size), tgt.reshape(-1))
        loss = loss / (1.0 + 0.5 * (L - 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            acc = evaluate(model, table, identity, args.train_len,
                           args.n_eval, args.batch, device, args.seed + 99)
            print(f"  step {step:>6}  loss {loss.item():.4f}  "
                  f"in-length acc {acc:.4f}  ({time.time() - t0:.0f}s)")

    print('\n  LENGTH GENERALIZATION')
    print('  eval T    x train    accuracy   vs chance')
    results = {}
    for T in args.eval_lens:
        acc = evaluate(model, table, identity, T, args.n_eval,
                       max(1, args.batch // max(1, T // args.train_len)),
                       device, args.seed + 199)
        results[T] = acc
        print(f"  {T:>6}    {T / args.train_len:>5.1f}x    {acc:>8.4f}   "
              f"{acc * size:>6.1f}x")
    print('=' * 70)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'group': args.group, 'order': size,
                       'solvable': solvable, 'arch': args.arch,
                       'rot_mode': args.rot_mode, 'params':
                       count_params(model), 'config': vars(args),
                       'accuracy_by_length': results}, f, indent=2,
                      default=str)
        print(f"wrote {args.json}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', default='A5', choices=list(GROUPS))
    p.add_argument('--arch', default='opera',
                   choices=['opera', 'transformer'])
    p.add_argument('--rot-mode', default='so3', choices=['so3', 'free'])
    p.add_argument('--norm-mode', default='layer',
                   choices=['layer', 'rms', 'blockrms'])
    p.add_argument('--act-mode', default='tanh', choices=['tanh', 'linear'])
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--nb', type=int, default=32)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--train-len', type=int, default=32)
    p.add_argument('--eval-lens', type=int, nargs='+',
                   default=[32, 64, 128, 256, 512])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--n-train', type=int, default=20000)
    p.add_argument('--n-eval', type=int, default=512)
    p.add_argument('--log-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--json', default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    run(args)


if __name__ == '__main__':
    main()
