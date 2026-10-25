"""Decode cost against context length: OPERA's Fenwick-incremental decoder
vs a transformer with a KV cache, at matched width and depth.

For each model, bytes are appended one at a time up to --max-ctx; at each
checkpoint length the script records
  - state: elements the decoder carries between steps (OPERA: frontier
    nodes + memories, OperaDecoder.state_numel; transformer: the K and V
    entries actually filled, 2 * layers * t * d), and its fp32 bytes;
  - time: mean wall-clock per appended byte over the --window bytes just
    before the checkpoint.
Weights are random: state size and step time do not depend on training.
Both decoders are checked against their model's full forward first.

  python experiments/bench_decode.py --d 512 --layers 2 --max-ctx 65536 --device cpu

Writes <out>.json and <out>.png (default $OPERA_RUNS/bench_decode_d<d>).
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'opera-chat'))
import repr_study as rs                                                 # noqa: E402
from arm_utils import OUT, _model_kwargs                                # noqa: E402
from opera_lm import OperaDecoder, OperaSpinorFenwickTree               # noqa: E402
from opera_transformer_baseline_v2 import (TransformerBaseline,         # noqa: E402
                                           apply_rope, rope_tables,
                                           _rms_norm_last_dim)


def build_opera(d, layers, device):
    """The scale-ladder architecture (fw_ arms) at width d, eager kernels."""
    arm = f'fw_d{d}_L{layers}_bench'
    rs.scale_arm(arm)
    kw = dict(vocab_size=259, d=d, nb=d // 4, num_layers=layers, pe_mode='none',
              fold_mode='left', rot_mode='free', fold_grade=rs.GRADE.get(arm),
              **_model_kwargs(rs.device_kernels(rs.ARCH.get(arm), 'cpu')))
    m = OperaSpinorFenwickTree(**kw)
    with torch.no_grad():                    # make the zero-init memory read live
        for g in m.hmem_gain:
            g.fill_(0.5)
    return m.to(device).eval()


class KVDecoder:
    """KV-cache decoding for TransformerBaseline: one query per step
    against the cached keys/values of every earlier position. Same ops as
    Block.forward (qk_norm, RoPE at the absolute position, soft cap)."""

    def __init__(self, model, max_ctx):
        self.m = model
        p = next(model.parameters())
        H, hd = model.nheads, model.hd
        self.K = [p.new_zeros(1, H, max_ctx, hd) for _ in model.blocks]
        self.V = [p.new_zeros(1, H, max_ctx, hd) for _ in model.blocks]
        self.rope = (rope_tables(max_ctx, hd, p.device)
                     if model.pe_mode == 'rope' else None)
        self.t = 0

    def state_numel(self):
        return 2 * len(self.m.blocks) * self.t * self.m.d

    @torch.no_grad()
    def append(self, token_id):
        m, t = self.m, self.t
        x = m.word_emb(torch.tensor([[token_id]], device=self.K[0].device))
        if m.pe_mode in ('sin', 'learned'):
            raise NotImplementedError('bench uses rope/nope transformers')
        for i, blk in enumerate(m.blocks):
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).reshape(1, 1, 3, blk.nheads, blk.hd).unbind(2)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))     # [1,H,1,hd]
            if blk.qk_norm:
                q = _rms_norm_last_dim(q, blk.q_norm_gain)
                k = _rms_norm_last_dim(k, blk.k_norm_gain)
            if self.rope is not None:
                cos, sin = self.rope
                q = apply_rope(q, cos[t:t + 1], sin[t:t + 1])
                k = apply_rope(k, cos[t:t + 1], sin[t:t + 1])
            self.K[i][:, :, t:t + 1] = k
            self.V[i][:, :, t:t + 1] = v
            Kc, Vc = self.K[i][:, :, :t + 1], self.V[i][:, :, :t + 1]
            if blk.attn_logit_cap:
                s = (q.float() @ Kc.float().transpose(-2, -1)) * blk.hd ** -0.5
                s = blk.attn_logit_cap * torch.tanh(s / blk.attn_logit_cap)
                att = torch.softmax(s, -1).to(Vc.dtype) @ Vc
            else:
                att = F.scaled_dot_product_attention(q, Kc, Vc)
            x = x + blk.proj(att.transpose(1, 2).reshape(1, 1, m.d))
            x = x + blk.ffn(blk.ln2(x))
        lg = m.head(m.ln_f(x))
        if m.logit_scale is not None:
            lg = lg * m.logit_scale
        self.t += 1
        return lg[0, 0]


def check_exact(name, dec, full_logits, toks):
    err = 0.0
    for t, tid in enumerate(toks):
        err = max(err, (dec.append(int(tid)) - full_logits[t]).abs().max().item())
    print(f'  {name}: decoder vs full forward over T={len(toks)}: max err {err:.1e}',
          flush=True)
    assert err < 1e-3, f'{name} decoder is not exact'


def run(name, make_dec, checkpoints, window, gen):
    """State is the MAXIMUM over the window before each checkpoint: OPERA's
    frontier holds popcount(t) nodes, so at a power of two it collapses to
    one node and a point sample there would understate it."""
    dec = make_dec()
    rows, t_prev, peak = [], None, 0
    marks = set(checkpoints)
    starts = {c - window for c in checkpoints}
    for t in range(1, max(checkpoints) + 1):
        if t - 1 in starts:
            t_prev = time.perf_counter()
            peak, step_s = 0, 0.0
        if t_prev is not None:
            t0 = time.perf_counter()
            dec.append(int(torch.randint(3, 259, (1,), generator=gen)))
            step_s += time.perf_counter() - t0
            peak = max(peak, dec.state_numel())
        else:
            dec.append(int(torch.randint(3, 259, (1,), generator=gen)))
        if t in marks:
            ms = step_s / window * 1e3
            n = peak
            t_prev = None
            rows.append(dict(ctx=t, state_numel=n, state_mib=n * 4 / 2**20,
                             ms_per_byte=ms))
            print(f'  {name:12s} ctx {t:7d}  state {n:>11,} el '
                  f'({n * 4 / 2**20:9.3f} MiB)  {ms:7.2f} ms/byte', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--d', type=int, default=512)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--max-ctx', type=int, default=65536)
    p.add_argument('--window', type=int, default=128,
                   help='bytes timed before each checkpoint')
    p.add_argument('--tf-pe', default='rope', choices=['rope', 'nope'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=0, help='torch threads (0 = default)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = a.out or os.path.join(OUT, f'bench_decode_d{a.d}')
    torch.manual_seed(0)

    opera = build_opera(a.d, a.layers, a.device)
    tf = TransformerBaseline(259, d=a.d, nheads=max(1, a.d // 64),
                             num_layers=a.layers, pe_mode=a.tf_pe).to(a.device).eval()
    n_op = sum(p.numel() for p in opera.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    print(f'OPERA d{a.d} L{a.layers}: {n_op / 1e6:.1f}M params; transformer '
          f'({a.tf_pe}, {a.d // 64} heads): {n_tf / 1e6:.1f}M params; device {a.device}',
          flush=True)

    # Exactness in full fp32: opera_lm.model turns on TF32 matmuls on CUDA,
    # and the decoder's single-row matmuls round differently from the
    # batched forward's (~1e-3 under TF32 -- a precision setting, not a
    # decoder error; device_check.py / triton_kernel --test do the same).
    # Timing below runs under the restored default.
    prec = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    toks = torch.randint(3, 259, (1, 300))
    with torch.no_grad():
        lo = opera(toks.to(a.device), torch.tensor([300], device=a.device)).logits[-1][0]
        lt = tf(toks.to(a.device), torch.tensor([300], device=a.device))[-1][0]
    check_exact('OPERA', OperaDecoder(opera), lo, toks[0])
    check_exact('transformer', KVDecoder(tf, 300), lt, toks[0])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prec[:2]
    torch.set_float32_matmul_precision(prec[2])

    ck = sorted({min(a.max_ctx, 2 ** k) for k in range(8, int(math.log2(a.max_ctx)) + 1)}
                | {a.max_ctx})
    ck = [c for c in ck if c > a.window]
    res = {}
    for name, make in (('OPERA', lambda: OperaDecoder(opera)),
                       ('transformer', lambda: KVDecoder(tf, a.max_ctx))):
        res[name] = run(name, make, ck, a.window, torch.Generator().manual_seed(1))

    meta = dict(d=a.d, layers=a.layers, device=a.device, tf_pe=a.tf_pe,
                params=dict(opera=n_op, transformer=n_tf), window=a.window,
                threads=torch.get_num_threads(),
                note='random weights; state = elements carried between steps; '
                     'transformer state counts filled K/V entries only')
    json.dump(dict(meta=meta, rows=res), open(out + '.json', 'w'), indent=1)
    print('wrote', out + '.json')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for name, color in (('OPERA', 'C0'), ('transformer', 'C1')):
        xs = [r['ctx'] for r in res[name]]
        label = 'OPERA (Fenwick decoder)' if name == 'OPERA' else \
            f'transformer + KV cache ({a.tf_pe})'
        axes[0].plot(xs, [r['state_mib'] for r in res[name]], 'o-', color=color, label=label)
        axes[1].plot(xs, [r['ms_per_byte'] for r in res[name]], 'o-', color=color, label=label)
    axes[0].set_ylabel('decoder state (MiB, fp32)')
    axes[1].set_ylabel('time per generated byte (ms)')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('context length (bytes)')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8.5)
    fig.suptitle(f'Decoding cost vs context, d={a.d}, {a.layers} layers, {a.device}',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out + '.png', dpi=180)
    print('wrote', out + '.png')


if __name__ == '__main__':
    main()
