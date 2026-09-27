"""Shared helpers for the Phase-1 instruments
(docs/OPERA_Redundancy_Research_2026-09-24.md §4): rebuild a finished
repr_study arm from its name, and run the model's layer loop exposing the
per-layer Fenwick block states.

The layer loop mirrors OperaSpinorFenwickTree.forward exactly for the
left fold (tree-input pre-norm / projection, fold, tree_out, stream
update); instruments/selftests compare its output against forward()."""
import inspect
import json
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('OPERA_RUNS', os.path.join(ROOT, 'runs_reprs'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from opera_lm.model import OperaSpinorFenwickTree  # noqa: E402


def load_corpus(T=1024, articles=20000):
    path = os.path.join(ROOT, 'assets', f'corpus_bytes_T{T}_a{articles}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _model_kwargs(arch):
    """Keep only constructor flags (ARCH also carries train-side ones such
    as aux_weight / samuon_gamma)."""
    params = set(inspect.signature(OperaSpinorFenwickTree.__init__).parameters)
    return {k: v for k, v in (arch or {}).items() if k in params}


def load_arm(name, vocab_size=259, device='cpu'):
    """Rebuild arm `name` from runs_reprs/repr_summary.json + its
    checkpoint. Strict state_dict load: an arm whose flags this helper
    does not reproduce fails loudly instead of loading as the wrong
    model."""
    import repr_study as rs
    rs.scale_arm(name)
    summ = json.load(open(os.path.join(OUT, 'repr_summary.json')))
    if name not in summ:
        raise SystemExit(f"arm {name} has not finished (not in summary)")
    r = summ[name]
    arm_dir = os.path.join(OUT, name)
    pts = [f for f in os.listdir(arm_dir)
           if f.endswith('.pt') and not f.endswith('_train_ckpt.pt')]
    if len(pts) != 1:
        raise SystemExit(f"{arm_dir}: expected one checkpoint, got {pts}")
    sd = torch.load(os.path.join(arm_dir, pts[0]), map_location='cpu',
                    weights_only=False)
    sd = sd.get('model', sd)
    layers = 1 + max(int(k.split('.')[1]) for k in sd
                     if k.startswith('fusion_gate.') and k.endswith('.weight'))
    kw = dict(vocab_size=vocab_size, d=r['d'], nb=r['nb'], num_layers=layers,
              pe_mode='none', fold_mode='left', rot_mode='free',
              fold_grade=rs.GRADE.get(name),
              **_model_kwargs(rs.device_kernels(rs.ARCH.get(name), str(device).split(':')[0])))
    m = OperaSpinorFenwickTree(**kw)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval(), kw


@torch.no_grad()
def layer_streams(model, token_ids):
    """Yield, per layer l, a dict with the layer's tree levels, the
    gathered Fenwick block states [B, T, S, d_tree] (slot 0 = oldest /
    largest block), per-position block count [T], block levels [T, S],
    the fold readout P (after tree_out, d_model) and the head input.
    Same op sequence as forward() (left fold)."""
    model._need_locks = False
    model._need_energy = False
    model._rot_dense_cache.clear()
    B, T = token_ids.shape
    current = model.word_emb(token_ids)
    for l in range(model.num_layers):
        x = current
        if model.tree_in_norm is not None:
            x = model.tree_in_norm[l](x)
        if model.tree_in is not None:
            x = model.tree_in[l](x)
        R_L, R_R, R_O = model.get_rotations(l)
        levels, _, _ = model.build_tree(x, l, R_L, R_R, R_O)
        offs, off = [], 0
        for lv in levels:
            offs.append(off)
            off += lv.shape[1]
        idx, count, S, lvl, _ = model._fenwick_indices(
            T, len(levels), offs, token_ids.device)
        flat = torch.cat(levels, dim=1)
        gathered = flat[:, idx.reshape(-1), :].reshape(B, T, S, -1)
        prefix = model.prefix_states(levels, T, l, R_L, R_R, R_O)
        if model.tree_out is not None:
            prefix = model.tree_out[l](prefix)
        mixed = model.cross_mlp[l](prefix)
        if model.resid_mode == 'add':
            current = current + mixed
        else:
            gate = model.blend_gate[l](prefix)
            current = gate * mixed + (1 - gate) * current
        head_in = (model.stream_norm[l](current)
                   if model.head_mode == 'stream' else prefix)
        yield dict(layer=l, gathered=gathered, count=count, lvl=lvl,
                   prefix=prefix, head_in=head_in)
