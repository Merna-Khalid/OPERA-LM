"""Trajectory-dynamics arm runs: incumbent vs the three Phase-1 arms at a
matched small budget.

Rung: the 5.9M word-level config (d=256, nb=64, L=2, pe-none, fold left,
rot free, norm layer, act tanh -- the incumbent winners), steps=3000,
batch=16, max_len=256, seed 42, same docs word-level data as the 22M
headline run (assets/vocab_docs.pkl). Iteration budget, NOT the headline
run: enough to rank arms and measure trajectory effects.

Runs sequentially (MPS contention), saves checkpoints to runs_arms/,
then reports each arm's learned parameter behavior (the plan's "log the
gate distribution" rule, satisfied post-hoc from the checkpoint):
  * homeo:     mean sigmoid(homeo_gate) per level, anchor norms
  * quotient:  mean sigmoid(quotient_gate.bias), |W| per layer
  * fold_adapt: |fold_adapt_w| per layer (distance from the zero init)

    /opt/anaconda3/envs/CWUW/bin/python3 experiments/trajectory_arms.py
"""
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.train import train

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs_arms')
VOCAB = os.path.join(os.path.dirname(__file__), '..', 'assets',
                     'vocab_docs.pkl')

ARMS = {
    'incumbent': {},
    'homeo': {'homeo_mode': 'on'},
    'quotient': {'node_paths': 4},          # eager-only arm
    'fold_adapt': {'fold_adapt': 'on'},
}

COMMON = dict(steps=3000, batch=16, max_len=256, eval_max_len=256,
              vocab_size=10000, d=256, nb=64, num_layers=2,
              lock_mode='none', pe_mode='none', fold_mode='left',
              rot_mode='free', norm_mode='layer', act_mode='tanh',
              device='mps', compile_mode='off', use_amp=True,
              gpu_data=True, warmup_steps=200, seed=42,
              save_every=1500, aux_frac=0.25)


def arm_param_stats(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    stats = {}
    if 'homeo_gate' in sd:
        g = torch.sigmoid(sd['homeo_gate'].float())      # [L, lvl, nb]
        stats['homeo_gate_mean_by_level'] = g.mean(dim=(0, 2)).tolist()
        stats['homeo_gate_mean'] = g.mean().item()
        stats['homeo_anchor_norm'] = sd['homeo_anchor'].float().norm(
            dim=-1).mean().item()
    if any(k.startswith('quotient_gate') for k in sd):
        b = torch.cat([sd[f'quotient_gate.{i}.bias'].float()
                       for i in range(2)])
        w = torch.cat([sd[f'quotient_gate.{i}.weight'].float().flatten()
                       for i in range(2)])
        stats['quotient_g3_sigmoid_bias_mean'] = torch.sigmoid(b).mean().item()
        stats['quotient_g3_weight_norm'] = w.norm().item()
    if 'fold_adapt_w' in sd:
        w = sd['fold_adapt_w'].float()
        stats['fold_adapt_w_norm_by_layer'] = w.norm(dim=(1, 2)).tolist()
    return stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tr, ts, tl, vocab, w2i, i2w = pickle.load(open(VOCAB, 'rb'))
    data = (tr, ts, tl, len(vocab))
    summary = {}
    results_log = os.path.join(OUT_DIR, 'opera_v8_0_results.jsonl')
    for name, flags in ARMS.items():
        print('=' * 74)
        print(f'ARM: {name}  {flags}')
        print('=' * 74, flush=True)
        # node_paths=4 asserts eager-only; the others use the metal kernel
        use_metal = flags.get('node_paths', 3) == 3
        expected = os.path.join(
            OUT_DIR, 'opera_v8_0_'
            + '+'.join((['pe-none', 'rotfree']
                        + (['metal'] if use_metal else [])
                        + ['amp', 'foreach', 'gpudata', 'aux0.25', 'wu200']
                        + ({'homeo': ['homeo'], 'quotient': ['np4'],
                            'fold_adapt': ['fadpt']}.get(name, [])))
                     ).replace('+', '_') + '.pt')
        if os.path.exists(expected):
            # already trained (rerun-safe): recover its eval from the log
            print(f'  checkpoint exists, skipping training: {expected}')
            ckpt = expected
            res = None
            with open(results_log) as f:
                for line in f:
                    r = json.loads(line)
                    if expected.endswith(r['config'].replace('+', '_')
                                         + '.pt'):
                        res = r
            assert res is not None, f"no results.jsonl entry for {name}"
        else:
            res = train(**COMMON, **flags, use_metal=use_metal,
                        data=data, idx2word=i2w, out_dir=OUT_DIR)
            assert os.path.exists(expected), \
                f"checkpoint not at expected path: {expected}"
            ckpt = expected
        summary[name] = dict(config=res['config'],
                             final_loss=res['final_loss'],
                             ppl_in_length=res['test_perplexity_in_length'],
                             params=res['params'], ckpt=ckpt,
                             arm_stats=arm_param_stats(ckpt))
        print(f"  -> {name}: final loss {res['final_loss']:.4f}, "
              f"PPL {res['test_perplexity_in_length']:.2f}, "
              f"ckpt {os.path.basename(ckpt)}", flush=True)
        with open(os.path.join(OUT_DIR, 'arms_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=float)
    print('=' * 74)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'ckpt'}
                      for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
