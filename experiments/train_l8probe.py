"""EXPLORATORY probe model for depth_survival.py: OPERA L=8, bytes T1024,
incumbent recipe (muon 0.02, include fusion_gate, wd 0.01), 1500 steps.

Not a comparison arm -- params grow with L. The measurement target is the
PER-LAYER trend within this one net (P_0..P_7 recency/rank/identity/CE),
so the reduced step count vs the 3000-step L=2 incumbent only affects
absolute levels, which are compared against this model's own layer 0.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opera_lm.train import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, 'assets', 'corpus_bytes_T1024_a20000.pkl'),
          'rb') as f:
    data, meta = pickle.load(f)

train(steps=1500, batch=8, max_len=1024,
      vocab_size=meta['vocab_size'], d=512, nb=128, num_layers=8,
      eval_max_len=2048, device='mps',
      pe_mode='none', fold_mode='left', rot_mode='free',
      data=data, seed=42, out_dir=os.path.join(ROOT, 'runs_reprs',
                                               'l8_probe'),
      tie=False, optimizer='muon', muon_lr=0.02,
      muon_include='fusion_gate', muon_wd=0.01,
      grad_checkpoint='layer', save_every=500)
