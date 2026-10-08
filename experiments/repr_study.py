"""Representation study: byte-level vs BPE OPERA, matched on TEXT.

THE MATCHING, AND THE ONE CONFOUND THAT SURVIVES IT
---------------------------------------------------
Both arms are built by `experiments/build_reprs.py` from the SAME
articles under the SAME seeded article-level split, and verified to
cover an identical 43,907,160 raw training bytes. With

    bytes  T=1024, batch 8  -> 8192 raw bytes of context per step
    bpe    T=290,  batch 8  -> 8195 raw bytes of context per step

step counts, tokens-of-text seen, and context windows all match. The
headline metric is **bits per byte**, the only quantity comparable
across tokenizations:

    bpb = (mean nats / unit) * (units / byte) / ln 2

with `units_per_byte` measured on the corpus (bytes 1.0000, bpe 0.2855).

PARAMETERS: TOTAL DIVERGES, COMPUTE DOES NOT
--------------------------------------------
At d=256 the two arms differ 10x in TOTAL parameters (0.89M vs 9.17M)
purely because embedding and head are `vocab x d` and vocab is 259 vs
16,384. Measured:

    arm          total       emb+head            non-embedding
    bytes_d256   893,191     132,608  (14.8%)    760,583
    bpe_d256   9,165,316   8,388,608  (91.5%)    776,708

**91.5% of the BPE model is a lookup table.** Its non-embedding
parameters -- every rotor, gate, norm and MLP that actually composes --
are within 2% of the byte model's. So `bytes_d256` vs `bpe_d256` is
already the compute-matched comparison: same architecture, same width,
same data, same steps, same bytes/step; only the input representation
and the size of the lookup table differ. That is the primary contrast
and it needs no correction.

Report total parameters alongside anyway, since a reader will ask, and
because it is the actual argument for byte-level at small scale: the
byte arm spends its budget on composition instead of on a table. The
width ladder (`bytes_d384`, `bytes_d512`) then shows what the byte side
buys by spending the saved parameters on compute -- 1.9M and 3.3M
non-embedding, i.e. 2.2x and 3.9x the BPE arm's compute, still at a
third of its total size.

    python experiments/repr_study.py --steps 3000
    python experiments/repr_study.py --arms bpe --steps 3000
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm.reprs import nats_to_bpb                      # noqa: E402
from opera_lm.train import train                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
# OPERA_RUNS relocates the run directory (Colab: a Drive folder, so
# checkpoints and the summary survive a disconnected runtime).
OUT = os.environ.get('OPERA_RUNS', os.path.join(ROOT, 'runs_reprs'))


def device_kernels(kw, device):
    """The arms name the Metal kernels (use_metal); on CUDA the same model
    runs on the Triton compose kernel instead (identical math: selftest /
    triton_kernel --test), and anywhere else on the eager path."""
    kw = dict(kw or {})
    if kw.pop('use_metal', False):
        if device == 'mps':
            kw['use_metal'] = True
        elif device == 'cuda':
            kw['use_triton'] = True
    return kw

# (name, repr, max_len, d, nb) -- corpora produced by build_reprs.py
ARMS = {
    'bytes_d256': ('bytes', 1024, 256, 64),
    'bytes_d384': ('bytes', 1024, 384, 96),
    'bytes_d512': ('bytes', 1024, 512, 128),
    'bpe_d256':   ('bpe',    290, 256, 64),
}

# Bistable-fold arms (docs/OPERA_Bistable_prereg.md). Same rung as
# bytes_d512 so its BPB (2.1317) is the incumbent number. 'mono' is the
# capacity-matched control: identical parameters, gain capped below 1 so
# bistability is unreachable. The contrast bi - mono isolates
# bistability rather than added capacity.
BIST = {'bist_off': 'off', 'bist_mono': 'mono', 'bist_bi': 'bi',
        'bist_bi_r1': 'bi', 'bist_bi_r4': 'bi'}
# rank of the gain bottleneck: 0 = per-block (falsified 2026-09-08),
# r>0 = the nb gains forced through r dimensions so they move together.
BIST_RANK = {'bist_bi_r1': 1, 'bist_bi_r4': 4}

# Over-relaxation arms (docs/OPERA_Relax_prereg.md). 'under' is the
# capacity-matched control: identical params, identical bitwise-incumbent
# init, gamma clamped at 1 so overshoot is unreachable.
RELAX = {'relax_off': 'off', 'relax_under': 'under', 'relax_over': 'over'}
# Level-balanced gradient (docs/OPERA_LevelGrad_prereg.md). beta=1.0 is
# bitwise the incumbent in BOTH forward and backward; beta>1 shifts the
# shared compose weights' gradient from shallow levels toward deep ones,
# normalised to mean 1 so it rebalances rather than rescales.
LGB = {'lgb_1_25': 1.25, 'lgb_1_5': 1.5, 'lgb_2_0': 2.0}
for _n in LGB:
    ARMS[_n] = ('bytes', 1024, 512, 128)

# RECIPE arms. Every byte run so far used train()'s defaults -- AdamW,
# cosine, no msup, no curriculum -- i.e. NONE of the improvements this
# project has already validated:
#   Muon lr 0.02 : 186.45 vs AdamW 224.88 (+38.4 PPL, pre-registered)
#   msup         : -9.87 PPL at the toy rung
#   curriculum   : parity on fewer tokens, stacks with msup
# RECIPE = the stacked recipe; the singles isolate each contribution.
RECIPE = {
    'rx_muon':   dict(optimizer='muon', muon_lr=0.02),
    'rx_msup':   dict(msup=True, msup_weight=0.1),
    'rx_full':   dict(optimizer='muon', muon_lr=0.02, msup=True,
                      msup_weight=0.1, curriculum=(128, 250),
                      lr_schedule='wsd'),
    # Stage 0 (docs/OPERA_Optimizer_prereg.md): pin the honest incumbent
    # before the LO-Muon comparison. 0a = fusion_gate into the Muon
    # partition (include-list, single variable, nothing else moves);
    # 0b = Muon-side decoupled weight decay (Moonshot recipe); then both.
    'rx_fgate':    dict(optimizer='muon', muon_lr=0.02,
                        muon_include='fusion_gate'),
    'rx_wd':       dict(optimizer='muon', muon_lr=0.02, muon_wd=0.01),
    'rx_fgate_wd': dict(optimizer='muon', muon_lr=0.02,
                        muon_include='fusion_gate', muon_wd=0.01),
    # Stage 1: LO-Muon (level-orthogonalized updates for the scale-tied
    # fusion weights; uniform = theory-free ablation, derived = the
    # Bernstein/Gluon-motivated w_l ~ 1/x_l weighting). Recipe = fgate+wd,
    # the expected stage-0 incumbent; amended if stage 0 crowns another.
    # Run ONLY after experiments/level_cosine.py (the kill-switch) says
    # the levels disagree.
    'lo_uniform': dict(optimizer='muon', muon_lr=0.02,
                       muon_include='fusion_gate', muon_wd=0.01,
                       lo_muon='uniform'),
    'lo_derived': dict(optimizer='muon', muon_lr=0.02,
                       muon_include='fusion_gate', muon_wd=0.01,
                       lo_muon='derived'),
}
for _n in RECIPE:
    ARMS[_n] = ('bytes', 1024, 512, 128)
# Over-relaxation RE-VALIDATION under the incumbent Muon recipe
# (OPERA_Relax_prereg.md ADDENDUM 2, 2026-09-09): the 2026-09-08 arms
# ran on the AdamW baseline; RM-H2 is the adopt-into-Path-A gate.
RELAX_M = {'relaxM_off': 'off', 'relaxM_under': 'under',
           'relaxM_over': 'over'}
RECIPE['relaxM_off'] = dict(optimizer='muon', muon_lr=0.02,
                            muon_include='fusion_gate', muon_wd=0.01)
RECIPE['relaxM_under'] = dict(optimizer='muon', muon_lr=0.02,
                              muon_include='fusion_gate', muon_wd=0.01)
RECIPE['relaxM_over'] = dict(optimizer='muon', muon_lr=0.02,
                             muon_include='fusion_gate', muon_wd=0.01)
# LEVEL-CONDITIONED compose weights (OPERA_LevelCond_prereg.md):
# lc_off re-runs the incumbent recipe in-family; lc_r8 is the arm.
RECIPE['lc_off'] = dict(optimizer='muon', muon_lr=0.02,
                        muon_include='fusion_gate', muon_wd=0.01,
                        level_cond_rank=0)
RECIPE['lc_r8'] = dict(optimizer='muon', muon_lr=0.02,
                       muon_include='fusion_gate', muon_wd=0.01,
                       level_cond_rank=8)
for _n in list(RELAX_M) + ['lc_off', 'lc_r8']:
    ARMS[_n] = ('bytes', 1024, 512, 128)
# SCALE-GRADED READOUT (docs/OPERA_ScaleGraded_prereg.md, REGISTERED
# 2026-09-13 before implementation): level-routed slot partition in the
# left fold. Slots [0,R) keep the classical fold; slots [R,nb) divide
# into G groups and a level-l block writes only group min(l, G-1).
# grade_off = fresh same-session incumbent control (recipe only, no
# grading) -- SG-H1's baseline. Zero new params in every arm.
GRADE = {'grade_r24': (24, 8), 'grade_full': (0, 8)}
for _n in ('grade_off', 'grade_r24', 'grade_full'):
    RECIPE[_n] = dict(optimizer='muon', muon_lr=0.02,
                      muon_include='fusion_gate', muon_wd=0.01)
    ARMS[_n] = ('bytes', 1024, 512, 128)
# STREAM / TREE-WIDTH ARMS (docs/OPERA_Stream_prereg.md -- set aside by
# Merna 2026-09-24; these run as EXPLORATORY arms, no gates). Stage 1
# (stream-side, all at d512 unless noted): st_off = fresh same-session
# incumbent; st_head = head reads LN(stream) (the fold head leaves the last
# layer's cross_mlp/blend_gate dead: 2.18M live of 3.29M); st_add = + additive
# pre-norm residual; st_auxw0 = aux-layer loss weight 0.5 -> 0; st_wide =
# fold-head incumbent at d640, the LIVE-param-matched control for st_head
# (3.32M vs 3.30M live). Stage 2 (fold/tree-side) runs on STAGE2_BASE,
# which stays the incumbent until a prereg amendment names the Stage-1
# winner; sx2_wide is the live-param-matched width control for sx2
# (d920 on the fold-head base, d800 on a stream-head base).
ARCH = {
    'st_off': {},
    'st_head': dict(head_mode='stream'),
    'st_add': dict(head_mode='stream', resid_mode='add'),
    'st_auxw0': dict(aux_weight=0.0),
    'st_wide': {},
    # Stage 1b: the residual question at depth (run with --layers 8
    # --steps 1500, the l8_probe budget; distinct names because the
    # summary is keyed by arm name).
    'st8_head': dict(head_mode='stream'),
    'st8_add': dict(head_mode='stream', resid_mode='add'),
    # Literature-scan arms (docs/OPERA_Literature_Scan_2026-09-24.md §5),
    # each single-variable on the incumbent: SAMuon-lite at the low end of
    # the paper's gamma grid (our batch is ~100x smaller than theirs);
    # identity-init rank-64 cross-slot mixing in the node; tied-gate 2x
    # state expansion.
    'x_sam': dict(samuon_gamma=3.54),
    'x_nmix': dict(node_mix_rank=64),
    'x_tie2': dict(state_mult=2, state_tie=True),
    # Redundancy-plan Phase 2 (docs/OPERA_Redundancy_Research_2026-09-24.md
    # §4/§5.3): innovation fold (2a) and causal disentangler (2b), rank 64,
    # each single-variable on the incumbent.
    'x_innov': dict(fold_innov_rank=64),
    'x_disent': dict(tree_disent_rank=64),
    # BASE 2 (2026-09-24): the stream head won (st_head -1.79% BPB,
    # -2.3% extrapolation vs st_off) and the downsweep fold computes the
    # same fold 2.6-2.7x faster per training step on MPS. Architecture
    # arms now run single-variable on this base.
    'b2_base': dict(head_mode='stream', fold_impl='downsweep'),
    'b2_tie2': dict(head_mode='stream', fold_impl='downsweep',
                    state_mult=2, state_tie=True),
    'b2_innov': dict(head_mode='stream', fold_impl='downsweep',
                     fold_innov_rank=64),
    'b2_disent': dict(head_mode='stream', fold_impl='downsweep',
                      tree_disent_rank=64),
    'b2_nmix': dict(head_mode='stream', fold_impl='downsweep',
                    node_mix_rank=64),
    # Seed replicates of b2_base (run with --seed 43 / --seed 44): the
    # measured noise band for base-2 comparisons. b2_base (seed 42) vs
    # st_head (same model, compact fold) differed by 0.7% BPB although
    # the two folds agree to bf16 rounding on the same weights/batch --
    # i.e. run-to-run trajectory variance, larger than the 0.25% seen
    # across incumbent runs.
    # Reruns of the three low-rank arms with lowrank_gain (direction on
    # Muon, magnitude on an AdamW scalar, zero-init): the plain versions
    # grew corrections of 1.3x / 6.6x / 30x their input under Muon.
    'b2g_innov': dict(head_mode='stream', fold_impl='downsweep',
                      fold_innov_rank=64, lowrank_gain=True),
    'b2g_disent': dict(head_mode='stream', fold_impl='downsweep',
                       tree_disent_rank=64, lowrank_gain=True),
    'b2g_nmix': dict(head_mode='stream', fold_impl='downsweep',
                     node_mix_rank=64, lowrank_gain=True),
    # Counterclockwise fold (fold_dir='both'): right-nested fold of the
    # prefix's blocks joined to the left fold by a zero-init per-slot gain
    # (path splitting, arXiv 1602.04478).
    'b2_ccw': dict(head_mode='stream', fold_impl='downsweep',
                   fold_dir='both'),
    # The ~100-byte plateau (docs/OPERA_Speed_and_Design_2026-09-24.md
    # §4.5): future-content objective (training-only head predicting the
    # hashed trigrams of the next 128 bytes, weight 0.1), and the base
    # trained 4x longer (run with --steps 12000) to test undertraining.
    'b2_fut': dict(head_mode='stream', fold_impl='downsweep',
                   future_bag=(128, 1024), future_weight=0.1),
    'b2_long': dict(head_mode='stream', fold_impl='downsweep'),
    # GRADIENT-SIDE arms against the ~100-byte plateau (gradient
    # starvation of the rare far-dependent positions): far-repeat loss
    # weighting (targets completing an 8-gram seen only >100 bytes back,
    # 4.7% of positions, weight 1+6), byte dropout 15% (posterior-collapse
    # fix), SAMuon-lite gamma 3.54 (bulk amplification).
    'b2_far': dict(head_mode='stream', fold_impl='downsweep',
                   far_weight=6.0, far_n=8, far_d=100),
    'b2_bdrop': dict(head_mode='stream', fold_impl='downsweep',
                     byte_dropout=0.15),
    'b2_sam': dict(head_mode='stream', fold_impl='downsweep',
                   samuon_gamma=3.54),
    # Speed: the fused Metal compose kernel (use_metal) -- same model as
    # eager (selftest test_metal_equivalence), 2.1x faster per training
    # step on top of the downsweep. Validation run: must land inside the
    # base-2 band (2.0058..2.0366) before becoming the default.
    'b2m_base': dict(head_mode='stream', fold_impl='downsweep',
                     use_metal=True),
    # second seed of the counterclockwise fold (b2_ccw: -1.3%, one seed),
    # on the (validated) Metal kernel
    'b2m_ccw_s43': dict(head_mode='stream', fold_impl='downsweep',
                        fold_dir='both', use_metal=True),
    # DEPTH (run with --layers 8): does depth move the ~128-byte MI onset?
    # d8_add: additive pre-norm residual (the blend highway closes at L=8:
    # prod(1-g) = 0.02); d8_blend: the current blend residual; d8_add_d256:
    # parameter-matched to the L=2/d512 base (~3.0M vs 3.3M).
    'd8_add': dict(head_mode='stream', resid_mode='add',
                   fold_impl='downsweep', use_metal=True),
    'd8_blend': dict(head_mode='stream', fold_impl='downsweep',
                     use_metal=True),
    'd8_add_d256': dict(head_mode='stream', resid_mode='add',
                        fold_impl='downsweep', use_metal=True),
    # Depth, retried with GPT-2 residual scaling (resid_init_scale='auto'
    # = 1/sqrt(2L)): additive residual + scaled init is near-isometric at
    # L=8 (1.05x/layer vs 1.38x add/default, 4.84x blend/default).
    'd8s_add': dict(head_mode='stream', resid_mode='add',
                    fold_impl='downsweep', use_metal=True,
                    resid_init_scale='auto'),
    'd8s_add_d256': dict(head_mode='stream', resid_mode='add',
                         fold_impl='downsweep', use_metal=True,
                         resid_init_scale='auto'),
    # RECALL (docs/OPERA_Recall_Research_2026-09-24.md §5b): quaternion
    # holographic memory, 128 slots per layer (512 dims = model width), on
    # the Metal base. Judged by the MI curve past ~128 bytes, the ICL
    # score (loss@500 - loss@50) and BPB vs the 5-run base band.
    'b2m_hmem128': dict(head_mode='stream', fold_impl='downsweep',
                        use_metal=True, hmem_nb=128),
    'b2m_hmem128_s43': dict(head_mode='stream', fold_impl='downsweep',
                            use_metal=True, hmem_nb=128),
    # the droop fix (Recall research §5d): per-slot multi-timescale decay,
    # fixed (content-independent) vs gated (data-dependent forget gate,
    # identical to fixed at init)
    'b2m_hmem128_fixed': dict(head_mode='stream', fold_impl='downsweep',
                              use_metal=True, hmem_nb=128,
                              hmem_decay='fixed'),
    'b2m_hmem128_gated': dict(head_mode='stream', fold_impl='downsweep',
                              use_metal=True, hmem_nb=128,
                              hmem_decay='gated'),
    'b2m_hmem128_gated_s43': dict(head_mode='stream', fold_impl='downsweep',
                                  use_metal=True, hmem_nb=128,
                                  hmem_decay='gated'),
    # short causal convolution (width 4, residual, zero-init taps) over the
    # memory's raw projection (Recall research §7: Canon layers / the
    # short conv of H3, Mamba, Based) -- keys describe 4 positions, not 1
    'b2m_hmem128_gated_conv4': dict(head_mode='stream', fold_impl='downsweep',
                                    use_metal=True, hmem_nb=128,
                                    hmem_decay='gated', hmem_conv=4),
    'b2m_hmem128_gated_conv4_s43': dict(head_mode='stream', fold_impl='downsweep',
                                        use_metal=True, hmem_nb=128,
                                        hmem_decay='gated', hmem_conv=4),
    # the gated memory trained 4x longer (run with --steps 12000), paired
    # with b2_long (the base at 12000 steps, seed 42)
    'b2m_hmem128_gated_long': dict(head_mode='stream', fold_impl='downsweep',
                                   use_metal=True, hmem_nb=128,
                                   hmem_decay='gated'),
    # SCALE-UP (Recall research §8): trained on the full Simple Wikipedia
    # pool (run with --packed assets/packed_bytes_T1024_all --steps 12000),
    # evaluated on the unchanged a20000 test sets. _full = same model as
    # b2m_hmem128_gated_long (data effect); w768 = 1.5x width, memory
    # slots = width / 4 as at d512 (width effect at equal tokens).
    'b2m_hmem128_gated_full': dict(head_mode='stream', fold_impl='downsweep',
                                   use_metal=True, hmem_nb=128,
                                   hmem_decay='gated'),
    'w768_hmem192_gated_full': dict(head_mode='stream', fold_impl='downsweep',
                                    use_metal=True, hmem_nb=192,
                                    hmem_decay='gated'),
    'b2g_nmix_s43': dict(head_mode='stream', fold_impl='downsweep',
                         node_mix_rank=64, lowrank_gain=True),
    'b2_tie2_s43': dict(head_mode='stream', fold_impl='downsweep',
                        state_mult=2, state_tie=True),
    'b2_base_s43': dict(head_mode='stream', fold_impl='downsweep'),
    'b2_base_s44': dict(head_mode='stream', fold_impl='downsweep'),
}
STAGE2_BASE = {}
ARCH.update({
    's2_base': dict(STAGE2_BASE),
    's2_foldgate': dict(STAGE2_BASE, fold_gate='separate'),
    's2_h0': dict(STAGE2_BASE, fold_h0=True),
    's2_sx2': dict(STAGE2_BASE, state_mult=2),
    's2_sx2_wide': dict(STAGE2_BASE),
})
for _n, _kw in ARCH.items():
    RECIPE[_n] = dict(optimizer='muon', muon_lr=0.02,
                      muon_include='fusion_gate', muon_wd=0.01, **_kw)
    ARMS[_n] = ('bytes', 1024, 512, 128)
ARMS['st_wide'] = ('bytes', 1024, 640, 160)
ARMS['w768_hmem192_gated_full'] = ('bytes', 1024, 768, 192)
ARMS['d8_add_d256'] = ('bytes', 1024, 256, 64)
ARMS['d8s_add_d256'] = ('bytes', 1024, 256, 64)
ARMS['s2_sx2_wide'] = (('bytes', 1024, 800, 200)
                       if STAGE2_BASE.get('head_mode') == 'stream'
                       else ('bytes', 1024, 920, 230))
for _n in RELAX:
    ARMS[_n] = ('bytes', 1024, 512, 128)
for _n, _m in BIST.items():
    ARMS[_n] = ('bytes', 1024, 512, 128)


# LARGE-MODEL arms (Recall research §8c), registered on demand from the
# name: fw_d{d}_L{L}[_suffix] = the gated-memory model at width d
# (memory slots d / 4, as in every memory arm), L layers; L > 2 uses the
# additive residual with GPT-2 init scaling. Trained on a packed pool
# (--packed, e.g. FineWeb-Edu bytes) and tested on --test-pkl.
LAYERS = {}


def scale_arm(name):
    import re
    m = re.match(r'^fw_d(\d+)_L(\d+)(?:_\w+)?$', name)
    if not m or name in ARMS:
        return
    d, L = int(m[1]), int(m[2])
    kw = dict(head_mode='stream', fold_impl='downsweep', use_metal=True,
              hmem_nb=d // 4, hmem_decay='gated')
    if L > 2:
        kw.update(resid_mode='add', resid_init_scale='auto')
    ARCH[name] = kw
    RECIPE[name] = dict(optimizer='muon', muon_lr=0.02,
                        muon_include='fusion_gate', muon_wd=0.01, **kw)
    ARMS[name] = ('bytes', 1024, d, d // 4)
    LAYERS[name] = L


def load_corpus(repr_mode, T, max_articles):
    cache = os.path.join(
        ROOT, 'assets', f'corpus_{repr_mode}_T{T}_a{max_articles}.pkl')
    if not os.path.exists(cache):
        raise SystemExit(
            f"missing {cache}\nrun: python experiments/build_reprs.py "
            f"--max-articles {max_articles}")
    with open(cache, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', nargs='+', default=list(ARMS))
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--max-articles', type=int, default=20000)
    p.add_argument('--eval-cap', type=int, default=2048,
                   help='cap on eval_max_len; affects extrapolation '
                        'buckets only, never the in-length BPB gate')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default=None)
    p.add_argument('--packed', default=None,
                   help='train on a packed pool (experiments/build_packed.py, '
                        'e.g. assets/packed_bytes_T1024_all); evaluation stays '
                        'on the --max-articles corpus test sets, whose articles '
                        'the pool excludes by construction')
    p.add_argument('--save-every', type=int, default=0,
                   help='mid-training checkpoint cadence (steps); with '
                        '--resume a restarted run continues exactly')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--muon-lr-scale', type=float, default=None,
                   help='multiply the Muon step size by this factor for the '
                        'rest of training; unlike --muon-lr it never changes '
                        'the checkpoint tag, so it is safe to vary across a '
                        '--resume of the same run (Recall research §8f: use '
                        'this, not a lower base LR, to calm a run that is '
                        'skipping non-finite gradients often)')
    p.add_argument('--max-lr-scale', type=float, default=None,
                   help='multiply the Adam-group max_lr by this factor '
                        '(recipe default 1e-3); pair with --muon-lr-scale '
                        'when raising --batch above the protocol 32 '
                        '(sqrt(batch/32) scaling). Unlike a raw --max-lr it '
                        'applies through the recipe, and the value must be '
                        'kept constant across --resume sessions of the same '
                        'run (train() puts max_lr in the checkpoint tag)')
    p.add_argument('--grad-clip', type=float, default=None,
                   help='gradient-norm clip threshold (default in train(): '
                        '1.0); also safe to vary across --resume')
    p.add_argument('--test-pkl', default=None,
                   help='in-training test sets from this .test.pkl '
                        '(build_fineweb_bytes.py) instead of the '
                        '--max-articles corpus; needs --packed')
    p.add_argument('--pool-test', default=None,
                   help='also score the finished arm on the pool test sets '
                        '(experiments/eval_pool.py), e.g. '
                        'assets/packed_bytes_T1024_all')
    args = p.parse_args()

    if args.device is None:
        import torch
        args.device = ('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available()
                       else 'cpu')
    os.makedirs(OUT, exist_ok=True)
    summary_path = os.path.join(OUT, 'repr_summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    for name in args.arms:
        if name in summary:
            print(f"[skip] {name} already in {summary_path}")
            continue
        scale_arm(name)
        repr_mode, T, d, nb = ARMS[name]
        layers = LAYERS.get(name, args.layers)
        if args.test_pkl:
            assert args.packed and repr_mode == 'bytes'
            with open(args.test_pkl, 'rb') as f:
                tp = pickle.load(f)
            data = ([], [c for _, c in tp['test_short']],
                    [c for _, c in tp['test_long']], 259)
            meta = {'units_per_byte': 1.0, 'vocab_size': 259,
                    'raw_bytes_train': None}
        else:
            data, meta = load_corpus(repr_mode, T, args.max_articles)
        if args.packed:
            assert repr_mode == 'bytes'
            data = ([],) + tuple(data[1:])        # train rows come from the pool
        upb = meta['units_per_byte']
        print(f"\n{'=' * 70}\n[{name}] repr={repr_mode} T={T} d={d} nb={nb} "
              f"V={meta['vocab_size']}")
        print(f"  units/byte {upb:.4f}  ->  "
              f"{T / upb:.0f} bytes of context, "
              f"{args.batch * T / upb:.0f} bytes/step")
        # PER-ARM out_dir. train()'s checkpoint name is built from the
        # config tag, which encodes pe/fold/rot/opt but NOT vocab_size or
        # max_len -- the only two things that differ between these arms.
        # A shared out_dir therefore makes every arm overwrite the last
        # one's checkpoint and results.jsonl. (Observed: the byte run was
        # about to clobber the finished BPE checkpoint.)
        arm_dir = os.path.join(OUT, name)
        os.makedirs(arm_dir, exist_ok=True)
        t0 = time.time()
        # eval_max_len drives ONLY the extrapolation buckets, never the
        # in-length PPL that H1 is computed from -- so capping it cannot
        # affect the study's gate. It is capped because the byte arm's
        # T*4 = 4096 final evaluation was killed on a 16 GB machine after
        # training had already completed (3000/3000 steps, loss 1.95),
        # losing the checkpoint. 2x extrapolation is enough to keep the
        # bucket report meaningful and fits in memory.
        eml = min(T * 4, args.eval_cap)
        recipe = dict(device_kernels(RECIPE.get(name, {}), args.device))
        if args.muon_lr_scale is not None:
            recipe['muon_lr_resume_scale'] = args.muon_lr_scale
        if args.max_lr_scale is not None:
            recipe['max_lr'] = recipe.get('max_lr', 1e-3) * args.max_lr_scale
        if args.grad_clip is not None:
            recipe['grad_clip'] = args.grad_clip
        res = train(
            steps=args.steps, batch=args.batch, max_len=T,
            vocab_size=meta['vocab_size'], d=d, nb=nb,
            num_layers=layers, eval_max_len=eml,
            device=args.device, pe_mode='none', fold_mode='left',
            rot_mode='free', data=data, seed=args.seed,
            out_dir=arm_dir, tie=False,
            fold_bistable=BIST.get(name, 'off'),
            bist_rank=BIST_RANK.get(name, 0),
            fold_relax={**RELAX, **RELAX_M}.get(name, 'off'),
            level_grad_balance=LGB.get(name, 1.0),
            fold_grade=GRADE.get(name),
            packed_data=(os.path.join(ROOT, args.packed)
                         if args.packed and not os.path.isabs(args.packed)
                         else args.packed),
            save_every=args.save_every, resume=args.resume,
            **recipe)
        mins = (time.time() - t0) / 60

        ppl = res['test_perplexity_in_length']
        nats = math.log(ppl)
        bpb = nats_to_bpb(nats, upb)
        summary[name] = {
            'repr': repr_mode, 'max_len': T, 'd': d, 'nb': nb,
            'vocab_size': meta['vocab_size'], 'params': res['params'],
            'steps': args.steps, 'batch': args.batch,
            'units_per_byte': upb,
            'context_bytes': T / upb,
            'bytes_per_step': args.batch * T / upb,
            'ppl_in_units': ppl, 'nats_per_unit': nats,
            'bpb': bpb, 'minutes': mins,
            'raw_bytes_train': meta['raw_bytes_train'],
            'ckpt_dir': arm_dir, 'config': res.get('config'),
            'fold_bistable': BIST.get(name, 'off'),
            'bist_rank': BIST_RANK.get(name, 0),
            'fold_relax': RELAX.get(name, 'off'),
            'level_grad_balance': LGB.get(name, 1.0),
            'fold_grade': (list(GRADE[name]) if name in GRADE else None),
            'arch': ARCH.get(name),
            'packed': args.packed, 'layers': layers,
            'test_pkl': args.test_pkl,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        if args.pool_test:
            from eval_pool import score_arm, test_sets
            pe = score_arm(name, test_sets(args.pool_test, T, args.test_pkl),
                           args.device, T)
            summary[name]['pool_eval'] = pe
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  [{name}] pool eval BPB: " +
                  "  ".join(f"{k} {v:.4f}" for k, v in pe.items()))
        print(f"  [{name}] PPL/unit {ppl:.2f}  nats/unit {nats:.4f}  "
              f"**BPB {bpb:.4f}**  params {res['params']:,}  "
              f"({mins:.0f} min)")

    print(f"\n{'=' * 70}\nREPRESENTATION STUDY — bits per byte (lower is better)")
    print(f"  {'arm':<12} {'repr':<6} {'params':>11} {'ctx bytes':>10} "
          f"{'PPL/unit':>9} {'BPB':>8}")
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]['bpb']):
        print(f"  {k:<12} {v['repr']:<6} {v['params']:>11,} "
              f"{v['context_bytes']:>10.0f} {v['ppl_in_units']:>9.2f} "
              f"{v['bpb']:>8.4f}")
    print(f"\n  wrote {summary_path}")
    print("  NOTE: PPL/unit is NOT comparable across representations "
          "(different\n  units); BPB is. Compare byte arms to the BPE arm "
          "at similar params.")


if __name__ == '__main__':
    main()
