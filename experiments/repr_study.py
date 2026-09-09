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
OUT = os.path.join(ROOT, 'runs_reprs')

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
for _n in RELAX:
    ARMS[_n] = ('bytes', 1024, 512, 128)
for _n, _m in BIST.items():
    ARMS[_n] = ('bytes', 1024, 512, 128)


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
    args = p.parse_args()

    if args.device is None:
        import torch
        args.device = ('mps' if torch.backends.mps.is_available()
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
        repr_mode, T, d, nb = ARMS[name]
        data, meta = load_corpus(repr_mode, T, args.max_articles)
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
        res = train(
            steps=args.steps, batch=args.batch, max_len=T,
            vocab_size=meta['vocab_size'], d=d, nb=nb,
            num_layers=args.layers, eval_max_len=eml,
            device=args.device, pe_mode='none', fold_mode='left',
            rot_mode='free', data=data, seed=args.seed,
            out_dir=arm_dir, tie=False,
            fold_bistable=BIST.get(name, 'off'),
            bist_rank=BIST_RANK.get(name, 0),
            fold_relax=RELAX.get(name, 'off'),
            level_grad_balance=LGB.get(name, 1.0),
            **RECIPE.get(name, {}))
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
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
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
