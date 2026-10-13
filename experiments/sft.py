"""Chat fine-tuning (SFT) of a pretrained OPERA arm on the byte-level chat
pool (experiments/build_chat_bytes.py), through opera_lm.train.train():
same model and recipe as the arm, weights from its final checkpoint
(init_weights_from), fresh optimizer, lower learning rate, short warmup,
conversations up to --T bytes. The loss covers whole conversations (the
pool's documented choice, as in opera-chat), not only assistant turns.

Length of the run, first match wins: --steps, else --epochs (passes over
the pool's conversations), else --hours (converted with --sec-per-step,
or with the A100 benchmark when that is not given).

Resumable: re-run the same command after a disconnect. The first launch
writes <out>/sft_plan.json and every later launch reuses its step count,
so a resumed run keeps the exact cosine schedule even if the benchmark or
the requested length changed in between. A deliberate retry with a
different length, batch, T, learning rate, pool or base checkpoint must
start fresh: rename the old <arm>_sft directory first.

  python experiments/sft.py --arm fw_d3072_L2_r28 \\
      --pool /content/data/chat_bytes_T2048 --epochs 1.5

Writes to $OPERA_RUNS/<arm>_sft; export with
  python experiments/export_hf.py --arm <arm> --ckpt-dir $OPERA_RUNS/<arm>_sft ...
"""
import argparse
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import repr_study as rs                                      # noqa: E402
from opera_lm.train import train                             # noqa: E402

SAMPLES = ["Hi! Who are you?",
           "Give me three tips for staying focused while studying.",
           "What is photosynthesis? Explain it simply."]
# train() reads at most the first 5,000 held-out sequences (its final eval;
# periodic evals use 200), so loading more only costs memory.
N_TEST = 5000
PLAN_KEYS = ('batch', 'T', 'lr_scale', 'pool', 'init_weights_from')


def estimate_sec_per_step(d, layers, batch, T):
    """Step time from the A100 benchmark (runs/bench.jsonl, measured at
    T=1024): look up the batch with the same bytes per step, then scale by
    tree depth, log2(T) / log2(1024) (one extra level at T=2048 = +10%)."""
    from ladder import est_ms, load_bench, n_params
    ms, how = est_ms(load_bench(os.path.join(rs.OUT, 'bench.jsonl')),
                     d, layers, max(1, batch * T // 1024), n_params(d, layers))
    if not ms:
        sys.exit('no benchmark (runs/bench.jsonl): pass --sec-per-step, '
                 '--steps or --epochs')
    return ms / 1000 * math.log2(T) / 10, how


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True)
    p.add_argument('--pool', required=True, help='chat pool prefix')
    p.add_argument('--T', type=int, default=2048,
                   help='training length in bytes; must be >= the length the '
                        'pool was built at, or final assistant turns (and '
                        'their EOS) get truncated')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--epochs', type=float, default=0,
                   help='passes over the pool\'s conversations '
                        '(steps = epochs * n_seqs / batch)')
    p.add_argument('--hours', type=float, default=1.0,
                   help='used when --steps and --epochs are both 0')
    p.add_argument('--sec-per-step', type=float, default=None,
                   help='step time for --hours (default: the A100 benchmark)')
    p.add_argument('--lr-scale', type=float, default=0.3,
                   help='fraction of the protocol peak learning rates '
                        '(AdamW 1e-3, Muon 0.02) -- the base for batch '
                        '16 x 2048 bytes, the protocol 32 x 1024 bytes per step')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    # ---- everything below up to train() is a pre-flight: fail in seconds,
    # not after hours of GPU time
    rs.scale_arm(args.arm)
    if args.arm not in rs.ARMS:
        sys.exit(f'unknown arm {args.arm!r}')
    _, _, d, nb = rs.ARMS[args.arm]
    layers = rs.LAYERS.get(args.arm, 2)
    pre_dir = os.path.join(rs.OUT, args.arm)
    if not os.path.isdir(pre_dir):
        sys.exit(f'{pre_dir}: no pretraining run for {args.arm}')
    pts = [f for f in os.listdir(pre_dir)
           if f.endswith('.pt') and not f.endswith('_train_ckpt.pt')]
    if len(pts) != 1:
        sys.exit(f'{pre_dir}: expected exactly one final checkpoint, found '
                 f'{pts or "none"} (is pretraining finished?)')
    init_from = os.path.join(pre_dir, pts[0])

    for suffix in ('.tokens.npy', '.offsets.npy', '.test.pkl', '.meta.json'):
        if not os.path.exists(args.pool + suffix):
            sys.exit(f'chat pool incomplete: {args.pool + suffix} missing')
    meta = json.load(open(args.pool + '.meta.json'))
    if args.T < meta.get('T', args.T):
        sys.exit(f'--T {args.T} < the pool\'s build length {meta["T"]}: '
                 f'conversations would be cut mid-turn, losing the final '
                 f'assistant turns and their EOS; rebuild the pool at --T '
                 f'{args.T} or train at --T {meta["T"]}')

    out_dir = os.path.join(rs.OUT, args.arm + '_sft')
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, 'sft_plan.json')
    request = dict(batch=args.batch, T=args.T, lr_scale=args.lr_scale,
                   pool=os.path.abspath(args.pool), init_weights_from=init_from)

    # the plan binds once the run has written a checkpoint; before that a
    # relaunch (e.g. with corrected options) simply re-plans
    started = any(f.endswith('.pt') for f in os.listdir(out_dir))
    if started and os.path.exists(plan_path):
        plan = json.load(open(plan_path))
        changed = [k for k in PLAN_KEYS if plan.get(k) != request[k]]
        if changed:
            sys.exit(f'{out_dir} belongs to a different SFT run ({", ".join(changed)} '
                     f'changed: {[(plan.get(k), request[k]) for k in changed]}). '
                     f'To start a new run, rename that directory first.')
        steps = plan['steps']
        print(f'resuming the SFT plan in {plan_path}: {steps} steps '
              f'(the length options of this launch are ignored)', flush=True)
    else:
        steps = args.steps
        if not steps and args.epochs:
            steps = math.ceil(args.epochs * meta['n_seqs'] / args.batch)
        if not steps:
            sps, how = ((args.sec_per_step, 'given') if args.sec_per_step
                        else estimate_sec_per_step(d, layers, args.batch, args.T))
            print(f'step time ~{sps:.3f}s ({how})', flush=True)
            steps = int(args.hours * 3600 / sps)
        if steps < 1:
            sys.exit('the requested length is zero steps')
        if started:
            # a run begun before sft_plan.json existed: adopt it. At the
            # default batch/T the step formula is unchanged, so the same
            # command reproduces the original step count.
            print(f'WARNING: {out_dir} has checkpoints but no plan; adopting '
                  f'them with {steps} steps -- the length options must match '
                  f'the original launch', flush=True)
        plan = dict(request, steps=steps)
        json.dump(plan, open(plan_path, 'w'), indent=2)

    epochs = steps * args.batch / meta['n_seqs']
    print(f"SFT {args.arm}: {steps} steps x {args.batch} x <= {args.T} bytes "
          f"(~{epochs:.2f} epochs of {meta['n_seqs']:,} conversations, "
          f"{meta['total_tokens'] / 1e6:.0f}M chat bytes)", flush=True)

    tp = pickle.load(open(args.pool + '.test.pkl', 'rb'))
    # the pool's held-out set is a seeded random sample, so its head is an
    # unbiased subset
    data = ([], [c for _, c in tp['test_short'][:N_TEST]], [], 259)
    del tp
    recipe = rs.device_kernels(rs.RECIPE[args.arm], args.device)
    recipe['muon_lr'] = recipe.get('muon_lr', 0.02) * args.lr_scale
    if args.device == 'cuda' and os.environ.get('OPERA_TRITON_CUSTOM_OP') == '1':
        # The custom-op path gives rot_free a gradient under torch.compile;
        # the default path (used for pretraining) leaves it frozen. A
        # Muon-routed parameter seeing its first gradient at full lr is the
        # step-147 divergence (Recall research §8f), so ramp its lr from 0.
        recipe['muon_fresh_substr'] = 'rot_free'
        recipe['muon_fresh_warmup'] = max(1, min(500, steps // 10))
        print(f"OPERA_TRITON_CUSTOM_OP=1: rot_free trains for the first time; "
              f"its Muon lr ramps over {recipe['muon_fresh_warmup']} steps", flush=True)

    # ---- the run
    res = train(steps=steps, batch=args.batch, max_len=args.T, vocab_size=259,
                d=d, nb=nb, num_layers=layers, eval_max_len=args.T,
                device=args.device, pe_mode='none', fold_mode='left',
                rot_mode='free', data=data, seed=0, out_dir=out_dir, tie=False,
                packed_data=args.pool, init_weights_from=init_from,
                max_lr=1e-3 * args.lr_scale, warmup_steps=min(200, steps // 10),
                save_every=args.save_every, resume=True, **recipe)
    ppl = res['test_perplexity_in_length']
    print(f"SFT done: held-out chat BPB {math.log2(ppl):.4f} "
          f"(whole conversations, first {len(data[1]):,} held-out)", flush=True)

    # ---- a few sample replies with the incremental decoder, from the exact
    # checkpoint this run wrote (not whatever else sits in the directory)
    import torch
    from chat_probe import load_sft_model
    from opera_lm.chat import format_prompt, stream_generate
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    ckpt = f"opera_v8_0_{res['config'].replace('+', '_')}.pt"
    m, _ = load_sft_model(args.arm, out_dir, args.device, ckpt_name=ckpt)
    for q in SAMPLES:
        text = ''
        for text in stream_generate(m, format_prompt([], q), max_new=300, seed=0):
            pass
        print(f"\n>>> {q}\n{text}", flush=True)


if __name__ == '__main__':
    main()
