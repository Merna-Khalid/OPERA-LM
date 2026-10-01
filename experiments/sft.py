"""Chat fine-tuning (SFT) of a pretrained OPERA arm on the byte-level chat
pool (experiments/build_chat_bytes.py), through opera_lm.train.train():
same model and recipe as the arm, weights from its final checkpoint
(init_weights_from), fresh optimizer, lower learning rate, short warmup,
conversations up to --T bytes. Resumable (--save-every / re-run).

  python experiments/sft.py --arm fw_d3072_L2_r16 \\
      --pool /content/data/chat_bytes_T2048 --hours 1.0

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--pool', required=True, help='chat pool prefix')
    p.add_argument('--T', type=int, default=2048)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--hours', type=float, default=1.0,
                   help='sets --steps from --sec-per-step when --steps is 0')
    p.add_argument('--sec-per-step', type=float, default=None,
                   help='training step time at this batch and T '
                        '(default: from the A100 benchmark)')
    p.add_argument('--lr-scale', type=float, default=0.3,
                   help='fraction of the pretraining peak learning rates')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    rs.scale_arm(args.arm)
    _, _, d, nb = rs.ARMS[args.arm]
    layers = rs.LAYERS.get(args.arm, 2)
    pre_dir = os.path.join(rs.OUT, args.arm)
    pts = [f for f in os.listdir(pre_dir)
           if f.endswith('.pt') and not f.endswith('_train_ckpt.pt')]
    assert len(pts) == 1, f"{pre_dir}: expected one final checkpoint, got {pts}"
    meta = json.load(open(args.pool + '.meta.json'))
    steps = args.steps
    if not steps:
        sps = args.sec_per_step
        if sps is None:
            # same bytes per step as pretraining (16 x 2048 = 32 x 1024),
            # +10% for the extra tree level at T=2048
            from ladder import est_ms, load_bench, n_params
            ms, how = est_ms(load_bench(os.path.join(rs.OUT, 'bench.jsonl')),
                             d, layers, 32, n_params(d, layers))
            assert ms, 'no benchmark (runs/bench.jsonl): pass --sec-per-step'
            sps = 1.1 * ms / 1000
            print(f"step time ~{sps:.3f}s ({how})", flush=True)
        steps = int(args.hours * 3600 / sps)
    epochs = steps * args.batch / meta['n_seqs']
    print(f"SFT {args.arm}: {steps} steps x {args.batch} x <= {args.T} bytes "
          f"(~{epochs:.2f} epochs of {meta['total_tokens'] / 1e6:.0f}M chat bytes)", flush=True)
    tp = pickle.load(open(args.pool + '.test.pkl', 'rb'))
    data = ([], [c for _, c in tp['test_short']], [], 259)
    out_dir = os.path.join(rs.OUT, args.arm + '_sft')
    os.makedirs(out_dir, exist_ok=True)
    recipe = rs.device_kernels(rs.RECIPE[args.arm], args.device)
    recipe['muon_lr'] = recipe.get('muon_lr', 0.02) * args.lr_scale
    res = train(steps=steps, batch=args.batch, max_len=args.T, vocab_size=259,
                d=d, nb=nb, num_layers=layers, eval_max_len=args.T,
                device=args.device, pe_mode='none', fold_mode='left',
                rot_mode='free', data=data, seed=0, out_dir=out_dir, tie=False,
                packed_data=args.pool, init_weights_from=os.path.join(pre_dir, pts[0]),
                max_lr=1e-3 * args.lr_scale, warmup_steps=min(200, steps // 10),
                save_every=args.save_every, resume=True, **recipe)
    ppl = res['test_perplexity_in_length']
    print(f"SFT done: held-out chat BPB {math.log(ppl) / math.log(2):.4f}", flush=True)

    # a few sample replies with the incremental decoder (eager, CPU-safe)
    from opera_lm import OperaSpinorFenwickTree
    from opera_lm.chat import format_prompt, stream_generate
    import torch
    from arm_utils import _model_kwargs
    kw = dict(vocab_size=259, d=d, nb=nb, num_layers=layers, pe_mode='none',
              fold_mode='left', rot_mode='free',
              **_model_kwargs(rs.device_kernels(rs.ARCH[args.arm], 'cpu')))
    m = OperaSpinorFenwickTree(**kw)
    ck = [f for f in os.listdir(out_dir) if f.endswith('.pt') and not f.endswith('_train_ckpt.pt')]
    m.load_state_dict(torch.load(os.path.join(out_dir, ck[0]), map_location='cpu',
                                 weights_only=True))
    m = m.to(args.device).eval()
    for q in SAMPLES:
        text = ''
        for text in stream_generate(m, format_prompt([], q), max_new=300, seed=0):
            pass
        print(f"\n>>> {q}\n{text}", flush=True)


if __name__ == '__main__':
    main()
