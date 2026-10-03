"""Export a trained OPERA run as a Hugging Face model folder
(config.json + model.safetensors + README.md) and optionally push it.

  python experiments/export_hf.py --arm fw_d3072_L2_r16 \\
      [--ckpt-dir $OPERA_RUNS/fw_d3072_L2_r16_sft] --out /content/export \\
      [--push your-name/opera-lm-chat]      # needs HF_TOKEN in the env

The model's constructor flags come from the arm name (repr_study), with
the kernel flags removed: the export runs on the eager path (CPU / any
GPU) and with opera_lm.incremental.OperaDecoder.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import repr_study as rs                                      # noqa: E402
from arm_utils import _model_kwargs                          # noqa: E402

CARD = """---
license: mit
library_name: opera-lm
tags: [byte-level, attention-free, no-positional-encoding, holographic-memory]
---
# {name}

OPERA-LM: an attention-free, byte-level language model. Bottom-up
composition over a binary (Fenwick) tree replaces self-attention, there is
no positional encoding, and in-context recall comes from a gated quaternion
holographic memory. Code: https://github.com/Merna-Khalid/OPERA-LM

- parameters: {params:,}
- width {d}, {layers} layers, {nb} memory slots per layer
- {stage}
{evals}
```python
from opera_lm.chat import load_model, format_prompt, stream_generate
model, cfg = load_model("{repo}")
for text in stream_generate(model, format_prompt([], "Hello! Who are you?")):
    pass
print(text)
```
Small research model: expect fluent-looking text, not reliable facts.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, help='pretraining arm, e.g. fw_d3072_L2_r16')
    p.add_argument('--ckpt-dir', default=None,
                   help='run folder with the final .pt (default: $OPERA_RUNS/<arm>)')
    p.add_argument('--out', required=True)
    p.add_argument('--push', default=None, help='Hub repo id, e.g. user/opera-lm-chat')
    p.add_argument('--stage', default=None, help='free-text line for the model card')
    args = p.parse_args()

    rs.scale_arm(args.arm)
    repr_mode, T, d, nb = rs.ARMS[args.arm]
    layers = rs.LAYERS.get(args.arm, 2)
    kw = dict(vocab_size=259, d=d, nb=nb, num_layers=layers, pe_mode='none',
              fold_mode='left', rot_mode='free',
              **_model_kwargs(rs.device_kernels(rs.ARCH.get(args.arm), 'cpu')))
    ckpt_dir = args.ckpt_dir or os.path.join(rs.OUT, args.arm)
    pts = [f for f in os.listdir(ckpt_dir)
           if f.endswith('.pt') and not f.endswith('_train_ckpt.pt')]
    if len(pts) != 1:
        sys.exit(f"{ckpt_dir}: expected one final checkpoint, got {pts}")
    sd = torch.load(os.path.join(ckpt_dir, pts[0]), map_location='cpu',
                    weights_only=True)
    sd = sd.get('model', sd)

    from opera_lm import OperaSpinorFenwickTree
    model = OperaSpinorFenwickTree(**kw)
    model.load_state_dict(sd, strict=True)
    params = sum(p.numel() for p in model.parameters())

    os.makedirs(args.out, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous().clone() for k, v in model.state_dict().items()},
              os.path.join(args.out, 'model.safetensors'))
    summ_path = os.path.join(rs.OUT, 'repr_summary.json')
    evals = {}
    if os.path.exists(summ_path):
        s = json.load(open(summ_path)).get(args.arm, {})
        evals = {'fineweb_bpb': s.get('bpb'), **(s.get('pool_eval') or {})}
    cfg = {'format': 'opera-chat-bytes-v1', 'arm': args.arm,
           'model_kwargs': kw, 'params': params, 'eos_id': 2,
           'byte_offset': 3, 'chat_max_len': 2048,
           'pretrain_eval_bpb': evals, 'source_ckpt': pts[0]}
    json.dump(cfg, open(os.path.join(args.out, 'config.json'), 'w'), indent=2)
    stage = args.stage or ('pretrained on FineWeb-Edu bytes, chat-tuned on smol-smoltalk'
                           if args.ckpt_dir else 'pretrained on FineWeb-Edu bytes')
    ev = ''.join(f"- {k}: {v:.4f} BPB\n" for k, v in evals.items() if isinstance(v, float))
    with open(os.path.join(args.out, 'README.md'), 'w') as f:
        f.write(CARD.format(name=args.push or args.arm, params=params, d=d,
                            layers=layers, nb=kw.get('hmem_nb', nb), stage=stage,
                            evals=('\nPretraining evaluation (held-out, bits per byte):\n' + ev)
                            if ev else '', repo=args.push or args.out))
    print(f"exported {args.arm} ({params:,} params) -> {args.out}")

    # Round-trip gate before anything is pushed: load the exported folder
    # back exactly the way a downloader will (chat.load_model: config.json
    # + safetensors) and require identical logits on a fixed prompt. A
    # constructor flag that changes BEHAVIOR without changing parameter
    # shapes (hmem_decay, resid_mode, fold_impl, ...) survives a strict
    # state_dict load but shows up here.
    from opera_lm.chat import format_prompt, load_model
    m2, _ = load_model(args.out, device='cpu')
    ids = format_prompt([], 'Hello! Who are you?')
    t = torch.tensor([ids]); lens = torch.tensor([len(ids)])
    with torch.no_grad():
        l1 = [l.float() for l in model(t, lens).logits]
        l2 = [l.float() for l in m2(t, lens).logits]
    diff = max((a - b).abs().max().item() for a, b in zip(l1, l2))
    assert diff < 1e-4, f'export round-trip FAILED: logits differ by {diff}'
    print(f'round-trip verified: exported folder reproduces the checkpoint '
          f'(max |dlogits| {diff:.2e})', flush=True)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get('HF_TOKEN'))
        api.create_repo(args.push, repo_type='model', exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type='model')
        print(f"pushed -> https://huggingface.co/{args.push}")


if __name__ == '__main__':
    main()
