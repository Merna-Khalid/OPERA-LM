"""train_chat.py -- thin CLI over opera_lm.train.train() for the chat pkl.

The pkl (prepare_data.py) already satisfies train()'s injected-data
contract (train/test_short <= max_len, test_long in (max_len,
eval_max_len]), so it is passed straight through as
data=(train, test_short, test_long, vocab_size) with idx2word=None.

Usage:
  python train_chat.py --data opera-chat/data_chat.pkl --steps 200
  python train_chat.py --data opera-chat/data_chat.pkl --device cuda \
      --steps 20000 --resume
"""
import argparse
import glob
import json
import os
import pickle

from chat_common import ensure_opera_lm, load_tokenizer

ensure_opera_lm()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="opera-chat/data_chat.pkl")
    p.add_argument("--tokenizer", default=None,
                   help="optional; only for a vocab-size sanity check")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-max-len", type=int, default=1024)
    p.add_argument("--d", type=int, default=640)
    p.add_argument("--nb", type=int, default=160)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--out-dir", default="runs_chat")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-msup", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--curriculum-t0", type=int, default=64,
                   help="curriculum starting length (default 64)")
    p.add_argument("--curriculum-every", type=int, default=250,
                   help="steps per curriculum doubling (default 250)")
    p.add_argument("--optimizer", default="adamw",
                   choices=["adamw", "muon"],
                   help="muon = roadmap T0.1: Muon on matrix params "
                        "(rot_free 3x3 maps, cross_mlp, head), AdamW on "
                        "embeddings/gates/gains")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--readout", default="none",
                   choices=["none", "multistate"],
                   help="multistate = roadmap T0.4: fixed learned reduction "
                        "over the prefix's raw Fenwick blocks")
    p.add_argument("--mem", default="none", choices=["none", "delta"],
                   help="delta = roadmap T1.4: delta-rule memory channel")
    p.add_argument("--mem-dim", type=int, default=128)
    p.add_argument("--compile", default="default",
                   choices=["default", "reduce-overhead", "max-autotune",
                            "off"])
    p.add_argument("--grad-checkpoint", default="", choices=["", "level", "layer"],
                   help="activation checkpointing to trade compute for "
                        "memory (~25-35%% slower steps): 'level' wraps "
                        "each compose node, 'layer' wraps each whole "
                        "layer (cheaper but incompatible with msup / "
                        "--no-msup is required for 'layer'). Use 'level' "
                        "if a curriculum stage OOMs (activation memory "
                        "grows with --curriculum's T_cur, not just "
                        "--max-len).")
    p.add_argument("--metal", action="store_true",
                   help="fused Metal kernels for the compose node "
                        "(Apple Silicon; now supports rot_mode='free', "
                        "not just 'so3')")
    p.add_argument("--triton", action="store_true",
                   help="fused Triton kernel for the compose node (CUDA; "
                        "same math/contract as --metal, ported from it -- "
                        "see opera_lm/triton_kernel.py). Requires a CUDA "
                        "device and `triton` installed; run "
                        "`python3 -m opera_lm.triton_kernel --test` on "
                        "the target GPU before a real training run, since "
                        "this path has not been hardware-verified.")
    p.add_argument("--init-weights-from", default=None,
                   help="load an existing checkpoint's weights before "
                        "training (short fine-tune phases, e.g. state "
                        "passing); independent of --resume")
    p.add_argument("--ddp", action="store_true",
                   help="multi-GPU data parallelism (e.g. Kaggle's 2xT4). "
                        "--batch is PER-GPU. Launch with torchrun, not "
                        "plain python: `torchrun --nproc_per_node=2 "
                        "train_chat.py --ddp --device cuda ...`. Forces "
                        "--compile off (see opera_lm.train.train's "
                        "docstring note); verified on CPU/gloo via "
                        "opera_lm.selftest, not yet on real multi-GPU "
                        "hardware -- report back what breaks.")
    a = p.parse_args()

    with open(a.data, "rb") as f:
        d = pickle.load(f)
    train_data, test_short, test_long = (d["train"], d["test_short"],
                                         d["test_long"])
    vocab_size = d["vocab_size"]
    assert train_data and test_short and test_long, \
        "train/test_short/test_long must all be non-empty"
    top = max(max(s) for s in train_data + test_short + test_long)
    assert top < vocab_size, f"token id {top} >= vocab_size {vocab_size}"
    if a.tokenizer:
        tv = load_tokenizer(a.tokenizer).get_vocab_size()
        assert tv == vocab_size, \
            f"tokenizer vocab {tv} != pkl vocab {vocab_size}"
    print(f"data: train={len(train_data)} short={len(test_short)} "
          f"long={len(test_long)} vocab={vocab_size}", flush=True)

    from opera_lm.train import train
    fold_gate_bias = (2.0, 0.0, -2.0)
    results = train(
        steps=a.steps, batch=a.batch, max_len=a.max_len,
        vocab_size=vocab_size, d=a.d, nb=a.nb, num_layers=a.num_layers,
        eval_max_len=a.eval_max_len, device=a.device,
        tie=True, msup=not a.no_msup, msup_weight=0.1,
        pe_mode='none', fold_mode='left', rot_mode='free',
        seed=a.seed, data_mode='chat', out_dir=a.out_dir,
        save_every=a.save_every, resume=a.resume,
        compile_mode=a.compile, gpu_data=True, aux_frac=0.25,
        max_lr=a.max_lr, warmup_steps=a.warmup_steps,
        fold_gate_bias=fold_gate_bias,
        curriculum=None if a.no_curriculum else (a.curriculum_t0, a.curriculum_every),
        data=(train_data, test_short, test_long, vocab_size),
        idx2word=None, optimizer=a.optimizer, muon_lr=a.muon_lr,
        readout_mode=a.readout, mem_mode=a.mem, mem_dim=a.mem_dim,
        use_metal=a.metal, use_triton=a.triton, init_weights_from=a.init_weights_from,
        ddp=a.ddp, grad_checkpoint=a.grad_checkpoint)

    # Under --ddp, train() returns None on every rank except 0 (see its
    # docstring note); only rank 0 does the post-training packaging
    # below, matching train()'s own rank-gating so ranks >0 don't race
    # rank 0 to write the same files.
    if a.ddp and int(os.environ.get("RANK", "0")) != 0:
        return

    # The final checkpoint is a raw state_dict; train()'s tag is derived
    # from its kwargs, so glob for the newest non-resume .pt instead of
    # reconstructing the filename (rank-tagged _train_ckpt_rankN.pt
    # files from --ddp are also excluded by the same suffix check).
    finals = [f for f in glob.glob(os.path.join(a.out_dir, "*.pt"))
              if not f.endswith("_train_ckpt.pt")
              and "_train_ckpt_rank" not in f]
    assert finals, f"no final .pt found in {a.out_dir}"
    ckpt = max(finals, key=os.path.getmtime)
    cfg = {"vocab_size": vocab_size, "d": a.d, "nb": a.nb,
           "num_layers": a.num_layers, "tie": True, "pe_mode": "none",
           "fold_mode": "left", "rot_mode": "free",
           "fold_gate_bias": list(fold_gate_bias),
           "readout_mode": a.readout, "mem_mode": a.mem,
           "mem_dim": a.mem_dim, "ckpt": ckpt}
    cfg_path = os.path.join(a.out_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"model_config.json -> {cfg_path}", flush=True)
    print(f"final ckpt: {ckpt}", flush=True)
    _ = results  # train() already printed/saved its own metrics


if __name__ == "__main__":
    main()
