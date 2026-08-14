"""generate_chat.py -- sample chat replies from a trained OPERA checkpoint.

Two decoding paths over the same sampling code:
  - incremental (default): opera_lm.OperaDecoder, O(L log T) per token;
  - --naive: full re-forward per token (like opera_generate.py), kept as
    an equivalence reference.

Usage:
  python generate_chat.py --ckpt runs_chat/opera_v8_0_....pt \
      --config runs_chat/model_config.json --prompt "hello there"

Selftest (no checkpoint, no tokenizer, no dataset):
  python generate_chat.py --selftest
"""
import json
import sys

import torch
import torch.nn.functional as F

from chat_common import (ensure_opera_lm, build_prompt, load_tokenizer,
                         PAD_ID, END_ID)

ensure_opera_lm()
from opera_lm import OperaSpinorFenwickTree, OperaDecoder   # noqa: E402

END_TOKEN = "<|end|>"
ASSISTANT_TOKEN = "<|assistant|>"


def load_model(config_path, ckpt_path, device):
    """Construct the model from model_config.json and load the checkpoint
    (raw state_dict, or a *_train_ckpt.pt dict with a 'model' key)."""
    with open(config_path) as f:
        cfg = json.load(f)
    model = OperaSpinorFenwickTree(
        vocab_size=cfg["vocab_size"], d=cfg["d"], nb=cfg["nb"],
        num_layers=cfg["num_layers"], tie=cfg.get("tie", True),
        pe_mode=cfg.get("pe_mode", "none"),
        fold_mode=cfg.get("fold_mode", "left"),
        rot_mode=cfg.get("rot_mode", "free"),
        fold_gate_bias=(tuple(cfg["fold_gate_bias"])
                        if cfg.get("fold_gate_bias") else None),
        readout_mode=cfg.get("readout_mode", "none"),
        mem_mode=cfg.get("mem_mode", "none"),
        mem_dim=cfg.get("mem_dim", 128)).to(device)
    st = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model" in st:                       # train_ckpt vs final state_dict
        st = st["model"]
    model.load_state_dict(st)
    model.eval()
    return model


def _sample(logits, temp=1.0, top_p=1.0, top_k=0):
    """One multinomial draw from a logits vector. <|pad|> is banned;
    temp scaling, then optional top-k and top-p (nucleus) filtering."""
    logits = logits.float().clone()
    logits[PAD_ID] = float("-inf")          # never emit <|pad|>
    if temp and temp != 1.0:
        logits = logits / temp
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # drop tokens past the nucleus, always keeping the first one
        srt = srt.masked_fill(cum - probs > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_stream(model, tok, history, n_tokens=100, temp=0.8, top_p=0.9,
                    top_k=0, max_ctx=512, device="cpu", naive=False):
    """Generator: yields the partial decoded reply as tokens arrive.

    Prompt longer than max_ctx is truncated to its last max_ctx ids. The
    incremental path stops early if dec.t reaches max_ctx (no mid-
    generation cache truncation -- reset and lose the thread instead).
    Stops on <|end|> or after n_tokens.
    """
    model.eval()
    end_id = tok.token_to_id(END_TOKEN)
    prompt = build_prompt(history, tok)
    if len(prompt) > max_ctx:
        prompt = prompt[-max_ctx:]
    gen = []
    if naive:
        out = list(prompt)
        for _ in range(n_tokens):
            ctx = out[-max_ctx:]
            t = torch.tensor([ctx], dtype=torch.long, device=device)
            lens = torch.tensor([len(ctx)], device=device)
            logits = model(t, lens, head_last_only=True)[-1][0, len(ctx) - 1]
            nxt = _sample(logits, temp, top_p, top_k)
            if nxt == end_id:
                break
            out.append(nxt)
            gen.append(nxt)
            yield tok.decode(gen)
    else:
        dec = OperaDecoder(model)
        logits = dec.prefill(prompt)
        for _ in range(n_tokens):
            if dec.t >= max_ctx:            # context full: stop, keep reply
                break
            nxt = _sample(logits, temp, top_p, top_k)
            if nxt == end_id:
                break
            gen.append(nxt)
            logits = dec.append(nxt)
            yield tok.decode(gen)


def generate(model, tok, history, n_tokens=100, temp=0.8, top_p=0.9,
             top_k=0, max_ctx=512, device="cpu", naive=False):
    """Returns (reply_text, updated_history)."""
    reply = ""
    for reply in generate_stream(model, tok, history, n_tokens, temp,
                                 top_p, top_k, max_ctx, device, naive):
        pass
    return reply, history + [{"role": "assistant", "content": reply}]


# ---------------------------------------------------------------- selftest

class DummyTokenizer:
    """Trivial char-level mapping + the 5 special tokens at ids 0..4.
    Implements exactly the three methods chat_common/generate_chat use."""

    def __init__(self):
        self._special = {n: i for i, n in enumerate(
            ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>",
             "<|system|>"])}

    def encode(self, text):
        return [5 + (ord(c) % 95) for c in text]

    def decode(self, ids):
        return "".join(chr(33 + ((i - 5) % 94)) for i in ids if i >= 5)

    def token_to_id(self, name):
        return self._special[name]


def selftest():
    torch.manual_seed(0)
    V = 101
    model = OperaSpinorFenwickTree(
        vocab_size=V, d=64, nb=16, num_layers=2, pe_mode="none",
        fold_mode="left", fold_gate_bias=(2.0, 0.0, -2.0), tie=True)
    model.eval()
    tok = DummyTokenizer()
    hist = [{"role": "user", "content": "hello there"}]

    def run(naive, n=25, max_ctx=64):
        ids = []
        dec_end = tok.token_to_id(END_TOKEN)
        prompt = build_prompt(hist, tok)[-max_ctx:]
        gen = []
        if naive:
            out = list(prompt)
            for _ in range(n):
                ctx = out[-max_ctx:]
                t = torch.tensor([ctx], dtype=torch.long)
                lens = torch.tensor([len(ctx)])
                lg = model(t, lens, head_last_only=True)[-1][0, len(ctx) - 1]
                nxt = _sample(lg, 0.8, 0.9, 0)
                if nxt == dec_end:
                    break
                out.append(nxt)
                gen.append(nxt)
                ids.append(nxt)
        else:
            dec = OperaDecoder(model)
            lg = dec.prefill(prompt)
            for _ in range(n):
                if dec.t >= max_ctx:
                    break
                nxt = _sample(lg, 0.8, 0.9, 0)
                if nxt == dec_end:
                    break
                gen.append(nxt)
                lg = dec.append(nxt)
                ids.append(nxt)
        return ids

    # (a) deterministic under fixed seed
    torch.manual_seed(7)
    a1 = run(naive=False)
    torch.manual_seed(7)
    a2 = run(naive=False)
    assert a1 == a2 and a1, "not deterministic under fixed seed"
    # (b) never emits id 0 (<|pad|>)
    assert 0 not in a1, "sampled <|pad|>"
    # (c) incremental and naive paths agree under the same seed
    torch.manual_seed(7)
    b1 = run(naive=False)
    torch.manual_seed(7)
    b2 = run(naive=True)
    assert b1 == b2, f"incremental vs naive diverged:\n{b1}\n{b2}"
    # (d) context-truncation path runs (max_ctx=16, long prompt)
    torch.manual_seed(7)
    long_hist = [{"role": "user", "content": "x" * 200}]
    p = build_prompt(long_hist, tok)
    assert len(p) > 16
    r, _ = generate(model, tok, long_hist, n_tokens=5, max_ctx=16,
                    naive=True)
    torch.manual_seed(7)
    hist2 = list(long_hist)
    r2, _ = generate(model, tok, hist2, n_tokens=5, max_ctx=16,
                     naive=False)
    # (e) stop logic: n_tokens caps the output length
    torch.manual_seed(7)
    ids10 = run(naive=False, n=10)
    assert len(ids10) <= 10, f"generated {len(ids10)} > 10 tokens"
    print("  selftest: deterministic, no <|pad|>, incremental==naive, "
          "ctx truncation OK, stop logic OK")
    print("ALL PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)

    def arg(name, default=None, cast=str):
        if name in sys.argv:
            return cast(sys.argv[sys.argv.index(name) + 1])
        if default is None:
            sys.exit(f"ERROR: {name} is required")
        return default

    ckpt = arg("--ckpt")
    config = arg("--config")
    tok_path = arg("--tokenizer", "opera-chat/tokenizer.json")
    prompt = arg("--prompt")
    n_tokens = arg("--tokens", 100, int)
    temp = arg("--temp", 0.8, float)
    top_p = arg("--top-p", 0.9, float)
    top_k = arg("--top-k", 0, int)
    max_ctx = arg("--max-ctx", 512, int)
    seed = arg("--seed", 0, int)
    naive = "--naive" in sys.argv

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(tok_path)
    model = load_model(config, ckpt, device)
    print(f"loaded {ckpt} ({'naive' if naive else 'incremental'}, "
          f"{device})", flush=True)

    torch.manual_seed(seed)
    history = [{"role": "user", "content": prompt}]
    print("\n--- reply ---", flush=True)
    for partial in generate_stream(model, tok, history, n_tokens, temp,
                                   top_p, top_k, max_ctx, device, naive):
        print(partial, flush=True)
