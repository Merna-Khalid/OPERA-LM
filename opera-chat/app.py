"""app.py -- Gradio chat app for the Hugging Face Space (runs locally too).

Model artifacts come from, in order:
  1. HF_REPO  -> huggingface_hub.snapshot_download(HF_REPO)
  2. MODEL_DIR -> a local directory
  3. ./deploy_artifacts (default)

The directory must contain tokenizer.json, model_config.json, and the
checkpoint named in the config. Generation runs on CPU.
"""
import os

import gradio as gr
import torch

MODEL_DIR = os.environ.get("MODEL_DIR")
HF_REPO = os.environ.get("HF_REPO", "")  # HF_REPO_DEFAULT

torch.set_num_threads(os.cpu_count() or 1)

from generate_chat import load_model, generate_stream   # noqa: E402
from chat_common import load_tokenizer                  # noqa: E402


def resolve_model_dir():
    if HF_REPO:
        from huggingface_hub import snapshot_download
        return snapshot_download(HF_REPO)
    return MODEL_DIR or "./deploy_artifacts"


mdir = resolve_model_dir()
print(f"loading model from {mdir}", flush=True)
tok = load_tokenizer(os.path.join(mdir, "tokenizer.json"))
with open(os.path.join(mdir, "model_config.json")) as f:
    import json
    cfg = json.load(f)
ckpt = cfg["ckpt"]
if not os.path.isabs(ckpt) or not os.path.exists(ckpt):
    ckpt = os.path.join(mdir, os.path.basename(ckpt))
model = load_model(os.path.join(mdir, "model_config.json"), ckpt, "cpu")
print("model ready", flush=True)


def respond(message, history, temperature, top_p, max_new):
    """Streaming reply: yields the partial text as tokens arrive."""
    hist = [{"role": m["role"], "content": m["content"]} for m in history
            if m["role"] in ("user", "assistant")]
    hist.append({"role": "user", "content": message})
    reply = ""
    for reply in generate_stream(model, tok, hist, n_tokens=int(max_new),
                                 temp=float(temperature),
                                 top_p=float(top_p), max_ctx=512,
                                 device="cpu"):
        yield reply
    if not reply:
        yield "(empty reply)"


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.8, label="temperature"),
        gr.Slider(0.0, 1.0, value=0.9, label="top-p"),
        gr.Slider(16, 150, value=100, step=1, label="max new tokens"),
    ],
    # gradio >= 6: ChatInterface is messages-native (the `type` kwarg was
    # removed); history arrives as {"role","content"} dicts.
    title="OPERA-LM Chat",
    description=(
        "Research prototype: a ~150M-parameter geometric language model "
        "(spinor Fenwick tree, NO positional encodings -- position is "
        "structure), trained from scratch on the full smoltalk corpus. "
        "Expect toy-quality replies and slow-ish CPU generation. "
        "Model: https://github.com/Merna-Khalid/OPERA-LM"),
)

if __name__ == "__main__":
    demo.launch()
