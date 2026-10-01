"""OPERA-LM chat demo (Hugging Face Space, CPU).

Loads the exported model (experiments/export_hf.py) from the model repo in
the MODEL_REPO variable and streams replies byte by byte with the
Fenwick-incremental decoder (opera_lm.chat).
"""
import os

import gradio as gr
import torch

from opera_lm.chat import format_prompt, load_model, stream_generate

MODEL_REPO = os.environ.get('MODEL_REPO', 'Merna-Khalid/opera-lm-chat')
torch.set_num_threads(max(1, os.cpu_count() or 1))
model, cfg = load_model(MODEL_REPO)
MAX_LEN = cfg.get('chat_max_len', 2048)

ABOUT = f"""
**OPERA-LM** — an attention-free, byte-level language model
({cfg['params'] / 1e6:.0f}M parameters). No self-attention and no positional
encoding: text is composed bottom-up over a binary (Fenwick) tree, and
in-context recall comes from a gated quaternion holographic memory.
It reads and writes raw UTF-8 bytes (no tokenizer).
Small research model: fluent-looking replies, not reliable facts.
[Code](https://github.com/Merna-Khalid/OPERA-LM) ·
[Model]({'https://huggingface.co/' + MODEL_REPO})
"""


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(c if isinstance(c, str) else c.get('text', '') for c in content)
    return str(content or '')


def _pairs(history):
    """Gradio 'messages' history -> [(user, assistant), ...]."""
    pairs, user = [], None
    for m in history or []:
        role, text = m.get('role'), _text(m.get('content'))
        if role == 'user':
            user = text
        elif role == 'assistant' and user is not None:
            pairs.append((user, text))
            user = None
    return pairs


def respond(message, history, max_new, temperature, top_p):
    ids = format_prompt(_pairs(history), _text(message), max_len=MAX_LEN - int(max_new))
    text = ''
    for text in stream_generate(model, ids, max_new=int(max_new),
                                temperature=float(temperature), top_p=float(top_p)):
        yield text
    if not text.strip():
        yield '…'


demo = gr.ChatInterface(
    respond,
    type='messages',
    title='OPERA-LM chat — no attention, no tokenizer',
    description=ABOUT,
    additional_inputs=[
        gr.Slider(32, 600, value=300, step=16, label='Max new bytes'),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label='Temperature'),
        gr.Slider(0.5, 1.0, value=0.9, step=0.01, label='Top-p'),
    ],
    examples=[['Hi! Who are you?'],
              ['Give me three tips for staying focused while studying.'],
              ['Explain photosynthesis in simple words.']],
)

if __name__ == '__main__':
    demo.queue().launch()
