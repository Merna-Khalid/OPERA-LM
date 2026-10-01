---
title: OPERA-LM Chat
emoji: 🌳
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Attention-free byte-level chat model
---

# OPERA-LM chat

An attention-free, byte-level language model: bottom-up composition over a
binary (Fenwick) tree instead of self-attention, no positional encoding,
and a gated quaternion holographic memory for in-context recall. Pretrained
on FineWeb-Edu bytes, chat-tuned on smol-smoltalk.

The model is loaded from the repo in the `MODEL_REPO` Space variable.
Code and research record: https://github.com/Merna-Khalid/OPERA-LM
