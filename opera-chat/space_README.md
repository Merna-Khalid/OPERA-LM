---
title: OPERA-LM Chat
emoji: 🌀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: 3.11
pinned: false
---

# OPERA-LM Chat

A chat demo for [OPERA-LM](https://github.com/Merna-Khalid/OPERA-LM):
a language model that replaces self-attention with bottom-up composition
over a binary (Fenwick) tree of spinor states in Cl(3), and replaces
positional encodings with nothing at all — position is the shape of the
computation.

This Space serves a BPE-tokenized checkpoint trained from scratch on the
full [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
`all` split (~1.04M conversations), using the incremental `OperaDecoder`
(O(log T) per generated token).

**Honesty box.** This is a ~150M-parameter research model trained on
~82M tokens total — five orders of magnitude less data than a real small
chat model (e.g. SmolLM2-135M used ~11T tokens). Expect grammatically
fluent but often incoherent replies: the model has learned local sentence
statistics, not world knowledge or reliable reasoning. There is no safety
tuning of any kind. It exists so the mechanism (no positional encodings,
geometric tree composition) can be poked at interactively, not as a
capability claim.
