"""Toy OPERA training run on random tokens -- no data download needed.

Builds a small OperaSpinorFenwickTree and trains a few steps with
lm_loss + msup_loss under a length curriculum (curriculum_len). The
random batches are generated ONCE per curriculum stage and reused, so
the loss visibly decreases as the model fits them.

Run: python examples/train_toy.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opera_lm import (OperaSpinorFenwickTree, lm_loss, msup_loss,
                      curriculum_len)


def main():
    torch.manual_seed(0)
    vocab_size, batch, steps = 101, 16, 60
    cur0, every, max_len = 8, 20, 32

    model = OperaSpinorFenwickTree(vocab_size=vocab_size, d=64, nb=16,
                                   num_layers=2, pe_mode='none',
                                   fold_mode='left', rot_mode='free')
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # One fixed random batch per curriculum stage (token ids >= 4 skip the
    # <pad>/<unk>/<bos>/<eos> slots; every position is a real target).
    gen = torch.Generator().manual_seed(0)
    stage_len = curriculum_len(0, cur0, every, max_len)
    token_ids, lengths = None, None
    for step in range(steps):
        T = curriculum_len(step, cur0, every, max_len)
        if T != stage_len:
            stage_len = T
            token_ids, lengths = None, None
        if token_ids is None:
            token_ids = torch.randint(4, vocab_size, (batch, T), generator=gen)
            lengths = torch.full((batch,), T, dtype=torch.long)

        out = model(token_ids, lengths, return_levels=True)
        loss, _, _ = lm_loss(out.logits, token_ids, lengths)
        loss = loss + 0.1 * msup_loss(model, out.levels, token_ids, lengths)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:4d}  len {T:3d}  loss {loss.item():.4f}",
                  flush=True)


if __name__ == '__main__':
    main()
