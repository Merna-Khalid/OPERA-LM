"""kagri_losses.py -- masked next-token loss for Kaggriculture behavior
cloning (Phase 4 of the plan). Same F.cross_entropy pattern as
opera_lm.losses.train_lm_loss, but masked by the explicit action-segment
boolean mask from prepare_kagri_data.py instead of lengths-only:
observation tokens are exogenous JSON, not model-generated content, and
must not be predicted-and-scored -- only positions where the TARGET
token belongs to the action segment count toward the loss.
"""
import torch
import torch.nn.functional as F


def masked_lm_loss(logits, token_ids, lengths, action_mask):
    """logits: [B,T,V] (the final-layer logits, e.g. model(...).logits[-1]).
    token_ids/action_mask: [B,T] (action_mask marks action-segment
    tokens). lengths: [B] (padding). Returns (loss, count) -- count is
    the number of scored (action-segment, non-pad) positions, useful
    for aggregating across batches without re-weighting by batch size."""
    B, T = token_ids.shape
    device = token_ids.device
    targets = token_ids[:, 1:]
    lg = logits[:, :-1, :]
    pos = torch.arange(T - 1, device=device)
    valid_len = (pos[None, :] + 1) < lengths[:, None]
    valid = valid_len & action_mask[:, 1:]
    loss_per = F.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), targets.reshape(-1),
        reduction='none').reshape(B, T - 1)
    count = valid.sum()
    loss = (loss_per * valid.float()).sum() / count.clamp(min=1).float()
    return loss, count


@torch.no_grad()
def action_exact_match(logits, token_ids, lengths, action_mask):
    """Fraction of action-segment target positions where argmax(logits)
    matches exactly -- a stricter, more interpretable behavior-cloning
    signal than perplexity (directly answers "does it predict the
    teacher's actual token here", not just "how surprised is it")."""
    B, T = token_ids.shape
    device = token_ids.device
    targets = token_ids[:, 1:]
    lg = logits[:, :-1, :]
    pos = torch.arange(T - 1, device=device)
    valid_len = (pos[None, :] + 1) < lengths[:, None]
    valid = valid_len & action_mask[:, 1:]
    pred = lg.argmax(dim=-1)
    correct = (pred == targets) & valid
    count = valid.sum().clamp(min=1)
    return correct.sum().float() / count.float()
