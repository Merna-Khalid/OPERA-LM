"""Loss functions"""
import torch
import torch.nn.functional as F

# ============================================================================
# LOSS (msup_loss is additive and vectorized)
# ============================================================================

def train_lm_loss(model, per_layer_prefix, token_ids, lengths,
                  aux_weight=0.5, aux_frac=1.0, final_logits=None):
    """lm_loss math with an optional aux-position subsample.

    Same normalization as lm_loss (total / (vcount * weight_total)). Aux
    layers' head+CE run on a random aux_frac subset of the T-1 positions,
    rescaled by 1/aux_frac (unbiased estimator of the full aux term).
    aux_frac=1.0 reproduces lm_loss EXACTLY. The final-layer term -- the
    one that defines reported PPL -- is always full-resolution."""
    B, T = token_ids.shape
    device = token_ids.device
    targets = token_ids[:, 1:]
    pos = torch.arange(T - 1, device=device)
    valid = (pos[None, :] + 1) < lengths[:, None]
    vcount = valid.sum()
    L = len(per_layer_prefix)
    total = 0.0
    final_sum = None
    for li, pref in enumerate(per_layer_prefix):
        if li == L - 1:
            logits = (final_logits if final_logits is not None
                      else model.apply_head(pref))
            lg = logits[:, :-1, :]
            loss_per = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), targets.reshape(-1),
                reduction='none').reshape(B, T - 1)
            masked = (loss_per * valid.float()).sum()
            total = total + masked
            final_sum = masked
        else:
            if aux_frac < 1.0:
                n_sub = max(1, int(round((T - 1) * aux_frac)))
                sub = torch.randperm(T - 1, device=device)[:n_sub]
                lg = model.apply_head(pref[:, :-1, :][:, sub, :])
                tgt = targets[:, sub]
                v = valid[:, sub]
            else:
                lg = model.apply_head(pref[:, :-1, :])
                tgt = targets
                v = valid
            loss_per = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                reduction='none').reshape(B, -1)
            masked = (loss_per * v.float()).sum() / aux_frac
            total = total + aux_weight * masked
    denom = vcount.clamp(min=1).float()
    weight_total = (1.0 + aux_weight * (L - 1))
    return total / (denom * weight_total), final_sum, vcount


def lm_loss(all_logits, token_ids, lengths, aux_weight=0.5):
    """Identical to v7.0. This (final layer term) defines the reported PPL."""
    B, T = token_ids.shape
    device = token_ids.device
    targets = token_ids[:, 1:]
    pos = torch.arange(T - 1, device=device)
    valid = (pos[None, :] + 1) < lengths[:, None]
    vcount = valid.sum()

    total = 0.0
    final_sum = None
    L = len(all_logits)
    for li, logits in enumerate(all_logits):
        lg = logits[:, :-1, :]
        loss_per = F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), targets.reshape(-1), reduction='none'
        ).reshape(B, T - 1)
        masked = (loss_per * valid.float()).sum()
        w = 1.0 if li == L - 1 else aux_weight
        total = total + w * masked
        if li == L - 1:
            final_sum = masked
    denom = vcount.clamp(min=1).float()
    weight_total = (1.0 + aux_weight * (L - 1))
    return total / (denom * weight_total), final_sum, vcount


def msup_loss(model, per_layer_levels, token_ids, lengths, aux_frac=1.0):
    """Multi-scale supervision, vectorized (one tensor op per tree level).

    An internal node at level l with index i summarizes span
    [i*2^l, (i+1)*2^l); it predicts the first token AFTER its span,
    i.e. token at position (i+1)*2^l, which is targets[:, (i+1)*2^l - 1].
    Nodes whose target falls outside the sentence are masked out.
    Uses the same (possibly tied) head; adds no parameters.
    Returns a mean loss over all valid (node, batch) pairs.

    aux_frac (v9 OOM mitigation): mirrors train_lm_loss's aux-position
    subsampling -- supervise only a random aux_frac fraction of each
    level's nodes, rescaled by 1/aux_frac (unbiased estimator of the
    full sum; the normalizing `count` below is still the TRUE full-
    resolution count, computed before subsampling). aux_frac=1.0
    reproduces the original, full-resolution loss exactly. Every level
    does a full-vocab head projection per node ([B, m, V]), none of it
    checkpointed, so at long T this dominates activation memory even
    after the compose node itself is checkpointed (--checkpoint level):
    a curriculum run that stopped OOMing in compose_pair_batch at
    T_cur=128 went on to OOM here instead, one call later in the same
    step, with grad_checkpoint='level' set but aux_frac left unwired on
    this call site -- this parameter is that fix."""
    B, T = token_ids.shape
    device = token_ids.device
    targets = token_ids[:, 1:]                                   # [B, T-1]

    total = 0.0
    count = 0
    for levels in per_layer_levels:
        for level_idx in range(1, len(levels)):
            span = 1 << level_idx
            if span >= T:
                break
            n_nodes = levels[level_idx].shape[1]
            node_ids = torch.arange(n_nodes, device=device)
            tgt_pos = (node_ids + 1) * span - 1                  # index into targets
            keep = tgt_pos < (T - 1)
            if not bool(keep.any()):
                continue
            node_ids = node_ids[keep]
            tgt_pos = tgt_pos[keep]
            m = node_ids.shape[0]

            # valid if the predicted position is inside the sentence --
            # computed at full resolution BEFORE subsampling, since count
            # (the loss normalizer) must reflect the true node population.
            valid = (tgt_pos[None, :] + 1) < lengths[:, None]    # [B, m]
            count = count + valid.sum()

            if aux_frac < 1.0:
                m_sub = max(1, int(round(m * aux_frac)))
                sub = torch.randperm(m, device=device)[:m_sub]
                node_ids = node_ids[sub]
                tgt_pos = tgt_pos[sub]
                valid = valid[:, sub]
                scale = 1.0 / aux_frac
            else:
                scale = 1.0

            states = levels[level_idx][:, node_ids, :]           # [B, m', d]
            logits = model.apply_head(states)                    # [B, m', V]
            tgt = targets[:, tgt_pos]                            # [B, m']

            loss_per = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                reduction='none').reshape(B, -1)
            total = total + (loss_per * valid.float()).sum() * scale
    if isinstance(count, int):                                   # no levels contributed
        return torch.zeros((), device=device)
    return total / count.clamp(min=1).float()
