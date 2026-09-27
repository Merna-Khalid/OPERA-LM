"""Loss functions"""
import torch
import torch.nn.functional as F

# ============================================================================
# LOSS (msup_loss is additive and vectorized)
# ============================================================================

def train_lm_loss(model, per_layer_prefix, token_ids, lengths,
                  aux_weight=0.5, aux_frac=1.0, final_logits=None,
                  target_weights=None):
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
            final_sum = masked                  # unweighted (logging / PPL)
            if target_weights is not None:
                # far-repeat weighting: reweighted final-layer term with
                # the SAME total weight as unweighted (weights normalized
                # to mean 1 over valid targets) -- redistributes gradient,
                # does not rescale the loss.
                w = target_weights * valid.float()
                w = w * (valid.float().sum() / w.sum().clamp_min(1e-6))
                masked = (loss_per * w).sum()
            total = total + masked
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


def lm_loss(all_logits, token_ids, lengths, aux_weight=0.5,
            target_weights=None):
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
        if li == L - 1:
            final_sum = masked
            if target_weights is not None:
                tw = target_weights * valid.float()
                tw = tw * (valid.float().sum() / tw.sum().clamp_min(1e-6))
                masked = (loss_per * tw).sum()
        total = total + w * masked
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
    for layer_idx, levels in enumerate(per_layer_levels):
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
            # tree width -> head width (identity unless state_mult > 1)
            states = model.node_readout(states, layer_idx)
            logits = model.apply_head(states)                    # [B, m', V]
            tgt = targets[:, tgt_pos]                            # [B, m']

            loss_per = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                reduction='none').reshape(B, -1)
            total = total + (loss_per * valid.float()).sum() * scale
    if isinstance(count, int):                                   # no levels contributed
        return torch.zeros((), device=device)
    return total / count.clamp(min=1).float()


def future_bag_loss(model, h, token_ids, lengths, n_pos=128):
    """Future-content auxiliary loss (model.future_bag = (W, K)).

    From the final layer's head input h [B, T, d] at a random subset of
    positions t, predict the distribution of hashed byte-trigrams in the
    next W tokens, (t, t+W]: soft-label cross-entropy against the
    normalized trigram counts. Only positions with t + W < length are used.
    Targets are future tokens, exactly like next-token targets; the model's
    inputs stay causal. Returns a scalar (0 if no valid position)."""
    W, K = model.future_bag
    B, T = token_ids.shape
    if T <= W + 1:
        return h.new_zeros(())
    x = token_ids
    x1 = F.pad(x, (1, 0))[:, :T]
    x2 = F.pad(x, (2, 0))[:, :T]
    hsh = (x * 1000003 + x1 * 10007 + x2 * 101) % K            # [B, T]
    C = torch.zeros(B, T + 1, K, device=x.device, dtype=torch.float32)
    C[:, 1:].scatter_(2, hsh.unsqueeze(-1), 1.0)
    C = C.cumsum(1)                                           # C[:, i] = counts in [0, i)
    g = torch.Generator(device='cpu').manual_seed(int(T))
    t = torch.randint(0, T - W, (B, n_pos), generator=g).to(x.device)
    valid = (t + W) < lengths[:, None]
    if not bool(valid.any()):
        return h.new_zeros(())
    bi = torch.arange(B, device=x.device)[:, None].expand(B, n_pos)
    tgt = C[bi, t + W + 1] - C[bi, t + 1]                    # trigrams ending in (t, t+W]
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1.0)
    logits = model.future_head(h[bi, t].float())
    ce = -(tgt * F.log_softmax(logits, dim=-1)).sum(-1)      # [B, n_pos]
    return (ce * valid.float()).sum() / valid.float().sum()


_FAR_HASH_P = (0x9E3779B97F4A7C15 - (1 << 64), 0x6C8E9CF570932BD5,
               0x2545F4914F6CDD1D, 0x5851F42D4C957F2D,
               0x14057B7EF767814F, 0x3C6EF372FE94F82B,
               0x1B873593CC9E2D51, 0x27D4EB2F165667C5)


def far_repeat_mask(token_ids, lengths, n=8, D=100):
    """FAR-REPEAT targets for loss reweighting (gradient starvation of the
    rare positions where only far context helps). Target position i (the
    token predicted from prefix t = i-1) is marked when the n-gram ENDING
    at i, x[i-n+1 .. i], occurred earlier in the sequence and its MOST
    RECENT earlier occurrence ended more than D tokens back -- i.e. the
    byte completes a repeat of content found only beyond D. Measured on
    held-out byte docs: 11% of 8-byte continuations recur only >100 bytes
    back. Uses target tokens only (like the loss). Returns bool [B, T-1]
    aligned with lm-loss targets (index t <-> token t+1). Computed on CPU
    (stable sort), returned on the input's device."""
    dev = token_ids.device
    x = token_ids.detach().cpu().long()
    lens = lengths.detach().cpu()
    B, T = x.shape
    h = torch.zeros_like(x)
    for k in range(n):
        h = h + F.pad(x, (k, 0))[:, :T] * _FAR_HASH_P[k % len(_FAR_HASH_P)]
    hs, order = torch.sort(h, dim=1, stable=True)
    prev_sorted = torch.full_like(order, -1)
    same = hs[:, 1:] == hs[:, :-1]
    prev_sorted[:, 1:] = torch.where(same, order[:, :-1],
                                     torch.full_like(order[:, 1:], -1))
    prev = torch.empty_like(prev_sorted).scatter_(1, order, prev_sorted)
    pos = torch.arange(T)[None, :]
    far = ((prev >= 0) & (pos - prev > D) & (pos >= n - 1)
           & (pos < lens[:, None]))
    return far[:, 1:].to(dev)
