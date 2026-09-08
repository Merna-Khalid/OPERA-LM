"""Training loop, batching, LR schedule, eval and probe utilities."""
import os
import json
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn

from .model import OperaSpinorFenwickTree, count_params, fold_work_counts
from .losses import lm_loss, train_lm_loss, msup_loss
from .data import load_data
from .packed import packed_stats

def _eval_batch(base, max_len, ref=256):
    """Scale an eval batch size down for max_len > ref (same principle as
    extrapolation_eval's per-bucket scaling below): activation memory per
    sequence grows ~linearly with T, so a batch tuned at ref tokens OOMs
    at max_len >> ref (e.g. a state-passing fine-tune with max_len=1024).
    A no-op when max_len <= ref, which covers every ordinary run."""
    return max(1, base * ref // max_len)


def _oom_backstop(fn, batch_size, device):
    """Call fn(batch_size), halving batch_size and retrying on GPU OOM.
    _eval_batch above only scales for max_len; it has no way to know the
    model's own size (d, num_layers), so a batch tuned for one config
    (e.g. the README's 20M-param reference) can still OOM on a much
    larger one. This is the same halving backstop extrapolation_eval
    already uses per-bucket, generalized to every other eval/diagnostic
    call site."""
    oom_types = (getattr(torch, 'OutOfMemoryError', RuntimeError), RuntimeError)
    eff = max(1, batch_size)
    while True:
        try:
            if device == 'cuda':
                torch.cuda.empty_cache()
            return fn(eff)
        except oom_types as e:
            if 'out of memory' not in str(e).lower():
                raise
            if device == 'cuda':
                torch.cuda.empty_cache()
            if eff == 1:
                raise
            eff = max(1, eff // 2)
            print(f"    (OOM; retrying at batch {eff})", flush=True)


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=1e-5,
           schedule='cosine', wsd_decay_frac=0.2):
    """LR at `step`. schedule='cosine' (default) is bitwise the incumbent
    formula. schedule='wsd' is warmup-STABLE-decay (MiniCPM/DeepSeek-v2):
    flat max_lr after warmup, then LINEAR decay to min_lr over the final
    wsd_decay_frac of total_steps. The stable phase is decoupled from any
    notion of "how far along we are", which is the point: a multi-session
    run (Kaggle quota) can extend total_steps on resume and simply stays
    in the stable phase longer -- no re-planning the whole curve -- while
    the final decay window still lands exactly where it should."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if schedule == 'cosine':
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    assert schedule == 'wsd', f"unknown lr schedule {schedule!r}"
    decay_start = max(warmup_steps,
                      int(total_steps * (1.0 - wsd_decay_frac)))
    if step < decay_start:
        return max_lr
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return max_lr + (min_lr - max_lr) * progress


def curriculum_len(step, cur0, every, max_len):
    """Length curriculum (v9, arm C): train at cur0 tokens, doubling every
    `every` steps, capped at max_len. Deterministic in `step` only, so
    --resume lands on the exact stage. Power-of-two stages keep every
    shape static within a stage (one compile per stage; the Fenwick
    index cache is keyed by T)."""
    return min(cur0 << (step // every), max_len)


def make_batch_full(sentences, max_len):
    B = len(sentences)
    token_ids = np.zeros((B, max_len), dtype=np.int64)
    lengths = np.zeros(B, dtype=np.int64)
    for b, s in enumerate(sentences):
        n = min(len(s), max_len)
        token_ids[b, :n] = s[:n]
        lengths[b] = n
    return torch.tensor(token_ids), torch.tensor(lengths)


class GpuBatchSource:
    """OPT: GPU-resident training set. One padded [N, max_len] tensor +
    lengths live on-device; a batch is one randint + one gather -- no
    per-step python sampling or host->device copy. A dedicated
    torch.Generator drives sampling; its state is checkpointed so
    --resume continues the EXACT batch stream."""

    def __init__(self, train_data, max_len, device, seed):
        N = len(train_data)
        ids = np.zeros((N, max_len), dtype=np.int64)
        lens = np.zeros(N, dtype=np.int64)
        for i, s in enumerate(train_data):
            n = min(len(s), max_len)
            ids[i, :n] = s[:n]
            lens[i] = n
        self.ids = torch.from_numpy(ids).to(device)
        self.lens = torch.from_numpy(lens).to(device)
        self.N = N
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def sample(self, batch):
        idx = torch.randint(self.N, (batch,), generator=self.gen,
                            device=self.ids.device)
        return self.ids[idx], self.lens[idx]

    def state_dict(self):
        return self.gen.get_state().cpu()

    def load_state_dict(self, st):
        # torch.load(map_location=device) moves EVERY checkpoint tensor to
        # the GPU -- including this saved RNG state -- but a CUDA
        # generator's set_state() strictly requires a CPU ByteTensor
        # (TypeError: RNG state must be a torch.ByteTensor). Coerce back.
        # (Merna's fix, regressed in opt4b-opt7 during the v8.3 port;
        # restored in opt7b with a selftest so it cannot regress again.)
        if isinstance(st, torch.Tensor):
            st = st.detach().to('cpu', torch.uint8)
        self.gen.set_state(st)


@torch.no_grad()
def compute_perplexity(model, test_data, max_len, batch_size, device):
    """Identical to v7.0: final-layer per-token PPL. Comparable across runs."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        if not batch:
            continue
        # OPT: pad to a FIXED max_len (identical math -- padded positions
        # never influence valid prefix states and are masked in the loss)
        # so a compiled model sees ONE input shape for every batch.
        token_ids, lengths = make_batch_full(batch, max_len)
        token_ids, lengths = token_ids.to(device), lengths.to(device)
        all_logits = model(token_ids, lengths, head_last_only=True).logits
        _, final_sum, vcount = lm_loss(all_logits, token_ids, lengths)
        total_loss += final_sum.item()
        total_tokens += vcount.item()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


@torch.no_grad()
def extrapolation_eval(model, test_long, train_max_len, eval_max_len, batch_size, device):
    model.eval()
    buckets = []
    lo = train_max_len + 1
    while lo <= eval_max_len:
        hi = min(lo + train_max_len - 1, eval_max_len)
        buckets.append((lo, hi))
        lo = hi + 1
    results = {}
    _oom = (getattr(torch, 'OutOfMemoryError', RuntimeError), RuntimeError)
    for (lo, hi) in buckets:
        sents = [s for s in test_long if lo <= len(s) <= hi]
        if len(sents) < 10:
            results[f"{lo}-{hi}"] = (None, len(sents))
            continue
        # opt7: activation memory per sequence scales ~linearly with T
        # (the fold's gathered tensor is B x T x S x d), so a batch tuned
        # at train_max_len OOMs at 4x that length. Scale batch inversely
        # with the bucket ceiling and halve on OOM as a backstop.
        eff = max(1, (batch_size * train_max_len) // hi)
        while True:
            try:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                ppl = compute_perplexity(model, sents, hi, eff, device)
                break
            except _oom as e:
                if 'out of memory' not in str(e).lower():
                    raise
                if device == 'cuda':
                    torch.cuda.empty_cache()
                if eff == 1:
                    raise
                eff = max(1, eff // 2)
                print(f"    (OOM in bucket {lo}-{hi}; retrying at "
                      f"batch {eff})", flush=True)
        results[f"{lo}-{hi}"] = (ppl, len(sents))
    model.train()
    return results


# ============================================================================
# RMT DIAGNOSTIC (fixed: spectrum of prefix-STATE covariance, CPU eig)
# ============================================================================

@torch.no_grad()
def rmt_states_diagnostic(model, test_data, batch_size, device, num_sentences=500):
    """Marchenko-Pastur analysis of final-layer prefix states.

    Collect n token-level prefix states h in R^d, form the covariance of
    the standardized states, and compare its spectrum to the MP law with
    gamma = d/n. Eigenvalues above the MP edge indicate learned collective
    structure (semantic/syntactic directions); a pure-MP spectrum would
    mean the representation is statistically indistinguishable from noise.

    Fixes vs v7.2: (1) operates on states (full-rank object), not on nb
    rotation matrices flattened to R^9 whose Gram has rank <= 9;
    (2) eigendecomposition on CPU (aten::_linalg_eigh has no MPS kernel).
    """
    model.eval()
    print("\n=== RMT Diagnostic (prefix-state spectrum) ===", flush=True)

    feats = []
    n_collected = 0
    for i in range(0, min(num_sentences, len(test_data)), batch_size):
        batch = test_data[i:i + batch_size]
        if not batch:
            continue
        bl = max(len(s) for s in batch)
        token_ids, lengths = make_batch_full(batch, bl)
        token_ids, lengths = token_ids.to(device), lengths.to(device)
        out = model(token_ids, lengths, return_states=True)
        h = out.states[-1]                                        # [B, T, d]
        for b in range(h.shape[0]):
            L = int(lengths[b].item())
            feats.append(h[b, :L, :].cpu())
            n_collected += L
    H = torch.cat(feats, dim=0).float()                           # [n, d] on CPU
    n, d = H.shape
    print(f"  collected {n} token states, d={d}", flush=True)

    # Standardize per dimension, covariance, spectrum (all on CPU).
    H = (H - H.mean(0, keepdim=True)) / (H.std(0, keepdim=True) + 1e-8)
    C = (H.t() @ H) / n                                           # [d, d]
    ev = torch.linalg.eigvalsh(C)                                 # ascending

    gamma = d / n
    mp_plus = (1 + math.sqrt(gamma)) ** 2
    mp_minus = max(0.0, (1 - math.sqrt(gamma)) ** 2)
    outliers = int((ev > mp_plus * 1.05).sum().item())
    bulk = ev[(ev >= mp_minus) & (ev <= mp_plus)]
    # participation ratio of the spectrum: effective number of directions
    pr = (ev.sum() ** 2 / (ev ** 2).sum()).item()

    print(f"  MP edge [{mp_minus:.3f}, {mp_plus:.3f}] (gamma={gamma:.4f})", flush=True)
    print(f"  eigenvalues above MP edge (+5%): {outliers} / {d}", flush=True)
    print(f"  top 5 eigenvalues: {[round(float(x), 2) for x in ev[-5:]]}", flush=True)
    print(f"  fraction of spectrum inside MP bulk: {bulk.numel() / d:.3f}", flush=True)
    print(f"  participation ratio (effective dims): {pr:.1f} / {d}", flush=True)
    model.train()
    return {
        'n_states': n, 'd': d, 'gamma': gamma,
        'mp_plus': mp_plus, 'mp_minus': mp_minus,
        'outliers': outliers,
        'top5': [float(x) for x in ev[-5:]],
        'bulk_fraction': bulk.numel() / d,
        'participation_ratio': pr,
    }


@torch.no_grad()
def energy_diagnostic(model, test_data, batch_size, device):
    """Versor-inspired diagnostic (arXiv:2602.10195): strict-SO(3) rotor
    composition preserves paravector norm exactly; OPERA's rot_mode='free'
    deliberately relaxes R_O's orthogonality, trading that guarantee away
    for capacity. This tracks the composed node's energy (mean per-block
    L2 norm) by tree depth BEFORE comp_norm/rms/blockrms rescales it --
    the only place a non-orthogonal R_O's drift is actually visible,
    since every norm_mode re-normalizes each node before it becomes the
    next level's input, erasing the signal from anything read post-norm
    (see compose_pair_batch's `_need_energy` gate and OperaOutput.energy).
    A roughly flat curve across depth means the raw composition is
    stable; one that collapses toward the norm's 1e-6 epsilon floor or
    grows by an order of magnitude across a handful of levels means
    'free' rotations are destabilizing the node even though the post-norm
    output looks fine -- exactly the failure mode this exists to catch
    before it shows up as a harder-to-diagnose loss regression."""
    if getattr(model, 'fold_mode', None) == 'scan':
        return None  # scan builds no tree -- nothing to measure
    model.eval()
    batch = test_data[:max(1, batch_size)]
    bl = max(len(s) for s in batch)
    token_ids, lengths = make_batch_full(batch, bl)
    token_ids, lengths = token_ids.to(device), lengths.to(device)
    out = model(token_ids, lengths, return_energy=True)
    print("\n=== Energy Diagnostic (pre-norm node energy by tree depth) ===",
          flush=True)
    summary = []
    for li, energies in enumerate(out.energy):
        curve = [round(e.mean().item(), 3) for e in energies]
        summary.append(curve)
        print(f"  layer {li} (levels 1..{len(curve)}): {curve}", flush=True)
    model.train()
    return summary


# ============================================================================
# TREE INSPECTION (identical to v7.0)
# ============================================================================

def tree_to_str(levels, locks, words, level_idx, node_idx):
    span_start = node_idx * (1 << level_idx)
    if span_start >= len(words):
        return ""
    if level_idx == 0:
        return words[span_start]
    left = tree_to_str(levels, locks, words, level_idx - 1, 2 * node_idx)
    right = tree_to_str(levels, locks, words, level_idx - 1, 2 * node_idx + 1)
    if not right:
        return left
    lock_val = locks[level_idx - 1][0, node_idx].item()
    return f"[{left} {right}](l={lock_val:.2f})"


@torch.no_grad()
def inspect_tree(model, sentences, idx2word, device, n=5):
    model.eval()
    if getattr(model, 'fold_mode', None) == 'scan':
        # Scan mode builds no tree; keep the prediction sample only.
        print("\n=== Scan mode: no tree to inspect (prediction sample) ===", flush=True)
        for sent in sentences[:n]:
            if len(sent) < 3:
                continue
            words = [idx2word.get(w, '<unk>') for w in sent]
            T = len(sent)
            token_ids, lengths = make_batch_full([sent], T)
            token_ids, lengths = token_ids.to(device), lengths.to(device)
            all_logits = model(token_ids, lengths).logits
            pred_idx = all_logits[-1][0, T - 2].argmax().item()
            pred_word = idx2word.get(pred_idx, '<unk>')
            print(f"  Sentence: {' '.join(words)}", flush=True)
            print(f"  Predict last word: {pred_word} (actual: {words[-1]})", flush=True)
            print()
        return
    print("\n=== Spinor Tree (sample, layer 0; lock shown is diagnostic) ===", flush=True)
    for sent in sentences[:n]:
        if len(sent) < 3:
            continue
        words = [idx2word.get(w, '<unk>') for w in sent]
        T = len(sent)
        token_ids, lengths = make_batch_full([sent], T)
        token_ids, lengths = token_ids.to(device), lengths.to(device)
        out = model(token_ids, lengths, return_tree=True)
        all_logits = out.logits
        levels, locks = out.tree[0]
        top = len(levels) - 1
        tree_str = tree_to_str(levels, locks, words, top, 0)
        pred_idx = all_logits[-1][0, T - 2].argmax().item()
        pred_word = idx2word.get(pred_idx, '<unk>')
        print(f"  Sentence: {' '.join(words)}", flush=True)
        print(f"  Tree:     {tree_str}", flush=True)
        print(f"  Predict last word: {pred_word} (actual: {words[-1]})", flush=True)
        print()
    model.train()



# ============================================================================
# TRAINING
# ============================================================================

def train(steps, batch, max_len, vocab_size, d, nb, num_layers, eval_max_len,
          device='cpu', lock_mode='none', tie=False, dropout=0.0, msup=False,
          msup_weight=0.1, pe_mode='sin', fold_mode='left',
          fold_rotors='shared', fold_scale=False,
          norm_mode='layer', act_mode='tanh', node_residual=False,
          tree_drop=0.0, grad_checkpoint='', use_metal=False, use_triton=False,
          use_amp=True, use_foreach=True, rot_mode='so3', seed=42,
          data_mode='sentences', docs_limit=100000, out_dir='.',
          save_every=0, resume=False,
          compile_mode='default', gpu_data=True, aux_frac=0.25,
           max_lr=1e-3, warmup_steps=500, lr_schedule='cosine',
           wsd_decay_frac=0.2, accum=1,
           oam_k=4, oam_charges='auto', oam_phi=0.7853981633974483,
          oam_shared_gate=False, oam_combine='compose', oam_pair='seq',
          rack_exitnorm=False, oam_transport='rack', oam_levelgate=False,
          oam_chan_emb=False, scan_salience=False, scan_decay_bias=-3.0,
          workspace=False, fold_gate_bias=None, curriculum=None,
          data=None, idx2word=None, optimizer='adamw', muon_lr=0.02,
          readout_mode='none', readout_max_slots=16,
          mem_mode='none', mem_dim=128, init_weights_from=None,
          homeo_mode='off', node_paths=3, fold_adapt='off',
          packed_data=None, ddp=False):
    # DDP (multi-GPU data parallelism, added for the Kaggle 2xT4 tier --
    # a single T4 measured ~8.5x slower than the project's A100, so real
    # multi-GPU throughput matters there in a way it didn't on Colab).
    # Launch via `torchrun --nproc_per_node=N script.py --ddp ...`;
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK, read here rather than
    # threaded through every caller. Scope of this first version:
    # DDP wraps ONLY the training forward/backward/step; all logging,
    # checkpointing, and eval run on rank 0 only, against the raw
    # (unwrapped) model -- eval never calls .backward() so it needs no
    # DDP wrapping or cross-rank sync at all. torch.compile is force-
    # disabled under ddp for this version: it has already caused two
    # hard-to-predict bugs in this codebase this session (the id()-cache
    # dynamo-guard issue, the P100 kernel-availability crash), and
    # DDP+compile interaction is a third, separately-tricky axis that
    # is not validated here -- a documented follow-up, not silently
    # broken. `device` stays the TYPE string ('cuda') used throughout
    # this function for branching; `torch_device` below is the actual
    # per-rank placement string ('cuda:LOCAL_RANK').
    rank, world_size, local_rank = 0, 1, 0
    if ddp:
        import torch.distributed as dist
        # nccl is the real, intended (and only performant) backend --
        # multi-GPU training is the whole point. gloo (CPU) is allowed
        # too, ONLY so this plumbing is unit-testable without a multi-GPU
        # machine (see selftest.py); it is not a supported real-training
        # configuration and gets no speed benefit.
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if device == 'cuda' else 'gloo')
        if device == 'cuda':
            torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    torch_device = (f'cuda:{local_rank}' if (ddp and device == 'cuda') else device)

    tag = [f'pe-{pe_mode}']
    assert not (msup and fold_mode == 'scan'), \
        "--msup reads tree levels; --fold scan builds no tree"
    if curriculum is not None:
        assert 8 <= curriculum[0] <= max_len and curriculum[1] >= 1, \
            "--curriculum T0:EVERY needs 8 <= T0 <= max_len, EVERY >= 1"
    assert accum >= 1, "--accum must be >= 1"
    if save_every and accum > 1:
        # Checkpoint/step-boundary alignment: optimizer updates land on
        # steps with (step+1) % accum == 0; the save branch below fires
        # on step % save_every == accum-1, which is always a boundary
        # step when save_every is a multiple of accum -- so a train ckpt
        # NEVER stores in-flight gradients and --resume stays exact.
        assert save_every % accum == 0, \
            f"save_every ({save_every}) must be a multiple of accum ({accum})"
    if data_mode != 'sentences': tag.append(f'data-{data_mode}')
    if data_mode == 'docs-en' and docs_limit != 100000:
        tag.append(f'lim{docs_limit}')
    if rot_mode != 'so3': tag.append('rotfree')
    if seed != 42: tag.append(f'seed{seed}')
    if norm_mode != 'layer': tag.append(norm_mode)
    if act_mode != 'tanh': tag.append('linact')
    if node_residual: tag.append('noderes')
    if tree_drop > 0: tag.append(f'tdrop{tree_drop}')
    if grad_checkpoint: tag.append(f'ckpt-{grad_checkpoint}')
    if use_metal: tag.append('metal')
    if use_triton: tag.append('triton')
    if use_amp: tag.append('amp')
    if use_foreach: tag.append('foreach')
    if compile_mode != 'off': tag.append(f'cmp-{compile_mode}')
    if gpu_data: tag.append('gpudata')
    if aux_frac != 1.0: tag.append(f'aux{aux_frac}')
    if max_lr != 1e-3: tag.append(f'lr{max_lr}')
    if warmup_steps != 500: tag.append(f'wu{warmup_steps}')
    if lr_schedule != 'cosine':
        tag.append(lr_schedule)
        if wsd_decay_frac != 0.2: tag.append(f'wsdf{wsd_decay_frac}')
    if accum > 1: tag.append(f'acc{accum}')
    if fold_mode != 'left': tag.append(f'fold-{fold_mode}')
    if fold_mode == 'oam':
        tag.append(f'k{oam_k}')
        if oam_charges != 'auto': tag.append('chgX')
        if float(oam_phi) == 0.0: tag.append('nocharge')
        if oam_shared_gate: tag.append('shgate')
        if oam_combine != 'compose': tag.append(f'comb-{oam_combine}')
        if oam_levelgate: tag.append('lvgate')
        if oam_chan_emb: tag.append('chanemb')
        if oam_pair != 'seq': tag.append(f'pair-{oam_pair}')
        if oam_transport != 'rack': tag.append(f'tr-{oam_transport}')
    if fold_mode == 'rack' and rack_exitnorm: tag.append('exitnorm')
    if fold_mode == 'scan' and scan_salience: tag.append('salience')
    if fold_mode == 'scan' and workspace: tag.append('ws')
    if fold_mode == 'scan' and scan_decay_bias != -3.0:
        tag.append(f'db{scan_decay_bias}')
    if fold_rotors != 'shared': tag.append('foldrot')
    if fold_scale: tag.append('foldscale')
    if fold_gate_bias is not None:
        tag.append('gb' + ','.join(str(float(x)) for x in fold_gate_bias))
    if curriculum is not None:
        tag.append(f'cur{curriculum[0]}x{curriculum[1]}')
    if tie: tag.append('tie')
    if dropout > 0: tag.append(f'drop{dropout}')
    if optimizer != 'adamw':
        tag.append(f'opt-{optimizer}')
        if optimizer == 'muon' and muon_lr != 0.02:
            tag.append(f'mlr{muon_lr}')
    if msup: tag.append('msup')
    if readout_mode != 'none': tag.append(f'ro-{readout_mode}')
    if mem_mode != 'none': tag.append(f'mem-{mem_mode}{mem_dim}')
    if homeo_mode != 'off': tag.append('homeo')
    if node_paths != 3: tag.append(f'np{node_paths}')
    if fold_adapt != 'off': tag.append('fadpt')
    if lock_mode != 'none': tag.append(lock_mode)
    tag = '+'.join(tag)

    if is_main:
        print(f"=== OPERA-LM v9.0 (Spinor Tree, tree-training line) [{tag}] ===", flush=True)
        print(f"  steps={steps}, batch={batch} (per-rank), train max_len={max_len}, eval_max_len={eval_max_len}", flush=True)
        if accum > 1 or ddp:
            print(f"  effective batch = {batch} x {world_size} ranks "
                  f"x {accum} accum = {batch * world_size * accum}", flush=True)
        if lr_schedule != 'cosine':
            print(f"  LR schedule: {lr_schedule} (decay over final "
                  f"{wsd_decay_frac:.0%} of steps)", flush=True)
        print(f"  d={d}, nb={nb}, num_layers={num_layers}, lock={lock_mode}, device={device}", flush=True)
        if ddp:
            print(f"  DDP: world_size={world_size}, effective batch="
                  f"{batch * world_size}, compile forced off", flush=True)
        print(f"  pe={pe_mode}, fold={fold_mode}, fold_rotors={fold_rotors}, "
              f"fold_scale={fold_scale}", flush=True)
        print(f"  norm={norm_mode}, act={act_mode}, node_residual={node_residual}", flush=True)
        print(f"  rot={rot_mode}, seed={seed}, data={data_mode}", flush=True)
        print(f"  out={out_dir}, save_every={save_every}, resume={resume}", flush=True)
        print(f"  tie={tie}, dropout={dropout}, msup={msup} (weight {msup_weight})", flush=True)
        print(f"  OPT: compile={compile_mode}, gpu_data={gpu_data}, "
              f"aux_frac={aux_frac}, amp={use_amp}, foreach={use_foreach}", flush=True)
        print(f"  References (train<=20, 4L): OPERA pe-none 70.92 / 1.65x / 1.68x;"
              f" RoPE transformer 70.71 / 1.60x / 1.68x", flush=True)
        mw, cw, _ = fold_work_counts(max_len)
        print(f"  Fold work @T={max_len}: {cw} row-composes/layer "
              f"(v7.7 masked: {mw}; {mw/cw:.2f}x less)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    if data is not None:
        # Injected data path (e.g. BPE chat corpora): caller supplies
        # (train_data, test_short, test_long, vocab_size) as lists of token-id
        # lists, bypassing the word-level Wikipedia load_data entirely.
        # idx2word stays None unless the caller provides one; tree inspection
        # is skipped then (it decodes ids to words).
        train_data, test_short, test_long, actual_vocab_size = data
    else:
        train_data, test_short, test_long, vocab, word2idx, idx2word = load_data(
            vocab_size, max_len, eval_max_len, data_mode=data_mode,
            docs_limit=docs_limit)
        actual_vocab_size = len(vocab)

    # Variation seed (r15): init + batch sampling only. Applied AFTER
    # load_data so the train/test split (seeded 42 internally) is
    # identical across seeds -- arms must share the exact same data.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = OperaSpinorFenwickTree(actual_vocab_size, d=d, nb=nb,
                                   num_layers=num_layers, lock_mode=lock_mode,
                                   tie=tie, dropout=dropout, pe_mode=pe_mode,
                                   fold_mode=fold_mode, fold_rotors=fold_rotors,
                                   fold_scale=fold_scale, norm_mode=norm_mode,
                                   act_mode=act_mode,
                                   node_residual=node_residual,
                                   tree_drop=tree_drop,
                                   grad_checkpoint=grad_checkpoint,
                                   use_metal=use_metal, use_triton=use_triton,
                                   rot_mode=rot_mode, oam_k=oam_k,
                                   oam_charges=oam_charges, oam_phi=oam_phi,
                                   oam_shared_gate=oam_shared_gate,
                                   oam_combine=oam_combine, oam_pair=oam_pair,
                                   rack_exitnorm=rack_exitnorm,
                                   oam_transport=oam_transport,
                                   oam_levelgate=oam_levelgate,
                                   oam_chan_emb=oam_chan_emb,
                                   scan_salience=scan_salience,
                                   scan_decay_bias=scan_decay_bias,
                                   workspace=workspace,
                                   fold_gate_bias=fold_gate_bias,
                                   readout_mode=readout_mode,
                                   readout_max_slots=readout_max_slots,
                                   mem_mode=mem_mode,
                                   mem_dim=mem_dim,
                                   homeo_mode=homeo_mode,
                                   node_paths=node_paths,
                                   fold_adapt=fold_adapt).to(torch_device)
    npar = count_params(model)
    if is_main:
        print(f"  Model params: {npar:,}", flush=True)
    if init_weights_from is not None:
        # short post-training phases (e.g. state-passing length-gen
        # fine-tunes) start from an existing checkpoint's WEIGHTS ONLY --
        # optimizer/schedule/curriculum/data for this call are independent
        # of whatever produced init_weights_from, so this is not --resume
        # (which requires an identical tag/config to find its own
        # train_ckpt and also restores optimizer+RNG state).
        sd = torch.load(init_weights_from, map_location=torch_device, weights_only=True)
        model.load_state_dict(sd)
        if is_main:
            print(f"  Initialized weights from {init_weights_from}", flush=True)
    if mem_mode == 'delta' and device == 'mps' and is_main:
        print(f"  WARNING: mem_mode='delta' on device='mps' is untested at "
              f"scale (docs/OPERA_Swarm_Notes.md, T1.4) -- the fp32 "
              f"chunked-WY scan on top of the fold measured ~4x step time "
              f"and an OOM kill in the one real run to date "
              f"(opera-chat/mem_run.log). Expect elevated memory/step "
              f"time; consider --device cuda or a smaller mem_dim/batch "
              f"until profiled.", flush=True)

    # OPT: torch.compile. The fold is compile-safe by construction (static
    # cached Fenwick indices; loop trip counts depend only on T). Eval is
    # padded to fixed T so one graph serves train and eval. Checkpoints
    # always save the RAW module (model), never the compiled wrapper.
    # v8.8: compile is now also enabled on MPS for --fold scan ONLY
    # (measured 2.12x step-time speedup, fwd err 6e-6; other folds on MPS
    # stay eager as before -- untested, out of scope). Force-disabled
    # under ddp for now -- see the docstring note at the top of train().
    model_c = model
    _compile_ok = (compile_mode != 'off' and not ddp
                   and (device == 'cuda'
                        or (device == 'mps' and fold_mode == 'scan')))
    if _compile_ok:
        # _layer_body specializes per layer_idx (num_layers) x per calling
        # context (train+autocast+grad vs eval+no_grad, plus kwarg variants),
        # which exhausts the DEFAULT recompile limit of 8 -- dynamo then
        # SILENTLY falls back to eager for the rest of the run (observed:
        # "hit config.recompile_limit (8)" 6 min in, zero speedup). Raise it.
        import torch._dynamo as _dynamo
        _dynamo.config.recompile_limit = 64
        kw = {'dynamic': False}
        if compile_mode != 'default':
            kw['mode'] = compile_mode
        model_c = torch.compile(model, **kw)
        if is_main:
            print(f"  torch.compile enabled (mode={compile_mode}, "
                  f"recompile_limit=64)", flush=True)
    elif compile_mode != 'off' and is_main:
        why = "ddp" if ddp else device
        print(f"  torch.compile skipped ({why}); running eager", flush=True)

    # OPT: GPU-resident training data (exact batch-stream resume via a
    # dedicated generator whose state rides in the checkpoint). Under ddp
    # each rank gets a DIFFERENT stream (seed + rank) -- otherwise every
    # rank would compute gradients on the identical batch and DDP's
    # all-reduce would just average N copies of the same gradient, zero
    # real parallelism benefit for 2x the power draw.
    # packed_data: path prefix of a packed pool (.tokens.npy/.offsets.npy,
    # see opera_lm.packed) -- replaces the resident tensor with an mmap
    # fetch per step. The generator and randint call are IDENTICAL to
    # GpuBatchSource, so the batch stream and checkpoint format are
    # unchanged; train_data is then only used for eval-side statistics
    # (and may be the empty list when the caller keeps just eval pools).
    batch_source = None
    if gpu_data:
        try:
            if packed_data:
                from .packed import PackedBatchSource
                batch_source = PackedBatchSource(packed_data, max_len,
                                                 torch_device, seed + rank)
                if is_main:
                    st_ = packed_stats(packed_data)
                    print(f"  packed batch source: {batch_source.N:,} seqs "
                          f"({st_['total_tokens'] / 2**20:.0f}M tokens, "
                          f"mmap) on {torch_device}"
                          + (f", {world_size} ranks" if ddp else ""),
                          flush=True)
            else:
                batch_source = GpuBatchSource(train_data, max_len,
                                              torch_device, seed + rank)
                if is_main:
                    print(f"  GPU batch source: {batch_source.N:,} sequences on "
                          f"{torch_device} ({batch_source.ids.numel() * 8 / 2**20:.0f} MiB)"
                          + (f", {world_size} ranks" if ddp else ""), flush=True)
        except Exception as e:
            if is_main:
                print(f"  WARNING: gpu_data failed ({e}); CPU sampling", flush=True)
            batch_source = None

    warmup = warmup_steps
    if optimizer == 'muon':
        # T0.1 (roadmap): Muon on matrix-shaped hidden params (rot_free's
        # per-block 3x3 maps, cross_mlp, untied head), AdamW on embeddings/
        # gates/gains/scalars. ONE optimizer object -> resume machinery
        # untouched. Per-group lr is re-set every step from the schedule.
        from .muon import Muon, split_muon_params
        muon_p, adam_p = split_muon_params(model)
        opt = Muon([
            {'params': muon_p, 'use_muon': True, 'lr': muon_lr},
            {'params': adam_p, 'use_muon': False, 'lr': max_lr},
        ], lr=max_lr)
        if is_main:
            print(f"  Muon: {sum(p.numel() for p in muon_p):,} matrix params "
                  f"(lr {muon_lr}) + AdamW: {sum(p.numel() for p in adam_p):,} "
                  f"(lr {max_lr})", flush=True)
    elif use_foreach:
        # AdamW(foreach=True): fused multi-tensor step. weight_decay=0.0
        # keeps the math equal to Adam so the recipe is unchanged.
        opt = torch.optim.AdamW(model.parameters(), lr=max_lr,
                                weight_decay=0.0, foreach=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    # OPT: half dtype must be NATIVE to the GPU. bf16 is only fast on
    # Ampere+ (A100/RTX30xx); Turing (T4) and older emulate it. Fall back
    # to fp16 (+GradScaler, below) where bf16 is unsupported.
    # MPS (v8.8, measured 2026-07-21, scan arm d=640 nb=160 L=4 vocab 10k):
    # autocast buys NO speed on MPS (fp32 1009 / fp16 1006 / bf16 1023
    # ms/step -- the model is kernel-launch-bound), but fp16 badly breaks
    # loss-trajectory parity with fp32 (max dev 0.57 over 50 seeded steps
    # vs bf16's 0.05; loss@20 6.32 vs 4.65 fp32). bf16: same speed as
    # fp32, near-fp32 numerics (fp32 exponent range), no GradScaler.
    if device == 'mps':
        amp_dtype = torch.bfloat16
    elif device == 'cuda':
        # NOTE: torch.cuda.is_bf16_supported() returns True on some torch
        # versions even when bf16 is only EMULATED (T4/Turing) -- and
        # emulated bf16 breaks inductor graphs ("does not support bfloat16
        # compilation natively, skipping") and runs slow. Gate on compute
        # capability instead: bf16 is native on Ampere+ (sm_80) only.
        cap = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16
    if use_amp and is_main:
        print(f"  AMP dtype: {amp_dtype}", flush=True)
    scaler = None
    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler(device)
        except Exception:
            scaler = None
            if is_main:
                print("  (no GradScaler on this backend; fp16 without scaling)", flush=True)
    import contextlib
    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type=device, dtype=amp_dtype)
        return contextlib.nullcontext()

    if ddp:
        # Wrap for the training step ONLY -- eval/inspection/checkpointing
        # below always use the raw `model` on rank 0, never `model_c`, so
        # DDP's forward-hook/gradient-sync machinery is never invoked
        # outside an actual backward() call.
        from torch.nn.parallel import DistributedDataParallel as _DDP
        # device_ids/output_device are CUDA-only; gloo (CPU, testing
        # only, see the note above) requires them to be None.
        # find_unused_parameters=True: train_lm_loss calls
        # model.apply_head() a SECOND time out-of-band (once per aux
        # layer, bypassing model_c/DDP entirely) on top of the head
        # usage already inside model_c's own forward (head_last_only) --
        # DDP's default single-forward-hook bucketing does not expect
        # the same parameters touched via two separate call sites in
        # one iteration and errors ("Expected to have finished
        # reduction..."), reproduced and confirmed fixed by this flag
        # on the CPU/gloo selftest below.
        model_c = _DDP(model, find_unused_parameters=True,
                       **({'device_ids': [local_rank], 'output_device': local_rank}
                          if device == 'cuda' else {}))

    # Colab checkpointing (v8.0): resume from a mid-training checkpoint.
    # RNG states are saved/restored so the batch stream continues exactly.
    # Under ddp: model/opt/step/global-RNG state is common to all ranks,
    # so every rank independently reads the same rank0-written file
    # (cheap, and simpler/less risky than reordering the DDP wrap above
    # to happen after this block so a broadcast would pick it up). Each
    # rank's GpuBatchSource stream is per-rank (seed + rank), so it is
    # saved/restored from its OWN separate rank-tagged file instead.
    train_ckpt = os.path.join(out_dir, f'opera_v8_0_{tag.replace("+","_")}_train_ckpt.pt')
    rank_ckpt = (train_ckpt.replace('_train_ckpt.pt', f'_train_ckpt_rank{rank}.pt')
                 if ddp else None)
    start_step = 0
    if resume and os.path.exists(train_ckpt):
        # weights_only=False: the checkpoint stores RNG states (numpy
        # objects) and is self-produced/trusted. Required on torch >= 2.6
        # where weights_only defaults to True.
        st = torch.load(train_ckpt, map_location=torch_device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        start_step = st['step'] + 1
        random.setstate(st['py_rng'])
        np.random.set_state(st['np_rng'])
        torch.set_rng_state(st['torch_rng'].cpu())
        if device == 'cuda' and st.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(st['cuda_rng'].cpu())
        if not ddp and batch_source is not None and st.get('gpu_rng') is not None:
            batch_source.load_state_dict(st['gpu_rng'])
        if is_main:
            print(f"  RESUMED from {train_ckpt} at step {start_step}", flush=True)
    if ddp and batch_source is not None and rank_ckpt and os.path.exists(rank_ckpt):
        rst = torch.load(rank_ckpt, map_location=torch_device, weights_only=False)
        batch_source.load_state_dict(rst['gpu_rng'])
        print(f"  rank {rank}: resumed its batch-source stream from {rank_ckpt}", flush=True)

    # Eval/inspection always run against the raw model, never the DDP
    # wrapper (no backward is ever called during eval, so no DDP sync
    # machinery is needed) -- for the non-ddp path this is unchanged
    # from before (model_c is model itself, or the compiled wrapper).
    eval_model = model if ddp else model_c

    init_ppl = _oom_backstop(
        lambda b: compute_perplexity(eval_model, test_short[:200], max_len, b, device),
        32, device) if is_main else None
    if is_main:
        print(f"  Initial per-token perplexity: {init_ppl:.2f} (chance ~ {actual_vocab_size}; "
              f"if this is >> chance, STOP - init is broken)", flush=True)

    t0 = time.time()
    nan_skips = 0            # batches skipped by the divergence guard
    last_loss_finite = True  # gates checkpointing (see guard comment below)
    for step in range(start_step, steps):
        lr = get_lr(step, warmup, steps, max_lr, schedule=lr_schedule,
                    wsd_decay_frac=wsd_decay_frac)
        for g in opt.param_groups:
            # Muon groups follow the same warmup/decay schedule, rescaled
            # to their own base lr (muon_lr).
            if optimizer == 'muon' and g.get('use_muon'):
                g['lr'] = lr * (muon_lr / max_lr)
            else:
                g['lr'] = lr

        if batch_source is not None:
            token_ids, lengths = batch_source.sample(batch)
        else:
            batch_sents = random.sample(train_data, min(batch, len(train_data)))
            token_ids, lengths = make_batch_full(batch_sents, max_len)
            token_ids, lengths = token_ids.to(device), lengths.to(device)

        # v9 arm C: LENGTH CURRICULUM. Slice the sampled batch to the
        # current stage length. Exact for stream data (a truncated doc
        # chunk is a valid sequence) and causal-exact: supervised
        # positions (< t_cur) see only tokens <= their position, so the
        # loss is bitwise the loss of a natively short batch (selftested).
        t_cur = None
        if curriculum is not None:
            t_cur = curriculum_len(step, curriculum[0], curriculum[1],
                                   max_len)
            if t_cur < token_ids.shape[1]:
                token_ids = token_ids[:, :t_cur]
                lengths = lengths.clamp(max=t_cur)

        # GRADIENT ACCUMULATION (accum > 1): micro-batches are sampled
        # EXACTLY as the incumbent stream (same RNG draws per step, so the
        # batch stream and --resume semantics are untouched); the optimizer
        # applies the averaged gradient only on BOUNDARY steps
        # ((step+1) % accum == 0, plus a flush on the final step so a
        # short trailing window is not silently dropped). At accum=1 every
        # step is a boundary and this block is bitwise the incumbent loop.
        # Under DDP, non-boundary micro-steps run under no_sync(): grads
        # accumulate locally and are all-reduced once per update instead
        # of once per micro-batch. The loss logged below is still the raw
        # micro-batch loss (not divided by accum).
        boundary = ((step + 1) % accum == 0) or (step == steps - 1)
        sync_ctx = (model_c.no_sync() if (ddp and not boundary)
                    else contextlib.nullcontext())
        with sync_ctx, amp_ctx():
            if msup:
                out = model_c(token_ids, lengths, return_levels=True)
                loss, _, _ = lm_loss(out.logits, token_ids, lengths)
                loss = loss + msup_weight * msup_loss(
                    model, out.levels, token_ids, lengths, aux_frac=aux_frac)
            else:
                # OPT: final-layer head inside the compiled graph; aux
                # layers' head computed on aux_frac of positions only.
                out = model_c(
                    token_ids, lengths, return_states=True, head_last_only=True)
                all_logits, states = out.logits, out.states
                # train_lm_loss calls apply_head(), a custom method, not
                # forward() -- DDP's wrapper only proxies forward()/
                # __call__, so model_c.apply_head AttributeErrors under
                # ddp (caught by the selftest). Use the raw `model` only
                # in that case; model and the module inside model_c are
                # the same underlying object, so gradients through
                # apply_head's params are still tracked by DDP's
                # reducer hooks regardless of which reference invokes
                # this op (same reasoning already applied to
                # msup_loss(model, ...) above). Left as model_c (not
                # touched) for the non-ddp path, where model_c may be a
                # torch.compile wrapper and this call site's compiled-
                # graph behavior is untouched/unverified by this change.
                head_model = model if ddp else model_c
                loss, _, _ = train_lm_loss(
                    head_model, states, token_ids, lengths,
                    aux_frac=aux_frac, final_logits=all_logits[0])

        # DIVERGENCE GUARD (added after the 2026-08-23 Kaggle 2xT4 run
        # NaN'd at ~step 7000 and the save branch then overwrote the last
        # healthy checkpoint with poisoned weights). A non-finite loss
        # means THIS batch's forward already blew up: backwarding it would
        # write inf/nan into .grad and (at a boundary) into the weights,
        # from which training never recovers. Instead the whole batch is
        # skipped -- no backward, no update, grads untouched. Under DDP
        # the skip decision is COLLECTIVE (all-reduced MAX of the finite
        # flag): if only one rank's batch went non-finite, the others
        # must skip too, or the next gradient all-reduce deadlocks.
        # The batch stream does NOT rewind -- the RNG draws for this step
        # are consumed, same as any other skipped-data policy, so resume
        # stays deterministic. `last_loss_finite` gates CHECKPOINTING:
        # once weights are poisoned the loss stays nan on every later
        # batch, so refusing to save while it is nan keeps the last
        # healthy state on disk instead of overwriting it (the exact loss
        # suffered on the run above).
        finite = bool(torch.isfinite(loss).all().item())
        if ddp and world_size > 1:
            import torch.distributed as dist
            t_flag = torch.tensor([float(finite)], device=torch_device)
            dist.all_reduce(t_flag, op=dist.ReduceOp.MIN)
            finite = bool(t_flag.item() > 0.5)
        if not finite:
            nan_skips += 1
            if is_main and (nan_skips == 1 or nan_skips % 50 == 0):
                print(f"    WARNING: non-finite loss at step {step} "
                      f"(batch skipped, no update; {nan_skips} skips so "
                      f"far)", flush=True)
            last_loss_finite = False
            continue
        last_loss_finite = True

        if scaler is not None:
            scaler.scale(loss / accum).backward()
            if boundary:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        else:
            (loss / accum).backward()
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        if is_main and (step % 200 == 0 or step == steps - 1):
            elapsed = time.time() - t0
            done = step - start_step + 1
            cur_txt = f"  T_cur {t_cur}" if t_cur is not None else ""
            skip_txt = f"  [nan-skips {nan_skips}]" if nan_skips else ""
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.5f}  "
                  f"({elapsed:.1f}s, {elapsed/done:.2f}s/step){cur_txt}{skip_txt}",
                  flush=True)

        # Save cadence under accumulation: fires on step % save_every ==
        # accum-1, which is always a boundary step (see the assert at the
        # top), so checkpoints land right AFTER an optimizer update and
        # never store in-flight gradients -- resume stays exact. At
        # accum=1 this is bitwise the incumbent cadence (step % save_every == 0).
        # GATED ON last_loss_finite: a poisoned run must not overwrite its
        # own last healthy checkpoint (2026-08-23 Kaggle lesson -- see the
        # divergence-guard comment above). The rank-tagged stream state is
        # still written: it carries no weights, and keeping it current
        # means a resumed-from-healthy-checkpoint run continues the same
        # batch streams.
        if (save_every and step > 0 and last_loss_finite
                and step % save_every == (accum - 1) % save_every):
            if is_main:
                torch.save({
                    'model': model.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'tag': tag,
                    'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': (torch.cuda.get_rng_state()
                                 if device == 'cuda' else None),
                    'gpu_rng': (batch_source.state_dict()
                                if (batch_source is not None and not ddp) else None),
                }, train_ckpt)
                print(f"    checkpoint -> {train_ckpt}", flush=True)
            if ddp and batch_source is not None:
                # Every rank's own stream state, not just rank 0's --
                # see the resume block above for why.
                torch.save({'gpu_rng': batch_source.state_dict()}, rank_ckpt)

        if is_main and step % 1000 == 0 and step > 0:
            ppl = _oom_backstop(
                lambda b: compute_perplexity(eval_model, test_short[:200], max_len, b, device),
                _eval_batch(32, max_len), device)
            print(f"    per-token perplexity: {ppl:.2f}", flush=True)
            # raw `model` (not eval_model/model_c): return_energy changes
            # forward()'s control flow, and model_c may be torch.compile'd
            # against the default (return_energy=False) path -- same
            # eager-only reasoning as inspect_tree/rmt_states_diagnostic
            # below, just hit periodically instead of only at the end.
            _oom_backstop(
                lambda b: energy_diagnostic(model, test_short[:b], b, device),
                _eval_batch(32, max_len), device)

    if is_main:
        print(f"\n=== Final Evaluation ===", flush=True)
        ppl_1k = _oom_backstop(
            lambda b: compute_perplexity(eval_model, test_short[:1000], max_len, b, device),
            _eval_batch(32, max_len), device)
        ppl = _oom_backstop(
            lambda b: compute_perplexity(eval_model, test_short[:5000], max_len, b, device),
            _eval_batch(32, max_len), device)
        print(f"  In-length per-token PPL (<= {max_len}): {ppl:.2f} on 5k test sentences", flush=True)
        print(f"  (legacy 1k-sentence eval for comparison with older runs: {ppl_1k:.2f})", flush=True)

    if not is_main:
        # All further code (extrapolation eval, tree inspection, RMT
        # diagnostic, final checkpoint, results.jsonl) is rank-0-only --
        # none of it calls model_c or any dist.* collective, so no
        # further cross-rank synchronization is needed; each rank just
        # tears down its own process group independently and exits.
        if ddp:
            import torch.distributed as dist
            dist.destroy_process_group()
        return None

    print(f"  NOTE: single-run differences under ~1.5 PPL are within seed+eval noise.", flush=True)

    print(f"\n=== Length Extrapolation (train <= {max_len}) ===", flush=True)
    extrap = extrapolation_eval(model, test_long, max_len, eval_max_len, 16, device)  # eager: one-off shapes
    for bucket, (bppl, n) in extrap.items():
        if bppl is None:
            print(f"  len {bucket}: insufficient data (n={n})", flush=True)
        else:
            ratio = bppl / ppl
            print(f"  len {bucket}: PPL {bppl:.2f}  (n={n}, {ratio:.2f}x in-length)", flush=True)

    if idx2word is not None:
        inspect_tree(model, test_short[:10], idx2word, device, n=5)

    rmt = _oom_backstop(
        lambda b: rmt_states_diagnostic(model, test_short, b, device, num_sentences=500),
        32, device)

    energy = _oom_backstop(
        lambda b: energy_diagnostic(model, test_short[:b], b, device),
        _eval_batch(32, max_len), device)

    ckpt_path = os.path.join(out_dir, f'opera_v8_0_{tag.replace("+", "_")}.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}", flush=True)

    results = {
        'model': 'opera_v9_tree', 'config': tag, 'pe': pe_mode,
        'rot': rot_mode, 'seed': seed, 'data': data_mode,
        'docs_limit': (docs_limit if data_mode == 'docs-en' else None),
        'fold': fold_mode, 'fold_rotors': fold_rotors, 'fold_scale': fold_scale,
        'norm': norm_mode, 'act': act_mode, 'node_residual': node_residual,
        'tree_drop': tree_drop,
        'lock_mode': lock_mode, 'tie': tie, 'dropout': dropout, 'msup': msup,
        'params': npar, 'd': d, 'nb': nb, 'num_layers': num_layers,
        'readout_mode': readout_mode, 'mem_mode': mem_mode,
        'mem_dim': (mem_dim if mem_mode != 'none' else None),
        'homeo_mode': homeo_mode, 'node_paths': node_paths,
        'fold_adapt': fold_adapt,
        'vocab_size': actual_vocab_size, 'max_len': max_len,
        'eval_max_len': eval_max_len, 'steps': steps,
        'final_loss': loss.item(),
        'nan_skips': nan_skips,
        'test_perplexity_in_length': ppl, 'test_perplexity_1k_legacy': ppl_1k,
        'extrapolation': {k: v[0] for k, v in extrap.items()},
        'init_perplexity': init_ppl,
        'rmt': rmt,
        'energy_by_depth': energy,
        'opt': {'compile': compile_mode, 'gpu_data': gpu_data,
                'aux_frac': aux_frac, 'amp': use_amp,
                'lr': max_lr, 'warmup': warmup_steps,
                'foreach': use_foreach,
                'optimizer': optimizer,
                'lr_schedule': lr_schedule,
                'wsd_decay_frac': (wsd_decay_frac
                                   if lr_schedule != 'cosine' else None),
                'accum': accum,
                'muon_lr': (muon_lr if optimizer == 'muon' else None),
                'oam': {'k': oam_k, 'charges': str(oam_charges),
                        'phi': oam_phi, 'shared_gate': oam_shared_gate,
                        'combine': oam_combine,
                        'pair': oam_pair,
                        'transport': oam_transport,
                        'levelgate': oam_levelgate,
                        'chan_emb': oam_chan_emb} if fold_mode == 'oam' else None,
                'rack_exitnorm': (rack_exitnorm
                                  if fold_mode == 'rack' else None),
                'salience': (scan_salience
                             if fold_mode == 'scan' else None),
                'workspace': (workspace
                              if fold_mode == 'scan' else None),
                'gate_bias': (list(fold_gate_bias)
                              if fold_gate_bias is not None else None),
                'curriculum': (list(curriculum)
                               if curriculum is not None else None)},
    }
    results_path = os.path.join(out_dir, 'opera_v8_0_results.jsonl')
    with open(results_path, 'a') as f:
        f.write(json.dumps(results) + '\n')
    print(f"Saved to {results_path}", flush=True)
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    return results

