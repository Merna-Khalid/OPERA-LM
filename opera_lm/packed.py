"""packed.py -- memory-mapped sequence pools + a lazy batch source.

For pools too large to unpickle into RAM every session (the Path A
FineWeb pretrain pool is ~250M tokens: ~9-10GB as Python int lists plus
a 2GB padded int64 tensor resident per GPU), pack the sequences ONCE
into a flat int32 array + int64 offsets stored as plain .npy files, and
memory-map them at session start: near-zero RAM, instant open, and each
step copies only its sampled batch (~64KB) to the device.

PackedBatchSource mirrors GpuBatchSource EXACTLY: same constructor
signature shape, same dedicated torch.Generator created on the same
device and seeded the same way (seed, or seed + rank under DDP), and
the identical randint call -- so the sampled INDEX STREAM and the
checkpointed generator-state format are bit-identical to the incumbent
source. Only the row-fetch differs: mmap slices instead of a device
gather. selftest verifies stream equality end-to-end.

Storage: a directory-less PAIR of .npy files (npz is a zip container
and cannot be mmapped):
    <prefix>.tokens.npy   int32 flat token ids, shape [total]
    <prefix>.offsets.npy  int64, shape [N + 1] (offsets[i]:offsets[i+1])
Sequences longer than max_len are truncated at PACK time (same
convention as GpuBatchSource's build step).
"""
import os

import numpy as np
import torch

__all__ = ["pack_pool", "PackedBatchSource", "packed_stats"]


def pack_pool(seqs, prefix, max_len):
    """Pack token-id sequences to <prefix>.tokens.npy / .offsets.npy.

    Truncation to max_len happens here (GpuBatchSource's convention), so
    a batch fetched later never needs slicing. Returns a stats dict.
    """
    tokens = np.empty(sum(min(len(s), max_len) for s in seqs), dtype=np.int32)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    pos = 0
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        tokens[pos:pos + n] = s[:n]
        pos += n
        offsets[i + 1] = pos
    assert pos == tokens.size
    for arr, name in ((tokens, "tokens"), (offsets, "offsets")):
        np.save(f"{prefix}.{name}.npy", arr)
    return {"prefix": prefix, "n_seqs": len(seqs), "total_tokens": int(pos),
            "max_len": max_len}


def packed_stats(prefix):
    """Header stats without loading the token array."""
    offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")
    return {"prefix": prefix, "n_seqs": int(offsets.size - 1),
            "total_tokens": int(offsets[-1])}


class PackedBatchSource:
    """Lazy mmap-backed twin of GpuBatchSource (see module docstring)."""

    def __init__(self, prefix, max_len, device, seed):
        self.tokens = np.load(f"{prefix}.tokens.npy", mmap_mode="r")
        self.offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")
        assert self.offsets[-1] == self.tokens.size, "corrupt packed pool"
        self.N = int(self.offsets.size - 1)
        self.max_len = max_len
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def sample(self, batch):
        # Identical RNG consumption to GpuBatchSource.sample: same
        # generator (device, seed), same randint call on the same device.
        idx = torch.randint(self.N, (batch,), generator=self.gen,
                            device=self.device)
        rows = idx.tolist()
        ids = np.zeros((len(rows), self.max_len), dtype=np.int64)
        lens = np.zeros(len(rows), dtype=np.int64)
        for b, i in enumerate(rows):
            i = int(i)
            n = min(int(self.offsets[i + 1] - self.offsets[i]), self.max_len)
            ids[b, :n] = self.tokens[self.offsets[i]:self.offsets[i] + n]
            lens[b] = n
        return (torch.from_numpy(ids).to(self.device),
                torch.from_numpy(lens).to(self.device))

    def state_dict(self):
        return self.gen.get_state().cpu()

    def load_state_dict(self, st):
        # Same CUDA-generator coercion as GpuBatchSource.load_state_dict.
        if isinstance(st, torch.Tensor):
            st = st.detach().to('cpu', torch.uint8)
        self.gen.set_state(st)
