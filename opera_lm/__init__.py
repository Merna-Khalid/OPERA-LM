"""opera-lm: Operadic Planar Equivariant Recursive Algebra language model.

Spinor states in Cl(3), balanced binary tree composition, Fenwick prefix
readout. No positional encodings -- position is structure.
"""
from .model import (OperaSpinorFenwickTree, OperaOutput, count_params,
                    fenwick_blocks, inject_geometry)
from .incremental import OperaDecoder, fenwick_blocks_of
from .muon import Muon, split_muon_params, zeropower_via_newtonschulz5
from .losses import lm_loss, train_lm_loss, msup_loss
from .train import (curriculum_len, get_lr, make_batch_full, GpuBatchSource,
                    train, compute_perplexity, extrapolation_eval)
from .data import load_data

__version__ = "0.9.0"

__all__ = [
    "OperaSpinorFenwickTree",
    "OperaOutput",
    "OperaDecoder",
    "fenwick_blocks_of",
    "Muon",
    "split_muon_params",
    "zeropower_via_newtonschulz5",
    "count_params",
    "fenwick_blocks",
    "inject_geometry",
    "lm_loss",
    "train_lm_loss",
    "msup_loss",
    "curriculum_len",
    "get_lr",
    "make_batch_full",
    "GpuBatchSource",
    "train",
    "load_data",
    "compute_perplexity",
    "extrapolation_eval",
    "__version__",
]
