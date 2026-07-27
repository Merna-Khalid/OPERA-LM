"""opera-lm: Operadic Planar Equivariant Recursive Algebra language model.

Spinor states in Cl(3), balanced binary tree composition, Fenwick prefix
readout. No positional encodings -- position is structure.
"""
from .model import OperaSpinorFenwickTree, count_params, fenwick_blocks
from .losses import lm_loss, train_lm_loss, msup_loss
from .train import (curriculum_len, get_lr, make_batch_full, GpuBatchSource,
                    train, compute_perplexity, extrapolation_eval)
from .data import load_data

__version__ = "0.9.0"

__all__ = [
    "OperaSpinorFenwickTree",
    "count_params",
    "fenwick_blocks",
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
