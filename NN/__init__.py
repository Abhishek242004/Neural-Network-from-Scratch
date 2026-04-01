"""NumPy-based neural network mini-framework.

This file exists to make the top-level `NN` directory a proper Python package so
example scripts (and user code) can import via `from NN...` when run from the
repo root.
"""

from . import activation, layers, losses, model, optimizer

__all__ = [
    "activation",
    "layers",
    "losses",
    "model",
    "optimizer",
]
