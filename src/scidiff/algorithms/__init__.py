"""High-level SciML algorithms with sensible defaults."""

from scidiff.algorithms.base import Algorithm
from scidiff.algorithms.pinn import PINN
from scidiff.algorithms.inverse_pinn import InversePINN
from scidiff.algorithms.deeponet import DeepONet
from scidiff.algorithms.fno import FNO
from scidiff.algorithms.adaptive_pinn import AdaptivePINN

__all__ = [
    "Algorithm",
    "PINN",
    "InversePINN",
    "DeepONet",
    "FNO",
    "AdaptivePINN",
]