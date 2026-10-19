"""SciDiff – Multi-backend Scientific Machine Learning (PINNs & more)."""

from __future__ import annotations

try:
    from scidiff._version import __version__
except ImportError:  # pragma: no cover
    # Fallback when the package is not installed / hatch-vcs has not run yet
    __version__ = "0.0.0.dev0"

from scidiff.backends import get_backend, set_backend

__all__ = [
    "__version__",
    "get_backend",
    "set_backend",
]