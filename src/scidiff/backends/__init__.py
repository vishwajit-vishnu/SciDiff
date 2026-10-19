from __future__ import annotations

from typing import Literal

from scidiff.backends.base import Backend

_BACKEND: Backend | None = None

BackendName = Literal["torch", "jax", "julia"]


def set_backend(name: BackendName) -> Backend:
    """Set the active backend globally."""
    global _BACKEND

    name = name.lower()  # type: ignore[assignment]
    if name == "torch":
        from scidiff.backends.torch import TorchBackend
        _BACKEND = TorchBackend()
    elif name == "jax":
        from scidiff.backends.jax import JaxBackend
        _BACKEND = JaxBackend()
    elif name == "julia":
        from scidiff.backends.julia import JuliaBackend
        _BACKEND = JuliaBackend()
    else:
        raise ValueError(f"Unknown backend '{name}'. Choose from: torch, jax, julia")

    return _BACKEND


def get_backend() -> Backend:
    """Return the currently active backend. Defaults to 'torch' if none set."""
    global _BACKEND
    if _BACKEND is None:
        return set_backend("torch")
    return _BACKEND