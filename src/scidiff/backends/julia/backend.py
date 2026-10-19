from __future__ import annotations

from typing import Any, Callable

from scidiff.backends.base import Backend


class JuliaBackend(Backend):
    """Julia backend using juliacall.

    This is a minimal skeleton. You will expand it to call
    Lux.jl / NeuralPDE.jl / custom Julia code later.
    """

    name = "julia"

    def __init__(self) -> None:
        try:
            from juliacall import Main as jl
            self.jl = jl
            # Example: make sure basic packages are available
            # self.jl.seval("using Pkg; Pkg.add([\"Lux\", \"NeuralPDE\"])")
        except ImportError as e:
            raise ImportError(
                "Julia backend requires `juliacall`. "
                "Install with: pip install scidiff[julia]"
            ) from e

    def array(self, data: Any, dtype: Any = None, device: Any = None) -> Any:
        # Convert Python/NumPy → Julia Array
        return self.jl.Array(data)

    def zeros(self, shape, dtype=None, device=None):
        return self.jl.zeros(shape)

    def ones(self, shape, dtype=None, device=None):
        return self.jl.ones(shape)

    def grad(self, func: Callable) -> Callable:
        raise NotImplementedError("Automatic differentiation via Julia backend coming soon")

    def value_and_grad(self, func: Callable) -> Callable:
        raise NotImplementedError("value_and_grad via Julia backend coming soon")

    def jit(self, func: Callable) -> Callable:
        # Julia has its own compilation model
        return func

    def to_numpy(self, x: Any):
        import numpy as np
        return np.asarray(x)