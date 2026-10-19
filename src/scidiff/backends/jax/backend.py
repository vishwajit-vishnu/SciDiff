from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from scidiff.backends.base import Backend


class JaxBackend(Backend):
    name = "jax"

    def array(self, data: Any, dtype: Any = None, device: Any = None) -> jnp.ndarray:
        return jnp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None, device=None):
        return jnp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None, device=None):
        return jnp.ones(shape, dtype=dtype)

    def grad(self, func: Callable) -> Callable:
        return jax.grad(func)

    def value_and_grad(self, func: Callable) -> Callable:
        return jax.value_and_grad(func)

    def jit(self, func: Callable) -> Callable:
        return jax.jit(func)

    def to_numpy(self, x: Any):
        return jax.device_get(x)