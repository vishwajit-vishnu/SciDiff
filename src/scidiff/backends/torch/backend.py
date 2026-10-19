from __future__ import annotations

from typing import Any, Callable

import torch

from scidiff.backends.base import Backend


class TorchBackend(Backend):
    name = "torch"

    def array(self, data: Any, dtype: Any = None, device: Any = None) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype, device=device)

    def zeros(self, shape, dtype=None, device=None):
        return torch.zeros(shape, dtype=dtype, device=device)

    def ones(self, shape, dtype=None, device=None):
        return torch.ones(shape, dtype=dtype, device=device)

    def grad(self, func: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            # Very minimal example – you will expand this later
            args = [a.requires_grad_(True) if isinstance(a, torch.Tensor) else a for a in args]
            out = func(*args, **kwargs)
            return torch.autograd.grad(out, args, create_graph=True)
        return wrapped

    def value_and_grad(self, func: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            args = [a.requires_grad_(True) if isinstance(a, torch.Tensor) else a for a in args]
            out = func(*args, **kwargs)
            grads = torch.autograd.grad(out, args, create_graph=True)
            return out, grads
        return wrapped

    def jit(self, func: Callable) -> Callable:
        try:
            return torch.compile(func)
        except Exception:
            # torch.compile is optional / version dependent
            # return the callable if some version does not support it
            return func

    def to_numpy(self, x: Any):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x