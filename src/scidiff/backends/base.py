from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Backend(ABC):
    """Abstract backend interface used throughout SciDiff."""

    name: str

    @abstractmethod
    def array(self, data: Any, dtype: Any = None, device: Any = None) -> Any:
        """Create an array / tensor on this backend."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None, device: Any = None) -> Any:
        ...

    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: Any = None, device: Any = None) -> Any:
        ...

    @abstractmethod
    def grad(self, func: Callable) -> Callable:
        """Return a function that computes gradients of `func`."""

    @abstractmethod
    def value_and_grad(self, func: Callable) -> Callable:
        """Return a function that computes both value and gradients."""

    @abstractmethod
    def jit(self, func: Callable) -> Callable:
        """Just-in-time compile `func` if the backend supports it."""

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """Convert backend array to NumPy."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"