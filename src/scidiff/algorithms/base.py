
from abc import ABC, abstractmethod
from typing import Any, Dict

class Algorithm(ABC):
    """Abstract base class for algorithms.

    Parameters
    ----------
    backend : str
        One of {"torch", "jax", "julia"}. Defaults to "torch".
    """
    def __init__(self, backend: str = "torch", **kwargs) -> None:
        from scidiff.backends import set_backend

        self.backend_name = backend
        self.backend = set_backend(backend)
        self.config: Dict[str, Any] = kwargs
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, data, pde=None, **kwargs) -> "Algorithm":
        """Train the algorithm. Must set self.is_fitted = True """

    @abstractmethod
    def predict(self, data, pde=None, **kwargs) -> Any:
        """Predict using the algorithm."""

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """Save the algorithm."""
        raise NotImplementedError(f"{self.__class__.__name__}.save() not implemented yet")

    @abstractmethod
    def load(self, path: str, **kwargs) -> "Algorithm":
        """Load the algorithm."""
        raise NotImplementedError(f"{self.__class__.__name__}.load() not implemented yet")

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, **self.config}

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(backend={self.backend}, status={status})"


