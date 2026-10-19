# Physics Informed Neural Network (forward problem)
from typing import Optional, Sequence

from scidiff.algorithms.base import Algorithm

class PINN(Algorithm)
    """
    Physics Informed Neural Network for forward problem
    
    Parameters:
    -----------
    residual_fn: callable
        Function that computed the PDE residual given network predictions
        and coordinates. Signature depends on backend
    """
    def __init__(
            self,
            residual_fn: Optional[Callable] = None,
            layers: Sequence[int] = (64, 64, 64),
            activation: str = "tanh",
            backend: str = "torch",
            **kwargs
    ):
        super().__init__()
        self.residual_fn = residual_fn
        self.layers = list(layers)
        self.activation = activation
        # Network will be created in fit
        self.network = None
        # States generated during training is stored in history
        self.history: Dict[str, list] = {}

    def fit(
            self
    ) -> None:
        # TODO: LAter
        pass


