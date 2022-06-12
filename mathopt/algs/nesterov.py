from dataclasses import dataclass
import numpy as np
from typing import List, Callable


from .core import Core
from ..callbacks.core import Callback


@dataclass
class Nesterov(Core):
    objective: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    n_iter: int

    stepsize: float
    momentum: Callable[[int], float] = lambda k: k / (k + 3)

    def _iter(self, x: np.ndarray, x_prev: np.ndarray, it: int) -> np.ndarray:
        y = x + self.momentum(it - 1) * (x - x_prev)
        return y - self.stepsize * self.grad(y)

    def run(
        self,
        x_init: np.ndarray,
        callbacks: List[Callback] = [],
    ) -> None:
        x = x_init - self.stepsize * self.grad(x_init)
        x_prev = x_init
        for i in range(self.n_iter):
            x_tmp = x.copy()
            x = self._iter(x, x_prev, i + 1)
            x_prev = x_tmp

            for c in callbacks:
                c.call(i, x, self.objective, self.grad)
