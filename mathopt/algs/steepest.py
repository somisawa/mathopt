from dataclasses import dataclass, field
import numpy as np
from typing import List, Callable


from .core import Core
from ..callbacks.core import Callback


@dataclass
class SteepestDescent(Core):
    objective: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    stepsize: float
    n_iter: int
    armijo: bool = field(default=False)

    def _iter(self, x: np.ndarray) -> np.ndarray:
        if self.armijo:
            return self._armijo_check(x)
        else:
            return x - self.stepsize * self.grad(x)

    def run(
        self,
        x_init: np.ndarray,
        callbacks: List[Callback] = [],
    ) -> None:
        x = x_init
        for i in range(self.n_iter):
            x = self._iter(x)

            for c in callbacks:
                c.call(i, x, self.objective, self.grad)

    def _armijo_check(self, x, eps=1e-3, tau=0.5, max_l=150):
        d = self.grad(x)

        for l in range(max_l):
            taul = tau**l
            armijo_point = x - self.stepsize * taul * d
            if (
                self.objective(armijo_point)
                <= self.objective(x) - eps * self.stepsize * taul * d.T @ d
            ):
                return armijo_point

        print("Reach max iteration of Armijo rule.")
        return armijo_point
