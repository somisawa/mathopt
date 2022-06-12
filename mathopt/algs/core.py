from typing import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Core:
    objective: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    stepsize: float
    n_iter: int

    def run(self):
        raise NotImplementedError
