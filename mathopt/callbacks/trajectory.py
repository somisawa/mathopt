import numpy as np
from .core import Callback


class Trajectory(Callback):
    def __init__(self, traj) -> None:
        super().__init__()

        self.traj = traj

    def call(self, i, x, obj, grad):
        self.traj = np.c_[self.traj, x]
