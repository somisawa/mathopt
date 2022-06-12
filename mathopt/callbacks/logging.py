from .core import Callback


class Logging(Callback):
    def call(self, i, x, obj, grad):
        print(f"{i + 1}-th iteration: Objective {obj(x)}")
