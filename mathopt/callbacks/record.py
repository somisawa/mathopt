from .core import Callback


class Record(Callback):
    def __init__(self, result=[]) -> None:
        super().__init__()

        self.result = result

    def call(self, i, x, obj, grad):
        self.result.append(obj(x))
