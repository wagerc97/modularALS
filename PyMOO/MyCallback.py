#!/usr/bin/python3
from pymoo.core.callback import Callback

class MyCallback(Callback):
    """
    Callback function to collect data during minimization.
    Source: https://pymoo.org/interface/callback.html
    """
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    #def notify(self, algorithm):
    def notify(self, algorithm, **kwargs):
        self.data["best"].append(algorithm.pop.get("F").min())
