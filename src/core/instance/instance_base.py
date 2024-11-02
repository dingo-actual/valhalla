from abc import abstractmethod

from typing import Any, Callable, Tuple, Sequence

class InstanceBase(object):
    def __init__(self, initializer: Callable[[], Tuple[Sequence[float], Any]]) -> None:
        self.solution, self.meta = initializer()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError