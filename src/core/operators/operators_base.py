from abc import abstractmethod
from operator import __add__, __mul__

from .generics import Population


class PopulationOperatorBase(object):
    def __init__(self) -> None:
        pass

    def __call__(self, pop: Population) -> None:
        self.op(pop)

    def __add__(self, other: "PopulationOperatorBase") -> "SequentialOperator":
        return SequentialOperator(self, other)

    @classmethod
    @abstractmethod
    def op(self, pop: Population) -> None:
        raise NotImplementedError
    
    
class PopulationOperatorForEach(PopulationOperatorBase):
    @classmethod
    @abstractmethod
    def op_single(self, Instance) -> None:
        raise NotImplementedError
    
    def op(self, pop: Population) -> None:
        for instance in pop:
            self.op_single(instance)


class SequentialOperator(PopulationOperatorBase):
    def __init__(self, first: PopulationOperatorBase, second: PopulationOperatorBase):
        self.first = first
        self.second = second

    def __add__(self, other: "PopulationOperatorBase") -> "SequentialOperator":
        return super().__add__(other)

    def op(self, arg: Population) -> None:
        self.first(arg)
        self.second(arg)
