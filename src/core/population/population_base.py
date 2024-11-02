from typing import Any, Callable, Sequence, Optional, Union, List


class PopulationBase(object):
    def __init__(
        self,
        initializer: Optional[Callable],
        subpopulations: Optional[Sequence["PopulationBase"]] = None,
        topology: Optional[
            Union[
                Callable[[Any], Union[Sequence[Sequence[int]], Sequence[Sequence[float]]]], 
                Union[Sequence[Sequence[int]], Sequence[Sequence[float]]]
            ]
        ] = None,
    ):
        self.subpopulations = subpopulations
        if self.subpopulations is not None:
            self.solutions = None
            if isinstance(topology, Callable):
                self.topology = topology(self.subpopulations)
            else:
                self.topology = topology
        else:
            if initializer is None:
                raise ValueError(
                    "one of initializer and subpopulations must be specified"
                )
            else:
                self.solutions = initializer()
                if isinstance(topology, Callable):
                    self.topology = topology(self.solutions)
                else:
                    self.topology = topology

    def __iter__(self):
        if self.subpopulations is None:
            out = self.solutions.__iter__()
        else:
            out = self.subpopulations.__iter__()
            
        return out
    
    def __len__(self):
        if self.subpopulations is None:
            out = self.solutions.__len__()
        else:
            out = self.subpopulations.__len__()
            
        return out
    
    def __getitem__(self, key):
        if self.subpopulations is None:
            out = self.solutions.__getitem__(key)
        else:
            out = self.subpopulations.__getitem__(key)
            
        return out
    
    def __setitem__(self, key, val):
        if self.subpopulations is None:
            out = self.solutions.__setitem__(key, val)
        else:
            out = self.subpopulations.__setitem__(key, val)
            
        return out
    
    def __str__(self):
        subpopulations_st = 'None' if self.subpopulations is None else ',\n'.join(pop.__str__() for pop in self)
        solutions_st = 'None' if self.solutions is None else ',\n'.join(solution.__str__() for solution in self)
        topology_st = 'None' if self.topology is None else '[\n'
        if self.topology is not None:
            topology_st += ',\n'.join(row for row in self.topology)
            topology_st += '\n]'
                
        return f'''Population(
              Subpopulations: {subpopulations_st},
              Solutions: {solutions_st},
              Topology: {topology_st}
            )'''

    @classmethod
    def neighbors(self, ix: int) -> Optional[Union[List[float], List[int]]]:
        out = None
        if self.topology is not None:
            out = self.topology[ix]
        return out
