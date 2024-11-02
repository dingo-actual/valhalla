from typing import Callable, List, Optional, Sequence, Tuple, Union

from numpy import ndarray

from ..core.population import PopulationBase
from .instance import ParticleInstance


class ParticleSwarmBase(PopulationBase):
    def __init__(
        self, 
        n_particles: int, 
        pos_len: int, 
        pos_initializer: Callable[[int], ndarray], 
        vel_initializer: Callable[[int], ndarray], 
        c_initializer: Union[Tuple[float, float], Callable[[], Tuple[float, float]]], 
        w_initializer: Optional[Union[float, Callable[[], float]]] = None,
        topology: Optional[Callable[[Sequence[ParticleInstance]], Union[List[List[int]], List[List[float]]]]] = None
    ) -> None:
        def _create_particle(
            pos_len: int, 
            pos_initializer: Callable[[int], ndarray], 
            vel_initializer: Callable[[int], ndarray], 
            c_initializer: Union[Tuple[float, float], Callable[[], Tuple[float, float]]],
            w_initializer: Optional[Union[float, Callable[[], float]]] = None
        ) -> ParticleInstance:
            return ParticleInstance(
                lambda: pos_initializer(pos_len),
                lambda: vel_initializer(pos_len),
                c_initializer,
                w_initializer
            )
        def _initialize():
            return [
                _create_particle(pos_len, pos_initializer, vel_initializer, c_initializer, w_initializer) for _ in range(n_particles)
            ]
        
        super(ParticleSwarmBase, self).__init__(_initialize, subpopulations=None, topology=topology)
        self.best_pos = None
        self.best_fitness = None

    def __iter__(self):
        return self.solutions.__iter__()