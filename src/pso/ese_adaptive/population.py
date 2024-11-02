from typing import Callable, Tuple, Union, Optional

from numpy import ndarray

from ..population import ParticleSwarmBase
    
    
class ParticleSwarmAPSOESE(ParticleSwarmBase):
    def __init__(
        self, 
        n_particles: int, 
        pos_len: int, 
        pos_initializer: Callable[[int], ndarray], 
        vel_initializer: Callable[[int], ndarray], 
        c_initializer: Union[Tuple[float, float], Callable[[], Tuple[float, float]]] = (2., 2.),
        w_initializer: Union[float, Callable[[], float]] = 0.9
    ) -> None:
        super(ParticleSwarmAPSOESE, self).__init__(
            n_particles,
            pos_len, 
            pos_initializer, 
            vel_initializer, 
            c_initializer,
            w_initializer
        )
        self.crnt_state = 'exploration'
        self.best_ix = None
        self.crnt_iter = 1
        self.stagnation = 0
        self.shock_mult = 1.
        self.elite_perturb_dims = 1
