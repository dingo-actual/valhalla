from typing import Callable, Tuple, Union

from numpy import ndarray, zeros
from numpy.random import rand

from ..population import ParticleSwarmBase
from ..instance import ParticleInstance
    
    
class ParticleSwarmAdaptiveComplexDirected(ParticleSwarmBase):
    def __init__(
        self, 
        n_particles: int, 
        pos_len: int, 
        pos_initializer: Callable[[int], ndarray], 
        vel_initializer: Callable[[int], ndarray], 
        c_initializer: Union[Tuple[float, float], Callable[[], Tuple[float, float]]],
        prob_rand_connection: float,
        dist_threshold: float,
        diff_eps: float,
        init_inertia: float,
        final_inertia: float,
        n_iter: int,
        fitness_fn: Callable[[ParticleInstance], float],
        dist_fn: Callable[[ParticleInstance, ParticleInstance], float],
        diff_fn: Callable[[float, float], float],
    ) -> None:
        super(ParticleSwarmAdaptiveComplexDirected, self).__init__(
            n_particles,
            pos_len, 
            pos_initializer, 
            vel_initializer, 
            c_initializer
        )
        self.prob_rand_connection = prob_rand_connection
        self.dist_threshold = dist_threshold
        self.diff_eps = diff_eps
        self.init_inertia = init_inertia
        self.final_inertia = final_inertia
        self.crnt_inertia = init_inertia
        self.n_iter = n_iter
        self.crnt_iter = 1
        self.fitness_fn = fitness_fn
        self.dist_fn = dist_fn
        self.diff_fn = diff_fn
        self.in_degrees = zeros((n_particles,))
        self.out_degrees = zeros((n_particles,))
        
        for particle in self:
            particle.meta.fitness = self.fitness_fn(particle)
            particle.meta.best_fitness = particle.meta.fitness
            particle.meta.best_pos = particle.solution
            if self.best_fitness is None:
                self.best_fitness = particle.meta.fitness
                self.best_pos = particle.solution
            elif particle.meta.fitness > self.best_fitness:
                self.best_fitness = particle.meta.fitness
                self.best_pos = particle.solution
            else:
                pass
        w = zeros((n_particles, n_particles))
        a = zeros((n_particles, n_particles))
        for ix in range(len(self)):
            for ix_other in range(ix):
                dist = self.dist_fn(self[ix], self[ix_other])
                diff_fitness = self.diff_fn(self[ix].meta.fitness, self[ix_other].meta.fitness)
                if (dist < self.dist_threshold) and (abs(diff_fitness) > self.diff_eps):
                    w[ix, ix_other] = diff_fitness
                    w[ix_other, ix] = diff_fitness
                    a[ix, ix_other] = 1.
                    a[ix_other, ix] = 1.
                elif rand() < self.prob_rand_connection:
                    w[ix, ix_other] = diff_fitness
                    w[ix_other, ix] = diff_fitness
                    a[ix, ix_other] = 1.
                    a[ix_other, ix] = 1.
        
        w_normalized = w / w.sum()
        
        self.topology = w_normalized
        self.adjacency = a
        self.in_degrees = w_normalized.sum(axis=1)
        self.out_degrees = w_normalized.sum(axis=0)
