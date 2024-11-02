from typing import Callable, Optional, Sequence

from numpy import ndarray, maximum, minimum
from numpy.random import rand

from .instance import ParticleInstance
from .population import ParticleSwarmBase
from ..core.operators import PopulationOperatorBase


class EvalFitness(PopulationOperatorBase):
    def __init__(self, fitness_fn: Callable[[ParticleInstance], float]):
        super(EvalFitness, self).__init__()
        self.fitness_fn = fitness_fn
        
    def op(self, swarm: ParticleSwarmBase) -> None:
        for particle in swarm:
            fitness = self.fitness_fn(particle)
            if swarm.best_fitness is None or fitness > swarm.best_fitness:
                swarm.best_fitness = fitness
                swarm.best_pos = particle.solution
            if particle.meta.best_fitness is None or fitness > particle.meta.best_fitness:
                particle.meta.best_fitness = fitness
                particle.meta.best_pos = particle.solution
            particle.meta.fitness = fitness
                
                
class VelocityUpdate(PopulationOperatorBase):
    def __init__(self, bound_lower: ndarray, bound_upper: ndarray, threshold: Optional[float] = None):
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper
        self.threshold = threshold        
    
    def op(self, swarm: ParticleSwarmBase) -> None:
        for ix, particle in enumerate(swarm):
            if swarm.topology is not None:
                p_best = self._neighbors_p_best(ix, swarm)
            else:
                p_best = swarm.best_pos
            particle.meta.vel += self._velocity_increment(
                particle.solution,
                particle.meta.best_pos,
                p_best,
                particle.meta.c1,
                particle.meta.c2
            )
            
            particle.meta.vel = maximum(particle.meta.vel, self.bound_lower)
            particle.meta.vel = minimum(particle.meta.vel, self.bound_upper)
            
    @staticmethod
    def _velocity_increment(pos: ndarray, p_best_self: ndarray, p_best: ndarray, c1: float, c2: float) -> ndarray:
        return c1 * rand(*pos.shape) * (p_best_self - pos) + c2 * rand(*pos.shape) * (p_best - pos)
        
    def _neighbors_p_best(self, particle_ix: int, swarm: ParticleSwarmBase) -> ndarray:
        neighbors = swarm.topology[particle_ix]
        if len(neighbors) == 0:
            out = swarm.topology[particle_ix].meta.best_pos
        elif sum(neighbors) == 0:
            out = swarm.topology[particle_ix].meta.best_pos
        elif isinstance(neighbors[0], float):
            if self.threshold is None:
                out = swarm.best_pos
            else:
                fitness_best = None
                for ix, dist in enumerate(neighbors):
                    if ix == 0:
                        p_best = swarm[ix].meta.best_pos
                        fitness_best = swarm[ix].meta.best_fitness
                    elif dist < self.threshold:
                        if swarm[ix].meta.best_fitness > fitness_best:
                            p_best = swarm[ix].meta.best_pos
                            fitness_best = swarm[ix].meta.best_fitness
                    else:
                        pass
                out = p_best
        elif isinstance(neighbors[0], int):
            fitness_best = swarm[neighbors[0]].meta.best_fitness
            p_best = swarm[neighbors[0]].meta.best_pos
            for ix in neighbors[1:]:
                if swarm[ix].meta.best_fitness > fitness_best:
                    p_best = swarm[ix].meta.best_pos
                    fitness_best = swarm[ix].meta.best_fitness
                else:
                    pass
            out = p_best
        else:
            raise RuntimeError
                
        return out
      
        
class PositionUpdate(PopulationOperatorBase):
    def __init__(self, bound_lower: ndarray, bound_upper: ndarray):
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper
        
    def op(self, swarm: ParticleSwarmBase) -> None:
        for particle in swarm:
            particle.solution += particle.meta.vel
            particle.solution = maximum(particle.solution, self.bound_lower)
            particle.solution = minimum(particle.solution, self.bound_upper)
                
                
class UpdateTopologyDist(PopulationOperatorBase):
    def __init__(self, dist_fn: Callable[[ParticleInstance, ParticleInstance], float]):
        super(UpdateTopologyDist, self).__init__()
        self.dist_fn = dist_fn
        
    def op(self, swarm: ParticleSwarmBase):
        for ix in enumerate(range(len(swarm))):
            swarm.topology[ix, ix] = self.dist_fn(swarm[ix], swarm[ix])
            for ix_other in range(ix):
                swarm.topology[ix, ix_other] = self.dist_fn(swarm[ix], swarm[ix_other])
                swarm.topology[ix_other, ix] = swarm.topology[ix, ix_other]
                
                
class UpdateTopologyPredicate(PopulationOperatorBase):
    def __init__(self, predicate_fn: Callable[[ParticleInstance, ParticleInstance], bool]):
        super(UpdateTopologyPredicate, self).__init__()
        self.predicate_fn = predicate_fn
        
    def op(self, swarm: ParticleSwarmBase):
        for ix in enumerate(range(len(swarm))):
            swarm.topology[ix] = []
            if self.predicate_fn(swarm[ix], swarm[ix]):
                swarm.topology[ix].append(ix)
            for ix_other in range(ix):
                if self.predicate_fn(swarm[ix], swarm[ix_other]):
                    swarm.topology[ix].append(ix_other)
                    swarm.topology[ix_other].append(ix)


class LocalSearch(PopulationOperatorBase):
    def __init__(
        self, 
        lr: float = 1e-3,
        choose_candidates = Callable[[ParticleSwarmBase], Sequence[int]],
        gradient_fn = Callable[[ParticleInstance], ndarray]
    ):
        self.lr = lr
        self.choose_candidates = choose_candidates
        self.gradient_fn = gradient_fn

    def op(self, swarm: ParticleSwarmBase) -> None:
        for ix in self.choose_candidates(swarm):
            particle = swarm[ix]
            particle.solution -= self.lr * self.gradient_fn(particle)
