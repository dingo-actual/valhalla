from numpy import maximum, minimum, ndarray, where, zeros_like
from numpy.random import rand

from .util import test_power_law
from ..instance import ParticleInstance
from .population import ParticleSwarmAdaptiveComplexDirected
from ...core.operators import PopulationOperatorBase


class EvalFitness(PopulationOperatorBase):
    def __init__(self):
        super(EvalFitness, self).__init__()
        
    def op(self, swarm: ParticleSwarmAdaptiveComplexDirected) -> None:
        for particle in swarm:
            fitness = swarm.fitness_fn(particle)
            if fitness > swarm.best_fitness:
                swarm.best_fitness = fitness
                swarm.best_pos = particle.solution
            if fitness > particle.meta.best_fitness:
                particle.meta.best_fitness = fitness
                particle.meta.best_pos = particle.solution
            particle.meta.fitness = fitness
                
                
class VelocityUpdate(PopulationOperatorBase):
    def __init__(self, bound_lower: ndarray, bound_upper: ndarray):
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper
    
    def op(self, swarm: ParticleSwarmAdaptiveComplexDirected) -> None:
        if True:
        # if test_power_law(swarm.in_degrees):
            w = (swarm.init_inertia - swarm.final_inertia) * (swarm.n_iter - swarm.crnt_iter) + swarm.final_inertia
            for ix, particle in enumerate(swarm):
                c2 = (1. + swarm.topology[ix, :].sum()) * particle.meta.c1
                p_best = self._neighbors_p_best(ix, swarm)
                particle.meta.vel = w * particle.meta.vel + self._velocity_increment(
                    particle.solution,
                    particle.meta.best_pos,
                    p_best,
                    particle.meta.c1,
                    c2
                )
                particle.meta.vel = maximum(particle.meta.vel, self.bound_lower)
                particle.meta.vel = minimum(particle.meta.vel, self.bound_upper)
            
    @staticmethod
    def _velocity_increment(pos: ndarray, p_best_self: ndarray, p_best: ndarray, c1: float, c2: ndarray) -> ndarray:
        return c1 * rand(*pos.shape) * (p_best_self - pos) + c2 * rand(*pos.shape) * (p_best - pos)
        
    def _neighbors_p_best(self, particle_ix: int, swarm: ParticleSwarmAdaptiveComplexDirected) -> ndarray:
        neighbors = where(swarm.adjacency[particle_ix, :] == 1.)[0]
        
        if len(neighbors) == 0:
            fitness_best = swarm[particle_ix].meta.best_fitness
            p_best = swarm[particle_ix].meta.best_pos
        else:
            fitness_best = swarm[neighbors[0]].meta.best_fitness
            p_best = swarm[neighbors[0]].meta.best_pos
        for ix in neighbors[1:]:
            if swarm[ix].meta.best_fitness > fitness_best:
                p_best = swarm[ix].meta.best_pos
                fitness_best = swarm[ix].meta.best_fitness

        out = p_best
                
        return out
      
        
class PositionUpdate(PopulationOperatorBase):
    def __init__(self, bound_lower: ndarray, bound_upper: ndarray):
        self.bound_lower = bound_lower
        self.bound_upper = bound_upper
        
    def op(self, swarm: ParticleSwarmAdaptiveComplexDirected) -> None:
        if True:
        #if test_power_law(swarm.in_degrees):
            for particle in swarm:
                particle.solution += particle.meta.vel
                particle.solution = maximum(particle.solution, self.bound_lower)
                particle.solution = minimum(particle.solution, self.bound_upper)
                
                
class UpdateTopology(PopulationOperatorBase):
    def __init__(self):
        super(UpdateTopology, self).__init__()
        
    def op(self, swarm: ParticleSwarmAdaptiveComplexDirected):
        w = zeros_like(swarm.topology)
        a = zeros_like(swarm.adjacency)
        for ix in range(len(swarm)):
            for ix_other in range(ix):
                dist = swarm.dist_fn(swarm[ix], swarm[ix_other])
                diff_fitness = swarm.diff_fn(swarm[ix].meta.fitness, swarm[ix_other].meta.fitness)
                diff_fitness_op = swarm.diff_fn(swarm[ix_other].meta.fitness, swarm[ix].meta.fitness)
                if (dist < swarm.dist_threshold) and (abs(diff_fitness) > swarm.diff_eps):
                    w[ix, ix_other] = diff_fitness
                    w[ix_other, ix] = diff_fitness_op
                    a[ix, ix_other] = 1.
                    a[ix_other, ix] = 1.
                elif rand() < swarm.prob_rand_connection:
                    w[ix, ix_other] = diff_fitness
                    w[ix_other, ix] = diff_fitness_op
                    a[ix, ix_other] = 1.
                    a[ix_other, ix] = 1.
        
        w_normalized = w / w.sum()
        
        swarm.topology = w_normalized
        swarm.adjacency = a
        swarm.in_degrees = w_normalized.sum(axis=1)
        swarm.out_degrees = w_normalized.sum(axis=0)
