from typing import Callable, Optional, Sequence, Tuple

from numpy import abs, all, argmax, argmin, min, max, mean, ndarray, exp, minimum, maximum, where
from numpy.random import choice, rand, randn, randint

from ..instance import ParticleInstance
from .population import ParticleSwarmAPSOESE
from ...core.operators import PopulationOperatorBase


class EvalFitness(PopulationOperatorBase):
    def __init__(
        self, 
        fitness_fn: Callable[[ParticleInstance], float], 
        dist_fn: Callable[[ParticleInstance, ParticleInstance], float],
        delta: Optional[float] = None,
        c_bounds: Tuple[float, float] = (1.5, 2.5),
        c_sum_bounds: Tuple[float, float] = (3., 4.),
        c_inc_mult: float = 0.1,
        elite_perturb_dims: int = 1,
        debug: bool = False
    ) -> None:
        super(EvalFitness, self).__init__()
        self.fitness_fn = fitness_fn
        self.dist_fn = dist_fn
        if delta is None:
            delta = 0.05 + 0.05 * rand()
        self.delta = delta
        self.c_lower, self.c_upper = c_bounds
        self.c_sum_lower, self.c_sum_upper = c_sum_bounds
        self.c_inc_mult = c_inc_mult
        self.elite_perturb_dims = elite_perturb_dims
        self.debug = debug
        
    def op(self, swarm: ParticleSwarmAPSOESE) -> None:
        fitnesses = []
        for ix, particle in enumerate(swarm):
            fitness = self.fitness_fn(particle)
            fitnesses.append(fitness)
            if swarm.best_fitness is None or fitness > swarm.best_fitness:
                swarm.best_fitness = fitness
                swarm.best_pos = particle.solution
                swarm.best_ix = ix
                swarm.stagnation = 0
                swarm.shock_mult = 1.
                swarm.elite_perturb_dims = self.elite_perturb_dims
            if particle.meta.best_fitness is None or fitness > particle.meta.best_fitness:
                particle.meta.best_fitness = fitness
                particle.meta.best_pos = particle.solution.copy()
            particle.meta.fitness = fitness
        
        new_best_ix = argmax(fitnesses)
        if new_best_ix != swarm.best_ix:
            swarm.best_ix = argmax(fitnesses)
            
        if self.debug:
            print(f'    Best Index: {swarm.best_ix}')
        
        mean_dists = [0. for _ in range(len(swarm))]
        for ix, particle in enumerate(swarm):
            for ix_other in range(ix):
                particle_other = swarm[ix_other]
                dist = self.dist_fn(particle, particle_other)
                mean_dists[ix] += dist
                mean_dists[ix_other] += dist

        mean_dists = [mean_dist / (len(swarm) - 1) for mean_dist in mean_dists]
        
        d_min = min(mean_dists)
        d_max = max(mean_dists)
        d_best = mean_dists[swarm.best_ix]
        
        f = (d_best - d_min) / (d_max - d_min)
        
        if self.debug:
            print(f'    d_min: {d_min}')
            print(f'    d_max: {d_max}')
            print(f'    d_best: {d_best}')
            print(f'    f: {f}')
        
        states = [
            'exploration',
            'exploitation',
            'convergence',
            'jumping_out'
        ]
        
        state_memberships = [
            self._membership_exploration(f),
            self._membership_exploitation(f),
            self._membership_convergence(f),
            self._membership_jumping_out(f)
        ]
        matching_states = [states[ix] for ix, mu in enumerate(state_memberships) if mu > 0.]
        
        if self.debug:
            print(f'    State Memberships: {state_memberships}')
        
        if not any(state == swarm.crnt_state for state in matching_states):
            swarm.crnt_state = states[argmax(state_memberships)]
            
        w_new = 1. / (1. + 1.5 * exp(-2.6 * f))
        
        if swarm.crnt_state == 'exploration':
            increment = self.delta * (self.c_inc_mult + rand() * (1. - self.c_inc_mult))
            c1_increment = increment
            c2_increment = -1. * increment
        elif swarm.crnt_state == 'exploitation':
            increment = self.delta * self.c_inc_mult * rand()
            c1_increment = increment
            c2_increment = -1. * increment
        elif swarm.crnt_state == 'convergence':
            increment = self.delta * self.c_inc_mult * rand()
            c1_increment = increment
            c2_increment = increment
        elif swarm.crnt_state == 'jumping_out':
            increment = self.delta * (self.c_inc_mult + rand() * (1. - self.c_inc_mult))
            c1_increment = -1. * increment
            c2_increment = increment
        
        for particle in swarm:
            particle.meta.w = w_new
            particle.meta.c1 += c1_increment
            particle.meta.c2 += c2_increment
            
            particle.meta.c1 = min([max([particle.meta.c1, self.c_lower]), self.c_upper])
            particle.meta.c2 = min([max([particle.meta.c2, self.c_lower]), self.c_upper])
            
            c_sum = particle.meta.c1 + particle.meta.c2
            if c_sum < self.c_sum_lower or c_sum > self.c_sum_upper:
                particle.meta.c1 = 4. * particle.meta.c1 / c_sum
                particle.meta.c2 = 4. * particle.meta.c2 / c_sum
                
        if self.debug:
            print(f'    w: {swarm[0].meta.w}')
            print(f'    c1: {swarm[0].meta.c1}')
            print(f'    c2: {swarm[0].meta.c2}')
    
    @staticmethod
    def _membership_exploration(f: float) -> float:
        out = None
        if f <= 0.4:
            out = 0.
        elif f <= 0.6:
            out = 5. * f - 2.
        elif f <= 0.7:
            out = 1.
        elif f <= 0.8:
            out = -10. * f + 8.
        elif f <= 1.0:
            out = 0.
        else:
            raise RuntimeError
        
        return out
    
    @staticmethod
    def _membership_exploitation(f: float) -> float:
        out = None
        if f <= 0.2:
            out = 0.
        elif f <= 0.3:
            out = 10. * f - 2.
        elif f <= 0.4:
            out = 1.
        elif f <= 0.6:
            out = -5. * f + 3.
        elif f <= 1.0:
            out = 0.
        else:
            raise RuntimeError
        
        return out
    
    @staticmethod
    def _membership_convergence(f: float) -> float:
        out = None
        if f <= 0.1:
            out = 1.
        elif f <= 0.3:
            out = -5. * f + 1.5
        elif f <= 1.0:
            out = 0.
        else:
            raise RuntimeError
        
        return out
    
    @staticmethod
    def _membership_jumping_out(f: float) -> float:
        out = None
        if f <= 0.7:
            out = 0.
        elif f <= 0.9:
            out = 5. * f - 3.5
        elif f <= 1.0:
            out = 1.
        else:
            raise RuntimeError
        
        return out

class ParameterUpdate(PopulationOperatorBase):
    def __init__(
        self, 
        vel_bound_lower: ndarray, 
        vel_bound_upper: ndarray, 
        pos_bound_lower: ndarray, 
        pos_bound_upper: ndarray, 
        fitness_fn: Callable[[ParticleInstance], float], 
        max_iter: int,
        sigma_min: float = 0.1, 
        sigma_max: float = 1.0,
        stagnation_shock_prob = 0.1,
        stagnation_shock_mult = 1.1,
        elite_perturb_dims = 1
    ):
        self.vel_bound_lower = vel_bound_lower
        self.vel_bound_upper = vel_bound_upper
        self.pos_bound_lower = pos_bound_lower
        self.pos_bound_upper = pos_bound_upper
        self.fitness_fn = fitness_fn
        self.max_iter = max_iter
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stagnation_shock_mult = stagnation_shock_mult
        self.stagnation_shock_prob = stagnation_shock_prob
        self.elite_perturb_dims = elite_perturb_dims
    
    def op(self, swarm: ParticleSwarmAPSOESE) -> None:
        for ix, particle in enumerate(swarm):
            if ix == swarm.best_ix:
                sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * swarm.crnt_iter / self.max_iter
                if swarm.stagnation > 0 and any(rand(swarm.stagnation) < self.stagnation_shock_prob):
                    sigma = self.sigma_max * swarm.shock_mult
                    if sigma >= max(self.pos_bound_upper - self.pos_bound_lower) / 9.:
                        sigma = max(self.pos_bound_upper - self.pos_bound_lower) / 9.
                    else:
                        swarm.shock_mult *= self.stagnation_shock_mult
                
                d = choice(len(particle.solution), size=self.elite_perturb_dims, replace=False)
                inc = (self.pos_bound_upper[d] - self.pos_bound_upper[d]) * sigma * randn(len(d))
                particle.solution[d] += inc
                fitness = self.fitness_fn(particle)
                particle.meta.fitness = fitness
                if fitness > particle.meta.best_fitness:
                    particle.meta.best_fitness = fitness
                    particle.meta.best_pos = particle.solution.copy()
                    
                if fitness > swarm.best_fitness:
                    swarm.best_fitness = fitness
                    swarm.best_pos = particle.solution.copy()
                    swarm.stagnation = 0
                    swarm.shock_mult = 1.
                    swarm.elite_perturb_dims = self.elite_perturb_dims
                else:
                    worst_ix = argmin([particle.meta.fitness for particle in swarm])
                    swarm[worst_ix] = particle.copy()
                    swarm.elite_perturb_dims += 1
                    swarm.elite_perturb_dims = min([swarm.elite_perturb_dims, len(swarm[0].solution)])
            else:
                particle.meta.vel = particle.meta.w * particle.meta.vel + self._velocity_increment(
                    particle.solution,
                    particle.meta.best_pos,
                    swarm.best_pos,
                    particle.meta.c1,
                    particle.meta.c2
                )
                particle.meta.vel = minimum(maximum(particle.meta.vel, self.vel_bound_lower), self.vel_bound_upper)
                
                particle.solution += particle.meta.vel
                particle.solution = minimum(maximum(particle.solution, self.pos_bound_lower), self.pos_bound_upper)
                
        swarm.crnt_iter += 1
        swarm.stagnation += 1

    @staticmethod
    def _velocity_increment(pos: ndarray, p_best_self: ndarray, p_best: ndarray, c1: float, c2: ndarray) -> ndarray:
        return c1 * rand(*pos.shape) * (p_best_self - pos) + c2 * rand(*pos.shape) * (p_best - pos)
