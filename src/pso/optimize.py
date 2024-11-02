from typing import Callable, Optional

from numpy import ndarray

from .population import ParticleSwarmBase
from .instance import ParticleInstance
from .operators import EvalFitness, VelocityUpdate, PositionUpdate, UpdateTopologyDist, UpdateTopologyPredicate

def pso_maximize(
    swarm: ParticleSwarmBase, 
    fitness_fn: Callable[[ParticleInstance], float],
    n_iter: int,
    bound_lower: ndarray,
    bound_upper: ndarray,
    vel_bound_lower: Optional[ndarray],
    vel_bound_upper: Optional[ndarray],
    term_cond_fn: Optional[Callable[[ParticleSwarmBase], bool]] = None,
    topology_dist_fn: Optional[Callable[[ParticleInstance, ParticleInstance], float]] = None,
    topology_predicate_fn: Optional[Callable[[ParticleInstance, ParticleInstance], bool]] = None,
    threshold: Optional[float] = None,
    verbosity: int = 0
) -> None:
    if verbosity > 0:
        print("Beginning optimization")
    
    if n_iter < 1:
        raise ValueError('n_iter must be at least 1')
    if (bound_upper < bound_lower).any():
        raise ValueError('all lower bounds must be less than or equal to upper bounds')
    
    if vel_bound_lower is None:
        vel_bound_lower = 0.5 * (bound_lower - bound_upper)
    if vel_bound_upper is None:
        vel_bound_upper = 0.5 * (bound_upper - bound_lower)
        
    if (vel_bound_upper < vel_bound_upper).any():
        raise ValueError('all velocity lower bounds must be less than or equal to upper bounds')
    
    if topology_dist_fn is not None:
        if topology_predicate_fn is not None:
            raise ValueError('at most one of topology_dist_fn and topology_predicate_fn and be specified')
        if threshold is None:
            raise ValueError('topology_dist_fn requires threshold to be specified')
        topology_upd = UpdateTopologyDist(topology_dist_fn)
    elif topology_predicate_fn is not None:
        topology_upd = UpdateTopologyPredicate(topology_predicate_fn)
    else:
        topology_upd = None
    
    eval_fitness = EvalFitness(fitness_fn)
    velocity_upd = VelocityUpdate(vel_bound_lower, vel_bound_upper, threshold)
    pos_upd = PositionUpdate(bound_lower, bound_upper)
    
    if topology_upd is None:
        pso_step = eval_fitness + velocity_upd + pos_upd
    else:
        pso_step = eval_fitness + velocity_upd + pos_upd + topology_upd
        
    for iter in range(n_iter):
        if verbosity > 1:
            print(f'  Iteration {iter + 1}')
        pso_step(swarm)
        if term_cond_fn is not None and term_cond_fn(swarm):
            break
        if verbosity > 1:
            print(f'    Current best fitness: {swarm.best_fitness}')
        if verbosity > 2:
            print(swarm)

    eval_fitness(swarm)

    if verbosity > 0:
        print(f'Optimization Complete\n  Final best fitness: {swarm.best_fitness}')