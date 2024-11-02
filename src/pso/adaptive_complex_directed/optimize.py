from typing import Optional

from numpy import ndarray

from .population import ParticleSwarmAdaptiveComplexDirected
from .operators import EvalFitness, VelocityUpdate, PositionUpdate, UpdateTopology

def acd_pso_maximize(
    swarm: ParticleSwarmAdaptiveComplexDirected, 
    bound_lower: ndarray,
    bound_upper: ndarray,
    vel_bound_lower: Optional[ndarray],
    vel_bound_upper: Optional[ndarray],
    term_weight_k: Optional[float] = None,
    eps: float = 1e-8,
    verbosity: int = 0
) -> None:
    if term_weight_k is not None and term_weight_k < 0.:
        raise ValueError('term_weight_k must be positive')
    
    if (bound_upper < bound_lower).any():
        raise ValueError('lower bounds must all be less than or equal to upper bounds')
    
    if vel_bound_lower is None:
        vel_bound_lower = 0.5 * (bound_lower - bound_upper)
    if vel_bound_upper is None:
        vel_bound_upper = 0.5 * (bound_upper - bound_lower)
        
    if (vel_bound_upper < vel_bound_lower).any():
        raise ValueError('velocity lower bounds must all be less than or equal to velocity upper bounds')
    
    if verbosity > 0:
        print("Beginning optimization")
    
    eval_fitness = EvalFitness()
    velocity_upd = VelocityUpdate(vel_bound_lower, vel_bound_upper)
    pos_upd = PositionUpdate(bound_lower, bound_upper)
    topology_upd = UpdateTopology()
    
    pso_step = eval_fitness + velocity_upd + pos_upd + topology_upd
        
    for iter_n in range(swarm.n_iter):
        if verbosity > 1:
            print(f'  Iteration {iter_n + 1}')
        pso_step(swarm)
        swarm.crnt_iter += 1
        if term_weight_k is not None:
            if (swarm.in_degrees.max() > term_weight_k) and (swarm.out_degrees.max() < eps):
                break
        if verbosity > 1:
            print(f'    Current best fitness: {swarm.best_fitness}')
        if verbosity > 2:
            print(swarm)