from typing import Any, Callable, Dict, Optional, Tuple

from numpy import ndarray

from .population import ParticleSwarmAPSOESE
from .operators import EvalFitness, ParameterUpdate
from ..instance import ParticleInstance
from ..operators import LocalSearch

def apso_ese_maximize(
    fitness_fn: Callable[[ParticleInstance], float],
    swarm: ParticleSwarmAPSOESE, 
    n_iter: int,
    dist_fn: Callable[[ParticleInstance, ParticleInstance], float],
    pos_bound_lower: ndarray,
    pos_bound_upper: ndarray,
    vel_bound_lower: ndarray,
    vel_bound_upper: ndarray,
    sigma_min: float = 0.1,
    sigma_max: float = 1.0,
    c_bounds: Tuple[float, float] = (1.5, 2.5),
    c_sum_bounds: Tuple[float, float] = (3., 4.),
    c_inc_mult: float = 0.1,
    delta: Optional[float] = None,
    stagnation_shock_prob: float = 0.1,
    stagnation_shock_mult: float = 1.1,
    elite_perturb_dims: int = 1,
    local_search_params: Optional[Dict[str, Any]] = None,
    verbosity: int = 0,
    debug: bool = False
) -> None:
    if n_iter < 1:
        raise ValueError('number of iterations must be at least 1')
    if (pos_bound_upper < pos_bound_lower).any():
        raise ValueError('lower bounds must all be less than or equal to upper bounds')
    if (vel_bound_upper < vel_bound_lower).any():
        raise ValueError('velocity lower bounds must all be less than or equal to velocity upper bounds')
    if sigma_min > sigma_max:
        raise ValueError('sigma_min must be less than or equal to sigma_max')
    if any(map(lambda c: c < 0, c_bounds)):
        raise ValueError('both c bounds must be positive')
    if any(map(lambda c_sum: c_sum < 0, c_sum_bounds)):
        raise ValueError('both c sum bounds must be positive')
    if c_inc_mult <= 0. or c_inc_mult >= 1.:
        raise ValueError('c increments must be between 0 and 1')
    if delta is not None and delta <= 0.:
        raise ValueError('delta must be nonnegative')
    if stagnation_shock_prob < 0. or stagnation_shock_prob > 1.:
        raise ValueError('stagnation shock probability must be between 0 and 1')
    if stagnation_shock_mult <= 0:
        raise ValueError('stagnation multiplier mult be positive')
    if elite_perturb_dims < 1:
        raise ValueError('number of elite perturbation dimensions must be at least 1')
    
    
    if verbosity > 0:
        print("Beginning optimization")
    
    eval_fitness = EvalFitness(
        fitness_fn=fitness_fn,
        dist_fn=dist_fn,
        delta=delta,
        c_bounds=c_bounds,
        c_sum_bounds=c_sum_bounds,
        c_inc_mult=c_inc_mult,
        debug=debug
    )
    param_upd = ParameterUpdate(
        vel_bound_lower=vel_bound_lower,
        vel_bound_upper=vel_bound_upper,
        pos_bound_lower=pos_bound_lower,
        pos_bound_upper=pos_bound_upper,
        fitness_fn=fitness_fn,
        max_iter=n_iter,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        stagnation_shock_prob=stagnation_shock_prob,
        stagnation_shock_mult=stagnation_shock_mult,
        elite_perturb_dims=elite_perturb_dims
    )
    
    if local_search_params is not None:
        local_search = LocalSearch(**local_search_params)
        pso_step = eval_fitness + param_upd + local_search
    else:
        pso_step = eval_fitness + param_upd
    
    for iter_n in range(n_iter):
        if verbosity > 1:
            print(f'  Iteration {iter_n + 1}')
        pso_step(swarm)
        
        if verbosity > 1:
            print(f'    Best fitness: {swarm.best_fitness}')
            best_fitness = max([particle.meta.fitness for particle in swarm])
            print(f'    Current best fitness: {best_fitness}')
            
        if verbosity > 2:
            print(f'    Current state:        {swarm.crnt_state}')
            #print(swarm)
            
    if verbosity > 1:
        print('Terminating -- maximum number of iterations reached')