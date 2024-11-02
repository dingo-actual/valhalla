from numpy import repeat
from scipy.spatial.distance import euclidean, cityblock

from .rastrigin import fitness_rastrigin_particle, gradient_fitness_rastrigin_particle, bounds as rastrigin_bounds
from .sphere import fitness_sphere_particle, gradient_sphere_particle, bounds as sphere_bounds
from ...pso import ParticleInstance, ParticleSwarmBase, pso_maximize
from ...pso.selectors import elite
from ...pso.initializers import uniform
from ...pso.adaptive_complex_directed import ParticleSwarmAdaptiveComplexDirected, acd_pso_maximize
from ...pso.ese_adaptive import ParticleSwarmAPSOESE, apso_ese_maximize


def particle_dist_euclidean(p1: ParticleInstance, p2: ParticleInstance) -> float:
    return euclidean(p1.solution, p2.solution)


def particle_dist_manhattan(p1: ParticleInstance, p2: ParticleInstance) -> float:
    return cityblock(p1.solution, p2.solution)


def main(args):
    test_name = args['test'][0]
    dim = args['dim'][0]
    n_particles = args['n_particles'][0]
    n_iter = args['n_iter'][0]
    optim = args['optimizer'][0]
    verbosity = args['verbosity'][0]
    debug = args.get('debug') is not None and args.get('debug') > 0
    
    fitness = tests[test_name]['fitness']
    gradient = tests[test_name]['gradient']
    bounds = tests[test_name]['bounds']
    bounds_vel = tuple(map(lambda x: 0.2 * x, bounds))
    
    pos_initializer = lambda n: uniform(n, *bounds)
    vel_initializer = lambda n: uniform(n, *bounds)
    c_initializer = (2., 2.)
    w_initializer = 0.9
    
    pos_bound_lower = repeat(bounds[0], dim)
    pos_bound_upper = repeat(bounds[1], dim)
    
    vel_bound_lower = repeat(bounds_vel[0], dim)
    vel_bound_upper = repeat(bounds_vel[1], dim)
    
    if optim == 'pso':
        
        topology_init = [[1. for _ in range(n_particles)] for _ in range(n_particles)]
                
        swarm = ParticleSwarmBase(
            n_particles=n_particles,
            pos_len=2*dim,
            pos_initializer=pos_initializer,
            vel_initializer=vel_initializer,
            c_initializer=c_initializer,
            topology=topology_init
        )
    
        pso_maximize(
            swarm,
            fitness_fn=fitness,
            n_iter=n_iter,
            bound_lower=pos_bound_lower,
            bound_upper=pos_bound_upper,
            vel_bound_lower=vel_bound_lower,
            vel_bound_upper=vel_bound_upper,
            topology_dist_fn=particle_dist_euclidean,
            threshold=2.,
            verbosity=verbosity
        )
    elif optim == 'acd_pso':
        swarm = ParticleSwarmAdaptiveComplexDirected(
            n_particles=n_particles,
            pos_len=dim,
            pos_initializer=pos_initializer,
            vel_initializer=vel_initializer,
            c_initializer=c_initializer,
            prob_rand_connection=0.1,
            dist_threshold=2 * dim,
            diff_eps=1e-8,
            init_inertia=w_initializer,
            final_inertia=0.4,
            n_iter=n_iter,
            fitness_fn=fitness,
            dist_fn=particle_dist_euclidean,
            diff_fn=lambda x, y: abs(x - y)
        )
        
        acd_pso_maximize(
            swarm=swarm,
            bound_lower=pos_bound_lower,
            bound_upper=pos_bound_upper,
            vel_bound_lower=vel_bound_lower,
            vel_bound_upper=vel_bound_upper,
            verbosity=verbosity
        )
    elif optim == 'ese_apso':
        swarm = ParticleSwarmAPSOESE(
            n_particles=n_particles,
            pos_len=dim,
            pos_initializer=pos_initializer,
            vel_initializer=vel_initializer,
            c_initializer=c_initializer,
            w_initializer=w_initializer
        )
        
        apso_ese_maximize(
            fitness_fn=fitness,
            swarm=swarm,
            n_iter=n_iter,
            dist_fn=particle_dist_euclidean,
            pos_bound_lower=pos_bound_lower,
            pos_bound_upper=pos_bound_upper,
            vel_bound_lower=vel_bound_lower,
            vel_bound_upper=vel_bound_upper,
            sigma_min=0.01,
            sigma_max=1.0,
            c_bounds=(1.5, 2.5),
            c_sum_bounds=(3., 4.),
            c_inc_mult=0.1,
            stagnation_shock_prob=0.1,
            stagnation_shock_mult=1.01,
            elite_perturb_dims=1,
            verbosity=verbosity,
            local_search_params={
                'lr': 1e-3,
                'choose_candidates': lambda swarm: elite(len(swarm)//4, swarm),
                'gradient_fn': gradient
            },
            debug=debug
        )
        
        print(swarm.best_pos)

tests = {
    'rast': {
        'fitness': fitness_rastrigin_particle,
        'gradient': gradient_fitness_rastrigin_particle,
        'bounds': rastrigin_bounds
    },
    'sphere': {
        'fitness': fitness_sphere_particle,
        'gradient': gradient_sphere_particle,
        'bounds': sphere_bounds
    }
}