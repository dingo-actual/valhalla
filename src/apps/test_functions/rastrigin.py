from numpy import cos, ndarray, pi, power, sin

from ...pso import ParticleInstance


def fitness_rastrigin(x: ndarray) -> float:
    return -1. * (power(x, 2) + 10. * cos(2. * pi * x) + 10.).sum()


def gradient_fitness_rastrigin(x: ndarray) -> ndarray:
    return -1. * (2. * x - 20. * pi * sin(2. * pi * x))


def fitness_rastrigin_particle(particle: ParticleInstance):
    return fitness_rastrigin(particle.solution)


def gradient_fitness_rastrigin_particle(particle: ParticleInstance):
    return gradient_fitness_rastrigin(particle.solution)


bounds = (-5.12, 5.12)