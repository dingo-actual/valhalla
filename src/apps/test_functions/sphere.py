from numpy import ndarray, power

from ...pso import ParticleInstance


def fitness_sphere(x: ndarray) -> float:
    return -1. * power(x, 2).sum()


def fitness_sphere_particle(particle: ParticleInstance):
    return fitness_sphere(particle.solution)


def gradient_sphere(x: ndarray) -> ndarray:
    return -2. * x


def gradient_sphere_particle(particle: ParticleInstance):
    return gradient_sphere(particle.solution)


bounds = (-10., 10.)