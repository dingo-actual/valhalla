from typing import Sequence

import numpy as np
from numpy import arange, argsort, sum
from numpy.random import choice

from .population import ParticleSwarmBase


def random(k: int, swarm: ParticleSwarmBase) -> Sequence[int]:
    return choice(len(swarm), size=k, replace=False).tolist()


def random_by_fitness(k: int, swarm: ParticleSwarmBase) -> Sequence[int]:
    fitnesses = [particle.meta.fitness for particle in swarm]
    total_fitness = sum(fitnesses)
    probs = [fitness / total_fitness for fitness in fitnesses]
    
    return choice(arange(len(swarm)), size=k, p=probs, replace=False)


def elite(k: int, swarm: ParticleSwarmBase) -> Sequence[int]:
    fitnesses = [particle.meta.fitness for particle in swarm]
    
    return argsort(fitnesses)[-k:].tolist()
