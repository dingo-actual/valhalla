from numpy import exp, log, ndarray, repeat
from numpy.random import rand, randn


def uniform(dim: int, lo: float = 0., hi: float = 1.) -> ndarray:
    return lo + (hi - lo) * rand(dim)


def normal(dim: int, mean: float = 0., std: float = 1.) -> ndarray:
    return mean + std * randn(dim)


def loguniform(dim: int, lo: float = 1e-10, hi: float = 1.) -> ndarray:
    lo_ = log(lo)
    hi_ = log(hi)
    return exp(uniform(dim, lo_, hi_))


def lognormal(dim: int, mean: float = 0., std: float = 1.) -> ndarray:
    return exp(normal(dim, mean, std))


def constant(dim: int, val: float) -> ndarray:
    return repeat(val, dim)