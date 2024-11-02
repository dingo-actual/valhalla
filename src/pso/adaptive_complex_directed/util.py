from numpy import ndarray
from scipy.stats import powerlaw, kstest


def test_power_law(xs: ndarray, p: float = 0.85) -> bool:
    a, loc, scale = powerlaw.fit(xs)
    sample = powerlaw.rvs(a, loc=loc, scale=scale, size=xs.shape)
    
    test_result = kstest(xs, sample)
    
    return test_result.pvalue > p
