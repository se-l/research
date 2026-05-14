import math
import numpy as np
from math import fabs, erf, erfc
from numba import njit
from numpy._typing import NDArray


@nb.njit(fastmath=True)
def get_d1(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    return (np.log(s / k) + (r - q + iv ** 2 / 2) * t) / (iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_d2(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    return (np.log(s / k) + (r - q - iv ** 2 / 2) * t) / (iv * np.sqrt(t))
    # return get_d1(s, k, t, iv, r, q) - iv * np.sqrt(t)


@nb.njit(fastmath=True)
def get_d1_derivative(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-d1**2/2) / np.sqrt(2 * np.pi)


@nb.njit(fastmath=True)
def price_call(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return s * np.exp(-q * t) * ndtr_numba_v(d1) - k * np.exp(-r * t) * ndtr_numba_v(d2)


@nb.njit(fastmath=True)
def price_put(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return k * np.exp(-r * t) * ndtr_numba_v(-d2) - s * np.exp(-q * t) * ndtr_numba_v(-d1)


@nb.njit(fastmath=True)
def delta_call(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * ndtr_numba_v(d1)


@nb.njit(fastmath=True)
def delta_put(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * (ndtr_numba_v(d1) - 1)


@nb.njit(fastmath=True)
def gamma(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return np.exp(-q * t) * d1_drv / (s * iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_vega(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return s * np.exp(-q * t) * np.sqrt(t) * d1_drv / 100


_SQRT2 = math.sqrt(2)
NPY_SQRT1_2 = 1.0 / np.sqrt(2)


@njit(cache=True, fastmath=True)
def ndtr_numba_v(arr: NDArray[np.float64]):
    if isinstance(arr, float):
        return ndtr_numba(arr)

    res = np.zeros_like(arr)
    for i, a in enumerate(arr):
        res[i] = ndtr_numba(a)
    return res


@njit(cache=True, fastmath=True)
def ndtr_numba(a):
    if np.isnan(a):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)

    else:
        y = 0.5 * erfc(z)

        if (x > 0):
            y = 1.0 - y

    return y


def cdf(x):
    """mu=0, sigma=1"""
    return 0.5 * (1 + math.erf(x / _SQRT2))



def speed_compare_cdf_calc():
    from scipy.special import ndtr
    from scipy.stats import norm
    import timeit

    for x in np.arange(-5, 5, 0.01):
        print(f'{x}: {ndtr(x)} {ndtr(x) == norm.cdf(x) == ndtr_numba(x)}')
        # print(f'{x}: {ndtr(x)} {ndtr(x) == norm.cdf(x) == ndtr_numba(x)}')

    def cdf_slow():
        return norm.cdf(np.arange(-5, 5, 0.1))

    def cdf_fast():
        return ndtr(np.arange(-5, 5, 0.1))

    def cdf_faster():
        return ndtr_numba_v(np.arange(-5, 5, 0.1))

    n = 1_000_000
    print(timeit.timeit('cdf_slow()', setup="from __main__ import cdf_slow", number=n), 'us')
    # 30
    print(timeit.timeit('cdf_fast()', setup="from __main__ import cdf_fast", number=n), 'us')
    # 3
    print(timeit.timeit('cdf_faster()', setup="from __main__ import cdf_faster", number=n), 'us')
    # 3