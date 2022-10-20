from collections.abc import Callable
import numpy as np
from scipy.special import roots_legendre

IntegralFunction = Callable[[Callable[float, float], float, float, int], float]


def trapezoidal(fn: Callable[[float], float], lower: float, upper: float,
                steps_number: int) -> float:
    xs = np.linspace(lower, upper, steps_number)
    h = (upper - lower) / steps_number

    return h * ((fn(xs[0]) + fn(xs[-1])) / 2 + sum(fn(xs[1:-1])))


def simpson(fn: Callable[[float], float], lower: float, upper: float,
            steps_number: int) -> float:
    xs = list(np.linspace(lower, upper, steps_number * 2 + 1))
    h = (upper - lower) / (steps_number * 2)
    odd_sum = sum(
        [fn(x) for (idx, x) in enumerate(xs[1:-1]) if (idx + 1) % 2 == 1])
    even_sum = sum(
        [fn(x) for (idx, x) in enumerate(xs[1:-1]) if (idx + 1) % 2 == 0])
    mean_value = (fn(lower) + fn(upper)) / 2

    return (2 / 3) * h * (mean_value + 2 * odd_sum + even_sum)


def gauss(fn: Callable[[float], float], lower: float, upper: float,
          steps_number: int) -> float:
    mean_point = (lower + upper) / 2
    mid_point = (upper - lower) / 2
    legendre_vals = list(zip(*roots_legendre(steps_number)))

    def t(i: int) -> (float, float):
        return legendre_vals[i][0]

    def A(i: int) -> (float, float):
        return legendre_vals[i][1]

    return mid_point * sum([
        A(i) * fn(mean_point + t(i) * mid_point) for i in range(steps_number)
    ])
