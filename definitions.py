import numpy as np
import computation
from tabulate import tabulate

# Variant 7
# Definitions
K = 2.4
L = 1.4
upper_bound = K + L
lower_bound = (K - L) / 2


def f(x: float) -> float:
    return (x + L) / (x**2 + x + K)


def real_integral(x: float) -> float:
    denom = np.sqrt(K - 0.25)
    long_computation = np.arctan(((x + 0.5) / denom)) * (L - 0.5) / denom
    return 0.5 * np.log(x**2 + x + K) + long_computation


# Compute value of analytical solution of the integral
real_value = real_integral(upper_bound) - real_integral(lower_bound)


def test_method(fn: computation.IntegralFunction) -> list[float]:
    steps = [4, 6, 8, 10]
    values = [fn(f, lower_bound, upper_bound, n)[0] for n in steps]
    error = np.abs(values[-1] - real_value)
    abs_diffs = np.abs(np.array(values) - real_value)
    print(abs_diffs)

    return values + [error]


def print_table():
    print(f"Real value of the integral: {real_value}")
    steps = [4, 6, 8, 10]
    trapezoidal_data = ["Trapezoidal"] + test_method(computation.trapezoidal)
    simpson_data = ["Simpson"] + test_method(computation.simpson)
    gauss_data = ["Gauss"] + test_method(computation.gauss)
    data = [trapezoidal_data, simpson_data, gauss_data]

    print(
        tabulate(data,
                 ["Steps count"] + steps + ["Absolute error"],
                 tablefmt="heavy_grid",
                 floatfmt=".12f"))
