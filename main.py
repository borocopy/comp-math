import numpy as np
import computation
from tabulate import tabulate

# Definitions
# Variant 7
K = 2.4
L = 1.4
upper_bound = K + L
lower_bound = (K - L) / 2


def f(x: float) -> float:
    return (x + L) / (x**2 + x + K)


def real_integral(x: float):
    denom = np.sqrt(K - 0.25)
    return 0.5 * np.log(x**2 + x + K) + np.arctan(((x + 0.5) / denom)) * (L - 0.5) / denom


real_value = real_integral(upper_bound) - real_integral(lower_bound)

print(f"Real value of the integral: {real_value}")
steps = [4, 6, 8]
trapezoidal_data = ["Trapezoidal"] + [
    computation.trapezoidal(f, lower_bound, upper_bound, n) for n in steps
]
simpson_data = ["Simpson"] + [
    computation.simpson(f, lower_bound, upper_bound, n) for n in steps
]
gauss_data = ["Gauss"] + [
    computation.gauss(f, lower_bound, upper_bound, n) for n in steps
]

data = [trapezoidal_data, simpson_data, gauss_data]

print(
    tabulate(
        data, ["Steps count"] + steps, tablefmt="heavy_grid", floatfmt=".8f"
    )
)
