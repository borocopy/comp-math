from collections.abc import Callable
import numpy as np

_gaussian_quadrature = {
    4: [
        (-0.861136, 0.347854),
        (-0.339981, 0.652145),
        (0.339981, 0.652145),
        (0.861136, 0.347854),
    ],
    6: [
        (-0.932464, 0.171324),
        (-0.661209, 0.360761),
        (-0.238619, 0.467913),
        (0.238619, 0.467913),
        (0.661209, 0.360761),
        (0.932464, 0.171324),
    ],
    8: [
        (-0.960289, 0.101228),
        (-0.796666, 0.222381),
        (-0.525532, 0.313706),
        (-0.183434, 0.362683),
        (0.183434, 0.362683),
        (0.525532, 0.313706),
        (0.796666, 0.222381),
        (0.960289, 0.101228),
    ]
}


def trapezoidal(
    fn: Callable[[float], float], lower: float, upper: float, steps_number: int
) -> float:
    xs = np.linspace(lower, upper, steps_number)
    h = (upper - lower) / steps_number

    return h * ((fn(xs[0]) + fn(xs[-1])) / 2 + sum(fn(xs[1:-1])))


def simpson(
    fn: Callable[[float], float], lower: float, upper: float, steps_number: int
) -> float:
    xs = list(np.linspace(lower, upper, steps_number * 2 + 1))
    h = (upper - lower) / (steps_number * 2)
    odd_sum = sum([
        fn(x) for (idx, x) in enumerate(xs[1:-1]) if (idx + 1) % 2 == 1
    ])
    even_sum = sum([
        fn(x) for (idx, x) in enumerate(xs[1:-1]) if (idx + 1) % 2 == 0
    ])
    mean_value = (fn(lower) + fn(upper)) / 2

    return (2 / 3) * h * (mean_value + 2 * odd_sum + even_sum)


def gauss(
    fn: Callable[[float], float], lower: float, upper: float, steps_number: int
) -> float:
    if steps_number not in [4, 6, 8]:
        raise ValueError("Gauss quadrature only supports intervals 4, 6 and 8")
    mean_point = (lower + upper) / 2
    mid_point = (upper - lower) / 2

    def A(i):
        return _gaussian_quadrature[steps_number][i][1]

    def t(i):
        return _gaussian_quadrature[steps_number][i][0]

    return mid_point * sum([
        A(i) * fn(mean_point + t(i) * mid_point) for i in range(steps_number)
    ])
