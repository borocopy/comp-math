import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import numpy as np
import computation
from definitions import f, lower_bound, upper_bound, real_value

mpl.use("pgf")

rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\\usepackage{polyglossia}')
rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
rc('text.latex', preamble=r'\usepackage{unicode-math}')
rc('text.latex', preamble=r'\setdefaultlanguage{russian}')
plt.rcParams["savefig.format"] = 'pdf'
markers = [
    '^', 'o', 's', 'p', 'x', 'd', '+', 'v', '*', '|', '<', '>', '1', '2', '3',
    '4', '8', 'h'
]
colors = [
    'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.margins(x=0, y=0)
fig, [ax1, ax2] = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)


def integrate(fn: computation.IntegralFunction, n: int) -> float:
    return fn(f, lower_bound, upper_bound, n)[0]


def generate_plot(name):
    xs = np.array([4, 6, 8, 10])
    ys_trapezoidal = np.array([integrate(computation.trapezoidal, n) for n in xs])
    ys_simpson = np.array([integrate(computation.simpson, n) for n in xs])
    ys_gauss = np.array([integrate(computation.gauss, n) for n in xs])

    ax1.set_ylim([0.99, 1.359])

    ax1.axhline(real_value, c="k", linestyle="--", alpha=0.8)
    ax1.plot(xs, ys_trapezoidal, "-ob", ms=5, alpha=0.5)
    ax1.plot(xs, ys_simpson, "*g", ms=8, alpha=0.5)
    ax1.plot(xs, ys_gauss, "or", ms=3)

    ax1.set_xlabel('Число точек разбиения $n$')
    ax1.set_ylabel('Значение аппроксимации')
    ax1.legend(["Истинное значение", "Метод трапеций", "Метод Симпсона", "Метод Гаусса"])

    ax2.axhline(real_value, c="k", linestyle="--", alpha=0.8)
    ax2.plot(xs, ys_simpson, "-og", linewidth=2, ms=5, alpha=0.5)
    ax2.plot(xs, ys_gauss, "-or", linewidth=2, ms=5, alpha=0.5)

    ax2.set_xlabel('Число точек разбиения $n$')
    ax2.set_ylabel('Значение аппроксимации')
    ax2.legend(["Истинное значение", "Метод Симпсона", "Метод Гаусса"])

    fig.suptitle("Зависимость значения аппроксимации от количества разбиений", fontsize=16)
    fig.align_ylabels([ax1, ax2])

    plt.savefig(name, bbox_inches='tight', pad_inches=0.3, dpi=300)
