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
fig = plt.figure(figsize=(8, 6))


def integrate(fn: computation.IntegralFunction, n: int) -> float:
    return fn(f, lower_bound, upper_bound, n)


xs = np.array([4, 6, 8, 10])
ys_trapezoidal = np.array([integrate(computation.trapezoidal, n) for n in xs])
ys_simpson = np.array([integrate(computation.simpson, n) for n in xs])
ys_gauss = np.array([integrate(computation.gauss, n) for n in xs])

plt.ylim([0.99, 1.359])

plt.axhline(real_value, c="k", linestyle="--", alpha=0.8)
plt.plot(xs, ys_trapezoidal, "-.ob", ms=6)
plt.plot(xs, ys_simpson, "og", ms=6)
plt.plot(xs, ys_gauss, "or", ms=3)

plt.xlabel('Число точек разбиения $n$')
plt.ylabel('Значение аппроксимации')
plt.legend(["Истинное значение", "Метод трапеций", "Метод Симпсона", "Метод Гаусса"])

plt.savefig('plot.png', bbox_inches='tight', pad_inches=0.3, dpi=300)
