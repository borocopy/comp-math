from plot import generate_plot
from tabulate import tabulate
from definitions import print_table, f, lower_bound, upper_bound
import computation

print_table()

for n in [4, 6, 8, 10]:
    xs = computation.trapezoidal(f, lower_bound, upper_bound, n)[1]
    print(f"Calculations for TRAPEZOIDAL method for n = {n}")
    print(
        tabulate([(x, f(x)) for x in xs], ['x_i', 'f_i=f(x_i)'],
                 tablefmt="heavy_grid"))
for n in [4, 6, 8, 10]:
    xs = computation.simpson(f, lower_bound, upper_bound, n)[1]
    print(f"Calculations for SIMPSON method for n = {n}")
    print(
        tabulate([(x, f(x)) for x in xs], ['x_i', 'f_i=f(x_i)'],
                 tablefmt="heavy_grid"))
for n in [4, 6, 8, 10]:
    xs = computation.gauss(f, lower_bound, upper_bound, n)[1]
    print(f"Calculations for GAUSS method for n = {n}")
    print(
        tabulate([(x, f(x)) for x in xs], ['x_i', 'f_i=f(x_i)'],
                 tablefmt="heavy_grid"))

name = "plot.png"
generate_plot(name)
print(f'Generated plot {name} at the root of the project')
