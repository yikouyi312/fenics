from dolfin import *
import sympy as sp
# Numerical test (Navier-Stokes equation) with exact solution
# It is hard to calculate force function by hand
# Use symbolic mathematical calculate to simplify it.
# define exact solution
# Example:
# u1 = pi*sin(t)*sin(2pi*y)*sin(pi*x)^2, u2 = -pi*sin(t)*sin(2pi*x)*sin(pi*y)^2
# p = sin(t)*cos(pi*x)*sin(pi*y)
# Input:
# nu, order: element degree
# Output:
# u_exact, p_exact, f(force function)

def NSEexactsolution(nu, order):
    t = 0.0
    # Exact solution
    x, y, s = sp.symbols('x[0] x[1] s')
    u1_exact = pi * sp.sin(s) * sp.sin(2 * pi * y) * (sp.sin(pi * x)) ** 2  # Exact velocity, x-component
    u2_exact = -pi * sp.sin(s) * sp.sin(2 * pi * x) * (sp.sin(pi * y)) ** 2  # Exact velocity, y-component
    p_exact = sp.sin(s) * sp.cos(pi * x) * sp.sin(pi * y)  # Exact pressure
    f1 = u1_exact.diff(s, 1) + u1_exact * u1_exact.diff(x, 1) + u2_exact * u1_exact.diff(y, 1) \
        - nu * sum(u1_exact.diff(xi, 2) for xi in (x, y)) + p_exact.diff(x, 1)  # Forcing, x-component
    f2 = u2_exact.diff(s, 1) + u1_exact * u2_exact.diff(x, 1) + u2_exact * u2_exact.diff(y, 1) \
        - nu * sum(u2_exact.diff(xi, 2) for xi in (x, y)) + p_exact.diff(y, 1)  # Forcing, y-component
    u1_exact = sp.simplify(u1_exact)  # Velocity simplification, x-component
    u2_exact = sp.simplify(u2_exact)  # Velocity simplification, y-component
    p_exact = sp.simplify(p_exact)  # Pressure simplification
    f1 = sp.simplify(f1)  # Forcing simplification
    f2 = sp.simplify(f2)
    u_exact = Expression((sp.printing.ccode(u1_exact), sp.printing.ccode(u2_exact)), degree=order,
                         s=t)  # Exact velocity expression
    p_exact = Expression(sp.printing.ccode(p_exact), degree=order, s=t)  # Exact pressure expression
    f = Expression((sp.printing.ccode(f1), sp.printing.ccode(f2)), degree=order, s=t)
    return u_exact, p_exact, f