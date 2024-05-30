import sympy as sp
from dolfin import *
import numpy as np


# define exact solution
def exactsolution1(m, n, alpha):
    """
    :return: symbolic analytic function
    """
    # Exact solution
    x, y, s = sp.symbols('x[0] x[1] s')
    u1_exact = - sp.sin(2 * s) * sp.cos(m * x) * sp.sin(n * y)  # u1
    u2_exact = sp.sin(2 * s) * sp.sin(m * x) * sp.cos(n * y)  # u2
    p_exact = -1.0 / 4.0 * (sp.cos(2 * m * x) + sp.cos(2 * n * y)) * (sp.sin(2 * s) ** 2)  # pressure
    return u1_exact, u2_exact, p_exact


def get_exact1(u1_exact, u2_exact, p_exact, nu, order, J, epsilon, seed):
    """
        Create J Navier-Stokes equations with slightly different initial conditions and body forces
        u = u_exact*(1+epsilon)
        output: u_list: list of function
    """
    np.random.seed(seed)
    # unsymmery
    # eps = np.random.normal(loc=0.0, scale=1.0, size=(3, J)) * epsilon
    # symmery
    eps = np.random.normal(loc=0.0, scale=1.0, size=(1, J)) * epsilon
    eps = np.vstack((eps, eps, eps))

    t = 0.0
    x, y, s = sp.symbols('x[0] x[1] s')
    u_list, p_list, f_list = [None] * J, [None] * J, [None] * J
    # calculate average result
    u1_exact_avg = 0
    u2_exact_avg = 0
    for i in range(J):
        u1_exact1 = (1 + eps[0][i]) * u1_exact
        u2_exact1 = (1 + eps[1][i]) * u2_exact
        p_exact1 = ((1 + eps[2][i]) ** 2) * p_exact
        u1_exact1 = sp.simplify(u1_exact1)  # Velocity simplification, x-component
        u2_exact1 = sp.simplify(u2_exact1)  # Velocity simplification, y-component
        u1_exact_avg += u1_exact1
        u2_exact_avg += u2_exact1
        f1 = u1_exact1.diff(s, 1) + u1_exact1 * u1_exact1.diff(x, 1) + u2_exact1 * u1_exact1.diff(y, 1) \
             - nu * sum(u1_exact1.diff(xi, 2) for xi in (x, y)) + p_exact1.diff(x, 1)  # Forcing, x-component
        f2 = u2_exact1.diff(s, 1) + u1_exact1 * u2_exact1.diff(x, 1) + u2_exact1 * u2_exact1.diff(y, 1) \
             - nu * sum(u2_exact1.diff(xi, 2) for xi in (x, y)) + p_exact1.diff(y, 1)  # Forcing, y-component
        # f1 = (2 * sp.cos(2 * s) + 2 * nu * sp.sin(2 * s)) * (- sp.cos(1 * x) * sp.sin(1 * y)) * (1 + eps[0][i])
        # f2 = (2 * sp.cos(2 * s) + 2 * nu * sp.sin(2 * s)) * (sp.sin(1 * x) * sp.cos(1 * y)) * (1 + eps[1][i])
        f1_1 = sp.simplify(f1)  # Forcing simplification
        f2_1 = sp.simplify(f2)
        u_exact1 = Expression((sp.printing.ccode(u1_exact1), sp.printing.ccode(u2_exact1)), degree=order,
                              s=t)  # Exact velocity expression
        p_exact1 = Expression(sp.printing.ccode(p_exact1), degree=order, s=t)  # Exact pressure expression
        f_1 = Expression((sp.printing.ccode(f1_1), sp.printing.ccode(f2_1)), degree=order, s=t)
        u_list[i] = u_exact1
        p_list[i] = p_exact1
        f_list[i] = f_1

    u1_exact_avg = sp.simplify(u1_exact_avg / J)
    u2_exact_avg = sp.simplify(u2_exact_avg / J)
    u_exact_avg = Expression((sp.printing.ccode(u1_exact_avg), sp.printing.ccode(u2_exact_avg)), degree=order,
                             s=t)
    return eps, u_list, p_list, f_list, u_exact_avg


def get_exact_random(u1_exact, u2_exact, p_exact, nu, order, J, epsilon, seed):
    """
        Create J Navier-Stokes equations with slightly different initial conditions and body forces
        u = u_exact*(1+epsilon)
        output: u_list: list of function
    """
    np.random.seed(seed)
    x, y, s = sp.symbols('x[0] x[1] s')
    # unsymmery
    # eps = np.random.normal(loc=0.0, scale=1.0, size=(3, J)) * epsilon
    # symmery
    eps = np.random.normal(loc=0.0, scale=1.0, size=(1, J)) * epsilon
    eps = np.vstack((eps, eps, eps))

    t = 0.0
    x, y, s = sp.symbols('x[0] x[1] s')
    u_list, p_list, f_list = [None] * J, [None] * J, [None] * J
    # calculate average result
    u1_exact_avg = 0
    u2_exact_avg = 0
    for i in range(J):
        perturbation1 = sp.cos(1.0 / (20.0 * eps[0][i]) * (s * y)) * sp.sin(1.0 / (20.0 * eps[0][i]) * s * x) * eps[0][i]
        perturbation2 = -sp.cos(1.0 / (20.0 * eps[1][i]) * (s * x)) * sp.sin(1.0 / (20.0 * eps[1][i]) * s * y) * eps[1][i]
        # perturbation1 = sp.cos(2 * eps[0][i] * x) * sp.sin(3.0 * eps[0][i] * y)
        # perturbation2 = sp.sin(3.0 * eps[1][i] * x) * sp.cos(2 * eps[1][i] * y)
        u1_exact1 = u1_exact + perturbation1
        u2_exact1 = u2_exact + perturbation2
        p_exact1 = p_exact
        u1_exact1 = sp.simplify(u1_exact1)  # Velocity simplification, x-component
        u2_exact1 = sp.simplify(u2_exact1)  # Velocity simplification, y-component
        u1_exact_avg += u1_exact1
        u2_exact_avg += u2_exact1
        f1 = u1_exact1.diff(s, 1) + u1_exact1 * u1_exact1.diff(x, 1) + u2_exact1 * u1_exact1.diff(y, 1) \
             - nu * sum(u1_exact1.diff(xi, 2) for xi in (x, y)) + p_exact1.diff(x, 1)  # Forcing, x-component
        f2 = u2_exact1.diff(s, 1) + u1_exact1 * u2_exact1.diff(x, 1) + u2_exact1 * u2_exact1.diff(y, 1) \
             - nu * sum(u2_exact1.diff(xi, 2) for xi in (x, y)) + p_exact1.diff(y, 1)  # Forcing, y-component
        # f1 = (2 * sp.cos(2 * s) + 2 * nu * sp.sin(2 * s)) * (- sp.cos(1 * x) * sp.sin(1 * y)) * (1 + eps[0][i])
        # f2 = (2 * sp.cos(2 * s) + 2 * nu * sp.sin(2 * s)) * (sp.sin(1 * x) * sp.cos(1 * y)) * (1 + eps[1][i])
        f1_1 = sp.simplify(f1)  # Forcing simplification
        f2_1 = sp.simplify(f2)
        u_exact1 = Expression((sp.printing.ccode(u1_exact1), sp.printing.ccode(u2_exact1)), degree=order,
                              s=t)  # Exact velocity expression
        p_exact1 = Expression(sp.printing.ccode(p_exact1), degree=order, s=t)  # Exact pressure expression
        f_1 = Expression((sp.printing.ccode(f1_1), sp.printing.ccode(f2_1)), degree=order, s=t)
        u_list[i] = u_exact1
        p_list[i] = p_exact1
        f_list[i] = f_1

    u1_exact_avg = sp.simplify(u1_exact_avg / J)
    u2_exact_avg = sp.simplify(u2_exact_avg / J)
    u_exact_avg = Expression((sp.printing.ccode(u1_exact_avg), sp.printing.ccode(u2_exact_avg)), degree=order,
                             s=t)
    return eps, u_list, p_list, f_list, u_exact_avg
