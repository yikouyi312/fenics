import sympy as sp
from dolfin import *
import numpy as np

def force1():
    x, y, s = sp.symbols('x[0] x[1] s')
    f1 = 0.0
    f2 = 0.0
    f_list = Expression((sp.printing.ccode(f1), sp.printing.ccode(f2)), degree=4, s=0.0)
    return f_list
def flow():
    x, y, s = sp.symbols('x[0] x[1] s')
    t = 0.0
    InflowExp1 = pow(0.41, -2) * sp.sin(np.pi * s / 8.0) * 6 * y * (0.41 - y)
    InflowExp2 = 0.0
    Inflow = Expression((sp.printing.ccode(InflowExp1), sp.printing.ccode(InflowExp2)), degree=4, s=0.0)
    return Inflow

def generate_u_init(order, J, tol, seed, issymmetry):
    np.random.seed(seed)
    if issymmetry:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(3, J)) * tol
        eps = np.vstack((eps, eps))
    else:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(6, J)) * tol
    u_list = [None] * J
    x, y, s = sp.symbols('x[0] x[1] s')
    for i in range(J):
        perturbation1 = sp.sin((3 * np.pi + eps[1][i]) * x) * sp.sin((3 * np.pi + eps[2][i]) * y)
        perturbation2 = sp.cos((3 * np.pi + eps[4][i]) * x) * sp.cos((3 * np.pi + eps[5][i]) * y)
        u1_cur = sp.simplify(eps[0][i] * perturbation1)
        u2_cur = sp.simplify(eps[3][i] * perturbation2)
        u_1 = Expression((sp.printing.ccode(u1_cur), sp.printing.ccode(u2_cur)), degree=order)
        u_list[i] = u_1
    return eps, u_list
