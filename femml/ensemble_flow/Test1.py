import sympy as sp
from dolfin import *
import numpy as np

def force1():
    x, y, s = sp.symbols('x[0] x[1] s')
    f1 = -4 * y * (1 - pow(x, 2) - pow(y, 2))
    f2 = 4 * x * (1 - pow(x, 2) - pow(y, 2))
    return f1, f2
def pertubation():
    x, y, s = sp.symbols('x[0] x[1] s')
    f1 = sp.sin(3 * np.pi * x) * sp.sin(3 * np.pi * y)
    f2 = sp.cos(3 * np.pi * x) * sp.cos(3 * np.pi * y)
    return f1, f2

def get_list(f1, f2, perx, pery, order, J, tol, seed, issymmetry):
    np.random.seed(seed)
    if issymmetry:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(1, J)) * tol
        eps = np.vstack((eps, eps))
    else:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(2, J)) * tol
    f1_list = [None] * J
    t = 0.0
    x, y, s = sp.symbols('x[0] x[1] s')
    for i in range(J):
        f1_cur = sp.simplify(f1 + eps[0][i] * perx)
        f2_cur = sp.simplify(f2 + eps[1][i] * pery)
        f_1 = Expression((sp.printing.ccode(f1_cur), sp.printing.ccode(f2_cur)), degree=order, s=t)
        f1_list[i] = f_1
    return eps, f1_list

def get_list1(f1, f2, order, J, tol, seed, issymmetry):
    np.random.seed(seed)
    if issymmetry:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(3, J)) * tol
        eps = np.vstack((eps, eps))
    else:
        eps = np.random.normal(loc=0.0, scale=1.0, size=(6, J)) * tol
    f1_list = [None] * J
    t = 0.0
    x, y, s = sp.symbols('x[0] x[1] s')
    for i in range(J):
        perturbation1 = sp.sin((3 * np.pi + eps[1][i]) * x) * sp.sin((3 * np.pi + eps[2][i]) * y)
        perturbation2 = sp.cos((3 * np.pi + eps[4][i]) * x) * sp.cos((3 * np.pi + eps[5][i]) * y)
        f1_cur = sp.simplify(f1 + eps[0][i] * perturbation1)
        f2_cur = sp.simplify(f2 + eps[3][i] * perturbation2)
        f_1 = Expression((sp.printing.ccode(f1_cur), sp.printing.ccode(f2_cur)), degree=order, s=t)
        f1_list[i] = f_1
    return eps, f1_list