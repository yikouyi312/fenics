import sympy as sp
from dolfin import *
import numpy as np

def forcefunction():
    x, y, s = sp.symbols('x[0] x[1] s')
    f1 = s*(-4*y*(1-pow(x, 2)-pow(y, 2)))
    f2 = s*(4*x*(1-pow(x, 2)-pow(y, 2)))
    return f1, f2
def get_list(f1, f2, order, J, tol):
    eps = np.random.randn(2, J) * tol
    f_list = [None] * J
    t = 0.0
    for i in range(J):
        f1_cur = sp.simplify((1 + eps[0][i]) * f1)
        f2_cur = sp.simplify((1 + eps[1][i]) * f2)
        f_list[i] = Expression((sp.printing.ccode(f1_cur), sp.printing.ccode(f2_cur)), degree=order,
                         s=t)
    return eps, f_list




