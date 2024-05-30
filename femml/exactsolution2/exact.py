import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from mesh_create import *

def exact(u_list,u_exact_avg, J, mesh, TIME_STEP, T):
    # create finite element
    V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    # create finite element space
    V = FunctionSpace(mesh, V_ele)

    # initial data list
    u_prev = [Function(V)] * J
    u = TrialFunction(V)

    N = int((T - 0.0) // TIME_STEP + 1)
    U_L2_list = np.zeros((J, N))
    U_Grad_list = np.zeros((J, N))
    AvgU_L2_list = [0.0] * N
    AvgU_Grad_list = [0.0] * N
    # Initial conditions
    t = 0.0
    count_time = 0
    while t < T:
        if t == 0:
            count_time = 0
        else:
            count_time += 1
        for i in range(J):
            u_list[i].s = t
            u_prev[i] = interpolate(u_list[i], V)
            U_L2_list[i][count_time] = assemble(inner(u_prev[i], u_prev[i]) * dx)
            U_Grad_list[i] = assemble(inner(nabla_grad(u_prev[i]), nabla_grad(u_prev[i])) * dx)
        u_exact_avg.s = t
        avg_fem = interpolate(u_exact_avg, V)
        AvgU_L2_list[count_time] = assemble(inner(avg_fem, avg_fem) * dx)
        AvgU_Grad_list[count_time] = assemble(inner(nabla_grad(avg_fem), nabla_grad(avg_fem)) * dx)
        t += TIME_STEP
    return U_L2_list, U_Grad_list, AvgU_L2_list, AvgU_Grad_list
