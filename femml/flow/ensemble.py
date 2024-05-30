import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from mesh_create import *
def ensemble(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element
    V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    W_ele = V_ele*Q_ele
    # create finite element space
    # V = VectorFunctionSpace(mesh, "Lagrange", 2)
    # Q = FunctionSpace(mesh, "Lagrange", 1)
    V = FunctionSpace(mesh, V_ele)
    Q = FunctionSpace(mesh, Q_ele)
    W = FunctionSpace(mesh, W_ele)
    # Define Trialfunction, Testfunction
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    #initial data list
    u_prime = [Function(V)] * J
    u_prev = [Function(V)] * J
    #initial error list
    uerror = [0] * J
    perror = [0] * J
    avguerror = 0
    nablauerror = [0] * J
    nablaperror = [0] * J
    nablaavguerror = 0
    # Initial conditions
    t = 0.0
    for i in range(J):
        u_list[i].s = t
        u_prev[i] = interpolate(u_list[i], V)
        p_list[i].s = t
    # loop
    count_time = 0
    while t < T:
        # Update current time
        t += TIME_STEP
        count_time += 1
        if count_time == 1:
            u_exact_avg.s = 0.0
            u_avg = interpolate(u_exact_avg, V)
            # update u_prime
            for i in range(J):
                u_prime[i].vector()[:] = interpolate(u_prev[i], V).vector() - interpolate(u_avg, V).vector()
        else:
            # update u_avg
            u_avg = cal_avg(u_prev, V, J)
            # update u_prime
            for i in range(J):
                u_prime[i].vector()[:] = u_prev[i].vector() - u_avg.vector()
            # calculate avg l2 norm
            u_exact_avg.s = t - TIME_STEP
            avg_fem = interpolate(u_exact_avg, V)
            L2error = assemble((avg_fem - u_avg) ** 2 * dx)
            #print("u_avg L2 norm error: " + str(L2error))
            avguerror += L2error * TIME_STEP
            L2error = assemble((nabla_grad(avg_fem) - nabla_grad(u_avg)) ** 2 * dx)
            #print("nabla u_avg L2 norm error: " + str(L2error))
            nablaavguerror += L2error * TIME_STEP
        #print("current time:" + str(t))
        # Weak form of the momentum equation
        # shared coefficient matrix
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx
               + b(u_avg, u, v) * dx
               + nu * a_1(u, v) * dx
               - p * div(v) * dx
               - q * div(u) * dx
               + p * q * 1e-10 * dx
               )
        A = assemble(A_u)
        # RHS for each realization
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # update the time on the source and boundary condtion
            f_list[i].s = t
            u_list[i].s = t
            p_list[i].s = t
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            bcs = []
            bcs.append(DirichletBC(W.sub(0), u_list[i], boundary))
            [bc.apply(A, B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            w_next = Function(W)
            solve(A, w_next.vector(), B)
            u_next, p_next = w_next.split(deepcopy=True)

            # L2 error
            u_fem = interpolate(u_list[i], V)
            p_fem = interpolate(p_list[i], Q)
            L2error = assemble(inner(u_fem - u_next, u_fem - u_next) * dx)
            #print("u L2 norm error: " + str(L2error))
            uerror[i] += L2error * TIME_STEP
            L2error = assemble(a_1(u_fem - u_next, u_fem - u_next) * dx)
            #print("nabla u L2 norm error: " + str(L2error))
            nablauerror[i] += L2error * TIME_STEP
            L2error = assemble(inner(p_fem - p_next, p_fem - p_next) * dx)
            #print("p L2 norm error: " + str(L2error))
            perror[i] += L2error * TIME_STEP
            L2error = assemble(a_1(p_fem - p_next, p_fem - p_next) * dx)
            nablaperror[i] += L2error * TIME_STEP
            #print("nabla p L2 norm error:" + str(L2error))
            #update
            u_prev[i] = u_next

    return uerror, perror, avguerror, nablauerror, nablaperror, nablaavguerror

