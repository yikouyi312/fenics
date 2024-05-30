import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from mesh_create import *
def ensemble_EVV_penalty(u_list, f_list, u_exact_avg, J, mesh, SPACE_STEP, TIME_STEP, T, nu, mu, epsilon, type = 2):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element space
    V = VectorFunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(mesh, "CG", 2)
    # Define Trialfunction, Testfunction
    u = TrialFunction(V)
    v = TestFunction(V)

    #initial data list
    u_prime = [Function(V)] * J
    u_prev = [Function(V)] * J
    #initial error list
    uerror = [0] * J
    avguerror = 0
    nablauerror = [0] * J
    nablaavguerror = 0
    # Initial conditions
    t = 0.0
    for i in range(J):
        u_list[i].s = t
        u_prev[i] = interpolate(u_list[i], V)
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
        EEV = Eddy_viscosity(type, mu, u_prime, V1, TIME_STEP, SPACE_STEP, J)
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx + b(u_avg, u, v) * dx + nu * a_1(u, v) * dx
               + 1 / epsilon * div(u) * div(v) * dx
               + b_2(EEV, u, v) * dx )
        A = assemble(A_u)
        # RHS for each realization
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # update the time on the source and boundary condtion
            f_list[i].s = t
            u_list[i].s = t
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            bcs = []
            bcs.append(DirichletBC(V, u_list[i], boundary))
            [bc.apply(A, B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            u_next = Function(V)
            solve(A, u_next.vector(), B)

            # L2 error
            u_fem = interpolate(u_list[i], V)
            L2error = assemble(inner(u_fem - u_next, u_fem - u_next) * dx)
            #print("u L2 norm error: " + str(L2error))
            uerror[i] += L2error * TIME_STEP
            L2error = assemble(a_1(u_fem - u_next, u_fem - u_next) * dx)
            #print("nabla u L2 norm error: " + str(L2error))
            nablauerror[i] += L2error * TIME_STEP
            #update
            u_prev[i] = u_next


    return uerror, avguerror, nablauerror, nablaavguerror



