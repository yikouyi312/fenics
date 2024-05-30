import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from mesh_create import *
def seperate_penalty(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu, epsilon):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element
    V = VectorFunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(mesh, "CG", 2)
    Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    Q = FunctionSpace(mesh, Q_ele)
    # Define Trialfunction, Testfunction
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    #initial data list
    u_prev = [Function(V)] * J
    #initial error list
    Uerror_L2 = [0.0] * J
    GradUerror_L2 = [0.0] * J
    Uerror_inf = [0.0] * J
    U_L2 = [0.0] * J
    U_Grad = [0.0] * J
    U_inf = [0.0] * J
    U_div_L2 = [0.0] * J
    U_div_inf = [0.0] * J

    Perror_L2 = [0.0] * J
    GradPerror_L2 = [0.0] * J
    Perror_inf = [0.0] * J
    P_L2 = [0.0] * J
    P_Grad = [0.0] * J
    P_inf = [0.0] * J

    AvgUerror_L2 = 0.0
    AvgGradUerror_L2 = 0.0
    AvgUerror_inf = 0.0
    AvgU_L2 = 0.0
    AvgU_Grad = 0.0
    AvgU_inf = 0.0
    AvgU_div_L2 = 0.0
    AvgU_div_inf = 0.0

    PRESSUREPENALTY = 1e-12
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
        if count_time > 1:
            # update u_avg
            u_avg = cal_avg(u_prev, V, J)
            # calculate avg l2 norm
            u_exact_avg.s = t - TIME_STEP
            avg_fem = interpolate(u_exact_avg, V)
            # Avg U norm, error norm
            L2error, gradL2error, L2u, Gradu = cal_norm(avg_fem, u_avg)
            AvgUerror_L2 += L2error * TIME_STEP
            AvgGradUerror_L2 += gradL2error * TIME_STEP
            AvgUerror_inf = max(AvgUerror_inf, L2error)
            AvgU_L2 += L2u * TIME_STEP
            AvgU_Grad += Gradu * TIME_STEP
            AvgU_inf = max(AvgU_inf, L2u)
            # Avg div u
            divunorm = cal_div(avg_fem)
            AvgU_div_L2 += divunorm * TIME_STEP
            AvgU_div_inf = max(AvgU_div_inf, divunorm)
        #print("current time:" + str(t))
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # Weak form of the momentum equation
            A_u = (1.0 / TIME_STEP * inner(u, v) * dx
                   + b(u_prev[i], u, v) * dx
                   + nu * a_1(u, v) * dx
                   + 1.0 / epsilon * div(u) * div(v) * dx
                   )
            A = assemble(A_u)
            # update the time on the source and boundary condition
            f_list[i].s = t - 0.5 * TIME_STEP
            u_list[i].s = t
            p_list[i].s = t
            # RHS for each realization
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            bcs = []
            bcs.append(DirichletBC(V, u_list[i], boundary))
            [bc.apply(A, B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            u_next = Function(V)
            solve(A, u_next.vector(), B)
            A_p = (Constant(epsilon) * dot(grad(p), grad(q)) * dx)
            A1 = assemble(A_p)
            p_next = Function(Q)
            B_p = (div(u_next) * div(nabla_grad(q)) * dx)
            # B_p = -(inner(nabla_grad(div(u_next)), nabla_grad(q)) * dx)
            B1 = assemble(B_p)
            solve(A1, p_next.vector(), B1)

            # U norm, error norm
            u_fem = interpolate(u_list[i], V)
            L2error, gradL2error, L2u, Gradu = cal_norm(u_fem, u_next)
            Uerror_L2[i] += L2error * TIME_STEP
            GradUerror_L2[i] += gradL2error * TIME_STEP
            Uerror_inf[i] = max(Uerror_inf[i], L2error)
            U_L2[i] += L2u * TIME_STEP
            U_Grad[i] += Gradu * TIME_STEP
            U_inf[i] = max(U_inf[i], L2u)

            # div U norm
            divunorm = cal_div(u_fem)
            U_div_L2[i] += divunorm * TIME_STEP
            U_div_inf[i] = max(U_div_inf[i], divunorm)

            # p norm, error norm
            p_fem = interpolate(p_list[i], Q)
            L2error, gradL2error, L2p, Gradp = cal_norm(p_fem, p_next)
            Perror_L2[i] += L2error * TIME_STEP
            GradPerror_L2[i] = gradL2error * TIME_STEP
            Perror_inf[i] = max(Perror_inf[i], np.sqrt(L2error))
            P_L2[i] += L2p * TIME_STEP
            P_Grad[i] += Gradp * TIME_STEP
            P_inf[i] = max(P_inf[i], np.sqrt(L2u))
            #update
            u_prev[i] = u_next
    return Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
           Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
           AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf

