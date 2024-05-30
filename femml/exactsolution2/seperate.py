import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from mesh_create import *
def seperate(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu):
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

    #error list
    # initial error list
    N = int((T - 0.0) // TIME_STEP + 1)
    Uerror_L2_list = np.zeros((J, N))
    GradUerror_L2_list = np.zeros((J, N))
    U_L2_list = np.zeros((J, N))
    U_Grad_list = np.zeros((J, N))
    U_div_L2_list = np.zeros((J, N))

    AvgUerror_L2_list = [0.0] * N
    AvgGradUerror_L2_list = [0.0] * N
    AvgU_L2_list = [0.0] * N
    AvgU_Grad_list = [0.0] * N
    AvgU_div_L2_list = [0.0] * N

    PRESSUREPENALTY = 1e-12
    #initial point value
    u_x = np.zeros((J, N))
    u_y = np.zeros((J, N))
    u_x_true = np.zeros((J, N))
    u_y_true = np.zeros((J, N))

    avg_u_x = [0.0] * N
    avg_u_y = [0.0] * N
    avg_u_x_true = [0.0] * N
    avg_u_y_true = [0.0] * N

    # Initial conditions
    t = 0.0
    for i in range(J):
        u_list[i].s = t
        u_prev[i] = interpolate(u_list[i], V)
        p_list[i].s = t
    # loop
    count_time = 0
    while count_time < N - 1:
        # Update current time
        t += TIME_STEP
        count_time += 1
        if count_time > 1:
            # update u_avg
            u_avg = cal_avg(u_prev, V, J)
            # get value at specific point of average solution
            avg_u_x[count_time], avg_u_y[count_time] = extract_value(u_avg, V)
            # calculate avg l2 norm
            u_exact_avg.s = t - TIME_STEP
            avg_fem = interpolate(u_exact_avg, V)
            # get value at specific point of average solution(true)
            avg_u_x_true[count_time], avg_u_y_true[count_time] = extract_value(avg_fem, V)
            # Avg U norm, error norm
            L2error, gradL2error, L2u, Gradu = cal_norm(avg_fem, u_avg)
            AvgUerror_L2 += L2error * TIME_STEP
            AvgGradUerror_L2 += gradL2error * TIME_STEP
            AvgUerror_inf = max(AvgUerror_inf, L2error)
            AvgU_L2 += L2u * TIME_STEP
            AvgU_Grad += Gradu * TIME_STEP
            AvgU_inf = max(AvgU_inf, L2u)
            # add to error list
            AvgUerror_L2_list[count_time] = np.sqrt(L2error)
            AvgGradUerror_L2_list[count_time] = np.sqrt(gradL2error)
            AvgU_L2_list[count_time] = np.sqrt(L2u)
            AvgU_Grad_list[count_time] = np.sqrt(Gradu)
            # Avg div u
            divunorm = cal_div(avg_fem)
            AvgU_div_L2 += divunorm * TIME_STEP
            AvgU_div_inf = max(AvgU_div_inf, divunorm)
            #add to div list
            AvgU_div_L2_list[count_time] = np.sqrt(divunorm)
        #print("current time:" + str(t))
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # Weak form of the momentum equation
            A_u = (1.0 / TIME_STEP * inner(u, v) * dx
                   + b(u_prev[i], u, v) * dx
                   + nu * a_1(u, v) * dx
                   - p * div(v) * dx
                   - q * div(u) * dx
                   + p * q * PRESSUREPENALTY * dx
                   )
            A = assemble(A_u)
            # update the time on the source and boundary condition
            f_list[i].s = t #- 0.5 * TIME_STEP
            u_list[i].s = t
            p_list[i].s = t
            # RHS for each realization
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx)
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
            # update
            u_prev[i] = u_next
            """
            calculate error
            """
            # get value at specific point of solution
            u_x[i][count_time], u_y[i][count_time] = extract_value(u_next, V)
            u_x_true[i][count_time], u_y_true[i][count_time] = extract_value(u_list[i], V)
            # U norm, error norm
            u_fem = interpolate(u_list[i], V)
            L2error, gradL2error, L2u, Gradu = cal_norm(u_fem, u_next)
            Uerror_L2[i] += L2error * TIME_STEP
            GradUerror_L2[i] += gradL2error * TIME_STEP
            Uerror_inf[i] = max(Uerror_inf[i], L2error)
            U_L2[i] += L2u * TIME_STEP
            U_Grad[i] += Gradu * TIME_STEP
            U_inf[i] = max(U_inf[i], L2u)
            # add to error list
            Uerror_L2_list[i][count_time] = np.sqrt(L2error)
            GradUerror_L2_list[i][count_time] = np.sqrt(gradL2error)
            U_L2_list[i][count_time] = np.sqrt(L2u)
            U_Grad_list[i][count_time] = np.sqrt(Gradu)
            # div U norm
            divunorm = cal_div(u_fem)
            U_div_L2[i] += divunorm * TIME_STEP
            U_div_inf[i] = max(U_div_inf[i], divunorm)
            # add to error list
            U_div_L2_list[i][count_time] = np.sqrt(float(divunorm))
            # p norm, error norm
            p_fem = interpolate(p_list[i], Q)
            L2error, gradL2error, L2p, Gradp = cal_norm(p_fem, p_next)
            Perror_L2[i] += L2error * TIME_STEP
            GradPerror_L2[i] = gradL2error * TIME_STEP
            Perror_inf[i] = max(Perror_inf[i], np.sqrt(L2error))
            P_L2[i] += L2p * TIME_STEP
            P_Grad[i] += Gradp * TIME_STEP
            P_inf[i] = max(P_inf[i], np.sqrt(L2u))

    return Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
           Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
           AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf,\
           Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
           AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list,\
           u_x, u_y, avg_u_x, avg_u_y, u_x_true, u_y_true, avg_u_x_true, avg_u_y_true


