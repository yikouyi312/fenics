from helper import *
from mesh_create import *
import time
import dijitso
def ensemble_time(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element
    V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    W_ele = V_ele * Q_ele
    # create finite element space
    V = FunctionSpace(mesh, V_ele)
    Q = FunctionSpace(mesh, Q_ele)
    W = FunctionSpace(mesh, W_ele)
    # Define Trialfunction, Testfunction
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    # initial data list
    u_prime = [Function(V)] * J
    u_prev = [Function(V)] * J
    N = int((T - 0.0) // TIME_STEP + 1)
    PRESSUREPENALTY = 1.0e-12
    time_A = 0.0
    time_b = 0.0
    time_solve = 0.0
    time_other = 0.0
    time_split = 0.0
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
        if count_time == 1:
            u_exact_avg.s = 0.0
            u_avg = interpolate(u_exact_avg, V)
            cur = time.time()
            # update u_prime
            for i in range(J):
                u_prime[i] = u_prev[i] - u_avg
            time_other += time.time() - cur
        else:
            cur = time.time()
            #cal average function
            u_avg = u_prev[0]
            for i in range(1, J):
                u_avg += u_prev[i]
            u_avg = u_avg / J
            # update u_prime
            for i in range(J):
                u_prime[i] = u_prev[i] - u_avg
            time_other += time.time() - cur
        # print("current time:" + str(t))
        # Weak form of the momentum equation
        # shared coefficient matrix
        print("t = " + str(t) + ", ensemble")
        u_exact_avg.s = t
        u_avg_fem = interpolate(u_exact_avg, V)
        cur = time.time()
        # u_avg = project(u_avg, V)
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx
               + b(u_avg, u, v) * dx
               + nu * a_1(u, v) * dx
               - p * div(v) * dx
               - q * div(u) * dx
               + p * q * PRESSUREPENALTY * dx
               )
        bcs = []
        bcs.append(DirichletBC(W.sub(0), u_avg_fem, boundary))
        # cur = time.time()
        A = assemble(A_u)
        [bc.apply(A) for bc in bcs]
        time_A += time.time() - cur
        cur = time.time()
        solver = LUSolver(A)
        #solver.parameters['reuse_factorization'] = True
        time_solve += time.time() - cur
        cur_max = 0.0
        # RHS for each realization
        for i in range(J):
            # print("current ensemble member:" + str(i))
            # update the time on the source and boundary condtion
            f_list[i].s = t #- 0.5 * TIME_STEP
            u_list[i].s = t
            p_list[i].s = t
            # get the RHS vector
            cur = time.time()
            # u_prime[i] = project(u_prime[i], V)
            # cur = time.time()
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            time_b += time.time() - cur
            # add boundary condition
            bcs = []
            bcs.append(DirichletBC(W.sub(0), u_list[i], boundary))
            cur = time.time()
            [bc.apply(B) for bc in bcs]
            time_b += time.time() - cur
            # solve A(u, p) = B
            # Define the solution fields involved
            w_next = Function(W)
            cur = time.time()
            #solve(A, w_next.vector(), B)
            solver.solve(w_next.vector(), B)
            time_solve += time.time() - cur
            cur = time.time()
            u_next, p_next = w_next.split()
            time_split += time.time() - cur
            # update
            cur = time.time()
            u_prev[i] = u_next
            time_other += time.time() - cur
        #     u_fem = interpolate(u_list[i], V)
        #     L2error = assemble(inner(u_fem - u_next, u_fem - u_next) * dx)
        #     cur_max = max(L2error, cur_max)
        # print(cur_max)

    return time_A, time_b, time_solve, time_other, time_A + time_b + time_solve + time_other + time_split, time_split


