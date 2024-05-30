import numpy as np

from helper import *
from mesh_create import *
def ensemble(u_init, f_list, J, mesh, mesh_size, t_init, TIME_STEP, T, nu):
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
    for i in range(J):
        u_prev[i] = project(u_init[i], V)
    N = int((T - t_init) / TIME_STEP + 1)
    PRESSUREPENALTY = 1.0e-12
    ldcount = int((T - t_init) / TIME_STEP) + 1
    drag_arr = np.zeros((J, ldcount))
    lift_arr = np.zeros((J, ldcount))
    jj = 0
    Energy = np.zeros((J, N))
    Enstropy = np.zeros((J, N))
    Divu = np.zeros((J, N))
    Energy_avg = [0.0] * N
    Enstropy_avg = [0.0] * N
    Divu_avg = [0.0] * N
    # calculate stability condition
    diffu = np.zeros((J, N))
    diffu_norm = np.zeros((J, N))
    # Initial conditions
    t = t_init
    for i in range(J):
        # load reference results
        Energy[i][0] = energy(u_prev[i])
        Enstropy[i][0] = enstrophy(u_prev[i])
        Divu[i][0] = cal_div(u_prev[i])
    u_avg = cal_avg(u_prev, V, J)
    Energy_avg[0] = energy(u_avg)
    Enstropy_avg[0] = enstrophy(u_avg)
    Divu_avg[0] = cal_div(u_avg)
    # lift and drag
    vd, vl, circle = lift_drag(V)
    # file
    # ufile = [File("./resultsV/ensemble_velocity" + str(int(i)) + ".pvd") for i in range(J)]
    # pfile = [File("./resultsP/ensemble_pressure" + str(int(i)) + ".pvd") for i in range(J)]
    frameRat = int(1.0 / TIME_STEP)
    # loop
    count_time = 0
    while count_time < N - 1:
        # Update current time
        t += TIME_STEP
        count_time += 1
        # update u_prime
        for i in range(J):
            u_prime[i] = u_prev[i] - u_avg
            diffu[i][count_time-1] = cal_diffu(u_prime[i], V)
            diffu_norm[i][count_time-1] = cal_diff_divu(u_prime[i])
        # Weak form of the momentum equation
        # shared coefficient matrix
        print("t = " + str(t) + ", ensemble")
        print(diffu[i][count_time - 1])
        print(diffu_norm[i][count_time - 1])
        # u_avg = project(u_avg, V)
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx
               + b(u_avg, u, v) * dx
               + nu * a_1(u, v) * dx
               - p * div(v) * dx
               - q * div(u) * dx
               + p * q * PRESSUREPENALTY * dx
               )
        # cur = time.time()
        A = assemble(A_u)
        bcs = boundary_condition(W.sub(0), W.sub(1), t)
        [bc.apply(A) for bc in bcs]
        solver = LUSolver(A)
        # solver.parameters["same_nonzero_pattern"] = True
        # RHS for each realization
        for i in range(J):
            # print("current ensemble member:" + str(i))
            # update the time on the source and boundary condition
            f_list.s = t
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list, v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            [bc.apply(B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            w_next = Function(W)
            #solve(A, w_next.vector(), B)
            solver.solve(w_next.vector(), B)
            u_next, p_next = w_next.split()
            # update
            u_prev[i] = u_next
            # save data
            # save initial result at t_init
            Enstropy[i][count_time] = enstrophy(u_next)
            Energy[i][count_time] = energy(u_next)
            Divu[i][count_time] = cal_div(u_next)
            # calculate drag_coefficient after t_init
            if count_time >= int(t_init / TIME_STEP):
                drag_arr[i][jj], lift_arr[i][jj] = calculate_lift_drag(u_next, p_next, nu, vd, vl)
                if i == J - 1:
                    jj += 1
            if count_time % frameRat == 0 and t <= 4.5:
                # ufile[i] << (u_next, t)
                # pfile[i] << (p_next, t)
                c = plot(u_next)
                plt.colorbar(c)
                plt.savefig('./fig/plotu/ensemble' + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
                speed = cal_speed(u_next)
                c = plot(project(speed, Q))
                plt.colorbar(c)
                plt.savefig('./fig/speed/ensemble' + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
        u_avg = cal_avg(u_prev, V, J)
        if count_time % frameRat == 0:
            c = plot(project(u_avg, V))
            plt.colorbar(c)
            plt.savefig('./fig/plotu_avg/ensemble' + '_t' + str(int(t)) + '.png')
            plt.close('all')
            speed = cal_speed(project(u_avg, V))
            c = plot(project(speed, Q))
            plt.colorbar(c)
            plt.savefig('./fig/speed_avg/ensemble' + '_t' + str(int(t)) + '.png')
            plt.close('all')
        Energy_avg[count_time] = energy(u_avg)
        Enstropy_avg[count_time] = enstrophy(u_avg)
        Divu_avg[count_time] = cal_div(u_avg)
    x = np.linspace(t_init, T, int((T - t_init) / TIME_STEP) + 1)
    for i in range(J):
        np.savetxt('./output/diff/ensemble' + str(i) + '.txt', diffu[i])
        np.savetxt('./output/diff_norm/ensemble' + str(i) + '.txt', diffu_norm[i])
        plt.figure(0)
        plt.plot(x, diffu[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('max u prime')
        plt.savefig('./fig/diff/ensemble' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, diffu_norm[i])
        plt.xlabel('t')
        plt.ylabel('u prime norm')
        plt.xlim((t_init, T))
        plt.savefig('./fig/diff_norm/ensemble' + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/lift/ensemble' + str(i) + '.txt', lift_arr[i])
        np.savetxt('./output/drag/ensemble' + str(i) + '.txt', drag_arr[i])
        plt.figure(0)
        plt.plot(x, lift_arr[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('lift')
        plt.savefig('./fig/lift/ensemble' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, drag_arr[i])
        plt.xlabel('t')
        plt.ylabel('drag')
        plt.xlim((t_init, T))
        plt.savefig('./fig/drag/ensemble' + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/energy/ensemble' + str(i) + '.txt', Energy[i])
        np.savetxt('./output/enstropy/ensemble' + str(i) + '.txt', Enstropy[i])
        np.savetxt('./output/divu/ensemble' + str(i) + '.txt', Divu[i])
        plt.figure(0)
        plt.plot(x, Energy[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('Energy')
        plt.savefig('./fig/energy/ensemble' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, Enstropy[i])
        plt.xlabel('t')
        plt.ylabel('Enstropy')
        plt.xlim((t_init, T))
        plt.savefig('./fig/enstropy/ensemble' + str(i) + '.png')
        plt.figure(2)
        plt.plot(x, Divu[i])
        plt.xlabel('t')
        plt.ylabel('Divu')
        plt.xlim((t_init, T))
        plt.savefig('./fig/divu/ensemble' + str(i) + '.png')
        plt.close('all')
    np.savetxt('./output/energy_avg/ensemble' + '.txt', Energy_avg)
    np.savetxt('./output/enstropy_avg/ensemble' + '.txt', Enstropy_avg)
    np.savetxt('./output/divu_avg/ensemble' + '.txt', Divu_avg)
    plt.figure(0)
    plt.plot(x, Energy_avg)
    plt.xlabel('t')
    plt.xlim((t_init, T))
    plt.ylabel('Energy')
    plt.savefig('./fig/energy_avg/ensemble' + '.png')
    plt.figure(1)
    plt.plot(x, Enstropy_avg)
    plt.xlabel('t')
    plt.ylabel('Enstropy')
    plt.xlim((t_init, T))
    plt.savefig('./fig/enstropy_avg/ensemble' + '.png')
    plt.figure(2)
    plt.plot(x, Divu_avg)
    plt.xlabel('t')
    plt.ylabel('Divu')
    plt.xlim((t_init, T))
    plt.savefig('./fig/divu_avg/ensemble' + '.png')
    plt.close('all')
    return

def ensemble_penalty(u_init, f_list, J, mesh, SPACE_STEP, TIME_STEP, t_init, T, nu, mu, beta, epsilon):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element space
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
    u_prime = [Function(V)] * J
    u1_prime = [Function(V)] * J
    u2_prime = [Function(V)] * J
    u_prev = [Function(V)] * J
    for i in range(J):
        u_prev[i] = project(u_init[i], V)

    N = int((T - 0.0) / TIME_STEP + 1)
    PRESSUREPENALTY = 1.0e-12
    ldcount = int((T - t_init) / TIME_STEP) + 1
    drag_arr = np.zeros((J, ldcount))
    lift_arr = np.zeros((J, ldcount))
    jj = 0
    Energy = np.zeros((J, N))
    Enstropy = np.zeros((J, N))
    Divu = np.zeros((J, N))
    Energy_avg = [0.0] * N
    Enstropy_avg = [0.0] * N
    Divu_avg = [0.0] * N
    # calculate stability condition
    diffu = np.zeros((J, N))
    diffu_norm = np.zeros((J, N))
    # Initial conditions
    t = t_init
    for i in range(J):
        # load reference results
        Energy[i][0] = energy(u_prev[i])
        Enstropy[i][0] = enstrophy(u_prev[i])
        Divu[i][0] = cal_div(u_prev[i])
    u_avg = cal_avg(u_prev, V, J)
    Energy_avg[0] = energy(u_avg)
    Enstropy_avg[0] = enstrophy(u_avg)
    Divu_avg[0] = cal_div(u_avg)
    # lift and drag
    vd, vl, circle = lift_drag(V)
    # file
    # ufile = [File("./resultsV/ensemble_penalty_velocity" + str(int(i)) + ".pvd") for i in range(J)]
    # pfile = [File("./resultsP/ensemble_penalty_pressure" + str(int(i)) + ".pvd") for i in range(J)]
    frameRat = int(1.0 / TIME_STEP)
    # loop
    count_time = 0
    while count_time < N - 1:
        # Update current time
        t += TIME_STEP
        count_time += 1
        # update u_prime
        for i in range(J):
            u_prime[i] = u_prev[i] - u_avg
            diffu[i][count_time-1] = cal_diffu(u_prime[i], V)
            diffu_norm[i][count_time-1] = cal_diff_divu(u_prime[i])
        # Weak form of the momentum equation
        print("t = " + str(t) + ", ensemble_penalty")
        # Weak form of the momentum equation
        # shared coefficient matrix
        #u_avg = project(u_avg, V)
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx
               + b(u_avg, u, v) * dx
               + nu * a_1(u, v) * dx
               + 1.0 / epsilon * div(u) * div(v) * dx
               )
        A = assemble(A_u)
        bcs = boundary_condition_penalty(V, t)
        [bc.apply(A) for bc in bcs]
        solver = LUSolver(A)
        # solver.parameters['reuse_factorization'] = True
        # RHS for each realization
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # update the time on the source and boundary condition
            f_list.s = t
            # RHS
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list, v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            [bc.apply(B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            u_next = Function(V)
            #solve(A, u_next.vector(), B)
            solver.solve(u_next.vector(), B)
            A_p = (Constant(epsilon) * dot(grad(p), grad(q)) * dx)
            A1 = assemble(A_p)
            p_next = Function(Q)
            B_p = (div(u_next) * div(nabla_grad(q)) * dx)
            # B_p = -(inner(nabla_grad(div(u_next)), nabla_grad(q)) * dx)
            B1 = assemble(B_p)
            solve(A1, p_next.vector(), B1)
            # update
            u_prev[i] = u_next
            # save data
            # save initial result at t_init
            Enstropy[i][count_time] = enstrophy(u_next)
            Energy[i][count_time] = energy(u_next)
            Divu[i][count_time] = cal_div(u_next)
            # calculate drag_coefficient after t_init
            if count_time >= int(t_init / TIME_STEP):
                drag_arr[i][jj], lift_arr[i][jj] = calculate_lift_drag(u_next, p_next, nu, vd, vl)
                if i == J - 1:
                    jj += 1
            if count_time % frameRat == 0:
                # ufile[i] << (u_next, t)
                # pfile[i] << (p_next, t)
                c = plot(u_next)
                plt.colorbar(c)
                plt.savefig('./fig/plotu/ensemble_penalty' + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
                speed = cal_speed(u_next)
                c = plot(project(speed, Q))
                plt.colorbar(c)
                plt.savefig('./fig/speed/ensemble_penalty' + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
        u_avg = cal_avg(u_prev, V, J)
        if count_time % frameRat == 0:
            c = plot(project(u_avg, V))
            plt.colorbar(c)
            plt.savefig('./fig/plotu_avg/ensemble_penalty' + '_t' + str(int(t)) + '.png')
            plt.close('all')
            speed = cal_speed(project(u_avg, V))
            c = plot(project(speed, Q))
            plt.colorbar(c)
            plt.savefig('./fig/speed_avg/ensemble_penalty' + '_t' + str(int(t)) + '.png')
            plt.close('all')
        Energy_avg[count_time] = energy(u_avg)
        Enstropy_avg[count_time] = enstrophy(u_avg)
        Divu_avg[count_time] = cal_div(u_avg)
    x = np.linspace(t_init, T, int((T - t_init) / TIME_STEP) + 1)
    for i in range(J):
        np.savetxt('./output/diff/ensemble_penalty' + str(i) + '.txt', diffu[i])
        np.savetxt('./output/diff_norm/ensemble_penalty' + str(i) + '.txt', diffu_norm[i])
        plt.figure(0)
        plt.plot(x, diffu[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('max u prime')
        plt.savefig('./fig/diff/ensemble_penalty' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, diffu_norm[i])
        plt.xlabel('t')
        plt.ylabel('u prime norm')
        plt.xlim((t_init, T))
        plt.savefig('./fig/diff_norm/ensemble_penalty' + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/lift/ensemble_penalty' + str(i) + '.txt', lift_arr[i])
        np.savetxt('./output/drag/ensemble_penalty' + str(i) + '.txt', drag_arr[i])
        plt.figure(0)
        plt.plot(x, lift_arr[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('lift')
        plt.savefig('./fig/lift/ensemble_penalty' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, drag_arr[i])
        plt.xlabel('t')
        plt.ylabel('drag')
        plt.xlim((t_init, T))
        plt.savefig('./fig/drag/ensemble_penalty' + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/energy/ensemble_penalty' + str(i) + '.txt', Energy[i])
        np.savetxt('./output/enstropy/ensemble_penalty' + str(i) + '.txt', Enstropy[i])
        np.savetxt('./output/divu/ensemble_penalty' + str(i) + '.txt', Divu[i])
        plt.figure(0)
        plt.plot(x, Energy[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('Energy')
        plt.savefig('./fig/energy/ensemble_penalty' + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, Enstropy[i])
        plt.xlabel('t')
        plt.ylabel('Enstropy')
        plt.xlim((t_init, T))
        plt.savefig('./fig/enstropy/ensemble_penalty' + str(i) + '.png')
        plt.figure(2)
        plt.plot(x, Divu[i])
        plt.xlabel('t')
        plt.ylabel('Divu')
        plt.xlim((t_init, T))
        plt.savefig('./fig/divu/ensemble_penalty' + str(i) + '.png')
        plt.close('all')
    np.savetxt('./output/energy_avg/ensemble_penalty' + '.txt', Energy_avg)
    np.savetxt('./output/enstropy_avg/ensemble_penalty' + '.txt', Enstropy_avg)
    np.savetxt('./output/divu_avg/ensemble_penalty' + '.txt', Divu_avg)
    plt.figure(0)
    plt.plot(x, Energy_avg)
    plt.xlabel('t')
    plt.xlim((t_init, T))
    plt.ylabel('Energy')
    plt.savefig('./fig/energy_avg/ensemble_penalty' + '.png')
    plt.figure(1)
    plt.plot(x, Enstropy_avg)
    plt.xlabel('t')
    plt.ylabel('Enstropy')
    plt.xlim((t_init, T))
    plt.savefig('./fig/enstropy_avg/ensemble_penalty' + '.png')
    plt.figure(2)
    plt.plot(x, Divu_avg)
    plt.xlabel('t')
    plt.ylabel('Divu')
    plt.xlim((t_init, T))
    plt.savefig('./fig/divu_avg/ensemble_penalty' + '.png')
    plt.close('all')
    return








