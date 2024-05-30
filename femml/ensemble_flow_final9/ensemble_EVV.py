from helper import *
from mesh_create import *
def stable_condition(TIME_STEP, mesh_size, nu, u, u_avg):
    return TIME_STEP/(nu * mesh_size) * assemble(inner(nabla_grad(u - u_avg), nabla_grad(u - u_avg)) * dx)
def ensemble_EVV(f_list, J, mesh, SPACE_STEP, TIME_STEP, t_init, T, nu, mu, type = 2):
    """
    Solves the incompressible Navier Stokes equations
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # create finite element
    V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
    V1_ele = FiniteElement("CG", mesh.ufl_cell(), 2)
    Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    W_ele = V_ele*Q_ele
    # create finite element space
    V = FunctionSpace(mesh, V_ele)
    V1 = FunctionSpace(mesh, V1_ele)
    Q = FunctionSpace(mesh, Q_ele)
    W = FunctionSpace(mesh, W_ele)
    # Define Trialfunction, Testfunction
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    #initial data list
    u_prime = [Function(V)] * J
    u1_prime = [Function(V)] * J
    u2_prime = [Function(V)] * J
    u_prev = [Function(V)] * J

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
    # stability condition value
    condition = np.zeros((J, N))
    # Initial conditions
    t = t_init
    for i in range(J):
        # load reference results
        Energy[i][0] = energy(u_prev[i])
        Enstropy[i][0] = enstrophy(u_prev[i])
        Divu[i][0] = cal_div(u_prev[i])
    u_avg = cal_avg(u_prev, V, J)
    for i in range(J):
        condition[i][0] = stable_condition(TIME_STEP, SPACE_STEP, nu, u_prev[i], u_avg)
    Energy_avg[0] = energy(u_avg)
    Enstropy_avg[0] = enstrophy(u_avg)
    Divu_avg[0] = cal_div(u_avg)
    # Boundary condition
    # noslip
    noslip = Constant((0.0, 0.0))
    # boundary condition
    bcs = []
    bcs.append(DirichletBC(W.sub(0), noslip, boundary))
    # lift and drag
    vd, vl, circle = lift_drag(V)
    # file
    # ufile = [File("./resultsV/ensemble_EVV_velocity" + "_evvtype" + str(type) + str(int(i)) + ".pvd") for i in range(J)]
    # pfile = [File("./resultsP/ensemble_EVV_pressure" + "_evvtype" + str(type) + str(int(i)) + ".pvd") for i in range(J)]
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
            diffu[i][count_time - 1] = cal_diffu(u_prime[i], V)
            diffu_norm[i][count_time - 1] = cal_diff_divu(u_prime[i])
        # Weak form of the momentum equation
        print("t = " + str(t) + ", ensemble_EVV")
        # shared coefficient matrix
        #EEV
        EEV = Eddy_viscosity(type, mu, u_prime, V, TIME_STEP, SPACE_STEP, J)
        # EEV = max_eddy(type, mu, u_prime, V1, TIME_STEP, SPACE_STEP, J)
        A_u = (1.0 / TIME_STEP * inner(u, v) * dx
               + b(u_avg, u, v) * dx
               + nu * a_1(u, v) * dx
               + b_2(EEV, u, v) * dx
               - p * div(v) * dx
               - q * div(u) * dx
               + p * q * PRESSUREPENALTY * dx
               )
        A = assemble(A_u)
        [bc.apply(A) for bc in bcs]
        solver = LUSolver(A)
        # solver.parameters['reuse_factorization'] = True
        # RHS for each realization
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # update the time on the source and boundary condition
            mint_val = smooth_bridge(t)
            f_list[i].s = mint_val
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx
                   - b(u_prime[i], u_prev[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            [bc.apply(B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            w_next = Function(W)
            #solve(A, w_next.vector(), B)
            solver.solve(w_next.vector(), B)
            u_next, p_next = w_next.split() # deepcopy=True
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
                plt.savefig('./fig/plotu/ensemble_EVV_type' + str(type) + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
                speed = cal_speed(u_next)
                c = plot(project(speed, Q))
                plt.colorbar(c)
                plt.savefig('./fig/speed/ensemble_EVV_type' + str(type) + str(i) + '_t' + str(int(t)) + '.png')
                plt.close('all')
        u_avg = cal_avg(u_prev, V, J)
        for i in range(J):
            condition[i][count_time] = stable_condition(TIME_STEP, SPACE_STEP, nu, u_prev[i], u_avg)
        if count_time % frameRat == 0:
            c = plot(project(u_avg, V))
            plt.colorbar(c)
            plt.savefig('./fig/plotu_avg/ensemble_EVV_type' + str(type) + '_t' + str(int(t)) + '.png')
            plt.close('all')
            speed = cal_speed(project(u_avg, V))
            c = plot(project(speed, Q))
            plt.colorbar(c)
            plt.savefig('./fig/speed_avg/ensemble_EVV_type' + str(type) + '_t' + str(int(t)) + '.png')
            plt.close('all')
        Energy_avg[count_time] = energy(u_avg)
        Enstropy_avg[count_time] = enstrophy(u_avg)
        Divu_avg[count_time] = cal_div(u_avg)
    x = np.linspace(t_init, T, int((T - t_init) / TIME_STEP) + 1)
    for i in range(J):
        np.savetxt('./output/condition/ensemble_EVV_type' + str(type) + str(i) + '.txt', condition[i])
        plt.figure(0)
        plt.plot(x, condition[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('condition number')
        plt.savefig('./fig/condition/ensemble_EVV_type' + str(type) + str(i) +  '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/diff/ensemble_EVV_type' + str(type) + str(i) + '.txt', diffu[i])
        np.savetxt('./output/diff_norm/ensemble_EVV_type' + str(type) + str(i) + '.txt', diffu_norm[i])
        plt.figure(0)
        plt.plot(x, diffu[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('max u prime')
        plt.savefig('./fig/diff/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, diffu_norm[i])
        plt.xlabel('t')
        plt.ylabel('u prime norm')
        plt.xlim((t_init, T))
        plt.savefig('./fig/diff_norm/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/lift/ensemble_EVV_type' + str(type) + str(i) + '.txt', lift_arr[i])
        np.savetxt('./output/drag/ensemble_EVV_type' + str(type) + str(i) + '.txt', drag_arr[i])
        plt.figure(0)
        plt.plot(x, lift_arr[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('lift')
        plt.savefig('./fig/lift/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, drag_arr[i])
        plt.xlabel('t')
        plt.ylabel('drag')
        plt.xlim((t_init, T))
        plt.savefig('./fig/drag/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.close('all')
    for i in range(J):
        np.savetxt('./output/energy/ensemble_EVV_type' + str(type) + str(i) + '.txt', Energy[i])
        np.savetxt('./output/enstropy/ensemble_EVV_type' + str(type) + str(i) + '.txt', Enstropy[i])
        np.savetxt('./output/divu/ensemble_EVV_type' + str(type) + str(i) + '.txt', Divu[i])
        plt.figure(0)
        plt.plot(x, Energy[i])
        plt.xlabel('t')
        plt.xlim((t_init, T))
        plt.ylabel('Energy')
        plt.savefig('./fig/energy/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.figure(1)
        plt.plot(x, Enstropy[i])
        plt.xlabel('t')
        plt.ylabel('Enstropy')
        plt.xlim((t_init, T))
        plt.savefig('./fig/enstropy/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.figure(2)
        plt.plot(x, Divu[i])
        plt.xlabel('t')
        plt.ylabel('Divu')
        plt.xlim((t_init, T))
        plt.savefig('./fig/divu/ensemble_EVV_type' + str(type) + str(i) + '.png')
        plt.close('all')
    np.savetxt('./output/energy_avg/ensemble_EVV_type' + str(type) + '.txt', Energy_avg)
    np.savetxt('./output/enstropy_avg/ensemble_EVV_type' + str(type) + '.txt', Enstropy_avg)
    np.savetxt('./output/divu_avg/ensemble_EVV_type' + str(type) + '.txt', Divu_avg)
    plt.figure(0)
    plt.plot(x, Energy_avg)
    plt.xlabel('t')
    plt.xlim((t_init, T))
    plt.ylabel('Energy')
    plt.savefig('./fig/energy_avg/ensemble_EVV_type' + str(type) + '.png')
    plt.figure(1)
    plt.plot(x, Enstropy_avg)
    plt.xlabel('t')
    plt.ylabel('Enstropy')
    plt.xlim((t_init, T))
    plt.savefig('./fig/enstropy_avg/ensemble_EVV_type' + str(type) + '.png')
    plt.figure(2)
    plt.plot(x, Divu_avg)
    plt.xlabel('t')
    plt.ylabel('Divu')
    plt.xlim((t_init, T))
    plt.savefig('./fig/divu_avg/ensemble_EVV_divu_avg_type' + str(type) + '.png')
    plt.close('all')
    return



