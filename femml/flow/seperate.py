import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
from helper import *
from mesh_create import *
def seperate(f_list, J, mesh, TIME_STEP, t_init, T, nu):
    """
    Solves the incompressible Navier Stokes equations
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    measure = assemble(1.0 * dx(mesh))  # size of the domain (for normalizing pressure)
    n = FacetNormal(mesh)
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
    #noslip
    noslip = Constant((0.0, 0.0))
    #boundary condition
    bcs = []
    bcs.append(DirichletBC(W.sub(0), noslip, boundary))

    # Lift and drag eventually
    vdExp = Expression(("0", "0"), degree=2)
    vlExp = Expression(("0", "0"), degree=2)
    vd = interpolate(vdExp, V)
    vl = interpolate(vlExp, V)
    circle = DirichletBC(V, (0., -1.), circle_boundary())
    circle.apply(vd.vector())
    circle = DirichletBC(V, (1., 0.), circle_boundary())
    circle.apply(vl.vector())

    #save last T - t_init second result
    time = int((T - t_init) / TIME_STEP)
    drag_arr = np.zeros((J, time))
    lift_arr = np.zeros((J, time))
    jj = 0

    # Dummy solution vector (for mixed solutions)
    # Timestepping--we do a while loop, could also choose a for loop (for jj in range(1,t_num): )
    frameNum = 40  # per second
    frameRat = int(1 / (frameNum * TIME_STEP))
    ufile = [None] * J
    pfile = [None] * J
    for i in range(J):
        ufile[i] = File("flow/seperate/velocity" + str(i) + ".pvd")
        pfile[i] = File("flow/seperate/pressure" + str(i) + ".pvd")

    # Initial conditions
    t = 0.0
    # initial data list
    u_prev = [Function(V)] * J
    # loop
    count_time = 0
    while t < T:
        # Update current time
        t += TIME_STEP
        count_time += 1
        mint_val = smooth_bridge(t)
        #print("current time:" + str(t))
        for i in range(J):
            #print("current ensemble member:" + str(i))
            # Weak form of the momentum equation
            A_u = (1.0 / TIME_STEP * inner(u, v) * dx
                   + b(u_prev[i], u, v) * dx
                   + nu * a_1(u, v) * dx
                   - p * div(v) * dx
                   - q * div(u) * dx
                   + p * q * 1e-10 * dx
                   )
            A = assemble(A_u)
            # update the time on the source
            f_list[i].s = t
            # RHS for each realization
            # get the RHS vector
            B_u = (1.0 / TIME_STEP * inner(u_prev[i], v) * dx + inner(f_list[i], v) * dx)
            B = assemble(B_u)
            # add boundary condition
            [bc.apply(A, B) for bc in bcs]
            # solve A(u, p) = B
            # Define the solution fields involved
            w_next = Function(W)
            solve(A, w_next.vector(), B)
            u_next, p_next = w_next.split()
            # update u_prev
            u_prev[i] = u_next
            print('t:' + str(t) + ', divu: ' + str(sqrt(assemble(pow(div(u_next), 2) * dx))))
            # save data
            # save initial result at t_init
            if count_time == int(1.0 / TIME_STEP):
                print("I collected an initial condition at t =" + str(t))
                # pdb.set_trace()
                filename_init_v = './Initializing/velocity_init/' + 'seperate' + str(i) + '.txt'
                filename_init_p = './Initializing/pressure_init/' + 'seperate' + str(i) + '.txt'
                u_init_hold = u_next.vector().get_local()
                p_init_hold = p_next.vector().get_local()
                np.savetxt(filename_init_v, u_init_hold)
                np.savetxt(filename_init_p, p_init_hold)
            # calculate drag_coefficient after t_init
            if count_time >= int(t_init / TIME_STEP):
                # plot solution
                # plot(u_next)
                # plt.show()
                drag_arr[i][jj] = assemble(nu * a_1(u_next, vd) * dx
                                        + convect(u_next, u_next, vd) * dx
                                        - c(p_next, vd) * dx)
                lift_arr[i][jj] = assemble(nu * a_1(u_next, vl) * dx
                                        + convect(u_next, u_next, vl) * dx
                                        - c(p_next, vl) * dx)
                jj = jj + 1

            if count_time % frameRat == 0:
                ufile[i] << (u_next, t)
                pfile[i] << (p_next, t)

    np.savetxt('Initializing/lift.txt', lift_arr)
    np.savetxt('Initializing/drag.txt', drag_arr)


