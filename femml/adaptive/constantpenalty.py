import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
import csv
from tabulate import tabulate

# save data
def save_tsv_dataset(data, file):
    csvfile = open(file, 'w', newline='\n')
    tsv_output = csv.writer(csvfile, delimiter='\t')
    tsv_output.writerows(data)
    csvfile.close()
def save_table_dataset(data, file, colname):
    table = tabulate(data, headers=col_name, tablefmt="grid", showindex="always")
    f = open(file, 'w')
    writer = csv.writer(f)
    writer.writerow(colname)
    for row in data:
        writer.writerow(row)
    f.close()
# Useful Expressions
def b(u, v, w):  # Skew-symmetry nonlinear term
    return .5 * (inner(dot(u, nabla_grad(v)), w) - inner(dot(u, nabla_grad(w)), v))
def a_1(u, v):  # Viscous term (grad(u),grad(v))
    return inner(nabla_grad(u), nabla_grad(v))

# define exact solution
def exactsolution(nu,order):
    t = 0.0
    # Exact solution
    x, y, s = sp.symbols('x[0] x[1] s')
    u1_exact = pi * sp.sin(s) * sp.sin(2 * pi * y) * (sp.sin(pi * x)) ** 2  # Exact velocity, x-component
    u2_exact = -pi * sp.sin(s) * sp.sin(2 * pi * x) * (sp.sin(pi * y)) ** 2  # Exact velocity, y-component
    p_exact = sp.sin(s) * sp.cos(pi * x) * sp.sin(pi * y)  # Exact pressure
    f1 = u1_exact.diff(s, 1) + u1_exact * u1_exact.diff(x, 1) + u2_exact * u1_exact.diff(y, 1) \
        - nu * sum(u1_exact.diff(xi, 2) for xi in (x, y)) + p_exact.diff(x, 1)  # Forcing, x-component
    f2 = u2_exact.diff(s, 1) + u1_exact * u2_exact.diff(x, 1) + u2_exact * u2_exact.diff(y, 1) \
        - nu * sum(u2_exact.diff(xi, 2) for xi in (x, y)) + p_exact.diff(y, 1)  # Forcing, y-component
    u1_exact = sp.simplify(u1_exact)  # Velocity simplification, x-component
    u2_exact = sp.simplify(u2_exact)  # Velocity simplification, y-component
    p_exact = sp.simplify(p_exact)  # Pressure simplification
    f1 = sp.simplify(f1)  # Forcing simplification
    f2 = sp.simplify(f2)
    u_exact = Expression((sp.printing.ccode(u1_exact), sp.printing.ccode(u2_exact)), degree=order,
                         s=t)  # Exact velocity expression
    p_exact = Expression(sp.printing.ccode(p_exact), degree=order, s=t)  # Exact pressure expression
    f = Expression((sp.printing.ccode(f1), sp.printing.ccode(f2)), degree=order, s=t)
    return u_exact, p_exact, f

def penalty(u_exact, p_exact, f, N_NODES, TIME_STEP, T, nu, epsilon):
    """
    Solves the incompressible Navier Stokes equations using penalty method,
    replace div u=0 with ∇ ⋅ u + epsilon p = 0, where We will employ
    the FEniCS Python Package
    Momentum:           ∂u/∂t -ν ∇²u + (u ⋅ ∇) u + ∇p = f
    Incompressibility:  ∇ ⋅ u = 0
    """
    # define mesh
    mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), N_NODES, N_NODES)
    # create finite element function space
    velocity_function_space = VectorFunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(velocity_function_space)
    v = TestFunction(velocity_function_space)
    # Define the solution fields involved
    u_prev = Function(velocity_function_space)  # u_n
    u_prev_2 = Function(velocity_function_space)  # u_(n-1)
    u_tent = Function(velocity_function_space)  # u^1_(n+1)
    u_next = Function(velocity_function_space)  # u_(n+1)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    velocity_boundary_conditions = DirichletBC(velocity_function_space, u_exact, boundary)
    # Specify initial conditions
    t = 0.0
    u_exact.s = t
    u_prev_2 = interpolate(u_exact, velocity_function_space)
    t = TIME_STEP
    u_exact.s = t
    u_prev = interpolate(u_exact, velocity_function_space)

    data = []
    m = velocity_function_space.dim()
    data1 = np.zeros((int((T - t)/TIME_STEP), m+1))
    i = 0
    while t < T:
        # Update current time
        t += TIME_STEP
        u_exact.s = t
        p_exact.s = t
        f.s = t  # update the time on the source
        print("current time:" + str(f.s))
        # Weak form of the momentum equation
        momentum_weak_form_lhs = (1.0 / TIME_STEP * inner(u, v) * dx + b(2 * u_prev - u_prev_2, u, v) * dx
                                  + 1.0 / epsilon * div(u) * div(v) * dx + nu * a_1(u, v) * dx)
        momentum_weak_form_rhs = 1.0 / TIME_STEP * inner(u_prev, v) * dx + inner(f, v) * dx
        momentum_assembled_system_matrix = assemble(momentum_weak_form_lhs)
        momentum_assembled_rhs = assemble(momentum_weak_form_rhs)
        velocity_boundary_conditions.apply(momentum_assembled_system_matrix, momentum_assembled_rhs)
        solve(momentum_assembled_system_matrix, u_tent.vector(), momentum_assembled_rhs)
        # Apply time filter
        u_next.assign(u_tent)
        # Advance in time
        u_prev_2.assign(u_prev)
        u_prev.assign(u_next)
        # Visualize interactively
        plot(u_next)
        plt.draw()
        plt.pause(0.02)
        plt.clf()
        # L2 error
        u_fem = interpolate(u_exact, velocity_function_space)
        error = (u_fem - u_next) ** 2 * dx(mesh)
        L2_err = np.sqrt(abs(assemble(error)))
        print("L2 norm error:" + str(L2_err))
        # Examining vertex values
        nodal_values_u = u_tent.vector()
        # convert the Vector object to a standard numpy array
        array_u = np.array(nodal_values_u)
        array_u = array_u.reshape((1, len(array_u)))
        vertex_values_u = u_tent.compute_vertex_values(mesh)
        unorm = norm(nodal_values_u, "L2")
        divunorm = np.sqrt(assemble(inner(div(u_tent), div(u_tent)) * dx(mesh)))
        gradunorm = np.sqrt(assemble(inner(nabla_grad(u_tent), nabla_grad(u_tent)) * dx(mesh)))
        # EST = divunorm/gradunorm
        data.append([t, epsilon, unorm, divunorm, gradunorm, L2_err])
        data1[i][0] = t
        data1[i][1:] = array_u
        i += 1
    return np.round(data, 6), data1


if __name__ == "__main__":
    # parameter setting
    N_NODES = 270  # Uniform meshes with 270 nodes per side on the boundary
    TIME_STEP = 0.05
    T = 0.2  # end time 1
    nu = 1  # Kinematic viscocity
    epsilon = 1e-5
    # exact solution
    u_exact, p_exact, f = exactsolution(nu, 4)
    # solve nse with penalty method
    data, data1 = penalty(u_exact, p_exact, f, N_NODES, TIME_STEP, T, nu, epsilon)
    print("Navier stokes equation with penalty method")
    # save data
    col_name = ['t', 'epsilon', 'ul2', 'divul2', 'gradul2', 'errorl2']
    save_table_dataset(data, "constantepsilon_norm", col_name)
    save_tsv_dataset(data1, "constantepsilon_vertexvalue")
