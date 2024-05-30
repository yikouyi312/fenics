from dolfin import *
import numpy as np
# Useful Expressions
def b(u, v, w):  # Skew-symmetry nonlinear term
    return 0.5 * (inner(dot(u, nabla_grad(v)), w) - inner(dot(u, nabla_grad(w)), v))
def a_1(u, v):  # Viscous term (grad(u),grad(v))
    return inner(nabla_grad(u), nabla_grad(v))
def convect(u, v, w):
    return dot(dot(u, nabla_grad(v)), w)
def c(p, u):
    return dot(p, div(u))
def b_2(EEV, u, v):
    return EEV * inner(nabla_grad(u), nabla_grad(v))
    #return inner(div(EEV * grad(u)), v)
# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

# Calculate quantity
def enstrophy(u):
    return 0.5 * assemble(pow(curl(u), 2) * dx)
def energy(u):
    return 0.5 * assemble(pow(u, 2) * dx)
def boundary_condition(V, Q, t):
    TOL = 1.0e-20
    # Boundary condition
    walls = 'near(x[1],0) || near(x[1],0.41)'
    inflow = 'near(x[0],0)'
    outflow = 'near(x[0],2.2)'
    cyl = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
    class OriginPoint(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < TOL and x[1] < TOL
    originpoint = OriginPoint()
    # Actual boundary conditions (inflow velocity and no slip
    InflowExp = Expression(('pow(0.41,-2) * sin(pi*t/8.0) * 6 * x[1] * (0.41-x[1])', '0'), t=t, degree=2)
    bcw = DirichletBC(V, Constant((0, 0)), walls)  # bc for top/bottom walls
    bcc = DirichletBC(V, Constant((0, 0)), cyl)  # bc for cylindar
    bci = DirichletBC(V, InflowExp, inflow)
    bco = DirichletBC(V, InflowExp, outflow)
    bcp = DirichletBC(Q, 0.0, originpoint, 'pointwise')
    bcs = [bcw, bcc, bci, bco, bcp]
    return bcs
def boundary_condition_penalty(V, t):
    TOL = 1.0e-20
    # Boundary condition
    walls = 'near(x[1],0) || near(x[1],0.41)'
    inflow = 'near(x[0],0)'
    outflow = 'near(x[0],2.2)'
    cyl = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
    # Actual boundary conditions (inflow velocity and no slip
    InflowExp = Expression(('pow(0.41,-2)*sin(pi*t/8.0)*6*x[1]*(0.41-x[1])', '0'), t=t, degree=2)
    bcw = DirichletBC(V, Constant((0, 0)), walls)  # bc for top/bottom walls
    bcc = DirichletBC(V, Constant((0, 0)), cyl)  # bc for cylindar
    bci = DirichletBC(V, InflowExp, inflow)
    bco = DirichletBC(V, InflowExp, outflow)
    bcs = [bcw, bcc, bci, bco]
    return bcs

def lift_drag(V):
    # Lift and drag eventually
    vdExp = Expression(("0", "0"), degree=2)
    vlExp = Expression(("0", "0"), degree=2)
    vd = interpolate(vdExp, V)
    vl = interpolate(vlExp, V)
    class circle_boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and \
                   ((x[0] - .2) ** 2 + (x[1] - .2) ** 2 < 0.0025 + 3 * DOLFIN_EPS)
    circle = DirichletBC(V, (1., 0.), circle_boundary())
    circle.apply(vd.vector())
    circle = DirichletBC(V, (0., 1.), circle_boundary())
    circle.apply(vl.vector())
    return vd, vl, circle

def calculate_lift_drag(u, p, nu, vd, vl):
    drag = - 20 * assemble(nu * a_1(u, vd) * dx
                        + convect(u, u, vd) * dx
                        - c(p, vd) * dx)
    lift = - 20 * assemble(nu * a_1(u, vl) * dx
                    + convect(u, u, vl) * dx
                    - c(p, vl) * dx)
    return drag, lift

#Define avg of u
def cal_avg(u_prev, V, J):
    u_avg = u_prev[0]
    for i in range(1, J):
        u_avg += u_prev[i]
    u_avg = u_avg / J
    return u_avg

# Defini EEV
def Eddy_viscosity(type, mu, u_prime, V1, time_step, space_step, J):
    u_prime_sum = inner(u_prime[0], u_prime[0])
    for i in range(1, J):
        u_prime_sum += inner(u_prime[i], u_prime[i])
    #EEV = Function(V1)
    u_prime_sum = u_prime_sum * 1 / J
    if type == 1:
        EEV = mu * sqrt(u_prime_sum) * 0.2
    else:
        EEV = mu * u_prime_sum * time_step
    return EEV

def max_eddy(type, mu, u_prime, V1, time_step, space_step, J):
    u_prime_max = inner(u_prime[0], u_prime[0])
    for i in range(1, J):
        u_prime_max = 0.5 * (u_prime_max + inner(u_prime[i], u_prime[i])
                             + abs(u_prime_max - inner(u_prime[i], u_prime[i])))
    if type == 1:
        EEV = mu * sqrt(u_prime_max) * 0.02
    else:
        EEV = mu * u_prime_max * time_step
    return EEV

def modify_Eddy(type, beta, u_prime, V1, time_step, space_step, eps, J):
    if type == 0:
        return 0.0
    # elif type == 1:
    #     constant = beta * np.power(space_step, 3.0 / 2.0) * time_step
    #     div_u_prim_norm_sum = pow(div(u_prime[0]), Constant(4.0))
    #     for i in range(1, J):
    #         div_u_prim_norm_sum += pow(div(u_prime[i]), Constant(4.0))
    #     div_u_prim_norm_sum = div_u_prim_norm_sum * 1 / J
    #     EEV = constant * sqrt(assemble(div_u_prim_norm_sum * dx))
    else:
        constant = beta * np.power(time_step, 7.0 / 4.0) / np.power(eps, 3.0 / 4.0)
        div_u_prim_norm_sum = pow(div(u_prime[0]), Constant(4.0))
        #div_u_prim_norm_sum = div_u_prim_norm_sum * 1 /
        for i in range(1, J):
            div_u_prim_norm_sum += pow(div(u_prime[i]), Constant(4.0))
        EEV = constant * sqrt(assemble(div_u_prim_norm_sum * dx)) / J

    return EEV

def cal_norm(u_fem, u):
    L2error = assemble(inner(u_fem - u, u_fem - u) * dx)
    gradL2error = assemble(inner(nabla_grad(u_fem - u), nabla_grad(u_fem - u)) * dx)
    unorm = assemble(inner(u_fem, u_fem) * dx)
    gradunorm = assemble(inner(nabla_grad(u_fem), nabla_grad(u_fem)) * dx)
    return L2error, gradL2error, unorm, gradunorm

def cal_div(u_fem):
    divu = assemble(inner(div(u_fem), div(u_fem)) * dx)
    return divu

def extract_value(solution, point_x=0.5, point_y=0.5):
    return solution(point_x, point_y)

def cal_speed(u):
    u1, u2 = split(u)
    return pow(u1, 2.0) + pow(u2, 2.0)

def cal_diffu(u_prime, V):
    difu = project(u_prime, V)
    return max(abs(max(difu.vector())), abs(min(difu.vector())))

def cal_diff_divu(u_prime):
    div_u_prim_norm = pow(div(u_prime), 4.0)
    divu = sqrt(assemble(div_u_prim_norm * dx))
    return divu


