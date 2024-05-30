from dolfin import *
import numpy as np
# Useful Expressions
def b(u, v, w):  # Skew-symmetry nonlinear term
    return 0.5 * (inner(dot(u, nabla_grad(v)), w) - inner(dot(u, nabla_grad(w)), v))
def a_1(u, v):  # Viscous term (grad(u),grad(v))
    return inner(nabla_grad(u), nabla_grad(v))
def b_2(EEV, u, v):
    return EEV * inner(nabla_grad(u), nabla_grad(v))
    #return inner(div(EEV * grad(u)), v)


#Define avg of u
def cal_avg(u_prev, V, J):
    u_avg = u_prev[0]
    for i in range(1, J):
        u_avg += u_prev[i]
    u_avg = u_avg / J
    return u_avg

def cal_prime(u_prev, V, u_avg, J):
    u_prime = [Function(V)] * J
    for i in range(J):
        u_prime[i] = project(u_prev[i] - u_avg, V)
    return u_prime

# Defini EEV
def Eddy_viscosity(type, mu, u_prime, V1, time_step, space_step, J):
    u_prime_sum = inner(u_prime[0], u_prime[0])
    for i in range(1, J):
        u_prime_sum += inner(u_prime[i], u_prime[i])
    #EEV = Function(V1)
    if type == 1:
        EEV = mu * sqrt(u_prime_sum) * space_step#0.02
    else:
        EEV = mu * u_prime_sum * time_step
    EEV = EEV / J
    return EEV
def max_eddy(type, mu, u_prime, V1, time_step, space_step, J):
    u_prime_max = inner(u_prime[0], u_prime[0])
    for i in range(1, J):
        u_prime_max = max(u_prime_max, inner(u_prime[i], u_prime[i]))
    if type == 1:
        EEV = mu * sqrt(u_prime_max) * 0.1
    else:
        EEV = mu * u_prime_max * time_step
    return EEV

def modify_Eddy(type, beta, u_prime, V1, time_step, space_step, eps, J):
    constant = beta * np.power(time_step, 7.0 / 4.0) / np.power(eps, 3.0 / 4.0)
    div_u_prim_norm_sum = pow(div(u_prime[0]), Constant(4.0))
    for i in range(1, J):
        div_u_prim_norm_sum += pow(div(u_prime[i]), Constant(4.0))
    EEV = constant / J * sqrt(assemble(div_u_prim_norm_sum * dx))
    # div_u_prim_norm_sum = sqrt(assemble(pow(div(u_prime[0]), Constant(4.0)) * dx))
    # for i in range(1, J):
    #     div_u_prim_norm_sum += sqrt(assemble(pow(div(u_prime[i]), Constant(4.0)) * dx))
    # EEV = constant / J
    return EEV
    return EEV

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

def cal_norm(u_fem, u):
    L2error = assemble(inner(u_fem - u, u_fem - u) * dx)
    gradL2error = assemble(inner(nabla_grad(u_fem - u), nabla_grad(u_fem - u)) * dx)
    unorm = assemble(inner(u_fem, u_fem) * dx)
    gradunorm = assemble(inner(nabla_grad(u_fem), nabla_grad(u_fem)) * dx)
    return L2error, gradL2error, unorm, gradunorm

def cal_div(u_fem):
    divu = assemble(inner(div(u_fem), div(u_fem)) * dx)
    return divu

def extract_value(solution, V, point_x=0.5, point_y=0.5):
    solution = project(solution, V)
    return solution(point_x, point_y)




