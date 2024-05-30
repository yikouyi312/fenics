from dolfin import *
import numpy as np


# Useful Expressions
def b(u, v, w):  # Skew-symmetry nonlinear term
    return 0.5 * (inner(dot(u, nabla_grad(v)), w) - inner(dot(u, nabla_grad(w)), v))

def c(p, u):
    return dot(p, div(u))

def a_1(u, v):  # Viscous term (grad(u),grad(v))
    return inner(grad(u), grad(v))


def convect(u, v, w):
    return dot(dot(u, nabla_grad(v)), w)


def b_2(EEV, u, v):
    return EEV * inner(grad(u), nabla_grad(v))
    # return inner(div(EEV * grad(u)), v)


# Define avg of u
def cal_avg(u_prev, V, J):
    u_avg = Function(V)
    u_avg.vector()[:] = u_prev[0].vector()
    for i in range(1, J):
        u_avg.vector()[:] += u_prev[i].vector()[:]
    u_avg.vector()[:] = u_avg.vector()[:] / J
    return u_avg


# Define EVV
def Eddy_viscosity(type, mu, u_prime, V1, time_step, space_step, J):
    u_prime_sum = Function(V1)
    u, v = u_prime[0].split(deepcopy=True)
    u_prime_sum.vector()[:] = u.vector()[:] ** 2 + v.vector()[:] ** 2
    for i in range(1, J):
        u, v = u_prime[i].split(deepcopy=True)
        u_prime_sum.vector()[:] += u.vector()[:] ** 2 + v.vector()[:] ** 2
    EEV = Function(V1)
    if type == 1:
        EEV.vector()[:] = 2 * mu * np.sqrt(u_prime_sum.vector()[:]) * space_step
    else:
        EEV.vector()[:] = 2 * mu * (u_prime_sum.vector()[:]) * time_step
    return EEV


# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary


# Smooth bridge (to allow force to increase slowly)
def smooth_bridge(t):
    if t > 1 + 1e-14:
        return 1.0
    elif abs(1 - t) > 1e-14:
        return np.exp(-np.exp(-1. / (1 - t) ** 2) / t ** 2)
    else:
        return 1.0
