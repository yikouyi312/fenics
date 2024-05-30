# importing os module
import os
from dolfin import *
import numpy as np
from mesh_create import flowovercylinder
from Test2 import force1, flow, generate_u_init
from seperate import separate, separate_penalty
from ensemble import ensemble, ensemble_penalty
from ensemble_EVV import ensemble_EVV
from ensemble_EVV_penalty import ensemble_EVV_penalty
from runner import runner
# Generate force function list(ensemble member number = J)
J = 10
# force function
f_list = force1()
# inflow
Inflow = flow()
# pertubation function
# perx, pery = pertubation()
# J force functions
order = 4
tol = 5e-2
seed = 1024
issymmetry = False
# eps_list, Inflow = get_list(InflowExp1, InflowExp2, perx, pery, order, J, tol, seed, issymmetry)
# eps_list, f_list = get_list(f1, f2, perx, pery, order, J, tol, seed, issymmetry)
# eps_list, f_list = get_list1(InflowExp1, InflowExp2, order, J, tol, seed, issymmetry)
eps_list, u_init = generate_u_init(order, J, tol, seed, issymmetry)
# parameter setting
t_init = 0.0
dt = 0.005
T = 8.0 + dt
t_num = int((T-t_init)/dt)
nu = 1./1000.
TOL = 1.e-20
tol_con = .01

# Define mesh
N = 32
mesh = flowovercylinder(N)
# plot(mesh)
space_size = mesh.hmin()
print(space_size)
space_size = mesh.hmax()
print(space_size)
measure = assemble(1.0 * dx(mesh)) #size of the domain (for normalizing pressure)
print('mesh area = ' + str(measure))
#create fold
parent_directory = "/Users/boweiouyang/Desktop/femml/cylinder1/output"
if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)
directory = "test"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)
# hyper-parameter
penalty_epsilon = 1e-2
mu = 0.55
beta = 1000.0

separate(u_init, f_list, J, mesh, N, dt, t_init, T, nu)
separate_penalty(u_init, f_list, J, mesh, N, dt, t_init, T, nu, penalty_epsilon)
ensemble(u_init, f_list, J, mesh, N, t_init, dt, T, nu)
ensemble_penalty(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon)
ensemble_EVV(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, type=1)
ensemble_EVV(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, type=2)
ensemble_EVV_penalty(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 0)
ensemble_EVV_penalty(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 0)
ensemble_EVV_penalty(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 1)
ensemble_EVV_penalty(u_init, f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 1)
# ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 2)
# ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 2)
