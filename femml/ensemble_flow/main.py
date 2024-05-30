# importing os module
import os
from dolfin import *
import numpy as np
from tabulate import tabulate
from mesh_create import offsetcylinder
from Test1 import force1, pertubation, get_list, get_list1
from seperate import separate, separate_penalty
from ensemble import ensemble, ensemble_penalty
from ensemble_EVV import ensemble_EVV
from ensemble_EVV_penalty import ensemble_EVV_penalty
import matplotlib.pyplot as plt
from runner import runner
# Generate force function list(ensemble member number = J)
J = 10
# force function
f1, f2 = force1()
# pertubation function
# perx, pery = pertubation()
# J force functions
order = 4
tol = 5 * 1e-2
seed = 1024
issymmetry = False
# eps_list, f_list = get_list(f1, f2, perx, pery, order, J, tol, seed, issymmetry)
eps_list, f_list = get_list1(f1, f2, order, J, tol, seed, issymmetry)

# parameter setting
t_init = 0.0
T = 16.0
dt = 0.025
t_num = int((T-t_init)/dt)
nu = 1./1800.
TOL = 1.e-10
tol_con = .01

# Define mesh
N = 15
M = 10
mesh = offsetcylinder(N, M)
space_size = mesh.hmin()
space_size = 0.3
print(space_size)
measure = assemble(1.0 * dx(mesh)) #size of the domain (for normalizing pressure)
print('mesh area = ' + str(measure))
#create fold
parent_directory = "/Users/boweiouyang/Desktop/femml/ensemble_flow_final9/output"
if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)
directory = "test"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)
# hyper-parameter
penalty_epsilon = 1e-2
mu = 6.0
beta = 550000

# save data
filepath = os.path.join(path, 'eps')
# Print data and Save data
f = open(filepath, 'w')
col_name = ['eps1', 'eps2', 'eps3', 'eps4',' eps5', 'eps6']
table = tabulate(np.transpose(eps_list), headers=col_name, tablefmt="grid", showindex="always")
print('pertubation_eps')
print(table)
print('\n')
f.write(table)
f.close()
# calculate avg eps
avgeps = np.mean(eps_list, axis=1).reshape((1, 6))
varianceeps = np.std(eps_list, axis=1).reshape((1, 6))
avgeps = np.vstack((avgeps, varianceeps))
filepath = os.path.join(path, 'epsavg')
f = open(filepath, 'w')
table = tabulate(avgeps, headers=col_name, tablefmt="grid", showindex="always")
print('pertubation_avg_eps')
print(table)
print('\n')
f.write(table)
f.close()


# separate(f_list, J, mesh, N, dt, t_init, T, nu)
# separate_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, penalty_epsilon)
# ensemble(f_list, J, mesh, space_size, t_init, dt, T, nu)
ensemble_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon)
ensemble_EVV(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, type=1)
ensemble_EVV(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, type=2)
ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 1)
ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 1)
ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 0)
ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 0)
# ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 1, type2 = 2)
# ensemble_EVV_penalty(f_list, J, mesh, space_size, dt, t_init, T, nu, mu, beta, penalty_epsilon, type1 = 2, type2 = 2)
