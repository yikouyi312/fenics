# importing os module
import os
from dolfin import *
import numpy as np
from mesh_create import offsetcylinder
from Test1 import force1, pertubation, get_list
from seperate import separate_reference
from runner import runner
# Define mesh
N = 50
M = 25
mesh = offsetcylinder(N, M)

# Generate force function list(ensemble member number = J)
J = 10
# force function
f1, f2 = force1()
# pertubation function
perx, pery = pertubation()
# J force functions
order = 4
tol = 1e-3
seed = 1024
issymmetry = True
eps_list, f_list = get_list(f1, f2, perx, pery, order, J, tol, seed, issymmetry)

# parameter setting
t_init = 12.0
T = 16.0
dt = 0.025
t_num = int((T-t_init)/dt)
nu = 1./800.
TOL = 1.e-10
tol_con = .01

# Define mesh
N = 25
M = 10
mesh = offsetcylinder(N, M)
space_size = mesh.hmin()
measure = assemble(1.0 * dx(mesh)) #size of the domain (for normalizing pressure)
print('mesh area = ' + str(measure))
#create fold
parent_directory = "/Users/boweiouyang/Desktop/femml/ensemble_flow/output"
if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)
directory = "test"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)
# hyper-parameter
penalty_epsilon = 1e-3
mu = 1.0
beta = 1.0


# for i in range(J):
#     input_file = HDF5File(mesh.mpi_comm(), './Initializing/velocity_init/' + "u" + str(i) + ".h5", "r")
#     input_file.read(u_prev[i], "solution")
#     input_file.close()
#     filename_init_v = './Initializing/velocity_init/' + 'separate' + str(i) + 'member' + 'check' + '.txt'
#     u_init_hold = u_prev[i].vector().get_local()
#     np.savetxt(filename_init_v, u_init_hold)

