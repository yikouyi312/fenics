from Test1 import *
from runner import *
from separate_time_test import *
# parameter setting
N_NODES = 50 # Uniform meshes with 10 nodes per side on the boundary
TIME_STEP = 0.05
T = 0.5  # end time 1
# Characteristic values for velocity, length
Re = 800
uRef = 1.0
lRef = 1.0  # L
nu = (uRef * lRef) / Re  # Kinematic viscocity
# hyperparameter setting
mu = 1.0
beta = 1e6
J = 1
penalty_epsilon = 1e-3
# exact solution
u1_exact, u2_exact, p_exact = exactsolution1(m=1, n=1, alpha=0)
# J NSE exact solution
pertubation_eps = 1e-3
set_seed = 320
eps, u_list, p_list, f_list, u_exact_avg = get_exact1(u1_exact, u2_exact, p_exact, nu, 4, J, pertubation_eps, set_seed)
eps, u1_list, u2_list, p_list, f1_list, f2_list, u1_exact_avg, u2_exact_avg\
    = get_exact2(u1_exact, u2_exact, p_exact, nu, 4, J, pertubation_eps, set_seed)
# Define mesh
mesh, space_size = mesh_create(N_NODES)
# plot(mesh)
#create fold
parent_directory = "/Users/boweiouyang/Desktop/femml/time/output"
directory = "test"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)

time_A, time_b, time_solve, time_other, time_all = \
    separate_time_test(u1_list, u2_list, p_list, f1_list, f2_list, u1_exact_avg, u2_exact_avg, J, mesh, TIME_STEP, T, nu)

print(time_A)
print(time_b)
print(time_solve)
print(time_other)
print(time_all)