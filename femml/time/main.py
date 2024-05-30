from Test1 import *
from runner_time import *
from ensemble_time import *
# parameter setting
N_NODES = 80 # Uniform meshes with 10 nodes per side on the boundary
TIME_STEP = 0.05
T = 2  # end time 1
# Characteristic values for velocity, length
Re = 2500 #800
uRef = 2 * np.pi
lRef = 2 * np.pi  # L
nu = (uRef * lRef) / Re  # Kinematic viscocity
# hyperparameter setting
J = 25
mu = 0.88 * J #1.0
beta = 66000000 #1e6
penalty_epsilon = 1e-3
# Pertubation setting
pertubation_eps = 5e-2#1e-3
set_seed = 320
issymmetry = True
# exact solution
u1_exact, u2_exact, p_exact = exactsolution1(m=1, n=1, alpha=0)
# J NSE exact solution
eps, u_list, p_list, f_list, u_exact_avg = get_exact_random(u1_exact, u2_exact, p_exact, nu, 4, J, pertubation_eps, set_seed, )
# Define mesh
mesh, space_size = mesh_create(N_NODES)
# plot(mesh)
#create fold
parent_directory = "/Users/boweiouyang/Desktop/femml/time/output"
directory = "test_true80"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)

runner_time(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, space_size, T, nu, mu, beta, penalty_epsilon, path)

