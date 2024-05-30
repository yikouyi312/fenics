import sympy as sp
import os
from Test1 import *
from runner import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
# parameter setting
N_NODES = 10  # Uniform meshes with 10 nodes per side on the boundary
TIME_STEP = 0.05
T = 2.0  # end time 1
# Characteristic values for velocity, length
Re = 2500
uRef = 1.0
lRef = 1.0  # L
nu = (uRef * lRef) / Re  # Kinematic viscocity
# hyperparameter setting
J = 25
mu = 22
beta = 660000
penalty_epsilon = 1e-3
# exact solution
u1_exact, u2_exact, p_exact = exactsolution1(m=1, n=1, alpha=0)
# J NSE exact solution
pertubation_eps = 5 * 1e-2
set_seed = 320
eps, u_list, p_list, f_list, u_exact_avg = get_exact_random(u1_exact, u2_exact, p_exact, nu, 4, J, pertubation_eps, set_seed)
# Define mesh
mesh, space_size = mesh_create(N_NODES)
# plot(mesh)
print(mesh.hmax())
print(mesh.hmin())
parent_directory = "/Users/boweiouyang/Desktop/femml/exactsolution2/output"
directory = "test1_sym_random_test_largebeta"
path = os.path.join(parent_directory, directory)
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '% s' created" % directory)
# create fold
x = np.arange(-0.15, 0.15, 0.001)
# plt.plot(x, stats.norm.pdf(x, 0, pertubation_eps))
plt.hist(eps[0, :], bins=10, edgecolor='black', alpha=0.75)
plt.xlabel('$\delta$')
plt.xlim((-0.15, 0.15))
plt.ylabel('count')
plt.savefig(os.path.join(path, 'eps.png'))
plt.close()
# save data
filepath = os.path.join(path, 'eps')
# Print data and Save data
f = open(filepath, 'w')
col_name = ['velocity1', 'velocity2', 'pressure']
table = tabulate(np.transpose(eps), headers=col_name, tablefmt="grid", showindex="always")
print('pertubation_eps')
print(table)
print('\n')
f.write(table)
f.close()
# calculate avg eps
avgeps = np.mean(eps, axis=1).reshape((1, 3))
varianceeps = np.std(eps, axis=1).reshape((1, 3))
avgeps = np.vstack((avgeps, varianceeps))
filepath = os.path.join(path, 'epsavg')
f = open(filepath, 'w')
col_name = ['velocity1', 'velocity2', 'pressure']
table = tabulate(avgeps, headers=col_name, tablefmt="grid", showindex="always")
print('pertubation_avg_eps')
print(table)
print('\n')
f.write(table)
f.close()
n = 9 #np.random.choice(list(range(J)))
filepath = os.path.join(path, 'parameter')
f = open(filepath, 'w')
f.write('Nodes = ' + str(N_NODES) + '\n')
f.write('nu = ' + str(nu) + '\n')
f.write('J = ' + str(J) + '\n')
f.write('mu = ' + str(mu) + '\n')
f.write('beta = ' + str(beta) + '\n')
f.write('epsilon_penalty = ' + str(penalty_epsilon) + '\n')
f.write(('random select = ' + str(n)) + '\n')
f.write(('seed = ' + str(set_seed)))
f.close()

runner(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, space_size, T, nu, mu, beta, penalty_epsilon, path, n)
