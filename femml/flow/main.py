import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from Test2 import *
from ensemble_EVV import *
from seperate import *
from ensemble import *
from ensemble_EVV_penalty import *
from mesh_create import *

# parameter setting
t_init = 0.0
t_final = 16.0
dt = 0.025
t_num = int((t_final-t_init)/dt)
nu = 1./800.
TOL = 1.e-10
tol_con = .01
J = 1
# force function
f1, f2 = forcefunction()
# J force functions
order = 4
tol = 0
eps_list, f_list = get_list(f1, f2, order, J, tol)
# Define mesh
N = 50
M = 10
mesh = offsetcylinder(N, M)
# solve nse with separate method
seperate(f_list, J, mesh, dt, t_init, t_final, nu)

# solve nse with ensemble method

# solve nse with ensemble_evv method

# solve nse with ensemble_evv_penalty method




