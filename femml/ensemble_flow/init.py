# importing os module
import os
import sympy as sp
from dolfin import *
import numpy as np
from mesh_create import offsetcylinder
from Test1 import force1
from seperate import separate_reference, separate_reference1
import matplotlib.pyplot as plt
# Generate force function
# force function
f1, f2 = force1()
x, y, s = sp.symbols('x[0] x[1] s')
t = 0.0
f_list = Expression((sp.printing.ccode(f1), sp.printing.ccode(f2)), degree=4, s=t)

# parameter setting
t_init = 0.0
T = 16.0
dt = 0.0125
t_num = int((T-t_init)/dt)
nu = 1./800.
TOL = 1.e-10
tol_con = .01


# Define mesh
N = 50
M = 10
mesh = offsetcylinder(N, M)
space_size = mesh.hmin()
measure = assemble(1.0 * dx(mesh)) #size of the domain (for normalizing pressure)
plot(mesh)
plt.show()
# separate_reference1(f_list, mesh, N, dt, t_init, T, nu)
# separate_reference(f_list, mesh, N, dt, t_init, T, nu)
