# importing os module
import os
import sympy as sp
from dolfin import *
import numpy as np
from mesh_create import flowovercylinder
from Test2 import force1, flow
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
dt = 0.0025
T = 8.0
t_num = int((T-t_init)/dt)
nu = 1./1000.
TOL = 1.e-20
tol_con = .01


# Define mesh
N = 64
mesh = flowovercylinder(N)
space_size = mesh.hmin()
print(space_size)
space_size = mesh.hmax()
print(space_size)
measure = assemble(1.0 * dx(mesh)) #size of the domain (for normalizing pressure)
plot(mesh)
plt.show()
# separate_reference(f_list, mesh, N, dt, t_init, T, nu)
# separate_reference1(f_list, mesh, N, dt, t_init, T, nu)
