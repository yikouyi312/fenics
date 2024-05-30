from dolfin import *
import sympy as sp
from dolfin import *
import numpy as np
import numpy as np
#list linear solver
#list_linear_solver_methods()
# list_krylov_solver_preconditioners()
# #info(UmfpackLUSolver.default_parameters(), True)
# info(LUSolver.default_parameters(), True)
# #info(PETScLUSolver.default_parameters(), True)
# solver = LinearVariationalSolver()
# solver.parameters.linear_solver = 'gmres'
# solver.parameters.preconditioner = 'ilu'
# prm = solver.parameters.krylov_solver  # short form
# prm.absolute_tolerance = 1E-7
# prm.relative_tolerance = 1E-4
# prm.maximum_iterations = 1000
N_NODES = 100
x0 = 0.0
x1 = 2 * np.pi
y0 = 0.0
y1 = 2 * np.pi
mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N_NODES, N_NODES)

Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
Q = FunctionSpace(mesh, Q_ele)

x, y, s = sp.symbols('x[0] x[1] s')
m = 1
n = 1
t = np.pi / 4.0
u1_exact = - sp.sin(2 * s) * sp.cos(m * x) * sp.sin(n * y)  # u1
u2_exact = sp.sin(2 * s) * sp.sin(m * x) * sp.cos(n * y)  # u2
u1 = Expression(sp.printing.ccode(u1_exact), degree=4,
                      s=t)
u2 = Expression(sp.printing.ccode(u2_exact), degree=4,
                      s=t)
u1 = project(u1, Q)
u2 = project(u2, Q)

u1 = u1 * u1
u2 = u2 * u2
max_u = max(u1, u2)
max_u = project(max_u, Q)
#max_u.vector()[:] = np.maximum(u1.vector()[:], u2.vector()[:])
print(u1.vector()[:])
print(u2.vector()[:])
print(max_u.vector()[:])
x = np.pi
y = 1
u1 = project(u1, Q)
u2 = project(u2, Q)
print(u1(x, y))
print(u2(x, y))
print(max_u(x,y))