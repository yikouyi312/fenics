from fenics import *
from dolfin import *
import numpy as np
# Parameter
m = 10
nu = 1
eps = 0.1
# Define exact solution
# Expression(formula, degree = )
# Expression is used to represent an exact solution which is used to evaluate the accuracy of a computed solution
# a higher degree must be used for the expression (one or two degrees
# higher)
utrue = Expression(('1 + x[0]*x[0] + 2*x[1]*x[1]', '2+x[0]*x[0]+2*x[1]*x[1]'), degree=2)
# Define source function
f = Expression(('-6', '1'), degree=0)
# Create mesh and define function space
mesh = UnitSquareMesh(m, m)
plot(mesh)
# create finite element function space V, 'P' = 'Lagrange', implying the standard Lagrange family of elements,
# third argument specifies the degree of the finite element
#  V.dim()
V = VectorFunctionSpace(mesh, 'P', 2)
W = FunctionSpace(mesh, 'P', 1)
# Defining the trial and test function
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(W)
q = TestFunction(W)
# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, utrue, boundary)

# Defining the variational problem
# LHS
a = inner(grad(u), grad(v))*dx + (1./eps)*div(u)*div(v) *dx
# RHS
L = inner(f, v) * dx
# Forming and solving the linear system
u = Function(V)
solve(a == L, u, bc)
# LHS
a = eps*p*q*dx
# RHS
L = -div(u)*q*dx
p = Function(W)
solve(a == L, p)

# Plot solution and mesh
# plot(u, title='Finite element solution')
# plot(mesh, title='Finite element mesh' )
plot(p)
# Plotting the solution using ParaView
# vtkfile = File('poisson/solution.pvd')
# vtkfile << u

# Computing the error
# Compute error in L2 norm
error_L2 = errornorm(utrue, u, 'L2')
# Compute maximum error at vertices
vertex_values_utrue = utrue.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_utrue - vertex_values_u))
# Examining degrees of freedom and vertex values
# degrees of freedom
nodal_values_u = u.vector()
# convert the Vector object to a standard numpy array
array_u = np.array(nodal_values_u)

