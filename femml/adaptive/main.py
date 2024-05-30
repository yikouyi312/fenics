from fenics import *
import numpy as np
# Create mesh and define function space
m = 10
mesh = UnitSquareMesh(m, m)
# create finite element function space V, 'P' = 'Lagrange', implying the standard Lagrange family of elements,
# third argument specifies the degree of the finite element
#  V.dim()
V = FunctionSpace(mesh, 'P', 1)
# Defining the trial and test function
u = TrialFunction(V)
v = TestFunction(V)
# Define exact solution
# Expression(formula, degree = )
# Expression is used to represent an exact solution which is used to evaluate the accuracy of a computed solution
# a higher degree must be used for the expression (one or two degrees
# higher)
utrue = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
# Define source function
f = Expression('-6', degree=0)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, utrue, boundary)

# Defining the variational problem
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Forming and solving the linear system
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u, title='Finite element solution')
plot(mesh, title='Finite element mesh' )
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
