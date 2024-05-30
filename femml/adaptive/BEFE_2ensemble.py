#Kiera & Xihui
#BEFE Attempt 1
#Doesn't yet include ensembles--just Green Taylor Vortex
#Sections copied from tutorial from Michael Mclaughlin

from dolfin import *
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import sympy as sp    
# from mshr import *
import pdb
###The above section is just things to import to make everything work nicely, I generally just copy this from old code every time


#Here, we initialize everything we need for the timestepping/flow
Re = 100.0 #Reynold's number
nu = 1./Re #viscosity
Pi = np.pi #pi, but to high accuracy
t_num = 200 #total iterations
t_0 = 0.0 #initial time
t_f = 1.0 #final time
t = t_0 #current time (start at t_0)
dt = (t_f - t_0)/t_num #delta t
TOL = 1.e-10
eps_init = 1.e-3


#Here, we set up the mesh. This is different than most fenics code as it's specifically a unit square. Commented out is a more general approach. 
m = 20
mesh = UnitSquareMesh(m,m)

## Define domain
#Length = 1.0
#Width = 1.0
#
#domain = Rectangle(Point(0.0, 0.0), Point(Length, Width)) 
#mesh = generate_mesh(geometry, m)

#Finite Element Space Creation for no penalty method
X = VectorElement('Lagrange',mesh.ufl_cell(),2) #Velocity        
Q = FiniteElement('Lagrange',mesh.ufl_cell(),1) #Pressure        
V = FunctionSpace(mesh,MixedElement([X,Q])) #Mixed   

##Finite Element Space Creation for Penalty method (Decoupled Velocity and Pressure)
#X1 = VectorFunctionSpace(mesh,"CG",2) #Velocity
#Q1 = FunctionSpace(mesh,"CG",1) #Pressure
#


# Exact solutions, forcing, etc. (THis is copied directly from Michael's code)
x,y,s = sp.symbols('x[0] x[1] s')                                                                  # Variable substitution
u1_exact = -sp.cos(Pi*x)*sp.sin(Pi*y)*sp.exp(-2.0*(Pi**2)*nu*s)                                    # Exact velocity, x-component
u2_exact = sp.sin(Pi*x)*sp.cos(Pi*y)*sp.exp(-2.0*(Pi**2)*nu*s)                                     # Exact velocity, y-component
p_exact = (-1./4)*(sp.cos(2*Pi*x)+sp.cos(2*Pi*y))*sp.exp(-4.0*(Pi**2)*nu*s)                        # Exact pressure
f1 = u1_exact.diff(s,1)+u1_exact*u1_exact.diff(x,1)+u2_exact*u1_exact.diff(y,1) \
    -nu*sum(u1_exact.diff(xi,2) for xi in (x,y))+p_exact.diff(x,1)                                 # Forcing, x-component
f2 = u2_exact.diff(s,1)+u1_exact*u2_exact.diff(x,1)+u2_exact*u2_exact.diff(y,1) \
    -nu*sum(u2_exact.diff(xi,2) for xi in (x,y))+p_exact.diff(y,1)                                 # Forcing, y-component

u1_exact = sp.simplify(u1_exact)                                                                   # Velocity simplification, x-component
u2_exact = sp.simplify(u2_exact)                                                                   # Velocity simplification, y-component
p_exact = sp.simplify(p_exact)                                                                     # Pressure simplification
f1 = sp.simplify(f1)                                                                               # Forcing simplification
f2 = sp.simplify(f2)                                                                               # Forcing simplification

u_exact = Expression((sp.printing.ccode(u1_exact),sp.printing.ccode(u2_exact)),degree=4,s=t)       # Exact velocity expression
p_exact = Expression(sp.printing.ccode(p_exact),degree=4,s=t)                                      # Exact pressure expression
f = Expression((sp.printing.ccode(f1),sp.printing.ccode(f2)),degree=4,s=t)                         # Exact forcing expression


#Useful Expressions
def a_1(u,v):  # Viscous term (grad(u),grad(v))
    return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):     # Skew-symmetrized nonlinear term
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))   
def convect(u,v,w):     # Nonlinear term (used for lift/drag if we're doing that)
    return inner(dot(u,nabla_grad(v)),w)

#Solution storage: u1
#Timestep n+1
tnPlus1_solutions = Function(V)  
(u1nP1,p1nP1) = tnPlus1_solutions.split(True)  
#Timestep n
tn_solutions = Function(V)  
(u1n,p1n) = tn_solutions.split(True) 

#Solution Storage: u2
#Timestep n+1
tnPlus1_solutions2 = Function(V)  
(u2nP1,p2nP1) = tnPlus1_solutions2.split(True)  
#Timestep n
tn_solutions2 = Function(V)  
(u2n,p2n) = tn_solutions2.split(True) 

#Test functions and trial functions
(u,p) = TrialFunctions(V)    # Trial functions
(v,q) = TestFunctions(V)      # Test functions

#Initial condition 
#(I'm not sure if it's the same one you used, but its the exact solution at t= 0 I believe)
u1n.assign((1+eps_init)*interpolate(u_exact,V.sub(0).collapse()))  # Sets initial velocity
u2n.assign((1-eps_init)*interpolate(u_exact,V.sub(0).collapse())) 
p1n.assign(((1+eps_init)**2)*interpolate(p_exact,V.sub(1).collapse()))  # Sets initial velocity
p2n.assign(((1-eps_init)**2)*interpolate(p_exact,V.sub(1).collapse())) 
t += dt #We un = u(0), so solving for unp1 is at time dt 

#Boundary Condition
def boundary(x,on_boundary):   # Boundary definition
    return on_boundary

#This picks out a node to define the pressure BC, same code should work for any domain
mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]))
originpoint = OriginPoint()


bc_u = DirichletBC(V.sub(0),u_exact,boundary)   # Velocity boundary condition
bc_p = DirichletBC(V.sub(1),p_exact,originpoint,'pointwise')   # Pressure boundary condition
bcs = [bc_u,bc_p]  #Both boundary conditions together

###Set up right hand side and left hand side--currently not ensemble as I'm less sure how to do that exactly
#This is exactly what it looks like--the right and left hand side of the time discretized NSE: All terms with u (what we solve for) on LHS, all terms independant on RHS. 

#Dummy solution vector (for mixed solutions)
w = Function(V)  
vel_err_array = np.zeros((t_num,1))
press_err_array = np.zeros((t_num,1))
cnt = 0
#Timestepping--we do a while loop, could also choose a for loop (for jj in range(1,t_num): )
while t <= t_f + TOL: #add TOL so we don't get issues with machine epsilon order errors
    print('Numerical time level: t = ',t)   #Tells us what our current time is 

    #Update exact solution, force
    u_exact.s = t
    p_exact.s = t
    f.s = t
    # LHS
    a = (1./dt)*inner(u,v)*dx + b((u1n+u2n)/2.,u,v)*dx + nu*a_1(u,v)*dx - inner(p,div(v))*dx - inner(div(u),q)*dx

    # RHS
    b1 = (1./dt)*inner(u1n,v)*dx+inner(f,v)*dx-b(u1n-((u1n+u2n)/2.),u1n,v)*dx
    b2 = (1./dt)*inner(u2n,v)*dx+inner(f,v)*dx-b(u2n-((u1n+u2n)/2.),u2n,v)*dx

    #Create system to solve
    A = assemble(a) #Make LHS a matrix (basically)
    B1 = assemble(b1) #Make RHS a vector (basically)
    B2 =assemble(b2)
    #Solve system
    [bc.apply(A,B1) for bc in bcs]   #Boundary condtions must apply 
    solve(A,w.vector(),B1) #Solve the linear system Aw = B
    (u1nP1,p1nP1) = w.split(True)   #Split w into pressure and velocity
    
    [bc.apply(A,B2) for bc in bcs] 
    solve(A,w.vector(),B2) #Solve the linear system Aw = B
    (u2nP1,p2nP1) = w.split(True)    
#    
    # Error calculations 
    vel_error = sqrt(assemble(inner(u1nP1/2.+u2nP1/2.-u_exact,u1nP1/2.+u2nP1/2.-u_exact)*dx)) 
    vel_err_array[cnt] = vel_error
    #square root of integral(|unP1-uexact|)^2--> Absolute L2 error (for average)
    print('Absolute velocity error is: ',vel_error)  
    #Relative error here
    vel_error2 = vel_error/sqrt(assemble(inner(u_exact,u_exact)*dx(mesh))) 
    print('Relative velocity error is: ',vel_error2) 
    #L1 pressure error
    press_error = np.sqrt(assemble(((p1nP1/2.+p2nP1/2.-p_exact)**2)**.5*dx(mesh))) 
    press_err_array[cnt] = press_error
    print('Absolute pressure error is: ',press_error)  
    #L1 relative pressure error 
    press_error2 = vel_error/np.sqrt(assemble((p_exact**2)**.5*dx(mesh))) 
    print('Relative pressure error is: ',press_error2) 
    cnt = cnt+1
    
    
    #Advance the time
    t += dt
    
    u1n.assign(u1nP1) # set un = unP1     
    p1n.assign(p1nP1) # set pn = pnP1
    u2n.assign(u2nP1) # set un = unP1     
    p2n.assign(p2nP1) # set pn = pnP1
    
    #Note: printing takes time, so if you want to test actual computing time, comment out print statements probably
    
print('L2 space, Linfinity time velocity error is: ', max(vel_err_array))    
print('L1 space, Linfinity time pressure error is: ', max(press_err_array))     












