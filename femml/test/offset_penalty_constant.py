from dolfin import *
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import sympy as sp    
from mshr import *
import pdb

#Declare initial setup parameters and stuff
t_init = 0.0
t_final = 16.0
dt = .0005
t_num = int((t_final-t_init)/dt)
nu = 1./100. 
eps_be = dt/1.
t = t_init
TOL = 1.e-10
tol_con = .01 #Size of div(u)???? 

#Generalized Offset Cylinders
circle_outx = 0.0
circle_outy = 0.0
circle_outr = 1.0
circle_inx = 0.5
circle_iny = 0.0
circle_inr = 0.1


N =55

domain = Circle(Point(circle_outx,circle_outy),circle_outr,200) - Circle(Point(circle_inx,circle_iny),circle_inr,50)
mesh = generate_mesh ( domain, N )

measure = assemble(1.0*dx(mesh)) #size of the domain (for normalizing pressure)

n = FacetNormal(mesh)

#Finite Element Space Creation for no penalty method
X = VectorFunctionSpace(mesh,"CG",2) # Velocity space
Q = FunctionSpace(mesh,"CG",1) # Pressure space


#Useful Expressions
def a_1(u,v):  # Viscous term (grad(u),grad(v))
    return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):     # Skew-symmetrized nonlinear term
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))   
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 


#Solution storage
    # Backward Euler
un = Function(X)
pn = Function(Q)
unPlus1 = Function(X)
pnPlus1 = Function(Q)

#Test functions and trial functions
# Finite element functions
u = TrialFunction(X)
p = TrialFunction(Q)

v = TestFunction(X)
q = TestFunction(Q)


####Define Functions For initial conditions
####Load in initial conditions
#filename_init_v = './Initializing/velocity_init'+str(N)+'.txt'
#initial_conV = np.loadtxt(filename_init_v) #Load in intitial condition in this example we chose not to start from T = 0
##Copy over initial conditions
#un.vector()[:] = np.array((initial_conV[:]))


#Boundary Condition
def u0_boundary(x, on_boundary):
    return on_boundary


#This picks out a node to define the pressure BC, same code should work for any domain
mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]))
originpoint = OriginPoint()

noslip = Constant((0.0, 0.0))

bcu = DirichletBC(X,noslip,u0_boundary) # Velocity boundary condition
#bcp = DirichletBC(Q, 0.0, originpoint, 'pointwise')
#bcs = [bcu,bcp]
#Smooth bridge (to allow force to increase slowly)
def smooth_bridge(t):
    if(t>1+1e-14):
        return 1.0
    elif(abs(1-t)>1e-14):
        return np.exp(-np.exp(-1./(1-t)**2)/t**2)
    else:
        return 1.0

f = Expression(("mint*(-4*x[1]*(1-pow(x[0],2)-pow(x[1],2)))",\
					"mint*(4*x[0]*(1-pow(x[0],2)-pow(x[1],2)))"),degree = 4, mint= 0.0)

#Lift and drag eventually
vdExp = Expression(("0","0"),degree=2)
vlExp =Expression(("0","0"),degree=2)
vd = interpolate(vdExp,X)
vl = interpolate(vlExp,X)
class circle_boundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and ((x[0]-.5)**2 + (x[1])**2 < 0.01 + 3*DOLFIN_EPS)
circle = DirichletBC(X,(0.,-1.),circle_boundary())
circle.apply(vd.vector())
circle = DirichletBC(X,(1.,0.),circle_boundary())
circle.apply(vl.vector())

ldtime = int(4./dt)
drag_arr = np.zeros((ldtime))
lift_arr = np.zeros((ldtime))
jj = 0
#System to solve

A_u = None
B_u = None
B_p = None




#Dummy solution vector (for mixed solutions)
#w = Function(X)  
count = 0
#Timestepping--we do a while loop, could also choose a for loop (for jj in range(1,t_num): )
frameNum = 40 # per second
frameRat = int(1.0/(frameNum*dt))

ufile = File("BE_penalty/resultsV/velocity.pvd")
pfile = File("BE_penalty/resultsP/pressure.pvd")



while t <= t_final + TOL: #add TOL so we don't get issues with machine epsilon order errors
    
    mint_val = smooth_bridge(t)
    f.mint = mint_val
    #pdb.set_trace()
    print('Numerical Time Level: t = '+ str(t))
    
    # LHS velocity
    a_u = (1./dt)*inner(u,v)*dx+nu*a_1(u,v)*dx+(1./eps_be)*div(u)*div(v)*dx + b(un,u,v)*dx 

    ## LHS pressure 
    a_p = eps_be*p*q*dx

    # RHS velocity
    b_u = (1./dt)*inner(un,v)*dx+inner(f,v)*dx
    # RHS pressure
    b_p = div(unPlus1)*q*dx

    #Create system to solve
    A_u = assemble(a_u) #Make LHS a matrix (basically)
    B_u = assemble(b_u) #Make RHS a vector (basically)
#    A_p = assemble(a_p)
#    B_p = assemble(b_p,tensor=B_p)
    #Solve system
    bcu.apply(A_u,B_u)   #Boundary condtions must apply 
    solve(A_u,unPlus1.vector(),B_u) #Solve the linear system Aw = B
#    solve(A_p,pnPlus1.vector(),B_p)
#    
    print(sqrt(assemble(pow(div(unPlus1),2)*dx)))
    
        
    #Advance the time
    count=count+1
    un.assign(unPlus1) # set un = unP1     
#    pn.assign(pnPlus1)
    t = t + dt

        
    if(count == int(12./dt)):
        print("I collected an initial condition at t =" + str(t))
        #pdb.set_trace()
        filename_init_v = './Initializing/velocity_init'+str(N)+'.txt'
#        filename_init_p = './Initializing/pressure_init'+str(N)+'.txt'
        u_init_hold = unPlus1.vector().get_local()
#        p_init_hold = pnPlus1.vector().get_local()
        np.savetxt(filename_init_v,u_init_hold)
#        np.savetxt(filename_init_p,p_init_hold)
##    
    if count >= int(1./dt):
        drag_arr[jj] = assemble(nu*a_1(unPlus1,vd)*dx + convect(unPlus1,unPlus1,vd)*dx - c(pnPlus1,vd)*dx)
        lift_arr[jj] = assemble(nu*a_1(unPlus1,vl)*dx + convect(unPlus1,unPlus1,vl)*dx - c(pnPlus1,vl)*dx)
        jj = jj + 1

    # if count%frameRat==0:
    #     ufile << (unPlus1,t)
#        pfile << (pnPlus1,t)




np.savetxt('Initializing/lift_65_0005.txt',lift_arr)
np.savetxt('Initializing/drag_65_0005.txt', drag_arr)
































