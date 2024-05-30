from dolfin import *

def element_create(mesh):
    # create finite element
    V_ele = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    V1_ele = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    Q_ele = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_ele = V_ele * Q_ele
    return V_ele, V1_ele, Q_ele, W_ele