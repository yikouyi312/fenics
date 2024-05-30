from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

def offsetcylinder(N, M):
    # Generalized Offset Cylinders
    circle_outx = 0.0
    circle_outy = 0.0
    circle_outr = 1.0
    circle_inx = 0.5
    circle_iny = 0.0
    circle_inr = 0.1
    domain = Circle(Point(circle_outx, circle_outy), circle_outr, 200) - Circle(Point(circle_inx, circle_iny),
                                                                                circle_inr, 50)
    mesh = generate_mesh(domain, N)
    return mesh

class circle_boundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and ((x[0]-.5)**2 + (x[1])**2 < 0.01 + 3*DOLFIN_EPS)

def flowovercylinder(N):
    channel = Rectangle(Point(0.0), Point(2.2, 0.41))
    cylindar = Circle(Point(0.2, 0.2), 0.05)
    domain = channel - cylindar
    mesh = generate_mesh(domain, N)
    return mesh

