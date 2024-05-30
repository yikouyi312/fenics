import numpy as np
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
def mesh_create(N_NODES):
    # define mesh
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N_NODES, N_NODES, "crossed")
    space_size = (1.0)/N_NODES
    return mesh, space_size



