from dolfin import *
import numpy as np

def mesh_create(N_NODES):
    # define mesh
    x0 = 0.0
    x1 = 2 * np.pi
    y0 = 0.0
    y1 = 2 * np.pi
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N_NODES, N_NODES, "crossed")
    space_size = (2 * np.pi)/N_NODES
    return mesh, space_size



