import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
from Test1 import *
from ensemble_EVV_time import *
from seperate_time import *
from ensemble_time import *
from ensemble_EVV_penalty_time import *
from mesh_create import *
from save_data import *
import csv
from tabulate import tabulate
file1 = 'function_list/eps'
file2 = 'function_list/test.h5'
J = 2
Re = 800
nu = 1.0 / Re

u1_exact, u2_exact, p_exact = exactsolution1()
# J NSE exact solution
pertubation_eps = 1e-3
eps, u_list, p_list, f_list, u_exact_avg = get_exact1(u1_exact, u2_exact, p_exact, nu, 2, J, pertubation_eps)

