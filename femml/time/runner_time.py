import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
import os
from mesh_create import *
from tabulate import tabulate
from seperate_time import *
from ensemble_time import *
from ensemble_EVV_time import *
from ensemble_EVV_penalty_time import *
import dijitso
from clear_cache import clear as clear_cache
def runner_time(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, space_size, T, nu, mu, beta, penalty_epsilon,
           parent_path):
    time_A_list = np.array([])
    time_b_list = np.array([])
    time_solve_list = np.array([])
    time_other_list = np.array([])
    time_total_list = np.array([])
    time_EVV_list = np.array([])
    time_EVV1_list = np.array([])
    time_split_list = np.array([])
    # solve nse with separate method
    # solve nse with ensemble_evv method
    type = 1
    time_A, time_b, time_solve, time_other, time_total, time_EVV, time_split \
        = ensemble_EVV_time(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, type)

    print('ensemble_EVV_type1 finished!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_EVV_list = np.append(time_EVV_list, time_EVV)
    time_split_list = np.append(time_split_list, time_split)

    # solve nse with ensemble_evv method penalty
    type = 1
    time_A, time_b, time_solve, time_other, time_total, time_EVV, time_EVV1 , time_split\
        = ensemble_EVV_penalty_time(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu,
                                    beta,
                                    penalty_epsilon, type)
    print('ensemble_EVV_penalty_type1 finished!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_EVV_list = np.append(time_EVV_list, time_EVV)
    time_EVV1_list = np.append(time_EVV1_list, time_EVV1)
    time_split_list = np.append(time_split_list, time_split)

    type = 2
    time_A, time_b, time_solve, time_other, time_total, time_EVV, time_split\
        = ensemble_EVV_time(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, type)
    print('ensemble_EVV_type2 finished!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_EVV_list = np.append(time_EVV_list, time_EVV)
    time_split_list = np.append(time_split_list, time_split)

    # solve nse with ensemble method penalty
    type = 2
    time_A, time_b, time_solve, time_other, time_total, time_EVV, time_EVV1, time_split\
        = ensemble_EVV_penalty_time(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu,
                                    beta,
                                    penalty_epsilon, type)

    print('ensemble_EVV_penalty_type2 finished!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_EVV_list = np.append(time_EVV_list, time_EVV)
    time_EVV1_list = np.append(time_EVV1_list, time_EVV1)
    time_split_list = np.append(time_split_list, time_split)

    time_A, time_b, time_solve, time_other, time_total, time_split  \
        = ensemble_time(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu)
    print('Ensemble finish!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_split_list = np.append(time_split_list, time_split)
    # solve nse seperate
    time_A, time_b, time_solve, time_other, time_total, time_split   \
        = seperate_time(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu)
    print('separete calculation Finished!\n')
    time_A_list = np.append(time_A_list, time_A)
    time_b_list = np.append(time_b_list, time_b)
    time_solve_list = np.append(time_solve_list, time_solve)
    time_other_list = np.append(time_other_list, time_other)
    time_total_list = np.append(time_total_list, time_total)
    time_split_list = np.append(time_split_list, time_split)
    filename = 'time'
    col_name = ['EVV-type1', 'penalty-type1', 'EVV-type2', 'penalty-type2',  'ensemble', 'separate']
    data = np.vstack((time_A_list, time_b_list, time_solve_list, time_other_list, time_split_list, time_total_list
                      , time_total_list - time_split_list))
    filepath = os.path.join(parent_path, filename)
    f = open(filepath, 'w')
    table = tabulate(data, headers=col_name, tablefmt="grid", showindex="always")
    f.write(table)
    f.close()
    print(table)
    print('EVV time\n')
    print("EVV-type1:" + str(time_EVV_list[0]) + '\n')
    print("EVV-type2:" + str(time_EVV_list[2]) + '\n')
    print("penalty-type1:" + str(time_EVV_list[1]) + '\n')
    print("penalty-type2:" + str(time_EVV_list[3]) + '\n')
    print('EVV1 time\n')
    print("penalty-type1:" + str(time_EVV1_list[0]) + '\n')
    print("penalty-type2:" + str(time_EVV1_list[1]) + '\n')

    return True
