import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import sympy as sp
from helper import *
import os
from mesh_create import *
from tabulate import tabulate
from seperate import *
from ensemble import *
from ensemble_EVV import *
from ensemble_EVV_penalty import *
from exact import *

def runner(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, space_size, T, nu, mu, beta, penalty_epsilon,
           parent_path, n):
    # initial dataset
    Uerror_L2_data = np.array([])
    GradUerror_L2_data = np.array([])
    Uerror_inf_data = np.array([])
    U_L2_data = np.array([])
    U_Grad_data = np.array([])
    U_inf_data = np.array([])
    U_div_L2_data = np.array([])
    U_div_inf_data = np.array([])

    Perror_L2_data = np.array([])
    GradPerror_L2_data = np.array([])
    Perror_inf_data = np.array([])
    P_L2_data = np.array([])
    P_Grad_data = np.array([])
    P_inf_data = np.array([])

    u_x_dif = np.array([])
    u_y_dif = np.array([])

    exact_L2_list, exact_Grad_list, exactAvgU_L2_list, exactAvgU_Grad_list = exact(u_list, u_exact_avg,  J, mesh, TIME_STEP, T)
    # solve nse with separate method
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list , AvgU_div_L2_list, \
    u_x, u_y, avg_u_x, avg_u_y, u_x_true, u_y_true, avg_u_x_true, avg_u_y_true\
        = seperate(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu)
    print('separete calculation Finished!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if  exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
    plot
    """
    t = np.linspace(0, T, int((T - 0) // TIME_STEP + 1))
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]

    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i+5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x_true[n])
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y_true[n])
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x_true)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y_true)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.concatenate((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.concatenate((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.concatenate((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.concatenate((U_L2_data, U_L2))
    U_Grad_data = np.concatenate((U_Grad_data, U_Grad))
    U_inf_data = np.concatenate((U_inf_data, U_inf))
    U_div_L2_data = np.concatenate((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.concatenate((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.concatenate((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.concatenate((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.concatenate((Perror_inf_data, Perror_inf))
    P_L2_data = np.concatenate((P_L2_data, P_L2))
    P_Grad_data = np.concatenate((P_Grad_data, P_Grad))
    P_inf_data = np.concatenate((P_inf_data, P_inf))

    AvgUerror_L2_data = np.array([AvgUerror_L2])
    AvgGradUerror_L2_data = np.array([AvgGradUerror_L2])
    AvgUerror_inf_data = np.array([AvgUerror_inf])
    AvgU_L2_data = np.array([AvgU_L2])
    AvgU_Grad_data = np.array([AvgU_Grad])
    AvgU_inf_data = np.array([AvgU_inf])
    AvgU_div_L2_data = np.array([AvgU_div_L2])
    AvgU_div_inf_data = np.array([AvgU_div_inf])

    u_x_dif = u_x_true[:][-1] - u_x[:][-1]
    u_y_dif = u_y_true[:][-1] - u_y[:][-1]

    # solve nse with ensemble method
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list,\
           u_x, u_y, avg_u_x, avg_u_y\
        = ensemble(u_list, p_list, f_list, u_exact_avg, J, mesh, TIME_STEP, T, nu)
    print('Ensemble finish!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
        plot
        """
    t = np.linspace(0, T, int((T - 0) // TIME_STEP + 1))
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]
    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i + 5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.vstack((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.vstack((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.vstack((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.vstack((U_L2_data, U_L2))
    U_Grad_data = np.vstack((U_Grad_data, U_Grad))
    U_inf_data = np.vstack((U_inf_data, U_inf))
    U_div_L2_data = np.vstack((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.vstack((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.vstack((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.vstack((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.vstack((Perror_inf_data, Perror_inf))
    P_L2_data = np.vstack((P_L2_data, P_L2))
    P_Grad_data = np.vstack((P_Grad_data, P_Grad))
    P_inf_data = np.vstack((P_inf_data, P_inf))

    AvgUerror_L2_data = np.vstack((AvgUerror_L2_data, AvgUerror_L2))
    AvgGradUerror_L2_data = np.vstack((AvgGradUerror_L2_data, AvgGradUerror_L2))
    AvgUerror_inf_data = np.vstack((AvgUerror_inf_data, AvgUerror_inf))
    AvgU_L2_data = np.vstack((AvgU_L2_data, AvgU_L2))
    AvgU_Grad_data = np.vstack((AvgU_Grad_data, AvgU_Grad))
    AvgU_inf_data = np.vstack((AvgU_inf_data, AvgU_inf))
    AvgU_div_L2_data = np.vstack((AvgU_div_L2_data, AvgU_div_L2))
    AvgU_div_inf_data = np.vstack((AvgU_div_inf_data, AvgU_div_inf))

    u_x_dif = np.vstack((u_x_dif, u_x_true[:][-1] - u_x[:][-1]))
    u_y_dif = np.vstack((u_y_dif, u_y_true[:][-1] - u_y[:][-1]))

    # solve nse with ensemble_evv method
    type = 1
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list, \
    u_x, u_y, avg_u_x, avg_u_y\
        = ensemble_EVV(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, type)
    print('ensemble_EVV_type1 finished!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
        plot
        """
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]
    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i + 5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.vstack((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.vstack((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.vstack((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.vstack((U_L2_data, U_L2))
    U_Grad_data = np.vstack((U_Grad_data, U_Grad))
    U_inf_data = np.vstack((U_inf_data, U_inf))
    U_div_L2_data = np.vstack((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.vstack((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.vstack((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.vstack((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.vstack((Perror_inf_data, Perror_inf))
    P_L2_data = np.vstack((P_L2_data, P_L2))
    P_Grad_data = np.vstack((P_Grad_data, P_Grad))
    P_inf_data = np.vstack((P_inf_data, P_inf))

    AvgUerror_L2_data = np.vstack((AvgUerror_L2_data, AvgUerror_L2))
    AvgGradUerror_L2_data = np.vstack((AvgGradUerror_L2_data, AvgGradUerror_L2))
    AvgUerror_inf_data = np.vstack((AvgUerror_inf_data, AvgUerror_inf))
    AvgU_L2_data = np.vstack((AvgU_L2_data, AvgU_L2))
    AvgU_Grad_data = np.vstack((AvgU_Grad_data, AvgU_Grad))
    AvgU_inf_data = np.vstack((AvgU_inf_data, AvgU_inf))
    AvgU_div_L2_data = np.vstack((AvgU_div_L2_data, AvgU_div_L2))
    AvgU_div_inf_data = np.vstack((AvgU_div_inf_data, AvgU_div_inf))

    u_x_dif = np.vstack((u_x_dif, u_x_true[:][-1] - u_x[:][-1]))
    u_y_dif = np.vstack((u_y_dif, u_y_true[:][-1] - u_y[:][-1]))

    type = 2
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list,\
           u_x, u_y, avg_u_x, avg_u_y\
        = ensemble_EVV(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, type)
    print('ensemble_EVV_type2 finished!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
        plot
        """
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]
    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i + 5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.vstack((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.vstack((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.vstack((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.vstack((U_L2_data, U_L2))
    U_Grad_data = np.vstack((U_Grad_data, U_Grad))
    U_inf_data = np.vstack((U_inf_data, U_inf))
    U_div_L2_data = np.vstack((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.vstack((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.vstack((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.vstack((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.vstack((Perror_inf_data, Perror_inf))
    P_L2_data = np.vstack((P_L2_data, P_L2))
    P_Grad_data = np.vstack((P_Grad_data, P_Grad))
    P_inf_data = np.vstack((P_inf_data, P_inf))

    AvgUerror_L2_data = np.vstack((AvgUerror_L2_data, AvgUerror_L2))
    AvgGradUerror_L2_data = np.vstack((AvgGradUerror_L2_data, AvgGradUerror_L2))
    AvgUerror_inf_data = np.vstack((AvgUerror_inf_data, AvgUerror_inf))
    AvgU_L2_data = np.vstack((AvgU_L2_data, AvgU_L2))
    AvgU_Grad_data = np.vstack((AvgU_Grad_data, AvgU_Grad))
    AvgU_inf_data = np.vstack((AvgU_inf_data, AvgU_inf))
    AvgU_div_L2_data = np.vstack((AvgU_div_L2_data, AvgU_div_L2))
    AvgU_div_inf_data = np.vstack((AvgU_div_inf_data, AvgU_div_inf))

    u_x_dif = np.vstack((u_x_dif, u_x_true[:][-1] - u_x[:][-1]))
    u_y_dif = np.vstack((u_y_dif, u_y_true[:][-1] - u_y[:][-1]))
# solve nse with ensemble_evv_penalty method
    type = 1
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list,\
           u_x, u_y, avg_u_x, avg_u_y\
        = ensemble_EVV_penalty(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, beta,
                               penalty_epsilon, type)
    print('ensemble_EVV_penalty_type1 finished!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
        plot
    """
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]
    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i + 5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.vstack((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.vstack((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.vstack((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.vstack((U_L2_data, U_L2))
    U_Grad_data = np.vstack((U_Grad_data, U_Grad))
    U_inf_data = np.vstack((U_inf_data, U_inf))
    U_div_L2_data = np.vstack((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.vstack((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.vstack((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.vstack((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.vstack((Perror_inf_data, Perror_inf))
    P_L2_data = np.vstack((P_L2_data, P_L2))
    P_Grad_data = np.vstack((P_Grad_data, P_Grad))
    P_inf_data = np.vstack((P_inf_data, P_inf))

    AvgUerror_L2_data = np.vstack((AvgUerror_L2_data, AvgUerror_L2))
    AvgGradUerror_L2_data = np.vstack((AvgGradUerror_L2_data, AvgGradUerror_L2))
    AvgUerror_inf_data = np.vstack((AvgUerror_inf_data, AvgUerror_inf))
    AvgU_L2_data = np.vstack((AvgU_L2_data, AvgU_L2))
    AvgU_Grad_data = np.vstack((AvgU_Grad_data, AvgU_Grad))
    AvgU_inf_data = np.vstack((AvgU_inf_data, AvgU_inf))
    AvgU_div_L2_data = np.vstack((AvgU_div_L2_data, AvgU_div_L2))
    AvgU_div_inf_data = np.vstack((AvgU_div_inf_data, AvgU_div_inf))

    u_x_dif = np.vstack((u_x_dif, u_x_true[:][-1] - u_x[:][-1]))
    u_y_dif = np.vstack((u_y_dif, u_y_true[:][-1] - u_y[:][-1]))

    type = 2
    Uerror_L2, GradUerror_L2, Uerror_inf, U_L2, U_Grad, U_inf, U_div_L2, U_div_inf, \
    Perror_L2, GradPerror_L2, Perror_inf, P_L2, P_Grad, P_inf, \
    AvgUerror_L2, AvgGradUerror_L2, AvgUerror_inf, AvgU_L2, AvgU_Grad, AvgU_inf, AvgU_div_L2, AvgU_div_inf, \
    Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list, AvgUerror_L2_list, \
    AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list,\
           u_x, u_y, avg_u_x, avg_u_y\
        = ensemble_EVV_penalty(u_list, p_list, f_list, u_exact_avg, J, mesh, space_size, TIME_STEP, T, nu, mu, beta,
                               penalty_epsilon, type)
    print('ensemble_EVV_penalty_type2 finished!\n')
    Uerror_L2_list = Uerror_L2_list / exact_L2_list
    GradUerror_L2_list = GradUerror_L2_list / exact_Grad_list
    N = int((T - 0.0) // TIME_STEP + 1)
    for i in range(N):
        if exactAvgU_L2_list[i] > 0:
            AvgUerror_L2_list[i] = AvgUerror_L2_list[i] / exactAvgU_L2_list[i]
        if exactAvgU_Grad_list[i] > 0:
            AvgGradUerror_L2_list[i] = AvgGradUerror_L2_list[i] / exactAvgU_Grad_list[i]
    """
            plot
            """
    plot_data_random = [Uerror_L2_list, GradUerror_L2_list, U_L2_list, U_Grad_list, U_div_L2_list]
    plot_data_avg = [AvgUerror_L2_list, AvgGradUerror_L2_list, AvgU_L2_list, AvgU_Grad_list, AvgU_div_L2_list]
    for i in range(len(plot_data_avg)):
        fig = plt.figure(i)
        plt.plot(t, plot_data_random[i][n])
        fig = plt.figure(i + 5)
        plt.plot(t, plot_data_avg[i])
    fig = plt.figure(10)
    plt.plot(t, u_x[n])
    fig = plt.figure(11)
    plt.plot(t, u_y[n])
    fig = plt.figure(12)
    plt.plot(t, avg_u_x)
    fig = plt.figure(13)
    plt.plot(t, avg_u_y)
    fig = plt.figure(14)
    plt.plot(t, u_x_true[n] - u_x[n])
    fig = plt.figure(15)
    plt.plot(t, u_y_true[n] - u_y[n])
    fig = plt.figure(16)
    plt.plot(t, [a - b for a, b in zip(avg_u_x_true, avg_u_x)])
    fig = plt.figure(17)
    plt.plot(t, [a - b for a, b in zip(avg_u_y_true, avg_u_y)])
    """
    combine data 
    """
    Uerror_L2_data = np.vstack((Uerror_L2_data, Uerror_L2))
    GradUerror_L2_data = np.vstack((GradUerror_L2_data, GradUerror_L2))
    Uerror_inf_data = np.vstack((Uerror_inf_data, Uerror_inf))
    U_L2_data = np.vstack((U_L2_data, U_L2))
    U_Grad_data = np.vstack((U_Grad_data, U_Grad))
    U_inf_data = np.vstack((U_inf_data, U_inf))
    U_div_L2_data = np.vstack((U_div_L2_data, U_div_L2))
    U_div_inf_data = np.vstack((U_div_inf_data, U_div_inf))

    Perror_L2_data = np.vstack((Perror_L2_data, Perror_L2))
    GradPerror_L2_data = np.vstack((GradPerror_L2_data, GradPerror_L2))
    Perror_inf_data = np.vstack((Perror_inf_data, Perror_inf))
    P_L2_data = np.vstack((P_L2_data, P_L2))
    P_Grad_data = np.vstack((P_Grad_data, P_Grad))
    P_inf_data = np.vstack((P_inf_data, P_inf))

    AvgUerror_L2_data = np.vstack((AvgUerror_L2_data, AvgUerror_L2))
    AvgGradUerror_L2_data = np.vstack((AvgGradUerror_L2_data, AvgGradUerror_L2))
    AvgUerror_inf_data = np.vstack((AvgUerror_inf_data, AvgUerror_inf))
    AvgU_L2_data = np.vstack((AvgU_L2_data, AvgU_L2))
    AvgU_Grad_data = np.vstack((AvgU_Grad_data, AvgU_Grad))
    AvgU_inf_data = np.vstack((AvgU_inf_data, AvgU_inf))
    AvgU_div_L2_data = np.vstack((AvgU_div_L2_data, AvgU_div_L2))
    AvgU_div_inf_data = np.vstack((AvgU_div_inf_data, AvgU_div_inf))
    u_x_dif = np.vstack((u_x_dif, u_x_true[:][-1] - u_x[:][-1]))
    u_y_dif = np.vstack((u_y_dif, u_y_true[:][-1] - u_y[:][-1]))

    data_list = [Uerror_L2_data, GradUerror_L2_data, Uerror_inf_data, U_L2_data, U_Grad_data, U_inf_data,
                 U_div_L2_data, U_div_inf_data,
                 Perror_L2_data, GradPerror_L2_data, Perror_inf_data, P_L2_data, P_Grad_data, P_inf_data,
                 AvgUerror_L2_data, AvgGradUerror_L2_data, AvgUerror_inf_data, AvgU_L2_data, AvgU_Grad_data,
                 AvgU_inf_data, AvgU_div_L2_data, AvgU_div_inf_data
                 ]
    filename = ['Uerror_L2', 'GradUerror_L2', 'Uerror_inf', 'U_L2', 'U_Grad', 'U_inf', 'U_div_L2', 'U_div_inf',
               'Perror_L2', 'GradPerror_L2', 'Perror_inf', 'P_L2', 'P_Grad', 'P_inf',
                'AvgUerror_L2', 'AvgGradUerror_L2', 'AvgUerror_inf', 'AvgU_L2', 'AvgU_Grad',
                'AvgU_inf', 'AvgU_div_L2', 'AvgU_div_inf'
                ]
    col_name = ['separate', 'ensemble', 'ensemble-EVV-type1', 'EVV-type2', 'EVV-penalty-type1',
               'EVV-penalty-type2']
    for i in range(len(data_list)):
        filepath = os.path.join(parent_path, filename[i])
        f = open(filepath, 'w')
        table = tabulate(np.transpose(np.sqrt(data_list[i])), headers=col_name, tablefmt="grid", showindex="always")
        f.write(table)
        f.close()

    filename = ['cal_Uerror_L2', 'cal_GradUerror_L2', 'cal_Uerror_inf', 'cal_U_L2', 'cal_U_Grad', 'cal_U_inf', 'cal_U_div_L2', 'cal_U_div_inf',
               'cal_Perror_L2', 'cal_GradPerror_L2', 'cal_Perror_inf', 'cal_P_L2', 'cal_P_Grad', 'cal_P_inf']
    for i in range(len(filename)):
        filepath = os.path.join(parent_path, filename[i])
        f = open(filepath, 'w')
        mean = np.mean(np.sqrt(data_list[i]), axis=1)
        std = np.std(np.sqrt(data_list[i]), axis=1)
        final = np.vstack((mean, std))
        table = tabulate(final, headers=col_name, tablefmt="grid", showindex="always")
        f.write(table)
        f.close()
    filepath = os.path.join(parent_path, 'u_x_dif_end')
    f = open(filepath, 'w')
    mean = np.mean(u_x_dif, axis=1)
    std = np.std(u_x_dif, axis=1)
    final = np.vstack((mean, std))
    table = tabulate(final, headers=col_name, tablefmt="grid", showindex="always")
    f.write(table)
    f.close()
    filepath = os.path.join(parent_path, 'u_y_dif_end')
    f = open(filepath, 'w')
    mean = np.mean(u_y_dif, axis=1)
    std = np.std(u_y_dif, axis=1)
    final = np.vstack((mean, std))
    table = tabulate(final, headers=col_name, tablefmt="grid", showindex="always")
    f.write(table)
    f.close()
    """
    plot 
    """

    x_label = 't'
    y_label = [r'$\frac{\|u-u_h\|_{2,0}}{\|u\|_{2,0}} $', r'$\frac{\|\nabla u-\nabla u_h\|_{2,0}}{\|u\|_{2,0}} $', r'$\|u_h\|_{2,0} $',
               r'$\|\nabla u_h \|_{2,0} $', r'$\|\nabla \cdot u_h\|_{2,0} $']
    plot_legend = [ 'Separate', 'Ensemble', 'EEV1', 'EEV2', 'BEEP-EEV3', 'BEEP-EEV4']
    plot_filename = ['Uerror_L2.png', 'GradUerror_L2.png', 'U_L2.png', 'U_Grad.png', 'U_div_L2.png']
    for i in range(len(y_label)):
        fig = plt.figure(i)
        plt.xlabel(x_label)
        plt.ylabel(y_label[i])
        plt.xlim((TIME_STEP, T))
        plt.legend(plot_legend)
        plt.savefig(os.path.join(parent_path, plot_filename[i]))
    plot_filename = ['AvgUerror_L2.png', 'AvgGradUerror_L2.png', 'AvgU_L2.png', 'AvgU_Grad.png', 'AvgU_div_L2.png']
    for i in range(len(y_label)):
        fig = plt.figure(5 + i)
        plt.xlabel(x_label)
        plt.ylabel(y_label[i])
        plt.xlim((TIME_STEP, T))
        plt.legend(plot_legend)
        plt.savefig(os.path.join(parent_path, plot_filename[i]))

    plot_legend = ['True', 'Separate', 'Ensemble', 'EEV1', 'EEV2', 'BEEP-EEV3', 'BEEP-EEV4']
    plot_filename = ['u_x.png', 'u_y.png', 'avg_u_x.png', 'avg_u_y.png']
    y_label = [r'$u_1(0.5,0.5)$', r'$u_2(0.5,0.5)$', r'$u_1(0.5,0.5)$', r'$u_2(0.5,0.5)$']
    for i in range(4):
        fig = plt.figure(10 + i)
        plt.xlabel(x_label)
        plt.ylabel(y_label[i])
        plt.xlim((TIME_STEP, T))
        plt.legend(plot_legend)
        plt.savefig(os.path.join(parent_path, plot_filename[i]))
    plot_legend = ['True', 'Separate', 'Ensemble', 'EEV1', 'EEV2', 'BEEP-EEV3', 'BEEP-EEV4']
    plot_filename = ['u_x_dif.png', 'u_y_dif.png', 'avg_u_x_dif.png', 'avg_u_y_dif.png']
    y_label = [r'$(u_{1, true} - u_1)(0.5,0.5)$', r'$(u_{2, true} - u_2)(0.5,0.5)$', r'$(u_{1, true} - u_1)(0.5,0.5)$',
               r'$(u_{2, true} - u_2)(0.5,0.5)$']
    for i in range(4):
        fig = plt.figure(14 + i)
        plt.xlabel(x_label)
        plt.ylabel(y_label[i])
        plt.xlim((TIME_STEP, T))
        plt.legend(plot_legend)
        plt.savefig(os.path.join(parent_path, plot_filename[i]))


    return True
