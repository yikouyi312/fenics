import numpy as np
import matplotlib.pyplot as plt
import os
t_init = 0.0
T = 8.005
TIME_STEP = 0.005
x = np.linspace(t_init, T, int((T - t_init) / TIME_STEP) + 1)
color_map = plt.cm.get_cmap('tab10', 15)
os.chdir('/Users/boweiouyang/Desktop/femml/cylinder1/output')
alpha_list = np.linspace(0.7, 1.0, 11)
linewidth_list = np.linspace(0.7, 1.5, 11)
label_list = [r'$ u_1$', r'$ u_2$', r'$ u_3$', r'$ u_4$', r'$ u_5$', r'$ u_6$', r'$ u_7$', r'$ u_8$', r'$ u_9$', r'$ u_{10}$']
# energy
for i in range(10):
    energy = np.loadtxt('./energy/ensemble_EVV_type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/energy_avg/ensemble_EVV_type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/spha/ensemble_EEV_type1' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./energy/ensemble_EVV_type2' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/energy_avg/ensemble_EVV_type2' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/spha/ensemble_EEV_type2' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./energy/ensemble_penalty_EVV_evvtype1_evv1type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/energy_avg/ensemble_penalty_EVV_evvtype1_evv1type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/spha/ensemble_penalty_EVV_evvtype1_evv1type1' + '.png')
plt.close()
# plt.show()


for i in range(10):
    energy = np.loadtxt('./energy/ensemble_penalty_EVV_evvtype2_evv1type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/energy_avg/ensemble_penalty_EVV_evvtype2_evv1type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/spha/ensemble_penalty_EVV_evvtype2_evv1type1' + '.png')
plt.close()
# plt.show()

# divu
for i in range(10):
    energy = np.loadtxt('./divu/ensemble_EVV_type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/divu_avg/ensemble_EVV_type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| \nabla\cdot u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/divu/ensemble_EEV_type1' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./divu/ensemble_EVV_type2' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/divu_avg/ensemble_EVV_type2' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\| \nabla\cdot u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/divu/ensemble_EEV_type2' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./divu/ensemble_penalty_EVV_evvtype1_evv1type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/divu_avg/ensemble_penalty_EVV_evvtype1_evv1type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\|\nabla\cdot u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/divu/ensemble_penalty_EVV_evvtype1_evv1type1' + '.png')
plt.close()
# plt.show()


for i in range(10):
    energy = np.loadtxt('./divu/ensemble_penalty_EVV_evvtype2_evv1type1' + str(i) + '.txt')
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/divu_avg/ensemble_penalty_EVV_evvtype2_evv1type1' + '.txt')
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel(r'$ \|\|\nabla\cdot u\|\|^2$')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/divu/ensemble_penalty_EVV_evvtype2_evv1type1' + '.png')
plt.close()
# plt.show()

# enstropy
nu = 1./1000.
for i in range(10):
    energy = np.loadtxt('./enstropy/ensemble_EVV_type1' + str(i) + '.txt')
    energy = energy * nu
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/enstropy_avg/ensemble_EVV_type1' + '.txt')
energy = energy * nu
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel('Enstropy')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/enstropy/ensemble_EEV_type1' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./enstropy/ensemble_EVV_type2' + str(i) + '.txt')
    energy = energy * nu
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/enstropy_avg/ensemble_EVV_type2' + '.txt')
energy = energy * nu
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel('Enstropy')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/enstropy/ensemble_EEV_type2' + '.png')
plt.close()
# plt.show()

for i in range(10):
    energy = np.loadtxt('./enstropy/ensemble_penalty_EVV_evvtype1_evv1type1' + str(i) + '.txt')
    energy = energy * nu
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/enstropy_avg/ensemble_penalty_EVV_evvtype1_evv1type1' + '.txt')
energy = energy * nu
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel('Enstropy')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/enstropy/ensemble_penalty_EVV_evvtype1_evv1type1' + '.png')
plt.close()
# plt.show()


for i in range(10):
    energy = np.loadtxt('./enstropy/ensemble_penalty_EVV_evvtype2_evv1type1' + str(i) + '.txt')
    energy = energy * nu
    plt.plot(x, energy, alpha=alpha_list[i], linewidth=linewidth_list[-i-1], color=color_map(i+1), label=label_list[i])
energy = np.loadtxt('/Users/boweiouyang/Desktop/femml/cylinder1/output/enstropy_avg/ensemble_penalty_EVV_evvtype2_evv1type1' + '.txt')
energy = energy * nu
plt.plot(x, energy, alpha=alpha_list[-1], linewidth=linewidth_list[0], color=color_map(14), marker='.', markevery=40, label=r'$ \langle u \rangle$')
plt.xlabel("t")
plt.ylabel('Enstropy')
plt.legend(loc='lower right')
plt.xlim((0, 8))
plt.savefig('./plot/enstropy/ensemble_penalty_EVV_evvtype2_evv1type1' + '.png')
plt.close()
# plt.show()

