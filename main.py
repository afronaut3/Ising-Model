import matplotlib.pyplot as plt
import numpy as np
import Metropolis as mp
import Ising_class
import time
import itertools, numpy as np, matplotlib.pyplot as plt
import Stats
import csv
from scipy.optimize import root_scalar
import pylab
import renormalization
import pickle

def enumerate_state_energies(dim, beta):
    total_cells = dim * dim
    num_states = 2 ** total_cells
    energies = np.empty(num_states, dtype=np.int64)
    for state in range(num_states):
        state_str = np.binary_repr(state, width=total_cells)
        bits = np.array([int(bit) for bit in state_str])
        spins = 2 * bits - 1
        grid = spins.reshape(dim, dim)
        right_interaction = np.sum(grid[:, :-1] * grid[:, 1:])
        bottom_interaction = np.sum(grid[:-1, :] * grid[1:, :])
        E = -(right_interaction + bottom_interaction)
        energies[state] = E
    return np.exp(-beta * energies)

beta = 0.3
dim = 3
a = Ising_class.Ising(dim)
simulated_freqs = mp.MCMC_state_number(a, beta)

theoretical_res = enumerate_state_energies(dim, beta)
bins = np.arange(2 ** (dim * dim) + 1)

simulated_hist, _ = np.histogram(simulated_freqs, bins=bins)
simulated_prob = simulated_hist / simulated_hist.sum()

theoretical_prob = theoretical_res / theoretical_res.sum()

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10,6))
plt.bar(bin_centers, simulated_prob, width=0.8, alpha=0.6, label='Simulated')
plt.plot(bin_centers, theoretical_prob, 'ro-', label='Theoretical')
plt.xlabel('State Number')
plt.ylabel('Probability')
plt.legend()
plt.show()

##############
#TIME TEST
##############
# beta = 0.3
# dim = 9
# a = Ising_class.Ising(dim)
# start_time = time.time()
# simulated_freqs = mp.MCMC(a, beta)
# print(f"the time it takes is {time.time()-start_time}")


# ######################
# # 2) RUN MCMC ON A 27×27 ISING SYSTEM FOR VARIOUS BETA
# ######################
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # Make sure to import your custom modules, for example:
# # import Ising_class, mp, Stats

# betas = [99]
# dim = 27

# ## Pre-compute two extreme cases for theoretical comparison:

# # Case when beta = 0: use random configurations.
# my_ising = Ising_class.Ising(dim)
# Msq_zero_list = []
# E_zero_list = []
# for _ in range(10000):
#     Msq_zero_list.append((my_ising.get_m())**2)
#     E_zero_list.append(my_ising.get_energy())
#     my_ising.randomize()

# (mean_zero_E, var_zero_E, err_zero_E, tau_zero_E) = Stats.Stats(np.array(E_zero_list))
# (mean_zero_M2, var_zero_M2, err_zero_M2, tau_zero_M2) = Stats.Stats(np.array(Msq_zero_list))

# # Case when beta = infinity (approximated by beta = 99): system is frozen at the minimum energy.
# E_ex_list = -2*dim * dim * np.ones(10000)
# Msq_ex_list = (dim * dim)**2 * np.ones(10000)
# (mean_ex_E, var_ex_E, err_ex_E, tau_ex_E) = Stats.Stats(E_ex_list)
# (mean_ex_M2, var_ex_M2, err_ex_M2, tau_ex_M2) = Stats.Stats(Msq_ex_list)

# # Loop over the beta values for simulation.
# for beta in betas:
#     model = Ising_class.Ising(dim)
#     energy_list, mag_list, snapshot = mp.MCMC(model, beta)
    
#     energy_list = np.array(energy_list)
#     mag_list = np.array(mag_list)
#     msq_list = mag_list**2
    
#     (mean_E, var_E, err_E, tau_E) = Stats.Stats(energy_list)
#     (mean_M2, var_M2, err_M2, tau_M2) = Stats.Stats(msq_list)
#     # (No need to reassign mean_M2; Stats.Stats already provided it.)
    
#     # ----- Display the Snapshot with a Discrete Colormap -----
#     # Create a discrete colormap mapping -1 to blue and +1 to red.
#     cmap_discrete = ListedColormap(['blue', 'red'])
#     # Define boundaries so that values less than 0 get the first color and values above 0 get the second.
#     norm = BoundaryNorm([-1.5, 0, 1.5], cmap_discrete.N)
#     plt.matshow(snapshot, cmap=cmap_discrete, norm=norm)
#     plt.title(f"Prototypical Snapshot (beta={beta})")
#     plt.colorbar()
#     plt.show()
    
#     # ----- Energy Histogram -----
#     plt.figure(figsize=(7, 5))
#     # For beta = 0 or beta = 99, overlay theoretical histogram with common bins.
#     if beta == 0:
#         theoretical_energy = np.array(E_zero_list)
#         bins_energy = np.linspace(min(energy_list.min(), theoretical_energy.min()),
#                                   max(energy_list.max(), theoretical_energy.max()), 20)
#     elif beta == 99:
#         theoretical_energy = np.array(E_ex_list)
#         bins_energy = np.linspace(min(energy_list.min(), theoretical_energy.min()),
#                                   max(energy_list.max(), theoretical_energy.max()), 20)
#     else:
#         bins_energy = 'auto'
    
#     # Plot the simulated histogram.
#     plt.hist(energy_list, bins=bins_energy, density=True,
#              alpha=0.7, color='skyblue', edgecolor='black', label='Simulated Histogram')
#     plt.title(f"Histogram of Energy (beta={beta})")
#     plt.xlabel("Energy")
#     plt.ylabel("Probability")
#     plt.grid(True, alpha=0.3)
#     plt.axvline(mean_E, color='k', linestyle='--',
#                 label=f"<E>={mean_E:.3f}±{err_E:.3f}")
    
#     # If we used fixed bins, enforce the same x-axis limits.
#     if isinstance(bins_energy, np.ndarray):
#         plt.xlim(bins_energy[0], bins_energy[-1])
    
#     # Overlay the theoretical histogram when appropriate.
#     if beta == 0 or beta == 99:
#         plt.hist(theoretical_energy, bins=bins_energy, density=True,
#                  color='red', alpha=0.5, label='Theoretical Histogram', histtype='step')
    
#     plt.legend()
#     plt.show()
    
#     # Print Energy statistics.
#     print(f"=== Results for beta={beta} ===")
#     print(f"<E> = {mean_E:.3f} ± {err_E:.3f}")
#     if beta == 0:
#         print(f"Theoretical <E> = {mean_zero_E:.3f} ± {err_zero_E:.3f}")
#     elif beta == 99:
#         print(f"Theoretical <E> = {mean_ex_E:.3f} ± {err_ex_E:.3f}")
    
#     # ----- M^2 Histogram -----
#     plt.figure(figsize=(7, 5))
#     if beta == 0:
#         theoretical_msq = np.array(Msq_zero_list)
#         bins_msq = np.linspace(min(msq_list.min(), theoretical_msq.min()),
#                                max(msq_list.max(), theoretical_msq.max()), 20)
#     elif beta == 99:
#         theoretical_msq = np.array(Msq_ex_list)
#         bins_msq = np.linspace(min(msq_list.min(), theoretical_msq.min()),
#                                max(msq_list.max(), theoretical_msq.max()), 20)
#     else:
#         bins_msq = 'auto'
    
#     # Plot the simulated M^2 histogram.
#     plt.hist(msq_list, bins=bins_msq, density=True,
#              alpha=0.7, color='lightgreen', edgecolor='black', label='Simulated Histogram')
#     plt.title(f"Histogram of $M^2$ (beta={beta})")
#     plt.xlabel("$M^2$")
#     plt.ylabel("Probability")
#     plt.grid(True, alpha=0.3)
#     plt.axvline(mean_M2, color='k', linestyle='--',
#                 label=f"<$M^2$>={mean_M2:.3f}±{err_M2:.3f}")
    
#     if isinstance(bins_msq, np.ndarray):
#         plt.xlim(bins_msq[0], bins_msq[-1])
    
#     # Overlay the theoretical M^2 histogram when appropriate.
#     if beta == 0 or beta == 99:
#         plt.hist(theoretical_msq, bins=bins_msq, density=True,
#                  color='red', alpha=0.5, label='Theoretical Histogram', histtype='step')
    
#     plt.legend()
#     plt.show()
    
#     # Print M^2 statistics.
#     if beta == 0:
#         print(f"Theoretical <M^2> = {mean_zero_M2:.3f} ± {err_zero_M2:.3f}")
#     elif beta == 99:
#         print(f"Theoretical <M^2> = {mean_ex_M2:.3f} ± {err_ex_M2:.3f}")
#     print(f"<M^2> = {mean_M2:.3f} ± {err_M2:.3f}, var($M^2$)={var_M2:.3f}, tau_M^2={tau_M2:.2f}\n")
# dimension = 27
# model = Ising_class.Ising(dimension)
# betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9, 1.0]  # Example values
# MSQ_list = []
# E_list = []
# err_list = []
# for beta in betas:
#     Energy_list,M_list,_ = mp.MCMC(model,beta)
#     M_list = np.array(M_list)
#     MSQ_list.append(np.mean(M_list**2))
#     E_list.append(Energy_list)
#     (mean_M2, var_M2, err_M2, tau_M2)   = Stats.Stats(M_list**2)
#     err_list.append(err_M2 if err_M2 >= 1e-6 else 1e-6)




# plt.plot(betas, MSQ_list)
# plt.errorbar(betas, MSQ_list, yerr=err_list, capsize=5)
# plt.title(r"$M^2$ vs $\beta$")
# plt.xlabel(r"$\beta$")
# plt.ylabel(r"$M^2$")
# plt.show()


# E_avg_list = np.mean(E_list, axis=1)

# betas = np.array(betas)
# dE_dBeta = np.gradient(E_avg_list, betas) 

# Cv = - (betas ** 2) * dE_dBeta 

# plt.plot(1 / betas, Cv)
# plt.title("Specific Heat vs Temperature")
# plt.xlabel("Temperature")
# plt.ylabel("Specific Heat")
# plt.show()


# with open(f'data{dimension}.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for beta, msq, e in zip(betas, MSQ_list, E_list):
#         writer.writerow([np.mean(msq)])








import scipy




dimension = 81
# my_ising = Ising_class.Ising(dimension)

# betas,R = renormalization.R(my_ising)
# with open(f'beta_coarsed{dimension}.csv', mode='w', newline="") as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerow(betas)

# with open('R.pkl', 'wb') as f:
#     pickle.dump(R, f)
beta_raw = np.arange(0,1.1,0.1)
with open('beta_coarsed81.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data = [float(x) for x in row]
        break    


plt.plot(np.arange(0,1.1,0.1), data, label=r"$R(\beta)$")
plt.plot(np.arange(0,1.1,0.01),np.arange(0,1.1,0.01),label = "x vs y")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$R(\beta)$")
plt.title(r"Plot of $R(\beta)$ vs. $\beta$")
plt.legend()
plt.grid(True)
plt.show()

R = scipy.interpolate.interp1d(beta_raw,data,fill_value="extrapolate")
testing_range = np.arange(0,1,0.001)
first_negative_index = None
subtracted = R(testing_range)-testing_range
for i, val in enumerate(subtracted[300:]):
    if val > 0:
        first_negative_index = i+300
        break


print(f"The transition beta is at = {first_negative_index/1000}")



plt.plot(np.arange(0,1.1,0.1), data, label=r"$R(\beta)$")
plt.plot(np.arange(0,1.1,0.01),np.arange(0,1.1,0.01),label = "x vs y")
plt.arrow(0.35, R(0.35),
          R(0.35) - 0.35, 0, 
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(0.35), R(0.35),
          0, R(R(0.35)) - R(0.35),
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(0.35), R(R(0.35)),
          R(R(0.35)) - R(0.35), 0,
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(R(0.35)), R(R(0.35)),
          0, R(R(R(0.35))) - R(R(0.35)),
          length_includes_head=True, head_width=0.01, head_length=0.02)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$R(\beta)$")
plt.title(r"Plot of $R(\beta)$ vs. $\beta$")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(np.arange(0,1.1,0.1), data, label=r"$R(\beta)$")
plt.plot(np.arange(0,1.1,0.01),np.arange(0,1.1,0.01),label = "x vs y")
plt.arrow(0.5, R(0.5),
          R(0.5) - 0.5, 0, 
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(0.5), R(0.5),
          0, R(R(0.5)) - R(0.5),
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(0.5), R(R(0.5)),
          R(R(0.5)) - R(0.5), 0,
          length_includes_head=True, head_width=0.01, head_length=0.02)

plt.arrow(R(R(0.5)), R(R(0.5)),
          0, R(R(R(0.5))) - R(R(0.5)),
          length_includes_head=True, head_width=0.01, head_length=0.02)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$R(\beta)$")
plt.title(r"Plot of $R(\beta)$ vs. $\beta$")
plt.legend()
plt.grid(True)
plt.show()

unstable_eq = first_negative_index/1000

range_x = np.arange(unstable_eq-0.05,unstable_eq+0.05,0.001)
range_y = R(range_x)
derivative = np.gradient(range_y,range_x)[len(range_x)//2]
critical_exp = 1/(np.emath.logn(3, derivative))
print(f"The critical exponent is {critical_exp}")







# dimension = 81
# betas = np.array([0.0,0.3,0.4,0.5,0.6,50])
# my_ising = Ising_class.Ising(dimension)
# for beta in betas:
#     _,_,proto = mp.MCMC(my_ising,beta)
#     plt.matshow(proto, cmap='bwr')
#     plt.title(f"Prototypical Snapshot (beta={beta})")
#     plt.show()
#     smaller_ising = renormalization.coarse_grain(my_ising,dimension//3)
#     plt.matshow(smaller_ising.grid, cmap='bwr')
#     plt.title(f"Prototypical Snapshot coarse-grained once (beta={beta})")
#     plt.show()
#     smallest_ising = renormalization.coarse_grain(smaller_ising,dimension//9)
#     plt.matshow(smallest_ising.grid, cmap='bwr')
#     plt.title(f"Prototypical Snapshot coarse-grained twice (beta={beta})")
#     plt.show()







# data_coarsed =np.array([754.3396131876941,
# 897.9684337107926,
# 1189.0356749173263,
# 2023.7982763803989,
# 8754.160837759295,
# 415096.03878144105,
# 530316.8665196913,
# 531347.9229381701,
# 531425.5403347028,
# 531438.9573103518,
# 531440.4163743862])
# betas = np.arange(0,1.1,0.1)
# plt.plot(betas,data_coarsed)
# plt.title(r"Coarsed $M^2$ vs $\beta$")
# plt.xlabel(r"$\beta$")
# plt.ylabel(r"$M^2$")
# plt.show()