import Ising_class
import numpy as np
import math
import matplotlib.pyplot as plt
import Stats
import itertools, numpy as np, matplotlib.pyplot as plt
import renormalization
def MCMC_state_number(my_ising,beta):
    dim = Ising_class.Ising.get_dimension(my_ising)
    N = dim*dim
    ret = []
    for sweep in range(10000):
        if(sweep < 21):
            continue
        for i in range(N):
            random_loc = np.random.randint(0,dim,2)
            delta_e = Ising_class.Ising.flip_bit(my_ising,random_loc)
            random_prob = np.random.rand()
            prob_to_change = np.exp(-1*beta*delta_e)
            if(prob_to_change <= random_prob):
                Ising_class.Ising.flip_bit(my_ising,random_loc)
        ret.append(Ising_class.Ising.get_state_number(my_ising))
    
    return np.array(ret)

def MCMC(my_ising, beta):
    dim = my_ising.get_dimension()  # Call the method on the instance
    N = dim * dim
    ret_E = []
    ret_M = []
    average_grid = np.zeros((dim, dim))

    for sweep in range(10000):
        if sweep < 21:
            continue
        for i in range(N):
            random_loc = np.random.randint(0, dim, 2)
            delta_e = my_ising.flip_bit(random_loc)
            random_prob = np.random.rand()
            prob_to_change = np.exp(-beta * delta_e)
            if prob_to_change <= random_prob:
                my_ising.flip_bit(random_loc)
        average_grid += my_ising.get_grid()
        ret_E.append(my_ising.get_energy())
        ret_M.append(my_ising.get_m())  
    average_grid = average_grid / 10000
    mask = np.where(average_grid > 0, 1, -1)
    return np.array(ret_E), np.array(ret_M), mask


def MCMC_coarse_grain(my_ising, beta):
    dim = my_ising.get_dimension()  # Call the method on the instance
    N = dim * dim
    ret_E = []
    ret_M = []
    average_grid = np.zeros((dim, dim))

    for sweep in range(10000):
        if sweep < 21:
            continue
        for i in range(N):
            random_loc = np.random.randint(0, dim, 2)
            delta_e = my_ising.flip_bit(random_loc)
            random_prob = np.random.rand()
            prob_to_change = np.exp(-beta * delta_e)
            if prob_to_change <= random_prob:
                my_ising.flip_bit(random_loc)
        new_ising = renormalization.coarse_grain(my_ising,dim//3)
        ret_E.append(new_ising.get_energy())
        ret_M.append(new_ising.get_m())  
    
    return np.array(ret_E), np.array(ret_M)

