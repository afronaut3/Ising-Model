import matplotlib.pyplot as plt
import numpy as np
import Metropolis as mp
import Ising_class
import time
import itertools, numpy as np, matplotlib.pyplot as plt
import Stats
import scipy
import csv
def coarse_grain(my_ising,dim):
    original_dimension =my_ising.get_dimension()
    if(dim>original_dimension or original_dimension % dim != 0):
        raise Exception("Invalid Target Dimension")
    ratio = original_dimension//dim
    new_grid = my_ising.get_grid()
    new_ising = Ising_class.Ising(dim)
    new_grid = new_grid.reshape(dim,dim,ratio,ratio).sum(axis = (2,3))
    new_grid = np.where(new_grid > 0,1,-1)
    new_ising.set_state(new_grid)
    return new_ising

def R(my_ising):
    dim = my_ising.get_dimension()
    coarse_dim = dim//3
    data = np.loadtxt(f'data{coarse_dim}.csv', delimiter=',')
    betas = np.arange(0,1.1,0.1)
    model = scipy.interpolate.interp1d(data,betas,fill_value="extrapolate")
    betas_CG = []
    MSQS_CG = []
    for beta in betas:
        _,Ms = mp.MCMC_coarse_grain(my_ising,beta)
        MSQ_CG = np.mean(Ms**2)
        MSQS_CG.append(MSQ_CG)
        betas_CG.append(model(MSQ_CG).item())
    with open(f'data_coarsed{dim}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for msq in MSQS_CG:
            writer.writerow([msq])    
    
    print(betas_CG)
    R = scipy.interpolate.interp1d(betas,betas_CG,fill_value="extrapolate")
    return betas_CG,R

     

        
        


