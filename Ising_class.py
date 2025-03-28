import numpy as np
import numba
from numba import int64
from numba.experimental import jitclass

class Ising:
    def __init__(self, dimension):
        self.grid = np.zeros((dimension, dimension), dtype=np.int64)
        self.energy = 0
        self.randomize()

    def recalculate_energy(self):
        horizontal_sum = np.sum(self.grid[1:, :] * self.grid[:-1, :])
        vertical_sum = np.sum(self.grid[:, 1:] * self.grid[:, :-1])
        return -1 * (horizontal_sum + vertical_sum)
    
    def get_grid(self):
        return self.grid
    
    def get_m(self):
        currstate = self.get_state_vector()
        return np.sum(currstate)
    
    def get_state_vector(self):
        dim = self.get_dimension()
        return (((self.grid))).reshape(dim*dim)
    

    def get_state_number(self):
        dim = self.get_dimension()
        #8 byte can store up to 2^64
        if(dim >=6):
            raise Exception("TOO LARGE DIMENSION")
        return np.sum(2**np.arange(dim*dim) * (self.get_state_vector()+1)//2)



    def get_dimension(self):
        return self.grid.shape[0]

    def get_energy(self):
        return self.energy

    def randomize(self):
        rand_arr = np.random.random((self.get_dimension(), self.get_dimension()))
        self.grid = np.where(rand_arr < 0.5, 1, -1)
        self.energy = self.recalculate_energy()

    def set_state(self,new_grid):
        self.grid = new_grid
        self.energy = self.recalculate_energy()


    def flip_bit(self, position):
        ## position means row,col

        my_spin = self.grid[position[0], position[1]]
        dim = self.grid.shape[0]
        self.grid[position[0], position[1]] *= -1
        delta_e = 0
        if position[0] != 0:
            delta_e += 2 * my_spin * self.grid[position[0]-1, position[1]]
        if position[1] != 0:
            delta_e += 2 * my_spin * self.grid[position[0], position[1]-1]
        if position[1] != dim - 1:
            delta_e += 2 * my_spin * self.grid[position[0], position[1]+1]
        if position[0] != dim - 1:
            delta_e += 2 * my_spin * self.grid[position[0]+1, position[1]]
        self.energy += delta_e
        return delta_e