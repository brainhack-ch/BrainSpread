''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import os

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian

from utils_vis import visualize_diffusion_timeplot

class DiffusionSimulation:
    def __init__(self, connect_matrix):
        self.beta = 1.5 # As in the Raj et al. papers

        #TODO: change init concentration with pet data
        self.init_concentration = 1 # initial concentration of misfolded proteins in the seeds

        self.iterations = int(1e3) #1000
        self.rois = 116 # AAL atlas has 116 rois
        self.tstar = 10.0 # total length of the simulation
        self.timestep = self.tstar / self.iterations
        self.cm = connect_matrix
        
    def run(self, inverse_log=True, downsample=True):
        ''' Run simulation. '''
        if inverse_log: self.calc_exponent()
        self.define_seeds()
        self.calc_laplacian()
        self.diffusion_final = self.iterate_spreading()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        
    def define_seeds(self):
        ''' Define Alzheimer seed regions manually. '''
        # Store initial misfolded proteins
        self.diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        # assign initial concentration of proteins in this region
        self.diffusion_init[[31, 32, 35, 36]] = self.init_concentration
        
    def calc_laplacian(self):
        # normed laplacian 
        adjacency = self.cm
        laplacian = scipy_laplacian(adjacency, normed=True)
        self.eigvals, self.eigvecs = np.linalg.eig(laplacian)
    
    def integration_step(self, x0, t):
        xt = self.eigvecs.T @ x0
        xt = np.diag(np.exp(-self.beta * t * self.eigvals)) @ xt
        return self.eigvecs @ xt  
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for _ in tqdm(range(self.iterations)):
            next_step = self.integration_step(diffusion[-1], self.timestep)
            diffusion.append(next_step)  
            
        return np.asarray(diffusion)   
 
    def calc_exponent(self):
        ''' Inverse operation to log1p. '''
        self.cm = np.expm1(self.cm)
 
    def downsample_matrix(self, matrix, target_len=int(1e3)):
        ''' Take every n-th sample when the matrix is longer than target length. '''
        current_len = matrix.shape[0]
        if current_len > target_len:
            factor = int(current_len/target_len)
            matrix = matrix[::factor, :] # downsampling
        return matrix
    
    def save_matrix(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'diffusion_matrix_over_time.csv'), 
                                self.diffusion_final, delimiter=",")
 
def load_connectivity_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def run_simulation(subject_path):    
    ''' Run simulation for single patient. '''
    
    connectivity_matrix_path = os.path.join(subject_path, 'connect_matrix_rough.csv')
    
    connect_matrix = load_connectivity_matrix(connectivity_matrix_path)
    simulation = DiffusionSimulation(connect_matrix)
    simulation.run()
    simulation.save_matrix(subject_path)
    visualize_diffusion_timeplot(simulation.diffusion_final, save_dir=subject_path)

def main():
    connectomes_dir = '../data/output'
    for subject in os.listdir(connectomes_dir):
        print(f'Simulation for subject: {subject}')
        subject_path = os.path.join(connectomes_dir, subject)
        run_simulation(subject_path)
    
if __name__ == '__main__':
    main()