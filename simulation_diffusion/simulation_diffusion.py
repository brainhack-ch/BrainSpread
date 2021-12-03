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
    
    def save_matrix(self, path):
        np.savetxt(path, self.diffusion_final, delimiter=",")
 
def load_connectivity_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def runPath(data_path_p, output_path_p):
    data_path = '../output/sub-AD4009/connect_matrix_rough.csv'

    output_path = '../output/sub-AD4009/diffusion_matrix.csv'
    
    connect_matrix = load_connectivity_matrix(data_path_p)
    simulation = DiffusionSimulation(connect_matrix)
    simulation.run()
    simulation.save_matrix(output_path_p)
    visualize_diffusion_timeplot(simulation.diffusion_final)
    return simulation.diffusion_final

def main():
    patients = ['sub-AD4009', 'sub-AD4215', 'sub-AD4500', 'sub-AD4892', 'sub-AD6264']
    i = 0
    diffusionMatrixes = np.zeros(5,1001,116).reshape((5,1001, 116))
    for p in patients:
        diffusionMatrixes[i] = runPath('../output/'+ p +'/connect_matrix_rough.csv', '../output/'+ p+'/diffusion_matrix.csv')
        i += 1
    
    
if __name__ == '__main__':
    main()