''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import os
import logging

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian

from utils_vis import visualize_diffusion_timeplot

logging.basicConfig(level=logging.INFO)

class DiffusionSimulation:
    def __init__(self, connect_matrix, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        
        self.beta = 1.5 # As in the Raj et al. papers
        self.iterations = int(1e3) #1000
        self.rois = 116 # AAL atlas has 116 rois
        self.tstar = 2.0 # total length of the simulation
        self.timestep = self.tstar / self.iterations
        self.cm = connect_matrix
        if concentrations is not None: 
            logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
        
    def run(self, inverse_log=True, downsample=True):
        ''' Run simulation. '''
        if inverse_log: self.calc_exponent()
        self.calc_laplacian()
        self.diffusion_final = self.iterate_spreading()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        
    def define_seeds(self, init_concentration=1):
        ''' Define Alzheimer seed regions manually. 
        
        Args:
            init_concentration (int): initial concentration of misfolded proteins in the seeds. '''
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        # assign initial concentration of proteins in this region
        diffusion_init[[31, 32, 35, 36]] = init_concentration
        return diffusion_init
        
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
    
    def save_diffusion_matrix(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'diffusion_matrix_over_time.csv'), 
                                self.diffusion_final, delimiter=",")
    
    def save_terminal_concentration(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'terminal_concentration.csv'),
                   self.diffusion_final[-1, :], delimiter=',')
        
 
def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def run_simulation(connectomes_dir, node_intensity_dir, subject):    
    ''' Run simulation for single patient. '''
    
    subject_path = os.path.join(connectomes_dir, subject)      
    
    connectivity_matrix_path = os.path.join(subject_path, 'connect_matrix_rough.csv')
    t0_concentration_path = os.path.join(node_intensity_dir, f'nodeIntensities_{subject}t0.csv')
    t1_concentration_path = os.path.join(node_intensity_dir, f'nodeIntensities_{subject}t1.csv')
    
    # load connectome
    connect_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brian regions
    t0_concentration = load_matrix(t0_concentration_path) 
    t1_concentration = load_matrix(t1_concentration_path)
            
    simulation = DiffusionSimulation(connect_matrix, t0_concentration)
    simulation.run()
    simulation.save_diffusion_matrix(subject_path)
    simulation.save_terminal_concentration(subject_path)
    visualize_diffusion_timeplot(simulation.diffusion_final, save_dir=subject_path)

def main():
    connectomes_dir = '../data/output'
    node_intensity_dir = '../data/node_intensity_data'
    
    patients = ['sub-AD4215', 'sub-AD4500', 'sub-AD6264']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(connectomes_dir, node_intensity_dir, subject)
    
if __name__ == '__main__':
    main()