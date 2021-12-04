''' Utilities funciton for visualizaiton purposes. '''

import os

import matplotlib.pyplot as plt
import numpy as np

def visualize_NIfTI_data(data, depths, time_idx=None):
    ''' Visualize provided depths from NIfTI data at the specific time. 
    
    Args:
        data (nibabel.nifti1.Nifti1Image): 3D/4D data [width, height, depth, [time]]
        depths (list; len(list)>1): list of depths indexes to visualize
        time_idx (int): index of time; if is None: do not consider time (data is 3D) 
    '''
    data = data.get_fdata()
    
    squeeze = False if len(depths)==1 else True
    fig, axs = plt.subplots(1, len(depths), squeeze=squeeze)
    for i, ax in enumerate(axs):
        ax.imshow(data[:, :, depths[i], time_idx])
        ax.set_title(f'depth number: {depths[i]}')
    plt.suptitle(f'time index: {time_idx}')
    plt.tight_layout()
    plt.show()
    
def visualize_diffusion_timeplot(matrix, timestep, total_time, save_dir=None):
    plt.figure(figsize=(15,3))
    plt.imshow(matrix.T) #, interpolation='nearest'
    plt.xlabel('# iterations')
    # TODO: change xticks and labels to time in years
    # plt.xticks(np.arange(0, total_time, step=timestep), labels=np.arange(0, total_time, step=timestep))
    # plt.xlabel('Time [years]' )
    plt.ylabel('ROIs')
    plt.colorbar()
    plt.title(f'Total time of simulation: {total_time} years')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'diffusion_over_time.png'))
    plt.show() 