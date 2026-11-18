#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
========================================================================
IMPORT PACKAGES:
========================================================================
'''
import glob
import os
import tqdm
from os.path import join as opj
import pandas as pd
import numpy as np
import copy
import sys
import itertools
import ast
from groo.groo import get_root
import nibabel as nib
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import rsatoolbox
import rsatoolbox.rdm as rsr
import matplotlib.pyplot as plt
#import scipy
root_folder = get_root(".my_hidden_root_file")
print(root_folder)

def create_script_named_folder(folder_name=None):
    """
    Creates a new directory with the name of the script file (without extension) or with the provided folder name.
    
    Args:
        folder_name (str, optional): The name of the folder to be created inside the script directory. 
            If not provided, the folder will have the same name as the script file (without extension).
    
    Returns:
        str: The path of the newly created directory.
    """
    
    # Assuming __file__ represents the current file name in a global scope
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    script_name_without_ext = os.path.splitext(script_name)[0]
    
    if folder_name is None:
        new_dir_path = os.path.join(script_dir, script_name_without_ext)
    else:
        new_dir_path = os.path.join(script_dir, script_name_without_ext, folder_name)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    return new_dir_path



'''
========================================================================
Subj and session info
========================================================================
'''


try:
    sub = 'sub-%s' % sys.argv[1]
    if len(sub) > 10: raise ValueError("Invalid sub ID")

except:
    print('ACHTUNG - DEVELOPER SUBJ CHOSEN')
    sub = 'sub-5'

sub_num = sub[4:]
ses = 'ses-01'

try:
    mask_name = sys.argv[2]
    print('This should be the mask name:', mask_name)
except:
    mask_name = 'mtl'
    print('cave this is run locally, using mtl as mask name')

# Get paths from main task
try:
    path_reward_learning = opj(root_folder, f'Behavior/Sess2/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_2.csv') # localizer_complete_participant_{sub_num}_session_{behavior_ses[-1]}_w_TR.csv')
    df_path_reward_learning = pd.read_csv(path_reward_learning, sep=';')
except:
    path_reward_learning = opj(root_folder, f'Behavior/Sess2/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_2_post_exp_completed.csv') # localizer_complete_participant_{sub_num}_session_{behavior_ses[-1]}_w_TR.csv')
    df_path_reward_learning = pd.read_csv(path_reward_learning, sep=';')

# drop all nas in base paths columns
df_path_reward_learning = df_path_reward_learning.dropna(subset=['base_path_A_num1', 'base_path_A_star_num3', 'base_path_B_num2', 'base_path_C_num4']).reset_index(drop=True)

# Get the base paths
base_path_A_num1 = ast.literal_eval(df_path_reward_learning['base_path_A_num1'][0])
base_path_A_star_num3 = ast.literal_eval(df_path_reward_learning['base_path_A_star_num3'][0])
base_path_B_num2 = ast.literal_eval(df_path_reward_learning['base_path_B_num2'][0])
base_path_C_num4 = ast.literal_eval(df_path_reward_learning['base_path_C_num4'][0])

conditions = list(itertools.chain.from_iterable([base_path_A_num1, base_path_A_star_num3, base_path_B_num2, base_path_C_num4]))
'''
========================================================================
# 3. Set up dataset object for the RDM 
# here we will loop over all main runs and create the RDM per run
========================================================================
'''

# This version loops over the runs and always creates the RDM in pairs of 2 runs. run with run +1


for run in [1, 4]:
    
    runs = [run, run+1]

    n_rep = 2 # number of repetitions of each condition, i.e. runs

    conds = np.array(['vid_' + x for x in conditions]) # conditions
    conds = np.repeat(conds, n_rep) # repeat each condition n_rep times

    sessions = np.tile(np.arange(n_rep), len(conditions)) # sessions
    sessions = np.array(['run_' + str(x+1) for x in sessions])

    des = {'subj': sub_num} # design matrix

    measurements = []
    for condition in conditions:
        for run in runs:
            print(f'loading tmap for {condition} in run {run}')
        
            # load tmap
            tmap_dir = opj(root_folder, 'Code', '7_RSA', 'Rewardlearning_reliability', '1_RSA_Session2_tvalues_RSAtoolbox_crossnobis', 't_maps_from_first_level_glm_masked')
            tmap_file_names = f'sub-{sub_num}_{ses}_{mask_name}_{condition}_run_{run}.npy'
            loaded_tmap = np.load(opj(tmap_dir, tmap_file_names))

            # append the masked beta map to the measurements array, giving us a 2D array of size n_observations (runs) x n_voxels (n_channels)
            measurements.append(loaded_tmap)


    n_voxels = loaded_tmap.shape[0]

    measurements_array = np.array(measurements)

    # design matrix
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(n_voxels)])} # channels: i.e. voxels in ROI

    obs_des = {'conds': conds, 'sessions': sessions} # observations: i.e. conditions and sessions

    #measurements is a 2D array of size n_observations x n_channels, e.g. [[voxel, voxel],[voxel, voxel]]
    print('dimensions of measurements should be n_observations (sessions*conditions) x n_channels (voxels): ')
    print(np.shape(measurements_array))

    # create a data matrix
    data = rsd.Dataset(measurements=measurements_array,
                            descriptors=des,
                            obs_descriptors=obs_des,
                            channel_descriptors=chn_des)

    '''
    ========================================================================
    # 4. get noise matrix based on residuals
    ========================================================================
    '''
    noise_matrices_dir = opj(root_folder, 'Code', '7_RSA', 'Rewardlearning_reliability', '1_RSA_Session2_tvalues_RSAtoolbox_crossnobis', 'residuals_from_first_level_glm')

    residuals = [] # create dict of 8 residuals, one for each run

    ind_noise = 1
    # let's see if the residuals are not singular, so we can actually use them!
    noise_singular = False
    for run in runs:

        noise_matrices_filename = f'mask_name_{mask_name}_sub_{sub}_run_{run}.npy'
        noise_matrices = np.load(opj(noise_matrices_dir, noise_matrices_filename))

        residuals.append(noise_matrices)
        ind_noise += 1
        print('============================================================')
        print('dimensions of noise_pres_res (noise matrix): ', np.shape(noise_matrices))
        print('dataset.n_channel: ', data.n_channel)
        print('============================================================')
        # check if the noise is all just nans
        if np.isnan(noise_matrices).all():
            noise_singular = True

    '''
    ========================================================================
    # 5. Add the residuals to the dataset and calculate neural RDM
    ========================================================================
    '''
    rdm_cv = rsatoolbox.rdm.calc_rdm(data, method='crossnobis', descriptor='conds', cv_descriptor='sessions', noise=residuals) #residuals
    

    path_rdms_from_rsatoolbox_with_residual_crossnobis = create_script_named_folder('rdms_from_rsatoolbox_with_residual_crossnobis')

    save_file = opj(path_rdms_from_rsatoolbox_with_residual_crossnobis, f'sub-{sub_num}_mask_{mask_name}_rdm_cv_run_{run}_and_run_{run+1}.pkl')

    # save the RDMs
    rdm_cv.save(save_file, file_type='pkl', overwrite=True)
    
    # save data for janis
    janis_dir = create_script_named_folder(f'janis_{mask_name}')
    # save rdm_cv.get_matrices()[0,:,:] as df after adding conditions as columns and rows
    rdm_cv_df = pd.DataFrame(rdm_cv.get_matrices()[0,:,:])
    rdm_cv_df.columns = conditions
    rdm_cv_df.index = conditions
    rdm_cv_df.to_csv(opj(janis_dir, f'sub-{sub_num}_mask_{mask_name}_rdm_cv_run_{run}_and_run_{run+1}.csv'), sep=';')

    print('============================================================')
    print(f'saved RDM  for subject {sub} in region {mask_name} to: ', save_file)
    print('============================================================')


    '''
    ========================================================================
    # 6. plot the RDM
    ========================================================================
    '''
    toying_around = 0
    # toying around
    if toying_around:
        fig, ax, ret_val = rsatoolbox.vis.show_rdm(rdm_cv)

        plt_rdm = rdm_cv.get_matrices()[0,:,:]

        # Plot the RDM
        fig, ax = plt.subplots()
        im = ax.imshow(plt_rdm, cmap='coolwarm')
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_xticklabels(conditions)
        ax.set_yticklabels(conditions)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.colorbar(im, ax=ax)
        plt.title(f"Neural RDM for subject {sub} in region {mask_name}")
        plt.show()

