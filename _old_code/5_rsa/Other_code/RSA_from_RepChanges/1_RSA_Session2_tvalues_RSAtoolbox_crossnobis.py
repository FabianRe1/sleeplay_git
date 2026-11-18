#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:53:36 2024

@author: renzadmin
"""
'''
========================================================================
IMPORT PACKAGES:
========================================================================
'''
#%%
import glob
import os
import tqdm
from os.path import join as opj
import pandas as pd
import copy
import sys
#import scipy
import numpy as np
#import datalad.api as dl
#from scipy.stats import sem, spearmanr, pearsonr
from nilearn import image, masking, signal, plotting
#from nilearn.decoding import Decoder
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel
#from sklearn import metrics
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from nilearn.masking import apply_mask, unmask
from nilearn.maskers import NiftiMasker
import seaborn as sns
import itertools
import ast
from groo.groo import get_root
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import gc
root_folder = get_root(".my_hidden_root_file")
print(root_folder)
# root_folder = opj(root_folder, 'Leipzig')
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
SETUP NECESSARY PATHS ETC:
========================================================================
'''

path_fmriprep = opj(root_folder, '3_derivatives')
path_masks = opj(root_folder, '4_decoding','repchanges-rewardlearning-masks-sigloss', 'output_rewardlearning')

TR = 1.5
use_reliability_mask = 1

try:
    sub = 'sub-%s' % sys.argv[1]
    if len(sub) > 10: raise ValueError("Invalid sub ID")

except:
    print('ACHTUNG - DEVELOPER SUBJ CHOSEN')
    sub = 'sub-5'


#%%
use_thresholded_masks=1
mask_threshold=50
sub_num = sub[4:]
behavior_ses = 'Sess2' # first main task session
ses = 'ses-01'

'''
========================================================================
LOAD BEHAVIORAL DATA FOR THE LOCALISER/TRAINING SCAN:
========================================================================
'''
sub_num = sub[4:]
path_rewardlearning_scan_pst_exp_completed = opj(root_folder, f'Behavior/{behavior_ses}/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_{behavior_ses[-1]}_post_exp_completed_longform_w_TR.csv')
path_rewardlearning_scan = opj(root_folder, f'Behavior/{behavior_ses}/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_{behavior_ses[-1]}_longform_w_TR.csv')
fixation_path_rewardlearning_scan_pst_exp_completed = opj(root_folder, f'Behavior/{behavior_ses}/subj_{sub_num}/fixation_maindata_complete_participant_{sub_num}_session_{behavior_ses[-1]}_post_exp_completed_longform.csv')
fixation_path_rewardlearning_scan = opj(root_folder, f'Behavior/{behavior_ses}/subj_{sub_num}/fixation_maindata_complete_participant_{sub_num}_session_{behavior_ses[-1]}_longform.csv')

if os.path.exists(path_rewardlearning_scan_pst_exp_completed):
    file_to_load = path_rewardlearning_scan_pst_exp_completed
elif os.path.exists(path_rewardlearning_scan):
    file_to_load = path_rewardlearning_scan
else:
    file_to_load = None

# If a file is selected, load and store it
if file_to_load is not None:
    df_behavior = pd.read_csv(file_to_load, sep=';', index_col=False)

# Load the fixation file
if os.path.exists(fixation_path_rewardlearning_scan_pst_exp_completed):
    fixation_file_to_load = fixation_path_rewardlearning_scan_pst_exp_completed
elif os.path.exists(fixation_path_rewardlearning_scan):
    fixation_file_to_load = fixation_path_rewardlearning_scan
else:
    fixation_file_to_load = None

# If a fixation file is selected, load and store it
if fixation_file_to_load is not None:
    df_fixation = pd.read_csv(fixation_file_to_load, sep=';', index_col=False)

#%% Generate a df with the onsets and stimuli for the first level GLM
# =============================================================================            

# create df_behavior_TR only keeping columns with TR in the namñe
df_behavior_TR = copy.deepcopy(df_behavior.filter(regex='onset', axis=1))
df_behavior_TR = copy.deepcopy(df_behavior_TR.filter(regex='block', axis=1))
# Add 'stimulus' column to df_behavior_TR
df_behavior_TR['stimulus'] = df_behavior['stimulus']
# Add 'Block_MRI' column to df_behavior_TR
df_behavior_TR['Block_MRI'] = df_behavior['Block_MRI']
# add reaction time
df_behavior_TR['slider_reaction_time'] = df_behavior['slider_reaction_time']

df_behavior_TR['new_run_info_onset'] = df_fixation['pre_stim_1_ITI_clock_block']
df_behavior_TR['new_run_info_duration'] = df_behavior_TR['onset_time_block'] - df_behavior_TR['new_run_info_onset']#.shift(1)

# Create a new empty DataFrame with the same columns as df_behavior_TR
df_behavior_TR_new = pd.DataFrame(columns=df_behavior_TR.columns)
# Iterate over the rows in df_behavior_TR
for idx, row in df_behavior_TR.iterrows():
    # Concatenate the current row
    df_behavior_TR_new = pd.concat([df_behavior_TR_new, row.to_frame().T], ignore_index=True)
    if pd.notna(row['slider_onset_block']): 
        # initalise 1 nan row df with columns as df_behavior_TR_new
        nan_row = pd.DataFrame([np.nan] * len(df_behavior_TR.columns)).T
        # match column names
        nan_row.columns = df_behavior_TR.columns
        df_behavior_TR_new = pd.concat([df_behavior_TR_new, nan_row], ignore_index=True)

# Reset the index of the new DataFrame
df_behavior_TR_new.reset_index(drop=True, inplace=True)
# Replace the original DataFrame with the new one
df_behavior_TR = df_behavior_TR_new
# go over every row in df_behavior_TR.TR_slider_onset_block_w_HRF and if it is not NaN, insert the value in the column TR_onset_time_block_w_HRF at index row+1 which should ocntian a nan
for index, row in df_behavior_TR.iterrows():
    slider_TR = row['slider_onset_block']
    if not pd.isna(slider_TR):
        df_behavior_TR.loc[index + 1, 'onset_time_block'] = slider_TR
        df_behavior_TR.loc[index + 1, 'stimulus'] = 'slider'

    block = row['Block_MRI']
    if pd.isna(block):
        df_behavior_TR.loc[index, 'Block_MRI'] = df_behavior_TR.loc[index-1, 'Block_MRI']
# shift every value of slider_reaction_time one row down
df_behavior_TR['slider_reaction_time'] = df_behavior_TR['slider_reaction_time'].shift(1)
df_behavior_TR['slider_reaction_time'] = df_behavior_TR['slider_reaction_time'].fillna(0.75)

# Insert the 'New path starts' info
df_behavior_TR_new = pd.DataFrame(columns=df_behavior_TR.columns)
for idx, row in df_behavior_TR.iterrows():
    if pd.notna(row['new_run_info_onset']): 
        # initalise 1 nan row df with columns as df_behavior_TR_new
        nan_row = pd.DataFrame([np.nan] * len(df_behavior_TR.columns)).T
        # match column names
        nan_row.columns = df_behavior_TR.columns
        nan_row['onset_time_block'] = row['new_run_info_onset']
        nan_row['stimulus'] = 'new_run_info'
        nan_row['Block_MRI'] = row['Block_MRI']
        nan_row['slider_reaction_time'] = row['new_run_info_duration']
        df_behavior_TR_new = pd.concat([df_behavior_TR_new, nan_row], ignore_index=True)

    df_behavior_TR_new = pd.concat([df_behavior_TR_new, row.to_frame().T], ignore_index=True)

df_behavior_TR = df_behavior_TR_new


# %% 
# =============================================================================            

# Blocks in rewardlearningiation
runs_rewardlearning = df_behavior_TR.Block_MRI.unique().astype(int)

# Get the number of runs in the rewardlearningiation
nruns_rewardlearning = len(np.unique(df_behavior_TR.Block_MRI))

# The TRs of the presented stimuli across blocks
grouped_rewardlearning = df_behavior_TR.groupby('Block_MRI')

# create a list of series, where each series contains the 'tr_idx' values for the corresponding block
trs_vis_scan1_test_rewardlearning = [group['TR_onset_time_block_w_HRF'] for _, group in sorted(grouped_rewardlearning, key=lambda x: x[0])]

# The stimuli label presented across blocks
labels_scan1_video_rewardlearning = df_behavior_TR.stimulus

# Block information across blocks
runvector_scan1_rewardlearning = df_behavior_TR.Block_MRI

# Get paths form main task to sort by stimuklus order for each path
try:
    path_reward_learning = opj(root_folder, f'Behavior/Sess2/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_2.csv') # rewardlearning_complete_participant_{sub_num}_session_{behavior_ses[-1]}_w_TR.csv')
    df_path_reward_learning = pd.read_csv(path_reward_learning, sep=';')
except:
    path_reward_learning = opj(root_folder, f'Behavior/Sess2/subj_{sub_num}/maindata_complete_participant_{sub_num}_session_2_post_exp_completed.csv') # rewardlearning_complete_participant_{sub_num}_session_{behavior_ses[-1]}_w_TR.csv')
    df_path_reward_learning = pd.read_csv(path_reward_learning, sep=';')

# drop all nas in base paths columns
df_path_reward_learning = df_path_reward_learning.dropna(subset=['base_path_A_num1', 'base_path_A_star_num3', 'base_path_B_num2', 'base_path_C_num4']).reset_index(drop=True)

# Get the base paths
base_path_A_num1 = ast.literal_eval(df_path_reward_learning['base_path_A_num1'][0])
base_path_A_star_num3 = ast.literal_eval(df_path_reward_learning['base_path_A_star_num3'][0])
base_path_B_num2 = ast.literal_eval(df_path_reward_learning['base_path_B_num2'][0])
base_path_C_num4 = ast.literal_eval(df_path_reward_learning['base_path_C_num4'][0])

keys = list(itertools.chain.from_iterable([base_path_A_num1, base_path_A_star_num3, base_path_B_num2, base_path_C_num4]))

#%%
'''
========================================================================
LOAD MASKS & CONFOUND FILES FROM THE LOCALISER/TRAINING SCAN
========================================================================
'''
parts = sub.split('-')
if len(parts[1]) == 1:
    mri_sub = f"{parts[0]}-0{parts[1]}"
else:
    mri_sub = sub
    
# load the visual mask task files
#path_mask_vis_task_scan1 = opj(path_masks, 'mask_visual', sub, '*', '*{ses}*task-gems*.nii.gz')
if use_thresholded_masks==1:
    path_mask_vis_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs', f'VIS_{mri_sub}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii')
    path_mask_vis_task_scan1_other_sess = opj(path_masks,mri_sub,ses,'ROIs', f'VIS_{mri_sub}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii')

path_mask_vis_task_scan1 = sorted(glob.glob(path_mask_vis_task_scan1), key=lambda f: os.path.basename(f))
path_mask_vis_task_scan1_localizer = [path_mask_vis_task_scan1[i] for i in runs_rewardlearning - 1] 


# load the hippocampus mask task files
#path_mask_hpc_task_scan1 = opj(path_masks, 'mask_hippocampus',sub, '*', '*{ses}*task-gems*.nii.gz')
if use_thresholded_masks==1:
    path_mask_hpc_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','HPC*T-' + str(mask_threshold) + '*nii')
path_mask_hpc_task_scan1 = sorted(glob.glob(path_mask_hpc_task_scan1), key=lambda f: os.path.basename(f))
path_mask_hpc_task_scan1_localizer = [path_mask_hpc_task_scan1[i] for i in runs_rewardlearning - 1] 

# load EC mask
if use_thresholded_masks==1:
    path_mask_erh_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','ERH*T-' + str(mask_threshold) + '*nii')
    path_mask_erh_task_scan1 = sorted(glob.glob(path_mask_erh_task_scan1), key=lambda f: os.path.basename(f))
path_mask_erh_task_scan1_localizer = [path_mask_erh_task_scan1[i] for i in runs_rewardlearning - 1] 
    
# load PFC mask
if use_thresholded_masks==1:
    path_mask_lpfc_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','LPFC*T-' + str(mask_threshold) + '*nii')
    path_mask_lpfc_task_scan1 = sorted(glob.glob(path_mask_lpfc_task_scan1), key=lambda f: os.path.basename(f))
path_mask_lpfc_task_scan1_localizer = [path_mask_lpfc_task_scan1[i] for i in runs_rewardlearning - 1] 

    
# load rostromiddle frontal mask
#if use_thresholded_masks==1:
#    path_mask_rmf_task_scan1 = opj(path_masks,sub,ses,'ROIs','RMFG*T-' + str(mask_threshold) + '*nii')
#    path_mask_rmf_task_scan1 = sorted(glob.glob(path_mask_rmf_task_scan1), key=lambda f: os.path.basename(f))
    
# load OFC mask
if use_thresholded_masks==1:
    path_mask_ofc_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','OFC*T-' + str(mask_threshold) + '*nii')
    path_mask_ofc_task_scan1 = sorted(glob.glob(path_mask_ofc_task_scan1), key=lambda f: os.path.basename(f))
path_mask_ofc_task_scan1_localizer = [path_mask_ofc_task_scan1[i] for i in runs_rewardlearning - 1] 

# # load ventricles mask
if use_thresholded_masks==1:
    path_mask_ventricles_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','ventricles*T-' + str(mask_threshold) + '*nii')
    path_mask_ventricles_task_scan1 = sorted(glob.glob(path_mask_ventricles_task_scan1), key=lambda f: os.path.basename(f))
path_mask_ventricles_task_scan1_localizer = [path_mask_ventricles_task_scan1[i] for i in runs_rewardlearning - 1] 

# load the medial temporal lobe task files
#path_mask_mtl_task_scan1 = opj(path_masks, 'mask_mtl',sub, '*', '*{ses}*task-gems*.nii.gz')
if use_thresholded_masks==1:
    path_mask_mtl_task_scan1 = opj(path_masks,mri_sub,ses,'ROIs','MTL*T-' + str(mask_threshold) + '*nii')
path_mask_mtl_task_scan1 = sorted(glob.glob(path_mask_mtl_task_scan1), key=lambda f: os.path.basename(f))
path_mask_mtl_task_scan1_localizer = [path_mask_mtl_task_scan1[i] for i in runs_rewardlearning - 1] 

# load the whole brain mask files
#path_mask_whole_brain_scan1 = opj(path_fmriprep, sub, ses, 'func','*{ses}*task-gems*T1w*brain_mask.nii.gz')
if use_thresholded_masks==1:
    path_mask_whole_brain_scan1=opj(path_masks,mri_sub,ses,'ROIs','wholebrain*T-' + str(mask_threshold) + '*nii')
path_mask_whole_brain_scan1 = sorted(glob.glob(path_mask_whole_brain_scan1), key=lambda f: os.path.basename(f))
path_mask_whole_brain_scan1_localizer = [path_mask_whole_brain_scan1[i] for i in runs_rewardlearning - 1] 

"""
TMP FOR DEVELOPING
load local file if mounted and not run on tardis otherwise takes long time to load
"""
# load the functional mri task files
if 'mount' in os.getcwd():
    path_func_task_scan1 = opj('/Users/renzadmin/MaxPlanck_PhD/PhD_Schuck/P1_ReplayedReps/project_code/RepChanges_MRI_analysis_code_and_data_Leipzig/Example_MRI_sub5',mri_sub,ses,'func',f'{mri_sub}_{ses}_task-rewardlearning*T1w_desc-preproc_bold.nii.gz')
else:   
    path_func_task_scan1 = opj(path_fmriprep,mri_sub,ses,'func',f'{mri_sub}_{ses}_task-rewardlearning*T1w_desc-preproc_bold.nii.gz')
path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
path_func_task_scan1_localizer = [path_func_task_scan1[i] for i in runs_rewardlearning - 1] 

# load the anatomical mri file:
path_anat_scan1 = opj(path_fmriprep, mri_sub, 'anat', f'{mri_sub}_desc-preproc_T1w.nii.gz')
path_anat_scan1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))

# load the confounds files
path_confs_task_scan1 = opj(path_fmriprep, mri_sub, ses, 'func','*task-rewardlearning*confounds_timeseries.tsv')
path_confs_task_scan1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
path_confs_task_scan1_localizer = [path_confs_task_scan1[i] for i in runs_rewardlearning - 1] 
#path_mask_vis_task_scan1_rewardlearning = [path_mask_vis_task_scan1[i] for i in runs_rewardlearning] 

'''
========================================================================
LOAD THE MRI DATA FROM THE LOCALISER/TRAINING SCAN:
========================================================================
'''

# =============================================================================
# association & Sequence
# =============================================================================
print('loading Leipzig loc data')

# load T1 mask:
#anat = image.load_img(path_anat_scan1[0]) path needs updating (not actually used currently)
# load visual mask:
mask_vis_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_vis_task_scan1)]
#mask_vis_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_vis_task_scan1_seq)]
#mask_vis_scan1 = mask_vis_scan1 + mask_vis_scan1_seq 
mask_vis_scan1 = masking.intersect_masks(mask_vis_scan1, connected=False, threshold=0)
# load hippocampus mask:
mask_hpc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_hpc_task_scan1)]
#mask_hpc_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_hpc_task_scan1_seq)]
#mask_hpc_scan1 = mask_hpc_scan1 + mask_hpc_scan1_seq
mask_hpc_scan1 = masking.intersect_masks(mask_hpc_scan1, connected=False, threshold=0)
# load entorhinal mask:
mask_erh_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_erh_task_scan1)]
#mask_erh_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_erh_task_scan1_seq)]
#mask_erh_scan1 = mask_erh_scan1 + mask_erh_scan1_seq
mask_erh_scan1 = masking.intersect_masks(mask_erh_scan1, connected=False, threshold=0)
# load PFC mask:
mask_lpfc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_lpfc_task_scan1)]
#mask_lpfc_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_lpfc_task_scan1_seq)]
#mask_lpfc_scan1 = mask_lpfc_scan1 + mask_lpfc_scan1_seq
mask_lpfc_scan1 = masking.intersect_masks(mask_lpfc_scan1, connected=False, threshold=0)
# load rostromiddle frontal mask:
#mask_rmf_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_rmf_task_scan1)]
#mask_rmf_scan1 = masking.intersect_masks(mask_rmf_scan1, connected=False, threshold=0)
# load rostromiddle frontal mask:
mask_ofc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_ofc_task_scan1)]
#mask_ofc_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_ofc_task_scan1_seq)]
#mask_ofc_scan1 = mask_ofc_scan1 + mask_ofc_scan1_seq
mask_ofc_scan1 = masking.intersect_masks(mask_ofc_scan1, connected=False, threshold=0)

#print('Loading ventricles mask for ubject %s' % sub)
mask_ventricles_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_ventricles_task_scan1)]
#mask_ventricles_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_ventricles_task_scan1_seq)]
#mask_ventricles_scan1 = mask_ventricles_scan1 + mask_ventricles_scan1_seq
mask_ventricles_scan1 = masking.intersect_masks(mask_ventricles_scan1, connected=False, threshold=0)

# load hippocampus mask:
mask_mtl_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_mtl_task_scan1)]
#mask_mtl_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_mtl_task_scan1_seq)]
#mask_mtl_scan1 = mask_mtl_scan1 + mask_mtl_scan1_seq
mask_mtl_scan1 = masking.intersect_masks(mask_mtl_scan1, connected=False, threshold=0)
# load wholebrain mask:
mask_wholebrain_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_whole_brain_scan1)]
#mask_wholebrain_scan1_seq = [image.load_img(i) for i in copy.deepcopy(path_mask_whole_brain_scan1_seq)]
#mask_wholebrain_scan1 = mask_wholebrain_scan1 + mask_wholebrain_scan1_seq
mask_wholebrain_scan1 = masking.intersect_masks(mask_wholebrain_scan1, connected=False)


masks_scan1 = {'lpfc': mask_lpfc_scan1, 'vis': mask_vis_scan1, 'hpc': mask_hpc_scan1, 'mtl': mask_mtl_scan1,'wholebrain': mask_wholebrain_scan1, 'erh': mask_erh_scan1, 'ofc': mask_ofc_scan1 , 'ventricles': mask_ventricles_scan1}

if use_reliability_mask:

    # lpfc
    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_lpfc_threshold_70_localizer_{behavior_ses}.nii')
    mask_lpfc_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_vis_threshold_70_localizer_{behavior_ses}.nii')
    mask_vis_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_hpc_threshold_70_localizer_{behavior_ses}.nii')
    mask_hpc_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_erh_threshold_70_localizer_{behavior_ses}.nii')
    mask_erh_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_lpfc_threshold_70_localizer_{behavior_ses}.nii')
    mask_lpfc_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_ofc_threshold_70_localizer_{behavior_ses}.nii')
    mask_ofc_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_mtl_threshold_70_localizer_{behavior_ses}.nii')
    mask_mtl_scan2 = image.load_img(path_to_load)

    path_to_load = opj(root_folder, 'Code', '8_voxel_based_reliability', f'generate_voxel_reliability_mask_maintask{ses[-1]}', f'{sub}_mask_wholebrain_threshold_70_localizer_{behavior_ses}.nii')
    mask_wholebrain_scan2 = image.load_img(path_to_load)

    # print sum of scan 1 and scan 2 for each mask
    print(f'lpfc: {mask_lpfc_scan1.get_fdata().sum()} {mask_lpfc_scan2.get_fdata().sum()}')
    print(f'vis: {mask_vis_scan1.get_fdata().sum()} {mask_vis_scan2.get_fdata().sum()}')
    print(f'hpc: {mask_hpc_scan1.get_fdata().sum()} {mask_hpc_scan2.get_fdata().sum()}')
    print(f'erh: {mask_erh_scan1.get_fdata().sum()} {mask_erh_scan2.get_fdata().sum()}')
    print(f'lpfc: {mask_lpfc_scan1.get_fdata().sum()} {mask_lpfc_scan2.get_fdata().sum()}')
    print(f'ofc: {mask_ofc_scan1.get_fdata().sum()} {mask_ofc_scan2.get_fdata().sum()}')
    print(f'mtl: {mask_mtl_scan1.get_fdata().sum()} {mask_mtl_scan2.get_fdata().sum()}')
    print(f'wholebrain: {mask_wholebrain_scan1.get_fdata().sum()} {mask_wholebrain_scan2.get_fdata().sum()}')

    masks_scan1 = {'lpfc': mask_lpfc_scan2, 'vis': mask_vis_scan2, 'hpc': mask_hpc_scan2, 'mtl': mask_mtl_scan2,'wholebrain': mask_wholebrain_scan2, 'erh': mask_erh_scan2, 'ofc': mask_ofc_scan2, 'ventricles': mask_ventricles_scan1}


# load functional data localizeriation
data_task_scan1 = [image.load_img(i) for i in path_func_task_scan1]
# load functional data sequential
#data_task_scan1_seq = [image.load_img(i) for i in path_func_task_scan1_seq]

# get confound values
confs_scan1 = [pd.read_csv(path_confs_task_scan1[cr], sep='\t') for cr in range(nruns_rewardlearning)]

# set confound variables of interest
conf_vars_localizer_all = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                'rot_x', 'rot_y', 'rot_z',
                'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'rot_x', 'rot_y', 'rot_z',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# %%

'''
========================================================================
Run GLM
========================================================================
'''
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix, plot_anat, plot_img, plot_stat_map
from nilearn.image import concat_imgs, mean_img
import nilearn.image as nimg
from nilearn.glm.first_level import make_first_level_design_matrix
import nibabel as nib

first_level_model = FirstLevelModel(
                            t_r=TR,
                            noise_model="ar1",
                            standardize=True,
                            smoothing_fwhm=4,
                            hrf_model="spm",
                            drift_model="cosine",
                            high_pass=1/128,
                            mask_img=masks_scan1['wholebrain'],
                            minimize_memory=False,
                            slice_time_ref=0.5
                        )

# Plotting options
plot_contrast=0
plot_stats_map=0
save_t_maps = 1
include_confounds_as_nuisance=1

# Number of stimuli, nuisance regressors, and drift regressors
num_stimuli = len(df_behavior_TR['stimulus'].unique())
stimuli = df_behavior_TR['stimulus'].unique().tolist()
contrast_vectors = {}


# prep residual per run output dir
residuals_from_first_level_glm = create_script_named_folder('residuals_from_first_level_glm')

# for mask_name in masks_scan1.keys():
############################################
mask_name = sys.argv[2]
# Here would be the mask loop
############################################


# Initialize a dictionary to hold for every stimukuls the t values for each run
dict_stim_by_t_map = {stim: {} for stim in keys}

# Fit the GLM for each run
############################################
for run in range(1, nruns_rewardlearning + 1):
    # nuisance regressors
    confs_scan1_run = confs_scan1[run - 1][conf_vars_localizer]

    # behavioral data
    df_run = df_behavior_TR[df_behavior_TR['Block_MRI'] == run]
    events_run = pd.DataFrame({
        'onset': df_run['onset_time_block'],
        'duration': df_run['slider_reaction_time'],
        'trial_type': [f'{stim}' for stim in df_run['stimulus']]
    })

    # Get fMRI data for the current run
    fmri_run_img = data_task_scan1[run - 1]
    frame_times = np.arange(fmri_run_img.shape[-1]) * TR
    design_matrix_run = make_first_level_design_matrix(frame_times, events=events_run)

    if include_confounds_as_nuisance:
        # add confounds to design matrix as additional columns
        for conf in confs_scan1_run.columns:
            # fill confs with 0s if NaN with .fillna(0)
            confs_scan1_run.loc[:, conf] = confs_scan1_run[conf].fillna(0).values
            design_matrix_run[conf] = confs_scan1_run[conf].values


    glm_result_run = first_level_model.fit(fmri_run_img, design_matrices=design_matrix_run)

    design_matrix = glm_result_run.design_matrices_[0]
    # plt.figure(figsize=(10, 4))
    # plot_design_matrix(design_matrix)
    # plt.title(f'Design Matrix for Run {run}')
    # plt.show()
    # plt.close()

    # save residuals after masking
    residuals = glm_result_run.residuals[0]
    # mask the residuals with the same mask
    residuals_masked = masking.apply_mask(residuals, masks_scan1[mask_name])
    #convert to array
    masked_res = np.array(residuals_masked)
    # apply shrinkage (recommended, followed form alex eperons code)
    noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(masked_res, method='shrinkage_eye') # or shrinkage_diag - get an estimate of noise, dof is n. regressors in glm-1 (but we can ignore this safely here)
    # save in residuals_from_first_level_glm
    np.save(f'{residuals_from_first_level_glm}/mask_name_{mask_name}_sub_{sub}_run_{run}.npy', noise_pres_res)



    # extract the t values for each run for each stimulus
    ############################################

    # Loop over each stimulus
    for stim in keys:
        # Create a zero vector for the contrast
        contrast_vector = np.zeros(len(design_matrix_run.columns))

        # Find the column index for this stimulus and run
        column_index = design_matrix_run.columns.tolist().index(stim)
        
        # Set the column index in the contrast vector to 1
        contrast_vector[column_index] = 1
        
        # Store the contrast vector in the dictionary with the stimulus as the key
        contrast_vectors[stim] = contrast_vector
            
        if plot_contrast:
            plot_contrast_matrix(contrast_vectors[stim], design_matrix=design_matrix)
            plt.title(f'Contrast Matrix for {stim}')
            plt.show()
            plt.close()

        # Compute the contrast map (e.g., z-score map)
        t_map = glm_result_run.compute_contrast(contrast_vectors[stim], stat_type='t') #output_type='effect_size')
    
        # Compute the mean image for the current run to use as background in plotting
        mean_image_run = mean_img(fmri_run_img)
        
        # Plot the statistical map using the mean image of the current run
        if plot_stats_map:
            plot_stat_map(
                t_map,
                bg_img=mean_image_run,  # Use the mean image as the background
                title=f't-map for stimulus: {stim}',
                threshold=2,
                display_mode='ortho',
                cut_coords=(0, 0, 0),
                colorbar=True
            )
            plt.show()
            plt.close()
        
        # Apply the mask to the t-map
        masked_t_map = masking.apply_mask(t_map, masks_scan1[mask_name])


        # apply mask and replot other masking way
        # masker= NiftiMasker(masks_scan1[f'{mask_name}'], target_affine= t_map.affine, target_shape=t_map.shape)
        # masker.fit()
        # masked_t_map= masker.inverse_transform(masker.transform_single_imgs(t_map)) 
        # if plot_stats_map:
        #     plot_stat_map(
        #         masked_t_map,
        #         bg_img=mean_image_run,  # Use the mean image as the background
        #         title=f't-map for stimulus: {stim} in mask {mask_name}',
        #         threshold=0,
        #         display_mode='ortho',
        #         cut_coords=(0, 0, 0),
        #         colorbar=True
        #     )
        #     plt.show()
        #     plt.close()


        if save_t_maps:

            t_maps_from_first_level_glm_masked = create_script_named_folder(f't_maps_from_first_level_glm_masked')
            t_maps_from_first_level_glm_unmasked = create_script_named_folder(f't_maps_from_first_level_glm_unmasked')

            t_map_filename_unmasked = opj(t_maps_from_first_level_glm_unmasked, f'{sub}_{ses}_{mask_name}_{stim}_run_{run}.nii')
            t_map_filename_masked = opj(t_maps_from_first_level_glm_masked, f'{sub}_{ses}_{mask_name}_{stim}_run_{run}.npy')

            nib.save(t_map, t_map_filename_unmasked)
            np.save(t_map_filename_masked, masked_t_map)

        

        # masked_t_map_flattened = masked_t_map.get_fdata().flatten()
        # masked_t_map_without_0s = masked_t_map_flattened[masked_t_map_flattened != 0]
        # dict_stim_by_t_map[stim][run] = masked_t_map_without_0s

        

    # Clear memory for the current run
    del glm_result_run, fmri_run_img, design_matrix_run, residuals, residuals_masked, masked_res
    gc.collect()


    