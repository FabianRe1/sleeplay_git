# -*- coding: utf-8 -*-
'''
========================================================================
#%% IMPORT PACKAGES:
========================================================================
'''
import glob
import os
import tqdm
from os.path import join as opj
import pandas as pd
import copy
import sys
import re
#import scipy
import numpy as np
#import datalad.api as dl
#from scipy.stats import sem, spearmanr, pearsonr
from nilearn import image, masking, signal, plotting
#from nilearn.decoding import Decoder

#from sklearn import metrics
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from groo.groo import get_root
from nilearn.maskers import NiftiMasker

from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix, plot_anat, plot_img, plot_stat_map
from nilearn.image import concat_imgs, mean_img
import nilearn.image as nimg
from nilearn.glm.first_level import make_first_level_design_matrix
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def combine_masks(roi_mask, correlation_mask):
    """
    Combine two masks using nilearn's intersect_masks function.
    This function will keep only the voxels that are non-zero in both masks.
    Args:
    - roi_mask: NIfTI image object for the ROI mask
    - correlation_mask: NIfTI image object for the correlation mask
    """
    combined_mask = masking.intersect_masks([roi_mask, correlation_mask], threshold=1, connected=False)
    return combined_mask

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
LOAD PROJECT PARAMETERS:
========================================================================
'''

#######
root_folder = get_root(".my_hidden_root_file")
path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
path_masks = opj(root_folder, 'code', '3_reliability_voxel_selection', 'combine_loc1_and_loc3_masks')
path_clf_dir = opj(root_folder, 'code', '4_decoding', '4.1_train_visual_classifier', 'classifiers')
path_behavior = opj(root_folder, 'behavior')

TR=1.5
csmooth = 4
temporal_smoothing_level = 3
mask_threshold=30
reliability_threshold = 70
mri_sessions_enc = 'ses-04'
use_thresholded_masks = 1
# behavior_ses = 'ses-04'

# ---------------------------------------------------------------------
# Subject handling:
#   - If called with an argument, use it as subject
#   - Accept "13", "s13", or "sub-13"
#   - Internally use behavior-style IDs "s13"
#   - On failure or no argument: fall back to default subject and warn
# ---------------------------------------------------------------------
default_subj = "s13"

if len(sys.argv) > 1:
    subj_arg = sys.argv[1]

    # Match forms like "sub-13" from the SLURM script
    m = re.match(r'^sub-(\d+)$', subj_arg)
    if m:
        sub = f"s{m.group(1)}"

    # Pure number ("13") -> "s13"
    elif re.match(r'^\d+$', subj_arg):
        sub = f"s{subj_arg}"

    # Already in behavior form ("s13")
    elif re.match(r'^s\d+$', subj_arg):
        sub = subj_arg

    else:
        print(
            f"WARNING: Unexpected subject format '{subj_arg}'. "
            f"Falling back to default subject {default_subj}."
        )
        sub = default_subj
        # Set paths to local
        path_fmriprep = '/Volumes/Untitled/mri/derivatives'
        print(f"Setting fmriprep path to local: {path_fmriprep}")
        path_masks = '/Volumes/Untitled/mri/3_reliability_voxel_masks_localizers_intersected'
        path_clf_dir = '/Volumes/Untitled/sleeplay_git/4_decoding/4.1_train_visual_classifier/classifiers'
        print(f"Setting classifier path to local: {path_clf_dir}")
        print(f"Setting masks path to local: {path_masks}")
else:
    print(
        f"WARNING: No subject specified on command line. "
        f"Using default subject {default_subj}."
    )
    sub = default_subj



'''
========================================================================
LOAD BEHAVIORAL DATA
========================================================================
'''

# Load df of localizer - this is the behavioral data for the pre task localizer session
sub_num = sub[1:]
sub_mri = f"sub-{sub_num}"

# load loc
path_behavior = opj(root_folder, 'behavior', sub, 'behav', 'loc', f's{sub_num}_df_loc1_w_HRF.csv')
file_list = glob.glob(path_behavior)
if len(file_list) == 0:
    raise FileNotFoundError(f"Could not find localizer behavior file with pattern: {path_behavior}")
df_behavior_loc = pd.read_csv(file_list[0], sep=None, engine='python')
# Build dict mapping each obj_enc_stim -> obj_cat (take first occurrence if duplicates)
obj_enc_to_cat = df_behavior_loc.groupby('obj_enc_stim', sort=False)['obj_cat'].first().to_dict()

# ---------------------------------------------------------------------
# Load main-task behavior using glob to find the highest step folder
# for nses_0, then decide encoding vs retrieval via label_type check
# ---------------------------------------------------------------------
nses_0_pattern = opj(
    root_folder, 'behavior', sub, 'behav', 'main',
    f'*_nses_0_step*'
)
nses_0_folders = sorted(glob.glob(nses_0_pattern))
if not nses_0_folders:
    raise FileNotFoundError(f"No nses_0_step* folders found matching: {nses_0_pattern}")

# Pick the folder with the highest step number
nses_0_folder = max(
    nses_0_folders,
    key=lambda f: int(re.search(r'step(\d+)', f).group(1))
)
print(f"[behavior] Found {len(nses_0_folders)} nses_0 folder(s): {[os.path.basename(f) for f in nses_0_folders]}")
print(f"[behavior] Using highest step: {os.path.basename(nses_0_folder)}")

path_behavior_best = opj(nses_0_folder, f'output_df_sub{sub_num}_w_HRF.csv')

# 1) Load the best (highest step) folder
try:
    df_best = pd.read_csv(path_behavior_best, sep=',')
except Exception as e:
    raise FileNotFoundError(
        f"Could not read behavior file: {path_behavior_best}. Error: {e}"
    )

# 2) Decide whether the best step folder contains both encoding + retrieval (label_type check)
required_label_types = {"sc_enc_fix", "d_no4sc"}

if "label_type" not in df_best.columns:
    print(
        "[behavior] best step folder has no 'label_type' column -> using it for encoding; "
        "falling back to step0 for retrieval."
    )
    use_best_for_both = False
    label_types_present = set()
else:
    label_types_present = set(df_best["label_type"].dropna().astype(str).unique().tolist())
    use_best_for_both = required_label_types.issubset(label_types_present)

if use_best_for_both:
    print(
        f"[behavior] Using highest step folder for encoding+retrieval (NO row subsetting). "
        f"label_type contains: {sorted(list(label_types_present))}"
    )
    df_behavior_enc = df_best.copy()
    df_behavior_retrieval = df_best.copy()
else:
    print(
        f"[behavior] Highest step folder missing expected label_type(s). "
        f"Need {sorted(list(required_label_types))}, found {sorted(list(label_types_present))}. "
        f"-> Using highest step for encoding; loading step0 for retrieval."
    )
    df_behavior_enc = df_best.copy()

    # Fall back to step0 folder for retrieval
    path_behavior_step0 = opj(
        root_folder, 'behavior', sub, 'behav', 'main',
        f'sub{sub_num}_main_task_nses_0_step0',
        f'output_df_sub{sub_num}_w_HRF.csv'
    )
    try:
        df_fallback = pd.read_csv(path_behavior_step0, sep=",")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load step0 behavior file for retrieval: {path_behavior_step0}. Error: {e}"
        )
    df_behavior_retrieval = df_fallback.copy()

# insert next to obj_enc_stim into obj_cat based on loc1 mapping
df_behavior_retrieval['obj_cat'] = df_behavior_retrieval['obj_ret_targ'].map(obj_enc_to_cat)

# forward fill trl_nr and run_nr columns to handle NaNs
if "trl_nr" in df_behavior_retrieval.columns:
    df_behavior_retrieval['trl_nr'] = df_behavior_retrieval['trl_nr'].ffill()
if "run_nr" in df_behavior_retrieval.columns:
    df_behavior_retrieval['run_nr'] = df_behavior_retrieval['run_nr'].ffill()

'''
========================================================================
LOAD MASKS & CONFOUND FILES FROM THE LOCALISER/TRAINING SCAN
========================================================================
'''
##########
# load the reliability masks
##########
path_vis_mask = f'{sub}_mask_vis_threshold_{reliability_threshold}_combined.nii'
mask_vis_scan1 = image.load_img(opj(path_masks, path_vis_mask))
    

##########
# load func data
###########
path_func_task_scan1 = opj(
    path_fmriprep,
    sub_mri,
    mri_sessions_enc,
    'func',
    f'{sub_mri}_{mri_sessions_enc}_task-mem*_space-T1w_desc-preproc_bold.nii.gz'
)
path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
data_task_scan_loc1 = [image.load_img(i) for i in path_func_task_scan1]

# print all dims of func data
for i in range(len(data_task_scan_loc1)):
    print(f'Func data scan 1 run {i} shape: {data_task_scan_loc1[i].shape}')

###########
# load the confounds files
############
path_confs_task_scan1 = opj(
    path_fmriprep,
    sub_mri,
    mri_sessions_enc,
    'func',
    f'{sub_mri}_{mri_sessions_enc}_task-mem*_run*_desc-confounds_timeseries.tsv'
)  # e.g., sub-13_ses-04_task-mem_run-01_desc-confounds_timeseries.tsv
path_confs_task_loc1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
confs_scan_loc1 = [pd.read_csv(path_confs_task_loc1[cr], sep='\t') for cr in range(len(path_confs_task_loc1))]

# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                'rot_x', 'rot_y', 'rot_z',
                'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

'''
========================================================================
Implement GLM to get beta/t-maps and save noise matrices (residuals from GLM) for multivariate noise normalization
========================================================================
'''

#first_level_model = FirstLevelModel(t_r=TR, hrf_model='glover')
first_level_model = FirstLevelModel(
                            t_r=TR,
                            #noise_model="ar1",
                            standardize=True,
                            hrf_model="spm",
                            drift_model="cosine",
                            high_pass=1/128,
                            smoothing_fwhm=4
                        )

plot_designmatrix = 0
plot_contrast = 0
plot_statmap = 0

# =============================================================================
# Design matrix per run
# =============================================================================
control_design_matrices = []
betas_per_run = []
runs = df_behavior['run_nr'].unique()
runs = runs[~pd.isna(runs)].astype(int)

stimulus_categories_scenes = ['canyons01.jpg', 'road01.jpg', 'forest03.jpg', 'desert03.jpg',
'mountain02.jpg', 'winter01.jpg', 'lake01.jpg', 'beach02.jpg',
'waterfall04.jpg', 'autumn03.jpg']

stimulus_categories_objects = ['vehicle', 'furniture', 'fruit', 'clothing', 'kitchen', 'sports', 'music',
'animal', 'tool', 'body']


# Loop through each run
for run in runs:
    run = int(run)
    # Filter df_behavior for the current run and create events DataFrame
    df_run = df_behavior[df_behavior['run_nr'] == run]
    
    # remove rows where encoding_category is NaN
    df_run = df_run[~df_run['encoding_category'].isna()]
    print(df_run.label_type.iloc[0])
    
    events_run = pd.DataFrame({
        'onset': df_run['time_onset'],
        'duration': [3.5 for stim in df_run['encoding_category']],# 0.750,  # Fixed duration for each condition
        'trial_type': df_run['encoding_category']})
    
    if df_run.label_type.iloc[0] == 'obj_enc_stim':
        stimulus_categories = stimulus_categories_objects
    elif df_run.label_type.iloc[0] == 'sc_enc_stim':
        stimulus_categories = stimulus_categories_scenes
    
    # Extract nuisance regressors for this run and replace NaNs with 0
    nuisance_regressors = confs_scan1[int(run)][conf_vars_localizer].fillna(0)
    nuisance_regressors = nuisance_regressors.values

    # Get func data
    fmri_run_img_correlation_analysis = data_task_scan1[run]  # run indexing starts from 1

    # Create an augmented design matrix that includes both events and nuisance regressors
    frame_times = np.arange(fmri_run_img_correlation_analysis.shape[-1]) * TR  # Adjust TR accordingly
    augmented_design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events_run,
        add_regs=nuisance_regressors,
        add_reg_names=conf_vars_localizer  # Names of nuisance regressors
    )
    
    
    # Fit the GLM with the augmented design matrix
    glm_result = first_level_model.fit(fmri_run_img_correlation_analysis, design_matrices=augmented_design_matrix)
    design_matrix = glm_result.design_matrices_[0]
    control_design_matrices.append(design_matrix)
    
    # Designmatrix plotting
    if plot_designmatrix:
        # Plot the design matrix
        plt.figure(figsize=(10, 4))
        plot_design_matrix(design_matrix)
        plt.title(f'Design Matrix for Run {run}')
        plt.show()

    # Initialize a dictionary to store beta estimates for this run
    mean_image_run = mean_img(data_task_scan1[run-1])
    
    # Creating a 3D array of empty lists
    betas_current_run_regressors_of_interest = np.empty(fmri_run_img_correlation_analysis.shape[0:3], dtype=object)
    
    # Filling the array with empty lists
    for x in range(betas_current_run_regressors_of_interest.shape[0]):
        for y in range(betas_current_run_regressors_of_interest.shape[1]):
            for z in range(betas_current_run_regressors_of_interest.shape[2]):
                betas_current_run_regressors_of_interest[x, y, z] = []


    # Compute and store beta estimates for each condition in this run
    for condition in stimulus_categories:
        contrast = np.zeros(len(design_matrix.columns))
        contrast[design_matrix.columns.tolist().index(condition)] = 1
        beta_estimate = glm_result.compute_contrast(contrast, output_type='effect_size')
        
        for i in range(betas_current_run_regressors_of_interest.shape[0]):
            for j in range(betas_current_run_regressors_of_interest.shape[1]):
                for k in range(betas_current_run_regressors_of_interest.shape[2]):
                    betas_current_run_regressors_of_interest[i, j, k].append(beta_estimate.get_fdata()[i, j, k])

    # Append a 3d array. each element is a list with betas for all regressors of interest in the current run
    betas_per_run.append(betas_current_run_regressors_of_interest)

    # Plot a contrast as sanity check
    num_regressors = len(design_matrix.columns)
    stimulus_on_contrast = np.zeros(num_regressors)
    stimulus_on_contrast[0:10] = 1
    if plot_contrast:
        plot_contrast_matrix(stimulus_on_contrast, design_matrix=design_matrix)
        plt.title(f'Contrast Matrix for Run {run}: Stimulus On vs Off')
        plt.show()
    # Plot the statistical map for the contrast
    z_map = glm_result.compute_contrast(stimulus_on_contrast, output_type="z_score")
    mean_image_run = mean_img(data_task_scan1[run-1])
    if plot_statmap:
        plot_stat_map(
            z_map,
            bg_img=mean_image_run,  # Use the mean image as the background
            title=f'Run {run}: Stimulus On vs Off',
            threshold=2.5,
            display_mode='ortho',
            cut_coords=(20, 0, 0),
            colorbar=True)
        plt.show()
        

