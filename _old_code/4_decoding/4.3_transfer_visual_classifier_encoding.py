# -*- coding: utf-8 -*-


'''
========================================================================
IMPORT PACKAGES:
========================================================================
'''
import glob
import os
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
from groo.groo import get_root
from nilearn.maskers import NiftiMasker
from joblib import dump, load
from tqdm import tqdm
import re

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

root_folder = get_root(".my_hidden_root_file")
path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
path_masks = opj(root_folder, 'code', '3_reliability_voxel_selection', 'combine_loc1_and_loc3_masks')
path_clf_dir = opj(root_folder, 'code', '4_decoding', '4.1_train_visual_classifier', 'classifiers')

TR=1.5
csmooth = 4
temporal_smoothing_level = 3
f_threshold=30
reliability_threshold = 70

# ---------------------------------------------------------------------
# Subject handling:
#   - If called with an argument, use it as subject
#   - Accept "13", "s13", or "sub-13"
#   - Internally use behavior-style IDs "s13"
#   - On failure or no argument: fall back to default subject and warn
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Subject + ROI handling
# ---------------------------------------------------------------------
default_subj = "s13"
default_mask = "vis"

if len(sys.argv) >= 3:
    subj_arg = sys.argv[1]
    cmask = sys.argv[2]
elif len(sys.argv) == 2:
    subj_arg = sys.argv[1]
    cmask = default_mask
    print(f"WARNING: No ROI specified. Using default ROI '{default_mask}'.")
else:
    subj_arg = None
    cmask = default_mask
    print(f"WARNING: No subject specified. Using default subject {default_subj} and ROI '{default_mask}'.")

if subj_arg is not None:
    m = re.match(r'^sub-(\d+)$', subj_arg)
    if m:
        sub = f"s{m.group(1)}"
    elif re.match(r'^\d+$', subj_arg):
        sub = f"s{subj_arg}"
    elif re.match(r'^s\d+$', subj_arg):
        sub = subj_arg
    else:
        print(f"WARNING: Unexpected subject format '{subj_arg}'. Falling back to default subject {default_subj}.")
        sub = default_subj
        path_fmriprep = '/Volumes/Untitled/mri/derivatives'
        print(f"Setting fmriprep path to local: {path_fmriprep}")
        path_masks = '/Volumes/Untitled/mri/3_reliability_voxel_masks_localizers_intersected'
        path_clf_dir = '/Volumes/Untitled/code/4_decoding/4.1_train_visual_classifier/classifiers'
        cmask = default_mask
else:
    sub = default_subj



# Define the sessions 

mri_sessions_enc = 'ses-04'

use_overall_classifier1_specific2 = 2


'''
========================================================================
LOAD BEHAVIORAL DATA FOR THE ENCODING session
========================================================================
'''

# Load df of localizer - this is the behavioral data for the pre task localizer session
sub_num = sub[1:]

# load loc
path_behavior = opj(root_folder, 'behavior', sub, 'behav', 'loc', f's{sub_num}_df_loc1_w_HRF.csv')
file_list = glob.glob(path_behavior)
df_behavior_loc1 = pd.read_csv(file_list[0], sep=None, engine='python')
# Build dict mapping each obj_enc_stim -> obj_cat (take first occurrence if duplicates)
obj_enc_to_cat = df_behavior_loc1.groupby('obj_enc_stim', sort=False)['obj_cat'].first().to_dict()

# Load encoding – find the nses_0 folder with the highest step number
nses_0_pattern = opj(root_folder, 'behavior', sub, 'behav', 'main', f'*_nses_0_step*')
nses_0_folders = sorted(glob.glob(nses_0_pattern))
if not nses_0_folders:
    raise FileNotFoundError(f"No nses_0_step* folders found matching: {nses_0_pattern}")
nses_0_folder = max(nses_0_folders, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
print(f"[behavior] Found {len(nses_0_folders)} nses_0 folder(s), using highest step: {os.path.basename(nses_0_folder)}")

path_behavior = opj(nses_0_folder, f'output_df_sub{sub_num}_w_HRF.csv')
df_behavior_encoding = pd.read_csv(path_behavior, sep=',')

# inset next to obj_enc_stim into obj_cat based on loc1 mapping
df_behavior_encoding['obj_cat'] = df_behavior_encoding['obj_enc_stim'].map(obj_enc_to_cat)

# forward fill trl_nr and run_nr columns to handle NaNs
df_behavior_encoding['trl_nr'] = df_behavior_encoding['trl_nr'].ffill()
df_behavior_encoding['run_nr'] = df_behavior_encoding['run_nr'].ffill()





'''
========================================================================
LOAD encoding func data, anatomical data and confounds
========================================================================
'''
path_func_task_scan1 = sorted(glob.glob(
    opj(
        path_fmriprep,
        f"sub-{sub_num}",
        mri_sessions_enc,          # must be exactly "ses-04"
        "func",
        f"sub-{sub_num}_{mri_sessions_enc}_task-mem*_space-T1w_desc-preproc_bold.nii.gz"
    )
))
print('=======================')
print(path_func_task_scan1)

data_task_scan_loc1 = [image.load_img(i) for i in path_func_task_scan1]

# print all dims of func data
for i in range(len(data_task_scan_loc1)):
    print(f'Func data scan 1 run {i} shape: {data_task_scan_loc1[i].shape}')

# Func data scan 1 run 0 shape: (70, 90, 77, 120)
# Func data scan 1 run 1 shape: (70, 90, 77, 208) * scenes
# Func data scan 1 run 2 shape: (70, 90, 77, 120)
# Func data scan 1 run 3 shape: (70, 90, 77, 551) * obj
# Func data scan 1 run 4 shape: (70, 90, 77, 254) * obj
# Func data scan 1 run 5 shape: (70, 90, 77, 120)
# Func data scan 1 run 6 shape: (70, 90, 77, 42)
# Func data scan 1 run 7 shape: (70, 90, 77, 33)
# Func data scan 1 run 8 shape: (70, 90, 77, 299)
# Func data scan 1 run 9 shape: (70, 90, 77, 453)
# Func data scan 1 run 10 shape: (70, 90, 77, 303)


# load the confounds files
path_confs_task_scan1 = opj(path_fmriprep, f'sub-{sub_num}', mri_sessions_enc, 'func', f'sub-{sub_num}_{mri_sessions_enc}_task-mem*_run*_desc-confounds_timeseries.tsv') # sub-13_ses-02_task-loc1_run-02_desc-confounds_timeseries.tsv
path_confs_task_loc1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
confs_scan_loc1 = [pd.read_csv(path_confs_task_loc1[cr], sep='\t') for cr in range(len(path_confs_task_loc1))]
print('=======================')
print(path_confs_task_loc1)
# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                'rot_x', 'rot_y', 'rot_z',
                'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# load the anatomical mri file - we do not need it here, but might want it for plotting. 
path_anat_scan1 = opj(path_fmriprep, f'sub-{sub_num}', 'anat', f'sub-{sub_num}_desc-preproc_T1w.nii.gz')
path_anat_scan_loc1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))




'''
========================================================================
Mask the data
========================================================================
'''

# path_vis_mask = f'{sub}_mask_vis_threshold_{reliability_threshold}_combined.nii'
# mask_vis_scan1 = image.load_img(opj(path_masks, path_vis_mask))

# data_masked_scan1_localizer=[]
# data_masked_scan1_localizer = [masking.apply_mask(i, mask_vis_scan1, smoothing_fwhm=csmooth) for i in data_task_scan_loc1]



path_roi_mask = f'sub-{sub_num}_mask_{cmask}_threshold_{reliability_threshold}_combined.nii' # sub-13_mask_mtl_threshold_70_combined.nii
mask_roi = image.load_img(opj(path_masks, path_roi_mask))

data_masked_scan1_localizer = [
    masking.apply_mask(i, mask_roi, smoothing_fwhm=csmooth) for i in data_task_scan_loc1
]

data_masked_detrend_scan1 = [signal.clean(signals=data_masked_scan1_localizer[cr],
                                    t_r=TR, detrend=True,
                                    standardize='zscore_sample',
                                    confounds=confs_scan_loc1[cr][conf_vars_localizer].fillna(0))
                                    for cr in range(len(data_masked_scan1_localizer))]

'''
========================================================================
LOAD THE CLASSIFIERS
clf.classes_
array(['animal', 'autumn03.jpg', 'beach02.jpg', 'body', 'canyons01.jpg',
       'clothing', 'desert03.jpg', 'forest03.jpg', 'fruit', 'furniture',
       'kitchen', 'lake01.jpg', 'mountain02.jpg', 'music', 'road01.jpg',
       'sports', 'tool', 'vehicle', 'waterfall04.jpg', 'winter01.jpg'],
      dtype=object)
========================================================================
'''

# Define parameters for loading (ensure they match what was used in 4.1)

# Construct path to classifier
# The classifier was saved in a folder named after the 4.1 script

if use_overall_classifier1_specific2 == 1:
    # Load Generic Classifier (Both)
    filename_clf = opj(path_clf_dir, f"s{sub_num}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_3.joblib")
    
    if os.path.exists(filename_clf):
        print(f"Loading GENERIC classifier from: {filename_clf}")
        clf_generic = load(filename_clf)
        clf_scenes = clf_generic
        clf_objects = clf_generic
    else:
        raise FileNotFoundError(f"Generic classifier not found at: {filename_clf}")

elif use_overall_classifier1_specific2 == 2:
    # Load Specific Classifiers
    
    # 1. Scene Classifier
    filename_clf_scenes = opj(path_clf_dir, f"s{sub_num}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_1.joblib")
    if os.path.exists(filename_clf_scenes):
        print(f"Loading SCENE classifier from: {filename_clf_scenes}")
        clf_scenes = load(filename_clf_scenes)
    else:
        raise FileNotFoundError(f"Scene classifier not found at: {filename_clf_scenes}")

    # 2. Object Classifier
    filename_clf_objects = opj(path_clf_dir, f"s{sub_num}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_2.joblib")
    if os.path.exists(filename_clf_objects):
        print(f"Loading OBJECT classifier from: {filename_clf_objects}")
        clf_objects = load(filename_clf_objects)
    else:
        raise FileNotFoundError(f"Object classifier not found at: {filename_clf_objects}")

'''
========================================================================
Apply classifier to encoding data (scenes)
========================================================================
'''
encoding_run1 = data_masked_detrend_scan1[1]
encoding_run3 = data_masked_detrend_scan1[3]
encoding_run4 = data_masked_detrend_scan1[4]

# Get all scene encodings
df_scenes = df_behavior_encoding[df_behavior_encoding['label_type'] == 'dn+sc_cue'].copy()
df_objects = df_behavior_encoding[df_behavior_encoding['label_type'] == 'dn+sc+obj_cue'].copy()

# Define mapping from behavior run_nr to data list index
# User specified: runs 1-3 correspond to encoding_run1 (idx 1), 3 (idx 3), and 4 (idx 4)
run_mapping = {1: 1, 2: 3, 3: 4} 

# Parameters
tr_window = np.arange(-4, 5) # -4 to +4 (9 TRs)
decoding_results = {} # Key: stimulus, Value: list of timecourses

# Iterate over trials
for idx, row in df_scenes.iterrows():
    run_nr = int(row['run_nr'])
        
    data_idx = run_mapping[run_nr]
    fmri_data = data_masked_detrend_scan1[data_idx]
    
    onset_tr = int(np.round(row['TR_time_onset_w_HRF']))
    target_class = row['sc_enc_stim']
        
    class_idx = np.where(clf_scenes.classes_ == target_class)[0][0]
    
    # Extract window
    window_idxs = onset_tr + tr_window
        
    window_idxs = (onset_tr + tr_window).astype(int)
    n_tp = fmri_data.shape[0]
    valid = (window_idxs >= 0) & (window_idxs < n_tp)
    # Pad out-of-bounds frames with zeros (data is z-scored so zero is neutral)
    X_window = np.zeros((len(window_idxs), fmri_data.shape[1]))
    if valid.any():
        X_window[valid, :] = fmri_data[window_idxs[valid], :]
    
    # Predict
    probs = clf_scenes.predict_proba(X_window)
    
    # Get probability of the target class
    target_prob_timecourse = probs[:, class_idx]
    
    if target_class not in decoding_results:
        decoding_results[target_class] = []
    decoding_results[target_class].append(target_prob_timecourse)

# ========================================================================
# PLOT 1: Summary (Mean per class + Grand Average)
# ========================================================================
plt_path = create_script_named_folder('decoding_timecourses_plots')

plt.figure(figsize=(12, 8))

all_timecourses = []

# Plot individual stimulus averages
for stim, timecourses in decoding_results.items():
    # Average for this stimulus
    stim_avg = np.mean(timecourses, axis=0)
    all_timecourses.extend(timecourses)
    
    # Plot individual stimulus average
    plt.plot(tr_window, stim_avg, alpha=0.8, linewidth=2, label=stim)

# Calculate and plot Grand Average
if all_timecourses:
    grand_avg = np.mean(all_timecourses, axis=0)
    plt.plot(tr_window, grand_avg, color='black', linewidth=6, label='Grand Average')

plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Onset +5seconds (TRs)')
plt.ylabel('Decoding Probability (Target Class)')
plt.title(f'Summary: Decoding Timecourse (Subject {sub} {cmask})\ntrained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save the figure
plt.savefig(opj(plt_path, f'{sub}_decoding_timecourses_summary_scene_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png'))
decoding_results_obj_path = create_script_named_folder('decoding_timecourses_data')
# save dict with decoding probs
np.save(opj(decoding_results_obj_path, f'{sub}_{cmask}_decoding_timecourses_scene_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'), decoding_results)
plt.show()

# # ========================================================================
# # PLOT 2: Panel per class (Individual trials + Class Mean)
# # ========================================================================
# n_classes = len(decoding_results)
# n_cols = 4
# n_rows = int(np.ceil(n_classes / n_cols))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
# axes = axes.flatten()

# sorted_stims = sorted(decoding_results.keys())

# for i, stim in enumerate(sorted_stims):
#     timecourses = decoding_results[stim]
#     ax = axes[i]
    
#     # Plot individual trials
#     for trial_tc in timecourses:
#         ax.plot(tr_window, trial_tc, color='gray', alpha=0.3, linewidth=1)
    
#     # Plot class mean
#     stim_avg = np.mean(timecourses, axis=0)
#     ax.plot(tr_window, stim_avg, color='firebrick', linewidth=3, label='Mean')
    
#     ax.set_title(f"{stim} (n={len(timecourses)})")
#     ax.axvline(0, color='black', linestyle='--')
#     ax.grid(True, alpha=0.3)
    
#     if i == 0:
#         ax.legend(loc='upper right')

# # Hide empty subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')

# fig.suptitle(f'Individual Trials per Stimulus Class (Subject {sub})\ntrained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}', fontsize=16)
# fig.text(0.5, 0.04, 'TRs from Onset', ha='center', fontsize=12)
# fig.text(0.04, 0.5, 'Decoding Probability', va='center', rotation='vertical', fontsize=12)
# plt.tight_layout(rect=[0.05, 0.05, 1, 0.97]) # Adjust for suptitle and axis labels
# plt.savefig(opj(plt_path, f'{sub}_decoding_timecourses_per_class_scene_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png'))
# plt.show()

'''
========================================================================
Apply classifier to encoding data (objects)
========================================================================
'''
# Parameters
decoding_results_obj = {} # Key: stimulus, Value: list of timecourses

# Iterate over trials
for idx, row in df_objects.iterrows():
    run_nr = int(row['run_nr'])
        
    data_idx = run_mapping[run_nr]
    fmri_data = data_masked_detrend_scan1[data_idx]
    
    onset_tr = int(np.round(row['TR_time_onset_w_HRF']))
    target_class = row['obj_cat']
        
    class_idx = np.where(clf_objects.classes_ == target_class)[0][0]
    
    # Extract window
    window_idxs = onset_tr + tr_window
        
    window_idxs = (onset_tr + tr_window).astype(int)
    n_tp = fmri_data.shape[0]
    valid = (window_idxs >= 0) & (window_idxs < n_tp)
    # Pad out-of-bounds frames with zeros (data is z-scored so zero is neutral)
    X_window = np.zeros((len(window_idxs), fmri_data.shape[1]))
    if valid.any():
        X_window[valid, :] = fmri_data[window_idxs[valid], :]
    
    # Predict
    probs = clf_objects.predict_proba(X_window)
    
    # Get probability of the target class
    target_prob_timecourse = probs[:, class_idx]
    
    if target_class not in decoding_results_obj:
        decoding_results_obj[target_class] = []
    decoding_results_obj[target_class].append(target_prob_timecourse)
    
plt_path = create_script_named_folder('decoding_timecourses_plots')

plt.figure(figsize=(12, 8))

all_timecourses = []

# Plot individual stimulus averages
for stim, timecourses in decoding_results_obj.items():
    # Average for this stimulus
    stim_avg = np.mean(timecourses, axis=0)
    all_timecourses.extend(timecourses)
    
    # Plot individual stimulus average
    plt.plot(tr_window, stim_avg, alpha=0.8, linewidth=2, label=stim)

# Calculate and plot Grand Average
if all_timecourses:
    grand_avg = np.mean(all_timecourses, axis=0)
    plt.plot(tr_window, grand_avg, color='black', linewidth=6, label='Grand Average')

plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Onset +5seconds (TRs)')
plt.ylabel('Decoding Probability (Target Class)')
plt.title(f'Summary: Decoding Timecourse (Subject {sub} {cmask})\ntrained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save the figure
plt.savefig(opj(plt_path, f'{sub}_decoding_timecourses_summary_obj_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png'))
# save dict with decoding probs
decoding_results_obj_path = create_script_named_folder('decoding_timecourses_data')
np.save(opj(decoding_results_obj_path, f'{sub}_{cmask}_decoding_timecourses_obj_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'), decoding_results_obj)
plt.show()