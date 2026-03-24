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
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import load_img, math_img
from nilearn.image import mean_img as mean_img_nilearn # optional

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
# path_masks = opj(root_folder, 'code', '3_reliability_voxel_selection', 'combine_loc1_and_loc3_masks')
path_mask = opj(root_folder, 'code', '2_get_threshold_roi', 'localizer_getThresholdedROI', 'thresholded_roi_output') 
path_clf_dir = opj(root_folder, 'code', '4_decoding', '4.1_train_visual_classifier', 'classifiers')
path_behavior = opj(root_folder, 'behavior')

TR=1.5
mri_sessions_enc = 'ses-04'

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
        path_anatiomy_mask = '/Volumes/Untitled/mri/derivatives/sub-13/anat'
        path_clf_dir = '/Volumes/Untitled/sleeplay_git/4_decoding/4.1_train_visual_classifier/classifiers'
        print(f"Setting classifier path to local: {path_clf_dir}")
        print(f"Setting masks path to local: {path_masks}")
else:
    print(
        f"WARNING: No subject specified on command line. "
        f"Using default subject {default_subj}."
    )
    sub = default_subj

sub_num = sub[1:]
sub_mri = f"sub-{sub_num}"
out_dir = Path(create_script_named_folder("glm_maps")) / f"sub-{sub_num}"
out_dir.mkdir(parents=True, exist_ok=True)


run_effect_paths = []
run_var_paths = []
'''
========================================================================
LOAD BEHAVIORAL DATA
========================================================================
'''

# Load df of localizer - this is the behavioral data for the pre task localizer session


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
    
# drop rows where label_type is NaN
df_behavior_retrieval = df_behavior_retrieval.dropna(subset=['label_type']).reset_index(drop=True)

'''
========================================================================
LOAD MASKS & CONFOUND FILES FROM THE LOCALISER/TRAINING SCAN
========================================================================
'''

##########
# load func data
###########
# path_func_task_scan1 = opj(
#     path_fmriprep,
#     sub_mri,
#     mri_sessions_enc,
#     'func',
#     f'{sub_mri}_{mri_sessions_enc}_task-mem*_space-T1w_desc-preproc_bold.nii.gz'
# )
# path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
# data_task_scan = [image.load_img(i) for i in path_func_task_scan1]

# Instead of space-T1w:
path_func_task_scan1 = opj(
path_fmriprep,
sub_mri,
mri_sessions_enc,
"func",
f"{sub_mri}_{mri_sessions_enc}_task-mem*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)
path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
data_task_scan = [image.load_img(i) for i in path_func_task_scan1]

# Keep track of indices BEFORE filtering
original_indices = list(range(len(data_task_scan)))
valid_indices = [i for i, run in enumerate(data_task_scan) if run.shape[-1] > 120]

# Filter func data
data_task_scan = [data_task_scan[i] for i in valid_indices]


##########
# load the masks
##########

# Load masks
get_files = opj(path_mask, f'sub-{sub_num}', mri_sessions_enc, f'wholebrain_sub-{sub_num}_{mri_sessions_enc}_space-T1w_T-30_run*.nii')
get_files_glob = sorted(glob.glob(get_files), key=lambda f: os.path.basename(f))
# Apply the same filter
masks_scan = [get_files_glob[i] for i in valid_indices]


# data_task_scan = [masking.apply_mask(i, masks_scan[cr]) for cr, i in enumerate(data_task_scan)]

# # inverse transform back to 4D nifti
# data_task_scan = [masking.unmask(data_task_scan[cr], masks_scan[cr]) for cr in range(len(data_task_scan))]

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
confs_scan = [pd.read_csv(path_confs_task_loc1[cr], sep='\t') for cr in range(len(path_confs_task_loc1))]

# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
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

# keep only those with more than 120 timepoints (to match func data)
confs_scan = [confs for i, confs in enumerate(confs_scan) if i in valid_indices]

# prrint the confound variables dimensions and see if they match the func data timepoints
for i in range(len(confs_scan)):
    print(f'Confounds scan 1 run {i} shape: {confs_scan[i].shape}')
    print(f'Func data scan 1 run {i} timepoints: {data_task_scan[i].shape[-1]}')

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

plot_designmatrix = 1
plot_contrast = 1
plot_statmap = 1

# =============================================================================
# Design matrix per run
# =============================================================================
control_design_matrices = []
betas_per_run = []
runs = df_behavior_retrieval['run_nr'].unique()
runs = runs[~pd.isna(runs)].astype(int)

# Loop through each run
for run in runs:
    run = int(run)
    # Filter df_behavior for the current run and create events DataFrame
    df_run = df_behavior_retrieval[df_behavior_retrieval['run_nr'] == run]
    
    # check that either sc_order_cue or obj_order_cue is present in df_run in label_type col otherwise continue
    if not any(df_run['label_type'].isin(['sc_order_cue', 'obj_order_cue'])):
        print(f"[run {run}] WARNING: No 'sc_order_cue' or 'obj_order_cue' found in label_type. Skipping run.")
        continue
    
    print(df_run['label_type'].unique())
    
    # ------------------------------------------------------------
    # If ret_resp is missing but ITIs has a value, copy ITIs -> ret_resp
    # (handles NaN and empty strings)
    # ------------------------------------------------------------
    # this is non remembered timeout
    ret_missing = df_run["ret_resp"].isna() | (df_run["ret_resp"].astype(str).str.strip() == "")
    iti_has_val = ~(df_run["ITIs"].isna() | (df_run["ITIs"].astype(str).str.strip() == ""))
    df_run.loc[ret_missing & iti_has_val, "ret_resp"] = df_run.loc[ret_missing & iti_has_val, "ITIs"]

    # ------------------------------------------------------------
    # Get q mark and following row
    # ------------------------------------------------------------
    # Sort by index to ensure proper row ordering
    df_run_sorted = df_run.sort_index()

    # Find q_mark rows
    q_mark_mask = df_run_sorted['label_type'] == 'q_mark'

    # Create a mask for q_mark rows OR rows immediately after q_mark
    next_row_mask = q_mark_mask.shift(1, fill_value=False)

    df_qmark_simple = df_run_sorted[q_mark_mask | next_row_mask].copy()

    # ------------------------------------------------------------
        


    # remembered: q_mark_press => remembered ; ret_fix => not remembered ; q_mark => <NA>
    df_qmark = df_qmark_simple.copy()
    df_qmark["remembered"] = np.select(
        [
            df_qmark["label_type"].eq("q_mark_press"),
            df_qmark["label_type"].eq("ret_fix"),
        ],
        [
            "remembered",
            "not_remembered",
        ],
        default=pd.NA,
    )

    # Rename the *values* in label_type for the response/timeout rows
    df_qmark["label_type2"] = df_qmark["label_type"].replace(
        {
            "q_mark_press": "response_ended_or_timeout",
            "ret_fix": "response_ended_or_timeout",
        }
    )
    
    # backwards fill ret_resp and remembered
    df_qmark['ret_resp'] = df_qmark['ret_resp'].bfill()
    df_qmark['remembered'] = df_qmark['remembered'].bfill()
    
    # subset rows where label_type is q_mark
    df_qmark = df_qmark[df_qmark['label_type'] == 'q_mark'].reset_index(drop=True)
    
    events_run = pd.DataFrame(
        {
            "onset": pd.to_numeric(df_qmark["t_onset"], errors="coerce"),
            "duration": pd.to_numeric(df_qmark["ret_resp"], errors="coerce"),
            "trial_type": df_qmark["remembered"].astype(str),
        }
    ).dropna(subset=["onset", "duration"])
    
    # events run asumng 1 sec duration
    # events_run = pd.DataFrame(
    #     {
    #         "onset": pd.to_numeric(df_qmark["t_onset"], errors="coerce"),
    #         "duration": 1.0,
    #         "trial_type": df_qmark["remembered"].astype(str),
    #     }
    # ).dropna(subset=["onset"])
    
    
    # Extract nuisance regressors for this run and replace NaNs with 0
    nuisance_regressors = confs_scan[int(run-1)][conf_vars_localizer].fillna(0)
    nuisance_regressors = nuisance_regressors.values

    fmri_run_img = data_task_scan[run - 1]

    frame_times = np.arange(fmri_run_img.shape[-1]) * TR
    augmented_design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events_run,
        add_regs=nuisance_regressors,
        add_reg_names=conf_vars_localizer,
        )

    glm_result = first_level_model.fit(
    fmri_run_img,
    design_matrices=augmented_design_matrix
    )
    design_matrix = glm_result.design_matrices_[0]
    control_design_matrices.append(design_matrix)
    
    if plot_designmatrix:
        plot_design_matrix(design_matrix)
        plt.title(f"Design Matrix - Run {run}")
        plt.savefig(opj(create_script_named_folder('figures'),
        f"sub-{sub_num}_run-{run}_design_matrix_retrieval_vs_rest.png"))
        plt.show()


    num_regressors = len(design_matrix.columns)
    stimulus_on_contrast = np.zeros(num_regressors)

    has_remembered = 'remembered' in design_matrix.columns
    has_not_remembered = 'not_remembered' in design_matrix.columns

    if not has_remembered and not has_not_remembered:
        print(f"[run {run}] WARNING: Neither 'remembered' nor 'not_remembered' in design matrix. Skipping contrast for this run.")
        continue

    if not has_remembered:
        print(f"[run {run}] WARNING: No 'remembered' column. Contrast will only use not_remembered.")
    if not has_not_remembered:
        print(f"[run {run}] WARNING: No 'not_remembered' column. Contrast will only use remembered.")

    if has_remembered:
        stimulus_on_contrast[design_matrix.columns.get_loc('remembered')] = 1
    if has_not_remembered:
        stimulus_on_contrast[design_matrix.columns.get_loc('not_remembered')] = -1
    
    if plot_contrast:
        plot_contrast_matrix(
        stimulus_on_contrast,
        design_matrix=design_matrix,
        )

        plt.show()


    # --- compute effect + variance ---
    eff_map = glm_result.compute_contrast(stimulus_on_contrast, output_type="effect_size")
    var_map = glm_result.compute_contrast(stimulus_on_contrast, output_type="effect_variance")


    eff_path = out_dir / f"run-{run:02d}_stimOn_eff.nii.gz"
    var_path = out_dir / f"run-{run:02d}_stimOn_var.nii.gz"
    eff_map.to_filename(eff_path)
    var_map.to_filename(var_path)


    run_effect_paths.append(str(eff_path))
    run_var_paths.append(str(var_path))


    # (optional) still compute z for QC visualization
    if plot_statmap:
        output_type = "stat"
        z_map = glm_result.compute_contrast(stimulus_on_contrast, output_type=output_type)
        mean_image_run = mean_img(fmri_run_img)
        plot_stat_map(
        z_map,
        bg_img=mean_image_run,
        title=f"Run {run}: Stimulus On vs Off, {output_type}",
        threshold=1.96,
        display_mode="ortho",
        cut_coords=(20, 0, 0),
        colorbar=True,
        )
        plt.savefig(opj(create_script_named_folder('figures'),
        f"sub-{sub_num}_run-{run}_zmap_retrieval_vs_rest.png"))
        plt.show()
        
        
        from nilearn.plotting import view_img
        from nilearn.image import mean_img
        from pathlib import Path

        # # Example: interactive view
        # view = view_img(
        # z_map, # or beta_map / psc_map
        # threshold=2.0, # set None for no threshold,
        # bg_img = mean_image_run,
        # symmetric_cmap=True,
        # annotate=True
        # )
        # view
        

    
# ============================================================
# Precision-weighted fixed-effects combine across runs (within subject)
# ============================================================
from nilearn.image import load_img, math_img
from pathlib import Path

assert len(run_effect_paths) == len(run_var_paths), "Mismatch: effects vs variances"
if len(run_effect_paths) == 0:
    raise ValueError("No run-level maps collected. run_effect_paths is empty.")

# Load run-level images
eff_imgs = [load_img(p) for p in run_effect_paths]
var_imgs = [load_img(p) for p in run_var_paths]

# If your contrasts are sometimes 4D with a singleton last dim, squeeze them to 3D
eff_imgs = [math_img("np.squeeze(img)", img=img) for img in eff_imgs]
var_imgs = [math_img("np.squeeze(img)", img=img) for img in var_imgs]

# Compute inv-variance and weighted effects
inv_var_imgs = [math_img("1.0 / (v + 1e-8)", v=v) for v in var_imgs]
w_eff_imgs   = [math_img("e / (v + 1e-8)", e=e, v=v) for e, v in zip(eff_imgs, var_imgs)]

# Sum across runs (build expressions dynamically)
sum_inv_var = math_img(
    " + ".join([f"i{i}" for i in range(len(inv_var_imgs))]),
    **{f"i{i}": inv_var_imgs[i] for i in range(len(inv_var_imgs))}
)

sum_w_eff = math_img(
    " + ".join([f"w{i}" for i in range(len(w_eff_imgs))]),
    **{f"w{i}": w_eff_imgs[i] for i in range(len(w_eff_imgs))}
)

# Precision-weighted mean effect and its variance
sub_eff = math_img("swe / (siv + 1e-8)", swe=sum_w_eff, siv=sum_inv_var)
sub_var = math_img("1.0 / (siv + 1e-8)", siv=sum_inv_var)

# Save subject-level maps
sub_eff_path = out_dir / "sublevel_qmark_eff.nii.gz"
sub_var_path = out_dir / "sublevel_qmark_var.nii.gz"
sub_eff.to_filename(sub_eff_path)
sub_var.to_filename(sub_var_path)

print("Saved subject-level precision-weighted maps:")
print("  effect:", sub_eff_path)
print("  var   :", sub_var_path)

# ============================================================
# Simple (unweighted) mean across runs — for QC / visualization
# ============================================================
from nilearn.image import mean_img, load_img
from pathlib import Path


# Load run-level effect maps
eff_imgs = [load_img(p) for p in run_effect_paths]


# Squeeze to 3D if needed
eff_imgs = [math_img("np.squeeze(img)", img=img) for img in eff_imgs]


# Compute simple average
sub_eff_mean = mean_img(eff_imgs)


# Save
sub_eff_mean_path = out_dir / "sublevel_qmark_eff_MEAN.nii.gz"
sub_eff_mean.to_filename(sub_eff_mean_path)


print("Saved simple mean effect map:")
print(" ", sub_eff_mean_path)
        
