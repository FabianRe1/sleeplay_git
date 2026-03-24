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
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img


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
path_mask = opj(root_folder, 'code', '2_get_threshold_roi', 'localizer_getThresholdedROI', 'thresholded_roi_output')
path_clf_dir = opj(root_folder, 'code', '4_decoding', '4.1_train_visual_classifier', 'classifiers')
path_behavior = opj(root_folder, 'behavior')

TR=1.5
mri_sessions_enc = 'ses-06'  # postsleep retrieval session



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
# Load main-task ENCODING behavior using glob to find the highest step
# folder for nses_0
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
print(f"[behavior] Using highest step for encoding: {os.path.basename(nses_0_folder)}")

path_behavior_enc_best = opj(nses_0_folder, f'output_df_sub{sub_num}_w_HRF.csv')

# 1) Load the best (highest step) folder for encoding
try:
    df_step0 = pd.read_csv(path_behavior_enc_best, sep=',')
except Exception as e:
    raise FileNotFoundError(
        f"Could not read encoding behavior file: {path_behavior_enc_best}. Error: {e}"
    )

# 2) Decide whether step0 contains encoding label_types
if "label_type" not in df_step0.columns:
    print("[behavior] step0 has no 'label_type' column -> using step0 as-is for encoding.")
    df_behavior_enc = df_step0.copy()
else:
    label_types_present = set(df_step0["label_type"].dropna().astype(str).unique().tolist())
    print(f"[behavior] Encoding step0 label_types: {sorted(list(label_types_present))}")
    df_behavior_enc = df_step0.copy()

# ---------------------------------------------------------------------
# Load POSTSLEEP RETRIEVAL behavior
# Use nses_2 which contains BOTH rounds of postsleep retrieval (6 runs)
# ---------------------------------------------------------------------
nses_2_pattern = opj(
    root_folder,
    'behavior',
    sub,
    'behav',
    'main',
    f'sub{sub_num}_main_task_nses_2_step*',
)
nses_2_folders = sorted(glob.glob(nses_2_pattern))

if not nses_2_folders:
    raise FileNotFoundError(
        f"No nses_2_step* folders found matching: {nses_2_pattern}"
    )

# Pick the folder with the highest step number
nses_2_folder = max(
    nses_2_folders,
    key=lambda f: int(re.search(r'step(\d+)', f).group(1))
)
print(f"[behavior] Found {len(nses_2_folders)} nses_2 folder(s): {nses_2_folders}")
print(f"[behavior] Using highest step: {os.path.basename(nses_2_folder)}")

path_behavior_ret = opj(nses_2_folder, f'output_df_sub{sub_num}_w_HRF.csv')

try:
    df_behavior_retrieval = pd.read_csv(path_behavior_ret, sep=',')
    print(f"[behavior] Loaded postsleep retrieval from: {path_behavior_ret}")
except Exception as e:
    raise FileNotFoundError(
        f"Could not load postsleep retrieval behavior file: {path_behavior_ret}. Error: {e}"
    )

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
LOAD MASKS & CONFOUND FILES FROM THE POSTSLEEP RETRIEVAL SCAN (ses-06)
========================================================================
'''

# Instead of space-T1w:
path_func_task_scan1 = opj(
path_fmriprep,
sub_mri,
mri_sessions_enc,
"func",
f"{sub_mri}_{mri_sessions_enc}_task-mem*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)


path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))

# Read only headers (no voxel data) to determine run lengths and pick the 6 longest
import nibabel as nib
run_lengths = [(i, nib.load(p).shape[-1]) for i, p in enumerate(path_func_task_scan1)]
n_total_runs = len(run_lengths)
if n_total_runs != 9:
    print(f"WARNING: Expected 9 total runs (3 short + 6 retrieval), but got {n_total_runs}.")

if n_total_runs < 6:
    raise ValueError(f"Expected at least 6 retrieval runs, but only got {n_total_runs} total runs.")

run_lengths_sorted = sorted(run_lengths, key=lambda x: x[1], reverse=True)
top6_indices = sorted([idx for idx, _ in run_lengths_sorted[:6]])  # keep original order

print(f"All run lengths: {[(i, l) for i, l in run_lengths]}")
print(f"Selected 6 longest runs (indices): {top6_indices} with lengths {[run_lengths[i][1] for i in top6_indices]}")

# Load only the 6 longest retrieval runs into memory
data_task_scan = [image.load_img(path_func_task_scan1[i]) for i in top6_indices]

##########
# load the masks
##########
mni_mask = load_mni152_brain_mask()  # MNI152 brain mask shipped with nilearn
target_img = data_task_scan[0]       # one of your MNI-space BOLD runs
mni_mask_resampled = resample_to_img(mni_mask, target_img, interpolation="nearest")

###########
# load the confounds files
############
path_confs_task_scan1 = opj(
    path_fmriprep,
    sub_mri,
    mri_sessions_enc,
    'func',
    f'{sub_mri}_{mri_sessions_enc}_task-mem*_run*_desc-confounds_timeseries.tsv'
)  # e.g., sub-13_ses-06_task-mem_run-01_desc-confounds_timeseries.tsv
path_confs_task_loc1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
confs_scan = [pd.read_csv(path_confs_task_loc1[cr], sep='\t') for cr in range(len(path_confs_task_loc1))]

# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'rot_x', 'rot_y', 'rot_z',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# Keep only the 6 longest runs (same as func data)
confs_scan = [confs_scan[i] for i in top6_indices]

# print the confound variables dimensions and see if they match the func data timepoints
for i in range(len(confs_scan)):
    print(f'Confounds scan 1 run {i} shape: {confs_scan[i].shape}')
    print(f'Func data scan 1 run {i} timepoints: {data_task_scan[i].shape[-1]}')

'''
========================================================================
Implement GLM to get beta/t-maps and save noise matrices (residuals from GLM) for multivariate noise normalization
========================================================================
'''



# --- MODIFY your FirstLevelModel definition to include mask_img ---
first_level_model = FirstLevelModel(
    t_r=TR,
    standardize=True,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=1/128,
    mask_img=mni_mask_resampled,
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
ls_augmented_design_matrices = []
ls_functional_data = []

# ---------------------------------------------------------------------
# IMPORTANT: behavioral run_nr values often refer to the *task run index*
# (e.g., 1-6), while data_task_scan is indexed 0..5 after filtering.
# Build an explicit mapping from observed behavioral run numbers -> [0..5].
# ---------------------------------------------------------------------
observed_run_nrs = sorted(runs.tolist())
run_nr_to_data_idx = {rn: i for i, rn in enumerate(observed_run_nrs)}
print(f"[postsleep] Behavioral run_nrs: {observed_run_nrs}")
print(f"[postsleep] run_nr -> data_idx mapping: {run_nr_to_data_idx}")

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

    # Map behavioral run_nr to data index
    data_idx = run_nr_to_data_idx.get(run)
    if data_idx is None or data_idx >= len(data_task_scan):
        print(f"[run {run}] WARNING: No matching data index for run_nr={run}. Skipping.")
        continue

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


    # Extract nuisance regressors for this run using mapped index
    nuisance_regressors = confs_scan[data_idx][conf_vars_localizer].fillna(0)
    nuisance_regressors = nuisance_regressors.values

    fmri_run_img = data_task_scan[data_idx]
    ls_functional_data.append(fmri_run_img)

    frame_times = np.arange(fmri_run_img.shape[-1]) * TR
    augmented_design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events_run,
        add_regs=nuisance_regressors,
        add_reg_names=conf_vars_localizer,
        )


    ls_augmented_design_matrices.append(augmented_design_matrix)




# =============================================================================
# Stack to 1 large GLM design matrix across runs and fit 1GLM across runs
# =============================================================================


# drop dead periods from functional data and design matrices if at the end of not_remembered and remembered conditions nothing is happening for more than 10 TRs
cleaned_functional_data = []
cleaned_design_matrices = []

for run_idx in range(len(ls_functional_data)):
    func_data = ls_functional_data[run_idx]
    design_matrix = ls_augmented_design_matrices[run_idx]

    # smooth functional data
    func_data = nimg.smooth_img(func_data, fwhm=4)

    print(f"\n[Run {run_idx}] BEFORE cleaning:")
    print(f"  Functional data shape: {func_data.shape}")
    print(f"  Design matrix shape: {design_matrix.shape}")

    # get last timepoint where remembered or not_remembered is 1
    # Handle case where a run has only one condition (all remembered or all not_remembered)
    cond_cols_present = [c for c in ['remembered', 'not_remembered'] if c in design_matrix.columns]
    if cond_cols_present:
        cond_mask = np.zeros(len(design_matrix), dtype=bool)
        for c in cond_cols_present:
            cond_mask |= (design_matrix[c] != 0).values
    else:
        cond_mask = np.zeros(len(design_matrix), dtype=bool)
    cond_indices = np.where(cond_mask)[0]
    if len(cond_indices) == 0:
        last_cond_timepoint = 0
    else:
        last_cond_timepoint = cond_indices[-1]

    # define cutoff timepoint
    cutoff_timepoint = last_cond_timepoint + 10  # 5 TRs after last condition

    # slice functional data and design matrix
    cleaned_func_data = func_data.slicer[..., :cutoff_timepoint+1]
    cleaned_design_matrix = design_matrix.iloc[:cutoff_timepoint+1, :].reset_index(drop=True)

    print(f"[Run {run_idx}] AFTER cleaning:")
    print(f"  Functional data shape: {cleaned_func_data.shape}")
    print(f"  Design matrix shape: {cleaned_design_matrix.shape}")

    cleaned_functional_data.append(cleaned_func_data)
    cleaned_design_matrices.append(cleaned_design_matrix)

# lengths (nTRs) per run
lengths_cleaned = [data.shape[-1] for data in cleaned_functional_data]
total_len = sum(lengths_cleaned)

def pad_to_global(run_idx: int, vec) -> pd.Series:
    """
    Take a length lengths_cleaned[run_idx] regressor (array/Series),
    and pad it with zeros so it lives in the correct block of a
    concatenated time axis across all runs.
    """
    vec = pd.Series(np.asarray(vec).squeeze()).reset_index(drop=True)
    if len(vec) != lengths_cleaned[run_idx]:
        raise ValueError(
            f"Run {run_idx+1}: expected length {lengths_cleaned[run_idx]}, got {len(vec)}"
        )

    left = sum(lengths_cleaned[:run_idx])
    right = total_len - left - lengths_cleaned[run_idx]

    return pd.concat(
        [pd.Series(np.zeros(left)), vec, pd.Series(np.zeros(right))],
        ignore_index=True
    )

# task regressors + run-specific intercepts (padded into global time axis)
n_runs = len(cleaned_design_matrices)
task_regressors = {}
for ri in range(n_runs):
    run_label = f"run{ri+1}"
    dm_cols = cleaned_design_matrices[ri].columns
    if 'remembered' in dm_cols:
        task_regressors[f'remembered_{run_label}'] = pad_to_global(ri, cleaned_design_matrices[ri]['remembered'])
    else:
        print(f"[run {ri+1}] WARNING: No 'remembered' column in design matrix. Skipping this regressor for run {ri+1}.")
    if 'not_remembered' in dm_cols:
        task_regressors[f'not_remembered_{run_label}'] = pad_to_global(ri, cleaned_design_matrices[ri]['not_remembered'])
    else:
        print(f"[run {ri+1}] WARNING: No 'not_remembered' column in design matrix. Skipping this regressor for run {ri+1}.")
    task_regressors[f'intercept_{run_label}'] = pad_to_global(ri, np.ones(lengths_cleaned[ri]))

def pad_drift_regressors(
    cleaned_design_matrices,
    lengths_cleaned,
    pad_to_global_fn,
    drift_prefix="drift",
    run_label_prefix="run",
    drop_constant=True,
):
    """
    Build a single (total_len x n_drift_total) DataFrame of *padded* drift regressors.

    For each run:
      - find drift columns matching drift_prefix (default: 'cosine*')
      - optionally drop 'constant' (recommended if you add intercept_runX yourself)
      - pad each drift regressor into the global time axis via pad_to_global_fn
      - rename columns to be unique per run: e.g. 'cosine_01_run2'

    Returns
    -------
    drift_all_padded : pd.DataFrame
        Rows = sum(lengths_cleaned)
        Cols = all drift regressors across runs, run-suffixed.
    drift_cols_by_run : list[list[str]]
        Drift column names detected per run (original names, before suffixing).
    """
    total_len = int(np.sum(lengths_cleaned))
    drift_blocks = []
    drift_cols_by_run = []

    for run_idx, dm in enumerate(cleaned_design_matrices):
        dm_run = dm.copy()

        if drop_constant and "constant" in dm_run.columns:
            dm_run = dm_run.drop(columns=["constant"])

        drift_cols = [c for c in dm_run.columns if c.startswith(drift_prefix)]
        drift_cols_by_run.append(drift_cols)

        if len(drift_cols) == 0:
            print(f"[pad_drift_regressors] WARNING: run {run_idx+1} has no drift cols starting with '{drift_prefix}'.")
            continue

        for c in drift_cols:
            padded = pad_to_global_fn(run_idx, dm_run[c])
            # make name stable + unique
            new_name = f"{c}_{run_label_prefix}{run_idx+1}"
            drift_blocks.append(padded.rename(new_name))

    if drift_blocks:
        drift_all_padded = pd.concat(drift_blocks, axis=1).reset_index(drop=True)
    else:
        drift_all_padded = pd.DataFrame(index=range(total_len))

    # sanity
    if len(drift_all_padded) != total_len:
        raise ValueError(
            f"drift_all_padded len {len(drift_all_padded)} != total_len {total_len}"
        )

    return drift_all_padded, drift_cols_by_run


# ---- usage ----
drift_all_padded, drift_cols_by_run = pad_drift_regressors(
    cleaned_design_matrices=cleaned_design_matrices,
    lengths_cleaned=lengths_cleaned,
    pad_to_global_fn=pad_to_global,
    drift_prefix="drift",      # or whatever your drift columns start with
    run_label_prefix="run",
    drop_constant=True,
)

print("Detected drift cols per run:", drift_cols_by_run)
print("Padded drift shape:", drift_all_padded.shape)


# 1) extract nuisance per run (same columns each time) and reset index
nuis_by_run = [
    dm.loc[:, conf_vars_localizer].reset_index(drop=True)
    for dm in cleaned_design_matrices
]

# 2) stack runs vertically -> ONE column per nuisance regressor
nuisance_regressors_all = pd.concat(nuis_by_run, axis=0, ignore_index=True)

# 3) build task part (already padded to global axis)
task_part = pd.DataFrame(task_regressors).reset_index(drop=True)

# 4) merge side-by-side
final_design_matrix = pd.concat([task_part, nuisance_regressors_all, drift_all_padded], axis=1)

# 5) sanity checks (this catches the usual reason it "still didn't work": length mismatch)
total_len = sum(lengths_cleaned)
assert len(nuisance_regressors_all) == total_len, f"nuis len {len(nuisance_regressors_all)} != {total_len}"
assert len(final_design_matrix) == total_len, f"final len {len(final_design_matrix)} != {total_len}"

# plot correlation matrix
# Compute correlation matrix
corr = final_design_matrix.corr()
# Plot correlation matrix with larger figure size
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, ha='right', fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()


plot_design_matrix(final_design_matrix)
plt.title(f"Design Matrix all runs")
plt.savefig(opj(create_script_named_folder('figures'),
f"sub-{sub_num}_allruns_design_matrix_.png"))
plt.show()

# stack the cleaned functional data across runs
stacked_func_data = concat_imgs(cleaned_functional_data)

# free intermediate data no longer needed
import gc
del cleaned_functional_data, cleaned_design_matrices, ls_functional_data, ls_augmented_design_matrices
del data_task_scan, confs_scan
gc.collect()

# fit 1GLM
glm_1GLM_result = first_level_model.fit(
    stacked_func_data,
    design_matrices=final_design_matrix
)

# compute contrasts
num_regressors_1GLM = len(final_design_matrix.columns)
stimulus_on_contrast_1GLM = np.zeros(num_regressors_1GLM)

# get column indices for all 'remembered' and 'not_remembered' runs
remembered_cols = [col for col in final_design_matrix.columns if col.startswith('remembered_run')]
not_remembered_cols = [col for col in final_design_matrix.columns if col.startswith('not_remembered_run')]

if not remembered_cols and not not_remembered_cols:
    raise ValueError("Neither 'remembered' nor 'not_remembered' columns found in any run. Cannot compute contrast.")

if not remembered_cols:
    print("WARNING: No 'remembered' columns found in any run. Contrast will only use not_remembered (negative).")
if not not_remembered_cols:
    print("WARNING: No 'not_remembered' columns found in any run. Contrast will only use remembered (positive).")

# set weights so that the positive and negative sides each sum to +1 and -1
# this ensures the contrast sums to 0 even when runs are missing a condition
n_rem = len(remembered_cols)
n_notrem = len(not_remembered_cols)
w_rem = 1.0 / n_rem if n_rem > 0 else 0
w_notrem = -1.0 / n_notrem if n_notrem > 0 else 0

for col in remembered_cols:
    stimulus_on_contrast_1GLM[final_design_matrix.columns.get_loc(col)] = w_rem
for col in not_remembered_cols:
    stimulus_on_contrast_1GLM[final_design_matrix.columns.get_loc(col)] = w_notrem

print(f"Contrast weights: remembered={w_rem:.3f} (x{n_rem}), not_remembered={w_notrem:.3f} (x{n_notrem}), sum={stimulus_on_contrast_1GLM.sum():.6f}")

# plot contrast
plot_contrast_matrix(
    stimulus_on_contrast_1GLM,
    design_matrix=final_design_matrix,
)

# show plot
output_type = "effect_size"  # "z_score" or "effect_size"
eff_map = glm_1GLM_result.compute_contrast(stimulus_on_contrast_1GLM, output_type=output_type)
mean_image = mean_img(stacked_func_data)
plot_stat_map(
eff_map,
bg_img=mean_image,
title=f"All runs: remembered vs not remembered, {output_type}",
threshold=1.96,
display_mode="ortho",
cut_coords=(20, 0, 0),
colorbar=True,
# save
output_file=opj(create_script_named_folder('figures'),
f"sub-{sub_num}_allruns_{output_type}_retrieval_vs_rest.png"))

eff_map = glm_1GLM_result.compute_contrast(stimulus_on_contrast_1GLM, output_type=output_type)
var_map = glm_1GLM_result.compute_contrast(stimulus_on_contrast_1GLM, output_type="effect_variance")

mni_mask_effgrid = resample_to_img(mni_mask, eff_map, interpolation="nearest")


# apply mask for clean visualization + saving
eff_map_masked = math_img("img * mask", img=eff_map, mask=mni_mask_effgrid)
var_map_masked = math_img("img * mask", img=var_map, mask=mni_mask_effgrid)

# plot ONLY masked
plot_stat_map(
    eff_map_masked,
    bg_img=mean_image,
    title=f"All runs: remembered vs not remembered, {output_type} (masked)",
    threshold=1.96,
    display_mode="ortho",
    cut_coords=(20, 0, 0),
    colorbar=True,
    output_file=opj(
        create_script_named_folder('figures'),
        f"sub-{sub_num}_allruns_{output_type}_retrieval_vs_rest_masked.png"
    )
)

# save masked maps
eff_path = out_dir / "Big_GLM_remem_vs_nonremem_effect.nii.gz"
var_path = out_dir / "Big_GLM_remem_vs_nonremem_var.nii.gz"
eff_map_masked.to_filename(eff_path)
var_map_masked.to_filename(var_path)

print("Saved masked maps:")
print("  eff:", eff_path)
print("  var:", var_path)
