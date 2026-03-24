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
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel
#from sklearn import metrics
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from groo.groo import get_root
from nilearn.maskers import NiftiMasker
from pathlib import Path

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
use_thresholded_masks = 1
TR = 1.5
mask_threshold = 30
root_folder = get_root(".my_hidden_root_file")
path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
path_behavior = opj(root_folder, 'behavior')

# Base folder where thresholded ROI files are saved by localizer_getThresholdedROI.py
path_thresholded_roi_output = opj(root_folder, 'code', '2_get_threshold_roi', 'localizer_getThresholdedROI','thresholded_roi_output')

path_reliability_masks = create_script_named_folder('reliability_masks')
path_masks_voxel_reliability = path_reliability_masks



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

        print(f"Setting masks path to local: {path_masks}")
else:
    print(
        f"WARNING: No subject specified on command line. "
        f"Using default subject {default_subj}."
    )
    sub = default_subj

# Ensure sub is in sXX format
if not re.match(r'^s\d+$', sub):
    sub = f"s{re.sub(r'[^0-9]', '', sub)}"

# ── Subjects to skip (failed experiments / missing data) ──────────────
# sub-31,33,35,39,49,56 : no behavioral data at all
# sub-54,55,57           : missing localizer behavioral data (no loc/ folder)
SKIP_SUBJECTS = {
    's31', 's33', 's35', 's39', 's49', 's56',   # no behavior folder
    's54', 's55', 's57',                         # no localizer behavior
}
if sub in SKIP_SUBJECTS:
    print(f"SKIP {sub}: known failed experiment – skipping.")
    sys.exit(0)


for ses in ['ses-02', 'ses-03']:
    
    if ses == 'ses-02':
        behavior_ses = 'loc1'
    elif ses == 'ses-03':
        behavior_ses = 'loc2'

    # NOTE: Skipping disabled to overwrite all subjects once.
    # ── Skip if all output masks already exist ────────────────────────────
    _roi_keys = [
        'lpfc', 'vis', 'cortical_memory', 'hpc', 'mtl', 'wholebrain',
        'erh', 'ofc', 'aud', 'somato', 'ventricles',
        'parietal_precuneus', 'parietal', 'precuneus', 'medial_ofc',
        'rostral_acc', 'caudal_acc', 'superior_frontal', 'rsc',
        'posterior_cingulate',
    ]
    _percentages = [10, 30, 50, 70, 90]
    _expected_files = [
        f"{path_masks_voxel_reliability}/{sub}_mask_{k}_threshold_{p}_localizer_{behavior_ses}.nii"
        for k in _roi_keys for p in _percentages
    ]
    if all(os.path.isfile(f) for f in _expected_files):
        print(f"SKIP {sub} {ses} ({behavior_ses}): all {len(_expected_files)} mask files already exist.")
        continue
    # ─────────────────────────────────────────────────────────────────────

    '''
    ========================================================================
    LOAD BEHAVIORAL DATA FOR THE LOCALISER/TRAINING SCAN:
    ========================================================================
    '''
    # Load data
    sub_num = sub[1:]
    path_localizer_scan = opj(root_folder, 'behavior', sub, 'behav', 'loc', f's{sub_num}_df_{behavior_ses}.csv')
    file_list = glob.glob(path_localizer_scan)
    # Autodetect delimiter so both ';' and ',' work
    df_behavior = pd.read_csv(file_list[0], sep=';')

    # Blocks in localizer session
    runs_localizer = df_behavior.run_nr.unique()
    runs_localizer = runs_localizer[~pd.isna(runs_localizer)].astype(int)
    nruns_localizer = len(runs_localizer)
    print(f"[DEBUG] df_behavior shape: {df_behavior.shape}, columns: {list(df_behavior.columns[:10])}")
    print(f"[DEBUG] run_nr unique (raw): {df_behavior.run_nr.unique()}")
    print(f"[DEBUG] runs_localizer (cleaned): {runs_localizer}, nruns={nruns_localizer}")


    # create encoding stimulus column and encoding category column
    def _exclusive_merge(df, col_a, col_b):
        if col_a not in df.columns:
            raise KeyError(f"Missing required column: {col_a}")
        if col_b not in df.columns:
            raise KeyError(f"Missing required column: {col_b}")

        s1 = df[col_a]
        s2 = df[col_b]

        # base non-null masks
        v1_base = ~pd.isna(s1)
        v2_base = ~pd.isna(s2)

        # consider empty/whitespace and common NA tokens as null (but keep original values when chosen)
        def invalid_text_mask(s, base_mask):
            txt = s.astype(str).str.strip().str.lower()
            return base_mask & (
                (txt == '') |
                (txt == 'na') | (txt == 'nan') | (txt == 'none') | (txt == 'null')
            )

        inv1 = invalid_text_mask(s1, v1_base)
        inv2 = invalid_text_mask(s2, v2_base)

        valid1 = v1_base & ~inv1
        valid2 = v2_base & ~inv2

        overlap = valid1 & valid2
        if overlap.any():
            idx = df.index[overlap].tolist()
            raise ValueError(f"Overlap detected between {col_a} and {col_b} at rows {idx[:20]} (showing up to 20); total overlaps: {overlap.sum()}")

        merged = pd.Series(np.nan, index=df.index, dtype=object)
        pick_a = valid1 & ~valid2
        pick_b = valid2 & ~valid1
        merged.loc[pick_a] = s1.loc[pick_a]
        merged.loc[pick_b] = s2.loc[pick_b]
        return merged

    # encoding_stimulus: obj_enc_stim + sc_enc_stim
    df_behavior['encoding_stimulus'] = _exclusive_merge(df_behavior, 'obj_enc_stim', 'sc_enc_stim')
    df_behavior['encoding_category'] = _exclusive_merge(df_behavior, 'obj_cat', 'sc_enc_stim')

    # Check encoding_stimulus and encoding_category if only encoding_category has an entry set it nan as it comes from obj retrieval
    _only_category = df_behavior['encoding_stimulus'].isna() & df_behavior['encoding_category'].notna()
    if _only_category.any():
        df_behavior.loc[_only_category, 'encoding_category'] = np.nan
        print(f"Cleared encoding_category for {int(_only_category.sum())} rows where encoding_stimulus is NaN.")
        
    # store behavior with extra columns under roiginal name
    df_behavior.to_csv(path_localizer_scan, sep=';', index=False)


    '''
    ========================================================================
    LOAD MASKS & CONFOUND FILES FROM THE LOCALISER/TRAINING SCAN
    ========================================================================
    '''

        
    # load the visual mask task files
    if use_thresholded_masks == 1:
        path_mask_vis_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'VIS_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_vis_task_scan1 = sorted(glob.glob(path_mask_vis_task_scan1), key=lambda f: os.path.basename(f))
    
    # load memory mask task files - CORTICAL_MEMORY
    if use_thresholded_masks == 1:
        path_mask_cortical_memory_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'CORTICAL_MEMORY_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_cortical_memory_task_scan1 = sorted(glob.glob(path_mask_cortical_memory_task_scan1), key=lambda f: os.path.basename(f))
        


    # load the hippocampus mask task files
    if use_thresholded_masks == 1:
        path_mask_hpc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'HPC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_hpc_task_scan1 = sorted(glob.glob(path_mask_hpc_task_scan1), key=lambda f: os.path.basename(f))

    # load EC mask
    if use_thresholded_masks == 1:
        path_mask_erh_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'ERH_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
        path_mask_erh_task_scan1 = sorted(glob.glob(path_mask_erh_task_scan1), key=lambda f: os.path.basename(f))
        
    # load PFC mask
    if use_thresholded_masks == 1:
        path_mask_lpfc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'LPFC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
        path_mask_lpfc_task_scan1 = sorted(glob.glob(path_mask_lpfc_task_scan1), key=lambda f: os.path.basename(f))

    # load OFC mask
    if use_thresholded_masks == 1:
        path_mask_ofc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'OFC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
        path_mask_ofc_task_scan1 = sorted(glob.glob(path_mask_ofc_task_scan1), key=lambda f: os.path.basename(f))

    # load the medial temporal lobe task files
    if use_thresholded_masks == 1:
        path_mask_mtl_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'MTL_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_mtl_task_scan1 = sorted(glob.glob(path_mask_mtl_task_scan1), key=lambda f: os.path.basename(f))

    # load the whole brain mask files
    if use_thresholded_masks == 1:
        path_mask_whole_brain_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'wholebrain_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_whole_brain_scan1 = sorted(glob.glob(path_mask_whole_brain_scan1), key=lambda f: os.path.basename(f))

    # load the auditory mask task files
    if use_thresholded_masks == 1:
        path_mask_aud_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'AUD_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_aud_task_scan1 = sorted(glob.glob(path_mask_aud_task_scan1), key=lambda f: os.path.basename(f))

    if use_thresholded_masks == 1:
        path_mask_ventricles_brain_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'ventricles_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_ventricles_brain_scan1 = sorted(glob.glob(path_mask_ventricles_brain_scan1), key=lambda f: os.path.basename(f))

    # load the somatosensory mask task files
    if use_thresholded_masks == 1:
        path_mask_somato_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'SOMATO_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_somato_task_scan1 = sorted(glob.glob(path_mask_somato_task_scan1), key=lambda f: os.path.basename(f))

    # load the parietal_precuneus mask task files
    if use_thresholded_masks == 1:
        path_mask_parietal_precuneus_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'PARIETAL_PRECUNEUS_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_parietal_precuneus_task_scan1 = sorted(glob.glob(path_mask_parietal_precuneus_task_scan1), key=lambda f: os.path.basename(f))

    # load the parietal mask task files
    if use_thresholded_masks == 1:
        path_mask_parietal_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'PARIETAL_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_parietal_task_scan1 = sorted(glob.glob(path_mask_parietal_task_scan1), key=lambda f: os.path.basename(f))

    # load the precuneus mask task files
    if use_thresholded_masks == 1:
        path_mask_precuneus_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'PRECUNEUS_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_precuneus_task_scan1 = sorted(glob.glob(path_mask_precuneus_task_scan1), key=lambda f: os.path.basename(f))

    # load the medial OFC mask task files
    if use_thresholded_masks == 1:
        path_mask_medial_ofc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'MEDIAL_OFC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_medial_ofc_task_scan1 = sorted(glob.glob(path_mask_medial_ofc_task_scan1), key=lambda f: os.path.basename(f))

    # load the rostral ACC mask task files
    if use_thresholded_masks == 1:
        path_mask_rostral_acc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'ROSTRAL_ACC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_rostral_acc_task_scan1 = sorted(glob.glob(path_mask_rostral_acc_task_scan1), key=lambda f: os.path.basename(f))

    # load the caudal ACC mask task files
    if use_thresholded_masks == 1:
        path_mask_caudal_acc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'CAUDAL_ACC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_caudal_acc_task_scan1 = sorted(glob.glob(path_mask_caudal_acc_task_scan1), key=lambda f: os.path.basename(f))

    # load the superior frontal mask task files
    if use_thresholded_masks == 1:
        path_mask_superior_frontal_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'SUPERIOR_FRONTAL_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_superior_frontal_task_scan1 = sorted(glob.glob(path_mask_superior_frontal_task_scan1), key=lambda f: os.path.basename(f))

    # load the RSC mask task files
    if use_thresholded_masks == 1:
        path_mask_rsc_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'RSC_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_rsc_task_scan1 = sorted(glob.glob(path_mask_rsc_task_scan1), key=lambda f: os.path.basename(f))

    # load the posterior cingulate mask task files
    if use_thresholded_masks == 1:
        path_mask_posterior_cingulate_task_scan1 = opj(
            path_thresholded_roi_output,
            f'sub-{sub_num}',
            ses,
            f'POSTERIOR_CINGULATE_sub-{sub_num}_{ses}_space-T1w_T-{mask_threshold}_run-*.nii',
        )
    path_mask_posterior_cingulate_task_scan1 = sorted(glob.glob(path_mask_posterior_cingulate_task_scan1), key=lambda f: os.path.basename(f))


    ##########
    # load the functional mri task files
    ##########
    path_func_task_scan1 = opj(
        path_fmriprep,
        f'sub-{sub_num}',
        ses,
        'func',
        f"sub-{sub_num}_{ses}_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz"
    )
    path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
    
    path_func_mask = opj(
        path_fmriprep,
        f'sub-{sub_num}',
        ses,
        'func',
        f"sub-{sub_num}_{ses}_task-loc*_run-*_desc-brain_mask.nii.gz"
        #sub-18_ses-02_task-loc1_run-13_desc-brain_mask.nii.gz

    )
    path_func_mask = sorted(glob.glob(path_func_mask), key=lambda f: os.path.basename(f))
    
    # # load the anatomical mri file:
    # path_anat_scan1 = opj(path_fmriprep, sub, 'anat', f'{sub}_desc-preproc_T1w.nii.gz')
    # path_anat_scan1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))

    # load the confounds files 
    path_confs_task_scan1 = opj(
        path_fmriprep,
        f'sub-{sub_num}',
        ses,
        'func',
        f'sub-{sub_num}_{ses}_task-*_run*_desc-confounds_timeseries.tsv',
    )  # sub-13_ses-02_task-loc1_run-02_desc-confounds_timeseries.tsv
    path_confs_task_scan1 = sorted(glob.glob(path_confs_task_scan1), key=lambda f: os.path.basename(f))
    print(f"[DEBUG] sub={sub}, ses={ses}, sub_num={sub_num}")
    print(f"[DEBUG] runs_localizer = {runs_localizer} (len={len(runs_localizer)})")
    print(f"[DEBUG] runs_localizer - 1 (indices) = {runs_localizer - 1}")
    print(f"[DEBUG] path_confs_task_scan1 has {len(path_confs_task_scan1)} files:")
    for idx, p in enumerate(path_confs_task_scan1):
        print(f"[DEBUG]   [{idx}] {os.path.basename(p)}")
    print(f"[DEBUG] path_func_task_scan1 has {len(path_func_task_scan1)} files:")
    for idx, p in enumerate(path_func_task_scan1):
        print(f"[DEBUG]   [{idx}] {os.path.basename(p)}")

    # ── Build run-keyed dicts so confound & BOLD files are matched by run ──
    # This handles subjects where behavior has more runs than fMRI (s38/s45)
    # and subjects with missing BOLD runs that cause index misalignment (s50).
    #
    # INDEXING: behavioral run numbers are 0-based, BIDS run labels are 1-based.
    #   behavioral run 0 → BIDS run-01, behavioral run N → BIDS run-(N+1).
    # We key the dicts by BEHAVIORAL run number so lookups in the per-run loop
    # (which iterates behavioral run numbers) work directly.
    def _extract_run_number(filepath):
        """Extract the integer run number from a BIDS-style filename."""
        m_run = re.search(r'run-(\d+)', os.path.basename(filepath))
        return int(m_run.group(1)) if m_run else None

    confs_by_run = {}  # keyed by behavioral run (= BIDS run - 1)
    for f in path_confs_task_scan1:
        rn = _extract_run_number(f)  # BIDS run number (1-based)
        if rn is not None:
            confs_by_run[rn - 1] = f

    func_by_run = {}  # keyed by behavioral run (= BIDS run - 1)
    for f in path_func_task_scan1:
        rn = _extract_run_number(f)
        if rn is not None:
            func_by_run[rn - 1] = f

    # Only keep behavioral runs that have BOTH a confound and a BOLD file
    available_runs = sorted(set(confs_by_run.keys()) & set(func_by_run.keys()))
    runs_localizer = np.array([r for r in runs_localizer if r in available_runs])
    nruns_localizer = len(runs_localizer)
    print(f"[DEBUG] available fMRI runs (conf ∩ func): {available_runs}")
    print(f"[DEBUG] runs_localizer after matching: {runs_localizer}, nruns={nruns_localizer}")

    path_confs_task_scan1_localizer = [confs_by_run[r] for r in runs_localizer]


    '''
    ========================================================================
    LOAD THE MRI DATA FROM THE LOCALISER/TRAINING SCAN:
    ========================================================================
    '''

    mask_vis_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_vis_task_scan1)]
    mask_vis_scan1 = masking.intersect_masks(mask_vis_scan1, connected=False, threshold=0)
    
    mask_cortical_memory_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_cortical_memory_task_scan1)]
    mask_cortical_memory_scan1 = masking.intersect_masks(mask_cortical_memory_scan1, connected=False, threshold=0)

    # load hippocampus mask:
    mask_hpc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_hpc_task_scan1)]
    mask_hpc_scan1 = masking.intersect_masks(mask_hpc_scan1, connected=False, threshold=0)

    # load entorhinal mask:
    mask_erh_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_erh_task_scan1)]
    mask_erh_scan1 = masking.intersect_masks(mask_erh_scan1, connected=False, threshold=0)

    # load PFC mask:
    mask_lpfc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_lpfc_task_scan1)]
    mask_lpfc_scan1 = masking.intersect_masks(mask_lpfc_scan1, connected=False, threshold=0)

    # load OFC mask:
    mask_ofc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_ofc_task_scan1)]
    mask_ofc_scan1 = masking.intersect_masks(mask_ofc_scan1, connected=False, threshold=0)

    # load hippocampus mask:
    mask_mtl_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_mtl_task_scan1)]
    mask_mtl_scan1 = masking.intersect_masks(mask_mtl_scan1, connected=False, threshold=0)

    # load wholebrain mask:
    mask_wholebrain_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_whole_brain_scan1)]
    mask_wholebrain_scan1 = masking.intersect_masks(mask_wholebrain_scan1, connected=False)

    mask_ventricles_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_ventricles_brain_scan1)]
    mask_ventricles_scan1 = masking.intersect_masks(mask_ventricles_scan1, connected=False, threshold=0)

    # load somatosensory mask
    mask_somato_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_somato_task_scan1)]
    mask_somato_scan1 = masking.intersect_masks(mask_somato_scan1, connected=False, threshold=0)

    # load auditory mask
    mask_aud_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_aud_task_scan1)]
    mask_aud_scan1 = masking.intersect_masks(mask_aud_scan1, connected=False, threshold=0)

    # load parietal_precuneus mask
    mask_parietal_precuneus_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_parietal_precuneus_task_scan1)]
    mask_parietal_precuneus_scan1 = masking.intersect_masks(mask_parietal_precuneus_scan1, connected=False, threshold=0)

    # load parietal mask
    mask_parietal_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_parietal_task_scan1)]
    mask_parietal_scan1 = masking.intersect_masks(mask_parietal_scan1, connected=False, threshold=0)

    # load precuneus mask
    mask_precuneus_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_precuneus_task_scan1)]
    mask_precuneus_scan1 = masking.intersect_masks(mask_precuneus_scan1, connected=False, threshold=0)

    # load medial OFC mask
    mask_medial_ofc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_medial_ofc_task_scan1)]
    mask_medial_ofc_scan1 = masking.intersect_masks(mask_medial_ofc_scan1, connected=False, threshold=0)

    # load rostral ACC mask
    mask_rostral_acc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_rostral_acc_task_scan1)]
    mask_rostral_acc_scan1 = masking.intersect_masks(mask_rostral_acc_scan1, connected=False, threshold=0)

    # load caudal ACC mask
    mask_caudal_acc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_caudal_acc_task_scan1)]
    mask_caudal_acc_scan1 = masking.intersect_masks(mask_caudal_acc_scan1, connected=False, threshold=0)

    # load superior frontal mask
    mask_superior_frontal_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_superior_frontal_task_scan1)]
    mask_superior_frontal_scan1 = masking.intersect_masks(mask_superior_frontal_scan1, connected=False, threshold=0)

    # load RSC mask
    mask_rsc_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_rsc_task_scan1)]
    mask_rsc_scan1 = masking.intersect_masks(mask_rsc_scan1, connected=False, threshold=0)

    # load posterior cingulate mask
    mask_posterior_cingulate_scan1 = [image.load_img(i) for i in copy.deepcopy(path_mask_posterior_cingulate_task_scan1)]
    mask_posterior_cingulate_scan1 = masking.intersect_masks(mask_posterior_cingulate_scan1, connected=False, threshold=0)

    # define mask dictionary
    masks_scan1 = {'lpfc': mask_lpfc_scan1, 'vis': mask_vis_scan1, 'cortical_memory': mask_cortical_memory_scan1, 'hpc': mask_hpc_scan1, 'mtl': mask_mtl_scan1,'wholebrain': mask_wholebrain_scan1, 'erh': mask_erh_scan1, 'ofc': mask_ofc_scan1, 'aud': mask_aud_scan1, 'somato': mask_somato_scan1, 'ventricles': mask_ventricles_scan1, 'parietal_precuneus': mask_parietal_precuneus_scan1, 'parietal': mask_parietal_scan1, 'precuneus': mask_precuneus_scan1, 'medial_ofc': mask_medial_ofc_scan1, 'rostral_acc': mask_rostral_acc_scan1, 'caudal_acc': mask_caudal_acc_scan1, 'superior_frontal': mask_superior_frontal_scan1, 'rsc': mask_rsc_scan1, 'posterior_cingulate': mask_posterior_cingulate_scan1}

    # load functional data localizeriation — keyed by BIDS run number
    data_task_scan1 = {rn: image.load_img(func_by_run[rn]) for rn in available_runs}
    # mask_task_scan1 = [image.load_img(i) for i in path_func_mask]
    
    # # apply brain mask to functional data
    # for run in range(len(data_task_scan1)):
    #     masker = NiftiMasker(mask_img=mask_task_scan1[run])
    #     data_task_scan1[run] = masker.fit_transform(data_task_scan1[run])
    #     data_task_scan1[run] = masker.inverse_transform(data_task_scan1[run])

    # get confound values — keyed by BIDS run number
    confs_scan1 = {rn: pd.read_csv(confs_by_run[rn], sep='\t') for rn in available_runs}

    # set confound variables of interest
    conf_vars_localizer = ['global_signal', 'framewise_displacement',
                    'trans_x', 'trans_y', 'trans_z',
                    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                    'rot_x', 'rot_y', 'rot_z',
                    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                    'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

    '''
    ========================================================================
    Implement voxelbased reliability
    confound regression über alle runs? (https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_Statistics/SPM_06_Stats_Running_1stLevel_Analysis.html)
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
    scene_betas = []   # track scene run betas separately
    object_betas = []  # track object run betas separately
    runs = df_behavior['run_nr'].unique()
    runs = runs[~pd.isna(runs)].astype(int)

    stimulus_categories_scenes = ['canyons01.jpg', 'road01.jpg', 'forest03.jpg', 'desert03.jpg',
    'mountain02.jpg', 'winter01.jpg', 'lake01.jpg', 'beach02.jpg',
    'waterfall04.jpg', 'autumn03.jpg']

    stimulus_categories_objects = ['vehicle', 'furniture', 'fruit', 'clothing', 'kitchen', 'sports', 'music',
    'animal', 'tool', 'body']


    # Loop through each run (only runs with matching confound + BOLD files)
    for run in runs:
        run = int(run)
        if run not in confs_scan1 or run not in data_task_scan1:
            print(f"[WARN] Run {run}: no matching confound or BOLD file on disk, skipping.")
            continue
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

        # Validate that all expected stimulus categories are present in this run
        run_categories = set(df_run['encoding_category'].unique())
        missing_categories = set(stimulus_categories) - run_categories
        if missing_categories:
            print(f"[WARN] Run {run}: missing {len(missing_categories)} categories: {missing_categories}. "
                  f"Only found {len(run_categories)}/{len(stimulus_categories)}. Skipping this run.")
            continue

        # Extract nuisance regressors for this run and replace NaNs with 0
        nuisance_regressors = confs_scan1[run][conf_vars_localizer].fillna(0)
        nuisance_regressors = nuisance_regressors.values

        # Get func data (keyed by BIDS run number)
        fmri_run_img_correlation_analysis = data_task_scan1[run]

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
        mean_image_run = mean_img(data_task_scan1[run])
        
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
        if stimulus_categories == stimulus_categories_scenes:
            scene_betas.append(betas_current_run_regressors_of_interest)
        else:
            object_betas.append(betas_current_run_regressors_of_interest)

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
        mean_image_run = mean_img(data_task_scan1[run])
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
            

    # =============================================================================
    # Helper: compute even/odd split-half correlation for a list of beta arrays
    # =============================================================================
    def _even_odd_correlation(betas_list, vol_shape, label):
        """Split betas_list into even-indexed and odd-indexed runs,
        average betas within each half, then compute voxel-wise Pearson r."""
        even_betas = [betas_list[i] for i in range(0, len(betas_list), 2)]
        odd_betas  = [betas_list[i] for i in range(1, len(betas_list), 2)]

        n_even = len(even_betas)
        n_odd  = len(odd_betas)
        print(f"[INFO] {label}: {len(betas_list)} runs -> {n_even} even, {n_odd} odd")

        if n_even == 0 or n_odd == 0:
            print(f"[WARN] {label}: not enough runs for even/odd split, filling with NaN")
            result = np.empty(vol_shape, dtype=object)
            for i in range(vol_shape[0]):
                for j in range(vol_shape[1]):
                    for k in range(vol_shape[2]):
                        result[i, j, k] = np.nan
            return result

        correlation_results_even = np.empty(vol_shape, dtype=object)
        correlation_results_odd  = np.empty(vol_shape, dtype=object)

        for i in range(vol_shape[0]):
            for j in range(vol_shape[1]):
                for k in range(vol_shape[2]):
                    # average betas across even runs
                    even_vals = [b[i, j, k] for b in even_betas]
                    correlation_results_even[i, j, k] = [
                        sum(vals) / n_even for vals in zip(*even_vals)
                    ]
                    # average betas across odd runs
                    odd_vals = [b[i, j, k] for b in odd_betas]
                    correlation_results_odd[i, j, k] = [
                        sum(vals) / n_odd for vals in zip(*odd_vals)
                    ]

        correlation_result = np.empty(vol_shape, dtype=object)
        for i in tqdm(range(vol_shape[0])):
            for j in range(vol_shape[1]):
                for k in range(vol_shape[2]):
                    even_data = np.array(correlation_results_even[i, j, k]).flatten()
                    odd_data  = np.array(correlation_results_odd[i, j, k]).flatten()
                    correlation, _ = pearsonr(even_data, odd_data)
                    correlation_result[i, j, k] = correlation
        return correlation_result

    vol_shape = fmri_run_img_correlation_analysis.shape[0:3]

    # =============================================================================
    # average betas odd and even runs scene encoding
    # =============================================================================
    correlation_result_scene = _even_odd_correlation(scene_betas, vol_shape, "scene")

    # =============================================================================
    # average betas odd and even runs object encoding
    # =============================================================================
    correlation_result_obj = _even_odd_correlation(object_betas, vol_shape, "object")

    correlation_result = np.empty(fmri_run_img_correlation_analysis.shape[0:3], dtype=object)

    # average the two correlation results (scene and object)
    for i in range(correlation_result_obj.shape[0]):
        for j in range(correlation_result_obj.shape[1]):
            for k in range(correlation_result_obj.shape[2]):
                correlation_result[i, j, k] = (correlation_result_scene[i, j, k] + correlation_result_obj[i, j, k]) / 2.0

    # =============================================================================
    # plot correlation values     
    # =============================================================================
    import nibabel as nib
    from nilearn.image import new_img_like
    import os

    # Concatenate all the runs along the time axis and compute the mean image
    all_runs_concatenated = concat_imgs(list(data_task_scan1.values()))
    mean_image = mean_img(all_runs_concatenated)

    # Ensure the correlation arrays are numeric float arrays
    correlation_result = np.asarray(correlation_result).astype(float)
    correlation_result_scene = np.asarray(correlation_result_scene).astype(float)
    correlation_result_obj = np.asarray(correlation_result_obj).astype(float)

    # Build nifti images for each correlation map using the mean image header/affine
    correlation_img_tmp = mean_image.__class__(correlation_result, affine=mean_image.affine, header=mean_image.header.copy())
    correlation_img_scene_tmp = mean_image.__class__(correlation_result_scene, affine=mean_image.affine, header=mean_image.header.copy())
    correlation_img_obj_tmp = mean_image.__class__(correlation_result_obj, affine=mean_image.affine, header=mean_image.header.copy())

    # Create output directory and save the three nifti files (scene, obj, combined)
    save_path = create_script_named_folder(opj('voxel_reliability_masks_visualization', sub, ses))
    nifti_fname = os.path.join(save_path, f"{sub}_corrmap_localizer_{behavior_ses}.nii.gz")
    nifti_scene_fname = os.path.join(save_path, f"{sub}_corrmap_scene_localizer_{behavior_ses}.nii.gz")
    nifti_obj_fname = os.path.join(save_path, f"{sub}_corrmap_obj_localizer_{behavior_ses}.nii.gz")

    nib.save(correlation_img_tmp, nifti_fname)
    nib.save(correlation_img_scene_tmp, nifti_scene_fname)
    nib.save(correlation_img_obj_tmp, nifti_obj_fname)

    # Filenames for saved visualizations
    mean_all = np.nanmean(correlation_result)
    std_all = np.nanstd(correlation_result)
    mean_scene = np.nanmean(correlation_result_scene)
    std_scene = np.nanstd(correlation_result_scene)
    mean_obj = np.nanmean(correlation_result_obj)
    std_obj = np.nanstd(correlation_result_obj)

    plot_fname_all = os.path.join(save_path, f"{sub}_mask_localizer_{behavior_ses}_voxel_reliability_mean_corr_{mean_all:.4f}.png")
    plot_fname_scene = os.path.join(save_path, f"{sub}_mask_localizer_{behavior_ses}_voxel_reliability_scene_mean_corr_{mean_scene:.4f}.png")
    plot_fname_obj = os.path.join(save_path, f"{sub}_mask_localizer_{behavior_ses}_voxel_reliability_obj_mean_corr_{mean_obj:.4f}.png")

    # Plot the combined map (already present) and save
    plot_stat_map(
        correlation_img_tmp,
        bg_img=mean_image,
        title=f'Combined Voxel-wise Correlation (odd vs even)\nmean:{mean_all:.4f} std:{std_all:.4f}',
        threshold=0.15,
        display_mode='ortho',
        cut_coords=(0, 0, 5),
        colorbar=True
    )
    plt.savefig(plot_fname_all, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the scene-only correlation map and save
    plot_stat_map(
        correlation_img_scene_tmp,
        bg_img=mean_image,
        title=f'Scene Voxel-wise Correlation (odd vs even)\nmean:{mean_scene:.4f} std:{std_scene:.4f}',
        threshold=0.15,
        display_mode='ortho',
        cut_coords=(0, 0, 5),
        colorbar=True
    )
    plt.savefig(plot_fname_scene, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the object-only correlation map and save
    plot_stat_map(
        correlation_img_obj_tmp,
        bg_img=mean_image,
        title=f'Object Voxel-wise Correlation (odd vs even)\nmean:{mean_obj:.4f} std:{std_obj:.4f}',
        threshold=0.15,
        display_mode='ortho',
        cut_coords=(0, 0, 5),
        colorbar=True
    )
    plt.savefig(plot_fname_obj, dpi=300, bbox_inches='tight')
    plt.close()
                
    # =============================================================================
    # apply the correlation values as additional mask 
    # =============================================================================
    masks_scan1_no_corr_mask = {'lpfc': mask_lpfc_scan1, 'vis': mask_vis_scan1, 'cortical_memory': mask_cortical_memory_scan1, 'hpc': mask_hpc_scan1, 'mtl': mask_mtl_scan1,'wholebrain': mask_wholebrain_scan1, 'erh': mask_erh_scan1, 'ofc': mask_ofc_scan1, 'aud': mask_aud_scan1, 'somato': mask_somato_scan1, 'ventricles': mask_ventricles_scan1,
                                'parietal_precuneus': mask_parietal_precuneus_scan1, 'parietal': mask_parietal_scan1, 'precuneus': mask_precuneus_scan1, 'medial_ofc': mask_medial_ofc_scan1, 'rostral_acc': mask_rostral_acc_scan1, 'caudal_acc': mask_caudal_acc_scan1, 'superior_frontal': mask_superior_frontal_scan1, 'rsc': mask_rsc_scan1, 'posterior_cingulate': mask_posterior_cingulate_scan1}

    # # Define a threshold for the correlation image (adjust as needed)
    # correlation_threshold = 0  # Example threshold

    # # Threshold the correlation image to create a binary mask
    # binary_correlation_mask = image.math_img("img > {}".format(correlation_threshold), img=correlation_img)
    # plotting.plot_epi(binary_correlation_mask)
        
    # # Apply the combined mask to each ROI
    # mask_vis_scan1_correlation_masked = combine_masks(mask_vis_scan1, binary_correlation_mask)
    # mask_hpc_scan1_correlation_masked = combine_masks(mask_hpc_scan1, binary_correlation_mask)
    # mask_erh_scan1_correlation_masked = combine_masks(mask_erh_scan1, binary_correlation_mask)
    # mask_lpfc_scan1_correlation_masked = combine_masks(mask_lpfc_scan1, binary_correlation_mask)
    # mask_ofc_scan1_correlation_masked = combine_masks(mask_ofc_scan1, binary_correlation_mask)
    # mask_mtl_scan1_correlation_masked = combine_masks(mask_mtl_scan1, binary_correlation_mask)
    # mask_wholebrain_scan1_correlation_masked = combine_masks(mask_wholebrain_scan1, binary_correlation_mask)

    # masks_scan1 = {'lpfc': mask_lpfc_scan1_correlation_masked, 'vis': mask_vis_scan1_correlation_masked, 'hpc': mask_hpc_scan1_correlation_masked, 'mtl': mask_mtl_scan1_correlation_masked,'wholebrain': mask_wholebrain_scan1_correlation_masked, 'erh': mask_erh_scan1_correlation_masked, 'ofc': mask_ofc_scan1_correlation_masked}

    # for key, mask in masks_scan1.items():
    #     plotting.plot_epi(masks_scan1_no_corr_mask[key], title=f"{key} fMRI Image before correlation exclusion")
    #     plotting.plot_epi(masks_scan1[key], title=f"{key} fMRI Image after correlation exclusion")
    #     plt.show()  # Display each plot 

    # loop over masks and over thresholds and save them as binary niftis
    for key, mask in masks_scan1_no_corr_mask.items():

        for percentage in [10, 30, 50, 70, 90]:
            
            # apply mask to correlation image. Now only within mask
            masker= NiftiMasker(mask, target_affine= mask.affine, target_shape=mask.shape)
            masker.fit()
            masked_correlation_img= masker.inverse_transform(masker.transform_single_imgs(correlation_img_tmp)) 

            # calculate the threshold to only keep the top percentage% of voxels - 70% --> most reliable 70% the smaller teh numbe rteh fewer voxels kept
            data_array = masked_correlation_img.get_fdata()
            non_zero_values = data_array[data_array != 0].flatten()
            threshold_index = int(len(non_zero_values) * (percentage / 100))
            sorted_values = np.sort(non_zero_values)[::-1]

            # Make sure there are enough non-zero elements
            if threshold_index < len(sorted_values):
                correlation_threshold = sorted_values[threshold_index]
            else:

                raise ValueError(f"Only {len(sorted_values)} non-zero elements found was looking for {threshold_index}th element.")
                
            # if correlation_threshold < 0 set to 0
            if correlation_threshold < 0:
                correlation_threshold = 0

            # Now you can use correlation_threshold in your masking as needed
            binary_correlation_mask = image.math_img(f"img > {correlation_threshold}", img=masked_correlation_img)

            # save mask with mask name, threshold and sub
            filename = f"{path_masks_voxel_reliability}/{sub}_mask_{key}_threshold_{percentage}_localizer_{behavior_ses}.nii"

            # Save the binary mask as a NIfTI file
            binary_correlation_mask.to_filename(filename)
            
            # plot and save the mask overlayed on mean image
            plot_fname_mask = os.path.join(save_path, f"{sub}_mask_{key}_threshold_{percentage}_localizer_{behavior_ses}_voxel_reliability_overlay.png")    
            plot_stat_map(
                binary_correlation_mask,
                bg_img=mean_image,
                title=f'{key} mask with {percentage}% voxel reliability threshold\ncorrelation threshold:{correlation_threshold:.4f}',
                threshold=0.5,
                display_mode='ortho',
                cut_coords=(0, 0, 5),
                colorbar=True   
            )
            #plt.savefig(plot_fname_mask, dpi=300, bbox_inches='tight')
            plt.close()

        masks_scan1[key] = binary_correlation_mask
        
                
                    
    # # # =============================================================================
    # # # Plot mean image with masks overlayed    
    # # # =============================================================================
            
    # for key in masks_scan1.keys():
    #     mean_img_data = mean_image.get_fdata()
    #     mask_data = masks_scan1[key].get_fdata()
    #     mask_data_no_corr = masks_scan1_no_corr_mask[key].get_fdata()

    #     # Create a figure with 3 subplots for different slices (axial, coronal, sagittal)
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #     # Select slices to display
    #     axial_slice = mean_img_data[:, :, mean_img_data.shape[2] // 2]
    #     coronal_slice = mean_img_data[:, mean_img_data.shape[1] // 2, :]
    #     sagittal_slice = mean_img_data[mean_img_data.shape[0] // 2, :, :]

    #     # Select corresponding slices from the masks
    #     axial_mask = mask_data[:, :, mask_data.shape[2] // 2]
    #     coronal_mask = mask_data[:, mask_data.shape[1] // 2, :]
    #     sagittal_mask = mask_data[mask_data.shape[0] // 2, :, :]

    #     axial_mask_no_corr = mask_data_no_corr[:, :, mask_data_no_corr.shape[2] // 2]
    #     coronal_mask_no_corr = mask_data_no_corr[:, mask_data_no_corr.shape[1] // 2, :]
    #     sagittal_mask_no_corr = mask_data_no_corr[mask_data_no_corr.shape[0] // 2, :, :]

    #     # Plot axial slice with both masks
    #     axes[0].imshow(np.rot90(axial_slice), cmap='gray')
    #     axes[0].imshow(np.rot90(axial_mask), cmap='Reds', alpha=0.8)
    #     axes[0].imshow(np.rot90(axial_mask_no_corr), cmap='Blues', alpha=0.2)
    #     axes[0].set_title(f'Axial Slice - {key} mask')

    #     # Plot coronal slice with both masks
    #     axes[1].imshow(np.rot90(coronal_slice), cmap='gray')
    #     axes[1].imshow(np.rot90(coronal_mask), cmap='Reds', alpha=0.8)
    #     axes[1].imshow(np.rot90(coronal_mask_no_corr), cmap='Blues', alpha=0.2)
    #     axes[1].set_title('Coronal Slice')

    #     # Plot sagittal slice with both masks
    #     axes[2].imshow(np.rot90(sagittal_slice), cmap='gray')
    #     axes[2].imshow(np.rot90(sagittal_mask), cmap='Reds', alpha=0.8)
    #     axes[2].imshow(np.rot90(sagittal_mask_no_corr), cmap='Blues', alpha=0.2)
    #     axes[2].set_title('Sagittal Slice')

    #     # Hide axis labels
    #     for ax in axes:
    #         ax.axis('off')
            

    #     plt.show()
                            

# %%
