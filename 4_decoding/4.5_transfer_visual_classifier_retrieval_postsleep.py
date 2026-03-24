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
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from groo.groo import get_root
from nilearn.maskers import NiftiMasker
from joblib import dump, load
from tqdm import tqdm
import re
import json

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


def build_sequence_dict_from_df_behavior_enc(
    df: pd.DataFrame,
    cue_col: str = "dn_cue",
    scene_col: str = "sc_enc_stim",
    obj_col: str = "obj_enc_stim",
    expected_n_obj: int = 5,
) -> dict:
    """
    Returns:
      {
        'day':   { 'autumn03.jpg': ['hand.png', ... x5], ... },
        'night': { ... }
      }
    Scenes kept in first-appearance order. Objects kept once (first 5 unique, in appearance order).
    """
    missing = [c for c in (cue_col, scene_col, obj_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"df_behavior_enc missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = {"day": {}, "night": {}}

    for cue, scene, obj in df[[cue_col, scene_col, obj_col]].itertuples(index=False, name=None):
        if pd.isna(cue) or pd.isna(scene):
            continue

        cue_s = str(cue).lower()
        if "day" in cue_s:
            dn = "day"
        elif "night" in cue_s:
            dn = "night"
        else:
            continue

        scene_key = str(scene)  # keep exactly as in sc_enc_stim

        if scene_key not in out[dn]:
            out[dn][scene_key] = []

        if pd.notna(obj):
            obj_s = str(obj)
            # keep only first 5 unique objects in appearance order
            if len(out[dn][scene_key]) < expected_n_obj and obj_s not in out[dn][scene_key]:
                out[dn][scene_key].append(obj_s)

    # Optional: verify we got exactly 5 per scene
    for dn in ("day", "night"):
        for sc, objs in out[dn].items():
            if len(objs) != expected_n_obj:
                print(f"WARNING: {dn} / {sc} has {len(objs)} objects (expected {expected_n_obj}).")

    return out


def map_sequence_objects_to_categories(
    seq_dict_obj: dict,
    obj_enc_to_cat: dict,
    keep_unmapped_as: str = "UNMAPPED",
    normalize_basename: bool = True,
) -> dict:
    """
    Maps nested object stimulus labels to object categories while preserving ordering.

    Args:
        seq_dict_obj: {'day': {scene: [obj_stim,...]}, 'night': {...}}
        obj_enc_to_cat: mapping from obj_enc_stim -> obj_cat (from localizer)
        keep_unmapped_as: value to use if an object cannot be mapped
        normalize_basename: if True, try mapping with os.path.basename(obj_stim) too

    Returns:
        seq_dict_cat with same structure but objects replaced by categories.
    """
    seq_dict_cat = {"day": {}, "night": {}}
    unmapped = []

    for dn, scenes in seq_dict_obj.items():
        if dn not in seq_dict_cat:
            seq_dict_cat[dn] = {}
        for scene, obj_list in scenes.items():
            mapped = []
            for obj_stim in obj_list:
                key = obj_stim
                cat = obj_enc_to_cat.get(key, None)

                if cat is None and normalize_basename:
                    cat = obj_enc_to_cat.get(os.path.basename(str(key)), None)

                if cat is None:
                    unmapped.append(key)
                    cat = keep_unmapped_as

                mapped.append(cat)

            seq_dict_cat[dn][scene] = mapped

    if unmapped:
        print(
            f"WARNING: {len(unmapped)} objects could not be mapped to categories "
            f"(showing up to 10): {unmapped[:10]}"
        )

    return seq_dict_cat

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
# Subject handling (mirrors 4.4):
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
        path_clf_dir = '/Volumes/Untitled/code/4_decoding/4.1_train_visual_classifier/classifiers'
        print(f"Setting classifier path to local: {path_clf_dir}")
        print(f"Setting masks path to local: {path_masks}")
else:
    print(
        f"WARNING: No subject specified on command line. "
        f"Using default subject {default_subj}."
    )
    sub = default_subj

# ROI handling: accept second argument for mask/ROI label
#   - Default: 'vis' (visual cortex)
#   - Alternatives: 'cortical_memory', 'hpc', etc.

if len(sys.argv) > 1:
    try:
        roi_arg = sys.argv[2]
    except Exception:
        roi_arg = "vis"
        print('CAVEAT: No ROI argument provided. Defaulting to "vis".')
    if re.match(r'^[a-z_]+$', roi_arg):  # Basic validation
        cmask = roi_arg
        print(f"Using ROI: {cmask}")
    else:
        print(
            f"WARNING: Unexpected ROI format '{roi_arg}'. "
        )
else:
    print(
        f"WARNING: No ROI specified on command line. "
        f"Using default ROI {default_roi}."
    )
    default_roi = "cortical_memory"

    cmask = default_roi

mri_sessions_enc = 'ses-06' # this is the last retrieval post sleep
use_overall_classifier1_specific2 = 2  # 1=overall, 2=specific


'''
========================================================================
LOAD BEHAVIORAL DATA FOR THE ENCODING session
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
required_enc_label_types = {"sc_enc_fix"}

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

# ---------------------------------------------------------------------
# Build the subject-specific sequences from ENCODING df
# (use df_behavior_enc, ideally filtered to sc_enc_fix)
# ---------------------------------------------------------------------
sequence_dict_obj = build_sequence_dict_from_df_behavior_enc(df_behavior_enc)  # object-stimulus version
sequence_dict_cat = map_sequence_objects_to_categories(sequence_dict_obj, obj_enc_to_cat)  # category version

if len(sequence_dict_obj['day']) == 0 or len(sequence_dict_obj['night']) == 0:
    raise ValueError("Could not build valid sequence_dict from encoding behavior data (step0).")

# Optional sanity check (prints scene order and object counts)
for dn in ("day", "night"):
    print(f"[sequence_dict] {dn}: {len(sequence_dict_cat[dn])} scenes")
    for sc, objs in sequence_dict_cat[dn].items():
        print(f"  {sc}: {len(objs)} objects -> {objs}")

# ---------------------------------------------------------------------
# Prepare RETRIEVAL df
# ---------------------------------------------------------------------
if "obj_ret_targ" not in df_behavior_retrieval.columns:
    raise KeyError(
        "df_behavior_retrieval missing required column 'obj_ret_targ'. "
        f"Available columns: {df_behavior_retrieval.columns.tolist()}"
    )

# insert next to obj_enc_stim into obj_cat based on loc1 mapping
df_behavior_retrieval['obj_cat'] = df_behavior_retrieval['obj_ret_targ'].map(obj_enc_to_cat)

# forward fill trl_nr and run_nr columns to handle NaNs
if "trl_nr" in df_behavior_retrieval.columns:
    df_behavior_retrieval['trl_nr'] = df_behavior_retrieval['trl_nr'].ffill()
if "run_nr" in df_behavior_retrieval.columns:
    df_behavior_retrieval['run_nr'] = df_behavior_retrieval['run_nr'].ffill()

'''
========================================================================
LOAD encoding func data
========================================================================
'''

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

# Expected: 9 runs total (3 short ~120 TPs + 6 longer retrieval runs)
# The 6 longest runs are selected as retrieval runs
'''
========================================================================
LOAD confounds
========================================================================
'''

# load the confounds files
path_confs_task_scan1 = opj(
    path_fmriprep,
    sub_mri,
    mri_sessions_enc,
    'func',
    f'{sub_mri}_{mri_sessions_enc}_task-mem*_run*_desc-confounds_timeseries.tsv'
)  # e.g., sub-13_ses-06_task-mem_run-01_desc-confounds_timeseries.tsv
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
Mask the data
========================================================================
'''
try:
    path_mask = f'{sub}_mask_{cmask}_threshold_{reliability_threshold}_combined.nii'
    mask_scan1 = image.load_img(opj(path_masks, path_mask))
except:

    path_mask = f'{sub_mri}_mask_{cmask}_threshold_{reliability_threshold}_combined.nii'
    mask_scan1 = image.load_img(opj(path_masks, path_mask))


data_masked_scan1_localizer=[]
data_masked_scan1_localizer = [masking.apply_mask(i, mask_scan1, smoothing_fwhm=csmooth) for i in data_task_scan_loc1]
data_masked_detrend_scan1 = [signal.clean(signals=data_masked_scan1_localizer[cr],
                                    t_r=TR, detrend=True,
                                    standardize='zscore_sample',
                                    confounds=confs_scan_loc1[cr][conf_vars_localizer].fillna(0))
                                    for cr in range(len(data_masked_scan1_localizer))]

# Select the 6 longest runs as retrieval runs (postsleep has 6 retrieval runs across 2 rounds)
n_total_runs = len(data_masked_detrend_scan1)
if n_total_runs != 9:
    print(f"WARNING: Expected 9 total runs (3 short + 6 retrieval), but got {n_total_runs}.")

run_lengths = [(i, run.shape[0]) for i, run in enumerate(data_masked_detrend_scan1)]
run_lengths_sorted = sorted(run_lengths, key=lambda x: x[1], reverse=True)
top6_indices = sorted([idx for idx, _ in run_lengths_sorted[:6]])  # keep original order

print(f"All run lengths: {[(i, l) for i, l in run_lengths]}")
print(f"Selected 6 longest runs (indices): {top6_indices} with lengths {[data_masked_detrend_scan1[i].shape[0] for i in top6_indices]}")

if len(data_masked_detrend_scan1) < 6:
    raise ValueError(f"Expected at least 6 retrieval runs, but only got {len(data_masked_detrend_scan1)} total runs.")

retrieval_runs = [data_masked_detrend_scan1[idx] for idx in top6_indices]

for i, idx in enumerate(top6_indices):
    print(f'Retrieval run {i} (original idx {idx}) shape: {data_masked_detrend_scan1[idx].shape}')

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

# Note: cmask is already defined in the ROI handling section above
# Construct path to classifier (mirrors 4.1; uses cmask which was configured above)

if use_overall_classifier1_specific2 == 1:
    # Load Generic Classifier (Both)
    filename_clf = opj(path_clf_dir, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_3.joblib")

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
    filename_clf_scenes = opj(path_clf_dir, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_1.joblib")
    if os.path.exists(filename_clf_scenes):
        print(f"Loading SCENE classifier from: {filename_clf_scenes}")
        clf_scenes = load(filename_clf_scenes)
    else:
        raise FileNotFoundError(f"Scene classifier not found at: {filename_clf_scenes}")

    # 2. Object Classifier
    filename_clf_objects = opj(path_clf_dir, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing_level}_only_scenes1_objects2_both3_2.joblib")
    if os.path.exists(filename_clf_objects):
        print(f"Loading OBJECT classifier from: {filename_clf_objects}")
        clf_objects = load(filename_clf_objects)
    else:
        raise FileNotFoundError(f"Object classifier not found at: {filename_clf_objects}")

rows = {
    "subject": [],                          # subj ID
    "run": [],                              # run number
    "trial": [],                            # trial number
    "tr": [],                               # TR relative to onset
    "object_scene_retrieval": [],           # 'object' or 'scene'
    "retrieval_target": [],                 # target stimulus (object or scene to be retreived)
    "retrieval_target_precise": [],         # precise target stimulus (object or scene to be retreived with full filename)
    "retrieval_cue": [],                    # cue stimulus (scene shown during object retrieval; day/night shown during scene retrieval)
    "retrieval_cue_position": [],           # position of the target in the sequence (1-5)
    "stimulus_class_from_classifier": [],   # class label from the classifier
    "target_classifier_matched": [],        # whether the classifier's target class matches the retrieval target (1 the classifier class matches the target, 0 otherwise)
    "probability_from_classifier": [],      # probability assigned by the classifier
    "behav_correct": [],                    # whether the retrieval was behaviorally correct
    "full_structured_sequence_object": [],          # full structured sequence of objects
    "full_structured_sequence_category": [],          # full structured sequence of category

}

# Serialize once; later we store it only in the first output row.
sequence_dict_obj_json = json.dumps(sequence_dict_obj, ensure_ascii=False, sort_keys=True)
sequence_dict_cat_json = json.dumps(sequence_dict_cat, ensure_ascii=False, sort_keys=True)
sequence_struct_written = False


'''
========================================================================
Apply classifier to object retreival data
========================================================================
'''


# Get all scene and object retrieval trials
df_objects = df_behavior_retrieval[df_behavior_retrieval['label_type'] == 'sc_no4obj'].copy()
df_scenes = df_behavior_retrieval[df_behavior_retrieval['label_type'] == 'd_no4sc'].copy()


# Map retrieval runs to indices
retrieval_data_list = retrieval_runs  # 6 retrieval runs

# ---------------------------------------------------------------------
# IMPORTANT: behavioral run_nr values often refer to the *task run index*
# (e.g., 1-6), while retrieval_data_list is indexed 0..5.
# Build an explicit mapping from observed run numbers -> [0..5].
# This fixes the "SKIP bad run_nr -> data_idx out of range" issue.
# ---------------------------------------------------------------------
try:
    run_nrs_obj = pd.to_numeric(df_objects.get('run_nr', pd.Series([], dtype=float)), errors='coerce').dropna().astype(int)
    run_nrs_scn = pd.to_numeric(df_scenes.get('run_nr', pd.Series([], dtype=float)), errors='coerce').dropna().astype(int)
    observed_run_nrs = sorted(set(run_nrs_obj.tolist()) | set(run_nrs_scn.tolist()))
except Exception:
    observed_run_nrs = []

run_nr_to_data_idx = {}
if len(observed_run_nrs) >= 1:
    # Prefer mapping based on the actually observed run numbers
    for i, rn in enumerate(observed_run_nrs[: len(retrieval_data_list)]):
        run_nr_to_data_idx[int(rn)] = i
else:
    # Fallback to the old assumption (1->0, 2->1, ..., 6->5)
    for i in range(len(retrieval_data_list)):
        run_nr_to_data_idx[i + 1] = i

# Time window around onset (in TRs)
tr_window = np.arange(0, 10)

# Storage for decoding results
decoding_results_scene_presented = {}   # key: scene stimulus (sc_enc_stim)
decoding_results_object_retrieved = {}  # key: object category (obj_cat)

for idx, row in df_objects.iterrows():
    run_nr_val = pd.to_numeric(row.get('run_nr', np.nan), errors='coerce')
    if pd.isna(run_nr_val):
        continue
    run_nr = int(run_nr_val)

    # Map run number (1,2,3,...) to index in retrieval_data_list (0,1,2,...)
    data_idx = run_nr_to_data_idx.get(run_nr, None)
    if data_idx is None or data_idx < 0 or data_idx >= len(retrieval_data_list):
        continue

    fmri_data = retrieval_data_list[data_idx]

    try:
        onset_tr = int(np.round(row['TR_time_onset']))
    except Exception:
        continue

    # Extract window
    window_idxs = (onset_tr + tr_window).astype(int)
    n_tp = fmri_data.shape[0]
    valid = (window_idxs >= 0) & (window_idxs < n_tp)

    # If any frame in the window is out-of-bounds, skip this trial
    if not valid.all():
        continue

    # All frames available — extract window
    X_window = fmri_data[window_idxs, :]

    # Retrieval cue (scene shown during object retrieval); used later when saving rows
    target_scene = row.get('ret_cue', np.nan)

    # ---------------------------
    # Scene classifier: presented scene
    # ---------------------------
    if 'ret_cue' in df_objects.columns:
        target_scene = row['ret_cue']
        if target_scene in clf_scenes.classes_:
            class_idx_scene = np.where(clf_scenes.classes_ == target_scene)[0][0]
            probs_scene = clf_scenes.predict_proba(X_window)

            # # plot every class over time, true class in red, others in grey
            # fig, ax = plt.subplots(figsize=(12, 6))
            # for class_idx, class_label in enumerate(clf_scenes.classes_):
            #     if class_idx == class_idx_scene:
            #         ax.plot(tr_window, probs_scene[:, class_idx],
            #                 color='red', linewidth=2.5, alpha=1.0, zorder=3,
            #                 label=f"{class_label} (target)")
            #     else:
            #         ax.plot(tr_window, probs_scene[:, class_idx],
            #                 color='grey', linewidth=1.0, alpha=0.5, zorder=1)
            # ax.set_title(f'Scene Classifier Probabilities over Time for Presented Scene: {target_scene}')
            # ax.set_xlabel('TRs from Onset')
            # ax.set_ylabel('Probability')
            # ax.axvline(0, color='red', linestyle='--')
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.tight_layout()
            # plt.show()

            scene_tc = probs_scene[:, class_idx_scene]

            if target_scene not in decoding_results_scene_presented:
                decoding_results_scene_presented[target_scene] = []
            decoding_results_scene_presented[target_scene].append(scene_tc)

    # ---------------------------
    # Object classifier: retrieved object (obj_cat)
    # ---------------------------
    target_object = row['obj_cat']
    if target_object in clf_objects.classes_:
        class_idx_obj = np.where(clf_objects.classes_ == target_object)[0][0]
        probs_obj = clf_objects.predict_proba(X_window)
        obj_tc = probs_obj[:, class_idx_obj]

        if target_object not in decoding_results_object_retrieved:
            decoding_results_object_retrieved[target_object] = []
        decoding_results_object_retrieved[target_object].append(obj_tc)


        # Fill in rows for each TR and each class of the classifier
        for t_idx, tr in enumerate(tr_window):
            for class_idx, class_label in enumerate(clf_objects.classes_):

                rows["subject"].append(sub)
                rows["run"].append(run_nr)
                rows["trial"].append(row["trl_nr"])
                rows["tr"].append(tr)
                rows["object_scene_retrieval"].append("object")
                rows["retrieval_target"].append(target_object)
                rows["retrieval_target_precise"].append(row["obj_ret_targ"])
                rows["retrieval_cue"].append(target_scene)
                rows["retrieval_cue_position"].append(
                    int(row.get("seq_pos", np.nan)[1]) if pd.notna(row.get("seq_pos", np.nan)) else np.nan
                )
                rows["stimulus_class_from_classifier"].append(class_label)
                rows["target_classifier_matched"].append(1 if class_label == target_object else 0)
                rows["probability_from_classifier"].append(probs_obj[t_idx, class_idx])

                # get from df_behavior_retrieval and current trl_nr the entry in catch_trial
                df_behavior_retrieval_subset = df_behavior_retrieval[df_behavior_retrieval["trl_nr"] == row["trl_nr"]]
                vals = df_behavior_retrieval_subset["catch_trial"].dropna()
                catch_trial_pressed = vals.iloc[0] if not vals.empty else np.nan
                rows["behav_correct"].append(catch_trial_pressed)

                if not sequence_struct_written:
                    rows["full_structured_sequence_object"].append(sequence_dict_obj_json)
                    rows["full_structured_sequence_category"].append(sequence_dict_cat_json)
                    sequence_struct_written = True
                else:
                    rows["full_structured_sequence_object"].append(np.nan)
                    rows["full_structured_sequence_category"].append(np.nan)


# Save decoding outputs for later group-level analysis (mirrors 4.4)
decoding_results_path = create_script_named_folder('decoding_timecourses_data')
np.save(
    opj(
        decoding_results_path,
        f'{sub}_ROI_{cmask}_decoding_timecourses_retrievalpostsleep_scene_presented_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'
    ),
    decoding_results_scene_presented
)
np.save(
    opj(
        decoding_results_path,
        f'{sub}_ROI_{cmask}_decoding_timecourses_retrievalpostsleep_object_retrieved_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'
    ),
    decoding_results_object_retrieved
)
np.save(
    opj(
        decoding_results_path,
        f'{sub}_ROI_{cmask}_decoding_timecourses_retrievalpostsleep_tr_window.npy'
    ),
    tr_window
)


# ========================================================================
# Quick check: scene decoding (is correct class peaking?)
# is the correct presented scene class peaking (this is the cue during object retrieval and visually presneted)
# ========================================================================
plt_path = create_script_named_folder('decoding_timecourses_plots')

plt.figure(figsize=(10, 6))

all_timecourses_scene = []

for stim, timecourses in decoding_results_scene_presented.items():
    stim_avg = np.mean(timecourses, axis=0)
    all_timecourses_scene.extend(timecourses)

    # one line per scene class (average over its trials)
    plt.plot(tr_window, stim_avg, alpha=0.7, linewidth=2, label=stim)

# Grand average across all scene timecourses
if all_timecourses_scene:
    grand_avg_scene = np.mean(all_timecourses_scene, axis=0)
    plt.plot(tr_window, grand_avg_scene, color='black', linewidth=4, label='Grand average')

plt.axvline(0, color='red', linestyle='--')
plt.xlabel('TRs from onset + 5 secs HRF lag')
plt.ylabel('P(correct scene class)')
plt.title(f'Scene decoding during object retrieval (subject {sub})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(opj(plt_path, f'ROI_{cmask}_scene_decoding_timecourses_subject_{sub}.png'))
plt.show()

# ========================================================================
# Quick check: object decoding (is correct class peaking?)
# this is actual retreival of the object category
# ========================================================================

plt.figure(figsize=(10, 6))
all_timecourses_object = []
for obj_cat, timecourses in decoding_results_object_retrieved.items():
    obj_avg = np.mean(timecourses, axis=0)
    all_timecourses_object.extend(timecourses)

    # one line per object category (average over its trials)
    plt.plot(tr_window, obj_avg, alpha=0.7, linewidth=2, label=obj_cat)
# Grand average across all object timecourses
if all_timecourses_object:
    grand_avg_object = np.mean(all_timecourses_object, axis=0)
    plt.plot(tr_window, grand_avg_object, color='black', linewidth=4, label='Grand average')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('TRs from onset + 5 secs HRF lag')
plt.ylabel('P(correct object class)')
plt.title(f'Object decoding during object retrieval (subject {sub})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(opj(plt_path, f'ROI_{cmask}_object_decoding_timecourses_subject_{sub}.png'))
plt.show()

'''
========================================================================
Apply classifier to scene retreival data
========================================================================
'''

# ============================================================================
# Scene retrieval analysis: decode retrieved scene (sc_ret_targ) on d_no4sc trials
# ============================================================================


decoding_results_scene_retrieved = {}  # key: retrieved scene (sc_ret_targ), value: list of timecourses

for idx, row in df_scenes.iterrows():
    run_nr_val = pd.to_numeric(row.get('run_nr', np.nan), errors='coerce')
    if pd.isna(run_nr_val):
        continue
    run_nr = int(run_nr_val)

    # Map run number (1,2,3,...) to index in retrieval_data_list (0,1,2,...)
    data_idx = run_nr_to_data_idx.get(run_nr, None)
    if data_idx is None or data_idx < 0 or data_idx >= len(retrieval_data_list):
        continue

    fmri_data = retrieval_data_list[data_idx]

    # Prefer TR_time_onset_w_HRF if available, otherwise compute from onset
    onset_tr = int(np.round(row['TR_time_onset']))


    # Extract window
    window_idxs = (onset_tr + tr_window).astype(int)
    n_tp = fmri_data.shape[0]
    valid = (window_idxs >= 0) & (window_idxs < n_tp)

    # If any frame in the window is out-of-bounds, skip this trial
    if not valid.all():
        continue

    # All frames available — extract window
    X_window = fmri_data[window_idxs, :]

    # Retrieved scene label
    target_scene_ret = row['sc_ret_targ']

    if target_scene_ret in clf_scenes.classes_:
        class_idx = np.where(clf_scenes.classes_ == target_scene_ret)[0][0]
        probs = clf_scenes.predict_proba(X_window)
        scene_tc = probs[:, class_idx]

        if target_scene_ret not in decoding_results_scene_retrieved:
            decoding_results_scene_retrieved[target_scene_ret] = []
        decoding_results_scene_retrieved[target_scene_ret].append(scene_tc)

        # Fill in rows for each TR and each class of the classifier
        for t_idx, tr in enumerate(tr_window):
            for class_idx, class_label in enumerate(clf_scenes.classes_):

                rows["subject"].append(sub)
                rows["run"].append(run_nr)
                rows["trial"].append(row["trl_nr"])
                rows["tr"].append(tr)
                rows["object_scene_retrieval"].append("scene")
                rows["retrieval_target"].append(target_scene_ret)
                rows["retrieval_target_precise"].append(target_scene_ret)
                rows["retrieval_cue"].append(row.get("ret_cue", np.nan))
                rows["retrieval_cue_position"].append(
                    int(row.get("seq_pos", np.nan)[1]) if pd.notna(row.get("seq_pos", np.nan)) else np.nan
                )
                rows["stimulus_class_from_classifier"].append(class_label)
                rows["target_classifier_matched"].append(1 if class_label == target_scene_ret else 0)
                rows["probability_from_classifier"].append(probs[t_idx, class_idx])

                # get from df_behavior_retrieval and current trl_nr the entry in catch_trial
                df_behavior_retrieval_subset = df_behavior_retrieval[df_behavior_retrieval["trl_nr"] == row["trl_nr"]]
                vals = df_behavior_retrieval_subset["catch_trial"].dropna()
                catch_trial_pressed = vals.iloc[0] if not vals.empty else np.nan
                rows["behav_correct"].append(catch_trial_pressed)

                if not sequence_struct_written:
                    rows["full_structured_sequence_object"].append(sequence_dict_obj_json)
                    rows["full_structured_sequence_category"].append(sequence_dict_cat_json)
                    sequence_struct_written = True
                else:
                    rows["full_structured_sequence_object"].append(np.nan)
                    rows["full_structured_sequence_category"].append(np.nan)

# save rows to dataframe
df_decoding_results = pd.DataFrame(rows)
# create output folder
output_folder = create_script_named_folder('decoding_results_tables_for_SODA')
# save to csv
df_decoding_results.to_csv(
    opj(
        output_folder,
        f'{sub}_ROI_{cmask}_decoding_results_retrievalpostsleep_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.csv'
    ),
    index=False
)

# Save scene-retrieval decoding outputs for later group-level analysis (mirrors 4.4)
decoding_results_path = create_script_named_folder('decoding_timecourses_data')
np.save(
    opj(
        decoding_results_path,
        f'{sub}_ROI_{cmask}_decoding_timecourses_retrievalpostsleep_scene_retrieved_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'
    ),
    decoding_results_scene_retrieved
)
# ------------------------------------------------------------------------
# Quick plot: is the correct retrieved scene class peaking?
# Color-code by cue type (ret_cue) and average retrieval position (seq_pos)
#   - night: Blues (light early -> dark late)
#   - day:   Reds  (light early -> dark late)
# ------------------------------------------------------------------------
plt_path = create_script_named_folder('decoding_timecourses_plots')

plt.figure(figsize=(10, 6))

all_timecourses_scene_ret = []

# 1) Helper to extract numeric position from seq_pos like '02.png' -> 2
def extract_pos(s):
    try:
        return int(os.path.splitext(str(s))[0])
    except Exception:
        return np.nan

df_scenes_pos = df_scenes.copy()
df_scenes_pos['pos_idx'] = df_scenes_pos['seq_pos'].apply(extract_pos)

# 2) For each retrieved scene (sc_ret_targ), get:
#    - average position
#    - dominant ret_cue (e.g., 'night.png' or 'day.png')
scene_stats = (
    df_scenes_pos
    .groupby('sc_ret_targ')
    .agg(
        pos_idx_mean=('pos_idx', 'mean'),
        ret_cue_mode=('ret_cue', lambda x: x.value_counts().idxmax())
    )
)

# Position normalization
pos_values = scene_stats['pos_idx_mean'].to_numpy()
pos_values = pos_values[~np.isnan(pos_values)]
if len(pos_values) > 0:
    pos_min, pos_max = pos_values.min(), pos_values.max()
else:
    pos_min, pos_max = 0.0, 1.0  # fallback

# Colormaps
# use later values in colormap for later positions (darker)
# use only the darker half of the colormaps (map norm [0..1] -> cmap [0.5..1])
cmap_night = lambda v: plt.cm.Blues(0.5 + 0.5 * np.clip(v, 0.0, 1.0))
cmap_day   = lambda v: plt.cm.Reds(  0.5 + 0.5 * np.clip(v, 0.0, 1.0))


for stim, timecourses in decoding_results_scene_retrieved.items():
    stim_avg = np.mean(timecourses, axis=0)
    all_timecourses_scene_ret.extend(timecourses)

    if stim in scene_stats.index:
        pos = scene_stats.loc[stim, 'pos_idx_mean']
        cue = str(scene_stats.loc[stim, 'ret_cue_mode']).lower()
    else:
        pos = np.nan
        cue = ''

    # Normalize position to [0,1] for light->dark mapping
    if np.isnan(pos) or pos_max == pos_min:
        norm_pos = 0.5
    else:
        norm_pos = (pos - pos_min) / (pos_max - pos_min)

    # Choose palette based on ret_cue
    if 'night' in cue:
        color = cmap_night(norm_pos)
    elif 'day' in cue:
        color = cmap_day(norm_pos)
    else:
        # fallback neutral if cue unknown
        color = plt.cm.Greys(norm_pos)

    label = stim
    if not np.isnan(pos):
        label += f" (pos~{pos:.1f}, cue={cue})"

    plt.plot(tr_window, stim_avg, color=color, alpha=0.9, linewidth=2, label=label)

# Grand average across all retrieved-scene timecourses
if all_timecourses_scene_ret:
    grand_avg_scene_ret = np.mean(all_timecourses_scene_ret, axis=0)
    plt.plot(tr_window, grand_avg_scene_ret, color='black', linewidth=4, label='Grand average')

plt.axvline(0, color='red', linestyle='--')
plt.xlabel('TRs from onset')
plt.ylabel('P(correct retrieved scene)')
plt.title(
    f'Scene retrieval decoding (d_no4sc, subject {sub})\n'
    'Color = ret_cue (night: blue, day: red) - position (light early → dark late)'
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(opj(plt_path, f'{sub}_ROI_{cmask}_scene_retrieval_decoding_d_no4sc_by_cue_and_position.png'))
plt.show()
