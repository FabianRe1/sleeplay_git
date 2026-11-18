# -*- coding: utf-8 -*-
"""4.3 – Transfer visual classifier to encoding session.

Decode scene and object categories from encoding fMRI data using classifiers
trained in 4.1 on localizer data.  Produces per-trial decoding timecourses
around stimulus onset and saves summary plots + raw data.
"""

import glob
import os
import re
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from groo.groo import get_root
from nilearn import image, masking, signal
from joblib import load

from utils import (
    parse_subject_arg,
    parse_roi_arg,
    script_output_dir,
    load_mask,
    CONFOUND_VARS,
)


# =====================================================================
# SETUP
# =====================================================================

root_folder = get_root(".my_hidden_root_file")
path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
path_masks = opj(root_folder, 'code', '3_reliability_voxel_selection', 'combine_loc1_and_loc3_masks')
path_clf_dir = opj(root_folder, 'code', '4_decoding_claude', '4.1_train_visual_classifier', 'classifiers')

TR = 1.5
csmooth = 4
temporal_smoothing_level = 3
f_threshold = 30
reliability_threshold = 70

sub = parse_subject_arg(sys.argv[1] if len(sys.argv) >= 2 else None, default="s13")
cmask = parse_roi_arg(sys.argv[2] if len(sys.argv) >= 3 else None, default="vis")
sub_num = sub[1:]

mri_sessions_enc = 'ses-04'
use_overall_classifier1_specific2 = 2


# =====================================================================
# BEHAVIORAL DATA
# =====================================================================

def load_behavioral_data(root_folder, sub, sub_num):
    """Load localizer and encoding behavioral data.

    Returns
    -------
    obj_enc_to_cat : dict
        Mapping from obj_enc_stim to obj_cat.
    df_behavior_encoding : pd.DataFrame
        Encoding behavior with obj_cat and forward-filled trl_nr / run_nr.
    """
    # Localizer behaviour
    path_loc = opj(root_folder, 'behavior', sub, 'behav', 'loc', f's{sub_num}_df_loc1_w_HRF.csv')
    file_list = glob.glob(path_loc)
    df_behavior_loc1 = pd.read_csv(file_list[0], sep=None, engine='python')
    obj_enc_to_cat = df_behavior_loc1.groupby('obj_enc_stim', sort=False)['obj_cat'].first().to_dict()

    # Encoding behaviour – find nses_0 folder with highest step
    nses_0_pattern = opj(root_folder, 'behavior', sub, 'behav', 'main', f'*_nses_0_step*')
    nses_0_folders = sorted(glob.glob(nses_0_pattern))
    if not nses_0_folders:
        raise FileNotFoundError(f"No nses_0_step* folders found matching: {nses_0_pattern}")
    nses_0_folder = max(nses_0_folders, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
    print(f"[behavior] Found {len(nses_0_folders)} nses_0 folder(s), using highest step: {os.path.basename(nses_0_folder)}")

    path_behavior = opj(nses_0_folder, f'output_df_sub{sub_num}_w_HRF.csv')
    df_behavior_encoding = pd.read_csv(path_behavior, sep=',')

    df_behavior_encoding['obj_cat'] = df_behavior_encoding['obj_enc_stim'].map(obj_enc_to_cat)
    df_behavior_encoding['trl_nr'] = df_behavior_encoding['trl_nr'].ffill()
    df_behavior_encoding['run_nr'] = df_behavior_encoding['run_nr'].ffill()

    return obj_enc_to_cat, df_behavior_encoding


# =====================================================================
# FUNCTIONAL DATA
# =====================================================================

def load_and_preprocess_func(path_fmriprep, sub_num, mri_sessions_enc, mask_img,
                             csmooth, TR):
    """Load encoding functional data, apply mask, smooth, detrend, z-score.

    Returns
    -------
    data_masked_detrend : list[np.ndarray]
        Cleaned data arrays (n_timepoints x n_voxels) per run.
    """
    # Functional images
    path_func = sorted(glob.glob(
        opj(path_fmriprep, f"sub-{sub_num}", mri_sessions_enc, "func",
            f"sub-{sub_num}_{mri_sessions_enc}_task-mem*_space-T1w_desc-preproc_bold.nii.gz")
    ))
    print('=======================')
    print(path_func)
    data_task = [image.load_img(i) for i in path_func]
    for i in range(len(data_task)):
        print(f'Func data scan 1 run {i} shape: {data_task[i].shape}')

    # Confounds
    path_confs = sorted(
        glob.glob(opj(path_fmriprep, f'sub-{sub_num}', mri_sessions_enc, 'func',
                       f'sub-{sub_num}_{mri_sessions_enc}_task-mem*_run*_desc-confounds_timeseries.tsv')),
        key=lambda f: os.path.basename(f)
    )
    print('=======================')
    print(path_confs)
    confs = [pd.read_csv(p, sep='\t') for p in path_confs]

    # Mask, smooth, detrend, z-score
    data_masked = [masking.apply_mask(i, mask_img, smoothing_fwhm=csmooth) for i in data_task]
    data_masked_detrend = [
        signal.clean(
            signals=data_masked[cr],
            t_r=TR,
            detrend=True,
            standardize='zscore_sample',
            confounds=confs[cr][CONFOUND_VARS].fillna(0),
        )
        for cr in range(len(data_masked))
    ]
    return data_masked_detrend


# =====================================================================
# CLASSIFIER LOADING
# =====================================================================

def load_classifiers(path_clf_dir, sub_num, cmask, csmooth, f_threshold,
                     reliability_threshold, temporal_smoothing_level,
                     use_overall_classifier1_specific2):
    """Load classifiers from disk.

    NOTE: In 4.3 the sub in the classifier filename uses s{sub_num}
    (unlike 4.4 which uses the sub variable directly).

    Returns
    -------
    clf_scenes, clf_objects
    """
    base = (f"s{sub_num}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{f_threshold}"
            f"_reliability_threshold_{reliability_threshold}"
            f"_tmpsmoothing_{temporal_smoothing_level}")

    if use_overall_classifier1_specific2 == 1:
        filename = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_3.joblib")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Generic classifier not found at: {filename}")
        print(f"Loading GENERIC classifier from: {filename}")
        clf_generic = load(filename)
        return clf_generic, clf_generic

    elif use_overall_classifier1_specific2 == 2:
        filename_scenes = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_1.joblib")
        filename_objects = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_2.joblib")
        if not os.path.exists(filename_scenes):
            raise FileNotFoundError(f"Scene classifier not found at: {filename_scenes}")
        if not os.path.exists(filename_objects):
            raise FileNotFoundError(f"Object classifier not found at: {filename_objects}")
        print(f"Loading SCENE classifier from: {filename_scenes}")
        print(f"Loading OBJECT classifier from: {filename_objects}")
        return load(filename_scenes), load(filename_objects)

    raise ValueError(f"Invalid use_overall_classifier1_specific2: {use_overall_classifier1_specific2}")


# =====================================================================
# DECODING
# =====================================================================

def decode_timecourses(df_trials, data_masked_detrend, clf, target_col,
                       run_mapping, tr_window):
    """Run trial-wise decoding around stimulus onset.

    Parameters
    ----------
    df_trials : pd.DataFrame
        Filtered behavioral rows (one per trial).
    data_masked_detrend : list[np.ndarray]
        Cleaned fMRI data per run.
    clf : classifier
        Trained sklearn classifier with predict_proba.
    target_col : str
        Column in df_trials holding the target class label.
    run_mapping : dict
        Maps behavioral run_nr -> index into data_masked_detrend.
    tr_window : np.ndarray
        TR offsets around onset (e.g. arange(-4, 5)).

    Returns
    -------
    decoding_results : dict
        {stimulus_label: [array of shape (len(tr_window),), ...]}
    """
    decoding_results = {}

    for idx, row in df_trials.iterrows():
        run_nr = int(row['run_nr'])
        data_idx = run_mapping[run_nr]
        fmri_data = data_masked_detrend[data_idx]

        onset_tr = int(np.round(row['TR_time_onset_w_HRF']))
        target_class = row[target_col]

        class_idx = np.where(clf.classes_ == target_class)[0][0]

        window_idxs = (onset_tr + tr_window).astype(int)
        n_tp = fmri_data.shape[0]
        valid = (window_idxs >= 0) & (window_idxs < n_tp)

        # Only decode valid TRs; out-of-bounds positions get NaN
        target_prob_timecourse = np.full(len(window_idxs), np.nan)
        if valid.any():
            X_valid = fmri_data[window_idxs[valid], :]
            probs = clf.predict_proba(X_valid)
            target_prob_timecourse[valid] = probs[:, class_idx]

        if target_class not in decoding_results:
            decoding_results[target_class] = []
        decoding_results[target_class].append(target_prob_timecourse)

    return decoding_results


# =====================================================================
# PLOTTING
# =====================================================================

def plot_summary(decoding_results, tr_window, sub, cmask,
                 use_overall_classifier1_specific2, label_type, plt_path):
    """Plot per-stimulus mean + grand average decoding timecourse.

    Parameters
    ----------
    label_type : str
        'scene' or 'obj' – used in title and filename.
    plt_path : str
        Directory to save the figure.
    """
    plt.figure(figsize=(12, 8))
    all_timecourses = []

    for stim, timecourses in decoding_results.items():
        stim_avg = np.mean(timecourses, axis=0)
        all_timecourses.extend(timecourses)
        plt.plot(tr_window, stim_avg, alpha=0.8, linewidth=2, label=stim)

    if all_timecourses:
        grand_avg = np.mean(all_timecourses, axis=0)
        plt.plot(tr_window, grand_avg, color='black', linewidth=6, label='Grand Average')

    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Onset +5seconds (TRs)')
    plt.ylabel('Decoding Probability (Target Class)')
    plt.title(f'Summary: Decoding Timecourse (Subject {sub} {cmask})\n'
              f'trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(opj(plt_path,
                    f'{sub}_decoding_timecourses_summary_{label_type}'
                    f'_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png'))
    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():
    # ---- Behavioral data ----
    obj_enc_to_cat, df_behavior_encoding = load_behavioral_data(root_folder, sub, sub_num)

    # ---- Functional data ----
    mask_img = load_mask(path_masks, sub, cmask, reliability_threshold)
    data_masked_detrend_scan1 = load_and_preprocess_func(
        path_fmriprep, sub_num, mri_sessions_enc, mask_img, csmooth, TR
    )

    # ---- Classifiers ----
    clf_scenes, clf_objects = load_classifiers(
        path_clf_dir, sub_num, cmask, csmooth, f_threshold,
        reliability_threshold, temporal_smoothing_level,
        use_overall_classifier1_specific2
    )

    # ---- Validate run mapping ----
    # Skipped runs (indices 0, 2) should be 120-TR restaurant runs
    for skip_idx in [0, 2]:
        n_trs = data_masked_detrend_scan1[skip_idx].shape[0]
        if n_trs != 120:
            raise ValueError(
                f"Run index {skip_idx} has {n_trs} TRs (expected 120 for restaurant). "
                f"Run mapping {{1: 1, 2: 3, 3: 4}} may be wrong for subject {sub}."
            )
    for enc_idx in [1, 3, 4]:
        n_trs = data_masked_detrend_scan1[enc_idx].shape[0]
        if n_trs == 120:
            print(f"WARNING: Encoding run index {enc_idx} has 120 TRs — "
                  f"might be a restaurant run for subject {sub}.")

    # ---- Prepare trial subsets ----
    df_scenes = df_behavior_encoding[df_behavior_encoding['label_type'] == 'dn+sc_cue'].copy()
    df_objects = df_behavior_encoding[df_behavior_encoding['label_type'] == 'dn+sc+obj_cue'].copy()

    run_mapping = {1: 1, 2: 3, 3: 4}
    tr_window = np.arange(-4, 5)  # 9 TRs

    # ---- Scene decoding ----
    decoding_results = decode_timecourses(
        df_scenes, data_masked_detrend_scan1, clf_scenes,
        target_col='sc_enc_stim', run_mapping=run_mapping, tr_window=tr_window
    )

    plt_path = script_output_dir(__file__, 'decoding_timecourses_plots')
    data_path = script_output_dir(__file__, 'decoding_timecourses_data')

    plot_summary(decoding_results, tr_window, sub, cmask,
                 use_overall_classifier1_specific2, 'scene', plt_path)
    np.save(opj(data_path,
                f'{sub}_{cmask}_decoding_timecourses_scene'
                f'_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'),
            decoding_results)

    # ---- Object decoding ----
    decoding_results_obj = decode_timecourses(
        df_objects, data_masked_detrend_scan1, clf_objects,
        target_col='obj_cat', run_mapping=run_mapping, tr_window=tr_window
    )

    plot_summary(decoding_results_obj, tr_window, sub, cmask,
                 use_overall_classifier1_specific2, 'obj', plt_path)
    np.save(opj(data_path,
                f'{sub}_{cmask}_decoding_timecourses_obj'
                f'_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'),
            decoding_results_obj)


if __name__ == '__main__':
    main()
