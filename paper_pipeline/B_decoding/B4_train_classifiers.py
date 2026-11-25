#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B4 - Train the stimulus classifiers on the two localizers.

For one participant and one ROI: logistic-regression classifiers for the ten
scenes and for the ten object categories, trained on both
the encoding trials (image shown) and the retrieval trials (word cue shown,
image recalled) of the localizers. Two modes are run one after the other:
  - leave-one-run-out cross-validation (accuracy tables and plots)
  - training on all data; the fitted classifier is saved for B6 and B7

Settings: 4 mm spatial smoothing, temporal smoothing 3, top 30 percent of
voxels by F statistic, 70th-percentile reliability masks.

Input   outputs/A2_add_tr_columns_localizer/sXX/sXX_df_loc{1,2}_w_HRF.csv
        outputs/B3_combine_localizer_masks/sub-XX_mask_<roi>_threshold_70_combined.nii
        <derivatives>/sub-XX/ses-0{2,3}/func/  BOLD (T1w space), confounds
                                                                    (read-only)
Output  outputs/B4_train_classifiers/classifiers/*.joblib
        outputs/B4_train_classifiers/df_classifier_results_csv/*.csv
        outputs/B4_train_classifiers/df_summary/df_mean_performance_sXX_Mask_<roi>.csv
        outputs/B4_train_classifiers/behavior_with_training_columns/sXX/
            sXX_df_loc{1,2}_w_HRF_with_training_columns.csv
Plots   plots/B4_train_classifiers/*.png   (cross-validation accuracy per stimulus)

Ported from 4_decoding_claude_retrieval/4.1_train_visual_classifier.py. The
scene and object classifiers are computed exactly as in the original.
Changes (2026-11-18):
  - the data of a region are masked, smoothed and cleaned once instead of once
    per mode and classifier (same result, about a sixth of the reading)
  - the combined 20-class classifier is no longer trained (not used later)
  - the SMOTE oversampling has a fixed seed
  - df_summary is written per participant and has a classifier column
  - the `_with_training_columns.csv` files go to this step's output folder
    instead of `behavior/`

Usage:  python B4_train_classifiers.py <subject> <roi>
        e.g.  python B4_train_classifiers.py sub-13 vis
"""

import glob
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nilearn import image, masking, signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from imblearn.over_sampling import SMOTE
from joblib import dump
from tqdm import tqdm

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DERIVATIVES_DIR, step_input_dir, step_output_dir, step_plot_dir
from decoding_utils import (
    parse_subject_arg,
    parse_roi_arg,
    ensure_dir,
    CONFOUND_VARS,
    load_func_and_confounds,
    mask_and_clean,
    load_mask,
    assert_func_conf_match,
)

# =====================================================================
# Constants
# =====================================================================

SCENE_CATEGORIES = [
    'canyons01.jpg', 'road01.jpg', 'forest03.jpg', 'desert03.jpg',
    'mountain02.jpg', 'winter01.jpg', 'lake01.jpg', 'beach02.jpg',
    'waterfall04.jpg', 'autumn03.jpg',
]

OBJECT_CATEGORIES = [
    'vehicle', 'furniture', 'fruit', 'clothing', 'kitchen',
    'sports', 'music', 'animal', 'tool', 'body',
]

TR = 1.5
SMOOTHING_LEVELS = [4]
TEMPORAL_SMOOTHING_LEVELS = [3]
F_THRESHOLDS = [30]
RELIABILITY_THRESHOLDS = [70]

WHAT_TO_DO = [0, 1]  # 0 = cross-val, 1 = train on everything

# Which classifiers to train: 1 = scenes (10 classes), 2 = objects (10 classes).
# The original also trained 3 = scenes and objects together (20 classes); no
# later step uses it, so it was dropped on 2026-11-18. Add 3 to get it back.
CLASSIFIER_SETS = [1, 2]

# Seed for the SMOTE oversampling. SMOTE only draws random samples when the
# classes are unbalanced; with balanced classes the seed has no effect.
SMOTE_SEED = 42


# =====================================================================
# Helper: exclusive merge (unchanged logic)
# =====================================================================

def _exclusive_merge(df, col_a, col_b):
    if col_a not in df.columns:
        raise KeyError(f"Missing required column: {col_a}")
    if col_b not in df.columns:
        raise KeyError(f"Missing required column: {col_b}")

    s1 = df[col_a]
    s2 = df[col_b]

    v1_base = ~pd.isna(s1)
    v2_base = ~pd.isna(s2)

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
        raise ValueError(
            f"Overlap detected between {col_a} and {col_b} at rows "
            f"{idx[:20]} (showing up to 20); total overlaps: {overlap.sum()}"
        )

    merged = pd.Series(np.nan, index=df.index, dtype=object)
    pick_a = valid1 & ~valid2
    pick_b = valid2 & ~valid1
    merged.loc[pick_a] = s1.loc[pick_a]
    merged.loc[pick_b] = s2.loc[pick_b]
    return merged


def adjust_series_indices(series_list):
    """Re-index a list of Series so indices are contiguous across entries."""
    for i in range(1, len(series_list)):
        last_index = series_list[i - 1].index[-1] + 1
        current_length = len(series_list[i])
        series_list[i].index = pd.RangeIndex(
            start=last_index, stop=last_index + current_length
        )
    return series_list


# =====================================================================
# Behavioral data loading — ENCODING + RETRIEVAL
# =====================================================================

def load_behavioral_data(path_localizer_behavior, sub, sub_num, loc_session, debug_mode):
    """Load and prepare behavioral data for one localizer session.

    Unlike the encoding-only version, this includes BOTH encoding and
    retrieval trials. The training_category column identifies the target
    category for both trial types.

    Returns
    -------
    df : DataFrame  (rows with valid training_category only)
    path_behavior : str
    """
    path_behavior = opj(
        path_localizer_behavior, sub,
        f's{sub_num}_df_{loc_session}_w_HRF.csv',
    )

    file_list = glob.glob(path_behavior)
    df = pd.read_csv(file_list[0], sep=';')

    # --- Encoding category (same as original) ---
    df['encoding_stimulus'] = _exclusive_merge(df, 'obj_enc_stim', 'sc_enc_stim')
    df['encoding_category'] = _exclusive_merge(df, 'obj_cat', 'sc_enc_stim')

    _only_category = df['encoding_stimulus'].isna() & df['encoding_category'].notna()
    if _only_category.any():
        df.loc[_only_category, 'encoding_category'] = np.nan

    # --- Retrieval category (NEW) ---
    # For scene retrieval: sc_ret_targ is the target scene filename (= category)
    # For object retrieval: obj_cat is the category
    df['retrieval_category'] = np.nan

    sc_ret_mask = df['label_type'] == 'sc_ret_cue'
    if sc_ret_mask.any():
        df.loc[sc_ret_mask, 'retrieval_category'] = df.loc[sc_ret_mask, 'sc_ret_targ']

    obj_ret_mask = df['label_type'] == 'obj_ret_cue'
    if obj_ret_mask.any():
        df.loc[obj_ret_mask, 'retrieval_category'] = df.loc[obj_ret_mask, 'obj_cat']

    # --- Combined training category ---
    # encoding_category for encoding trials, retrieval_category for retrieval trials
    df['training_category'] = df['encoding_category'].fillna(df['retrieval_category'])

    # --- Track trial phase (for later inspection of per-phase performance) ---
    df['trial_phase'] = np.nan
    df.loc[df['encoding_category'].notna(), 'trial_phase'] = 'encoding'
    df.loc[df['retrieval_category'].notna(), 'trial_phase'] = 'retrieval'

    # Print summary
    n_enc = (df['trial_phase'] == 'encoding').sum()
    n_ret = (df['trial_phase'] == 'retrieval').sum()
    print(f"  {loc_session}: {n_enc} encoding + {n_ret} retrieval = {n_enc + n_ret} training trials")

    # Save enriched df (into this step's output folder, not next to the input)
    path_enriched = step_output_dir(__file__, opj('behavior_with_training_columns', sub))
    df.to_csv(
        opj(path_enriched,
            os.path.basename(path_behavior).replace('.csv', '_with_training_columns.csv')),
        sep=';', index=False)

    # Keep only rows with a valid training category
    df = df.dropna(subset=['training_category']).reset_index(drop=True)
    return df, path_behavior


def prepare_localizer_behavior(path_localizer_behavior, sub, sub_num, debug_mode):
    """Load and combine behavioral data for both localizer sessions.

    Returns
    -------
    df_behavior, labels, runvector, TR_video_onset, runs, nruns_1, nruns_2
    """
    # Localizer 1 ----------------------------------------------------------
    df_loc1, _ = load_behavioral_data(
        path_localizer_behavior, sub, sub_num, 'loc1', debug_mode
    )
    runs_1 = df_loc1.run_nr.unique()
    runs_1 = runs_1[~pd.isna(runs_1)].astype(int)
    nruns_1 = len(runs_1)

    grouped_1 = df_loc1.groupby('run_nr')
    trs_loc1 = [
        group['TR_time_onset_w_HRF']
        for _, group in sorted(grouped_1, key=lambda x: x[0])
    ]
    labels_1 = df_loc1.training_category
    runvec_1 = df_loc1.run_nr.copy()

    # Localizer 2 ----------------------------------------------------------
    df_loc2, _ = load_behavioral_data(
        path_localizer_behavior, sub, sub_num, 'loc2', debug_mode
    )
    runs_2 = df_loc2.run_nr.unique() + 13
    runs_2 = runs_2[~pd.isna(runs_2)].astype(int)
    nruns_2 = len(np.unique(df_loc2.run_nr))

    grouped_2 = df_loc2.groupby('run_nr')
    trs_loc2 = [
        group['TR_time_onset_w_HRF']
        for _, group in sorted(grouped_2, key=lambda x: x[0])
    ]
    labels_2 = df_loc2.training_category
    runvec_2 = df_loc2.run_nr.copy() + 13

    # Combine ---------------------------------------------------------------
    trs_combined = trs_loc1 + trs_loc2
    TR_video_onset = adjust_series_indices(trs_combined)

    df_behavior = pd.concat([df_loc1, df_loc2]).reset_index(drop=True)
    labels = pd.concat([labels_1, labels_2], ignore_index=True).reset_index(drop=True)
    runvector = pd.concat((runvec_1, runvec_2)).reset_index(drop=True)

    nruns = nruns_1 + nruns_2
    runs = np.concatenate((runs_1, runs_2))

    return df_behavior, labels, runvector, TR_video_onset, runs, nruns_1, nruns_2


# =====================================================================
# Functional data loading
# =====================================================================

def load_func_data(path_fmriprep, sub_num, session, debug_mode):
    """Load functional images and confounds for one localizer session."""
    path_func = opj(
        path_fmriprep, f'sub-{sub_num}', session, 'func',
        f"sub-{sub_num}_{session}_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz",
    )
    func_paths = sorted(glob.glob(path_func), key=lambda f: os.path.basename(f))
    func_imgs = [image.load_img(i) for i in func_paths]
    print('=======================')
    print(func_paths)
    print('=======================')

    path_confs = opj(
        path_fmriprep, f'sub-{sub_num}', session, 'func',
        f'sub-{sub_num}_{session}_task-*_run*_desc-confounds_timeseries.tsv',
    )
    conf_paths = sorted(glob.glob(path_confs), key=lambda f: os.path.basename(f))
    print('=======================')
    print(conf_paths)
    print('=======================')
    assert_func_conf_match(func_paths, conf_paths)
    confounds = [pd.read_csv(p, sep='\t') for p in conf_paths]

    return func_imgs, confounds


def apply_mask_and_clean(func_imgs, confounds, mask_img, csmooth, nruns):
    """Mask, smooth and clean functional data."""
    masked = [
        masking.apply_mask(img, mask_img, smoothing_fwhm=csmooth)
        for img in func_imgs
    ]
    cleaned = [
        signal.clean(
            signals=masked[cr],
            t_r=TR,
            detrend=True,
            standardize='zscore_sample',
            confounds=confounds[cr][CONFOUND_VARS].fillna(0),
        )
        for cr in range(nruns)
    ]
    return cleaned


# =====================================================================
# TR alignment check
# =====================================================================

def check_tr_alignment(TR_video_onset, data_cleaned, nruns):
    """Drop TRs where behavior extends beyond MRI length."""
    trs_filtered = []
    for j in range(nruns):
        n_mri_trs = data_cleaned[j].shape[0]
        print(f'localizer: TRs from behavior {TR_video_onset[j].iloc[-1]}; '
              f'TRs recorded in MRI {n_mri_trs}')
        if TR_video_onset[j].iloc[-1] >= n_mri_trs:
            print(f'Too few TRs at run {j} in the localizer')
        new_series = TR_video_onset[j][TR_video_onset[j] < n_mri_trs]
        trs_filtered.append(new_series)

    valid_indices = np.concatenate([s.index.values for s in trs_filtered])
    return trs_filtered, valid_indices


# =====================================================================
# Temporal smoothing
# =====================================================================

def apply_temporal_smoothing(
    temporal_smoothing, data_stacked, trs_filtered, data_cleaned, nruns
):
    """Average neighbouring TRs according to the smoothing level."""
    if temporal_smoothing == 0:
        print('No temporal smoothing')
        return data_stacked

    offset_map = {1: [-1, 0], 2: [-1, 0, 1], 3: [-1, 0, 1, 2]}
    offsets = offset_map.get(temporal_smoothing)
    if offsets is None:
        return data_stacked

    smoothed_parts = []
    for cr in range(nruns):
        n_trs_mri = data_cleaned[cr].shape[0]
        trs = trs_filtered[cr].astype(int).values
        n_voxels = data_cleaned[cr].shape[1]
        accum = np.zeros((len(trs), n_voxels))
        count = np.zeros((len(trs), 1))
        for offset in offsets:
            shifted = trs + offset
            valid = (shifted >= 0) & (shifted < n_trs_mri)
            accum[valid] += data_cleaned[cr][shifted[valid]]
            count[valid] += 1
        smoothed_parts.append(accum / count)
    return np.vstack(smoothed_parts)


# =====================================================================
# Category filtering
# =====================================================================

def filter_categories(only_scenes1_objects2_both3, curr_data, curr_labels,
                      curr_runvector, curr_df):
    """Filter to scenes, objects or both. Returns filtered copies."""
    if only_scenes1_objects2_both3 == 1:
        sel_idx = curr_df['training_category'].isin(SCENE_CATEGORIES)
    elif only_scenes1_objects2_both3 == 2:
        sel_idx = curr_df['training_category'].isin(OBJECT_CATEGORIES)
    else:
        return curr_data, curr_labels, curr_runvector, curr_df

    curr_data = curr_data[sel_idx.values]
    curr_labels = curr_labels[sel_idx].reset_index(drop=True)
    curr_runvector = curr_runvector[sel_idx].reset_index(drop=True)
    curr_df = curr_df.loc[sel_idx].reset_index(drop=True)
    return curr_data, curr_labels, curr_runvector, curr_df


# =====================================================================
# Build classifier pipeline
# =====================================================================

# Inverse regularisation strength C of the logistic regression, per ROI. Default 1.0
# (the original setting). For mtl a stronger regularisation was adopted on 2026-11-19
# after an exploration on the localizer data only (7_rsa_exploratory/14_mtl_decoder):
# C = 0.001 raises cross-validated accuracy in mtl for objects (13.5 -> 14.7 %, shown
# images 15.0 -> 17.1 %; recalled images 11.9 -> 12.3 %, not significant). Not tested
# in other ROIs. Note: stronger regularisation gives flatter class probabilities, so
# probability differences in mtl are on a smaller scale than in the other ROIs.
CLASSIFIER_C = {'mtl': 0.001}
DEFAULT_C = 1.0


def build_clf(cthreshold, cmask=None):
    """Build the standard logistic-regression pipeline (C depends on the ROI, see CLASSIFIER_C)."""
    return make_pipeline(
        StandardScaler(),
        SelectPercentile(f_classif, percentile=cthreshold),
        LogisticRegression(
            C=CLASSIFIER_C.get(cmask, DEFAULT_C), solver='lbfgs', penalty='l2',
            multi_class='ovr', max_iter=10000,
            class_weight='balanced', random_state=42,
        ),
    )


def remove_nan_labels(data, labels):
    """Exclude samples whose labels are NaN."""
    excl = np.where(pd.Series(labels).isna())
    if len(excl[0]) > 0:
        data = np.delete(data, excl, 0)
        labels = np.delete(labels, excl, 0)
    return data, labels


# =====================================================================
# Cross-validation
# =====================================================================

def run_cross_validation(curr_data, curr_labels, curr_runvector,
                         curr_runs, cthreshold, cmask=None):
    """Leave-one-run-out cross-validation."""
    predictions = None
    predictions_prob = None

    for iteration, irun in enumerate(
        tqdm(curr_runs, desc='Leave-one-run-out', unit='run')
    ):
        test_runs = [irun]
        train_runs = curr_runs[~pd.Series(curr_runs).isin(test_runs)]

        train_data = curr_data[curr_runvector.isin(train_runs)]
        test_data = curr_data[curr_runvector.isin(test_runs)]
        train_labels = curr_labels[curr_runvector.isin(train_runs)]
        test_labels = curr_labels[curr_runvector.isin(test_runs)]

        train_data, train_labels = remove_nan_labels(train_data, train_labels)
        train_data_re, train_labels_re = SMOTE(random_state=SMOTE_SEED).fit_resample(
            train_data, train_labels
        )

        clf = build_clf(cthreshold, cmask)
        clf.fit(train_data_re, train_labels_re)

        preds = clf.predict(test_data)
        preds_prob = clf.predict_proba(test_data)

        if iteration == 0:
            predictions = preds
            predictions_prob = preds_prob
        else:
            predictions = np.concatenate((predictions, preds))
            predictions_prob = np.concatenate((predictions_prob, preds_prob))

    return predictions, predictions_prob


# =====================================================================
# Train final classifier (all data)
# =====================================================================

def train_final_classifier(curr_data, curr_labels, cthreshold, out_path, cmask=None):
    """Train on everything and save the fitted pipeline."""
    train_data, train_labels = remove_nan_labels(curr_data, curr_labels)
    train_data_re, train_labels_re = SMOTE(random_state=SMOTE_SEED).fit_resample(
        train_data, train_labels
    )
    clf = build_clf(cthreshold, cmask)
    clf.fit(train_data_re, train_labels_re)
    dump(clf, out_path)
    return clf


# =====================================================================
# Save results & plot
# =====================================================================

def save_cv_results(df_temp, predictions, predictions_prob, cmask, csmooth,
                    cthreshold, reliability_threshold, temporal_smoothing,
                    sub, only_scenes1_objects2_both3, path_csv, path_plots):
    """Attach predictions to the behavioural df, save CSV and plot."""
    size_df = len(df_temp)

    df_temp['predicted_labels'] = list(predictions)
    df_temp['predicted_probabilities'] = predictions_prob.tolist()
    df_temp['mask'] = cmask
    df_temp['smoothing'] = np.repeat(csmooth, size_df)
    df_temp['fstat_percentile'] = np.repeat(cthreshold, size_df)
    df_temp['reliability_threshold'] = reliability_threshold
    df_temp['temporal_smoothing'] = temporal_smoothing
    df_temp['label_match'] = (
        df_temp['training_category'] == df_temp['predicted_labels']
    )

    # Save CSV
    filename_df = opj(
        path_csv,
        f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{sub}"
        f"_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}"
        f"_reliability_threshold_{reliability_threshold}"
        f"_tmpsmoothing_{temporal_smoothing}.csv",
    )
    df_temp.to_csv(filename_df)

    # --- Overall performance ---
    match_percentages = (
        df_temp.groupby('training_category')['label_match'].mean() * 100
    )
    mean_performance = match_percentages.mean()

    # --- Per-phase performance (encoding vs retrieval) ---
    phase_perf = {}
    for phase in ['encoding', 'retrieval']:
        phase_df = df_temp[df_temp['trial_phase'] == phase]
        if len(phase_df) > 0:
            phase_perf[phase] = (
                phase_df.groupby('training_category')['label_match'].mean() * 100
            ).mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(match_percentages.index, match_percentages.values)
    colors = plt.cm.PuBu(match_percentages.values / 100)
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])

    ax.set_title(
        f"{sub} Enc+Ret combined | "
        f"Mask_{cmask}_Smooth_{csmooth}_Thr_{cthreshold} "
        f"rel_{reliability_threshold} "
        f"tmpsmooth_{temporal_smoothing}\n"
        f"Avg_{mean_performance:.2f}%"
    )
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy (%)')

    # Sub-group annotations
    text_lines = []

    # Scene / object breakdown
    present_scene = [c for c in SCENE_CATEGORIES if c in match_percentages.index]
    present_object = [c for c in OBJECT_CATEGORIES if c in match_percentages.index]
    if present_scene:
        scene_perf = match_percentages.loc[present_scene].mean()
        text_lines.append(f"Scene mean: {scene_perf:.2f}% ({len(present_scene)} cats)")
    if present_object:
        object_perf = match_percentages.loc[present_object].mean()
        text_lines.append(f"Object mean: {object_perf:.2f}% ({len(present_object)} cats)")

    # Encoding / retrieval breakdown
    for phase, perf in phase_perf.items():
        text_lines.append(f"{phase.capitalize()} mean: {perf:.2f}%")

    if text_lines:
        ax.text(
            0.01, 0.98, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
        )

    ax.axhline(
        y=1 / len(df_temp.training_category.unique()) * 100,
        color='red', linestyle='--', label='Chance level',
    )
    plt.xticks(rotation=90)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.1f}%', ha='center', va='bottom')
    ax.legend()

    filename_plot = opj(
        path_plots,
        f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}"
        f"_reliability_threshold_{reliability_threshold}"
        f"_tmpsmoothing_{temporal_smoothing}"
        f"_only_scenes1_objects2_both3_{only_scenes1_objects2_both3}.png",
    )
    plt.savefig(filename_plot, bbox_inches='tight')
    plt.show()

    return mean_performance


# =====================================================================
# MAIN
# =====================================================================

def main():
    debug_mode = 0

    # --- Paths ---
    path_localizer_behavior = str(step_input_dir('A2_add_tr_columns_localizer'))
    path_fmriprep = str(DERIVATIVES_DIR)
    path_masks = str(step_input_dir('B3_combine_localizer_masks'))

    # --- Subject & ROI ---
    sub = parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None)
    sub_num = sub[1:]
    roi_arg = parse_roi_arg(sys.argv[2] if len(sys.argv) > 2 else None)
    mask_labels = [roi_arg]

    mri_sessions_loc1 = 'ses-02'
    mri_sessions_loc2 = 'ses-03'

    # --- Output directories ---
    path_clf = str(step_output_dir(__file__, 'classifiers'))
    path_csv = str(step_output_dir(__file__, 'df_classifier_results_csv'))
    path_plots = str(step_plot_dir(__file__))
    path_summary = str(step_output_dir(__file__, 'df_summary'))

    # --- Load behavioral data (encoding + retrieval) ---
    print("=" * 60)
    print("TRAINING MODE: Encoding + Retrieval trials combined")
    print("=" * 60)

    (df_behavior, labels_all, runvector_all, TR_video_onset,
     runs_all, nruns_1, nruns_2) = prepare_localizer_behavior(
        path_localizer_behavior, sub, sub_num, debug_mode
    )
    nruns = nruns_1 + nruns_2

    print(f"\nTotal training samples: {len(df_behavior)}")
    print(f"  Encoding: {(df_behavior['trial_phase'] == 'encoding').sum()}")
    print(f"  Retrieval: {(df_behavior['trial_phase'] == 'retrieval').sum()}")
    print(f"  Categories: {sorted(labels_all.unique())}")

    # --- Load functional data ---
    func_loc1, confs_loc1 = load_func_data(
        path_fmriprep, sub_num, mri_sessions_loc1, debug_mode
    )
    func_loc2, confs_loc2 = load_func_data(
        path_fmriprep, sub_num, mri_sessions_loc2, debug_mode
    )

    # --- Main loop ---
    # The data of a region are masked, smoothed and cleaned once and then used
    # for every mode (cross-validation, final training) and every classifier.
    # (The original repeated this step for each mode and classifier, with the
    # same result each time.)
    def _base_name(cmask, csmooth, cthreshold, reliability_threshold,
                   temporal_smoothing, clf_set):
        return (
            f"{sub}_Mask_{cmask}_Smooth_{csmooth}"
            f"_Threshold_{cthreshold}"
            f"_reliability_threshold_{reliability_threshold}"
            f"_tmpsmoothing_{temporal_smoothing}"
            f"_only_scenes1_objects2_both3_{clf_set}"
        )

    def _output_file(mode, cmask, csmooth, cthreshold, reliability_threshold,
                     temporal_smoothing, clf_set):
        """File that marks a (mode, classifier) combination as done."""
        if mode:
            return opj(path_clf, _base_name(cmask, csmooth, cthreshold,
                                            reliability_threshold,
                                            temporal_smoothing, clf_set) + ".joblib")
        return opj(
            path_csv,
            f"only_scenes1_objects2_both3_{clf_set}_{sub}"
            f"_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}"
            f"_reliability_threshold_{reliability_threshold}"
            f"_tmpsmoothing_{temporal_smoothing}.csv",
        )

    summary_rows = []

    for cmask in mask_labels:
        print('Running ' + cmask + ' mask')

        for reliability_threshold in RELIABILITY_THRESHOLDS:
            # Load mask
            path_mask = (
                f'sub-{sub_num}_mask_{cmask}_threshold_'
                f'{reliability_threshold}_combined.nii'
            )
            mask_img = image.load_img(opj(path_masks, path_mask))

            for csmooth in SMOOTHING_LEVELS:
                print(f'Smoothing {csmooth}fwhm')

                # Nothing to do if every output of this region already exists
                expected = [
                    _output_file(mode, cmask, csmooth, cthreshold,
                                 reliability_threshold, temporal_smoothing, clf_set)
                    for mode in WHAT_TO_DO
                    for clf_set in CLASSIFIER_SETS
                    for temporal_smoothing in TEMPORAL_SMOOTHING_LEVELS
                    for cthreshold in F_THRESHOLDS
                ]
                if all(os.path.isfile(f) for f in expected):
                    print(f"  SKIP {cmask}: all {len(expected)} outputs already exist")
                    continue

                # Mask, smooth, clean (once per region)
                print(f'Running wholebrain smoothing and applying '
                      f'anatomical {cmask} mask on loc 1 data')
                cleaned_loc1 = apply_mask_and_clean(
                    func_loc1, confs_loc1, mask_img, csmooth, nruns_1
                )
                print(f'Running wholebrain smoothing and applying '
                      f'anatomical {cmask} mask on loc 2 data')
                cleaned_loc2 = apply_mask_and_clean(
                    func_loc2, confs_loc2, mask_img, csmooth, nruns_2
                )
                data_cleaned = cleaned_loc1 + cleaned_loc2

                # TR alignment check
                trs_filtered, valid_indices = check_tr_alignment(
                    TR_video_onset, data_cleaned, nruns
                )

                runvector = runvector_all[valid_indices]
                labels = labels_all[valid_indices]
                df_scan = df_behavior.loc[
                    df_behavior.index.isin(valid_indices)
                ]

                # Select relevant TRs from MRI data
                data_selected = [
                    data_cleaned[cr][trs_filtered[cr].astype(int)]
                    for cr in range(nruns)
                ]
                data_stacked = np.vstack(data_selected)

                for temporal_smoothing in TEMPORAL_SMOOTHING_LEVELS:
                    data_smoothed = apply_temporal_smoothing(
                        temporal_smoothing, data_stacked,
                        trs_filtered, data_cleaned, nruns,
                    )

                    for use_everything_for_training in WHAT_TO_DO:
                        if use_everything_for_training == 0:
                            print('===================\nCross validation\n===================')
                        else:
                            print('===================\nTrain on everything to transfer\n===================')

                        for only_scenes1_objects2_both3 in CLASSIFIER_SETS:
                            # Filter categories
                            (curr_data, curr_labels, curr_runvector,
                             curr_df) = filter_categories(
                                only_scenes1_objects2_both3,
                                data_smoothed.copy(), labels.copy(),
                                runvector.copy(), df_scan.copy(),
                            )

                            # Sanity checks
                            n_enc = (curr_df['trial_phase'] == 'encoding').sum()
                            n_ret = (curr_df['trial_phase'] == 'retrieval').sum()
                            print(f"DEBUG: data shape: {curr_data.shape} "
                                  f"(enc={n_enc}, ret={n_ret})")
                            print(f"DEBUG: Unique classes: {curr_labels.unique()}")
                            print(f"DEBUG: Runs: {len(np.unique(curr_runvector))}")

                            for r in np.unique(curr_runvector):
                                n_samples = sum(curr_runvector == r)
                                if n_samples < 5:
                                    print(f"WARNING: Run {r} has only "
                                          f"{n_samples} samples after filtering!")

                            curr_runs = np.unique(curr_runvector)

                            for cthreshold in F_THRESHOLDS:
                                print(f'Feature selection at {cthreshold}th percentile')

                                _check_file = _output_file(
                                    use_everything_for_training, cmask, csmooth,
                                    cthreshold, reliability_threshold,
                                    temporal_smoothing, only_scenes1_objects2_both3,
                                )
                                if os.path.isfile(_check_file):
                                    print(f"  SKIP: output already exists -> "
                                          f"{os.path.basename(_check_file)}")
                                    continue

                                if use_everything_for_training:
                                    # Train on everything, save classifier
                                    print('Training on encoding + retrieval, '
                                          'saving classifier')
                                    train_final_classifier(
                                        curr_data, curr_labels,
                                        cthreshold, _check_file, cmask,
                                    )
                                else:
                                    # Cross-validation
                                    print('Running leave-one-run-out CV '
                                          '(encoding + retrieval)')
                                    predictions, predictions_prob = (
                                        run_cross_validation(
                                            curr_data, curr_labels,
                                            curr_runvector, curr_runs,
                                            cthreshold, cmask,
                                        )
                                    )

                                    df_temp = curr_df.copy()
                                    mean_perf = save_cv_results(
                                        df_temp, predictions, predictions_prob,
                                        cmask, csmooth, cthreshold,
                                        reliability_threshold,
                                        temporal_smoothing, sub,
                                        only_scenes1_objects2_both3,
                                        path_csv, path_plots,
                                    )

                                    summary_rows.append({
                                        'subj': sub,
                                        'mask': cmask,
                                        'classifier': {1: 'scene', 2: 'object', 3: 'both'}[
                                            only_scenes1_objects2_both3],
                                        'smoothing': csmooth,
                                        'fstat_percentile': cthreshold,
                                        'reliability_threshold': reliability_threshold,
                                        'temporal_smoothing': temporal_smoothing,
                                        'mean_performance': mean_perf,
                                    })

    # Summary of the cross-validated accuracy: one file per participant and
    # region. (The original appended every participant to one file per region
    # without a classifier column, so jobs running in parallel could overwrite
    # each other's rows and the scene and object rows replaced one another.)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        for mask_label, subset in df_summary.groupby('mask'):
            subset.to_csv(
                opj(path_summary, f"df_mean_performance_{sub}_Mask_{mask_label}.csv"),
                index=False,
            )


if __name__ == '__main__':
    main()
