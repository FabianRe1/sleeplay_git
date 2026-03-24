# -*- coding: utf-8 -*-
"""
Unified retrieval decoding script for presleep and postsleep sessions.

Merges the original 4.4 (presleep) and 4.5 (postsleep) scripts into a single
parametrised pipeline.

CLI arguments:
    sys.argv[1] : subject  ("13", "s13", or "sub-13")
    sys.argv[2] : ROI      ("vis", "cortical_memory", "hpc", ...)
    sys.argv[3] : pre_post ("presleep" or "postsleep"; default "presleep")

Output goes to:
    presleep  -> 4.4_transfer_visual_classifier_retrieval_presleep/
    postsleep -> 4.4_transfer_visual_classifier_retrieval_postsleep/
"""

import glob
import os
from os.path import join as opj
import sys

# Import utils from the original pipeline directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "4_decoding_claude"))
import re
import json

import numpy as np
import pandas as pd
from nilearn import image, masking, signal
from joblib import load
from matplotlib import pyplot as plt

from groo.groo import get_root
from utils import (
    parse_subject_arg,
    parse_roi_arg,
    script_output_dir,
    load_mask,
    load_classifiers,
    CONFOUND_VARS,
    build_sequence_dict_from_df_behavior_enc,
    map_sequence_objects_to_categories,
)


# =====================================================================
# Session-specific configuration
# =====================================================================

SESSION_CONFIG = {
    "presleep": {
        "mri_session": "ses-04",
        "nses_num": "0",
        "nses_glob_pattern": "*_nses_0_step*",
        "min_runs_after_filter": 6,
        "expected_exact_runs": True,      # presleep: must be exactly 6
        "n_retrieval_runs": 3,            # take last 3
        "output_folder_prefix": "4.4_transfer_visual_classifier_retrieval_presleep",
        "file_label": "retrievalpresleep",
    },
    "postsleep": {
        "mri_session": "ses-06",
        "nses_num": "2",
        "nses_glob_pattern": "*_nses_2_step*",
        "min_runs_after_filter": 6,
        "expected_exact_runs": False,     # postsleep: at least 6, up to 9
        "n_retrieval_runs": 6,            # take 6 longest
        "output_folder_prefix": "4.4_transfer_visual_classifier_retrieval_postsleep",
        "file_label": "retrievalpostsleep",
    },
}


# =====================================================================
# Output directory helper (mirrors original create_script_named_folder
# but routes to the correct output prefix)
# =====================================================================

def _output_dir(subfolder: str, output_prefix: str) -> str:
    """Create and return an output subdirectory under *output_prefix*."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = opj(script_dir, output_prefix, subfolder) if subfolder else opj(script_dir, output_prefix)
    os.makedirs(out, exist_ok=True)
    return out


# =====================================================================
# Behavioral data loading
# =====================================================================

def load_localizer_behavior(root_folder: str, sub: str):
    """Load localizer behaviour and return obj_enc_to_cat mapping."""
    sub_num = sub[1:]
    path_behavior = opj(root_folder, 'behavior', sub, 'behav', 'loc', f's{sub_num}_df_loc1_w_HRF.csv')
    file_list = glob.glob(path_behavior)
    if len(file_list) == 0:
        raise FileNotFoundError(f"Could not find localizer behavior file with pattern: {path_behavior}")
    df_behavior_loc = pd.read_csv(file_list[0], sep=None, engine='python')
    obj_enc_to_cat = df_behavior_loc.groupby('obj_enc_stim', sort=False)['obj_cat'].first().to_dict()
    return df_behavior_loc, obj_enc_to_cat


def load_main_task_behavior(root_folder: str, sub: str, cfg: dict):
    """Load encoding and retrieval behaviour DataFrames.

    For presleep (nses_0): encoding and retrieval may come from the same
    folder (highest step), with a fallback to step0 for retrieval.

    For postsleep (nses_2): encoding always comes from nses_0 (highest step),
    and retrieval comes from the nses_2 folder.
    """
    sub_num = sub[1:]
    nses_num = cfg["nses_num"]

    # ------------------------------------------------------------------
    # Always load encoding from nses_0 (highest step)
    # ------------------------------------------------------------------
    nses_0_pattern = opj(root_folder, 'behavior', sub, 'behav', 'main', '*_nses_0_step*')
    nses_0_folders = sorted(glob.glob(nses_0_pattern))
    if not nses_0_folders:
        raise FileNotFoundError(f"No nses_0_step* folders found matching: {nses_0_pattern}")

    nses_0_folder = max(
        nses_0_folders,
        key=lambda f: int(re.search(r'step(\d+)', f).group(1))
    )
    print(f"[behavior] Found {len(nses_0_folders)} nses_0 folder(s): {[os.path.basename(f) for f in nses_0_folders]}")
    print(f"[behavior] Using highest step for encoding: {os.path.basename(nses_0_folder)}")

    path_behavior_enc_best = opj(nses_0_folder, f'output_df_sub{sub_num}_w_HRF.csv')
    try:
        df_best = pd.read_csv(path_behavior_enc_best, sep=',')
    except Exception as e:
        raise FileNotFoundError(
            f"Could not read encoding behavior file: {path_behavior_enc_best}. Error: {e}"
        )

    if nses_num == "0":
        # ----- PRESLEEP: encoding + retrieval may live in the same folder -----
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

    else:
        # ----- POSTSLEEP (nses_2): encoding from nses_0, retrieval from nses_2 -----
        df_behavior_enc = df_best.copy()

        nses_2_pattern = opj(
            root_folder, 'behavior', sub, 'behav', 'main',
            f'sub{sub_num}_main_task_nses_{nses_num}_step*',
        )
        nses_2_folders = sorted(glob.glob(nses_2_pattern))
        if not nses_2_folders:
            raise FileNotFoundError(f"No nses_{nses_num}_step* folders found matching: {nses_2_pattern}")

        nses_folder = max(
            nses_2_folders,
            key=lambda f: int(re.search(r'step(\d+)', f).group(1))
        )
        print(f"[behavior] Found {len(nses_2_folders)} nses_{nses_num} folder(s): {nses_2_folders}")
        print(f"[behavior] Using highest step: {os.path.basename(nses_folder)}")

        path_behavior_ret = opj(nses_folder, f'output_df_sub{sub_num}_w_HRF.csv')
        try:
            df_behavior_retrieval = pd.read_csv(path_behavior_ret, sep=',')
            print(f"[behavior] Loaded postsleep retrieval from: {path_behavior_ret}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load postsleep retrieval behavior file: {path_behavior_ret}. Error: {e}"
            )

    return df_behavior_enc, df_behavior_retrieval


def prepare_retrieval_df(df_behavior_retrieval: pd.DataFrame, obj_enc_to_cat: dict) -> pd.DataFrame:
    """Add obj_cat column and forward-fill trl_nr / run_nr."""
    if "obj_ret_targ" not in df_behavior_retrieval.columns:
        raise KeyError(
            "df_behavior_retrieval missing required column 'obj_ret_targ'. "
            f"Available columns: {df_behavior_retrieval.columns.tolist()}"
        )
    df_behavior_retrieval['obj_cat'] = df_behavior_retrieval['obj_ret_targ'].map(obj_enc_to_cat)
    if "trl_nr" in df_behavior_retrieval.columns:
        df_behavior_retrieval['trl_nr'] = df_behavior_retrieval['trl_nr'].ffill()
    if "run_nr" in df_behavior_retrieval.columns:
        df_behavior_retrieval['run_nr'] = df_behavior_retrieval['run_nr'].ffill()
    return df_behavior_retrieval


# =====================================================================
# Functional data loading and preprocessing
# =====================================================================

def load_and_preprocess_fmri(
    path_fmriprep: str, sub_mri: str, mri_session: str,
    mask_img, csmooth: float, TR: float, conf_vars: list[str],
) -> list[np.ndarray]:
    """Load functional images + confounds, mask, smooth, detrend, z-score."""
    # Load functional images
    path_func = opj(
        path_fmriprep, sub_mri, mri_session, 'func',
        f'{sub_mri}_{mri_session}_task-mem*_space-T1w_desc-preproc_bold.nii.gz'
    )
    path_func = sorted(glob.glob(path_func), key=lambda f: os.path.basename(f))
    data_func = [image.load_img(i) for i in path_func]

    for i in range(len(data_func)):
        print(f'Func data run {i} shape: {data_func[i].shape}')

    # Load confounds
    path_confs = opj(
        path_fmriprep, sub_mri, mri_session, 'func',
        f'{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv'
    )
    path_confs = sorted(glob.glob(path_confs), key=lambda f: os.path.basename(f))
    confs = [pd.read_csv(p, sep='\t') for p in path_confs]

    # Mask, smooth, detrend, z-score
    data_masked = [masking.apply_mask(i, mask_img, smoothing_fwhm=csmooth) for i in data_func]
    data_cleaned = [
        signal.clean(
            signals=data_masked[cr],
            t_r=TR, detrend=True,
            standardize='zscore_sample',
            confounds=confs[cr][conf_vars].fillna(0),
        )
        for cr in range(len(data_masked))
    ]
    return data_cleaned


def select_retrieval_runs(data_cleaned: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """Filter runs by length (>120 TRs) and select retrieval runs.

    Presleep: expect exactly 6 runs after filtering, take last 3.
    Postsleep: expect at least 6, take 6 longest from up to 9.
    """
    data_filtered = [run for run in data_cleaned if run.shape[0] > 120]

    if cfg["expected_exact_runs"]:
        # Presleep: exactly 6 runs expected
        if len(data_filtered) != 6:
            raise ValueError(
                f"Expected exactly 6 runs after filtering (>120 TRs), "
                f"but got {len(data_filtered)}."
            )
        n_ret = cfg["n_retrieval_runs"]  # 3
        retrieval_runs = data_filtered[-n_ret:]
        for i, run in enumerate(retrieval_runs):
            print(f'Retrieval run {i} shape: {run.shape}')
        return retrieval_runs

    else:
        # Postsleep: at least 6, select 6 longest
        if len(data_filtered) < 6:
            raise ValueError(
                f"Expected at least 6 retrieval runs, but only got "
                f"{len(data_filtered)} total runs after filtering."
            )
        n_total = len(data_cleaned)
        if n_total != 9:
            print(f"WARNING: Expected 9 total runs (3 short + 6 retrieval), but got {n_total}.")

        run_lengths = [(i, run.shape[0]) for i, run in enumerate(data_filtered)]
        run_lengths_sorted = sorted(run_lengths, key=lambda x: x[1], reverse=True)
        top6_indices = sorted([idx for idx, _ in run_lengths_sorted[:6]])

        print(f"All run lengths: {run_lengths}")
        print(f"Selected 6 longest runs (indices): {top6_indices} "
              f"with lengths {[data_filtered[i].shape[0] for i in top6_indices]}")

        retrieval_runs = [data_filtered[idx] for idx in top6_indices]
        for i, idx in enumerate(top6_indices):
            print(f'Retrieval run {i} (original idx {idx}) shape: {data_filtered[idx].shape}')
        return retrieval_runs


# =====================================================================
# Run-number mapping
# =====================================================================

def build_run_nr_mapping(
    df_objects: pd.DataFrame,
    df_scenes: pd.DataFrame,
    n_retrieval_runs: int,
) -> dict[int, int]:
    """Build explicit mapping from behavioral run_nr values to data indices."""
    try:
        run_nrs_obj = pd.to_numeric(
            df_objects.get('run_nr', pd.Series([], dtype=float)), errors='coerce'
        ).dropna().astype(int)
        run_nrs_scn = pd.to_numeric(
            df_scenes.get('run_nr', pd.Series([], dtype=float)), errors='coerce'
        ).dropna().astype(int)
        observed_run_nrs = sorted(set(run_nrs_obj.tolist()) | set(run_nrs_scn.tolist()))
    except Exception:
        observed_run_nrs = []

    run_nr_to_data_idx = {}
    if len(observed_run_nrs) >= 1:
        for i, rn in enumerate(observed_run_nrs[:n_retrieval_runs]):
            run_nr_to_data_idx[int(rn)] = i
    else:
        for i in range(n_retrieval_runs):
            run_nr_to_data_idx[i + 1] = i
    return run_nr_to_data_idx


# =====================================================================
# Classifier application: object retrieval
# =====================================================================

def decode_object_retrieval(
    df_objects: pd.DataFrame,
    df_behavior_retrieval: pd.DataFrame,
    retrieval_data_list: list[np.ndarray],
    run_nr_to_data_idx: dict[int, int],
    clf_scenes, clf_objects,
    tr_window: np.ndarray,
    sub: str,
    rows: dict,
    sequence_dict_obj_json: str,
    sequence_dict_cat_json: str,
    sequence_struct_written: bool,
) -> tuple[dict, dict, bool]:
    """Apply scene + object classifiers to object retrieval trials (sc_no4obj).

    Returns
    -------
    decoding_results_scene_presented : dict
    decoding_results_object_retrieved : dict
    sequence_struct_written : bool (updated flag)
    """
    decoding_results_scene_presented = {}
    decoding_results_object_retrieved = {}

    for idx, row in df_objects.iterrows():
        run_nr_val = pd.to_numeric(row.get('run_nr', np.nan), errors='coerce')
        if pd.isna(run_nr_val):
            continue
        run_nr = int(run_nr_val)

        data_idx = run_nr_to_data_idx.get(run_nr, None)
        if data_idx is None or data_idx < 0 or data_idx >= len(retrieval_data_list):
            continue

        fmri_data = retrieval_data_list[data_idx]

        try:
            onset_tr = int(np.round(row['TR_time_onset']))
        except Exception:
            continue

        window_idxs = (onset_tr + tr_window).astype(int)
        n_tp = fmri_data.shape[0]
        valid = (window_idxs >= 0) & (window_idxs < n_tp)
        if not valid.any():
            continue

        # Only decode valid TRs; out-of-bounds get NaN
        X_valid = fmri_data[window_idxs[valid], :]
        target_scene = row.get('ret_cue', np.nan)

        # Scene classifier: presented scene
        if 'ret_cue' in df_objects.columns:
            target_scene = row['ret_cue']
            if target_scene in clf_scenes.classes_:
                class_idx_scene = np.where(clf_scenes.classes_ == target_scene)[0][0]
                probs_scene_valid = clf_scenes.predict_proba(X_valid)
                scene_tc = np.full(len(tr_window), np.nan)
                scene_tc[valid] = probs_scene_valid[:, class_idx_scene]

                if target_scene not in decoding_results_scene_presented:
                    decoding_results_scene_presented[target_scene] = []
                decoding_results_scene_presented[target_scene].append(scene_tc)

        # Object classifier: retrieved object (obj_cat)
        target_object = row['obj_cat']
        if target_object in clf_objects.classes_:
            class_idx_obj = np.where(clf_objects.classes_ == target_object)[0][0]
            probs_obj_valid = clf_objects.predict_proba(X_valid)
            # Build full probability array with NaN for out-of-bounds
            probs_obj = np.full((len(tr_window), len(clf_objects.classes_)), np.nan)
            probs_obj[valid] = probs_obj_valid
            obj_tc = probs_obj[:, class_idx_obj]

            if target_object not in decoding_results_object_retrieved:
                decoding_results_object_retrieved[target_object] = []
            decoding_results_object_retrieved[target_object].append(obj_tc)

            # Look up catch_trial once per trial
            df_behavior_retrieval_subset = df_behavior_retrieval[df_behavior_retrieval["trl_nr"] == row["trl_nr"]]
            vals = df_behavior_retrieval_subset["catch_trial"].dropna()
            catch_trial_pressed = vals.iloc[0] if not vals.empty else np.nan

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
                    rows["behav_correct"].append(catch_trial_pressed)

                    if not sequence_struct_written:
                        rows["full_structured_sequence_object"].append(sequence_dict_obj_json)
                        rows["full_structured_sequence_category"].append(sequence_dict_cat_json)
                        sequence_struct_written = True
                    else:
                        rows["full_structured_sequence_object"].append(np.nan)
                        rows["full_structured_sequence_category"].append(np.nan)

    return decoding_results_scene_presented, decoding_results_object_retrieved, sequence_struct_written


# =====================================================================
# Classifier application: scene retrieval
# =====================================================================

def decode_scene_retrieval(
    df_scenes: pd.DataFrame,
    df_behavior_retrieval: pd.DataFrame,
    retrieval_data_list: list[np.ndarray],
    run_nr_to_data_idx: dict[int, int],
    clf_scenes,
    tr_window: np.ndarray,
    sub: str,
    rows: dict,
    sequence_dict_obj_json: str,
    sequence_dict_cat_json: str,
    sequence_struct_written: bool,
) -> tuple[dict, bool]:
    """Apply scene classifier to scene retrieval trials (d_no4sc).

    Returns
    -------
    decoding_results_scene_retrieved : dict
    sequence_struct_written : bool (updated flag)
    """
    decoding_results_scene_retrieved = {}

    for idx, row in df_scenes.iterrows():
        run_nr_val = pd.to_numeric(row.get('run_nr', np.nan), errors='coerce')
        if pd.isna(run_nr_val):
            continue
        run_nr = int(run_nr_val)

        data_idx = run_nr_to_data_idx.get(run_nr, None)
        if data_idx is None or data_idx < 0 or data_idx >= len(retrieval_data_list):
            continue

        fmri_data = retrieval_data_list[data_idx]
        onset_tr = int(np.round(row['TR_time_onset']))

        window_idxs = (onset_tr + tr_window).astype(int)
        n_tp = fmri_data.shape[0]
        valid = (window_idxs >= 0) & (window_idxs < n_tp)
        if not valid.any():
            continue

        # Only decode valid TRs; out-of-bounds get NaN
        X_valid = fmri_data[window_idxs[valid], :]
        target_scene_ret = row['sc_ret_targ']

        if target_scene_ret in clf_scenes.classes_:
            probs_valid = clf_scenes.predict_proba(X_valid)
            # Build full probability array with NaN for out-of-bounds
            probs = np.full((len(tr_window), len(clf_scenes.classes_)), np.nan)
            probs[valid] = probs_valid

            class_idx = np.where(clf_scenes.classes_ == target_scene_ret)[0][0]
            scene_tc = probs[:, class_idx]

            if target_scene_ret not in decoding_results_scene_retrieved:
                decoding_results_scene_retrieved[target_scene_ret] = []
            decoding_results_scene_retrieved[target_scene_ret].append(scene_tc)

            # Look up catch_trial once per trial
            df_behavior_retrieval_subset = df_behavior_retrieval[df_behavior_retrieval["trl_nr"] == row["trl_nr"]]
            vals = df_behavior_retrieval_subset["catch_trial"].dropna()
            catch_trial_pressed = vals.iloc[0] if not vals.empty else np.nan

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
                    rows["behav_correct"].append(catch_trial_pressed)

                    if not sequence_struct_written:
                        rows["full_structured_sequence_object"].append(sequence_dict_obj_json)
                        rows["full_structured_sequence_category"].append(sequence_dict_cat_json)
                        sequence_struct_written = True
                    else:
                        rows["full_structured_sequence_object"].append(np.nan)
                        rows["full_structured_sequence_category"].append(np.nan)

    return decoding_results_scene_retrieved, sequence_struct_written


# =====================================================================
# Plotting functions
# =====================================================================

def plot_scene_decoding(
    decoding_results_scene_presented: dict,
    tr_window: np.ndarray,
    sub: str, cmask: str,
    plt_path: str,
):
    """Scene decoding timecourse per subject (presented scene cue)."""
    plt.figure(figsize=(10, 6))
    all_timecourses_scene = []

    for stim, timecourses in decoding_results_scene_presented.items():
        stim_avg = np.mean(timecourses, axis=0)
        all_timecourses_scene.extend(timecourses)
        plt.plot(tr_window, stim_avg, alpha=0.7, linewidth=2, label=stim)

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


def plot_object_decoding(
    decoding_results_object_retrieved: dict,
    tr_window: np.ndarray,
    sub: str, cmask: str,
    plt_path: str,
):
    """Object decoding timecourse per subject."""
    plt.figure(figsize=(10, 6))
    all_timecourses_object = []

    for obj_cat, timecourses in decoding_results_object_retrieved.items():
        obj_avg = np.mean(timecourses, axis=0)
        all_timecourses_object.extend(timecourses)
        plt.plot(tr_window, obj_avg, alpha=0.7, linewidth=2, label=obj_cat)

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


def plot_scene_retrieval_by_cue_and_position(
    decoding_results_scene_retrieved: dict,
    df_scenes: pd.DataFrame,
    tr_window: np.ndarray,
    sub: str, cmask: str,
    plt_path: str,
):
    """Scene retrieval decoding with color-coding by cue and position."""

    def extract_pos(s):
        try:
            return int(os.path.splitext(str(s))[0])
        except Exception:
            return np.nan

    plt.figure(figsize=(10, 6))
    all_timecourses_scene_ret = []

    df_scenes_pos = df_scenes.copy()
    df_scenes_pos['pos_idx'] = df_scenes_pos['seq_pos'].apply(extract_pos)

    scene_stats = (
        df_scenes_pos
        .groupby('sc_ret_targ')
        .agg(
            pos_idx_mean=('pos_idx', 'mean'),
            ret_cue_mode=('ret_cue', lambda x: x.value_counts().idxmax())
        )
    )

    pos_values = scene_stats['pos_idx_mean'].to_numpy()
    pos_values = pos_values[~np.isnan(pos_values)]
    if len(pos_values) > 0:
        pos_min, pos_max = pos_values.min(), pos_values.max()
    else:
        pos_min, pos_max = 0.0, 1.0

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

        if np.isnan(pos) or pos_max == pos_min:
            norm_pos = 0.5
        else:
            norm_pos = (pos - pos_min) / (pos_max - pos_min)

        if 'night' in cue:
            color = cmap_night(norm_pos)
        elif 'day' in cue:
            color = cmap_day(norm_pos)
        else:
            color = plt.cm.Greys(norm_pos)

        label = stim
        if not np.isnan(pos):
            label += f" (pos~{pos:.1f}, cue={cue})"

        plt.plot(tr_window, stim_avg, color=color, alpha=0.9, linewidth=2, label=label)

    if all_timecourses_scene_ret:
        grand_avg_scene_ret = np.mean(all_timecourses_scene_ret, axis=0)
        plt.plot(tr_window, grand_avg_scene_ret, color='black', linewidth=4, label='Grand average')

    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('TRs from onset')
    plt.ylabel('P(correct retrieved scene)')
    plt.title(
        f'Scene retrieval decoding (d_no4sc, subject {sub})\n'
        'Color = ret_cue (night: blue, day: red) - position (light early \u2192 dark late)'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(opj(plt_path, f'{sub}_ROI_{cmask}_scene_retrieval_decoding_d_no4sc_by_cue_and_position.png'))
    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():
    # -----------------------------------------------------------------
    # Parse CLI arguments
    # -----------------------------------------------------------------
    sub = parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None)
    default_roi = "vis"
    cmask = parse_roi_arg(sys.argv[2] if len(sys.argv) > 2 else None, default=default_roi)

    pre_post = "presleep"
    if len(sys.argv) > 3:
        pre_post_arg = sys.argv[3].lower().strip()
        if pre_post_arg in SESSION_CONFIG:
            pre_post = pre_post_arg
        else:
            print(f"WARNING: Unrecognised pre_post argument '{sys.argv[3]}'. Defaulting to 'presleep'.")

    cfg = SESSION_CONFIG[pre_post]
    print(f"[config] Running {pre_post} analysis")
    print(f"[config] Subject: {sub}, ROI: {cmask}, MRI session: {cfg['mri_session']}")

    # -----------------------------------------------------------------
    # Setup paths
    # -----------------------------------------------------------------
    root_folder = get_root(".my_hidden_root_file")
    path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
    path_masks = opj(root_folder, 'code', '3_reliability_voxel_selection', 'combine_loc1_and_loc3_masks')
    path_clf_dir = opj(os.path.dirname(os.path.abspath(__file__)), '4.1_train_visual_classifier', 'classifiers')

    TR = 1.5
    csmooth = 4
    temporal_smoothing_level = 3
    f_threshold = 30
    reliability_threshold = 70
    use_overall_classifier1_specific2 = 2

    mri_session = cfg["mri_session"]
    file_label = cfg["file_label"]
    output_prefix = cfg["output_folder_prefix"]

    sub_num = sub[1:]
    sub_mri = f"sub-{sub_num}"

    # -----------------------------------------------------------------
    # Load behavioral data
    # -----------------------------------------------------------------
    _, obj_enc_to_cat = load_localizer_behavior(root_folder, sub)
    df_behavior_enc, df_behavior_retrieval = load_main_task_behavior(root_folder, sub, cfg)

    # Build sequence dicts from encoding df
    sequence_dict_obj = build_sequence_dict_from_df_behavior_enc(df_behavior_enc)
    sequence_dict_cat = map_sequence_objects_to_categories(sequence_dict_obj, obj_enc_to_cat)

    if len(sequence_dict_obj['day']) == 0 or len(sequence_dict_obj['night']) == 0:
        raise ValueError("Could not build valid sequence_dict from encoding behavior data.")

    for dn in ("day", "night"):
        print(f"[sequence_dict] {dn}: {len(sequence_dict_cat[dn])} scenes")
        for sc, objs in sequence_dict_cat[dn].items():
            print(f"  {sc}: {len(objs)} objects -> {objs}")

    # Prepare retrieval df
    df_behavior_retrieval = prepare_retrieval_df(df_behavior_retrieval, obj_enc_to_cat)

    # -----------------------------------------------------------------
    # Load and preprocess fMRI data
    # -----------------------------------------------------------------
    mask_img = load_mask(path_masks, sub, cmask, reliability_threshold)
    data_cleaned = load_and_preprocess_fmri(
        path_fmriprep, sub_mri, mri_session,
        mask_img, csmooth, TR, CONFOUND_VARS,
    )

    # -----------------------------------------------------------------
    # Load classifiers
    # -----------------------------------------------------------------
    use_overall = (use_overall_classifier1_specific2 == 1)
    clf_scenes, clf_objects = load_classifiers(
        path_clf_dir, sub, cmask, csmooth, f_threshold,
        reliability_threshold, temporal_smoothing_level,
        use_overall=use_overall,
    )
    if use_overall:
        print(f"Loading GENERIC classifier")
    else:
        print(f"Loading SCENE + OBJECT classifiers (specific)")

    # -----------------------------------------------------------------
    # Select retrieval runs
    # -----------------------------------------------------------------
    retrieval_data_list = select_retrieval_runs(data_cleaned, cfg)

    # -----------------------------------------------------------------
    # Get retrieval trial subsets
    # -----------------------------------------------------------------
    df_objects = df_behavior_retrieval[df_behavior_retrieval['label_type'] == 'sc_no4obj'].copy()
    df_scenes = df_behavior_retrieval[df_behavior_retrieval['label_type'] == 'd_no4sc'].copy()

    # Build run-number -> data-index mapping
    run_nr_to_data_idx = build_run_nr_mapping(
        df_objects, df_scenes, cfg["n_retrieval_runs"]
    )

    # -----------------------------------------------------------------
    # Prepare output rows dict
    # -----------------------------------------------------------------
    tr_window = np.arange(0, 10)

    rows = {
        "subject": [],
        "run": [],
        "trial": [],
        "tr": [],
        "object_scene_retrieval": [],
        "retrieval_target": [],
        "retrieval_target_precise": [],
        "retrieval_cue": [],
        "retrieval_cue_position": [],
        "stimulus_class_from_classifier": [],
        "target_classifier_matched": [],
        "probability_from_classifier": [],
        "behav_correct": [],
        "full_structured_sequence_object": [],
        "full_structured_sequence_category": [],
    }

    sequence_dict_obj_json = json.dumps(sequence_dict_obj, ensure_ascii=False, sort_keys=True)
    sequence_dict_cat_json = json.dumps(sequence_dict_cat, ensure_ascii=False, sort_keys=True)
    sequence_struct_written = False

    # -----------------------------------------------------------------
    # Apply classifier to object retrieval data
    # -----------------------------------------------------------------
    decoding_results_scene_presented, decoding_results_object_retrieved, sequence_struct_written = \
        decode_object_retrieval(
            df_objects, df_behavior_retrieval, retrieval_data_list,
            run_nr_to_data_idx, clf_scenes, clf_objects, tr_window,
            sub, rows, sequence_dict_obj_json, sequence_dict_cat_json,
            sequence_struct_written,
        )

    # Save object-retrieval decoding outputs
    decoding_results_path = _output_dir('decoding_timecourses_data', output_prefix)
    np.save(
        opj(decoding_results_path,
            f'{sub}_ROI_{cmask}_decoding_timecourses_{file_label}_scene_presented_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'),
        decoding_results_scene_presented,
    )
    np.save(
        opj(decoding_results_path,
            f'{sub}_ROI_{cmask}_decoding_timecourses_{file_label}_object_retrieved_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'),
        decoding_results_object_retrieved,
    )
    np.save(
        opj(decoding_results_path,
            f'{sub}_ROI_{cmask}_decoding_timecourses_{file_label}_tr_window.npy'),
        tr_window,
    )

    # Plots: scene + object decoding
    plt_path = _output_dir('decoding_timecourses_plots', output_prefix)
    plot_scene_decoding(decoding_results_scene_presented, tr_window, sub, cmask, plt_path)
    plot_object_decoding(decoding_results_object_retrieved, tr_window, sub, cmask, plt_path)

    # -----------------------------------------------------------------
    # Apply classifier to scene retrieval data
    # -----------------------------------------------------------------
    decoding_results_scene_retrieved, sequence_struct_written = decode_scene_retrieval(
        df_scenes, df_behavior_retrieval, retrieval_data_list,
        run_nr_to_data_idx, clf_scenes, tr_window,
        sub, rows, sequence_dict_obj_json, sequence_dict_cat_json,
        sequence_struct_written,
    )

    # Save results table
    df_decoding_results = pd.DataFrame(rows)
    output_folder = _output_dir('decoding_results_tables_for_SODA', output_prefix)
    df_decoding_results.to_csv(
        opj(output_folder,
            f'{sub}_ROI_{cmask}_decoding_results_{file_label}_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.csv'),
        index=False,
    )

    # Save scene-retrieval decoding timecourses
    np.save(
        opj(decoding_results_path,
            f'{sub}_ROI_{cmask}_decoding_timecourses_{file_label}_scene_retrieved_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy'),
        decoding_results_scene_retrieved,
    )

    # Plot: scene retrieval by cue and position
    plot_scene_retrieval_by_cue_and_position(
        decoding_results_scene_retrieved, df_scenes, tr_window,
        sub, cmask, plt_path,
    )

    print(f"\n[DONE] {pre_post} retrieval decoding for {sub}, ROI={cmask}")


if __name__ == "__main__":
    main()
