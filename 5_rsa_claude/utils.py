# -*- coding: utf-8 -*-
"""
Shared utilities for the RSA pipeline.

Reuses core functions from folder 6 (parse_subject_id, behavioral loading,
functional data loading) and adds RSA-specific utilities for mask loading,
event extraction, and run selection.
"""

from __future__ import annotations

import glob
import json
import os
import re
from os.path import join as opj
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img

from config import (
    PATH_FMRIPREP,
    PATH_BEHAVIOR,
    PATH_RELIABILITY_MASKS,
    ROOT_FOLDER,
    RSA_SESSION_MAP,
    RSA_RELIABILITY_THRESHOLD,
    CONFOUND_COLUMNS,
    TR,
)


# =====================================================================
# Subject ID handling (same as folder 6)
# =====================================================================

def parse_subject_id(subj_arg: str) -> tuple[str, str, str]:
    """Parse a subject argument into consistent formats.

    Accepts: "13", "s13", "sub-13"
    Returns: (sub_behav, sub_num, sub_mri) e.g. ("s13", "13", "sub-13")
    """
    m = re.match(r"^sub-(\d+)$", subj_arg)
    if m:
        sub_num = m.group(1)
    elif re.match(r"^\d+$", subj_arg):
        sub_num = subj_arg
    elif re.match(r"^s(\d+)$", subj_arg):
        sub_num = subj_arg[1:]
    else:
        raise ValueError(
            f"Cannot parse subject ID '{subj_arg}'. "
            f"Expected formats: '13', 's13', or 'sub-13'."
        )
    return f"s{sub_num}", sub_num, f"sub-{sub_num}"


# =====================================================================
# Behavioral data loading
# =====================================================================

def load_localizer_category_mapping(sub_behav: str, sub_num: str) -> dict:
    """Load localizer behavioral data and return obj_enc_stim -> obj_cat mapping."""
    path = opj(
        ROOT_FOLDER, "behavior", sub_behav, "behav", "loc",
        f"s{sub_num}_df_loc1_w_HRF.csv",
    )
    files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"Localizer behavior file not found: {path}")
    df = pd.read_csv(files[0], sep=None, engine="python")
    mapping = df.groupby("obj_enc_stim", sort=False)["obj_cat"].first().to_dict()
    print(f"[loc] Loaded {len(mapping)} stimulus->category mappings from localizer.")
    return mapping


def _find_highest_step_folder(pattern: str) -> str:
    """Find the folder matching *pattern* with the highest step number."""
    folders = sorted(glob.glob(pattern))
    if not folders:
        raise FileNotFoundError(f"No folders matching: {pattern}")
    best = max(folders, key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    print(f"  Found {len(folders)} folder(s), using: {os.path.basename(best)}")
    return best


def load_retrieval_behavior(
    sub_behav: str,
    sub_num: str,
    session: str,
    obj_enc_to_cat: dict,
) -> pd.DataFrame:
    """Load behavioral data for a retrieval session.

    Parameters
    ----------
    session : str
        One of "presleep", "postsleep", "postsleep_3runs".

    Returns
    -------
    pd.DataFrame
        Retrieval behavior with obj_cat mapped and NaNs handled.
    """
    sess_cfg = RSA_SESSION_MAP[session]
    behav_nses = sess_cfg["behav_nses"]

    if session == "presleep":
        # Presleep uses nses_0 (same file as encoding session)
        nses_pattern = opj(
            ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
            f"*_{behav_nses}_step*",
        )
        print(f"[behavior:{session}] Searching: {nses_pattern}")
        best_folder = _find_highest_step_folder(nses_pattern)
        path_csv = opj(best_folder, f"output_df_sub{sub_num}_w_HRF.csv")
        df_best = pd.read_csv(path_csv, sep=",")

        # Check if highest-step folder has both encoding + retrieval label_types
        required = {"sc_enc_fix", "d_no4sc"}
        if "label_type" in df_best.columns:
            present = set(df_best["label_type"].dropna().astype(str).unique())
            if required.issubset(present):
                print(f"  Using highest step for retrieval.")
                df_ret = df_best.copy()
            else:
                print(f"  Highest step missing {required - present}; loading step0.")
                path_step0 = opj(
                    ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
                    f"sub{sub_num}_main_task_{behav_nses}_step0",
                    f"output_df_sub{sub_num}_w_HRF.csv",
                )
                df_ret = pd.read_csv(path_step0, sep=",")
        else:
            print(f"  No label_type column; using step0.")
            path_step0 = opj(
                ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
                f"sub{sub_num}_main_task_{behav_nses}_step0",
                f"output_df_sub{sub_num}_w_HRF.csv",
            )
            df_ret = pd.read_csv(path_step0, sep=",")

    elif session in ("postsleep", "postsleep_3runs"):
        nses_pattern = opj(
            ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
            f"sub{sub_num}_main_task_{behav_nses}_step*",
        )
        print(f"[behavior:{session}] Searching: {nses_pattern}")
        best_folder = _find_highest_step_folder(nses_pattern)
        path_csv = opj(best_folder, f"output_df_sub{sub_num}_w_HRF.csv")
        df_ret = pd.read_csv(path_csv, sep=",")
        print(f"  Loaded postsleep retrieval from: {path_csv}")
    else:
        raise ValueError(f"Unknown session: {session}")

    # Map object categories
    df_ret["obj_cat"] = df_ret["obj_ret_targ"].map(obj_enc_to_cat)

    # Forward-fill run_nr and trl_nr
    for col in ["trl_nr", "run_nr"]:
        if col in df_ret.columns:
            df_ret[col] = df_ret[col].ffill()

    # Drop rows with NaN label_type
    df_ret = df_ret.dropna(subset=["label_type"]).reset_index(drop=True)

    n_runs = df_ret["run_nr"].nunique() if "run_nr" in df_ret.columns else "?"
    print(f"  Loaded {len(df_ret)} rows across {n_runs} runs.")
    return df_ret


# =====================================================================
# Mask loading
# =====================================================================

def load_reliability_mask(
    sub_behav: str,
    roi: str,
    threshold: int = RSA_RELIABILITY_THRESHOLD,
) -> nib.Nifti1Image:
    """Load combined reliability mask for a subject and ROI.

    Expected path:
        {PATH_RELIABILITY_MASKS}/{sub_behav}_mask_{roi}_threshold_{threshold}_combined.nii
    """
    # Try sXX naming first (e.g. s13_mask_vis_...), then sub-XX (e.g. sub-13_mask_parietal_...)
    mask_dir = str(PATH_RELIABILITY_MASKS)
    fname = f"{sub_behav}_mask_{roi}_threshold_{threshold}_combined.nii"
    mask_path = opj(mask_dir, fname)
    if not os.path.exists(mask_path):
        sub_mri = f"sub-{sub_behav[1:]}"
        fname_alt = f"{sub_mri}_mask_{roi}_threshold_{threshold}_combined.nii"
        mask_path_alt = opj(mask_dir, fname_alt)
        if os.path.exists(mask_path_alt):
            mask_path = mask_path_alt
        else:
            raise FileNotFoundError(
                f"Reliability mask not found: tried {fname} and {fname_alt}"
            )
    mask_img = image.load_img(mask_path)
    n_voxels = int(mask_img.get_fdata().sum())
    print(f"[mask] Loaded {roi} mask for {sub_behav}: {n_voxels} voxels")
    return mask_img


def get_mni_mask(reference_img):
    """Load MNI152 brain mask and resample to match a reference image."""
    mni_mask = load_mni152_brain_mask()
    return resample_to_img(mni_mask, reference_img, interpolation="nearest")


def resample_mask_to_func(mask_img, func_img):
    """Resample a mask to match the functional data space if needed."""
    if mask_img.shape[:3] != func_img.shape[:3]:
        print(f"  Resampling mask {mask_img.shape[:3]} -> {func_img.shape[:3]}")
        return resample_to_img(mask_img, func_img, interpolation="nearest")
    return mask_img


# =====================================================================
# Functional data and confound loading
# =====================================================================

def _parse_run_id(filepath: str) -> str:
    """Extract a unique run identifier from a BIDS filename."""
    basename = os.path.basename(filepath)
    m = re.search(r"task-(mem[^_]+(?:_run-\d+)?)", basename)
    return m.group(1) if m else basename


def _handle_restarted_runs(
    paths: list[str],
    selected_indices: list[int],
    run_lengths: list[tuple[int, int]],
) -> list[int]:
    """Detect restarted runs (duplicate run IDs) and keep the later one."""
    run_id_to_indices: dict[str, list[int]] = {}
    for idx in selected_indices:
        rid = _parse_run_id(paths[idx])
        run_id_to_indices.setdefault(rid, []).append(idx)

    cleaned = []
    for rid, indices in sorted(run_id_to_indices.items()):
        if len(indices) > 1:
            kept = indices[-1]
            dropped = indices[:-1]
            print(f"  WARNING [RESTART]: Run ID '{rid}' has {len(indices)} files "
                  f"(indices {indices}). Keeping later one (idx={kept}, "
                  f"{dict(run_lengths)[kept]} TRs). "
                  f"Dropped: {[(d, dict(run_lengths)[d]) for d in dropped]}.")
            cleaned.append(kept)
        else:
            cleaned.append(indices[0])

    return sorted(cleaned)


def load_functional_data(
    sub_mri: str,
    session: str,
    path_fmriprep: str = PATH_FMRIPREP,
) -> tuple[list, list[int]]:
    """Load preprocessed BOLD runs for an RSA session.

    For presleep: loads all ses-04 runs, filters, keeps last 3 (retrieval).
    For postsleep: loads ses-06 runs, filters, keeps first N.

    Returns
    -------
    func_imgs : list of Nifti1Image
    selected_indices : list of int (original indices for confound alignment)
    """
    sess_cfg = RSA_SESSION_MAP[session]
    mri_session = sess_cfg["mri_session"]
    min_trs = sess_cfg["min_run_trs"]
    select_first_n = sess_cfg.get("select_first_n")
    retrieval_runs = sess_cfg.get("retrieval_runs")

    pattern = opj(
        path_fmriprep, sub_mri, mri_session, "func",
        f"{sub_mri}_{mri_session}_task-mem*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    paths = sorted(glob.glob(pattern), key=lambda f: os.path.basename(f))

    if not paths:
        raise FileNotFoundError(f"No functional files found: {pattern}")

    print(f"[func:{session}] Found {len(paths)} files:")
    for i, p in enumerate(paths):
        print(f"    [{i}] {os.path.basename(p)}")

    # Step 1: Read all run lengths
    run_lengths = [(i, nib.load(p).shape[-1]) for i, p in enumerate(paths)]
    print(f"  Run lengths: {run_lengths}")

    # Step 2: Filter by minimum TRs
    selected_indices = []
    for i, n_trs in run_lengths:
        if n_trs > min_trs:
            selected_indices.append(i)
        else:
            print(f"  Dropping run {i} ({os.path.basename(paths[i])}): "
                  f"{n_trs} TRs <= {min_trs}")

    print(f"  After filtering: {len(selected_indices)} runs (indices {selected_indices})")

    # Step 3: Detect and resolve restarted runs
    selected_indices = _handle_restarted_runs(paths, selected_indices, run_lengths)

    # Step 4: Select retrieval runs
    if retrieval_runs == "last_3":
        # Presleep: take last 3 runs (retrieval runs from ses-04)
        if len(selected_indices) < 3:
            raise ValueError(
                f"Expected >= 3 runs after filtering for presleep, "
                f"got {len(selected_indices)}"
            )
        print(f"  Presleep: taking last 3 of {len(selected_indices)} runs")
        selected_indices = selected_indices[-3:]
    elif select_first_n is not None and len(selected_indices) > select_first_n:
        print(f"  Keeping first {select_first_n} of {len(selected_indices)} runs")
        selected_indices = selected_indices[:select_first_n]

    print(f"  Final selected run indices: {selected_indices}")
    func_imgs = [image.load_img(paths[i]) for i in selected_indices]
    print(f"  Loaded {len(func_imgs)} runs into memory.")
    return func_imgs, selected_indices


def load_confounds(
    sub_mri: str,
    session: str,
    selected_indices: list[int],
    path_fmriprep: str = PATH_FMRIPREP,
) -> list[pd.DataFrame]:
    """Load confound TSVs for selected runs, subset to CONFOUND_COLUMNS."""
    sess_cfg = RSA_SESSION_MAP[session]
    mri_session = sess_cfg["mri_session"]

    pattern = opj(
        path_fmriprep, sub_mri, mri_session, "func",
        f"{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv",
    )
    paths = sorted(glob.glob(pattern), key=lambda f: os.path.basename(f))

    if not paths:
        raise FileNotFoundError(f"No confound files found: {pattern}")

    confs = [pd.read_csv(paths[i], sep="\t") for i in selected_indices]

    confs_clean = []
    for i, df in enumerate(confs):
        missing = [c for c in CONFOUND_COLUMNS if c not in df.columns]
        if missing:
            print(f"  WARNING: Run {selected_indices[i]} missing confounds: {missing}")
        cols_present = [c for c in CONFOUND_COLUMNS if c in df.columns]
        confs_clean.append(df[cols_present].fillna(0))

    return confs_clean


# =====================================================================
# RSA-specific event extraction
# =====================================================================

def extract_rsa_events_for_run(
    df_behavior: pd.DataFrame,
    run_nr: int,
    obj_enc_to_cat: dict,
) -> pd.DataFrame:
    """Extract events with item-level trial_types for RSA GLM.

    For object retrieval trials (sc_no4obj): trial_type = obj_cat (category)
    For scene retrieval trials (d_no4sc): trial_type = sc_ret_targ (scene name)

    Returns events DataFrame with columns: onset, duration, trial_type, modality
    """
    df_run = df_behavior[df_behavior["run_nr"] == run_nr].copy()

    events_list = []

    # Process each retrieval trial type
    for _, row in df_run.iterrows():
        label_type = str(row.get("label_type", ""))

        if label_type == "sc_no4obj":
            # Object retrieval: use category as condition
            obj_target = row.get("obj_ret_targ")
            if pd.isna(obj_target):
                continue
            cat = obj_enc_to_cat.get(obj_target)
            if cat is None:
                continue
            trial_type = cat
            modality = "object"
        elif label_type == "d_no4sc":
            # Scene retrieval: use scene name as condition
            sc_target = row.get("sc_ret_targ")
            if pd.isna(sc_target):
                continue
            trial_type = str(sc_target)
            modality = "scene"
        else:
            continue

        onset = pd.to_numeric(row.get("t_onset"), errors="coerce")
        duration = pd.to_numeric(row.get("ret_resp"), errors="coerce")

        if pd.isna(onset):
            continue
        if pd.isna(duration) or duration <= 0:
            duration = 3.5  # fallback: fixed duration

        events_list.append({
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "modality": modality,
        })

    events = pd.DataFrame(events_list)
    if events.empty:
        print(f"  [run {run_nr}] WARNING: No RSA events found!")
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "modality"])

    n_obj = (events["modality"] == "object").sum()
    n_sc = (events["modality"] == "scene").sum()
    n_conds = events["trial_type"].nunique()
    print(f"  [run {run_nr}] RSA events: {n_obj} object, {n_sc} scene, "
          f"{n_conds} unique conditions")
    return events


def get_conditions_for_modality(
    df_behavior: pd.DataFrame,
    obj_enc_to_cat: dict,
    modality: str,
) -> list[str]:
    """Get sorted list of unique condition names for a modality.

    Parameters
    ----------
    modality : str
        "object" (returns 10 category names) or "scene" (returns 10 scene names).
    """
    if modality == "object":
        obj_targets = df_behavior.loc[
            df_behavior["label_type"] == "sc_no4obj", "obj_ret_targ"
        ].dropna()
        cats = obj_targets.map(obj_enc_to_cat).dropna().unique()
        return sorted(cats)
    elif modality == "scene":
        sc_targets = df_behavior.loc[
            df_behavior["label_type"] == "d_no4sc", "sc_ret_targ"
        ].dropna().unique()
        return sorted([str(s) for s in sc_targets])
    else:
        raise ValueError(f"Unknown modality: {modality}")


# =====================================================================
# Sequence structure extraction
# =====================================================================

def load_encoding_behavior(sub_behav: str, sub_num: str) -> pd.DataFrame:
    """Load encoding behavioral data (nses_0) to extract sequence structure."""
    nses_pattern = opj(
        ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
        f"*_nses_0_step*",
    )
    best_folder = _find_highest_step_folder(nses_pattern)
    path_csv = opj(best_folder, f"output_df_sub{sub_num}_w_HRF.csv")
    df = pd.read_csv(path_csv, sep=",")
    return df


def build_sequence_structure(
    df_enc: pd.DataFrame,
    obj_enc_to_cat: dict,
) -> dict:
    """Build day/night sequence structure from encoding behavioral data.

    Uses the dn_cue column (day/night context), sc_enc_stim (scene),
    and obj_enc_stim (object) to determine which scenes and object
    categories belong to which context.

    Returns
    -------
    dict with keys:
        "scene_to_context": {scene_name: "day"|"night"}
        "category_to_context": {category_name: "day"|"night"}
        "scene_to_categories": {scene_name: [category_names]}
        "context_to_scenes": {"day": [scenes], "night": [scenes]}
        "context_to_categories": {"day": [categories], "night": [categories]}
    """
    result = {
        "scene_to_context": {},
        "category_to_context": {},
        "scene_to_categories": {},
        "context_to_scenes": {"day": [], "night": []},
        "context_to_categories": {"day": [], "night": []},
    }

    # Check required columns
    required_cols = ["dn_cue", "sc_enc_stim", "obj_enc_stim"]
    missing = [c for c in required_cols if c not in df_enc.columns]
    if missing:
        print(f"  WARNING: Missing columns for sequence structure: {missing}")
        return result

    # Build scene -> context mapping
    for _, row in df_enc.iterrows():
        cue = row.get("dn_cue")
        scene = row.get("sc_enc_stim")
        obj = row.get("obj_enc_stim")

        if pd.isna(cue) or pd.isna(scene):
            continue

        dn = None
        cue_str = str(cue).lower()
        if "day" in cue_str:
            dn = "day"
        elif "night" in cue_str:
            dn = "night"
        if dn is None:
            continue

        scene_str = str(scene)
        result["scene_to_context"][scene_str] = dn

        # Map objects to categories and link to scenes
        if pd.notna(obj):
            cat = obj_enc_to_cat.get(str(obj))
            if cat is not None:
                result["category_to_context"][cat] = dn
                if scene_str not in result["scene_to_categories"]:
                    result["scene_to_categories"][scene_str] = []
                if cat not in result["scene_to_categories"][scene_str]:
                    result["scene_to_categories"][scene_str].append(cat)

    # Build context -> items lists
    for scene, ctx in result["scene_to_context"].items():
        if scene not in result["context_to_scenes"][ctx]:
            result["context_to_scenes"][ctx].append(scene)
    for cat, ctx in result["category_to_context"].items():
        if cat not in result["context_to_categories"][ctx]:
            result["context_to_categories"][ctx].append(cat)

    print(f"  Sequence structure: "
          f"day={len(result['context_to_scenes']['day'])} scenes / "
          f"{len(result['context_to_categories']['day'])} categories, "
          f"night={len(result['context_to_scenes']['night'])} scenes / "
          f"{len(result['context_to_categories']['night'])} categories")

    return result


# =====================================================================
# Condition list I/O
# =====================================================================

def save_conditions(conditions: list[str], path: str) -> None:
    """Save condition list as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"  Saved {len(conditions)} conditions to {path}")


def load_conditions(path: str) -> list[str]:
    """Load condition list from JSON."""
    with open(path) as f:
        return json.load(f)
