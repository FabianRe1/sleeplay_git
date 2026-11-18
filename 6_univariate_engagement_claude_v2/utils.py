# -*- coding: utf-8 -*-
"""
Shared utilities for the univariate engagement GLM pipeline (v2).

Functions for subject ID parsing, behavioral data loading,
confound loading, functional data loading, and design matrix helpers.
"""

from __future__ import annotations

import glob
import gc
import os
import re
from os.path import join as opj
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img

from config import (
    PATH_FMRIPREP,
    PATH_BEHAVIOR,
    ROOT_FOLDER,
    SESSION_MAP,
    CONFOUND_COLUMNS,
    TR,
)


# =====================================================================
# Subject ID handling
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
    """Load behavioral data for a retrieval session (encoding or postsleep).

    Parameters
    ----------
    session : str
        One of "encoding" or "postsleep".

    Returns
    -------
    pd.DataFrame
        Retrieval behavior with obj_cat mapped and NaNs handled.
    """
    sess_cfg = SESSION_MAP[session]
    behav_nses = sess_cfg["behav_nses"]

    if session == "encoding":
        # For encoding: use nses_0, with fallback logic for step folders
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
                print(f"  Using highest step for both encoding+retrieval.")
                df_ret = df_best.copy()
            else:
                print(f"  Highest step missing {required - present}; loading step0 for retrieval.")
                path_step0 = opj(
                    ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
                    f"sub{sub_num}_main_task_{behav_nses}_step0",
                    f"output_df_sub{sub_num}_w_HRF.csv",
                )
                df_ret = pd.read_csv(path_step0, sep=",")
        else:
            print(f"  No label_type column; using step0 for retrieval.")
            path_step0 = opj(
                ROOT_FOLDER, "behavior", sub_behav, "behav", "main",
                f"sub{sub_num}_main_task_{behav_nses}_step0",
                f"output_df_sub{sub_num}_w_HRF.csv",
            )
            df_ret = pd.read_csv(path_step0, sep=",")

    elif session in ("postsleep", "postsleep_3runs"):
        # For postsleep (both variants): use nses_2
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
# Functional data and confound loading
# =====================================================================

def _parse_run_id(filepath: str) -> str:
    """Extract a unique run identifier from a BIDS filename.

    Uses the full task + run combination so that different runs of the same
    task (e.g., task-mem1_run-01, task-mem1_run-02) are NOT treated as
    duplicates.  Only truly restarted runs (same task AND run number
    appearing twice) will be grouped.

    Examples:
        ...task-mem1_run-03_space-...  → 'mem1_run-03'
        ...task-mem03_space-...        → 'mem03'          (no run entity)
    """
    basename = os.path.basename(filepath)
    # Try task + run first (e.g., task-mem1_run-03)
    m = re.search(r"task-(mem[^_]+(?:_run-\d+)?)", basename)
    return m.group(1) if m else basename


def _handle_restarted_runs(
    paths: list[str],
    selected_indices: list[int],
    run_lengths: list[tuple[int, int]],
) -> list[int]:
    """Detect restarted runs (duplicate run IDs) and keep the later one.

    A restarted run produces two files for the same task block. The later
    file (higher index in sorted file list) is the correct one because it
    was the successful re-run.

    Returns the updated selected_indices with duplicates resolved.
    """
    # Group selected indices by their run ID
    run_id_to_indices: dict[str, list[int]] = {}
    for idx in selected_indices:
        rid = _parse_run_id(paths[idx])
        run_id_to_indices.setdefault(rid, []).append(idx)

    cleaned = []
    for rid, indices in sorted(run_id_to_indices.items()):
        if len(indices) > 1:
            # Multiple files with same run ID — keep the later one
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


# Expected number of task runs after filtering resting state (<=120 TRs)
_EXPECTED_TASK_RUNS = {
    "encoding": 6,          # 3 encoding + 3 retrieval blocks
    "postsleep": 6,         # 6 retrieval blocks
    "postsleep_3runs": 6,   # same session as postsleep; select_first_n trims to 3
}


def load_functional_data(
    sub_mri: str,
    session: str,
    path_fmriprep: str = PATH_FMRIPREP,
) -> tuple[list, list[int]]:
    """Load preprocessed BOLD runs for a session.

    Pipeline:
      1. Find all files matching the BIDS glob pattern.
      2. Drop runs with ≤120 TRs (resting state / calibration scans).
      3. Detect restarted runs (duplicate run IDs) → keep the later file.
      4. Validate the surviving count against expectations.
      5. Optionally take the first N runs by position (select_first_n).

    Returns
    -------
    func_imgs : list of Nifti1Image
        Functional images for valid runs.
    selected_indices : list of int
        Original run indices that were selected (needed for confound alignment).
    """
    sess_cfg = SESSION_MAP[session]
    mri_session = sess_cfg["mri_session"]
    min_trs = sess_cfg["min_run_trs"]
    select_first_n = sess_cfg.get("select_first_n")

    pattern = opj(
        path_fmriprep, sub_mri, mri_session, "func",
        f"{sub_mri}_{mri_session}_task-mem*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    paths = sorted(glob.glob(pattern), key=lambda f: os.path.basename(f))

    if not paths:
        raise FileNotFoundError(f"No functional files found: {pattern}")

    # Log all filenames for traceability
    print(f"[func:{session}] Found {len(paths)} files:")
    for i, p in enumerate(paths):
        print(f"    [{i}] {os.path.basename(p)}")

    # Step 1: Read all run lengths
    run_lengths = [(i, nib.load(p).shape[-1]) for i, p in enumerate(paths)]
    print(f"  Run lengths: {run_lengths}")

    # Step 2: Filter by minimum TRs (drop resting state ≤120)
    selected_indices = []
    for i, n_trs in run_lengths:
        if n_trs > min_trs:
            selected_indices.append(i)
        else:
            print(f"  Dropping run {i} ({os.path.basename(paths[i])}): "
                  f"{n_trs} TRs <= {min_trs} (resting state / too short).")

    n_after_filter = len(selected_indices)
    print(f"  After filtering: {n_after_filter} runs (indices {selected_indices})")

    # Step 3: Detect and resolve restarted runs
    selected_indices = _handle_restarted_runs(paths, selected_indices, run_lengths)
    n_after_restart = len(selected_indices)

    # Step 4: Validate against expected count
    base_session = "postsleep" if session == "postsleep_3runs" else session
    expected = _EXPECTED_TASK_RUNS.get(base_session)
    if expected is not None and n_after_restart != expected:
        print(f"  *** FLAG: {sub_mri} {session}: expected {expected} task runs "
              f"after filtering, got {n_after_restart}! "
              f"Indices: {selected_indices}. "
              f"This is UNCONVENTIONAL — check data manually.")

    # Step 5: Take first N valid runs by position (if requested)
    if select_first_n is not None and len(selected_indices) > select_first_n:
        print(f"  Keeping first {select_first_n} of {len(selected_indices)} "
              f"valid runs: indices {selected_indices[:select_first_n]}")
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
    sess_cfg = SESSION_MAP[session]
    mri_session = sess_cfg["mri_session"]

    pattern = opj(
        path_fmriprep, sub_mri, mri_session, "func",
        f"{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv",
    )
    paths = sorted(glob.glob(pattern), key=lambda f: os.path.basename(f))

    if not paths:
        raise FileNotFoundError(f"No confound files found: {pattern}")

    confs = [pd.read_csv(paths[i], sep="\t") for i in selected_indices]

    # Subset to columns of interest, fill NaN with 0
    confs_clean = []
    for i, df in enumerate(confs):
        missing = [c for c in CONFOUND_COLUMNS if c not in df.columns]
        if missing:
            print(f"  WARNING: Run {selected_indices[i]} missing confounds: {missing}")
        cols_present = [c for c in CONFOUND_COLUMNS if c in df.columns]
        confs_clean.append(df[cols_present].fillna(0))

    return confs_clean


def get_mni_mask(reference_img):
    """Load MNI152 brain mask and resample to match a reference image."""
    mni_mask = load_mni152_brain_mask()
    return resample_to_img(mni_mask, reference_img, interpolation="nearest")


# =====================================================================
# Event extraction from behavioral data
# =====================================================================

def compute_mean_remembered_rt(df_behavior: pd.DataFrame) -> float:
    """Compute mean response time across all remembered (button-press) trials.

    Looks at q_mark_press rows across all runs for this subject/session.
    Used as the duration for non-remembered trials where the subject
    timed out without pressing.
    """
    press_mask = df_behavior["label_type"] == "q_mark_press"
    rts = pd.to_numeric(df_behavior.loc[press_mask, "ret_resp"], errors="coerce").dropna()

    if len(rts) == 0:
        raise ValueError("No q_mark_press trials found — cannot compute mean remembered RT.")

    mean_rt = float(rts.mean())
    print(f"  Mean remembered RT: {mean_rt:.3f}s (from {len(rts)} trials)")
    return mean_rt


def extract_events_for_run(
    df_behavior: pd.DataFrame,
    run_nr: int,
    nonremem_duration: Optional[float] = None,
) -> pd.DataFrame:
    """Extract remembered/not_remembered events for a single run.

    Parameters
    ----------
    nonremem_duration : float, optional
        If provided, use this fixed duration (in seconds) for all
        not_remembered trials instead of the ITI/timeout duration.
        Typically set to the mean remembered RT for the subject.

    Returns a nilearn-compatible events DataFrame with columns:
    onset, duration, trial_type.
    """
    df_run = df_behavior[df_behavior["run_nr"] == run_nr].copy()

    # Check for cue trials
    has_cues = df_run["label_type"].isin(["sc_order_cue", "obj_order_cue"]).any()
    if not has_cues:
        print(f"  [run {run_nr}] No cue trials found. Returning empty events.")
        return pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # Fill missing ret_resp with ITIs (timeout = not remembered)
    # This ensures all trials have a duration; non-remembered durations
    # are overridden below if nonremem_duration is provided.
    if "ITIs" in df_run.columns:
        ret_missing = df_run["ret_resp"].isna() | (df_run["ret_resp"].astype(str).str.strip() == "")
        iti_has_val = ~(df_run["ITIs"].isna() | (df_run["ITIs"].astype(str).str.strip() == ""))
        df_run.loc[ret_missing & iti_has_val, "ret_resp"] = df_run.loc[ret_missing & iti_has_val, "ITIs"]

    # Get q_mark rows and the row immediately following each
    df_sorted = df_run.sort_index()
    q_mark_mask = df_sorted["label_type"] == "q_mark"
    next_row_mask = q_mark_mask.shift(1, fill_value=False)
    df_qm = df_sorted[q_mark_mask | next_row_mask].copy()

    # Classify: q_mark_press -> remembered, ret_fix -> not_remembered
    df_qm["remembered"] = np.select(
        [
            df_qm["label_type"].eq("q_mark_press"),
            df_qm["label_type"].eq("ret_fix"),
        ],
        ["remembered", "not_remembered"],
        default=pd.NA,
    )

    # Back-fill from response row to q_mark row
    df_qm["ret_resp"] = df_qm["ret_resp"].bfill()
    df_qm["remembered"] = df_qm["remembered"].bfill()

    # Keep only q_mark rows (now enriched with response info)
    df_qm = df_qm[df_qm["label_type"] == "q_mark"].reset_index(drop=True)

    events = pd.DataFrame({
        "onset": pd.to_numeric(df_qm["t_onset"], errors="coerce"),
        "duration": pd.to_numeric(df_qm["ret_resp"], errors="coerce"),
        "trial_type": df_qm["remembered"].astype(str),
    }).dropna(subset=["onset", "duration"])

    # Override non-remembered durations with mean remembered RT
    if nonremem_duration is not None:
        not_rem_mask = events["trial_type"] == "not_remembered"
        n_replaced = not_rem_mask.sum()
        if n_replaced > 0:
            old_durations = events.loc[not_rem_mask, "duration"]
            events.loc[not_rem_mask, "duration"] = nonremem_duration
            print(f"  [run {run_nr}] Replaced {n_replaced} non-remembered durations: "
                  f"mean {old_durations.mean():.3f}s -> {nonremem_duration:.3f}s")

    n_rem = (events["trial_type"] == "remembered").sum()
    n_not = (events["trial_type"] == "not_remembered").sum()
    print(f"  [run {run_nr}] Events: {n_rem} remembered, {n_not} not_remembered.")
    return events


# =====================================================================
# Design matrix helpers
# =====================================================================

def pad_to_global(run_idx: int, vec, lengths: list[int]) -> pd.Series:
    """Pad a run-level regressor into a global (concatenated) time axis."""
    vec = pd.Series(np.asarray(vec).squeeze()).reset_index(drop=True)
    total_len = sum(lengths)
    left = sum(lengths[:run_idx])
    right = total_len - left - lengths[run_idx]
    return pd.concat(
        [pd.Series(np.zeros(left)), vec, pd.Series(np.zeros(right))],
        ignore_index=True,
    )


def build_padded_drift_regressors(
    design_matrices: list[pd.DataFrame],
    lengths: list[int],
    drift_prefix: str = "drift",
) -> pd.DataFrame:
    """Extract drift regressors from per-run design matrices and pad globally.

    Drops the 'constant' column (handled separately via run intercepts).
    """
    total_len = sum(lengths)
    blocks = []

    for ri, dm in enumerate(design_matrices):
        drift_cols = [c for c in dm.columns if c.startswith(drift_prefix)]
        if "constant" in dm.columns:
            drift_cols = [c for c in drift_cols if c != "constant"]

        for c in drift_cols:
            padded = pad_to_global(ri, dm[c], lengths)
            blocks.append(padded.rename(f"{c}_run{ri + 1}"))

    if blocks:
        return pd.concat(blocks, axis=1).reset_index(drop=True)
    return pd.DataFrame(index=range(total_len))
