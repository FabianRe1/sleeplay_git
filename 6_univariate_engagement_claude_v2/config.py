# -*- coding: utf-8 -*-
"""
Shared configuration for the univariate engagement GLM pipeline (v2).

All paths, constants, and subject/session definitions live here so that
individual analysis scripts remain DRY.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (detected via hidden marker file, with cluster fallback)
# ---------------------------------------------------------------------------
try:
    from groo.groo import get_root
    ROOT_FOLDER: Path = Path(get_root(".my_hidden_root_file"))
except Exception:
    # Fallback: assume the code lives at <root>/sleeplay_git/6_univariate_engagement_v2
    ROOT_FOLDER = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------------
PATH_FMRIPREP: str = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
PATH_BEHAVIOR: Path = ROOT_FOLDER / "behavior"
PATH_MASKS: Path = ROOT_FOLDER / "sleeplay_git" / "2_get_threshold_roi" / "localizer_getThresholdedROI" / "thresholded_roi_output"

# Output root — all results go under the v2 script directory
SCRIPT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Acquisition parameters
# ---------------------------------------------------------------------------
TR: float = 1.5  # seconds

# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------
# Maps a human-readable session name to (MRI session ID, behavioral nses key)
SESSION_MAP: dict[str, dict] = {
    "encoding": {
        "mri_session": "ses-04",
        "behav_nses": "nses_0",
        "description": "Main task encoding (Day 1)",
        "min_run_trs": 120,       # drop runs with <=120 TRs (resting state)
        "select_first_n": None,   # no cap — keep all valid runs (expect 6)
    },
    "postsleep": {
        "mri_session": "ses-06",
        "behav_nses": "nses_2",
        "description": "Postsleep retrieval — all 6 runs (Day 2)",
        "min_run_trs": 120,        # drop runs with <=120 TRs (resting state)
        "select_first_n": 6,       # take the first 6 valid runs by position
    },
    "postsleep_3runs": {
        "mri_session": "ses-06",
        "behav_nses": "nses_2",
        "description": "Postsleep retrieval — first 3 runs only (matched to encoding)",
        "min_run_trs": 120,        # drop runs with <=120 TRs (resting state)
        "select_first_n": 3,       # first 3 valid runs by position
    },
}

# All valid session names (used for argparse choices)
ALL_SESSIONS: list[str] = list(SESSION_MAP.keys())

# ---------------------------------------------------------------------------
# Confound variables (18 — matches the reference LCI pipeline)
# ---------------------------------------------------------------------------
CONFOUND_COLUMNS: list[str] = [
    "global_signal",
    "framewise_displacement",
    "trans_x", "trans_y", "trans_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x", "rot_y", "rot_z",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05",
]

# ---------------------------------------------------------------------------
# GLM parameters
# ---------------------------------------------------------------------------
SMOOTHING_FWHM: float = 4.0    # mm, applied at first level
HIGH_PASS: float = 1.0 / 128   # Hz (128 s period)
HRF_MODEL: str = "spm"
DRIFT_MODEL: str = "cosine"

# Dead-period trimming: how many TRs to keep after the last task event
TRIM_BUFFER_TRS: int = 10

# ---------------------------------------------------------------------------
# Second-level parameters
# ---------------------------------------------------------------------------
GROUP_SMOOTHING_FWHM: float | None = None  # no additional smoothing at group level
N_PERMUTATIONS: int = 5000
CLUSTER_THRESHOLD: int = 10    # minimum cluster size for tables
T_CUTOFF_UNCORR: float = 3.1  # |t| cutoff for cluster tables

# ---------------------------------------------------------------------------
# ROI definitions (Harvard-Oxford atlas)
# ---------------------------------------------------------------------------
ROI_DEFINITIONS: dict[str, dict] = {
    "hippocampus": {
        "atlas": "sub-maxprob-thr25-2mm",  # subcortical
        "labels": ["Hippocampus"],
    },
    "parietal": {
        "atlas": "cort-maxprob-thr25-2mm",  # cortical
        "labels": [
            "Precuneous Cortex",
            "Cingulate Gyrus, posterior division",
            "Angular Gyrus",
        ],
    },
}
