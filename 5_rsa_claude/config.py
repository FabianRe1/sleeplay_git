# -*- coding: utf-8 -*-
"""
Shared configuration for the RSA pipeline.

Extends the core project config from folder 6 with RSA-specific settings.
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
    ROOT_FOLDER = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Core paths (same as folder 6)
# ---------------------------------------------------------------------------
PATH_FMRIPREP: str = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
PATH_BEHAVIOR: Path = ROOT_FOLDER / "behavior"

# Output root — all results go under the RSA script directory
SCRIPT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Acquisition parameters
# ---------------------------------------------------------------------------
TR: float = 1.5  # seconds

# ---------------------------------------------------------------------------
# Confound variables (18 — same as folder 6)
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
NOISE_MODEL: str = "ar1"       # AR1 for RSA (needed for proper residuals)

# ---------------------------------------------------------------------------
# RSA session definitions
# ---------------------------------------------------------------------------
RSA_SESSION_MAP: dict[str, dict] = {
    "presleep": {
        "mri_session": "ses-04",
        "behav_nses": "nses_0",
        "description": "Presleep retrieval (last 3 runs of ses-04)",
        "min_run_trs": 120,
        "retrieval_runs": "last_3",   # take last 3 of all valid runs
    },
    "postsleep": {
        "mri_session": "ses-06",
        "behav_nses": "nses_2",
        "description": "Postsleep retrieval — all 6 runs (Day 2)",
        "min_run_trs": 120,
        "select_first_n": 6,
    },
    "postsleep_3runs": {
        "mri_session": "ses-06",
        "behav_nses": "nses_2",
        "description": "Postsleep retrieval — first 3 runs only (matched to presleep)",
        "min_run_trs": 120,
        "select_first_n": 3,
    },
}

# ---------------------------------------------------------------------------
# RSA ROI definitions
# ---------------------------------------------------------------------------
RSA_ROIS: list[str] = ["vis", "hpc", "mtl", "parietal"]
RSA_RELIABILITY_THRESHOLD: int = 70

# Path to combined reliability masks from localizer sessions
PATH_RELIABILITY_MASKS: Path = (
    ROOT_FOLDER / "sleeplay_git" / "3_reliability_voxel_selection"
    / "combine_loc1_and_loc3_masks"
)

# ---------------------------------------------------------------------------
# Crossnobis noise estimation
# ---------------------------------------------------------------------------
NOISE_METHOD: str = "shrinkage_eye"  # rsatoolbox.data.noise.prec_from_residuals
