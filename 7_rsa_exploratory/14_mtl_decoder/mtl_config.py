# -*- coding: utf-8 -*-
"""Shared settings for the MTL decoder exploration (14_mtl_decoder).

Model selection on the LOCALIZER only. Nothing here reads main-task data.
Everything is written to 7_rsa_exploratory/outputs/14_mtl_decoder/.
"""

import glob
import os
import re
import sys
from os.path import join as opj

HERE = os.path.dirname(os.path.abspath(__file__))
RSA_DIR = os.path.dirname(HERE)                         # 7_rsa_exploratory/
CODE_ROOT = os.path.dirname(RSA_DIR)                    # sleeplay_git/
ROOT_FOLDER = os.path.dirname(CODE_ROOT)                # Sleeplay_Hummel/

OUT_ROOT = opj(RSA_DIR, "outputs", "14_mtl_decoder")
CACHE_DIR = opj(OUT_ROOT, "cache")
RESULT_DIR = opj(OUT_ROOT, "results")
SUMMARY_DIR = opj(OUT_ROOT, "summary")

PATH_FMRIPREP = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
PATH_BEHAVIOR = opj(ROOT_FOLDER, "behavior")
PATH_REL_MASKS = opj(CODE_ROOT, "3_reliability_voxel_selection", "combine_loc1_and_loc3_masks")
PATH_ANAT_MASKS = opj(CODE_ROOT, "2_get_threshold_roi", "localizer_getThresholdedROI",
                      "thresholded_roi_output")
PATH_EXISTING_CV = opj(CODE_ROOT, "4_decoding_claude_retrieval", "4.1_train_visual_classifier",
                       "df_classifier_results_csv")

TR = 1.5
SMOOTHING_LEVELS = [0, 2, 4]            # mm FWHM; 0 = none
REL_THRESHOLDS = [10, 30, 50, 70, 90]
ANAT_SIGNAL_THRESHOLD = 30              # the T-30 anatomical masks are the base of the reliability masks
ANAT_ROIS = {"mtl": "MTL", "hpc": "HPC", "phc": "PHPC", "erh": "ERH"}
LOC_SESSIONS = [("loc1", "ses-02"), ("loc2", "ses-03")]

# Participants whose combined masks are older than their reliability maps
STALE_MASK_SUBJECTS = ["s32", "s36", "s40", "s41", "s42", "s44", "s47"]


def subjects_with_current_results():
    """Participants (sXX) that have a current MTL cross-validation table."""
    files = glob.glob(opj(PATH_EXISTING_CV, "only_scenes1_objects2_both3_2_s*_Mask_mtl_*.csv"))
    subs = sorted({re.search(r"both3_2_(s\d+)_Mask", os.path.basename(f)).group(1) for f in files})
    return subs


def add_pipeline_to_path():
    """Make the paper-pipeline modules importable (read-only use)."""
    for p in (opj(CODE_ROOT, "paper_pipeline"), opj(CODE_ROOT, "paper_pipeline", "B_decoding")):
        if p not in sys.path:
            sys.path.insert(0, p)
