# -*- coding: utf-8 -*-
"""Shared settings for the exploratory RSA (folder 7_rsa_exploratory).

Everything this folder writes goes to ``outputs/`` next to this file.
Nothing here writes to ``paper_pipeline/``, ``behavior/`` or the derivatives.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from os.path import join as opj

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)                       # sleeplay_git/
ROOT_FOLDER = os.path.dirname(CODE_ROOT)                # Sleeplay_Hummel/
OUT_ROOT = opj(HERE, "outputs")

PATH_FMRIPREP = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
PATH_MASKS = opj(CODE_ROOT, "3_reliability_voxel_selection", "combine_loc1_and_loc3_masks")
PATH_DECODING = opj(CODE_ROOT, "4_decoding_claude_retrieval")

TR = 1.5
SMOOTHING_FWHM = 4
RELIABILITY_THRESHOLD = 70
TR_WINDOW = list(range(0, 10))        # TRs after cue onset that are stored per trial

# ROIs: the three thesis ROIs first, then the rest of the decoding set + hpc
ROIS = [
    "vis", "mtl", "parietal_precuneus",
    "hpc", "parietal", "precuneus", "cortical_memory", "superior_frontal",
    "posterior_cingulate", "rsc", "medial_ofc", "rostral_acc", "caudal_acc",
]

SESSIONS = ["presleep", "postsleep"]


def import_decoding_module():
    """Import 4.4 (validated behaviour loading and run selection) as a module.

    4.4 puts ``4_decoding_claude`` on sys.path itself and imports ``utils``
    from there, so this folder must not contain a module called ``utils``.
    """
    path = opj(PATH_DECODING, "4.4_transfer_visual_classifier_retrieval.py")
    spec = importlib.util.spec_from_file_location("decoding_4_4", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["decoding_4_4"] = mod
    spec.loader.exec_module(mod)
    return mod
