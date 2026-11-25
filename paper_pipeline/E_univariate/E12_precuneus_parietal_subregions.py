#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E12 - Precuneus and parietal cortex in finer subdivisions, in native space.

E7 (main model `ret_press`) and E10 (single trials matched for search time)
use the anatomical regions of B1, in which the precuneus and the inferior
parietal cortex are one region each, both hemispheres together. This step
repeats both analyses, unchanged, on finer subdivisions taken from each
participant's own FreeSurfer parcellation, to see where a change between the
three retrievals sits.

Regions (voxels inside the brain mask, at least 10):
  Desikan-Killiany labels of fMRIPrep (func, space-T1w, aparcaseg), per hemisphere:
      PREC_L / _R        precuneus                  (1025 / 2025)
      INFPAR_L / _R      inferior parietal          (1008 / 2008)
      SUPPAR_L / _R      superior parietal          (1029 / 2029)
      SUPRAMARG_L / _R   supramarginal              (1031 / 2031)
      RSC_L / _R         isthmus of the cingulate   (1010 / 2010)
      PCC_L / _R         posterior cingulate        (1023 / 2023)
    and both hemispheres together: PRECUNEUS, PARIETAL (= inferior parietal, as in B1),
    PARIETAL_PRECUNEUS, SUPPAR, SUPRAMARG, RSC, PCC, and CAUDAL_ACC for reference.
  Destrieux labels (FreeSurfer aparc.a2009s+aseg.mgz, resampled to the BOLD grid
  with nearest neighbour; left 11100 + n, right 12100 + n), both hemispheres:
      D_PRECUNEUS_G 30, D_SUBPARIETAL_S 72, D_PARIETOOCC_S 66, D_MARGINAL_S 47,
      D_PCC_DORSAL_G 9, D_PCC_VENTRAL_G 10, D_SUPPARIETAL_G 27, D_INTRAPARIETAL_S 57,
      D_ANGULAR_G 25, D_SUPRAMARGINAL_G 26

Reliable voxels only. Every region is also computed on its reliable voxels:
<region>__r70 and <region>__r30. Reliable = inside the combined localizer
reliability mask of B3 (even/odd reliability of the category responses in the
localizers; the masks keep the most reliable 70 / 30 percent of an ROI per
localizer and are joined over both localizers, which leaves about 80 / 53 percent
of the voxels) of any of precuneus, parietal (inferior parietal), rsc,
posterior_cingulate or caudal_acc. 70 is the threshold the decoding uses.
Superior parietal and supramarginal cortex have no such mask; a reliable version
is written only where at least 30 voxels remain.

Smoothing. E12_SMOOTH=<mm> smooths the BOLD data with a Gaussian kernel of that
width before the region means are taken (as D1 / E3 do with 4 mm). Results then
go to the subfolder smooth<mm>/. Without it the data are not smoothed (as E7 / E10).

Everything else (data, models, nuisance terms, matching) is the code of E7 and
E10; only the region function, the smoothing and the output folder are replaced.

Input   as E7 / E10, plus outputs/B3_combine_localizer_masks/sub-XX_mask_<roi>_threshold_<pct>_combined.nii and <derivatives>/sourcedata/freesurfer/sub-XX/mri/aparc.a2009s+aseg.mgz
Output  outputs/E12_precuneus_parietal_subregions/[smooth<mm>/]
          E7/participants_ret_press/<sub>_region_estimates.csv, E7/group_levels_ret_press.csv,
          E7/group_differences_ret_press.csv
          E10/participants[_scene]/<sub>_trials.csv, E10/group_matched[_scene].csv,
          E10/participant_matched[_scene].csv
Usage:  python E12_precuneus_parietal_subregions.py <subject>
        python E12_precuneus_parietal_subregions.py group

Not a port of the original pipeline: added 2026-11-24. Exploratory: many regions,
none chosen in advance.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image, masking
from nilearn.image import resample_to_img

os.environ["E7_MODELS"] = "ret_press"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import E7_native_regions as e7
import E10_matched_search_time as e10
from config import DERIVATIVES_DIR, step_input_dir, step_output_dir

SMOOTH = float(os.environ.get("E12_SMOOTH", 0))
RELIABILITY = (70, 30)
RELIABLE_ROIS = ("precuneus", "parietal", "rsc", "posterior_cingulate", "caudal_acc")
MIN_RELIABLE = 30

DK = {"PREC": 25, "INFPAR": 8, "SUPPAR": 29, "SUPRAMARG": 31, "RSC": 10, "PCC": 23}
DK_BOTH = {"PRECUNEUS": (25,), "PARIETAL": (8,), "PARIETAL_PRECUNEUS": (8, 25), "SUPPAR": (29,),
           "SUPRAMARG": (31,), "RSC": (10,), "PCC": (23,), "CAUDAL_ACC": (2,)}
DESTRIEUX = {"D_PRECUNEUS_G": 30, "D_SUBPARIETAL_S": 72, "D_PARIETOOCC_S": 66, "D_MARGINAL_S": 47,
             "D_PCC_DORSAL_G": 9, "D_PCC_VENTRAL_G": 10, "D_SUPPARIETAL_G": 27, "D_INTRAPARIETAL_S": 57,
             "D_ANGULAR_G": 25, "D_SUPRAMARGINAL_G": 26}


def fine_regions(sub_mri, brain_img, bm):
    """name -> boolean over the voxels of the brain mask (same contract as d1.anatomical_regions)."""
    def on_grid(path):
        img = nib.load(path)
        if img.shape[:3] != brain_img.shape or not np.allclose(img.affine, brain_img.affine, atol=1e-3):
            img = resample_to_img(img, brain_img, interpolation="nearest", force_resample=True, copy_header=True)
        return np.asarray(img.dataobj).astype(int)[bm]

    dk = on_grid(sorted(glob.glob(opj(str(DERIVATIVES_DIR), sub_mri, "ses-04", "func",
                                      f"{sub_mri}_ses-04_task-*_space-T1w_desc-aparcaseg_dseg.nii.gz")))[0])
    des = on_grid(opj(str(DERIVATIVES_DIR), "sourcedata", "freesurfer", sub_mri, "mri", "aparc.a2009s+aseg.mgz"))
    out = {}
    for name, n in DK.items():
        out[name + "_L"], out[name + "_R"] = dk == 1000 + n, dk == 2000 + n
    for name, ns in DK_BOTH.items():
        out[name] = np.isin(dk, [1000 + n for n in ns] + [2000 + n for n in ns])
    for name, n in DESTRIEUX.items():
        out[name] = (des == 11100 + n) | (des == 12100 + n)
    out = {k: v for k, v in out.items() if v.sum() >= 10}
    b3 = str(step_input_dir("B3_combine_localizer_masks"))
    for pct in RELIABILITY:
        files = [opj(b3, f"{sub_mri}_mask_{roi}_threshold_{pct}_combined.nii") for roi in RELIABLE_ROIS]
        if not all(os.path.exists(f) for f in files):
            print(f"no reliability masks ({pct}) for {sub_mri}: reliable-voxel regions skipped", flush=True)
            continue
        rel = np.any([on_grid(f) > 0 for f in files], axis=0)
        for k in [k for k in out if "__r" not in k]:
            if (out[k] & rel).sum() >= MIN_RELIABLE:
                out[f"{k}__r{pct}"] = out[k] & rel
    return out


def _route(step):
    step = (f"smooth{SMOOTH:g}/" if SMOOTH else "") + step
    return lambda script_file, subfolder=None: step_output_dir(__file__, step + ("/" + subfolder if subfolder else ""))


class _Masking:
    """nilearn.masking for E7 / E10, with the BOLD data smoothed before the mask is applied."""
    intersect_masks = staticmethod(masking.intersect_masks)

    @staticmethod
    def apply_mask(img, mask):
        return masking.apply_mask(image.smooth_img(img, SMOOTH) if SMOOTH else img, mask)


e7.masking = e10.masking = _Masking


b1_regions = e7.d1.anatomical_regions             # the regions of B1 (used by E13 for VIS, MTL, HPC, PHPC)
e7.d1.anatomical_regions = fine_regions            # e7.d1 and e10.d1 are the same module
e7.step_output_dir = _route("E7")
e10.step_output_dir = _route("E10")


def _kind(kind):
    e10.KIND, e10.OTHER = kind, ("scene" if kind == "object" else "object")
    e10.TAG = "" if kind == "object" else "_" + kind


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "group":
        e7.group()
        for kind in ("object", "scene"):
            _kind(kind)
            e10.group()
    else:
        sub = e7.parse_subject_arg(arg)
        e7.participant(sub)
        for kind in ("object", "scene"):
            _kind(kind)
            e10.participant(sub)
