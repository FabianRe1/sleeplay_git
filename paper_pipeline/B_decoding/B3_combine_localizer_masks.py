#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B3 - Combine the reliability masks of the two localizers.

For every participant, ROI and reliability threshold: the union of the loc1
and loc2 reliability masks (a voxel is kept if it is reliable in at least one
localizer; `intersect_masks` with threshold=0).

Input   outputs/B2_voxel_reliability_masks/reliability_masks/
            sXX_mask_<roi>_threshold_<pct>_localizer_loc{1,2}.nii
Output  outputs/B3_combine_localizer_masks/
            sub-XX_mask_<roi>_threshold_<pct>_combined.nii

Ported from 3_reliability_voxel_selection/combine_loc1_and_loc3_masks.py;
logic unchanged. One difference in file handling: the ROI name is matched
against the file name instead of the full path, so the result cannot depend
on where the pipeline is stored.

Existing masks are always rebuilt. The original skipped files that already
existed, which left seven participants with combined masks older than their
reliability masks (found 2026-11-18).

Usage:  python B3_combine_localizer_masks.py
"""

import glob
import os
import re
import sys
from pathlib import Path

from nilearn import image, masking

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir

ROIS = ['vis', 'mtl', 'cortical_memory', 'parietal_precuneus', 'parietal', 'precuneus',
        'medial_ofc', 'rostral_acc', 'caudal_acc', 'superior_frontal', 'rsc',
        'posterior_cingulate', 'hpc', 'somato']
# B3_ROIS=a,b limits the run to these ROIs (e.g. to add masks for a new ROI
# without rebuilding the others); hpc and somato were added 2026-11-25
if os.environ.get("B3_ROIS"):
    ROIS = os.environ["B3_ROIS"].split(",")


def combine_masks(mask_loc_1, mask_loc_2):
    """
    Combine two masks using nilearn's intersect_masks function.

    threshold 1 means that the voxels should be non-zero in both masks
    threshold 0 means that the voxels should be non-zero in at least one mask
    """
    combined_mask = masking.intersect_masks([mask_loc_1, mask_loc_2], threshold=0, connected=False)
    return combined_mask


def process_and_save_masks(input_folder, output_folder, ROIS):
    # Pattern: sXX_mask_YYY_threshold_ZZ_localizer_loc1.nii
    mask_files_1 = glob.glob(os.path.join(input_folder, '*_localizer_loc1.nii'))

    print(f"Found {len(mask_files_1)} files for loc1 in {input_folder}")

    n_saved = 0
    for roi in ROIS:

        for mask_file_1 in mask_files_1:

            base_name = os.path.basename(mask_file_1)

            if roi not in base_name:
                continue

            # Corresponding loc2 file
            corresponding_name = base_name.replace('localizer_loc1', 'localizer_loc2')
            mask_file_2 = os.path.join(input_folder, corresponding_name)

            if os.path.exists(mask_file_2):
                new_name = base_name.replace('localizer_loc1', 'combined')
                # Replace sXX with sub-XX
                new_name = re.sub(r's(\d+)', r'sub-\1', new_name)

                combined_mask_filename = os.path.join(output_folder, new_name)

                mask_loc_1 = image.load_img(mask_file_1)
                mask_loc_2 = image.load_img(mask_file_2)

                combined_mask = combine_masks(mask_loc_1, mask_loc_2)

                combined_mask.to_filename(combined_mask_filename)
                n_saved += 1
                print(f"Saved combined mask to {combined_mask_filename}")
            else:
                print(f"Corresponding file not found for {base_name}: {corresponding_name}")
    return n_saved


out_path = str(step_output_dir(__file__))
path_masks_voxel_reliability = str(step_input_dir('B2_voxel_reliability_masks', 'reliability_masks'))

n_saved = process_and_save_masks(path_masks_voxel_reliability, out_path, ROIS=ROIS)
print(f"\nB3 done: {n_saved} masks written to {out_path}")
