# -*- coding: utf-8 -*-
"""Per-run GLM to extract t-maps and noise precision matrices for RSA.

For each retrieval run, fits a GLM with individual item conditions
(10 object categories + 10 scenes = ~20 regressors) and extracts:
  - t-maps per condition per ROI (saved as .npy)
  - Residual-based precision matrices per ROI (for crossnobis noise normalization)

Usage
-----
    python 01_glm_tmaps_residuals.py --subject sub-13 --session presleep
    python 01_glm_tmaps_residuals.py --subject 13 --session postsleep
"""

import argparse
import gc
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from nilearn import masking
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from matplotlib import pyplot as plt

import rsatoolbox.data.noise

from config import (
    SCRIPT_DIR,
    TR,
    CONFOUND_COLUMNS,
    SMOOTHING_FWHM,
    HIGH_PASS,
    HRF_MODEL,
    DRIFT_MODEL,
    NOISE_MODEL,
    NOISE_METHOD,
    RSA_ROIS,
)
from utils import (
    parse_subject_id,
    load_localizer_category_mapping,
    load_retrieval_behavior,
    load_functional_data,
    load_confounds,
    load_reliability_mask,
    get_mni_mask,
    resample_mask_to_func,
    extract_rsa_events_for_run,
    save_conditions,
)


def main():
    parser = argparse.ArgumentParser(description="RSA GLM: t-maps + residuals")
    parser.add_argument("--subject", required=True, help="Subject ID (13, s13, or sub-13)")
    parser.add_argument("--session", required=True,
                        choices=["presleep", "postsleep", "postsleep_3runs"])
    parser.add_argument("--roi", choices=RSA_ROIS, default=None,
                        help="Single ROI to process (default: all ROIs)")
    args = parser.parse_args()

    sub_behav, sub_num, sub_mri = parse_subject_id(args.subject)
    session = args.session

    print(f"\n{'='*60}")
    print(f"RSA GLM: {sub_mri}, session={session}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # Output directories
    # -----------------------------------------------------------------
    out_base = opj(SCRIPT_DIR, "01_glm_tmaps_residuals", session)
    out_tmaps = opj(out_base, "t_maps", sub_mri)
    out_noise = opj(out_base, "noise_matrices", sub_mri)
    out_conds = opj(out_base, "conditions", sub_mri)
    out_figs = opj(out_base, "figures")
    for d in [out_tmaps, out_noise, out_conds, out_figs]:
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------
    # Load behavioral data
    # -----------------------------------------------------------------
    obj_enc_to_cat = load_localizer_category_mapping(sub_behav, sub_num)
    df_behavior = load_retrieval_behavior(sub_behav, sub_num, session, obj_enc_to_cat)

    # -----------------------------------------------------------------
    # Load functional data
    # -----------------------------------------------------------------
    func_imgs, selected_indices = load_functional_data(sub_mri, session)
    confounds_list = load_confounds(sub_mri, session, selected_indices)

    # -----------------------------------------------------------------
    # Load ROI masks
    # -----------------------------------------------------------------
    rois_to_run = [args.roi] if args.roi else RSA_ROIS
    roi_masks = {}
    for roi in rois_to_run:
        try:
            mask = load_reliability_mask(sub_behav, roi)
            # Resample to functional space if needed
            mask = resample_mask_to_func(mask, func_imgs[0])
            roi_masks[roi] = mask
        except FileNotFoundError as e:
            print(f"  WARNING: Skipping ROI {roi}: {e}")

    if not roi_masks:
        raise RuntimeError("No ROI masks could be loaded. Aborting.")

    # Get MNI mask for GLM fitting (broad mask)
    mni_mask = get_mni_mask(func_imgs[0])

    # -----------------------------------------------------------------
    # Determine behavioral run numbers for retrieval
    # -----------------------------------------------------------------
    # For presleep: retrieval runs are the higher run_nr values
    # For postsleep: all runs are retrieval
    all_run_nrs = sorted(df_behavior["run_nr"].dropna().unique().astype(int))
    retrieval_label_types = {"sc_no4obj", "d_no4sc"}

    # Filter to runs that contain retrieval trials
    retrieval_run_nrs = []
    for rn in all_run_nrs:
        df_rn = df_behavior[df_behavior["run_nr"] == rn]
        if df_rn["label_type"].isin(retrieval_label_types).any():
            retrieval_run_nrs.append(rn)

    # Match behavioral runs to functional runs
    n_func = len(func_imgs)
    if session == "presleep":
        # For presleep, we selected last 3 functional runs (retrieval)
        # Match to last 3 behavioral retrieval runs
        retrieval_run_nrs = retrieval_run_nrs[-n_func:]
    else:
        retrieval_run_nrs = retrieval_run_nrs[:n_func]

    if len(retrieval_run_nrs) != n_func:
        print(f"  WARNING: {len(retrieval_run_nrs)} behavioral retrieval runs "
              f"vs {n_func} functional runs. Using min of both.")
        n_use = min(len(retrieval_run_nrs), n_func)
        retrieval_run_nrs = retrieval_run_nrs[:n_use]
        func_imgs = func_imgs[:n_use]
        confounds_list = confounds_list[:n_use]

    print(f"\nBehavioral retrieval runs: {retrieval_run_nrs}")
    print(f"Functional runs: {n_func}")

    # -----------------------------------------------------------------
    # Collect all conditions across runs (for consistent ordering)
    # -----------------------------------------------------------------
    all_conditions = set()
    for run_nr in retrieval_run_nrs:
        events = extract_rsa_events_for_run(df_behavior, run_nr, obj_enc_to_cat)
        all_conditions.update(events["trial_type"].unique())
    all_conditions = sorted(all_conditions)
    print(f"\nAll conditions across runs ({len(all_conditions)}): {all_conditions}")

    # Save condition list
    save_conditions(all_conditions, opj(out_conds, f"{sub_mri}_{session}_conditions.json"))

    # -----------------------------------------------------------------
    # Per-run GLM
    # -----------------------------------------------------------------
    for run_idx, (run_nr, func_img, confs) in enumerate(
        zip(retrieval_run_nrs, func_imgs, confounds_list)
    ):
        print(f"\n--- Run {run_idx} (behavioral run_nr={run_nr}) ---")

        # Extract events
        events = extract_rsa_events_for_run(df_behavior, run_nr, obj_enc_to_cat)
        if events.empty:
            print(f"  Skipping run {run_nr}: no events")
            continue

        # Build nilearn events (drop modality column)
        events_nilearn = events[["onset", "duration", "trial_type"]].copy()

        # Build design matrix
        n_trs = func_img.shape[-1]
        frame_times = np.arange(n_trs) * TR
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events=events_nilearn,
            hrf_model=HRF_MODEL,
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
        )

        # Add confounds
        nuisance = confs.values
        for ci, col_name in enumerate(confs.columns):
            if ci < nuisance.shape[1]:
                design_matrix[col_name] = nuisance[:, ci]

        # Save design matrix figure
        fig_path = opj(out_figs, f"{sub_mri}_{session}_run{run_idx}_design_matrix.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_design_matrix(design_matrix, ax=ax)
        ax.set_title(f"{sub_mri} {session} run{run_idx} (behav run_nr={run_nr})")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Fit GLM
        glm = FirstLevelModel(
            t_r=TR,
            noise_model=NOISE_MODEL,
            standardize=True,
            hrf_model=HRF_MODEL,
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
            smoothing_fwhm=SMOOTHING_FWHM,
            mask_img=mni_mask,
            minimize_memory=False,  # needed for residuals
        )
        glm_result = glm.fit(func_img, design_matrices=design_matrix)
        dm_columns = glm_result.design_matrices_[0].columns.tolist()

        # Extract t-maps and noise for each ROI
        for roi, roi_mask in roi_masks.items():
            # --- T-maps per condition ---
            for condition in all_conditions:
                if condition not in dm_columns:
                    # Condition not present in this run — save NaN placeholder
                    print(f"    {roi}/{condition}: not in design matrix, saving NaN")
                    np.save(
                        opj(out_tmaps, f"{sub_mri}_{session}_{roi}_{condition}_run_{run_idx}.npy"),
                        np.array([]),
                    )
                    continue

                contrast_vec = np.zeros(len(dm_columns))
                contrast_vec[dm_columns.index(condition)] = 1
                t_map = glm_result.compute_contrast(contrast_vec, stat_type="t")
                masked_tmap = masking.apply_mask(t_map, roi_mask)

                save_path = opj(
                    out_tmaps,
                    f"{sub_mri}_{session}_{roi}_{condition}_run_{run_idx}.npy",
                )
                np.save(save_path, masked_tmap)

            # --- Noise precision matrix from residuals ---
            residuals = glm_result.residuals[0]
            masked_residuals = masking.apply_mask(residuals, roi_mask)
            masked_res_arr = np.array(masked_residuals)

            noise_prec = rsatoolbox.data.noise.prec_from_residuals(
                masked_res_arr, method=NOISE_METHOD,
            )
            noise_path = opj(
                out_noise,
                f"{sub_mri}_{session}_{roi}_noise_run_{run_idx}.npy",
            )
            np.save(noise_path, noise_prec)
            print(f"    {roi}: saved {len(all_conditions)} t-maps + noise matrix "
                  f"({masked_res_arr.shape[1]} voxels)")

        # Clean up
        del glm_result, func_img, residuals
        gc.collect()

    print(f"\nDone. Outputs in: {out_base}")


if __name__ == "__main__":
    main()
