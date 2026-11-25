# -*- coding: utf-8 -*-
"""Compute cross-validated (crossnobis) RDMs from t-maps and noise matrices.

For each modality (objects=10 categories, scenes=10 scenes) separately:
  1. Load masked t-maps and noise precision matrices per run
  2. Build rsatoolbox Dataset with runs as CV folds
  3. Compute crossnobis RDM
  4. Save RDM as pickle + CSV + visualization

Usage
-----
    python 02_compute_rdms.py --subject sub-13 --session presleep --roi vis
"""

import argparse
import os
from os.path import join as opj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr

from config import SCRIPT_DIR, RSA_ROIS
from utils import (
    parse_subject_id,
    load_localizer_category_mapping,
    load_retrieval_behavior,
    load_conditions,
    get_conditions_for_modality,
)


def compute_rdm_for_modality(
    sub_mri: str,
    session: str,
    roi: str,
    modality: str,
    all_conditions: list[str],
    modality_conditions: list[str],
    tmaps_dir: str,
    noise_dir: str,
    n_runs: int,
) -> rsatoolbox.rdm.RDMs | None:
    """Load t-maps, build Dataset, compute crossnobis RDM for one modality."""

    print(f"\n  --- {modality} RDM ({len(modality_conditions)} conditions) ---")

    # Determine which runs have all conditions present
    valid_runs = []
    for run_idx in range(n_runs):
        all_present = True
        for cond in modality_conditions:
            fpath = opj(tmaps_dir, f"{sub_mri}_{session}_{roi}_{cond}_run_{run_idx}.npy")
            if not os.path.exists(fpath):
                all_present = False
                break
            arr = np.load(fpath)
            if arr.size == 0:
                all_present = False
                break
        if all_present:
            valid_runs.append(run_idx)

    if len(valid_runs) < 2:
        print(f"    WARNING: Only {len(valid_runs)} valid runs. "
              f"Need >=2 for crossnobis. Skipping.")
        return None

    print(f"    Valid runs: {valid_runs}")

    # Load measurements: (n_conditions * n_runs, n_voxels)
    measurements = []
    conds_labels = []
    session_labels = []

    for cond in modality_conditions:
        for run_idx in valid_runs:
            fpath = opj(tmaps_dir, f"{sub_mri}_{session}_{roi}_{cond}_run_{run_idx}.npy")
            tmap = np.load(fpath)
            measurements.append(tmap)
            conds_labels.append(cond)
            session_labels.append(f"run_{run_idx}")

    measurements_arr = np.array(measurements)
    n_voxels = measurements_arr.shape[1]
    print(f"    Measurements shape: {measurements_arr.shape} "
          f"({len(modality_conditions)} conds x {len(valid_runs)} runs, {n_voxels} voxels)")

    # Load noise precision matrices
    noise_list = []
    for run_idx in valid_runs:
        noise_path = opj(noise_dir, f"{sub_mri}_{session}_{roi}_noise_run_{run_idx}.npy")
        noise_prec = np.load(noise_path)
        noise_list.append(noise_prec)

    # Build rsatoolbox Dataset
    obs_des = {
        "conds": np.array(conds_labels),
        "sessions": np.array(session_labels),
    }
    chn_des = {
        "voxels": np.array([f"voxel_{i}" for i in range(n_voxels)]),
    }
    des = {"subj": sub_mri}

    data = rsd.Dataset(
        measurements=measurements_arr,
        descriptors=des,
        obs_descriptors=obs_des,
        channel_descriptors=chn_des,
    )

    # Compute crossnobis RDM
    rdm_cv = rsatoolbox.rdm.calc_rdm(
        data,
        method="crossnobis",
        descriptor="conds",
        cv_descriptor="sessions",
        noise=noise_list,
    )

    print(f"    RDM computed: {rdm_cv.n_cond} conditions")
    return rdm_cv


def save_rdm(rdm, conditions, save_dir, prefix):
    """Save RDM as pickle, CSV, and visualization."""
    os.makedirs(save_dir, exist_ok=True)

    # Save as pickle
    pkl_path = opj(save_dir, f"{prefix}.pkl")
    rdm.save(pkl_path, file_type="pkl", overwrite=True)

    # Save as labeled CSV
    rdm_matrix = rdm.get_matrices()[0]
    df_rdm = pd.DataFrame(rdm_matrix, columns=conditions, index=conditions)
    csv_path = opj(save_dir, f"{prefix}.csv")
    df_rdm.to_csv(csv_path)

    print(f"    Saved RDM to {pkl_path} and {csv_path}")
    return rdm_matrix


def plot_rdm(rdm_matrix, conditions, title, save_path):
    """Plot RDM as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rdm_matrix, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(conditions, fontsize=8)
    plt.colorbar(im, ax=ax, label="Crossnobis distance")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved figure: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute crossnobis RDMs")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True,
                        choices=["presleep", "postsleep", "postsleep_3runs"])
    parser.add_argument("--roi", required=True, choices=RSA_ROIS)
    args = parser.parse_args()

    sub_behav, sub_num, sub_mri = parse_subject_id(args.subject)
    session = args.session
    roi = args.roi

    print(f"\n{'='*60}")
    print(f"Compute RDMs: {sub_mri}, session={session}, roi={roi}")
    print(f"{'='*60}\n")

    # Paths
    glm_base = opj(SCRIPT_DIR, "01_glm_tmaps_residuals", session)
    tmaps_dir = opj(glm_base, "t_maps", sub_mri)
    noise_dir = opj(glm_base, "noise_matrices", sub_mri)
    conds_path = opj(glm_base, "conditions", sub_mri, f"{sub_mri}_{session}_conditions.json")

    out_base = opj(SCRIPT_DIR, "02_compute_rdms", session)
    out_rdms = opj(out_base, "rdms", sub_mri)
    out_figs = opj(out_base, "figures")
    os.makedirs(out_rdms, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    # Load condition list
    all_conditions = load_conditions(conds_path)
    print(f"Loaded {len(all_conditions)} conditions: {all_conditions}")

    # Load behavioral data for modality splitting
    obj_enc_to_cat = load_localizer_category_mapping(sub_behav, sub_num)
    df_behavior = load_retrieval_behavior(sub_behav, sub_num, session, obj_enc_to_cat)

    # Determine number of runs from saved files
    run_files = [f for f in os.listdir(noise_dir)
                 if f.startswith(f"{sub_mri}_{session}_{roi}_noise_run_")]
    n_runs = len(run_files)
    print(f"Found {n_runs} noise matrices (runs)")

    # Process each modality
    for modality in ["object", "scene"]:
        modality_conds = get_conditions_for_modality(df_behavior, obj_enc_to_cat, modality)
        # Filter to conditions that are actually present in the GLM
        modality_conds = [c for c in modality_conds if c in all_conditions]

        if len(modality_conds) < 2:
            print(f"\n  Skipping {modality}: only {len(modality_conds)} conditions")
            continue

        rdm = compute_rdm_for_modality(
            sub_mri, session, roi, modality,
            all_conditions, modality_conds,
            tmaps_dir, noise_dir, n_runs,
        )

        if rdm is None:
            continue

        # Save
        prefix = f"{sub_mri}_{session}_{roi}_{modality}s_rdm"
        rdm_matrix = save_rdm(rdm, modality_conds, out_rdms, prefix)

        # Plot
        plot_rdm(
            rdm_matrix, modality_conds,
            f"{sub_mri} | {session} | {roi} | {modality}s\nCrossnobis RDM",
            opj(out_figs, f"{prefix}.png"),
        )

    print(f"\nDone. Outputs in: {out_base}")


if __name__ == "__main__":
    main()
