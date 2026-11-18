# -*- coding: utf-8 -*-
"""
01_first_level_glm.py — First-level GLM for remembered vs not_remembered.

Fits a single concatenated GLM across all valid runs for one subject and
one session. Outputs effect-size and effect-variance maps for second-level
analysis.

Usage:
    python 01_first_level_glm.py --subject sub-13 --session encoding
    python 01_first_level_glm.py --subject sub-13 --session postsleep
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt

from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import concat_imgs, load_img, mean_img, math_img, resample_to_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix, plot_stat_map
import nilearn.image as nimg

from config import (
    TR,
    SMOOTHING_FWHM,
    HIGH_PASS,
    HRF_MODEL,
    DRIFT_MODEL,
    CONFOUND_COLUMNS,
    TRIM_BUFFER_TRS,
    SCRIPT_DIR,
)
from utils import (
    parse_subject_id,
    load_localizer_category_mapping,
    load_retrieval_behavior,
    load_functional_data,
    load_confounds,
    get_mni_mask,
    compute_mean_remembered_rt,
    extract_events_for_run,
    pad_to_global,
    build_padded_drift_regressors,
)


def _session_suffix(session: str, trim: bool) -> str:
    """Return directory suffix, appending '_trimmed' when trimming is on."""
    return f"{session}_trimmed" if trim else session


def make_output_dir(session: str, sub_num: str, trim: bool = False) -> Path:
    """Create and return the output directory for this subject/session."""
    tag = _session_suffix(session, trim)
    out = SCRIPT_DIR / f"01_first_level_glm_{tag}" / "glm_maps" / f"sub-{sub_num}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_figures_dir(session: str, trim: bool = False) -> Path:
    """Create and return the figures directory for this session."""
    tag = _session_suffix(session, trim)
    out = SCRIPT_DIR / f"01_first_level_glm_{tag}" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


# =====================================================================
# Main analysis
# =====================================================================

def run_first_level(subject: str, session: str, trim: bool = False) -> None:
    t_start = time.time()

    # --- Parse subject ---
    sub_behav, sub_num, sub_mri = parse_subject_id(subject)
    trim_label = " [TRIMMED]" if trim else ""
    print(f"{'=' * 70}")
    print(f"First-level GLM: {sub_mri} | session={session}{trim_label}")
    print(f"{'=' * 70}")

    out_dir = make_output_dir(session, sub_num, trim)
    fig_dir = make_figures_dir(session, trim)

    # --- Load behavioral data ---
    obj_enc_to_cat = load_localizer_category_mapping(sub_behav, sub_num)
    df_behavior = load_retrieval_behavior(sub_behav, sub_num, session, obj_enc_to_cat)

    # --- Compute mean remembered RT (used as duration for non-remembered trials) ---
    mean_remembered_rt = compute_mean_remembered_rt(df_behavior)

    # --- Load functional data ---
    func_imgs, selected_indices = load_functional_data(sub_mri, session)
    confs = load_confounds(sub_mri, session, selected_indices)

    # --- MNI mask ---
    mni_mask = get_mni_mask(func_imgs[0])

    # Verify confound/func alignment
    for i, (f, c) in enumerate(zip(func_imgs, confs)):
        n_trs_func = f.shape[-1]
        n_trs_conf = len(c)
        print(f"  Run {i}: func={n_trs_func} TRs, confounds={n_trs_conf} rows")
        if n_trs_func != n_trs_conf:
            print(f"  WARNING: TR count mismatch in run {i}!")

    # --- Build run-nr -> data-index mapping ---
    runs_in_behavior = sorted(
        df_behavior["run_nr"].dropna().unique().astype(int).tolist()
    )
    run_nr_to_idx = {rn: i for i, rn in enumerate(runs_in_behavior)}
    print(f"\nBehavioral run_nrs: {runs_in_behavior}")
    print(f"run_nr -> data_idx: {run_nr_to_idx}")

    # --- FirstLevelModel ---
    first_level_model = FirstLevelModel(
        t_r=TR,
        standardize=True,
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS,
        mask_img=mni_mask,
    )

    # --- Per-run design matrices ---
    per_run_design_matrices: list[pd.DataFrame] = []
    per_run_func_data: list = []

    for run_nr in runs_in_behavior:
        data_idx = run_nr_to_idx[run_nr]
        if data_idx >= len(func_imgs):
            print(f"[run {run_nr}] No matching functional data (idx={data_idx}). Skipping.")
            continue

        events = extract_events_for_run(df_behavior, run_nr, nonremem_duration=mean_remembered_rt)
        if events.empty:
            print(f"[run {run_nr}] No valid events. Skipping.")
            continue

        fmri_img = func_imgs[data_idx]
        nuisance = confs[data_idx][
            [c for c in CONFOUND_COLUMNS if c in confs[data_idx].columns]
        ].values

        frame_times = np.arange(fmri_img.shape[-1]) * TR
        dm = make_first_level_design_matrix(
            frame_times,
            events=events,
            add_regs=nuisance,
            add_reg_names=[c for c in CONFOUND_COLUMNS if c in confs[data_idx].columns],
        )

        per_run_design_matrices.append(dm)
        per_run_func_data.append(fmri_img)

    if not per_run_design_matrices:
        print("ERROR: No valid runs found. Exiting.")
        sys.exit(1)

    print(f"\nBuilt design matrices for {len(per_run_design_matrices)} runs.")

    # --- Smooth (and optionally trim dead periods) ---
    cleaned_func = []
    cleaned_dms = []

    for ri in range(len(per_run_func_data)):
        func_data = nimg.smooth_img(per_run_func_data[ri], fwhm=SMOOTHING_FWHM)
        dm = per_run_design_matrices[ri]
        orig_len = func_data.shape[-1]

        if trim:
            # Trim after last HRF activity + buffer (optional, off by default)
            cond_cols = [c for c in ["remembered", "not_remembered"] if c in dm.columns]
            if cond_cols:
                cond_active = np.zeros(len(dm), dtype=bool)
                for c in cond_cols:
                    cond_active |= (dm[c] != 0).values
                active_trs = np.where(cond_active)[0]
                last_active = active_trs[-1] if len(active_trs) > 0 else 0
            else:
                last_active = 0
            cutoff = min(last_active + TRIM_BUFFER_TRS, orig_len)
            func_data = func_data.slicer[..., : cutoff + 1]
            dm = dm.iloc[: cutoff + 1].reset_index(drop=True)
            print(f"  Run {ri}: {orig_len} -> {func_data.shape[-1]} TRs (trimmed)")
        else:
            print(f"  Run {ri}: {orig_len} TRs")

        cleaned_func.append(func_data)
        cleaned_dms.append(dm)

    # Free memory
    del per_run_func_data, func_imgs
    gc.collect()

    # --- Build concatenated design matrix ---
    lengths = [f.shape[-1] for f in cleaned_func]
    total_len = sum(lengths)
    n_runs = len(cleaned_dms)

    # Task regressors + run intercepts (padded)
    task_regs: dict[str, pd.Series] = {}
    for ri in range(n_runs):
        label = f"run{ri + 1}"
        for cond in ["remembered", "not_remembered"]:
            if cond in cleaned_dms[ri].columns:
                task_regs[f"{cond}_{label}"] = pad_to_global(
                    ri, cleaned_dms[ri][cond], lengths
                )
            else:
                print(f"  WARNING: '{cond}' missing in run {ri + 1}.")
        task_regs[f"intercept_{label}"] = pad_to_global(
            ri, np.ones(lengths[ri]), lengths
        )

    # Drift regressors (padded, constant dropped)
    drift_df = build_padded_drift_regressors(cleaned_dms, lengths)

    # Nuisance regressors (stacked vertically — shared columns across runs)
    conf_cols = [c for c in CONFOUND_COLUMNS if c in cleaned_dms[0].columns]
    nuis_by_run = [dm[conf_cols].reset_index(drop=True) for dm in cleaned_dms]
    nuis_stacked = pd.concat(nuis_by_run, axis=0, ignore_index=True)

    # Assemble final design matrix
    task_df = pd.DataFrame(task_regs).reset_index(drop=True)
    final_dm = pd.concat([task_df, nuis_stacked, drift_df], axis=1)
    assert len(final_dm) == total_len, f"DM length {len(final_dm)} != {total_len}"

    # --- Drop zero-variance task regressors ---
    # A regressor with no events (all zeros after convolution) acts as a
    # spurious intercept and must be removed before contrast construction.
    task_col_names = [c for c in final_dm.columns
                      if c.startswith("remembered_run") or c.startswith("not_remembered_run")]
    zero_var_cols = [c for c in task_col_names if final_dm[c].var() == 0]
    if zero_var_cols:
        print(f"\n  WARNING: Dropping {len(zero_var_cols)} zero-variance task regressor(s):")
        for zc in zero_var_cols:
            print(f"    - {zc}")
        final_dm = final_dm.drop(columns=zero_var_cols)

    print(f"\nFinal design matrix: {final_dm.shape}")
    print(f"  Columns: {list(final_dm.columns)}")

    # --- Save design matrix figure ---
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_design_matrix(final_dm, ax=ax)
    ax.set_title(f"{sub_mri} | {session} | design matrix")
    fig.savefig(
        fig_dir / f"{sub_mri}_{session}_design_matrix.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- Concatenate functional data and fit ---
    print("\nConcatenating functional data...")
    stacked_func = concat_imgs(cleaned_func)

    del cleaned_func
    gc.collect()

    print("Fitting GLM...")
    glm_result = first_level_model.fit(stacked_func, design_matrices=final_dm)
    print("GLM fitted successfully.")

    # --- Compute contrast: remembered > not_remembered ---
    n_cols = len(final_dm.columns)
    contrast_vec = np.zeros(n_cols)

    rem_cols = [c for c in final_dm.columns if c.startswith("remembered_run")]
    notrem_cols = [c for c in final_dm.columns if c.startswith("not_remembered_run")]

    if not rem_cols and not notrem_cols:
        print("ERROR: No remembered or not_remembered columns. Cannot compute contrast.")
        sys.exit(1)

    n_rem = len(rem_cols)
    n_notrem = len(notrem_cols)

    # --- Contrast 1: remembered > not_remembered ---
    # Only compute if BOTH conditions have at least one regressor.
    # When one condition is entirely missing (e.g., subject always remembers),
    # this contrast is undefined — skip it but still compute task > baseline.
    can_compute_remem = n_rem > 0 and n_notrem > 0
    if can_compute_remem:
        w_rem = 1.0 / n_rem
        w_notrem = -1.0 / n_notrem
        for col in rem_cols:
            contrast_vec[final_dm.columns.get_loc(col)] = w_rem
        for col in notrem_cols:
            contrast_vec[final_dm.columns.get_loc(col)] = w_notrem

        weight_sum = contrast_vec.sum()
        print(f"\nContrast 1: remembered ({n_rem} runs, w={w_rem:.3f}) > "
              f"not_remembered ({n_notrem} runs, w={w_notrem:.3f})")
        print(f"  Remembered runs:     {rem_cols}")
        print(f"  Not-remembered runs: {notrem_cols}")
        print(f"  Sum of weights: {weight_sum:.6f}")
    else:
        missing = "remembered" if not rem_cols else "not_remembered"
        print(f"\nWARNING: No '{missing}' regressors survived variance check. "
              f"Skipping remem_vs_nonremem contrast (requires both conditions).")
        print(f"  Remembered runs:     {rem_cols}")
        print(f"  Not-remembered runs: {notrem_cols}")

    # --- Contrast 2: task > baseline (all task regressors positive) ---
    task_baseline_vec = np.zeros(n_cols)
    n_task_total = n_rem + n_notrem
    w_task = 1.0 / n_task_total if n_task_total > 0 else 0
    for col in rem_cols + notrem_cols:
        task_baseline_vec[final_dm.columns.get_loc(col)] = w_task

    print(f"\nContrast 2: task > baseline ({n_task_total} regressors, w={w_task:.3f})")
    print(f"  Sum of weights: {task_baseline_vec.sum():.6f}")

    # --- Build contrast dict (only include computable contrasts) ---
    contrasts = {}
    if can_compute_remem:
        contrasts["remem_vs_nonremem"] = contrast_vec
    contrasts["task_vs_baseline"] = task_baseline_vec

    # --- Compute and save maps ---
    # Use task_vs_baseline for MNI mask (always available)
    mni_eff = resample_to_img(
        load_mni152_brain_mask(),
        glm_result.compute_contrast(task_baseline_vec, output_type="effect_size"),
        interpolation="nearest",
    )

    for cname, cvec in contrasts.items():
        eff_map = glm_result.compute_contrast(cvec, output_type="effect_size")
        var_map = glm_result.compute_contrast(cvec, output_type="effect_variance")
        eff_masked = math_img("img * mask", img=eff_map, mask=mni_eff)
        var_masked = math_img("img * mask", img=var_map, mask=mni_eff)

        eff_path = out_dir / f"Big_GLM_{cname}_effect.nii.gz"
        var_path = out_dir / f"Big_GLM_{cname}_var.nii.gz"
        eff_masked.to_filename(eff_path)
        var_masked.to_filename(var_path)
        print(f"  Saved {cname}: {eff_path}")

    # --- Save stat map figures ---
    mean_image = mean_img(stacked_func)
    for cname, cvec in contrasts.items():
        eff_img = load_img(str(out_dir / f"Big_GLM_{cname}_effect.nii.gz"))
        label = "remem > not_remem" if "vs_nonremem" in cname else "task > baseline"
        plot_stat_map(
            eff_img,
            bg_img=mean_image,
            title=f"{sub_mri} | {session} | {label}",
            threshold=1.96,
            display_mode="ortho",
            cut_coords=(20, 0, 0),
            colorbar=True,
            output_file=str(fig_dir / f"{sub_mri}_{session}_{cname}_effect_map.png"),
        )
    plt.close("all")

    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed / 60:.1f} min")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="First-level GLM: remembered vs not_remembered."
    )
    parser.add_argument(
        "--subject", "-s", required=True,
        help="Subject ID (e.g., sub-13, s13, or 13).",
    )
    parser.add_argument(
        "--session", required=True,
        choices=["encoding", "postsleep", "postsleep_3runs"],
        help="Session to analyze.",
    )
    parser.add_argument(
        "--trim", action="store_true", default=False,
        help="Enable dead-period trimming (off by default). "
             "Results go to separate *_trimmed directories.",
    )
    args = parser.parse_args()
    run_first_level(args.subject, args.session, trim=args.trim)
