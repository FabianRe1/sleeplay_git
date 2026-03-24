# -*- coding: utf-8 -*-
"""
02_group_level_glm.py — Second-level (group) analysis.

Collects first-level effect maps from all subjects, runs a one-sample
t-test (SecondLevelModel), applies FDR/Bonferroni/permutation correction,
extracts ROI statistics, and saves cluster tables.

Usage:
    python 02_group_level_glm.py --session encoding
    python 02_group_level_glm.py --session postsleep
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.image import (
    load_img,
    new_img_like,
    math_img,
    mean_img,
)
from nilearn.datasets import (
    load_mni152_brain_mask,
    fetch_atlas_harvard_oxford,
)
from nilearn.image import resample_to_img
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.reporting import get_clusters_table
from nilearn.maskers import NiftiMasker

from config import (
    SCRIPT_DIR,
    GROUP_SMOOTHING_FWHM,
    N_PERMUTATIONS,
    CLUSTER_THRESHOLD,
    T_CUTOFF_UNCORR,
    ROI_DEFINITIONS,
)


# =====================================================================
# Helpers
# =====================================================================

def robust_abs_max(img, q: float = 99.5) -> float:
    x = img.get_fdata()
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(np.abs(x), q)) if x.size else 0.0


def save_stat_figure(
    img, title: str, fname: Path,
    bg_img=None, threshold=None, vmax=None,
) -> None:
    fig = plot_stat_map(
        img, bg_img=bg_img, title=title,
        threshold=threshold, display_mode="ortho",
        cut_coords=(20, 0, 0), colorbar=True,
        vmax=vmax, symmetric_cbar=True,
    )
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close("all")


def save_glass_figure(
    img, title: str, fname: Path,
    threshold=None, vmax=None,
) -> None:
    fig = plot_glass_brain(
        img, title=title, threshold=threshold,
        display_mode="lyrz", colorbar=True,
        vmax=vmax, symmetric_cbar=True,
    )
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close("all")


def build_roi_mask(atlas_name: str, label_substrings: list[str]):
    """Build a binary ROI mask from Harvard-Oxford atlas labels."""
    atlas = fetch_atlas_harvard_oxford(atlas_name, data_dir=str(SCRIPT_DIR / "nilearn_data"))
    atlas_img = load_img(atlas.maps)
    labels = atlas.labels

    hits = []
    for sub in label_substrings:
        matched = [
            i for i, lab in enumerate(labels)
            if isinstance(lab, str) and sub.lower() in lab.lower()
        ]
        if not matched:
            print(f"  WARNING: No labels matched '{sub}' in {atlas_name}")
        hits.extend(matched)

    if not hits:
        raise ValueError(
            f"No atlas labels matched any of {label_substrings} in {atlas_name}.\n"
            f"Available labels: {labels[:30]}"
        )

    expr = " + ".join([f"(img == {i})" for i in hits])
    return math_img(f"({expr}) > 0", img=atlas_img)


# =====================================================================
# Main
# =====================================================================

# Contrasts produced by the first-level script
CONTRAST_NAMES = ["remem_vs_nonremem", "task_vs_baseline"]


def run_group_level(session: str) -> None:
    t_start = time.time()
    print(f"{'=' * 70}")
    print(f"Group-level GLM: session={session}")
    print(f"{'=' * 70}")

    fl_root = SCRIPT_DIR / f"01_first_level_glm_{session}" / "glm_maps"
    if not fl_root.exists():
        raise FileNotFoundError(
            f"First-level output not found: {fl_root}\n"
            f"Run 01_first_level_glm.py --session {session} first."
        )

    for contrast in CONTRAST_NAMES:
        print(f"\n{'─' * 60}")
        print(f"  Contrast: {contrast}")
        print(f"{'─' * 60}")
        _run_group_for_contrast(session, fl_root, contrast)

    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed / 60:.1f} min")


def _run_group_for_contrast(session: str, fl_root: Path, contrast: str) -> None:
    sub_eff_maps = sorted(
        fl_root.glob(f"sub-*/Big_GLM_{contrast}_effect.nii.gz")
    )
    if not sub_eff_maps:
        print(f"  No maps found for {contrast}. Skipping.")
        return

    subjects = [p.parent.name for p in sub_eff_maps]
    n_subs = len(subjects)
    print(f"Found {n_subs} subjects: {subjects[:5]}{'...' if n_subs > 5 else ''}")

    out_dir = SCRIPT_DIR / f"02_group_level_glm_{session}" / "group_maps" / contrast
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MNI mask ---
    mni_mask = resample_to_img(
        load_mni152_brain_mask(),
        load_img(str(sub_eff_maps[0])),
        interpolation="nearest",
    )

    # --- Fit second-level model ---
    design_matrix = pd.DataFrame({"intercept": 1.0}, index=subjects)

    print("\nFitting SecondLevelModel...")
    slm = SecondLevelModel(
        smoothing_fwhm=GROUP_SMOOTHING_FWHM,
        mask_img=mni_mask,
    ).fit(
        [str(p) for p in sub_eff_maps],
        design_matrix=design_matrix,
    )

    group_t = slm.compute_contrast("intercept", output_type="stat")
    group_z = slm.compute_contrast("intercept", output_type="z_score")
    group_p = slm.compute_contrast("intercept", output_type="p_value")

    # Cohen's d = t / sqrt(n)
    d_data = group_t.get_fdata() / np.sqrt(n_subs)
    group_d = new_img_like(group_t, d_data)

    print("  Group maps computed.")

    # --- Multiple comparisons correction (parametric) ---
    t_fdr, thr_fdr = threshold_stats_img(
        group_t, alpha=0.05, height_control="fdr",
        two_sided=True, cluster_threshold=0,
    )
    t_bonf, thr_bonf = threshold_stats_img(
        group_t, alpha=0.05, height_control="bonferroni",
        two_sided=True, cluster_threshold=0,
    )

    print(f"  FDR |t| threshold: {float(thr_fdr):.4f}")
    print(f"  Bonferroni |t| threshold: {float(thr_bonf):.4f}")

    # --- Non-parametric permutation test ---
    print(f"\nRunning permutation test ({N_PERMUTATIONS} perms)...")
    perm = non_parametric_inference(
        second_level_input=[str(p) for p in sub_eff_maps],
        design_matrix=design_matrix,
        second_level_contrast="intercept",
        n_perm=N_PERMUTATIONS,
        two_sided_test=True,
        smoothing_fwhm=None,
        n_jobs=-1,
    )
    print("  Permutation test complete.")

    # --- Save all maps ---
    group_t.to_filename(out_dir / "group_t.nii.gz")
    group_z.to_filename(out_dir / "group_z.nii.gz")
    group_p.to_filename(out_dir / "group_p.nii.gz")
    group_d.to_filename(out_dir / "group_cohens_d.nii.gz")
    t_fdr.to_filename(out_dir / "group_t_FDR_q05.nii.gz")
    t_bonf.to_filename(out_dir / "group_t_BONF_a05.nii.gz")

    if isinstance(perm, dict):
        for k in ["logp_max_t", "logp_max_stat", "logp_max_size", "logp_max_tfce", "logp"]:
            if k in perm and hasattr(perm[k], "to_filename"):
                perm[k].to_filename(out_dir / f"perm_{k}.nii.gz")
                print(f"  Saved perm[{k}]")
    elif hasattr(perm, "to_filename"):
        perm.to_filename(out_dir / "perm_map.nii.gz")

    print(f"\nSaved group maps to: {out_dir}")

    # --- Figures ---
    t_vmax = robust_abs_max(group_t)

    save_stat_figure(
        group_t, f"Group t (unthresholded) — {session}",
        out_dir / "viz_group_t_unthr.png", vmax=t_vmax,
    )
    save_glass_figure(
        group_t, f"Group t (unthresholded) — {session}",
        out_dir / "viz_group_t_unthr_glass.png", vmax=t_vmax,
    )
    save_stat_figure(
        t_fdr, f"Group t (FDR q=0.05) — {session}",
        out_dir / "viz_group_t_fdr.png", threshold=0.0, vmax=t_vmax,
    )
    save_glass_figure(
        t_fdr, f"Group t (FDR q=0.05) — {session}",
        out_dir / "viz_group_t_fdr_glass.png", threshold=0.0, vmax=t_vmax,
    )
    save_stat_figure(
        t_bonf, f"Group t (Bonferroni) — {session}",
        out_dir / "viz_group_t_bonf.png", threshold=0.0, vmax=t_vmax,
    )
    save_glass_figure(
        group_d, f"Group Cohen's d — {session}",
        out_dir / "viz_group_d_glass.png", vmax=robust_abs_max(group_d),
    )

    print("  Saved figures.")

    # --- Cluster tables ---
    clusters_pos = get_clusters_table(
        group_t, stat_threshold=T_CUTOFF_UNCORR,
        cluster_threshold=CLUSTER_THRESHOLD, two_sided=False,
    )
    clusters_neg = get_clusters_table(
        math_img("-img", img=group_t), stat_threshold=T_CUTOFF_UNCORR,
        cluster_threshold=CLUSTER_THRESHOLD, two_sided=False,
    )
    clusters_pos.to_csv(out_dir / "clusters_positive_t.csv", index=False)
    clusters_neg.to_csv(out_dir / "clusters_negative_t.csv", index=False)
    print(f"  Cluster tables saved ({len(clusters_pos)} pos, {len(clusters_neg)} neg clusters).")

    # --- ROI analysis ---
    print("\nROI analysis:")
    roi_dir = out_dir / "roi_maps"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_summary = []

    for roi_name, roi_cfg in ROI_DEFINITIONS.items():
        print(f"\n  {roi_name}:")
        mask = build_roi_mask(roi_cfg["atlas"], roi_cfg["labels"])
        mask_rs = resample_to_img(mask, group_t, interpolation="nearest")

        # Mask t-map
        t_roi = math_img("t * m", t=group_t, m=mask_rs)
        t_roi.to_filename(roi_dir / f"group_t_{roi_name}.nii.gz")

        # Mask corrected maps
        t_fdr_roi = math_img("t * m", t=t_fdr, m=mask_rs)
        t_bonf_roi = math_img("t * m", t=t_bonf, m=mask_rs)
        t_fdr_roi.to_filename(roi_dir / f"group_t_FDR_{roi_name}.nii.gz")
        t_bonf_roi.to_filename(roi_dir / f"group_t_BONF_{roi_name}.nii.gz")

        # Summary statistics
        masker = NiftiMasker(mask_img=mask_rs)
        t_vals = masker.fit_transform(group_t).flatten()
        mean_t = float(np.nanmean(t_vals))
        max_t = float(np.nanmax(t_vals))
        min_t = float(np.nanmin(t_vals))
        print(f"    Mean t = {mean_t:.4f}, Max t = {max_t:.4f}, Min t = {min_t:.4f}")

        roi_summary.append({
            "roi": roi_name,
            "mean_t": mean_t,
            "max_t": max_t,
            "min_t": min_t,
            "n_voxels": len(t_vals),
        })

    pd.DataFrame(roi_summary).to_csv(out_dir / "roi_summary.csv", index=False)
    print(f"\n  ROI summary saved to {out_dir / 'roi_summary.csv'}")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group-level GLM: one-sample t-test on effect maps."
    )
    parser.add_argument(
        "--session", required=True,
        choices=["encoding", "postsleep", "postsleep_3runs"],
        help="Session to analyze.",
    )
    args = parser.parse_args()
    run_group_level(args.session)
