# -*- coding: utf-8 -*-
"""
03_encoding_vs_postsleep.py — Paired comparison: encoding vs postsleep.

Runs a paired t-test (SecondLevelModel) on first-level effect maps from
encoding and postsleep sessions. Tests whether brain engagement during
retrieval differs before vs. after sleep.

Processes both contrasts:
  - remem_vs_nonremem  (remembered > not_remembered)
  - task_vs_baseline   (all task conditions > implicit baseline)

Usage:
    python 03_encoding_vs_postsleep.py --postsleep postsleep
    python 03_encoding_vs_postsleep.py --postsleep postsleep_3runs
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
)
from nilearn.datasets import load_mni152_brain_mask, fetch_atlas_harvard_oxford
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


CONTRAST_NAMES = ["remem_vs_nonremem", "task_vs_baseline"]


def robust_abs_max(img, q: float = 99.5) -> float:
    x = img.get_fdata()
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(np.abs(x), q)) if x.size else 0.0


def run_paired_comparison(postsleep_session: str = "postsleep") -> None:
    t_start = time.time()
    print(f"{'=' * 70}")
    print(f"Paired comparison: encoding vs {postsleep_session}")
    print(f"{'=' * 70}")

    enc_root = SCRIPT_DIR / "01_first_level_glm_encoding" / "glm_maps"
    post_root = SCRIPT_DIR / f"01_first_level_glm_{postsleep_session}" / "glm_maps"

    for root, name in [(enc_root, "encoding"), (post_root, postsleep_session)]:
        if not root.exists():
            raise FileNotFoundError(
                f"{name} first-level maps not found: {root}\n"
                f"Run 01_first_level_glm.py --session {name} first."
            )

    for contrast in CONTRAST_NAMES:
        print(f"\n{'─' * 60}")
        print(f"  Contrast: {contrast}")
        print(f"{'─' * 60}")
        _run_paired_for_contrast(
            postsleep_session, enc_root, post_root, contrast
        )

    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed / 60:.1f} min")


def _run_paired_for_contrast(
    postsleep_session: str,
    enc_root: Path,
    post_root: Path,
    contrast: str,
) -> None:
    map_glob = f"sub-*/Big_GLM_{contrast}_effect.nii.gz"

    enc_maps = {
        p.parent.name: p
        for p in sorted(enc_root.glob(map_glob))
    }
    post_maps = {
        p.parent.name: p
        for p in sorted(post_root.glob(map_glob))
    }

    common_subs = sorted(set(enc_maps.keys()) & set(post_maps.keys()))
    n_subs = len(common_subs)

    if n_subs == 0:
        print(f"  No subjects found with both sessions for {contrast}. Skipping.")
        return

    enc_only = set(enc_maps.keys()) - set(post_maps.keys())
    post_only = set(post_maps.keys()) - set(enc_maps.keys())
    if enc_only:
        print(f"  Subjects with encoding only (excluded): {sorted(enc_only)}")
    if post_only:
        print(f"  Subjects with postsleep only (excluded): {sorted(post_only)}")

    print(f"Using {n_subs} subjects with both sessions.")

    out_dir = (
        SCRIPT_DIR / f"03_encoding_vs_{postsleep_session}" / "group_maps" / contrast
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build paired design matrix ---
    all_maps = (
        [str(enc_maps[s]) for s in common_subs]
        + [str(post_maps[s]) for s in common_subs]
    )

    condition = np.array([-1] * n_subs + [1] * n_subs, dtype=float)

    subject_dummies = np.zeros((2 * n_subs, n_subs), dtype=float)
    for i in range(n_subs):
        subject_dummies[i, i] = 1
        subject_dummies[n_subs + i, i] = 1

    dm = pd.DataFrame(
        np.column_stack([condition, subject_dummies]),
        columns=["postsleep_minus_encoding"] + [f"sub_{s}" for s in common_subs],
    )

    # --- MNI mask ---
    mni_mask = resample_to_img(
        load_mni152_brain_mask(),
        load_img(all_maps[0]),
        interpolation="nearest",
    )

    # --- Fit ---
    print("\nFitting paired SecondLevelModel...")
    slm = SecondLevelModel(
        smoothing_fwhm=GROUP_SMOOTHING_FWHM,
        mask_img=mni_mask,
    ).fit(all_maps, design_matrix=dm)

    paired_t = slm.compute_contrast(
        "postsleep_minus_encoding", output_type="stat"
    )
    paired_z = slm.compute_contrast(
        "postsleep_minus_encoding", output_type="z_score"
    )

    # Cohen's d
    d_data = paired_t.get_fdata() / np.sqrt(n_subs)
    paired_d = new_img_like(paired_t, d_data)

    # Corrections
    t_fdr, thr_fdr = threshold_stats_img(
        paired_t, alpha=0.05, height_control="fdr",
        two_sided=True, cluster_threshold=0,
    )
    t_bonf, thr_bonf = threshold_stats_img(
        paired_t, alpha=0.05, height_control="bonferroni",
        two_sided=True, cluster_threshold=0,
    )

    print(f"  FDR |t| threshold: {float(thr_fdr):.4f}")
    print(f"  Bonferroni |t| threshold: {float(thr_bonf):.4f}")

    # --- Non-parametric permutation test ---
    print(f"\nRunning permutation test ({N_PERMUTATIONS} perms) on paired contrast...")
    perm = non_parametric_inference(
        second_level_input=all_maps,
        design_matrix=dm,
        second_level_contrast="postsleep_minus_encoding",
        n_perm=N_PERMUTATIONS,
        two_sided_test=True,
        smoothing_fwhm=None,
        n_jobs=-1,
    )
    print("  Permutation test complete.")

    # --- Save ---
    paired_t.to_filename(out_dir / "paired_t_post_minus_enc.nii.gz")
    paired_z.to_filename(out_dir / "paired_z_post_minus_enc.nii.gz")
    paired_d.to_filename(out_dir / "paired_d_post_minus_enc.nii.gz")
    t_fdr.to_filename(out_dir / "paired_t_FDR_q05.nii.gz")
    t_bonf.to_filename(out_dir / "paired_t_BONF_a05.nii.gz")

    if isinstance(perm, dict):
        for k in ["logp_max_t", "logp_max_stat", "logp_max_size", "logp_max_tfce", "logp"]:
            if k in perm and hasattr(perm[k], "to_filename"):
                perm[k].to_filename(out_dir / f"perm_{k}.nii.gz")
                print(f"  Saved perm[{k}]")
    elif hasattr(perm, "to_filename"):
        perm.to_filename(out_dir / "perm_map.nii.gz")

    print(f"\nSaved paired-test maps to: {out_dir}")

    # --- Figures ---
    t_vmax = robust_abs_max(paired_t)

    for img, label, fname in [
        (paired_t, f"Paired t (post - enc) [{contrast}]", "viz_paired_t_unthr_glass.png"),
        (t_fdr, f"Paired t FDR q=0.05 [{contrast}]", "viz_paired_t_fdr_glass.png"),
        (t_bonf, f"Paired t Bonferroni [{contrast}]", "viz_paired_t_bonf_glass.png"),
    ]:
        thr = 0.0 if "fdr" in fname or "bonf" in fname else None
        fig = plot_glass_brain(
            img, title=label, threshold=thr,
            display_mode="lyrz", colorbar=True,
            vmax=t_vmax, symmetric_cbar=True,
        )
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close("all")

    # --- Cluster tables ---
    clusters_pos = get_clusters_table(
        paired_t, stat_threshold=T_CUTOFF_UNCORR,
        cluster_threshold=CLUSTER_THRESHOLD, two_sided=False,
    )
    clusters_neg = get_clusters_table(
        math_img("-img", img=paired_t), stat_threshold=T_CUTOFF_UNCORR,
        cluster_threshold=CLUSTER_THRESHOLD, two_sided=False,
    )
    clusters_pos.to_csv(out_dir / "clusters_positive_t.csv", index=False)
    clusters_neg.to_csv(out_dir / "clusters_negative_t.csv", index=False)

    # --- ROI summary ---
    roi_summary = []
    for roi_name, roi_cfg in ROI_DEFINITIONS.items():
        atlas = fetch_atlas_harvard_oxford(roi_cfg["atlas"], data_dir=str(SCRIPT_DIR / "nilearn_data"))
        atlas_img = load_img(atlas.maps)
        labels = atlas.labels
        hits = []
        for sub in roi_cfg["labels"]:
            hits.extend([
                i for i, lab in enumerate(labels)
                if isinstance(lab, str) and sub.lower() in lab.lower()
            ])
        expr = " + ".join([f"(img == {i})" for i in hits])
        mask = math_img(f"({expr}) > 0", img=atlas_img)
        mask_rs = resample_to_img(mask, paired_t, interpolation="nearest")

        masker = NiftiMasker(mask_img=mask_rs)
        t_vals = masker.fit_transform(paired_t).flatten()
        roi_summary.append({
            "roi": roi_name,
            "mean_t": float(np.nanmean(t_vals)),
            "max_t": float(np.nanmax(t_vals)),
            "min_t": float(np.nanmin(t_vals)),
        })
        print(f"  {roi_name}: mean t = {roi_summary[-1]['mean_t']:.4f}")

    pd.DataFrame(roi_summary).to_csv(out_dir / "roi_summary_paired.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired comparison: encoding vs postsleep."
    )
    parser.add_argument(
        "--postsleep", default="postsleep",
        choices=["postsleep", "postsleep_3runs"],
        help="Which postsleep variant to compare against encoding (default: postsleep = all 6 runs).",
    )
    args = parser.parse_args()
    run_paired_comparison(args.postsleep)
