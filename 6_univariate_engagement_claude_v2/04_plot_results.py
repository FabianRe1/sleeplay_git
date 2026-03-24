# -*- coding: utf-8 -*-
"""
04_plot_results.py — Publication-ready figures for the univariate engagement analysis.

Produces TWO comparison folders, each with subfolder per contrast:
  04_figures/encoding_vs_postsleep/remem_vs_nonremem/
  04_figures/encoding_vs_postsleep/task_vs_baseline/
  04_figures/encoding_vs_postsleep_3runs/remem_vs_nonremem/
  04_figures/encoding_vs_postsleep_3runs/task_vs_baseline/

Each subfolder contains:
  fig1  — Whole-brain glass brains (FDR-corrected) for both sessions
  fig2  — ROI mean & peak group t-values (bar plot)
  fig3  — Paired comparison glass brain (postsleep - encoding)
  fig4  — Subject-level ROI violin plots with paired lines
  fig5  — Ortho slices at hippocampus / precuneus / medial frontal
  fig6  — ROI summary table (CSV + annotated bar chart)

Usage:
    python 04_plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from nilearn.image import load_img, math_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.maskers import NiftiMasker

from config import SCRIPT_DIR, ROI_DEFINITIONS, T_CUTOFF_UNCORR

NILEARN_DATA = str(SCRIPT_DIR / "nilearn_data")

CONTRAST_NAMES = ["remem_vs_nonremem", "task_vs_baseline"]

CONTRAST_LABELS = {
    "remem_vs_nonremem": "Remembered > Not remembered",
    "task_vs_baseline": "Task > Baseline",
}

# =====================================================================
# Publication style
# =====================================================================

COLORS = {
    "encoding": "#0072B2",         # blue
    "postsleep": "#D55E00",        # vermillion
    "postsleep_3runs": "#E69F00",  # amber
    "paired": "#009E73",           # bluish green
    "ns": "#999999",
}

SESSION_LABELS = {
    "encoding": "Encoding (evening)",
    "postsleep": "Postsleep (morning, 6 runs)",
    "postsleep_3runs": "Postsleep (morning, 3 runs)",
}


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _savefig(fig, out_dir: Path, name: str) -> None:
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# =====================================================================
# ROI mask builder (reusable)
# =====================================================================

def build_roi_masks() -> dict:
    roi_masks = {}
    for roi_name, roi_cfg in ROI_DEFINITIONS.items():
        atlas = fetch_atlas_harvard_oxford(roi_cfg["atlas"], data_dir=NILEARN_DATA)
        atlas_img = load_img(atlas.maps)
        labels = atlas.labels
        hits = []
        for sub in roi_cfg["labels"]:
            hits.extend([
                i for i, lab in enumerate(labels)
                if isinstance(lab, str) and sub.lower() in lab.lower()
            ])
        expr = " + ".join([f"(img == {i})" for i in hits])
        roi_masks[roi_name] = math_img(f"({expr}) > 0", img=atlas_img)
    return roi_masks


# =====================================================================
# Figure 1: Whole-brain glass brains (FDR)
# =====================================================================

def plot_fig1_glass_brains(
    enc_session: str, post_session: str, contrast: str, out_dir: Path,
) -> None:
    sessions = [enc_session, post_session]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))

    for i, session in enumerate(sessions):
        map_dir = SCRIPT_DIR / f"02_group_level_glm_{session}" / "group_maps" / contrast
        fdr_path = map_dir / "group_t_FDR_q05.nii.gz"
        t_path = map_dir / "group_t.nii.gz"

        if not fdr_path.exists():
            print(f"  WARNING: {fdr_path} not found.")
            continue

        t_fdr = load_img(str(fdr_path))
        group_t = load_img(str(t_path))
        t_data = group_t.get_fdata()
        vmax = float(np.nanpercentile(np.abs(t_data[np.isfinite(t_data)]), 99.5))

        plot_glass_brain(
            t_fdr, threshold=0.0, display_mode="lyrz",
            colorbar=True, vmax=vmax, symmetric_cbar=True,
            axes=axes[i], plot_abs=False,
        )
        axes[i].set_title(SESSION_LABELS.get(session, session), fontsize=9)

    fig.suptitle(f"FDR-corrected group t — {CONTRAST_LABELS[contrast]}", fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir, "fig1_glass_brain_fdr")
    print("  Saved fig1: glass brain FDR maps.")


# =====================================================================
# Figure 2: ROI mean & peak t-values (grouped bars)
# =====================================================================

def plot_fig2_roi_bars(
    enc_session: str, post_session: str, contrast: str, out_dir: Path,
) -> None:
    sessions = [enc_session, post_session]
    roi_names = list(ROI_DEFINITIONS.keys())
    n_rois = len(roi_names)

    data = {}
    for session in sessions:
        csv_path = (
            SCRIPT_DIR / f"02_group_level_glm_{session}" / "group_maps"
            / contrast / "roi_summary.csv"
        )
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found.")
            continue
        data[session] = pd.read_csv(csv_path).set_index("roi")

    if len(data) < 2:
        print("  Insufficient ROI data. Skipping fig2.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    bar_width = 0.35
    x = np.arange(n_rois)

    for ax, metric, ylabel in [
        (axes[0], "mean_t", "Mean group t-value"),
        (axes[1], "max_t", "Peak group t-value"),
    ]:
        for j, session in enumerate(sessions):
            if session not in data:
                continue
            df = data[session]
            vals = [df.loc[roi, metric] if roi in df.index else 0 for roi in roi_names]
            offset = (j - 0.5) * bar_width
            ax.bar(
                x + offset, vals, bar_width,
                color=COLORS.get(session, "#333"),
                label=SESSION_LABELS.get(session, session),
                edgecolor="white", linewidth=0.5,
            )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", " ").capitalize() for r in roi_names])
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(frameon=False, fontsize=6)

    fig.suptitle(
        f"ROI group-level statistics — {CONTRAST_LABELS[contrast]}",
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "fig2_roi_mean_peak_bars")
    print("  Saved fig2: ROI mean & peak bar plots.")


# =====================================================================
# Figure 3: Paired comparison glass brain
# =====================================================================

def plot_fig3_paired(post_session: str, contrast: str, out_dir: Path) -> None:
    paired_dir = (
        SCRIPT_DIR / f"03_encoding_vs_{post_session}" / "group_maps" / contrast
    )
    fdr_path = paired_dir / "paired_t_FDR_q05.nii.gz"
    t_path = paired_dir / "paired_t_post_minus_enc.nii.gz"

    if not t_path.exists():
        print("  Paired maps not found. Skipping fig3.")
        return

    paired_t = load_img(str(t_path))
    t_data = paired_t.get_fdata()
    vmax = float(np.nanpercentile(np.abs(t_data[np.isfinite(t_data)]), 99.5))

    fig, axes = plt.subplots(2, 1, figsize=(7, 4))

    plot_glass_brain(
        paired_t, threshold=T_CUTOFF_UNCORR,
        display_mode="lyrz", colorbar=True,
        vmax=vmax, symmetric_cbar=True,
        axes=axes[0], plot_abs=False,
    )
    axes[0].set_title(
        f"Postsleep - Encoding (uncorrected |t|>{T_CUTOFF_UNCORR}) [{CONTRAST_LABELS[contrast]}]",
        fontsize=8,
    )

    if fdr_path.exists():
        t_fdr = load_img(str(fdr_path))
        plot_glass_brain(
            t_fdr, threshold=0.0,
            display_mode="lyrz", colorbar=True,
            vmax=vmax, symmetric_cbar=True,
            axes=axes[1], plot_abs=False,
        )
        axes[1].set_title(
            f"Postsleep - Encoding (FDR q=0.05) [{CONTRAST_LABELS[contrast]}]",
            fontsize=8,
        )
    else:
        axes[1].text(0.5, 0.5, "FDR map not available",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    _savefig(fig, out_dir, "fig3_paired_glass")
    print("  Saved fig3: paired comparison glass brain.")


# =====================================================================
# Figure 4: Subject-level ROI violins with paired lines & stats
# =====================================================================

def plot_fig4_roi_violins(
    enc_session: str, post_session: str, contrast: str,
    roi_masks: dict, out_dir: Path,
) -> pd.DataFrame:
    """Returns the subject-level DataFrame for reuse."""
    sessions = [enc_session, post_session]
    roi_names = list(ROI_DEFINITIONS.keys())
    effect_file = f"Big_GLM_{contrast}_effect.nii.gz"

    all_data = []
    for session in sessions:
        fl_root = SCRIPT_DIR / f"01_first_level_glm_{session}" / "glm_maps"
        if not fl_root.exists():
            continue
        sub_maps = sorted(fl_root.glob(f"sub-*/{effect_file}"))
        if not sub_maps:
            continue

        for roi_name in roi_names:
            mask_rs = resample_to_img(
                roi_masks[roi_name], load_img(str(sub_maps[0])),
                interpolation="nearest",
            )
            masker = NiftiMasker(mask_img=mask_rs)
            for sub_path in sub_maps:
                sub_id = sub_path.parent.name
                vals = masker.fit_transform(load_img(str(sub_path))).flatten()
                all_data.append({
                    "subject": sub_id,
                    "session": session,
                    "roi": roi_name,
                    "mean_effect": float(np.nanmean(vals)),
                    "peak_effect": float(np.nanmax(np.abs(vals))),
                })

    if not all_data:
        print("  No subject-level data. Skipping fig4.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    fig, axes = plt.subplots(1, len(roi_names), figsize=(3.5 * len(roi_names), 3.5))
    if len(roi_names) == 1:
        axes = [axes]

    for i, roi_name in enumerate(roi_names):
        ax = axes[i]
        df_roi = df[df["roi"] == roi_name]

        enc_data = df_roi[df_roi["session"] == enc_session].set_index("subject")["mean_effect"]
        post_data = df_roi[df_roi["session"] == post_session].set_index("subject")["mean_effect"]
        common = sorted(set(enc_data.index) & set(post_data.index))

        positions = [0, 1]
        violin_data = []
        colors_list = []
        for session in sessions:
            subset = df_roi[df_roi["session"] == session]["mean_effect"].values
            if len(subset) > 0:
                violin_data.append(subset)
                colors_list.append(COLORS.get(session, "#333"))

        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=positions,
                showmeans=True, showmedians=False, showextrema=False,
            )
            for pc, color in zip(parts["bodies"], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            parts["cmeans"].set_color("k")
            parts["cmeans"].set_linewidth(1.5)

        if common:
            enc_vals = enc_data.loc[common].values
            post_vals = post_data.loc[common].values
            for ev, pv in zip(enc_vals, post_vals):
                ax.plot([0, 1], [ev, pv], color="#888888", lw=0.4, alpha=0.4)

            t_stat, p_val = stats.ttest_rel(enc_vals, post_vals)
            y_top = max(np.max(enc_vals), np.max(post_vals)) * 1.1
            sig = _sig_label(p_val)
            ax.plot([0, 0, 1, 1], [y_top, y_top * 1.03, y_top * 1.03, y_top],
                    lw=0.8, color="k")
            ax.text(0.5, y_top * 1.04, f"{sig}\np={p_val:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            SESSION_LABELS.get(enc_session, enc_session),
            SESSION_LABELS.get(post_session, post_session),
        ], fontsize=6.5)
        ax.set_title(roi_name.replace("_", " ").capitalize(), fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel(f"Mean effect size\n({CONTRAST_LABELS[contrast]})")

    fig.suptitle(
        f"Subject-level ROI effects — {CONTRAST_LABELS[contrast]}",
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "fig4_roi_violins_paired")
    print("  Saved fig4: ROI violin plots with paired lines.")
    return df


# =====================================================================
# Figure 5: Ortho slices
# =====================================================================

def plot_fig5_ortho(
    enc_session: str, post_session: str, contrast: str, out_dir: Path,
) -> None:
    coords_of_interest = [
        (0, -24, -16),   # hippocampus
        (0, -60, 30),    # precuneus / PCC
        (0, 0, 50),      # medial frontal
    ]
    coord_labels = ["Hippocampus", "Precuneus / PCC", "Medial frontal"]
    sessions = [enc_session, post_session]

    fig, axes = plt.subplots(
        len(coords_of_interest), len(sessions),
        figsize=(6 * len(sessions), 2.0 * len(coords_of_interest)),
    )

    for col, session in enumerate(sessions):
        map_dir = SCRIPT_DIR / f"02_group_level_glm_{session}" / "group_maps" / contrast
        t_path = map_dir / "group_t.nii.gz"
        fdr_path = map_dir / "group_t_FDR_q05.nii.gz"
        if not t_path.exists():
            continue

        group_t = load_img(str(t_path))
        t_data = group_t.get_fdata()
        vmax = float(np.nanpercentile(np.abs(t_data[np.isfinite(t_data)]), 99.5))
        img_to_plot = load_img(str(fdr_path)) if fdr_path.exists() else group_t
        threshold = 0.0 if fdr_path.exists() else T_CUTOFF_UNCORR

        for row, (coords, label) in enumerate(zip(coords_of_interest, coord_labels)):
            ax = axes[row, col]
            plot_stat_map(
                img_to_plot, threshold=threshold,
                display_mode="ortho", cut_coords=coords,
                colorbar=(row == 0 and col == len(sessions) - 1),
                vmax=vmax, symmetric_cbar=True, axes=ax,
            )
            if col == 0:
                ax.set_title(f"{label}", fontsize=8, loc="left")

        axes[0, col].set_title(
            f"{SESSION_LABELS.get(session, session)}",
            fontsize=9, fontweight="bold",
        )

    fig.suptitle(f"Ortho slices — {CONTRAST_LABELS[contrast]}", fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir, "fig5_ortho_slices")
    print("  Saved fig5: ortho slices.")


# =====================================================================
# Figure 6: ROI summary comparison (mean + peak, with change arrows)
# =====================================================================

def plot_fig6_roi_summary(
    enc_session: str, post_session: str, contrast: str,
    subject_df: pd.DataFrame, out_dir: Path,
) -> None:
    """Combined ROI summary: group-level mean/peak + paired change + stats."""
    roi_names = list(ROI_DEFINITIONS.keys())
    sessions = [enc_session, post_session]

    group_data = {}
    for session in sessions:
        csv_path = (
            SCRIPT_DIR / f"02_group_level_glm_{session}" / "group_maps"
            / contrast / "roi_summary.csv"
        )
        if csv_path.exists():
            group_data[session] = pd.read_csv(csv_path).set_index("roi")

    paired_csv = (
        SCRIPT_DIR / f"03_encoding_vs_{post_session}" / "group_maps"
        / contrast / "roi_summary_paired.csv"
    )
    paired_data = pd.read_csv(paired_csv).set_index("roi") if paired_csv.exists() else None

    if len(group_data) < 2:
        print("  Insufficient data for fig6.")
        return

    rows = []
    for roi in roi_names:
        row = {"ROI": roi.replace("_", " ").capitalize()}
        for session in sessions:
            label = "Enc" if session == enc_session else "Post"
            df = group_data[session]
            if roi in df.index:
                row[f"{label} mean t"] = df.loc[roi, "mean_t"]
                row[f"{label} peak t"] = df.loc[roi, "max_t"]
                row[f"{label} min t"] = df.loc[roi, "min_t"]

        if paired_data is not None and roi in paired_data.index:
            row["Paired mean t (post-enc)"] = paired_data.loc[roi, "mean_t"]

        if not subject_df.empty:
            enc_vals = subject_df[
                (subject_df["roi"] == roi) & (subject_df["session"] == enc_session)
            ].set_index("subject")["mean_effect"]
            post_vals = subject_df[
                (subject_df["roi"] == roi) & (subject_df["session"] == post_session)
            ].set_index("subject")["mean_effect"]
            common = sorted(set(enc_vals.index) & set(post_vals.index))
            if common:
                t_stat, p_val = stats.ttest_rel(enc_vals.loc[common], post_vals.loc[common])
                row["t-stat (paired)"] = t_stat
                row["p-value (paired)"] = p_val
                row["Mean change"] = float(post_vals.loc[common].mean() - enc_vals.loc[common].mean())
                row["N subjects"] = len(common)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "roi_comparison_summary.csv", index=False)
    print(f"  Saved ROI comparison table to roi_comparison_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    x = np.arange(len(roi_names))
    bar_w = 0.35

    # Panel 1: Mean t
    ax = axes[0]
    for j, session in enumerate(sessions):
        label = SESSION_LABELS.get(session, session)
        df = group_data[session]
        vals = [df.loc[roi, "mean_t"] if roi in df.index else 0 for roi in roi_names]
        offset = (j - 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, color=COLORS.get(session, "#333"),
               label=label, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Group mean t-value")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").capitalize() for r in roi_names])
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("Mean t in ROI", fontweight="bold")

    # Panel 2: Peak t
    ax = axes[1]
    for j, session in enumerate(sessions):
        label = SESSION_LABELS.get(session, session)
        df = group_data[session]
        vals = [df.loc[roi, "max_t"] if roi in df.index else 0 for roi in roi_names]
        offset = (j - 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, color=COLORS.get(session, "#333"),
               label=label, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Group peak t-value")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").capitalize() for r in roi_names])
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("Peak t in ROI", fontweight="bold")

    # Panel 3: Paired change (postsleep - encoding) with error bars
    ax = axes[2]
    if not subject_df.empty:
        change_means = []
        change_sems = []
        p_vals = []
        for roi in roi_names:
            enc_vals = subject_df[
                (subject_df["roi"] == roi) & (subject_df["session"] == enc_session)
            ].set_index("subject")["mean_effect"]
            post_vals = subject_df[
                (subject_df["roi"] == roi) & (subject_df["session"] == post_session)
            ].set_index("subject")["mean_effect"]
            common = sorted(set(enc_vals.index) & set(post_vals.index))
            if common:
                diffs = post_vals.loc[common].values - enc_vals.loc[common].values
                change_means.append(np.mean(diffs))
                change_sems.append(stats.sem(diffs))
                _, p = stats.ttest_rel(enc_vals.loc[common], post_vals.loc[common])
                p_vals.append(p)
            else:
                change_means.append(0)
                change_sems.append(0)
                p_vals.append(1.0)

        bar_colors = [COLORS["paired"]] * len(roi_names)
        bars = ax.bar(x, change_means, 0.5, yerr=change_sems,
                      color=bar_colors, edgecolor="white", linewidth=0.5,
                      capsize=3, error_kw={"lw": 0.8})

        for xi, (m, p) in enumerate(zip(change_means, p_vals)):
            y_pos = m + change_sems[xi] + 0.02 if m >= 0 else m - change_sems[xi] - 0.05
            ax.text(xi, y_pos, _sig_label(p), ha="center", fontsize=7, fontweight="bold")

    ax.set_ylabel("Mean effect change\n(postsleep - encoding)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").capitalize() for r in roi_names])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Paired change", fontweight="bold")

    fig.suptitle(
        f"ROI comparison — {CONTRAST_LABELS[contrast]}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "fig6_roi_summary")
    print("  Saved fig6: ROI summary comparison.")


# =====================================================================
# Main — runs both comparisons x both contrasts
# =====================================================================

def run_comparison(
    enc_session: str, post_session: str, contrast: str, roi_masks: dict,
) -> None:
    out_dir = SCRIPT_DIR / "04_figures" / f"encoding_vs_{post_session}" / contrast
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Comparison: {enc_session} vs {post_session} | {CONTRAST_LABELS[contrast]}")
    print(f"Output: {out_dir}")
    print(f"{'=' * 60}\n")

    plot_fig1_glass_brains(enc_session, post_session, contrast, out_dir)
    plot_fig2_roi_bars(enc_session, post_session, contrast, out_dir)
    plot_fig3_paired(post_session, contrast, out_dir)
    subject_df = plot_fig4_roi_violins(enc_session, post_session, contrast, roi_masks, out_dir)
    plot_fig5_ortho(enc_session, post_session, contrast, out_dir)
    plot_fig6_roi_summary(enc_session, post_session, contrast, subject_df, out_dir)


def main() -> None:
    set_pub_style()
    print("Building ROI masks...")
    roi_masks = build_roi_masks()

    for contrast in CONTRAST_NAMES:
        run_comparison("encoding", "postsleep", contrast, roi_masks)
        run_comparison("encoding", "postsleep_3runs", contrast, roi_masks)

    print(f"\nAll figures saved under: {SCRIPT_DIR / '04_figures'}")


if __name__ == "__main__":
    main()
