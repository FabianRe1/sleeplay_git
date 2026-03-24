# -*- coding: utf-8 -*-
"""Publication-ready figures for the Sleeplay decoding & SODA pipeline.

Generates clean, journal-quality figures from the output of steps 4.2–4.7.
Follows the visual styling of the reference paper (Renz et al., LCI Replay).

Usage
-----
    python 5.0_publication_figures.py [ROI] [presleep|postsleep]

    Defaults: ROI=vis, presleep

Output
------
    5.0_publication_figures/<ROI>/  — SVG + PDF figures
"""

import ast
import os
import sys
import glob
import re
from os.path import join as opj

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, ttest_rel, sem as scipy_sem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "4_decoding_claude"))
from utils import fdr_bh, ensure_dir

# =====================================================================
# STYLE SETUP  (matches reference paper: Renz et al.)
# =====================================================================

def setup_style():
    """Set publication-quality matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,       # editable text in PDFs
        "ps.fonttype": 42,
    })


# =====================================================================
# COLOR PALETTE
# =====================================================================

COLORS = {
    "retrieved": "#3274A1",       # blue
    "control": "#E1812C",         # orange
    "matched": "#034e87",         # dark blue
    "unmatched": "#888888",       # grey
    "shuffled_ret": "#8CB8D0",    # light blue
    "shuffled_ctrl": "#F0C08A",   # light orange
    "grand_avg": "#000000",
    "chance": "#CC0000",
    "sig_star": "#000000",
    "sem_band_alpha": 0.25,
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _save(fig, out_dir, name):
    """Save figure as SVG and PDF."""
    for ext in ("svg", "pdf"):
        fig.savefig(opj(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved: {name}.svg/.pdf")


def _despine(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold panel label (a, b, c, ...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


# =====================================================================
# FIGURE 1: CLASSIFIER PERFORMANCE PER ROI
# =====================================================================

def fig_classifier_performance(results_dir: str, out_dir: str, condition: int = 1):
    """Bar plot of decoding accuracy per ROI with subject dots.

    Parameters
    ----------
    results_dir : str
        Path to ``4.1_train_visual_classifier/df_classifier_results_csv/``
    condition : int
        1=scenes, 2=objects
    """
    from scipy.stats import ttest_1samp

    cond_label = {1: "Scenes", 2: "Objects"}.get(condition, f"Cond {condition}")

    # Load all CSVs for the default parameters
    pattern = f"only_scenes1_objects2_both3_{condition}_*_Smooth_4_Threshold_30_reliability_threshold_70_tmpsmoothing_3.csv"
    paths = sorted(glob.glob(opj(results_dir, pattern)))
    if not paths:
        print(f"  [SKIP] No CSVs for condition={condition}")
        return

    # Collect per-subject, per-mask accuracies
    mask_data = {}
    for p in paths:
        df = pd.read_csv(p, index_col=0)
        fname = os.path.basename(p)
        # Extract mask and subject
        m_mask = re.search(r"Mask_([a-z_]+)_Smooth", fname)
        m_subj = re.search(r"_(s\d+)_Mask", fname)
        if not m_mask or not m_subj:
            continue
        mask, subj = m_mask.group(1), m_subj.group(1)
        acc = df["label_match"].mean() * 100.0
        mask_data.setdefault(mask, {})[subj] = acc

    if not mask_data:
        return

    masks = sorted(mask_data.keys())
    n_masks = len(masks)
    n_classes = 10  # scenes or objects
    chance = 100.0 / n_classes

    fig, ax = plt.subplots(figsize=(max(3.5, n_masks * 0.65), 3.0))

    x = np.arange(n_masks)
    means, sems_vals = [], []
    for mask in masks:
        vals = np.array(list(mask_data[mask].values()))
        means.append(vals.mean())
        sems_vals.append(scipy_sem(vals) if len(vals) > 1 else 0)

    ax.bar(x, means, yerr=sems_vals, width=0.6,
           color=COLORS["retrieved"], edgecolor="#333", linewidth=0.8,
           capsize=3, error_kw={"linewidth": 1.0, "capthick": 1.0},
           alpha=0.8, zorder=1)

    # Subject dots
    rng = np.random.default_rng(42)
    for i, mask in enumerate(masks):
        vals = np.array(list(mask_data[mask].values()))
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", s=12, alpha=0.5,
                   linewidths=0, zorder=3)

        # T-test vs chance
        if len(vals) > 2:
            _, p = ttest_1samp(vals, chance)
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(i, means[i] + sems_vals[i] + 2, star,
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(chance, color=COLORS["chance"], ls="--", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(masks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Classifier performance — {cond_label}")
    _despine(ax)

    fig.tight_layout()
    _save(fig, out_dir, f"fig1_classifier_accuracy_{cond_label.lower()}")


# =====================================================================
# FIGURE 2: RETRIEVAL DECODING TIMECOURSE
# =====================================================================

def fig_retrieval_timecourse(soda_csv_dir: str, roi: str, out_dir: str, pre_post: str):
    """Matched vs unmatched classifier probability timecourse with FDR stars."""
    csv_files = sorted(glob.glob(opj(soda_csv_dir, "*.csv")))
    csv_files = [f for f in csv_files if f"ROI_{roi}" in f]
    if not csv_files:
        print(f"  [SKIP] No SODA input CSVs for ROI={roi}")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    stacked = pd.concat(dfs, ignore_index=True)

    d = stacked[["subject", "tr", "target_classifier_matched", "probability_from_classifier"]].copy()
    for col in d.columns[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    d["target_classifier_matched"] = d["target_classifier_matched"].astype(int)

    subj_mean = d.groupby(["subject", "target_classifier_matched", "tr"], as_index=False)[
        "probability_from_classifier"
    ].mean()

    def _pivot(val):
        tmp = subj_mean[subj_mean["target_classifier_matched"] == val]
        return tmp.pivot(index="tr", columns="subject", values="probability_from_classifier").sort_index()

    wide_m = _pivot(1)
    wide_u = _pivot(0)

    fig, ax = plt.subplots(figsize=(3.75, 3.0))

    for label, wide, color in [
        ("Unmatched", wide_u, COLORS["unmatched"]),
        ("Matched", wide_m, COLORS["matched"]),
    ]:
        mean = wide.mean(axis=1)
        se = wide.std(axis=1, ddof=1) / np.sqrt(wide.count(axis=1))
        ax.fill_between(mean.index, (mean - se).values, (mean + se).values,
                        color=color, alpha=COLORS["sem_band_alpha"], linewidth=0)
        ax.plot(mean.index, mean.values, color=color, lw=1.5, label=label)

    # FDR-corrected paired t-tests per TR
    all_trs = sorted(set(wide_m.index) & set(wide_u.index))
    pvals = np.full(len(all_trs), np.nan)
    for i, tr in enumerate(all_trs):
        a, b = wide_m.loc[tr], wide_u.loc[tr]
        ok = ~a.isna() & ~b.isna()
        if ok.sum() >= 3:
            _, pvals[i] = ttest_rel(a[ok].values, b[ok].values)

    rejected, _ = fdr_bh(pvals)
    y_star = ax.get_ylim()[1] * 0.97
    sig_trs = [tr for tr, rej in zip(all_trs, rejected) if rej]
    if sig_trs:
        ax.scatter(sig_trs, [y_star] * len(sig_trs), marker="*", s=30,
                   color=COLORS["sig_star"], zorder=10)

    ax.set_xlabel("TR from retrieval onset")
    ax.set_ylabel("P(target class)")
    ax.set_title(f"Retrieval decoding — {roi.upper()}")
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)

    fig.tight_layout()
    _save(fig, out_dir, f"fig2_retrieval_timecourse_{roi}_{pre_post}")


# =====================================================================
# FIGURE 3: SODA TIMECOURSE (Retrieved vs Control)
# =====================================================================

def fig_soda_timecourse(soda_results_dir: str, roi: str, out_dir: str, pre_post: str):
    """Mean SODA slope timecourse: retrieved vs control vs shuffle variants ± SEM."""
    csv_path = opj(soda_results_dir, f"{roi}_soda_results_per_combination.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] SODA results not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    soda_cols = [
        "retrieved_soda", "control_soda",
        "retrieved_soda_shuffled", "retrieved_soda_shuffled_selected",
    ]
    for col in ["tr"] + soda_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available = [c for c in soda_cols if c in df.columns]
    subj_tr = df.groupby(["subject", "tr"], as_index=False)[available].mean()
    agg = subj_tr.groupby("tr")[available].agg(["mean", "std", "count"])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    line_defs = [
        ("retrieved_soda",                   "Retrieved",      COLORS["retrieved"], 1.5, "-"),
        ("control_soda",                     "Control",        COLORS["control"],   1.5, "-"),
        ("retrieved_soda_shuffled",          "Shuffled",       COLORS["shuffled_ret"], 1.2, "--"),
        ("retrieved_soda_shuffled_selected", "Shuf (strict)",  "#A0D0A0",           1.2, "--"),
    ]

    for soda_col, label, color, lw, ls in line_defs:
        if soda_col not in available:
            continue
        m = agg[(soda_col, "mean")]
        se = agg[(soda_col, "std")] / np.sqrt(agg[(soda_col, "count")])
        ax.fill_between(m.index, (m - se).values, (m + se).values,
                        color=color, alpha=COLORS["sem_band_alpha"], linewidth=0)
        ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label)

    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel("TR from retrieval onset")
    ax.set_ylabel("SODA slope")
    ax.set_title(f"Sequential reactivation — {roi.upper()}")
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)

    fig.tight_layout()
    _save(fig, out_dir, f"fig3_soda_timecourse_{roi}_{pre_post}")


# =====================================================================
# FIGURE 4: SODA BY RETRIEVED POSITION (5-panel)
# =====================================================================

def fig_soda_by_position(soda_results_dir: str, roi: str, out_dir: str, pre_post: str):
    """5-panel SODA timecourse split by cue position."""
    csv_path = opj(soda_results_dir, f"{roi}_soda_results_per_combination.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] SODA results not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    for col in ["tr", "retrieved_soda", "control_soda", "retrieved_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    positions = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8), sharey=True, sharex=True)

    for ax, pos in zip(axes, positions):
        df_pos = df[df["retrieved_position"] == pos]
        if df_pos.empty:
            ax.set_title(f"Pos {pos}\n(no data)", fontsize=10)
            _despine(ax)
            continue

        subj_tr = df_pos.groupby(["subject", "tr"], as_index=False)[
            ["retrieved_soda", "control_soda"]
        ].mean()
        agg = subj_tr.groupby("tr")[["retrieved_soda", "control_soda"]].agg(["mean", "std", "count"])

        for soda_col, color in [
            ("retrieved_soda", COLORS["retrieved"]),
            ("control_soda", COLORS["control"]),
        ]:
            m = agg[(soda_col, "mean")]
            se = agg[(soda_col, "std")] / np.sqrt(agg[(soda_col, "count")])
            ax.fill_between(m.index, (m - se).values, (m + se).values,
                            color=color, alpha=COLORS["sem_band_alpha"], linewidth=0)
            ax.plot(m.index, m.values, color=color, lw=1.5)

        ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
        n_sub = subj_tr["subject"].nunique()
        ax.set_title(f"Pos {pos} (N={n_sub})", fontsize=10)
        ax.set_xlabel("TR", fontsize=9)
        _despine(ax)

    axes[0].set_ylabel("SODA slope", fontsize=10)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["retrieved"], lw=1.5, label="Retrieved"),
        Line2D([0], [0], color=COLORS["control"], lw=1.5, label="Control"),
    ]
    axes[-1].legend(handles=legend_elements, frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(f"SODA by cue position — {roi.upper()} ({pre_post})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"fig4_soda_by_position_{roi}_{pre_post}")


# =====================================================================
# GENERIC 6-CONDITION BOXPLOT
# =====================================================================

# Condition definitions used across all metric figures
CONDITION_DEFS = [
    ("retrieved",          "Retrieved",      "#3274A1"),
    ("control",            "Control",        "#E1812C"),
    ("shuf_retrieved",     "Shuf Ret",       "#8CB8D0"),
    ("shuf_control",       "Shuf Ctrl",      "#F0C08A"),
    ("shuf_sel_retrieved", "Shuf Sel Ret",   "#A0D0A0"),
    ("shuf_sel_control",   "Shuf Sel Ctrl",  "#D0A0D0"),
]


def _boxplot_6conditions(
    subj_df: pd.DataFrame,
    col_map: dict[str, str],
    ax,
    ylabel: str = "",
    title: str = "",
):
    """Draw a boxplot with subject dots for up to 6 conditions.

    Parameters
    ----------
    subj_df : DataFrame
        Subject-level aggregated data (one row per subject).
    col_map : dict
        Maps condition key (e.g. "retrieved") to column name in subj_df.
    """
    available = [(key, label, color) for key, label, color in CONDITION_DEFS
                 if key in col_map and col_map[key] in subj_df.columns]
    if len(available) < 2:
        ax.set_title(f"{title}\n(insufficient data)")
        _despine(ax)
        return

    plot_data = [subj_df[col_map[key]].dropna().values for key, _, _ in available]
    plot_labels = [label for _, label, _ in available]
    plot_colors = [color for _, _, color in available]

    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)
    for whisker in bp["whiskers"]:
        whisker.set_color("#666")
    for cap in bp["caps"]:
        cap.set_color("#666")

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   color=color, edgecolor="k", linewidth=0.5,
                   s=20, zorder=5, alpha=0.8)

    # Paired t-test: retrieved vs control (first two conditions)
    if len(plot_data[0]) >= 3 and len(plot_data[1]) >= 3:
        n = min(len(plot_data[0]), len(plot_data[1]))
        t_stat, p_val = ttest_rel(plot_data[0][:n], plot_data[1][:n])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        all_vals = np.concatenate(plot_data)
        y_max = np.nanmax(all_vals)
        y_min = np.nanmin(all_vals)
        bracket_y = y_max + (y_max - y_min) * 0.08
        bracket_h = (y_max - y_min) * 0.03
        ax.plot([0, 0, 1, 1],
                [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                lw=1.0, color="black")
        ax.text(0.5, bracket_y + bracket_h * 2, sig,
                ha="center", va="bottom", fontsize=10)
        title_suffix = f"\nRet vs Ctrl: t={t_stat:.2f}, p={p_val:.3f}"
    else:
        title_suffix = ""

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}{title_suffix}", fontsize=10)
    _despine(ax)


# =====================================================================
# FIGURE 5: ABSOLUTE SLOPE (unsigned SODA)
# =====================================================================

def _safe_parse_prob_list(val):
    """Parse a probability list from CSV.

    Handles two formats:
      - Python list: [0.257, 0.001, ...]
      - NumPy repr:  [np.float64(0.257), np.float64(0.001), ...]
    """
    if isinstance(val, list):
        return [float(x) for x in val if not (isinstance(x, float) and np.isnan(x))]
    if not isinstance(val, str) or not val.strip():
        return []
    # Try ast.literal_eval first (plain Python lists)
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    # Fall back: extract all numbers via regex (handles np.float64(...) format)
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", val)
    if nums:
        return [float(x) for x in nums]
    return []


def fig_absolute_slope(soda_results_dir: str, roi: str, out_dir: str, pre_post: str):
    """Boxplot of absolute SODA slope + scatterplot of mean classifier probability."""
    csv_path = opj(soda_results_dir, f"{roi}_soda_results_per_combination.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] SODA results not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    abs_cols = {
        "retrieved":          "retrieved_soda_abs",
        "control":            "control_soda_abs",
        "shuf_retrieved":     "retrieved_soda_shuffled_abs",
        "shuf_control":       "control_soda_shuffled_abs",
        "shuf_sel_retrieved": "retrieved_soda_shuffled_selected_abs",
        "shuf_sel_control":   "control_soda_shuffled_selected_abs",
    }
    for col in abs_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [c for c in abs_cols.values() if c in df.columns]
    subj = df.groupby("subject")[available_cols].mean().dropna()

    # --- Compute mean classifier probability per trial ---
    prob_cols_map = {
        "retrieved":          "retrieved_probs",
        "control":            "control_probs",
        "shuf_retrieved":     "retrieved_probs_shuffle",
        "shuf_sel_retrieved": "retrieved_probs_shuffle_selected",
    }
    has_probs = any(c in df.columns for c in prob_cols_map.values())

    if has_probs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5.5, 3.5))
        axes = [axes]

    # Panel 1: absolute slope boxplot
    _boxplot_6conditions(subj, abs_cols, axes[0],
                         ylabel="|SODA slope|",
                         title=f"Absolute slope — {roi.upper()}")

    # Panel 2: mean classifier probability scatterplot
    if has_probs and len(axes) > 1:
        ax = axes[1]

        # Compute trial-level mean probability for each condition
        trial_probs = {}
        for cond_key, prob_col in prob_cols_map.items():
            if prob_col not in df.columns:
                continue
            means = []
            for _, row in df.iterrows():
                probs = _safe_parse_prob_list(row[prob_col])
                if probs:
                    means.append(np.nanmean(probs))
                else:
                    means.append(np.nan)
            df[f"_mean_prob_{cond_key}"] = means
            trial_probs[cond_key] = f"_mean_prob_{cond_key}"

        # Plot as strip/scatter across conditions
        cond_order = [
            ("retrieved",      "Retrieved",      COLORS["retrieved"]),
            ("control",        "Control",        COLORS["control"]),
            ("shuf_retrieved", "Shuffled",       COLORS["shuffled_ret"]),
            ("shuf_sel_retrieved", "Shuf (strict)", "#A0D0A0"),
        ]
        available_conds = [(k, l, c) for k, l, c in cond_order
                           if k in trial_probs]

        rng = np.random.default_rng(42)
        positions = []
        labels = []
        cond_vals = {}  # store subject-level arrays for significance testing
        for i, (key, label, color) in enumerate(available_conds):
            col = trial_probs[key]
            # Subject-level means for the scatter
            subj_means = df.groupby("subject")[col].mean().dropna()
            vals = subj_means.values
            cond_vals[key] = subj_means
            jitter = rng.normal(0, 0.08, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color=color, edgecolor="k", linewidth=0.4,
                       s=25, alpha=0.7, zorder=5)
            # Box for distribution
            ax.boxplot([vals], positions=[i], widths=0.4,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, alpha=0.3, edgecolor="black", linewidth=0.8),
                       medianprops=dict(color="black", linewidth=1.4),
                       whiskerprops=dict(color="#666"),
                       capprops=dict(color="#666"))
            positions.append(i)
            labels.append(label)

        # Significance tests: Retrieved vs each other condition
        if "retrieved" in cond_vals and len(cond_vals) >= 2:
            all_vals_flat = np.concatenate([v.values for v in cond_vals.values()])
            y_max = np.nanmax(all_vals_flat)
            y_min = np.nanmin(all_vals_flat)
            y_range = y_max - y_min
            bracket_h = y_range * 0.03

            ret_series = cond_vals["retrieved"]
            level = 0
            for j, (key, label, _) in enumerate(available_conds):
                if key == "retrieved":
                    continue
                other_series = cond_vals[key]
                common = sorted(set(ret_series.index) & set(other_series.index))
                if len(common) < 3:
                    continue
                t_stat, p_val = ttest_rel(
                    ret_series.loc[common].values,
                    other_series.loc[common].values,
                )
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

                bracket_y = y_max + y_range * (0.08 + level * 0.12)
                pos_a, pos_b = 0, j
                ax.plot([pos_a, pos_a, pos_b, pos_b],
                        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                        lw=0.8, color="black")
                ax.text((pos_a + pos_b) / 2, bracket_y + bracket_h * 1.5,
                        f"{sig}\np={p_val:.3f}",
                        ha="center", va="bottom", fontsize=7)
                level += 1

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Mean P(target class)")
        ax.set_title(f"Classifier probability — {roi.upper()}")
        _despine(ax)

    fig.tight_layout()
    _save(fig, out_dir, f"fig5_absolute_slope_{roi}_{pre_post}")


# =====================================================================
# FIGURE 6: SODA VARIANCE
# =====================================================================

def fig_soda_variance(soda_results_dir: str, roi: str, out_dir: str, pre_post: str):
    """Boxplot of SODA variance (per trial, averaged per subject) across 6 conditions."""
    csv_path = opj(soda_results_dir, f"{roi}_soda_results_per_combination.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] SODA results not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    soda_cols = {
        "retrieved":          "retrieved_soda",
        "control":            "control_soda",
        "shuf_retrieved":     "retrieved_soda_shuffled",
        "shuf_control":       "control_soda_shuffled",
        "shuf_sel_retrieved": "retrieved_soda_shuffled_selected",
        "shuf_sel_control":   "control_soda_shuffled_selected",
    }
    for col in soda_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Variance per trial (across TRs), then mean per subject
    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in df.columns]

    available_cols = [c for c in soda_cols.values() if c in df.columns]
    trial_var = df.groupby(group_cols)[available_cols].var()
    # Rename columns to var_ prefix for clarity
    var_col_map = {}
    for key, col in soda_cols.items():
        var_name = f"var_{col}"
        if col in available_cols:
            trial_var = trial_var.rename(columns={col: var_name})
            var_col_map[key] = var_name

    subj = trial_var.reset_index().groupby("subject")[list(var_col_map.values())].mean().dropna()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _boxplot_6conditions(subj, var_col_map, ax,
                         ylabel="SODA variance",
                         title=f"SODA variance — {roi.upper()} ({pre_post})")
    fig.tight_layout()
    _save(fig, out_dir, f"fig6_soda_variance_{roi}_{pre_post}")


# =====================================================================
# FIGURE 7: SINUSOIDAL AMPLITUDE (windowed one-cycle)
# =====================================================================

def fig_sinus_amplitude(sinus_dir: str, roi: str, out_dir: str, pre_post: str):
    """Boxplot of |amplitude| from windowed sinusoidal fit across 6 conditions."""
    csv_path = opj(sinus_dir, roi, f"{roi}_windowed_fit_results_{pre_post}.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Sinus results not found: {csv_path}")
        return

    mdf = pd.read_csv(csv_path)

    amp_cols = {
        "retrieved":          "retrieved_w_amplitude",
        "control":            "control_w_amplitude",
        "shuf_retrieved":     "shuf_retrieved_w_amplitude",
        "shuf_control":       "shuf_control_w_amplitude",
        "shuf_sel_retrieved": "shuf_sel_retrieved_w_amplitude",
        "shuf_sel_control":   "shuf_sel_control_w_amplitude",
    }

    # Take absolute amplitude, then average per subject
    available_cols = [c for c in amp_cols.values() if c in mdf.columns]
    for col in available_cols:
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce").abs()

    subj = mdf.groupby("subject")[available_cols].mean().dropna()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _boxplot_6conditions(subj, amp_cols, ax,
                         ylabel="|Amplitude| (windowed sinusoid)",
                         title=f"Sinusoidal amplitude — {roi.upper()} ({pre_post})")
    fig.tight_layout()
    _save(fig, out_dir, f"fig7_sinus_amplitude_{roi}_{pre_post}")


# =====================================================================
# FIGURE 8: FFT BAND POWER
# =====================================================================

def fig_fft_band_power(fft_dir: str, roi: str, out_dir: str, pre_post: str):
    """Boxplot of FFT band power across 6 conditions."""
    csv_path = opj(fft_dir, roi, f"{roi}_fft_results_{pre_post}.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] FFT results not found: {csv_path}")
        return

    mdf = pd.read_csv(csv_path)

    bp_cols = {
        "retrieved":          "retrieved_band_power",
        "control":            "control_band_power",
        "shuf_retrieved":     "shuf_retrieved_band_power",
        "shuf_control":       "shuf_control_band_power",
        "shuf_sel_retrieved": "shuf_sel_retrieved_band_power",
        "shuf_sel_control":   "shuf_sel_control_band_power",
    }

    available_cols = [c for c in bp_cols.values() if c in mdf.columns]
    for col in available_cols:
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

    subj = mdf.groupby("subject")[available_cols].mean().dropna()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _boxplot_6conditions(subj, bp_cols, ax,
                         ylabel="Band power (0.08–0.20 cyc/TR)",
                         title=f"FFT band power — {roi.upper()} ({pre_post})")
    fig.tight_layout()
    _save(fig, out_dir, f"fig8_fft_band_power_{roi}_{pre_post}")


# =====================================================================
# FIGURE 9: BAYESIAN MODEL COMPARISON (log10 BF)
# =====================================================================

def fig_bayesian_bf(bayes_dir: str, roi: str, out_dir: str, pre_post: str):
    """Boxplot of log10 Bayes factor (replay vs null) across 6 conditions."""
    csv_path = opj(bayes_dir, roi, f"{roi}_bayesian_results_{pre_post}.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Bayesian results not found: {csv_path}")
        return

    mdf = pd.read_csv(csv_path)

    bf_cols = {
        "retrieved":          "retrieved_log10_bf",
        "control":            "control_log10_bf",
        "shuf_retrieved":     "shuf_retrieved_log10_bf",
        "shuf_control":       "shuf_control_log10_bf",
        "shuf_sel_retrieved": "shuf_sel_retrieved_log10_bf",
        "shuf_sel_control":   "shuf_sel_control_log10_bf",
    }

    available_cols = [c for c in bf_cols.values() if c in mdf.columns]
    for col in available_cols:
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce")

    subj = mdf.groupby("subject")[available_cols].mean().dropna()

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _boxplot_6conditions(subj, bf_cols, ax,
                         ylabel="log$_{10}$ BF (replay vs null)",
                         title=f"Bayesian evidence — {roi.upper()} ({pre_post})")
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    fig.tight_layout()
    _save(fig, out_dir, f"fig9_bayesian_bf_{roi}_{pre_post}")


# =====================================================================
# FIGURE 10: ALL METRICS OVERVIEW (one panel per metric, 4 conditions)
# =====================================================================

# Reduced condition set: retrieved, control, shuffled, shuffled_selected
CONDITION_DEFS_4 = [
    ("retrieved",          "Retrieved",    "#3274A1"),
    ("control",            "Control",      "#E1812C"),
    ("shuf_retrieved",     "Shuffled",     "#8CB8D0"),
    ("shuf_sel_retrieved", "Shuf (strict)", "#A0D0A0"),
]


def _boxplot_4conditions(subj_df, col_map, ax, ylabel="", title=""):
    """Like _boxplot_6conditions but for the reduced 4-condition set.

    Draws significance brackets for:
      - Retrieved vs Control (positions 0 vs 1)
      - Retrieved vs Shuffled (positions 0 vs 2)
      - Retrieved vs Shuf-strict (positions 0 vs 3)
    """
    available = [(key, label, color) for key, label, color in CONDITION_DEFS_4
                 if key in col_map and col_map[key] in subj_df.columns]
    if len(available) < 2:
        ax.set_title(f"{title}\n(insufficient data)", fontsize=9)
        _despine(ax)
        return

    plot_data = [subj_df[col_map[key]].dropna().values for key, _, _ in available]
    plot_labels = [label for _, label, _ in available]
    plot_colors = [color for _, _, color in available]

    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)
    for whisker in bp["whiskers"]:
        whisker.set_color("#666")
    for cap in bp["caps"]:
        cap.set_color("#666")

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   color=color, edgecolor="k", linewidth=0.5,
                   s=15, zorder=5, alpha=0.7)

    # Significance brackets: Retrieved (0) vs each other condition
    all_vals = np.concatenate(plot_data)
    y_max = np.nanmax(all_vals)
    y_min = np.nanmin(all_vals)
    y_range = y_max - y_min
    bracket_h = y_range * 0.03

    comparisons = []  # (pos_a, pos_b, offset_level)
    for j in range(1, len(plot_data)):
        comparisons.append((0, j, j - 1))

    for pos_a, pos_b, level in comparisons:
        n = min(len(plot_data[pos_a]), len(plot_data[pos_b]))
        if n < 3:
            continue
        _, p_val = ttest_rel(plot_data[pos_a][:n], plot_data[pos_b][:n])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        bracket_y = y_max + y_range * (0.08 + level * 0.12)
        ax.plot([pos_a, pos_a, pos_b, pos_b],
                [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                lw=0.8, color="black")
        ax.text((pos_a + pos_b) / 2, bracket_y + bracket_h * 1.5, sig,
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    _despine(ax)


def fig_all_metrics_overview(
    soda_results_dir: str,
    sinus_dir: str,
    fft_dir: str,
    bayes_dir: str,
    roi: str,
    out_dir: str,
    pre_post: str,
):
    """5-panel figure: one boxplot per metric, 4 conditions each."""

    # --- Load all data sources ---
    panels = []  # list of (subj_df, col_map, ylabel, title)

    soda_csv = opj(soda_results_dir, f"{roi}_soda_results_per_combination.csv")
    sinus_csv = opj(sinus_dir, roi, f"{roi}_windowed_fit_results_{pre_post}.csv")
    fft_csv = opj(fft_dir, roi, f"{roi}_fft_results_{pre_post}.csv")
    bayes_csv = opj(bayes_dir, roi, f"{roi}_bayesian_results_{pre_post}.csv")

    # 1. Absolute slope
    if os.path.exists(soda_csv):
        df = pd.read_csv(soda_csv)
        abs_map = {
            "retrieved":          "retrieved_soda_abs",
            "control":            "control_soda_abs",
            "shuf_retrieved":     "retrieved_soda_shuffled_abs",
            "shuf_sel_retrieved": "retrieved_soda_shuffled_selected_abs",
        }
        for c in abs_map.values():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        avail = [c for c in abs_map.values() if c in df.columns]
        subj = df.groupby("subject")[avail].mean().dropna()
        panels.append((subj, abs_map, "|SODA slope|", "Absolute slope"))
    else:
        print(f"  [SKIP] SODA CSV not found for absolute slope")

    # 2. SODA variance
    if os.path.exists(soda_csv):
        df = pd.read_csv(soda_csv)
        var_src = {
            "retrieved":          "retrieved_soda",
            "control":            "control_soda",
            "shuf_retrieved":     "retrieved_soda_shuffled",
            "shuf_sel_retrieved": "retrieved_soda_shuffled_selected",
        }
        for c in var_src.values():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
                      if c in df.columns]
        avail_src = [c for c in var_src.values() if c in df.columns]
        trial_var = df.groupby(group_cols)[avail_src].var()
        var_map = {}
        for key, col in var_src.items():
            vname = f"var_{col}"
            if col in avail_src:
                trial_var = trial_var.rename(columns={col: vname})
                var_map[key] = vname
        subj = trial_var.reset_index().groupby("subject")[list(var_map.values())].mean().dropna()
        panels.append((subj, var_map, "SODA variance", "Variance"))

    # 3. Sinusoidal amplitude
    if os.path.exists(sinus_csv):
        mdf = pd.read_csv(sinus_csv)
        amp_map = {
            "retrieved":          "retrieved_w_amplitude",
            "control":            "control_w_amplitude",
            "shuf_retrieved":     "shuf_retrieved_w_amplitude",
            "shuf_sel_retrieved": "shuf_sel_retrieved_w_amplitude",
        }
        for c in amp_map.values():
            if c in mdf.columns:
                mdf[c] = pd.to_numeric(mdf[c], errors="coerce").abs()
        avail = [c for c in amp_map.values() if c in mdf.columns]
        subj = mdf.groupby("subject")[avail].mean().dropna()
        panels.append((subj, amp_map, "|Amplitude|", "Sinusoidal fit"))
    else:
        print(f"  [SKIP] Sinus CSV not found")

    # 4. FFT band power
    if os.path.exists(fft_csv):
        mdf = pd.read_csv(fft_csv)
        bp_map = {
            "retrieved":          "retrieved_band_power",
            "control":            "control_band_power",
            "shuf_retrieved":     "shuf_retrieved_band_power",
            "shuf_sel_retrieved": "shuf_sel_retrieved_band_power",
        }
        for c in bp_map.values():
            if c in mdf.columns:
                mdf[c] = pd.to_numeric(mdf[c], errors="coerce")
        avail = [c for c in bp_map.values() if c in mdf.columns]
        subj = mdf.groupby("subject")[avail].mean().dropna()
        panels.append((subj, bp_map, "Band power", "FFT"))
    else:
        print(f"  [SKIP] FFT CSV not found")

    # 5. Bayesian log10 BF
    if os.path.exists(bayes_csv):
        mdf = pd.read_csv(bayes_csv)
        bf_map = {
            "retrieved":          "retrieved_log10_bf",
            "control":            "control_log10_bf",
            "shuf_retrieved":     "shuf_retrieved_log10_bf",
            "shuf_sel_retrieved": "shuf_sel_retrieved_log10_bf",
        }
        for c in bf_map.values():
            if c in mdf.columns:
                mdf[c] = pd.to_numeric(mdf[c], errors="coerce")
        avail = [c for c in bf_map.values() if c in mdf.columns]
        subj = mdf.groupby("subject")[avail].mean().dropna()
        panels.append((subj, bf_map, "log$_{10}$ BF", "Bayesian"))
    else:
        print(f"  [SKIP] Bayesian CSV not found")

    if not panels:
        print("  [SKIP] No metric data available for overview figure")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.5), squeeze=False)
    axes = axes[0]

    for i, (subj_df, col_map, ylabel, title) in enumerate(panels):
        _add_panel_label(axes[i], chr(ord("a") + i))
        _boxplot_4conditions(subj_df, col_map, axes[i], ylabel=ylabel, title=title)
        # Add zero line for Bayesian panel
        if "log" in ylabel.lower():
            axes[i].axhline(0, color="black", lw=0.5, ls="--", zorder=0)

    fig.suptitle(f"Sequentiality metrics — {roi.upper()} ({pre_post})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"fig10_all_metrics_overview_{roi}_{pre_post}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    setup_style()

    roi = sys.argv[1] if len(sys.argv) > 1 else "vis"
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = ensure_dir(opj(script_dir, "5.0_publication_figures", roi))

    print(f"Generating publication figures for ROI={roi}, {pre_post}")
    print(f"Output: {out_dir}\n")

    # Paths to pipeline outputs
    clf_results_dir = opj(script_dir, "4.1_train_visual_classifier", "df_classifier_results_csv")
    if pre_post == "presleep":
        soda_input_dir = opj(script_dir, "4.4_transfer_visual_classifier_retrieval_presleep", "decoding_results_tables_for_SODA")
        soda_results_dir = opj(script_dir, "4.6_SODA_based_on_singlesubj_4.4_presleep", "SODA_results_csv")
    else:
        soda_input_dir = opj(script_dir, "4.4_transfer_visual_classifier_retrieval_postsleep", "decoding_results_tables_for_SODA")
        soda_results_dir = opj(script_dir, "4.6_SODA_based_on_singlesubj_4.4_postsleep", "SODA_results_csv")
    sinus_dir = opj(script_dir, "4.7_sinus_1cycle")
    fft_dir = opj(script_dir, "4.8_fft")
    bayes_dir = opj(script_dir, "4.9_bayesian_replay")

    # --- Generate figures ---
    print("--- Figure 1: Classifier performance ---")
    if os.path.isdir(clf_results_dir):
        fig_classifier_performance(clf_results_dir, out_dir, condition=1)
        fig_classifier_performance(clf_results_dir, out_dir, condition=2)
    else:
        print(f"  [SKIP] Classifier results dir not found: {clf_results_dir}")

    print("\n--- Figure 2: Retrieval decoding timecourse ---")
    if os.path.isdir(soda_input_dir):
        fig_retrieval_timecourse(soda_input_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] SODA input dir not found: {soda_input_dir}")

    print("\n--- Figure 3: SODA timecourse ---")
    if os.path.isdir(soda_results_dir):
        fig_soda_timecourse(soda_results_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] SODA results dir not found: {soda_results_dir}")

    print("\n--- Figure 4: SODA by position ---")
    if os.path.isdir(soda_results_dir):
        fig_soda_by_position(soda_results_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] SODA results dir not found")

    print("\n--- Figure 5: Absolute slope ---")
    if os.path.isdir(soda_results_dir):
        fig_absolute_slope(soda_results_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] SODA results dir not found")

    print("\n--- Figure 6: SODA variance ---")
    if os.path.isdir(soda_results_dir):
        fig_soda_variance(soda_results_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] SODA results dir not found")

    print("\n--- Figure 7: Sinusoidal amplitude ---")
    if os.path.isdir(sinus_dir):
        fig_sinus_amplitude(sinus_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] Sinus dir not found: {sinus_dir}")

    print("\n--- Figure 8: FFT band power ---")
    if os.path.isdir(fft_dir):
        fig_fft_band_power(fft_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] FFT dir not found: {fft_dir}")

    print("\n--- Figure 9: Bayesian model comparison ---")
    if os.path.isdir(bayes_dir):
        fig_bayesian_bf(bayes_dir, roi, out_dir, pre_post)
    else:
        print(f"  [SKIP] Bayesian dir not found: {bayes_dir}")

    print("\n--- Figure 10: All metrics overview (4 conditions) ---")
    fig_all_metrics_overview(soda_results_dir, sinus_dir, fft_dir, bayes_dir,
                             roi, out_dir, pre_post)

    print(f"\nDone. All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
