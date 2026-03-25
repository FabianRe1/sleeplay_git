# -*- coding: utf-8 -*-
"""Thesis chapter figures — individual panels for manual composition.

Generates one PDF per panel from the master replay dataframe (5.1).
Only absolute SODA (abssoda) metric is used.
Filenames encode: plottype_metric_scope.pdf

Condition tests (pre vs post) use a linear mixed model
(replay_diff ~ is_post + (1|subject)) matching 5.2_replay_statistics.
Results are shown as brackets with significance stars on session plots.

Usage
-----
    python 5.0_thesis_figures.py

Output
------
    5.0_thesis_figures/
        # Section 2.4 — Replay detection
        collapsed_abssoda_diff_per_roi.pdf
        session_abssoda_diff_vis.pdf  (etc. for each ROI)
        # Section 2.5 — Sleep changes
        session_alltrials_abssoda_diff_per_roi.pdf
        session_objects_abssoda_diff_parietal.pdf  (etc. for each ROI)
        # Section 2.6 — Brain-behavior
        brainbehavior_abssoda_diff_vis_presleep.pdf (etc. for each ROI × variant × session)
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, pearsonr

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = opj(SCRIPT_DIR, "5.0_thesis_figures")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

ROIS = ["vis", "mtl", "parietal"]
ROI_LABELS = {"vis": "Visual cortex", "mtl": "MTL", "parietal": "Parietal"}

# (ret_col, ctrl_col, short_name, display_name)
METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "abssoda", "Abs SODA"),
]

COLORS = {
    "presleep": "#7BAFD4",
    "postsleep": "#D4534E",
    "vis": "#3274A1",
    "mtl": "#E1812C",
    "parietal": "#2CA02C",
}


def setup_style():
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, name):
    fig.savefig(opj(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}")


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "\u2020"
    return "n.s."


def _add_bracket(ax, x0, x1, y, text, h=None):
    """Draw a bracket between x0 and x1 at height y with annotation text."""
    if h is None:
        ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        h = ylim_range * 0.03
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, color="black")
    ax.text((x0 + x1) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


def _fit_lme(df, formula, group_var="subject"):
    """Fit LME with subject random intercept (Powell optimizer), matching 5.2."""
    df_clean = df.dropna(subset=[group_var]).copy()
    dep_var = formula.split("~")[0].strip()
    if dep_var in df_clean.columns:
        df_clean = df_clean.dropna(subset=[dep_var])
    model = smf.mixedlm(formula, df_clean, groups=df_clean[group_var])
    return model.fit(reml=True, method="powell")


def _prepost_lme(df, roi, ret_col, ctrl_col):
    """LME: replay_diff ~ is_post + (1|subject), matching 5.2 approach.

    Returns (z_value, p_value) for the is_post coefficient.
    """
    sub = df[df["roi"] == roi].copy()
    sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
    sub["is_post"] = (sub["session"] == "postsleep").astype(int)
    sub = sub.dropna(subset=["replay_diff"])
    if sub["subject"].nunique() < 3:
        return np.nan, np.nan
    try:
        res = _fit_lme(sub, "replay_diff ~ is_post")
        return res.tvalues["is_post"], res.pvalues["is_post"]
    except Exception:
        return np.nan, np.nan


def _bar_with_dots(ax, positions, means, sems, subject_vals, colors, labels,
                   ylabel="", width=0.6):
    rng = np.random.default_rng(42)
    for pos, m, se, vals, color in zip(positions, means, sems, subject_vals, colors):
        ax.bar(pos, m, yerr=se, width=width, color=color, edgecolor="#333",
               linewidth=0.8, capsize=3, alpha=0.7, zorder=1,
               error_kw={"linewidth": 1.0, "capthick": 1.0})
        jitter = rng.uniform(-width / 4, width / 4, size=len(vals))
        ax.scatter(pos + jitter, vals, color="black", s=10, alpha=0.35,
                   linewidths=0, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)


def _compute_diff_stats(df, roi, ret_col, ctrl_col, session=None):
    """Compute subject-level diff means and t-test for a given ROI/session."""
    if session:
        sub = df[(df["roi"] == roi) & (df["session"] == session)].copy()
    else:
        sub = df[df["roi"] == roi].copy()
    sub["diff"] = sub[ret_col] - sub[ctrl_col]
    smeans = sub.groupby("subject")["diff"].mean().dropna()
    if len(smeans) < 3:
        return smeans, np.nan, np.nan
    t, p = ttest_1samp(smeans, 0)
    return smeans, t, p


# =====================================================================
# SECTION 2.4 — Replay detection
# =====================================================================

def panel_collapsed_diff_per_roi(df, ret_col, ctrl_col, metric_name, metric_short):
    """Bar plot of diff (ret - ctrl) per ROI, collapsed across sessions."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

    for i, roi in enumerate(ROIS):
        smeans, t, p = _compute_diff_stats(df, roi, ret_col, ctrl_col)
        positions.append(i)
        means.append(smeans.mean())
        sems.append(smeans.std() / np.sqrt(len(smeans)))
        all_vals.append(smeans.values)
        colors.append(COLORS[roi])
        labels.append(ROI_LABELS[roi])
        y_top = smeans.mean() + smeans.std() / np.sqrt(len(smeans)) + smeans.std() * 0.15
        ax.text(i, y_top, _sig(p), ha="center", va="bottom", fontsize=10)

    _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                   ylabel=f"{metric_name}\n(ret $-$ ctrl)")
    ax.set_title(f"Sequential reactivation (collapsed)")
    fig.tight_layout()
    _save(fig, f"collapsed_{metric_short}_diff_per_roi")


def panel_session_diff_per_roi_single(df, roi, ret_col, ctrl_col, metric_name, metric_short):
    """Bar plot: pre vs post diff for a single ROI."""
    fig, ax = plt.subplots(figsize=(2.5, 3.0))
    positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

    for i, session in enumerate(["presleep", "postsleep"]):
        smeans, t, p = _compute_diff_stats(df, roi, ret_col, ctrl_col, session)
        positions.append(i)
        means.append(smeans.mean())
        sems.append(smeans.std() / np.sqrt(len(smeans)))
        all_vals.append(smeans.values)
        colors.append(COLORS[session])
        labels.append("Pre" if session == "presleep" else "Post")
        y_top = max(smeans.mean() + smeans.std() / np.sqrt(len(smeans)),
                    smeans.mean()) + smeans.std() * 0.15
        ax.text(i, y_top, _sig(p), ha="center", va="bottom", fontsize=10)

    _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                   ylabel=f"{metric_name} (ret $-$ ctrl)")
    ax.set_title(ROI_LABELS[roi])

    # Pre vs post paired t-test bracket
    z_pp, p_pp = _prepost_lme(df, roi, ret_col, ctrl_col)
    if not np.isnan(p_pp):
        y_max = max(m + se for m, se in zip(means, sems))
        y_max = max(y_max, max(v.max() for v in all_vals if len(v) > 0))
        bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
        _add_bracket(ax, 0, 1, bracket_y, f"pre vs post: {_sig(p_pp)}")

    fig.tight_layout()
    _save(fig, f"session_{metric_short}_diff_{roi}")


# =====================================================================
# SECTION 2.5 — Sleep changes
# =====================================================================

def panel_session_alltrials_grouped(df, ret_col, ctrl_col, metric_name, metric_short):
    """Grouped bar: pre vs post per ROI, all trials."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    width = 0.35
    x = np.arange(len(ROIS))
    rng = np.random.default_rng(42)

    for j, (session, offset, color) in enumerate([
            ("presleep", -width / 2, COLORS["presleep"]),
            ("postsleep", width / 2, COLORS["postsleep"])]):
        for i, roi in enumerate(ROIS):
            smeans, t, p = _compute_diff_stats(df, roi, ret_col, ctrl_col, session)
            m = smeans.mean()
            se = smeans.std() / np.sqrt(len(smeans))
            ax.bar(x[i] + offset, m, yerr=se, width=width, color=color,
                   edgecolor="#333", linewidth=0.6, capsize=2, alpha=0.7,
                   error_kw={"linewidth": 0.8}, zorder=1)
            jitter = rng.uniform(-width / 4, width / 4, size=len(smeans))
            ax.scatter(x[i] + offset + jitter, smeans.values, color="black",
                       s=8, alpha=0.3, linewidths=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([ROI_LABELS[r] for r in ROIS])
    ax.set_ylabel(f"{metric_name}\n(ret $-$ ctrl)")
    ax.set_title("All trials: pre vs post sleep")
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    ax.legend(["Pre-sleep", "Post-sleep"], frameon=False, fontsize=9)

    # Pre vs post paired t-tests per ROI
    ylim = ax.get_ylim()
    y_bracket = ylim[1]
    for i, roi in enumerate(ROIS):
        z_pp, p_pp = _prepost_lme(df, roi, ret_col, ctrl_col)
        if not np.isnan(p_pp):
            _add_bracket(ax, x[i] - width / 2, x[i] + width / 2, y_bracket,
                         _sig(p_pp))

    fig.tight_layout()
    _save(fig, f"session_alltrials_{metric_short}_diff_per_roi")


def panel_session_objects_single_roi(df_obj, roi, ret_col, ctrl_col,
                                      metric_name, metric_short):
    """Bar plot: pre vs post for objects only, single ROI, single metric."""
    fig, ax = plt.subplots(figsize=(2.5, 3.0))
    positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

    for i, session in enumerate(["presleep", "postsleep"]):
        smeans, t, p = _compute_diff_stats(df_obj, roi, ret_col, ctrl_col, session)
        positions.append(i)
        means.append(smeans.mean())
        sems.append(smeans.std() / np.sqrt(len(smeans)))
        all_vals.append(smeans.values)
        colors.append(COLORS[session])
        labels.append("Pre" if session == "presleep" else "Post")
        y_top = max(smeans.mean() + smeans.std() / np.sqrt(len(smeans)),
                    smeans.mean()) + smeans.std() * 0.15
        ax.text(i, y_top, _sig(p), ha="center", va="bottom", fontsize=10)

    _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                   ylabel=f"{metric_name} (ret $-$ ctrl)")
    ax.set_title(f"{ROI_LABELS[roi]} — objects only")

    # Pre vs post paired t-test bracket
    z_pp, p_pp = _prepost_lme(df_obj, roi, ret_col, ctrl_col)
    if not np.isnan(p_pp):
        y_max = max(m + se for m, se in zip(means, sems))
        y_max = max(y_max, max(v.max() for v in all_vals if len(v) > 0))
        bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
        _add_bracket(ax, 0, 1, bracket_y, f"pre vs post: {_sig(p_pp)}")

    fig.tight_layout()
    _save(fig, f"session_objects_{metric_short}_diff_{roi}")


# =====================================================================
# SECTION 2.6 — Brain-behavior
# =====================================================================

def panel_brainbehavior(df_obj, roi, variant, ret_col, ctrl_col,
                        metric_name, metric_short, session_filter=None):
    """Scatter: replay metric vs accuracy at subject level."""
    if session_filter:
        sub = df_obj[(df_obj["roi"] == roi) & (df_obj["session"] == session_filter)].copy()
        session_label = session_filter
    else:
        sub = df_obj[df_obj["roi"] == roi].copy()
        session_label = "collapsed"

    sub["behav_correct"] = pd.to_numeric(sub["behav_correct"], errors="coerce")

    if variant == "diff":
        sub["replay"] = sub[ret_col] - sub[ctrl_col]
        ylabel = f"{metric_name} (ret $-$ ctrl)"
    else:
        sub["replay"] = sub[ret_col]
        ylabel = f"{metric_name} (raw ret)"

    sub = sub.dropna(subset=["replay", "behav_correct"])
    smeans = sub.groupby("subject").agg(
        replay=("replay", "mean"), acc=("behav_correct", "mean")
    ).dropna()

    if len(smeans) < 5:
        return

    r, p = pearsonr(smeans["replay"], smeans["acc"])

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(smeans["replay"], smeans["acc"], color=COLORS[roi],
               edgecolor="black", linewidth=0.5, s=35, alpha=0.8, zorder=3)

    z = np.polyfit(smeans["replay"], smeans["acc"], 1)
    x_line = np.linspace(smeans["replay"].min(), smeans["replay"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="black", lw=1.0, ls="--", zorder=2)

    ax.set_xlabel(ylabel)
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title(f"{ROI_LABELS[roi]} ({session_label})\nr = {r:.3f}, p = {p:.4f}")
    fig.tight_layout()
    _save(fig, f"brainbehavior_{metric_short}_{variant}_{roi}_{session_label}")


# =====================================================================
# HELPERS — trial matching
# =====================================================================

def _subsample_postsleep_first_half(df):
    """Keep presleep as-is, keep only first half of postsleep per subject × ROI.

    Matches the approach in 5.2_replay_statistics (analysis 2b).
    """
    df_pre = df[df["session"] == "presleep"].copy()
    df_post = df[df["session"] == "postsleep"].copy()

    keep_idx = []
    for (subj, roi), grp in df_post.groupby(["subject", "roi"]):
        half = len(grp) // 2
        keep_idx.extend(grp.index[:half].tolist())
    df_post_half = df_post.loc[keep_idx]

    return pd.concat([df_pre, df_post_half], ignore_index=True)


# =====================================================================
# MAIN
# =====================================================================

def _run_panels(df, df_obj, out_dir, label="full"):
    """Generate all panels for a given dataframe into out_dir."""
    global OUT_DIR
    orig_out = OUT_DIR
    OUT_DIR = out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    for ret_col, ctrl_col, metric_short, metric_name in METRICS:
        print(f"\n{'='*50}")
        print(f"  METRIC: {metric_name} ({metric_short}) [{label}]")
        print(f"{'='*50}")

        # --- Section 2.4: Replay detection ---
        print("\n  Section 2.4 — Replay detection:")
        panel_collapsed_diff_per_roi(df, ret_col, ctrl_col, metric_name, metric_short)
        for roi in ROIS:
            panel_session_diff_per_roi_single(df, roi, ret_col, ctrl_col,
                                              metric_name, metric_short)

        # --- Section 2.5: Sleep changes ---
        print("\n  Section 2.5 — Sleep changes:")
        panel_session_alltrials_grouped(df, ret_col, ctrl_col, metric_name, metric_short)
        for roi in ROIS:
            panel_session_objects_single_roi(df_obj, roi, ret_col, ctrl_col,
                                            metric_name, metric_short)

        # --- Section 2.6: Brain-behavior (per session only, no collapsed) ---
        print("\n  Section 2.6 — Brain-behavior:")
        for roi in ROIS:
            for variant in ["diff", "raw"]:
                for session in ["presleep", "postsleep"]:
                    panel_brainbehavior(df_obj, roi, variant, ret_col, ctrl_col,
                                        metric_name, metric_short,
                                        session_filter=session)

    print(f"\n  Panels saved to: {OUT_DIR}")
    OUT_DIR = orig_out


def main():
    setup_style()

    df = pd.read_csv(MASTER_CSV)
    df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")
    df_obj = df[df["object_scene_retrieval"] == "object"].copy()

    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")
    print(f"Object trials: {len(df_obj)}")

    # --- Full version (all trials) ---
    print(f"\n{'#'*60}")
    print(f"  FULL VERSION (all trials)")
    print(f"{'#'*60}")
    _run_panels(df, df_obj, opj(SCRIPT_DIR, "5.0_thesis_figures"), label="full")

    # --- Matched version (postsleep first-half only) ---
    df_matched = _subsample_postsleep_first_half(df)
    df_obj_matched = df_matched[df_matched["object_scene_retrieval"] == "object"].copy()

    ref_roi = ROIS[0]
    n_pre = len(df_matched[(df_matched["session"] == "presleep") & (df_matched["roi"] == ref_roi)])
    n_post = len(df_matched[(df_matched["session"] == "postsleep") & (df_matched["roi"] == ref_roi)])
    print(f"\n{'#'*60}")
    print(f"  MATCHED VERSION (postsleep first-half)")
    print(f"  Presleep trials/ROI: {n_pre}, Postsleep trials/ROI: {n_post}")
    print(f"{'#'*60}")
    _run_panels(df_matched, df_obj_matched,
                opj(SCRIPT_DIR, "5.0_thesis_figures_matched"), label="matched")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
