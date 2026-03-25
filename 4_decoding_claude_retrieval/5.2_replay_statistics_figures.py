# -*- coding: utf-8 -*-
"""Publication-ready figures for 5.2 replay statistics.

Reads the master replay dataframe and the 5.2 statistics CSVs,
then produces one PDF per panel.  Figures cover both the full
(unmatched) and trial-matched analyses.

Usage
-----
    python 5.2_replay_statistics_figures.py

Output
------
    5.2_replay_statistics_figures/
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
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe",
                 "master_replay_metrics.csv")
OUT_DIR = opj(SCRIPT_DIR, "5.2_replay_statistics_figures")

ROIS = ["vis", "mtl", "parietal"]
ROI_LABELS = {"vis": "Visual", "mtl": "MTL", "parietal": "Parietal"}

# Only Abs SODA for thesis figures
RET_COL = "retrieved_soda_abs"
CTRL_COL = "control_soda_abs"
METRIC_NAME = "Abs SODA"

COLORS = {
    "presleep": "#7BAFD4",
    "postsleep": "#D4534E",
    "vis": "#3274A1",
    "mtl": "#E1812C",
    "parietal": "#2CA02C",
    "object": "#8C564B",
    "scene": "#9467BD",
    "correct": "#2CA02C",
    "incorrect": "#D62728",
}

# =====================================================================
# STYLE
# =====================================================================

def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =====================================================================
# HELPERS
# =====================================================================

def _save(fig, name):
    fig.savefig(opj(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.pdf")


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def _sig_dagger(p):
    """Include dagger for marginal significance."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "\u2020"
    return "n.s."


def _add_bracket(ax, x0, x1, y, text, h=None):
    """Draw a comparison bracket."""
    if h is None:
        ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        h = ylim_range * 0.02
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y],
            lw=0.9, color="black", clip_on=False)
    ax.text((x0 + x1) / 2, y + h * 1.15, text,
            ha="center", va="bottom", fontsize=8, clip_on=False)
    # Expand y-axis to make room for bracket + text
    cur_top = ax.get_ylim()[1]
    needed = y + h * 3.5
    if needed > cur_top:
        ax.set_ylim(top=needed)


def _subject_diff_means(df, roi, ret_col, ctrl_col, session=None):
    """Subject-level mean of (ret - ctrl) for a given ROI/session."""
    if session:
        sub = df[(df["roi"] == roi) & (df["session"] == session)]
    else:
        sub = df[df["roi"] == roi]
    diffs = sub.assign(diff=sub[ret_col] - sub[ctrl_col])
    return diffs.groupby("subject")["diff"].mean().dropna()


def _bar_with_dots(ax, positions, means, sems, subject_vals, colors,
                   labels, ylabel="", width=0.55):
    """Bar + individual subject dots."""
    rng = np.random.default_rng(42)
    for pos, m, se, vals, color in zip(positions, means, sems,
                                        subject_vals, colors):
        ax.bar(pos, m, yerr=se, width=width, color=color,
               edgecolor="#333", linewidth=0.7, capsize=3,
               alpha=0.75, zorder=1,
               error_kw={"linewidth": 0.9, "capthick": 0.9})
        jitter = rng.uniform(-width / 4, width / 4, size=len(vals))
        ax.scatter(pos + jitter, vals, color="black", s=12, alpha=0.3,
                   linewidths=0, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)


def _subsample_postsleep_first_half(df):
    """Keep presleep as-is; keep first half of postsleep per subject x ROI."""
    df_pre = df[df["session"] == "presleep"]
    df_post = df[df["session"] == "postsleep"]
    keep_idx = []
    for (subj, roi), grp in df_post.groupby(["subject", "roi"]):
        half = len(grp) // 2
        keep_idx.extend(grp.index[:half].tolist())
    return pd.concat([df_pre, df_post.loc[keep_idx]], ignore_index=True)


def _lme_session_p(df, roi, ret_col, ctrl_col):
    """LME: replay_diff ~ is_post + (1|subject). Returns p for is_post.

    Matches the approach in 5.2_replay_statistics.
    """
    sub = df[df["roi"] == roi].copy()
    sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
    sub["is_post"] = (sub["session"] == "postsleep").astype(int)
    sub = sub.dropna(subset=["replay_diff"])
    if sub["subject"].nunique() < 3:
        return np.nan
    try:
        model = smf.mixedlm("replay_diff ~ is_post", sub,
                             groups=sub["subject"])
        res = model.fit(reml=True, method="powell")
        return res.pvalues["is_post"]
    except Exception:
        return np.nan


# =====================================================================
# ANALYSIS 2 — Replay detection (one-sample t-test vs 0)
# =====================================================================

def fig_replay_detection_collapsed(df, suffix=""):
    """Bar plot: replay diff per ROI, collapsed across sessions."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

    for i, roi in enumerate(ROIS):
        smeans = _subject_diff_means(df, roi, RET_COL, CTRL_COL)
        t, p = ttest_1samp(smeans, 0)
        positions.append(i)
        means.append(smeans.mean())
        sems.append(smeans.sem())
        all_vals.append(smeans.values)
        colors.append(COLORS[roi])
        labels.append(ROI_LABELS[roi])
        # Significance star above bar
        y_top = smeans.mean() + smeans.sem() + smeans.std() * 0.12
        y_top = max(y_top, smeans.values.max() * 0.5)
        ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                   ylabel="Replay strength\n(retrieved \u2212 control)")
    ax.set_title("Replay detection (collapsed)")
    fig.tight_layout()
    _save(fig, f"2_detection_collapsed{suffix}")


def fig_replay_detection_per_session(df, suffix=""):
    """One panel per ROI: pre vs post bars with one-sample t-test stars."""
    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

        for i, session in enumerate(["presleep", "postsleep"]):
            slabel = "postsleep_first_half" if (suffix == "_matched"
                     and session == "postsleep") else session
            smeans = _subject_diff_means(df, roi, RET_COL, CTRL_COL, session)
            t, p = ttest_1samp(smeans, 0)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(COLORS[session])
            labels.append("Pre" if session == "presleep" else "Post")
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]}")
        fig.tight_layout()
        _save(fig, f"2_detection_{roi}{suffix}")


# =====================================================================
# ANALYSIS 3 — Session effect (pre vs post)
# =====================================================================

def fig_session_effect_alltrials(df, suffix=""):
    """Grouped bar: pre vs post per ROI, all trials, with LME p-value."""
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    width = 0.32
    x = np.arange(len(ROIS))
    rng = np.random.default_rng(42)

    # Compute LME p-values for session effect from data
    lme_p = {}
    for roi in ROIS:
        p = _lme_session_p(df, roi, RET_COL, CTRL_COL)
        if not np.isnan(p):
            lme_p[roi] = p

    bar_handles = []
    for j, (session, offset, color, label) in enumerate([
            ("presleep", -width / 2, COLORS["presleep"], "Pre"),
            ("postsleep", width / 2, COLORS["postsleep"], "Post")]):
        for i, roi in enumerate(ROIS):
            smeans = _subject_diff_means(df, roi, RET_COL, CTRL_COL, session)
            m, se = smeans.mean(), smeans.sem()
            b = ax.bar(x[i] + offset, m, yerr=se, width=width, color=color,
                       edgecolor="#333", linewidth=0.6, capsize=2, alpha=0.75,
                       error_kw={"linewidth": 0.8}, zorder=1,
                       label=label if i == 0 else None)
            jitter = rng.uniform(-width / 5, width / 5, size=len(smeans))
            ax.scatter(x[i] + offset + jitter, smeans.values, color="black",
                       s=8, alpha=0.25, linewidths=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([ROI_LABELS[r] for r in ROIS])
    ax.set_ylabel("Replay strength\n(retrieved \u2212 control)")
    ax.set_title("Session effect \u2014 all trials")
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # LME brackets
    ylim = ax.get_ylim()
    y_bracket = ylim[1] + (ylim[1] - ylim[0]) * 0.02
    for i, roi in enumerate(ROIS):
        if roi in lme_p:
            _add_bracket(ax, x[i] - width / 2, x[i] + width / 2,
                         y_bracket, _sig(lme_p[roi]))

    fig.tight_layout()
    _save(fig, f"3a_session_alltrials{suffix}")


def fig_session_effect_objects(df_obj, suffix=""):
    """Per-ROI: pre vs post for object trials, with LME p-value."""
    lme_p = {}
    for roi in ROIS:
        p = _lme_session_p(df_obj, roi, RET_COL, CTRL_COL)
        if not np.isnan(p):
            lme_p[roi] = p

    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

        for i, session in enumerate(["presleep", "postsleep"]):
            smeans = _subject_diff_means(df_obj, roi, RET_COL, CTRL_COL,
                                         session)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(COLORS[session])
            labels.append("Pre" if session == "presleep" else "Post")
            # One-sample stars
            t, p = ttest_1samp(smeans, 0)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 objects")

        # LME pre-vs-post bracket
        if roi in lme_p:
            y_max = max(m + se for m, se in zip(means, sems))
            y_max = max(y_max, max(v.max() for v in all_vals))
            bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
            _add_bracket(ax, 0, 1, bracket_y,
                         f"LME: {_sig(lme_p[roi])}")

        fig.tight_layout()
        _save(fig, f"3b_session_objects_{roi}{suffix}")


def fig_session_effect_scenes(df_scene, suffix=""):
    """Per-ROI: pre vs post for scene trials."""
    lme_p = {}
    for roi in ROIS:
        p = _lme_session_p(df_scene, roi, RET_COL, CTRL_COL)
        if not np.isnan(p):
            lme_p[roi] = p

    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

        for i, session in enumerate(["presleep", "postsleep"]):
            smeans = _subject_diff_means(df_scene, roi, RET_COL, CTRL_COL,
                                         session)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(COLORS[session])
            labels.append("Pre" if session == "presleep" else "Post")
            t, p = ttest_1samp(smeans, 0)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 scenes")

        if roi in lme_p:
            y_max = max(m + se for m, se in zip(means, sems))
            y_max = max(y_max, max(v.max() for v in all_vals))
            bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
            _add_bracket(ax, 0, 1, bracket_y,
                         f"LME: {_sig(lme_p[roi])}")

        fig.tight_layout()
        _save(fig, f"3c_session_scenes_{roi}{suffix}")


# =====================================================================
# ANALYSIS 4 — Accuracy effect
# =====================================================================

def fig_accuracy_effect(df, suffix=""):
    """Bar: correct vs incorrect replay per ROI (objects only, collapsed)."""
    df_obj = df[df["object_scene_retrieval"] == "object"].copy()
    df_obj["behav_correct"] = pd.to_numeric(df_obj["behav_correct"],
                                             errors="coerce")

    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

        for i, (acc_val, label, color) in enumerate([
                (1, "Correct", COLORS["correct"]),
                (0, "Incorrect", COLORS["incorrect"])]):
            sub = df_obj[(df_obj["roi"] == roi)
                         & (df_obj["behav_correct"] == acc_val)]
            sub = sub.assign(diff=sub[RET_COL] - sub[CTRL_COL])
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            if len(smeans) < 3:
                continue
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(color)
            labels.append(label)
            t, p = ttest_1samp(smeans, 0)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 85))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        if len(positions) == 2:
            _bar_with_dots(ax, positions, means, sems, all_vals, colors,
                           labels,
                           ylabel="Replay strength\n(retrieved \u2212 control)")
            ax.set_title(f"{ROI_LABELS[roi]} \u2014 accuracy")
            fig.tight_layout()
            _save(fig, f"4_accuracy_{roi}{suffix}")
        else:
            plt.close(fig)


# =====================================================================
# ANALYSIS 5 — Trial type (object vs scene)
# =====================================================================

def fig_trial_type(df, suffix=""):
    """Bar: object vs scene replay per ROI, collapsed."""
    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = [], [], [], [], [], []

        for i, (tt, label, color) in enumerate([
                ("object", "Object", COLORS["object"]),
                ("scene", "Scene", COLORS["scene"])]):
            sub = df[(df["roi"] == roi)
                     & (df["object_scene_retrieval"] == tt)]
            sub = sub.assign(diff=sub[RET_COL] - sub[CTRL_COL])
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(color)
            labels.append(label)
            t, p = ttest_1samp(smeans, 0)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 85))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 trial type")
        fig.tight_layout()
        _save(fig, f"5_trialtype_{roi}{suffix}")


# =====================================================================
# ANALYSIS 7 — Brain-behavior correlations
# =====================================================================

def fig_brain_behavior(df_obj, suffix=""):
    """Scatter: replay diff vs accuracy, per ROI x session."""
    df_obj = df_obj.copy()
    df_obj["behav_correct"] = pd.to_numeric(df_obj["behav_correct"],
                                             errors="coerce")

    for roi in ROIS:
        for session in ["presleep", "postsleep", "collapsed"]:
            if session == "collapsed":
                sub = df_obj[df_obj["roi"] == roi]
            else:
                sub = df_obj[(df_obj["roi"] == roi)
                             & (df_obj["session"] == session)]

            sub = sub.assign(replay=sub[RET_COL] - sub[CTRL_COL])
            sub = sub.dropna(subset=["replay", "behav_correct"])
            smeans = sub.groupby("subject").agg(
                replay=("replay", "mean"),
                acc=("behav_correct", "mean")
            ).dropna()

            if len(smeans) < 5:
                continue

            r, p = pearsonr(smeans["replay"], smeans["acc"])

            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            ax.scatter(smeans["replay"], smeans["acc"],
                       color=COLORS[roi], edgecolor="white",
                       linewidth=0.4, s=30, alpha=0.85, zorder=3)

            # Regression line
            z = np.polyfit(smeans["replay"], smeans["acc"], 1)
            x_line = np.linspace(smeans["replay"].min(),
                                 smeans["replay"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line),
                    color="black", lw=1.0, ls="--", zorder=2, alpha=0.6)

            ax.set_xlabel("Replay strength (ret \u2212 ctrl)")
            ax.set_ylabel("Retrieval accuracy")

            # Session label for title
            if session == "collapsed":
                sess_label = "collapsed"
            elif session == "presleep":
                sess_label = "pre-sleep"
            else:
                sess_label = "post-sleep"

            sig_marker = _sig(p)
            ax.set_title(f"{ROI_LABELS[roi]} ({sess_label})\n"
                         f"r = {r:.2f}, p = {p:.3f} {sig_marker}",
                         fontsize=10)

            fig.tight_layout()
            _save(fig, f"7_brainbehavior_{roi}_{session}{suffix}")


# =====================================================================
# ANALYSIS 9 — Shuffled baseline validation
# =====================================================================

SHUF_RET_COL = "retrieved_soda_shuffled_abs"
SHUF_CTRL_COL = "control_soda_shuffled_abs"


def fig_shuffled_vs_real(df, suffix=""):
    """Grouped bar: real vs shuffled replay per ROI, collapsed."""
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    width = 0.32
    x = np.arange(len(ROIS))
    rng = np.random.default_rng(42)

    for j, (ret, ctrl, label, color) in enumerate([
            (RET_COL, CTRL_COL, "Real", "#4C72B0"),
            (SHUF_RET_COL, SHUF_CTRL_COL, "Shuffled", "#AAAAAA")]):
        offset = -width / 2 if j == 0 else width / 2
        for i, roi in enumerate(ROIS):
            smeans = _subject_diff_means(df, roi, ret, ctrl)
            m, se = smeans.mean(), smeans.sem()
            t, p = ttest_1samp(smeans, 0)
            ax.bar(x[i] + offset, m, yerr=se, width=width, color=color,
                   edgecolor="#333", linewidth=0.6, capsize=2, alpha=0.75,
                   error_kw={"linewidth": 0.8}, zorder=1,
                   label=label if i == 0 else None)
            jitter = rng.uniform(-width / 5, width / 5, size=len(smeans))
            ax.scatter(x[i] + offset + jitter, smeans.values, color="black",
                       s=8, alpha=0.25, linewidths=0, zorder=3)
            y_top = max(m + se, np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.08
            ax.text(x[i] + offset, y_top, _sig_dagger(p),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ROI_LABELS[r] for r in ROIS])
    ax.set_ylabel("Replay strength\n(retrieved \u2212 control)")
    ax.set_title("Real vs shuffled labels")
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, f"9_shuffled_vs_real{suffix}")


def fig_shuffled_per_session(df, suffix=""):
    """Per-ROI: real vs shuffled, split by session."""
    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        width = 0.32
        x = np.arange(2)  # pre, post
        rng = np.random.default_rng(42)

        for j, (ret, ctrl, label, color) in enumerate([
                (RET_COL, CTRL_COL, "Real", COLORS[roi]),
                (SHUF_RET_COL, SHUF_CTRL_COL, "Shuffled", "#AAAAAA")]):
            offset = -width / 2 if j == 0 else width / 2
            for i, session in enumerate(["presleep", "postsleep"]):
                smeans = _subject_diff_means(df, roi, ret, ctrl, session)
                m, se = smeans.mean(), smeans.sem()
                t, p = ttest_1samp(smeans, 0)
                ax.bar(x[i] + offset, m, yerr=se, width=width, color=color,
                       edgecolor="#333", linewidth=0.6, capsize=2,
                       alpha=0.75, error_kw={"linewidth": 0.8}, zorder=1,
                       label=label if i == 0 else None)
                jitter = rng.uniform(-width / 5, width / 5,
                                     size=len(smeans))
                ax.scatter(x[i] + offset + jitter, smeans.values,
                           color="black", s=8, alpha=0.25,
                           linewidths=0, zorder=3)
                y_top = max(m + se, np.percentile(smeans.values, 85))
                y_top += smeans.std() * 0.08
                ax.text(x[i] + offset, y_top, _sig_dagger(p),
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_ylabel("Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 real vs shuffled")
        ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        _save(fig, f"9_shuffled_{roi}{suffix}")


# =====================================================================
# ANALYSIS 11 — Within-session temporal dynamics
# =====================================================================

def fig_run_effects(df, suffix=""):
    """Per-ROI: early vs late postsleep runs, with LME bracket."""
    df_post = df[df["session"] == "postsleep"].copy()

    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = (
            [], [], [], [], [], [])

        sub = df_post[df_post["roi"] == roi].copy()
        sub["replay_diff"] = sub[RET_COL] - sub[CTRL_COL]

        for i, (runs, label, color) in enumerate([
                ([1, 2, 3], "Early", "#7BAFD4"),
                ([4, 5, 6], "Late", "#D4534E")]):
            run_sub = sub[sub["run"].isin(runs)]
            smeans = run_sub.groupby("subject")["replay_diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(color)
            labels.append(label)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Replay strength\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 post-sleep")

        # LME: replay_diff ~ run + (1|subject) — bracket
        sub_clean = sub.dropna(subset=["replay_diff"])
        try:
            model = smf.mixedlm("replay_diff ~ run", sub_clean,
                                groups=sub_clean["subject"])
            res = model.fit(reml=True, method="powell")
            p_run = res.pvalues["run"]
            y_max = max(m + se for m, se in zip(means, sems))
            y_max = max(y_max, max(v.max() for v in all_vals))
            bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
            _add_bracket(ax, 0, 1, bracket_y, f"LME: {_sig(p_run)}")
        except Exception:
            pass

        fig.tight_layout()
        _save(fig, f"11_runeffect_{roi}{suffix}")


# =====================================================================
# ANALYSIS 13 — Fit quality as moderator
# =====================================================================

R2_RET_COL = "retrieved_w_r_squared"
R2_CTRL_COL = "control_w_r_squared"
SINUS_RET_COL = "retrieved_w_amplitude"
SINUS_CTRL_COL = "control_w_amplitude"


def fig_r2_median_split(df, suffix=""):
    """Per-ROI: high vs low R² replay, collapsed, with paired t-test."""
    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = (
            [], [], [], [], [], [])

        sub = df[df["roi"] == roi].copy()
        sub["replay_diff"] = sub[SINUS_RET_COL] - sub[SINUS_CTRL_COL]
        sub = sub.dropna(subset=["replay_diff", R2_RET_COL])
        medians = sub.groupby("subject")[R2_RET_COL].transform("median")
        sub["r2_high"] = (sub[R2_RET_COL] >= medians).astype(int)

        for i, (val, label, color) in enumerate([
                (1, "High R\u00b2", "#2CA02C"),
                (0, "Low R\u00b2", "#D62728")]):
            grp = sub[sub["r2_high"] == val]
            smeans = grp.groupby("subject")["replay_diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(color)
            labels.append(label)
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="Sinus amplitude\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 R\u00b2 split")

        # Paired t-test bracket (high vs low)
        hi = sub[sub["r2_high"] == 1].groupby("subject")[
            "replay_diff"].mean()
        lo = sub[sub["r2_high"] == 0].groupby("subject")[
            "replay_diff"].mean()
        common = hi.dropna().index.intersection(lo.dropna().index)
        if len(common) >= 3:
            from scipy.stats import ttest_rel
            t_hl, p_hl = ttest_rel(hi.loc[common], lo.loc[common])
            y_max = max(m + se for m, se in zip(means, sems))
            y_max = max(y_max, max(v.max() for v in all_vals))
            bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
            _add_bracket(ax, 0, 1, bracket_y, _sig(p_hl))

        fig.tight_layout()
        _save(fig, f"13a_r2split_{roi}{suffix}")


def fig_r2_as_metric(df, suffix=""):
    """Bar: R² diff (ret - ctrl) per ROI, collapsed + per session."""
    # Collapsed
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    positions, means, sems, all_vals, colors, labels = (
        [], [], [], [], [], [])

    for i, roi in enumerate(ROIS):
        sub = df[df["roi"] == roi].copy()
        sub["r2_diff"] = sub[R2_RET_COL] - sub[R2_CTRL_COL]
        smeans = sub.groupby("subject")["r2_diff"].mean().dropna()
        t, p = ttest_1samp(smeans, 0)
        positions.append(i)
        means.append(smeans.mean())
        sems.append(smeans.sem())
        all_vals.append(smeans.values)
        colors.append(COLORS[roi])
        labels.append(ROI_LABELS[roi])
        y_top = smeans.mean() + smeans.sem() + smeans.std() * 0.12
        y_top = max(y_top, np.percentile(smeans.values, 90))
        ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                   ylabel="R\u00b2 difference\n(retrieved \u2212 control)")
    ax.set_title("Fit quality as replay metric")
    fig.tight_layout()
    _save(fig, f"13c_r2metric_collapsed{suffix}")


def fig_r2_session_effect(df, suffix=""):
    """Per-ROI: pre vs post R² diff, with LME bracket."""
    for roi in ROIS:
        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        positions, means, sems, all_vals, colors, labels = (
            [], [], [], [], [], [])

        sub = df[df["roi"] == roi].copy()
        sub["r2_diff"] = sub[R2_RET_COL] - sub[R2_CTRL_COL]

        for i, session in enumerate(["presleep", "postsleep"]):
            s_sub = sub[sub["session"] == session]
            smeans = s_sub.groupby("subject")["r2_diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            positions.append(i)
            means.append(smeans.mean())
            sems.append(smeans.sem())
            all_vals.append(smeans.values)
            colors.append(COLORS[session])
            labels.append("Pre" if session == "presleep" else "Post")
            y_top = max(smeans.mean() + smeans.sem(),
                        np.percentile(smeans.values, 90))
            y_top += smeans.std() * 0.1
            ax.text(i, y_top, _sig_dagger(p), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        _bar_with_dots(ax, positions, means, sems, all_vals, colors, labels,
                       ylabel="R\u00b2 difference\n(retrieved \u2212 control)")
        ax.set_title(f"{ROI_LABELS[roi]} \u2014 R\u00b2 session")

        # LME: r2_diff ~ is_post + (1|subject)
        sub_clean = sub.dropna(subset=["r2_diff"]).copy()
        sub_clean["is_post"] = (
            sub_clean["session"] == "postsleep").astype(int)
        try:
            model = smf.mixedlm("r2_diff ~ is_post", sub_clean,
                                groups=sub_clean["subject"])
            res = model.fit(reml=True, method="powell")
            p_sess = res.pvalues["is_post"]
            y_max = max(m + se for m, se in zip(means, sems))
            y_max = max(y_max, max(v.max() for v in all_vals))
            bracket_y = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
            _add_bracket(ax, 0, 1, bracket_y, f"LME: {_sig(p_sess)}")
        except Exception:
            pass

        fig.tight_layout()
        _save(fig, f"13d_r2session_{roi}{suffix}")


# =====================================================================
# SUMMARY FOREST PLOT — Replay detection across ROIs and sessions
# =====================================================================

def fig_forest_detection(df, suffix=""):
    """Forest plot of effect sizes (Cohen's d) for replay detection."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    rows_data = []
    y_pos = 0
    y_positions = []
    y_labels = []
    y_colors = []

    for session_label, session in [("Collapsed", None),
                                    ("Pre-sleep", "presleep"),
                                    ("Post-sleep", "postsleep")]:
        for roi in ROIS:
            smeans = _subject_diff_means(df, roi, RET_COL, CTRL_COL, session)
            n = len(smeans)
            d = smeans.mean() / smeans.std(ddof=1) if smeans.std(ddof=1) > 0 else 0
            se_d = np.sqrt(1 / n + d**2 / (2 * n))
            t, p = ttest_1samp(smeans, 0)

            rows_data.append((d, se_d, p))
            y_positions.append(y_pos)
            y_labels.append(f"{session_label} \u2014 {ROI_LABELS[roi]}")
            y_colors.append(COLORS[roi])
            y_pos += 1
        y_pos += 0.5  # Gap between session groups

    for i, (d, se_d, p) in enumerate(rows_data):
        ax.errorbar(d, y_positions[i], xerr=1.96 * se_d, fmt="o",
                    color=y_colors[i], markersize=6, capsize=3,
                    linewidth=1.2, markeredgecolor="black",
                    markeredgewidth=0.4)
        # Significance label
        star = _sig(p)
        ax.text(d + 1.96 * se_d + 0.03, y_positions[i], star,
                va="center", ha="left", fontsize=8)

    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Effect size (Cohen's d)")
    ax.set_title("Replay detection \u2014 effect sizes")
    ax.invert_yaxis()
    # Pad right side so significance labels don't clip
    xmax = max(d + 1.96 * se_d for d, se_d, _ in rows_data)
    ax.set_xlim(left=-0.35, right=xmax + 0.18)
    fig.tight_layout()
    _save(fig, f"2_forest_detection{suffix}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(MASTER_CSV)
    df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")
    df_obj = df[df["object_scene_retrieval"] == "object"].copy()
    df_scene = df[df["object_scene_retrieval"] == "scene"].copy()

    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")
    print(f"Object: {len(df_obj)}, Scene: {len(df_scene)}")
    print(f"Output: {OUT_DIR}\n")

    # === FULL (unmatched) ===
    print("=" * 50)
    print("  FULL (unmatched)")
    print("=" * 50)

    print("\n  Analysis 2 — Replay detection:")
    fig_replay_detection_collapsed(df)
    fig_replay_detection_per_session(df)
    fig_forest_detection(df)

    print("\n  Analysis 3 — Session effect:")
    fig_session_effect_alltrials(df)
    fig_session_effect_objects(df_obj)
    fig_session_effect_scenes(df_scene)

    print("\n  Analysis 4 — Accuracy:")
    fig_accuracy_effect(df)

    print("\n  Analysis 5 — Trial type:")
    fig_trial_type(df)

    print("\n  Analysis 7 — Brain-behavior:")
    fig_brain_behavior(df_obj)

    print("\n  Analysis 9 — Shuffled validation:")
    fig_shuffled_vs_real(df)
    fig_shuffled_per_session(df)

    print("\n  Analysis 11 — Run effects:")
    fig_run_effects(df)

    print("\n  Analysis 13 — Fit quality:")
    fig_r2_median_split(df)
    fig_r2_as_metric(df)
    fig_r2_session_effect(df)

    # === MATCHED (postsleep first-half) ===
    print("\n" + "=" * 50)
    print("  MATCHED (postsleep first-half)")
    print("=" * 50)

    df_m = _subsample_postsleep_first_half(df)
    df_obj_m = df_m[df_m["object_scene_retrieval"] == "object"].copy()
    df_scene_m = df_m[df_m["object_scene_retrieval"] == "scene"].copy()

    ref_roi = ROIS[0]
    n_pre = len(df_m[(df_m["session"] == "presleep")
                      & (df_m["roi"] == ref_roi)])
    n_post = len(df_m[(df_m["session"] == "postsleep")
                       & (df_m["roi"] == ref_roi)])
    print(f"  Pre: {n_pre}, Post: {n_post} trials/ROI\n")

    print("  Analysis 2 — Replay detection (matched):")
    fig_replay_detection_collapsed(df_m, suffix="_matched")
    fig_replay_detection_per_session(df_m, suffix="_matched")
    fig_forest_detection(df_m, suffix="_matched")

    print("\n  Analysis 3 — Session effect (matched):")
    fig_session_effect_alltrials(df_m, suffix="_matched")
    fig_session_effect_objects(df_obj_m, suffix="_matched")
    fig_session_effect_scenes(df_scene_m, suffix="_matched")

    print("\n  Analysis 4 — Accuracy (matched):")
    fig_accuracy_effect(df_m, suffix="_matched")

    print("\n  Analysis 5 — Trial type (matched):")
    fig_trial_type(df_m, suffix="_matched")

    print("\n  Analysis 7 — Brain-behavior (matched):")
    fig_brain_behavior(df_obj_m, suffix="_matched")

    print(f"\nDone. All panels saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
