#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Position-adaptive SODA and template matching.

Model assumption: When cued at position k, participants replay items 1→2→...→k.
  - Position 1: no replay needed (0 preceding items) → no slope expected
  - Position 2: replay 1→2 (slope from 2 points, noisy)
  - Position 3+: replay 1→...→k, meaningful slope
  - Position 5: maximum replay (4 preceding items, full chain)

Computes:
  1. Adaptive SODA: slope over items [1..k] instead of always [1..5]
  2. Adaptive template matching: expected sinusoidal pattern for k items,
     ISI explored as free parameter since replay speed is unknown
  3. Full vs adaptive comparison of position effects
  4. Split by scene/object retrieval

Usage:
    python 4.6c_position_adaptive_SODA.py [ROI] [presleep|postsleep]
"""

import os
import sys
import ast
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, pearsonr
from matplotlib import pyplot as plt


# =====================================================================
# HELPERS
# =====================================================================

def parse_prob_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return [float(v) for v in ast.literal_eval(x)]
        except (ValueError, SyntaxError):
            pass
    return [np.nan] * 5


def load_soda_results(csv_dir, roi):
    pattern = opj(csv_dir, f"{roi}_soda_results_per_combination.csv")
    files = glob.glob(pattern)
    if not files:
        files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
        files = [f for f in files if roi in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No SODA results for ROI '{roi}' in {csv_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    for col in ["tr", "retrieved_soda", "control_soda",
                 "retrieved_soda_abs", "control_soda_abs",
                 "retrieved_position", "behav_correct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["retrieved_probs", "control_probs"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_prob_list)

    return df


# =====================================================================
# 1. ADAPTIVE SODA: SLOPE OVER ITEMS [1..k]
# =====================================================================

def adaptive_soda_slope(probs, cue_position):
    """Compute SODA slope using only items 1..cue_position.

    Returns NaN if cue_position < 2 (can't compute slope from 1 point).
    NaN handling: NaNs are dropped but original position indices are preserved.
    Requires at least 2 valid items.
    """
    k = int(cue_position)
    if k < 2 or k > len(probs):
        return np.nan
    truncated = np.array([probs[i] for i in range(k)], dtype=float)
    positions = np.arange(1, k + 1, dtype=float)
    valid = ~np.isnan(truncated)
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(positions[valid], truncated[valid], 1)[0])


def compute_adaptive_soda_per_trial(df):
    """For each trial × TR, compute adaptive SODA (truncated to cue position).

    Also computes the standard full-sequence SODA for comparison.
    """
    results = []

    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in df.columns]

    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))

        # Get cue position (constant within trial)
        pos_vals = trial_df["retrieved_position"].dropna()
        if len(pos_vals) == 0:
            continue
        cue_pos = int(pos_vals.iloc[0])
        trial_info["cue_position"] = cue_pos
        trial_info["n_items_to_replay"] = max(0, cue_pos - 1)

        # Per-TR SODA values
        trial_df = trial_df.sort_values("tr")
        adaptive_slopes = []
        full_slopes = []

        for _, row in trial_df.iterrows():
            ret_probs = row["retrieved_probs"]
            ctrl_probs = row["control_probs"]

            # Full SODA (all 5 items, NaN-tolerant like seq_fitter)
            ret_arr = np.array(ret_probs, dtype=float)
            positions_full = np.arange(1, 6, dtype=float)
            valid_full = ~np.isnan(ret_arr)
            if valid_full.sum() >= 2:
                full_slope = float(np.polyfit(positions_full[valid_full], ret_arr[valid_full], 1)[0])
            else:
                full_slope = np.nan

            # Adaptive SODA (items 1..k)
            adapt_slope = adaptive_soda_slope(ret_probs, cue_pos)

            # Adaptive control
            adapt_ctrl = adaptive_soda_slope(ctrl_probs, cue_pos)

            adaptive_slopes.append(adapt_slope)
            full_slopes.append(full_slope)

        adaptive_slopes = np.array(adaptive_slopes)
        full_slopes = np.array(full_slopes)

        # Trial-level summary metrics
        trial_info["adaptive_soda_mean"] = np.nanmean(adaptive_slopes)
        trial_info["adaptive_soda_abs_mean"] = np.nanmean(np.abs(adaptive_slopes))
        trial_info["adaptive_soda_variance"] = np.nanvar(adaptive_slopes)
        trial_info["adaptive_soda_ptt"] = np.nanmax(adaptive_slopes) - np.nanmin(adaptive_slopes) \
            if not np.all(np.isnan(adaptive_slopes)) else np.nan

        trial_info["full_soda_mean"] = np.nanmean(full_slopes)
        trial_info["full_soda_abs_mean"] = np.nanmean(np.abs(full_slopes))
        trial_info["full_soda_variance"] = np.nanvar(full_slopes)
        trial_info["full_soda_ptt"] = np.nanmax(full_slopes) - np.nanmin(full_slopes) \
            if not np.all(np.isnan(full_slopes)) else np.nan

        # Store timecourses as lists for template matching
        trial_info["adaptive_slopes_tc"] = adaptive_slopes.tolist()
        trial_info["full_slopes_tc"] = full_slopes.tolist()

        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 2. POSITION-ADAPTIVE TEMPLATE MATCHING
# =====================================================================

def sinusoidal_hrf(t_seconds, onset_s=0.0, peak_s=5.0, width_s=5.0):
    """Simplified sinusoidal HRF: half-sine peaking at onset + peak_s.

    Returns a non-negative response within [onset, onset + width].
    """
    t_rel = t_seconds - onset_s
    response = np.zeros_like(t_seconds)
    in_window = (t_rel >= 0) & (t_rel <= width_s)
    # Half-sine: peaks at width/2
    response[in_window] = np.sin(np.pi * t_rel[in_window] / width_s)
    return response


def generate_adaptive_template(n_trs, n_items, isi_s, tr=1.5,
                                hrf_peak_s=5.0, hrf_width_s=5.0,
                                onset_lag_s=0.0):
    """Generate expected SODA slope timecourse for n_items replayed with given ISI.

    Parameters
    ----------
    n_trs : int
        Number of TRs in the timecourse.
    n_items : int
        Number of items replayed (= cue_position).
    isi_s : float
        Inter-stimulus interval in seconds between successive items.
    tr : float
        Repetition time in seconds.
    hrf_peak_s, hrf_width_s : float
        HRF shape parameters (half-sine peak time and width).
    onset_lag_s : float
        Delay before replay starts (seconds after cue onset).

    Returns
    -------
    slopes : array (n_trs,)
        Unit-norm SODA slope timecourse.
    """
    if n_items < 2:
        return np.zeros(n_trs)

    t = np.arange(n_trs) * tr  # time in seconds

    # HRF response for each item
    probs = np.zeros((n_items, n_trs))
    for i in range(n_items):
        item_onset_s = onset_lag_s + i * isi_s
        probs[i] = sinusoidal_hrf(t, onset_s=item_onset_s,
                                   peak_s=hrf_peak_s, width_s=hrf_width_s)

    # Compute SODA slopes at each TR
    positions = np.arange(1, n_items + 1, dtype=float)
    slopes = np.array([
        np.polyfit(positions, probs[:, ti], 1)[0]
        for ti in range(n_trs)
    ])

    norm = np.linalg.norm(slopes)
    return slopes / norm if norm > 0 else slopes


def adaptive_template_matching(soda_tc, cue_position, n_trs, tr=1.5,
                                isi_range=(0.1, 4.0), n_isi=30,
                                onset_range=(0.0, 3.0), n_onset=10):
    """Match observed SODA timecourse to position-adaptive templates.

    Searches over ISI × onset-lag grid to find best-matching template.

    Returns
    -------
    dict with best_r, best_abs_r, best_isi, best_onset_lag
    """
    soda_tc = np.asarray(soda_tc, dtype=float)
    n_items = int(cue_position)

    if n_items < 2:
        return {"best_r": np.nan, "best_abs_r": np.nan,
                "best_isi": np.nan, "best_onset_lag": np.nan}

    isi_values = np.linspace(isi_range[0], isi_range[1], n_isi)
    onset_values = np.linspace(onset_range[0], onset_range[1], n_onset)

    best_r = -np.inf
    best_isi = np.nan
    best_onset = np.nan

    for isi in isi_values:
        for onset in onset_values:
            template = generate_adaptive_template(
                n_trs, n_items, isi, tr=tr, onset_lag_s=onset,
            )
            valid = ~np.isnan(soda_tc) & ~np.isnan(template)
            if valid.sum() < 4:
                continue
            # Check template has variance (not flat)
            if np.std(template[valid]) < 1e-10:
                continue
            r, _ = pearsonr(soda_tc[valid], template[valid])
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r = r
                best_isi = isi
                best_onset = onset

    best_abs_r = abs(best_r) if np.isfinite(best_r) else np.nan

    return {
        "best_r": best_r if np.isfinite(best_r) else np.nan,
        "best_abs_r": best_abs_r,
        "best_isi": best_isi,
        "best_onset_lag": best_onset,
    }


def compute_adaptive_template_matching(trial_metrics_df, tr=1.5):
    """Apply adaptive template matching to each trial."""
    results = []

    for idx, row in trial_metrics_df.iterrows():
        cue_pos = row["cue_position"]

        # Use adaptive SODA timecourse
        adapt_tc = row.get("adaptive_slopes_tc", None)
        full_tc = row.get("full_slopes_tc", None)

        info = {}

        # Adaptive template matching
        if adapt_tc is not None and cue_pos >= 2:
            tc = np.array(adapt_tc)
            tm = adaptive_template_matching(tc, cue_pos, len(tc), tr=tr)
            info["adapt_tm_r"] = tm["best_r"]
            info["adapt_tm_abs_r"] = tm["best_abs_r"]
            info["adapt_tm_isi"] = tm["best_isi"]
            info["adapt_tm_onset"] = tm["best_onset_lag"]
        else:
            info["adapt_tm_r"] = np.nan
            info["adapt_tm_abs_r"] = np.nan
            info["adapt_tm_isi"] = np.nan
            info["adapt_tm_onset"] = np.nan

        # Full template matching (for comparison, using cue_pos items)
        if full_tc is not None and cue_pos >= 2:
            tc = np.array(full_tc)
            tm_full = adaptive_template_matching(tc, cue_pos, len(tc), tr=tr)
            info["full_tm_r"] = tm_full["best_r"]
            info["full_tm_abs_r"] = tm_full["best_abs_r"]
            info["full_tm_isi"] = tm_full["best_isi"]
            info["full_tm_onset"] = tm_full["best_onset_lag"]
        else:
            info["full_tm_r"] = np.nan
            info["full_tm_abs_r"] = np.nan
            info["full_tm_isi"] = np.nan
            info["full_tm_onset"] = np.nan

        results.append(info)

    tm_df = pd.DataFrame(results)
    return pd.concat([trial_metrics_df.reset_index(drop=True), tm_df], axis=1)


# =====================================================================
# 3. PLOTTING
# =====================================================================

def plot_full_vs_adaptive_position(metrics_df, roi, out_dir):
    """Compare full-sequence vs adaptive SODA position effects."""
    if "cue_position" not in metrics_df.columns:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Full-sequence SODA by position ---
    for ax, (metric, label) in zip(axes[0], [
        ("full_soda_abs_mean", "Full |SODA| (all 5 items)"),
        ("full_soda_variance", "Full SODA variance"),
        ("full_soda_ptt", "Full SODA peak-to-trough"),
    ]):
        if metric not in metrics_df.columns:
            ax.set_visible(False)
            continue
        _plot_position_bars(ax, metrics_df, metric, label, color="#E1812C")

    # --- Row 2: Adaptive SODA by position ---
    for ax, (metric, label) in zip(axes[1], [
        ("adaptive_soda_abs_mean", "Adaptive |SODA| (items 1..k)"),
        ("adaptive_soda_variance", "Adaptive SODA variance"),
        ("adaptive_soda_ptt", "Adaptive SODA peak-to-trough"),
    ]):
        if metric not in metrics_df.columns:
            ax.set_visible(False)
            continue
        _plot_position_bars(ax, metrics_df, metric, label, color="#3274A1")

    fig.suptitle(
        f"Full (top) vs Adaptive (bottom) SODA by Cue Position — {roi}\n"
        "Adaptive: only items 1..k used for slope at position k",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_01_full_vs_adaptive_position.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_position_bars(ax, df, metric, label, color="#3274A1"):
    """Bar plot of metric by cue position with individual subject lines."""
    subj_pos = df.groupby(["subject", "cue_position"])[metric].mean().reset_index()
    pos_stats = subj_pos.groupby("cue_position")[metric].agg(["mean", "std", "count"])
    pos_stats["sem"] = pos_stats["std"] / np.sqrt(pos_stats["count"])

    positions = pos_stats.index
    ax.bar(positions, pos_stats["mean"], yerr=pos_stats["sem"],
           color=color, alpha=0.6, capsize=4, edgecolor="white")

    for sub in subj_pos["subject"].unique():
        sd = subj_pos[subj_pos["subject"] == sub]
        ax.plot(sd["cue_position"], sd[metric], "o-", alpha=0.2, ms=3, color="grey")

    ax.set_xlabel("Cue position")
    ax.set_ylabel(label)
    ax.set_xticks([int(p) for p in positions])
    ax.spines[["top", "right"]].set_visible(False)

    # Spearman correlation
    valid = subj_pos.dropna(subset=[metric])
    if len(valid) > 5:
        rho, p = spearmanr(valid["cue_position"], valid[metric])
        ax.set_title(f"{label}\nrho={rho:.3f}, p={p:.3f}")
    else:
        ax.set_title(label)


def plot_template_matching_results(metrics_df, roi, out_dir):
    """Template matching results: ISI distribution, position effects."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- 1a: Best ISI distribution ---
    ax = axes[0, 0]
    if "adapt_tm_isi" in metrics_df.columns:
        isi_vals = metrics_df["adapt_tm_isi"].dropna()
        if len(isi_vals) > 0:
            ax.hist(isi_vals, bins=30, color="#3274A1", alpha=0.7, edgecolor="white")
            ax.axvline(isi_vals.median(), color="red", ls="--", lw=2,
                       label=f"median={isi_vals.median():.2f}s")
            ax.set_xlabel("Best-fit ISI (seconds)")
            ax.set_ylabel("Count")
            ax.set_title(f"Optimal ISI across trials\n(M={isi_vals.mean():.2f}s, Md={isi_vals.median():.2f}s)")
            ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # --- 1b: Best ISI by position ---
    ax = axes[0, 1]
    if "adapt_tm_isi" in metrics_df.columns and "cue_position" in metrics_df.columns:
        valid = metrics_df.dropna(subset=["adapt_tm_isi", "cue_position"])
        valid = valid[valid["cue_position"] >= 2]
        if len(valid) > 0:
            pos_isi = valid.groupby("cue_position")["adapt_tm_isi"].agg(["mean", "std", "count"])
            pos_isi["sem"] = pos_isi["std"] / np.sqrt(pos_isi["count"])
            ax.bar(pos_isi.index, pos_isi["mean"], yerr=pos_isi["sem"],
                   color="#3274A1", alpha=0.7, capsize=4, edgecolor="white")
            ax.set_xlabel("Cue position")
            ax.set_ylabel("Best-fit ISI (s)")
            ax.set_title("Optimal ISI by cue position")
            ax.set_xticks([int(p) for p in pos_isi.index])
    ax.spines[["top", "right"]].set_visible(False)

    # --- 1c: Onset lag distribution ---
    ax = axes[0, 2]
    if "adapt_tm_onset" in metrics_df.columns:
        onset_vals = metrics_df["adapt_tm_onset"].dropna()
        if len(onset_vals) > 0:
            ax.hist(onset_vals, bins=30, color="#9467BD", alpha=0.7, edgecolor="white")
            ax.set_xlabel("Best-fit onset lag (seconds)")
            ax.set_ylabel("Count")
            ax.set_title(f"Replay onset delay\n(M={onset_vals.mean():.2f}s)")
    ax.spines[["top", "right"]].set_visible(False)

    # --- 2a: Template |r| by position ---
    ax = axes[1, 0]
    if "adapt_tm_abs_r" in metrics_df.columns:
        _plot_position_bars(ax, metrics_df[metrics_df["cue_position"] >= 2],
                            "adapt_tm_abs_r",
                            "Template |r| (adaptive)", color="#2CA02C")
    ax.spines[["top", "right"]].set_visible(False)

    # --- 2b: Template |r| adaptive vs control by trial type ---
    ax = axes[1, 1]
    if "adapt_tm_abs_r" in metrics_df.columns and "object_scene_retrieval" in metrics_df.columns:
        for ttype, color, marker in [("object", "#3274A1", "o"), ("scene", "#E1812C", "s")]:
            sub_df = metrics_df[
                (metrics_df["object_scene_retrieval"] == ttype) &
                (metrics_df["cue_position"] >= 2)
            ]
            if len(sub_df) == 0:
                continue
            subj = sub_df.groupby("subject")["adapt_tm_abs_r"].mean()
            ax.scatter(np.random.default_rng(42).normal({"object": 0, "scene": 1}[ttype], 0.05, len(subj)),
                       subj.values, color=color, alpha=0.6, s=30, marker=marker, label=ttype)
            ax.errorbar({"object": 0, "scene": 1}[ttype], subj.mean(),
                        yerr=subj.sem(), color=color, capsize=5, lw=2, fmt="none")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Object", "Scene"])
        ax.set_ylabel("Template |r|")
        ax.set_title("Template match by trial type")
        ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # --- 2c: Example best-fit templates for different positions ---
    ax = axes[1, 2]
    tr = 1.5
    median_isi = metrics_df["adapt_tm_isi"].median() if "adapt_tm_isi" in metrics_df.columns else 1.0
    if np.isnan(median_isi):
        median_isi = 1.0

    colors_pos = {2: "#FFC107", 3: "#FF9800", 4: "#F44336", 5: "#9C27B0"}
    for k in [2, 3, 4, 5]:
        template = generate_adaptive_template(10, k, median_isi, tr=tr)
        ax.plot(np.arange(10), template, "o-", color=colors_pos[k], lw=2,
                ms=4, label=f"k={k} items")
    ax.set_xlabel("TR")
    ax.set_ylabel("Expected SODA slope (norm.)")
    ax.set_title(f"Template shapes at median ISI={median_isi:.2f}s")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="grey", ls=":", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Adaptive Template Matching — {roi}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_02_template_matching.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_trial_type_position_interaction(metrics_df, roi, out_dir):
    """Position effects split by scene vs object retrieval."""
    if "object_scene_retrieval" not in metrics_df.columns:
        return

    trial_types = sorted(metrics_df["object_scene_retrieval"].dropna().unique())
    metrics_to_plot = [
        ("adaptive_soda_abs_mean", "Adaptive |SODA|"),
        ("full_soda_abs_mean", "Full |SODA|"),
        ("adapt_tm_abs_r", "Template |r|"),
    ]
    metrics_to_plot = [(m, l) for m, l in metrics_to_plot if m in metrics_df.columns]

    if len(metrics_to_plot) == 0 or len(trial_types) == 0:
        return

    n_met = len(metrics_to_plot)
    fig, axes = plt.subplots(len(trial_types), n_met,
                             figsize=(5.5 * n_met, 5 * len(trial_types)), squeeze=False)

    print("\n--- Position × Trial-Type Interaction ---")

    for row_idx, ttype in enumerate(trial_types):
        df_sub = metrics_df[metrics_df["object_scene_retrieval"] == ttype]

        for col_idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]

            # Filter to positions with enough data
            df_valid = df_sub.dropna(subset=[metric, "cue_position"])
            if metric.startswith("adapt"):
                df_valid = df_valid[df_valid["cue_position"] >= 2]

            if len(df_valid) < 5:
                ax.set_title(f"{ttype}: {label}\n(insufficient data)")
                continue

            _plot_position_bars(ax, df_valid, metric, f"{ttype}: {label}")

            # Print stats
            subj_pos = df_valid.groupby(["subject", "cue_position"])[metric].mean().reset_index()
            valid = subj_pos.dropna(subset=[metric])
            if len(valid) > 5:
                rho, p = spearmanr(valid["cue_position"], valid[metric])
                print(f"  {ttype} — {label}: rho={rho:.3f}, p={p:.3f}")

    fig.suptitle(f"Position Effects by Trial Type — {roi}\n"
                 "Adaptive: items 1..k | Full: items 1..5",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_03_position_by_trialtype.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_n_items_vs_sequentiality(metrics_df, roi, out_dir):
    """Direct test: does number of replayed items predict sequentiality?"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, label) in zip(axes, [
        ("adaptive_soda_abs_mean", "Adaptive |SODA|"),
        ("adaptive_soda_ptt", "Adaptive peak-to-trough"),
        ("adapt_tm_abs_r", "Template |r|"),
    ]):
        if metric not in metrics_df.columns:
            ax.set_visible(False)
            continue

        df_valid = metrics_df.dropna(subset=[metric, "n_items_to_replay"])
        if metric.startswith("adapt"):
            df_valid = df_valid[df_valid["n_items_to_replay"] >= 1]

        if len(df_valid) < 5:
            ax.set_title(f"{label}\n(insufficient data)")
            continue

        subj_n = df_valid.groupby(["subject", "n_items_to_replay"])[metric].mean().reset_index()
        n_stats = subj_n.groupby("n_items_to_replay")[metric].agg(["mean", "std", "count"])
        n_stats["sem"] = n_stats["std"] / np.sqrt(n_stats["count"])

        ax.bar(n_stats.index, n_stats["mean"], yerr=n_stats["sem"],
               color="#3274A1", alpha=0.7, capsize=4, edgecolor="white")

        for sub in subj_n["subject"].unique():
            sd = subj_n[subj_n["subject"] == sub]
            ax.plot(sd["n_items_to_replay"], sd[metric], "o-", alpha=0.2, ms=3, color="grey")

        ax.set_xlabel("Number of items to replay (k − 1)")
        ax.set_ylabel(label)
        ax.set_xticks([int(n) for n in n_stats.index])
        ax.spines[["top", "right"]].set_visible(False)

        rho, p = spearmanr(subj_n["n_items_to_replay"], subj_n[metric])
        ax.set_title(f"{label}\nrho={rho:.3f}, p={p:.3f}")
        print(f"  n_items_to_replay vs {label}: rho={rho:.3f}, p={p:.3f}")

    fig.suptitle(
        f"Number of Replayed Items vs Sequentiality — {roi}\n"
        "Direct prediction: more items to replay → more sequential reactivation",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_04_n_items_vs_sequentiality.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    roi = sys.argv[1] if len(sys.argv) > 1 else None
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{pre_post}", "SODA_results_csv")

    csv_files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    available_rois = [
        os.path.basename(f).replace("_soda_results_per_combination.csv", "")
        for f in csv_files
    ]

    if roi:
        if roi not in available_rois:
            raise ValueError(f"ROI '{roi}' not found. Available: {available_rois}")
        rois_to_process = [roi]
    else:
        rois_to_process = available_rois

    TR = 1.5
    print(f"Processing {len(rois_to_process)} ROIs: {rois_to_process}")
    print(f"Session: {pre_post}, TR: {TR}s")

    for roi_name in rois_to_process:
        print(f"\n{'=' * 60}")
        print(f"ROI: {roi_name}")
        print(f"{'=' * 60}")

        out_dir = opj(script_dir, "4.6c_position_adaptive", roi_name)
        os.makedirs(out_dir, exist_ok=True)

        # Load data
        df = load_soda_results(csv_dir, roi_name)
        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

        # Step 1: Compute adaptive SODA
        print("Computing adaptive SODA (items 1..k per trial)...")
        trial_metrics = compute_adaptive_soda_per_trial(df)
        print(f"  {len(trial_metrics)} trials")

        # Add trial type info
        if "object_scene_retrieval" in df.columns:
            group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
            group_cols = [c for c in group_cols if c in df.columns]
            type_map = df.groupby(group_cols)["object_scene_retrieval"].first()
            trial_metrics = trial_metrics.merge(
                type_map.reset_index()[group_cols],
                on=[c for c in group_cols if c in trial_metrics.columns],
                how="left",
                suffixes=("", "_dup"),
            )

        # Step 2: Adaptive template matching
        print("Computing adaptive template matching (ISI × onset search)...")
        print("  (this may take a minute...)")
        trial_metrics = compute_adaptive_template_matching(trial_metrics, tr=TR)

        # Save results
        # Drop timecourse columns for CSV (they're lists, not useful in CSV)
        save_cols = [c for c in trial_metrics.columns if "_tc" not in c]
        trial_metrics[save_cols].to_csv(
            opj(out_dir, f"{roi_name}_adaptive_metrics.csv"), index=False,
        )

        # Step 3: Position statistics
        print("\n--- Position Summary ---")
        for pos in sorted(trial_metrics["cue_position"].dropna().unique()):
            pos_data = trial_metrics[trial_metrics["cue_position"] == pos]
            n = len(pos_data)
            adapt_mean = pos_data["adaptive_soda_abs_mean"].mean()
            full_mean = pos_data["full_soda_abs_mean"].mean()
            print(f"  Position {int(pos)}: n={n}, "
                  f"adaptive |SODA|={adapt_mean:.5f}, "
                  f"full |SODA|={full_mean:.5f}, "
                  f"items_to_replay={int(max(0, pos-1))}")

        # Step 4: Plots
        print("\nGenerating plots...")
        plot_full_vs_adaptive_position(trial_metrics, roi_name, out_dir)
        plot_template_matching_results(trial_metrics, roi_name, out_dir)
        plot_trial_type_position_interaction(trial_metrics, roi_name, out_dir)
        plot_n_items_vs_sequentiality(trial_metrics, roi_name, out_dir)

        print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
