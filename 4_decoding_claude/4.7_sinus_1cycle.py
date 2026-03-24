# -*- coding: utf-8 -*-
"""Windowed single-cycle sinusoidal fitting of SODA timecourses.

Following Wittkuhn & Schuck (2021, Nature Communications).

Usage
-----
    python 4.7_sinus_1cycle.py [ROI] [presleep|postsleep]
"""

import os
import sys
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt

from utils import script_output_dir


# =====================================================================
# 1. WINDOWED SINGLE-CYCLE SINUSOIDAL MODEL
# =====================================================================

def windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
    """Single-cycle sinusoidal model with smooth on/off ramps.

    y(t) = A * sin(2π/λ * (t - d)) * W(t; d, λ, taper) + c
    """
    t = np.asarray(t, dtype=float)
    phase = 2 * np.pi / duration * (t - onset)
    sine_part = amplitude * np.sin(phase)

    active = (t >= onset) & (t <= onset + duration)
    taper_len = taper_frac * duration
    window = np.zeros_like(t)
    window[active] = 1.0

    if taper_len > 0:
        rise = active & (t < onset + taper_len)
        if rise.any():
            window[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - onset) / taper_len))
        fall = active & (t > onset + duration - taper_len)
        if fall.any():
            window[fall] = 0.5 * (1 - np.cos(np.pi * (onset + duration - t[fall]) / taper_len))

    return sine_part * window + offset


def _windowed_for_curvefit(t, amplitude, duration, onset, offset):
    return windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15)


def _full_cycle_visible(onset, duration, t_min, t_max, min_cycle_frac=0.90):
    """Check that at least *min_cycle_frac* of the cycle falls within the data range.

    This ensures both the positive and negative lobes of the sinusoid are
    captured by the data, preventing half-cycle fits that chase a single
    strong deflection.
    """
    cycle_start = onset
    cycle_end = onset + duration
    overlap_start = max(cycle_start, t_min)
    overlap_end = min(cycle_end, t_max)
    overlap = max(0.0, overlap_end - overlap_start)
    return (overlap / duration) >= min_cycle_frac


def fit_windowed_sinusoid(
    trs: np.ndarray,
    soda_values: np.ndarray,
    duration_min: float = 3.0,
    duration_max: float = 12.0,
    n_restarts: int = 10,
    min_cycle_frac: float = 0.90,
) -> dict:
    """Fit the windowed single-cycle sinusoidal model to a SODA timecourse.

    Parameters
    ----------
    min_cycle_frac : float
        Minimum fraction of the full cycle that must fall within the data
        range.  Set to 0.90 by default so that both the positive and
        negative phases of the sinusoid are required to be visible.
    """
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(soda_values)

    if valid.sum() < 4:
        return _empty_windowed_result(len(trs))

    t, y = trs[valid], soda_values[valid]
    t_min, t_max = t.min(), t.max()
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    # Tighten onset: cycle must start no earlier than data start and must
    # leave room for at least duration_min within the data range.
    onset_lower = t_min
    onset_upper = max(t_min, t_max - duration_min)

    lower = [-np.inf, duration_min, onset_lower, -np.inf]
    upper = [np.inf, duration_max, onset_upper, np.inf]

    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, (t_max - t_min) * 0.8, t_min, y_mean]
        elif restart == 1:
            p0 = [-a0, (t_max - t_min) * 0.8, t_min, y_mean]
        else:
            p0 = [
                rng.uniform(-a0 * 2, a0 * 2),
                rng.uniform(duration_min, min(duration_max, t_max - t_min + 2)),
                rng.uniform(onset_lower, onset_lower + (onset_upper - onset_lower) * 0.5),
                y_mean + rng.normal(0, a0 * 0.1),
            ]
        # Clip p0 to bounds
        p0 = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, lower, upper)]
        try:
            popt, _ = curve_fit(_windowed_for_curvefit, t, y, p0=p0, bounds=(lower, upper), maxfev=5000)
            # Reject fits where full cycle is not visible in the data
            if not _full_cycle_visible(popt[2], popt[1], t_min, t_max, min_cycle_frac):
                continue
            residual = np.sum((y - _windowed_for_curvefit(t, *popt)) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_windowed_result(len(trs))

    amplitude, duration, onset, offset = best_result
    y_hat_valid = _windowed_for_curvefit(t, *best_result)
    ss_res = np.sum((y - y_hat_valid) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    y_hat_full = np.full(len(trs), np.nan)
    y_hat_full[valid] = y_hat_valid

    return {
        "amplitude": amplitude, "duration": duration,
        "onset": onset, "offset": offset,
        "frequency": 1.0 / duration, "r_squared": r_squared,
        "fit_success": True, "y_hat": y_hat_full,
    }


def _empty_windowed_result(n_trs: int) -> dict:
    return {
        "amplitude": np.nan, "duration": np.nan,
        "onset": np.nan, "offset": np.nan,
        "frequency": np.nan, "r_squared": np.nan,
        "fit_success": False, "y_hat": np.full(n_trs, np.nan),
    }


# =====================================================================
# 4. TRIAL-LEVEL COMPUTATION
# =====================================================================

def compute_trial_level_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Fit windowed single-cycle sinusoid to each trial's SODA timecourse."""
    soda_cols = {
        "retrieved": "retrieved_soda",
        "control": "control_soda",
        "shuf_retrieved": "retrieved_soda_shuffled",
        "shuf_control": "control_soda_shuffled",
        "shuf_sel_retrieved": "retrieved_soda_shuffled_selected",
        "shuf_sel_control": "control_soda_shuffled_selected",
    }

    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"] if c in df.columns]
    results = []

    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))
        trial_df = trial_df.sort_values("tr")
        trs = trial_df["tr"].values

        for cond_label, col_name in soda_cols.items():
            if col_name not in trial_df.columns:
                continue
            vals = trial_df[col_name].values

            w_fit = fit_windowed_sinusoid(trs, vals)

            trial_info[f"{cond_label}_w_amplitude"] = w_fit["amplitude"]
            trial_info[f"{cond_label}_w_duration"] = w_fit["duration"]
            trial_info[f"{cond_label}_w_onset"] = w_fit["onset"]
            trial_info[f"{cond_label}_w_frequency"] = w_fit["frequency"]
            trial_info[f"{cond_label}_w_r_squared"] = w_fit["r_squared"]
            trial_info[f"{cond_label}_w_fit_success"] = w_fit["fit_success"]

        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 5. PLOTTING
# =====================================================================

def plot_example_fits(df, metrics_df, soda_col="retrieved_soda", n_examples=8, save_path=None):
    """Plot example windowed sinusoidal fits sorted by R²."""
    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"] if c in metrics_df.columns]

    mask = metrics_df["retrieved_w_fit_success"].fillna(False)
    valid = metrics_df[mask].copy()
    if valid.empty:
        print("No trials with successful windowed fits.")
        return

    # Show best and worst fits
    best = valid.nlargest(n_examples // 2, "retrieved_w_r_squared")
    worst = valid.nsmallest(n_examples // 2, "retrieved_w_r_squared")
    examples = pd.concat([best, worst])

    n_plots = len(examples)
    if n_plots == 0:
        return

    ncols = min(4, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for i, (_, row) in enumerate(examples.iterrows()):
        ax = axes[i // ncols, i % ncols]

        mask_trial = pd.Series(True, index=df.index)
        for col in group_cols:
            if col in df.columns:
                mask_trial &= df[col] == row[col]
        trial_data = df[mask_trial].sort_values("tr")

        if trial_data.empty:
            ax.set_visible(False)
            continue

        trs = trial_data["tr"].values
        soda_vals = trial_data[soda_col].values
        t_fine = np.linspace(trs.min(), trs.max(), 200)

        ax.plot(trs, soda_vals, "ko", ms=5, zorder=5, label="Data")

        if row["retrieved_w_fit_success"]:
            w_fit = fit_windowed_sinusoid(trs, soda_vals)
            y_w = windowed_sinusoid(t_fine, w_fit["amplitude"], w_fit["duration"], w_fit["onset"], w_fit["offset"])
            ax.plot(t_fine, y_w, "r-", lw=2, alpha=0.8, label="Windowed")

        ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.set_title(f"R²={row['retrieved_w_r_squared']:.2f}", fontsize=8)
        ax.set_xlabel("TR", fontsize=8)
        ax.set_ylabel("SODA", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)

    for j in range(n_plots, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Windowed single-cycle sinusoidal fits (best & worst R²)", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_amplitude_summary(metrics_df, save_path=None):
    """Boxplot of windowed sinusoidal |amplitude| across all 6 conditions."""
    conditions = [
        "retrieved", "control",
        "shuf_retrieved", "shuf_control",
        "shuf_sel_retrieved", "shuf_sel_control",
    ]
    cond_labels = [
        "Retrieved", "Control",
        "Shuf Ret", "Shuf Ctrl",
        "Shuf Sel Ret", "Shuf Sel Ctrl",
    ]
    colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A", "#A0D0A0", "#D0A0D0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: R² distribution for retrieved
    ax = axes[0]
    r2_col = "retrieved_w_r_squared"
    if r2_col in metrics_df.columns:
        vals = metrics_df[r2_col].dropna()
        ax.hist(vals, bins=40, color="#3274A1", alpha=0.7, edgecolor="white")
        ax.set_xlabel("R² (windowed sinusoid)")
        ax.set_ylabel("Count")
        ax.set_title(f"Windowed fit quality (retrieved)\nmedian R²={vals.median():.3f}")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Amplitude boxplot across conditions
    ax = axes[1]
    amp_cols = [f"{c}_w_amplitude" for c in conditions]
    available = [c for c in amp_cols if c in metrics_df.columns]
    if len(available) >= 2 and "subject" in metrics_df.columns:
        subj = metrics_df.groupby("subject")[available].apply(lambda x: x.abs().mean()).dropna()
        if len(subj) >= 3:
            plot_data = [subj[c].values for c in available]
            plot_labels = [cond_labels[amp_cols.index(c)] for c in available]
            plot_colors = [colors[amp_cols.index(c)] for c in available]

            bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            rng = np.random.default_rng(42)
            for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
                jitter = rng.normal(0, 0.06, size=len(vals))
                ax.scatter(np.full_like(vals, i) + jitter, vals,
                           color=color, edgecolor="k", linewidth=0.5, s=25, zorder=5, alpha=0.8)

            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("|Amplitude| (windowed model)")
            ax.set_title("Windowed amplitude across conditions")

            ret_col, ctrl_col = "retrieved_w_amplitude", "control_w_amplitude"
            if ret_col in available and ctrl_col in available:
                t_stat, p_val = ttest_rel(subj[ret_col].values, subj[ctrl_col].values)
                sig_str = f"p={p_val:.1e}" if p_val < 0.001 else f"p={p_val:.3f}"
                print(f"  Ret vs Ctrl |amplitude|: t({len(subj[ret_col])-1})={t_stat:.3f}, {sig_str}")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_windowed_parameters(metrics_df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax_idx, (param, xlabel, title) in enumerate([
        ("duration", "Duration λ (TRs)", "Cycle duration"),
        ("onset", "Onset d (TRs)", "Retrieval onset"),
    ]):
        ax = axes[ax_idx]
        for cond, color, label in [("retrieved", "#3274A1", "Retrieved"), ("control", "#E1812C", "Control")]:
            col = f"{cond}_w_{param}"
            if col in metrics_df.columns:
                ax.hist(metrics_df[col].dropna(), bins=30, color=color, alpha=0.5, edgecolor="white", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    d_col, o_col = "retrieved_w_duration", "retrieved_w_onset"
    if d_col in metrics_df.columns and o_col in metrics_df.columns:
        v = metrics_df.dropna(subset=[d_col, o_col])
        scatter = ax.scatter(v[o_col], v[d_col],
                             c=v.get("behav_correct", pd.Series(np.nan)),
                             cmap="RdBu", s=10, alpha=0.4, edgecolors="none")
        ax.set_xlabel("Onset d (TRs)")
        ax.set_ylabel("Duration λ (TRs)")
        ax.set_title("Onset vs duration (retrieved)")
        if "behav_correct" in v.columns:
            plt.colorbar(scatter, ax=ax, label="Accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 6. ENTRY POINT
# =====================================================================

def main():
    roi_filter = sys.argv[1] if len(sys.argv) > 1 else None
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{pre_post}", "SODA_results_csv")

    csv_files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    all_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if "ROI" not in df.columns:
            roi_name = os.path.basename(fp).replace("_soda_results_per_combination.csv", "")
            df["ROI"] = roi_name
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    numeric_cols = [
        "tr", "retrieved_soda", "control_soda", "behav_correct",
        "retrieved_soda_shuffled", "control_soda_shuffled",
        "retrieved_soda_shuffled_selected", "control_soda_shuffled_selected",
    ]
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    if roi_filter:
        master_df = master_df[master_df["ROI"] == roi_filter]
        if master_df.empty:
            raise ValueError(f"ROI '{roi_filter}' not found.")

    rois = sorted(master_df["ROI"].unique())
    print(f"Found {len(rois)} ROIs: {rois}")

    all_roi_summaries = []

    for roi in rois:
        print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

        df = master_df[master_df["ROI"] == roi].copy()
        out_dir = opj(script_dir, "4.7_sinus_1cycle", roi)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Fitting windowed single-cycle sinusoid per trial...")

        metrics_df = compute_trial_level_fits(df)
        print(f"Computed fits for {len(metrics_df)} trials")

        metrics_df.to_csv(opj(out_dir, f"{roi}_windowed_fit_results_{pre_post}.csv"), index=False)

        for cond in ["retrieved", "control"]:
            r2_col = f"{cond}_w_r_squared"
            if r2_col in metrics_df.columns:
                vals = metrics_df[r2_col].dropna()
                print(f"  {cond}: median R²={vals.median():.3f}, mean R²={vals.mean():.3f}")

        print("Generating plots...")

        plot_example_fits(df, metrics_df, save_path=opj(out_dir, f"{roi}_example_fits_{pre_post}.png"))
        plot_amplitude_summary(metrics_df, save_path=opj(out_dir, f"{roi}_amplitude_summary_{pre_post}.png"))
        plot_windowed_parameters(metrics_df, save_path=opj(out_dir, f"{roi}_windowed_parameters_{pre_post}.png"))

        roi_summary = {"ROI": roi, "n_subjects": df["subject"].nunique(), "n_trials": len(metrics_df)}

        for cond_a, cond_b, label in [
            ("retrieved_w_amplitude", "control_w_amplitude", "ret_vs_ctrl"),
            ("retrieved_w_amplitude", "shuf_retrieved_w_amplitude", "ret_vs_shuf"),
            ("retrieved_w_amplitude", "shuf_sel_retrieved_w_amplitude", "ret_vs_shuf_sel"),
        ]:
            if cond_a in metrics_df.columns and cond_b in metrics_df.columns:
                subj = metrics_df.groupby("subject")[[cond_a, cond_b]].apply(lambda x: x.abs().mean()).dropna()
                if len(subj) >= 3:
                    t, p = ttest_rel(subj[cond_a], subj[cond_b])
                    print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj)})")
                    roi_summary[f"t_{label}"] = t
                    roi_summary[f"p_{label}"] = p

        all_roi_summaries.append(roi_summary)
        print(f"\nDone with ROI {roi}.")

    summary_dir = opj(script_dir, "4.7_sinus_1cycle", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = opj(summary_dir, f"windowed_fit_summary_{pre_post}.csv")
    summary_df = pd.DataFrame(all_roi_summaries)
    # Merge with existing summary to avoid overwriting other ROIs
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["ROI"].isin(summary_df["ROI"])]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to {summary_path}.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
