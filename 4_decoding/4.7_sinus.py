# -*- coding: utf-8 -*-
"""
Sinusoidal fitting of SODA (sequentiality) timecourses.

Rationale
---------
The SODA metric at each TR reflects sequential ordering of classifier
probabilities. During successful retrieval of a 5-item sequence, SODA
should trace a sinusoidal pattern: negative slope when early items peak
(early items high, late items low) transitioning to positive slope when
late items peak. Averaging raw SODA across trials cancels signal when
trials are phase-shifted. Instead, we fit a sinusoidal model per trial
and use the *amplitude* as a phase-invariant measure of sequential
retrieval strength.

Model
-----
For each trial's SODA timecourse y(t) over TRs t:

    y(t) = A · sin(2π·f·t + φ) + c

Parameters:
    A     : amplitude (retrieval strength; larger = stronger sequentiality)
    f     : frequency (retrieval speed; cycles per TR)
    φ     : phase (retrieval onset timing)
    c     : offset (baseline SODA)

We use scipy.optimize.curve_fit with sensible bounds and initial guesses
derived from the data.

Usage
-----
    python sinusoidal_soda_fit.py <ROI> [path_to_soda_csv]

If path_to_soda_csv is not given, it looks for the default output from
the SODA analysis script.
"""

import os
import sys
import glob
import ast
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, ttest_1samp, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt


def create_script_named_folder(folder_name=None):
    """
    Creates a new directory with the name of the script file (without extension) or with the provided folder name.
    
    Args:
        folder_name (str, optional): The name of the folder to be created inside the script directory. 
            If not provided, the folder will have the same name as the script file (without extension).
    
    Returns:
        str: The path of the newly created directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    script_name_without_ext = os.path.splitext(script_name)[0]
    
    if folder_name is None:
        new_dir_path = os.path.join(script_dir, script_name_without_ext)
    else:
        new_dir_path = os.path.join(script_dir, script_name_without_ext, folder_name)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    return new_dir_path


# =====================================================================
# 1. SINUSOIDAL MODEL AND FITTING
# =====================================================================

def sinusoidal_model(t, amplitude, frequency, phase, offset):
    """Sinusoidal model: A * sin(2π*f*t + φ) + c"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def fit_sinusoid_to_trial(
    trs: np.ndarray,
    soda_values: np.ndarray,
    f_min: float = 0.03,
    f_max: float = 0.45,
    n_restarts: int = 5,
) -> dict:
    """
    Fit a sinusoidal model to a single trial's SODA timecourse.

    Parameters
    ----------
    trs : array, shape (n_trs,)
        TR indices (e.g., 0, 1, 2, ..., 9).
    soda_values : array, shape (n_trs,)
        SODA slope values at each TR.
    f_min, f_max : float
        Bounds on frequency in cycles/TR.
        f_min=0.03 -> at most ~1 cycle over 30 TRs (very slow)
        f_max=0.45 -> at most ~1 cycle per 2.2 TRs (fast, near Nyquist)
    n_restarts : int
        Number of random restarts to avoid local minima.

    Returns
    -------
    dict with keys:
        amplitude, frequency, phase, offset : fitted parameters
        r_squared : variance explained
        fit_success : bool
        y_hat : fitted values
    """
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)

    valid = ~np.isnan(soda_values)
    if valid.sum() < 4:  # need > number of parameters
        return _empty_fit_result(len(trs))

    t = trs[valid]
    y = soda_values[valid]

    # Data-driven initial guesses
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    # Bounds: amplitude >= 0 (we allow curve_fit to find sign via phase)
    #         frequency in [f_min, f_max]
    #         phase in [-2π, 2π] (unconstrained enough for curve_fit)
    #         offset unconstrained
    lower = [0.0,    f_min, -2 * np.pi, -np.inf]
    upper = [np.inf, f_max,  2 * np.pi,  np.inf]

    best_result = None
    best_residual = np.inf

    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            # Data-driven initial guess
            p0 = [a0, 0.15, 0.0, y_mean]
        else:
            # Random restarts
            p0 = [
                rng.uniform(0, a0 * 2),
                rng.uniform(f_min, f_max),
                rng.uniform(-np.pi, np.pi),
                y_mean + rng.normal(0, a0 * 0.1),
            ]

        try:
            popt, pcov = curve_fit(
                sinusoidal_model, t, y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=5000,
            )
            y_hat_valid = sinusoidal_model(t, *popt)
            residual = np.sum((y - y_hat_valid) ** 2)

            if residual < best_residual:
                best_residual = residual
                best_result = popt

        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_fit_result(len(trs))

    amplitude, frequency, phase, offset = best_result

    # Compute R² on valid data
    y_hat_valid = sinusoidal_model(t, *best_result)
    ss_res = np.sum((y - y_hat_valid) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Full timecourse prediction (including NaN positions)
    y_hat_full = np.full(len(trs), np.nan)
    y_hat_full[valid] = y_hat_valid

    return {
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "offset": offset,
        "r_squared": r_squared,
        "fit_success": True,
        "y_hat": y_hat_full,
    }


def _empty_fit_result(n_trs):
    return {
        "amplitude": np.nan,
        "frequency": np.nan,
        "phase": np.nan,
        "offset": np.nan,
        "r_squared": np.nan,
        "fit_success": False,
        "y_hat": np.full(n_trs, np.nan),
    }


# =====================================================================
# 2. COMPLEMENTARY METRICS (phase-invariant)
# =====================================================================

def soda_variance(soda_values: np.ndarray) -> float:
    """Variance of SODA timecourse — higher if sinusoidal pattern present."""
    v = np.asarray(soda_values, dtype=float)
    v = v[~np.isnan(v)]
    return np.var(v) if len(v) >= 2 else np.nan


def soda_abs_mean(soda_values: np.ndarray) -> float:
    """Mean of absolute SODA — high if strong slopes regardless of sign."""
    v = np.asarray(soda_values, dtype=float)
    v = v[~np.isnan(v)]
    return np.mean(np.abs(v)) if len(v) >= 1 else np.nan


def soda_peak_to_trough(soda_values: np.ndarray) -> float:
    """Peak-to-trough range of SODA timecourse."""
    v = np.asarray(soda_values, dtype=float)
    v = v[~np.isnan(v)]
    return np.ptp(v) if len(v) >= 2 else np.nan


def soda_spectral_power(
    soda_values: np.ndarray,
    f_min: float = 0.05,
    f_max: float = 0.45,
) -> float:
    """
    Power in the expected frequency band via FFT.
    Following Wittkuhn & Schuck (2021): power spectral density of the
    sequentiality metric distinguishes sub- vs supra-second sequences.
    """
    v = np.asarray(soda_values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 4:
        return np.nan

    v = v - np.mean(v)  # remove DC
    freqs = np.fft.rfftfreq(len(v), d=1.0)  # in cycles/TR
    power = np.abs(np.fft.rfft(v)) ** 2

    band = (freqs >= f_min) & (freqs <= f_max)
    return np.mean(power[band]) if band.any() else np.nan


# =====================================================================
# 3. MAIN ANALYSIS
# =====================================================================


def compute_trial_level_metrics(
    df: pd.DataFrame,
    soda_col: str = "retrieved_soda",
    control_col: str = "control_soda",
    shuffled_ret_col: str = "retrieved_soda_shuffled",
    shuffled_ctrl_col: str = "control_soda_shuffled",
) -> pd.DataFrame:
    """
    For each unique trial, compute sinusoidal fit and complementary metrics.
    """
    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    # Check which group cols exist
    group_cols = [c for c in group_cols if c in df.columns]

    results = []

    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))

        trial_df = trial_df.sort_values("tr")
        trs = trial_df["tr"].values
        soda_vals = trial_df[soda_col].values

        # Also get control SODA for comparison
        control_vals = trial_df[control_col].values if control_col in trial_df.columns else None

        # --- Sinusoidal fit ---
        fit = fit_sinusoid_to_trial(trs, soda_vals)

        trial_info["sin_amplitude"] = fit["amplitude"]
        trial_info["sin_frequency"] = fit["frequency"]
        trial_info["sin_phase"] = fit["phase"]
        trial_info["sin_offset"] = fit["offset"]
        trial_info["sin_r_squared"] = fit["r_squared"]
        trial_info["sin_fit_success"] = fit["fit_success"]

        # --- Complementary metrics ---
        trial_info["soda_variance"] = soda_variance(soda_vals)
        trial_info["soda_abs_mean"] = soda_abs_mean(soda_vals)
        trial_info["soda_peak_to_trough"] = soda_peak_to_trough(soda_vals)
        trial_info["soda_spectral_power"] = soda_spectral_power(soda_vals)

        # --- Same for control ---
        if control_vals is not None:
            fit_ctrl = fit_sinusoid_to_trial(trs, control_vals)
            trial_info["ctrl_sin_amplitude"] = fit_ctrl["amplitude"]
            trial_info["ctrl_sin_r_squared"] = fit_ctrl["r_squared"]
            trial_info["ctrl_soda_variance"] = soda_variance(control_vals)
            trial_info["ctrl_soda_abs_mean"] = soda_abs_mean(control_vals)
            trial_info["ctrl_soda_spectral_power"] = soda_spectral_power(control_vals)

        # --- Shuffled retrieved ---
        shuf_ret_vals = trial_df[shuffled_ret_col].values if shuffled_ret_col in trial_df.columns else None
        if shuf_ret_vals is not None:
            fit_shuf_ret = fit_sinusoid_to_trial(trs, shuf_ret_vals)
            trial_info["shuf_ret_sin_amplitude"] = fit_shuf_ret["amplitude"]
            trial_info["shuf_ret_sin_r_squared"] = fit_shuf_ret["r_squared"]
            trial_info["shuf_ret_soda_variance"] = soda_variance(shuf_ret_vals)
            trial_info["shuf_ret_soda_abs_mean"] = soda_abs_mean(shuf_ret_vals)
            trial_info["shuf_ret_soda_spectral_power"] = soda_spectral_power(shuf_ret_vals)

        # --- Shuffled control ---
        shuf_ctrl_vals = trial_df[shuffled_ctrl_col].values if shuffled_ctrl_col in trial_df.columns else None
        if shuf_ctrl_vals is not None:
            fit_shuf_ctrl = fit_sinusoid_to_trial(trs, shuf_ctrl_vals)
            trial_info["shuf_ctrl_sin_amplitude"] = fit_shuf_ctrl["amplitude"]
            trial_info["shuf_ctrl_sin_r_squared"] = fit_shuf_ctrl["r_squared"]
            trial_info["shuf_ctrl_soda_variance"] = soda_variance(shuf_ctrl_vals)
            trial_info["shuf_ctrl_soda_abs_mean"] = soda_abs_mean(shuf_ctrl_vals)
            trial_info["shuf_ctrl_soda_spectral_power"] = soda_spectral_power(shuf_ctrl_vals)

        # --- Behavior ---
        if "behav_correct" in trial_df.columns:
            trial_info["behav_correct"] = trial_df["behav_correct"].dropna().iloc[0] if trial_df["behav_correct"].dropna().shape[0] > 0 else np.nan

        # --- Retrieval position ---
        if "retrieved_position" in trial_df.columns:
            trial_info["retrieved_position"] = trial_df["retrieved_position"].dropna().iloc[0] if trial_df["retrieved_position"].dropna().shape[0] > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 4. PLOTTING
# =====================================================================

def plot_amplitude_by_accuracy(
    metrics_df: pd.DataFrame,
    metric_col: str = "sin_amplitude",
    ctrl_col: str = "ctrl_sin_amplitude",
    title_suffix: str = "",
    save_path: str = None,
):
    """
    Compare sinusoidal amplitude between correct vs incorrect trials.
    Also compare retrieved vs control within each accuracy bin.
    """
    df = metrics_df.dropna(subset=[metric_col, "behav_correct"]).copy()
    if df.empty:
        print(f"No data for {metric_col} vs behav_correct")
        return

    # Subject-level averages
    subj = df.groupby(["subject", "behav_correct"]).agg(
        retrieved_amp=(metric_col, "mean"),
        control_amp=(ctrl_col, "mean") if ctrl_col in df.columns else (metric_col, "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Retrieved amplitude by accuracy
    ax = axes[0]
    for acc_val, color, label in [(1, "#2166ac", "Correct"), (0, "#b2182b", "Incorrect")]:
        vals = subj[subj["behav_correct"] == acc_val]["retrieved_amp"]
        if len(vals) > 0:
            ax.bar(label, vals.mean(), yerr=vals.sem(), color=color, alpha=0.7, capsize=5)
            # Individual subjects
            ax.scatter(
                [label] * len(vals), vals,
                color=color, alpha=0.4, zorder=3, s=20,
            )
    ax.set_ylabel(f"Mean {metric_col}")
    ax.set_title(f"Retrieved: {metric_col} by accuracy")

    # Panel 2: Retrieved vs Control (correct trials only)
    ax = axes[1]
    correct_subj = subj[subj["behav_correct"] == 1]
    if len(correct_subj) > 1:
        for i, (col, color, label) in enumerate([
            ("retrieved_amp", "#2166ac", "Retrieved"),
            ("control_amp", "#636363", "Control"),
        ]):
            vals = correct_subj[col]
            ax.bar(label, vals.mean(), yerr=vals.sem(), color=color, alpha=0.7, capsize=5)
            ax.scatter([label] * len(vals), vals, color=color, alpha=0.4, zorder=3, s=20)

        # Paired t-test
        if len(correct_subj) >= 3:
            t, p = ttest_rel(correct_subj["retrieved_amp"], correct_subj["control_amp"])
            ax.set_title(f"Correct trials: Retrieved vs Control\nt={t:.2f}, p={p:.3f}")
        else:
            ax.set_title("Correct trials: Retrieved vs Control")
    ax.set_ylabel(f"Mean amplitude")

    fig.suptitle(f"Sinusoidal amplitude analysis {title_suffix}", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_example_trial_fits(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    soda_col: str = "retrieved_soda",
    n_examples: int = 8,
    save_path: str = None,
):
    """
    Show example sinusoidal fits: top N by R² (good fits) and bottom N (poor fits).
    """
    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in metrics_df.columns]

    good = metrics_df[metrics_df["sin_fit_success"]].nlargest(n_examples, "sin_r_squared")
    poor = metrics_df[metrics_df["sin_fit_success"]].nsmallest(n_examples, "sin_r_squared")
    examples = pd.concat([good, poor])

    n_plots = len(examples)
    if n_plots == 0:
        print("No successful fits to plot.")
        return

    ncols = min(4, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, (_, row) in enumerate(examples.iterrows()):
        ax = axes[i // ncols, i % ncols]

        # Get trial data
        mask = pd.Series(True, index=df.index)
        for col in group_cols:
            if col in df.columns:
                mask &= df[col] == row[col]
        trial_data = df[mask].sort_values("tr")

        if trial_data.empty:
            ax.set_visible(False)
            continue

        trs = trial_data["tr"].values
        soda_vals = trial_data[soda_col].values

        ax.plot(trs, soda_vals, "ko-", ms=4, lw=1.5, label="Data")

        # Overlay fit
        if row["sin_fit_success"]:
            t_fine = np.linspace(trs.min(), trs.max(), 100)
            y_fit = sinusoidal_model(
                t_fine, row["sin_amplitude"], row["sin_frequency"],
                row["sin_phase"], row["sin_offset"],
            )
            ax.plot(t_fine, y_fit, "r-", lw=2, alpha=0.8, label="Fit")

        ax.axhline(0, color="grey", ls="--", lw=0.8)
        behav = row.get("behav_correct", "?")
        ax.set_title(
            f"R²={row['sin_r_squared']:.2f}, A={row['sin_amplitude']:.3f}\n"
            f"f={row['sin_frequency']:.2f}, correct={behav}",
            fontsize=9,
        )
        ax.set_xlabel("TR")
        ax.set_ylabel("SODA")

    # Hide unused axes
    for j in range(n_plots, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Example sinusoidal fits (top: best R², bottom: worst R²)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_summary(
    metrics_df: pd.DataFrame,
    save_path: str = None,
):
    """
    Summary panel: distributions of amplitude, R², frequency;
    retrieved vs control comparisons.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: Distributions
    for ax, col, label in zip(
        axes[0],
        ["sin_amplitude", "sin_r_squared", "sin_frequency"],
        ["Amplitude", "R²", "Frequency (cycles/TR)"],
    ):
        vals = metrics_df[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=30, color="#2166ac", alpha=0.7, edgecolor="white")
            ax.axvline(vals.mean(), color="red", ls="--", lw=2)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{label}\nmean={vals.mean():.4f}, median={vals.median():.4f}")

    # Row 2: 4-condition comparisons (subject-level), mirroring boxplot layout
    metric_quads = [
        ("sin_amplitude", "ctrl_sin_amplitude", "shuf_ret_sin_amplitude", "shuf_ctrl_sin_amplitude", "Amplitude"),
        ("soda_variance", "ctrl_soda_variance", "shuf_ret_soda_variance", "shuf_ctrl_soda_variance", "Variance"),
        ("soda_spectral_power", "ctrl_soda_spectral_power", "shuf_ret_soda_spectral_power", "shuf_ctrl_soda_spectral_power", "Spectral power"),
    ]
    bar_labels = ["Retrieved", "Control", "Ret (shuf)", "Ctrl (shuf)"]
    bar_colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A"]

    for ax, (ret_col, ctrl_col, shuf_ret_col, shuf_ctrl_col, label) in zip(axes[1], metric_quads):
        cols = [ret_col, ctrl_col, shuf_ret_col, shuf_ctrl_col]
        available = [c for c in cols if c in metrics_df.columns]
        if len(available) < 2:
            ax.set_visible(False)
            continue

        subj = metrics_df.groupby("subject")[available].mean().dropna()
        if len(subj) < 3:
            ax.text(0.5, 0.5, "Too few subjects", ha="center", va="center", transform=ax.transAxes)
            continue

        # Bar plot for all available conditions
        positions = list(range(len(available)))
        means = [subj[c].mean() for c in available]
        sems = [subj[c].sem() for c in available]
        col_labels = [bar_labels[cols.index(c)] for c in available]
        col_colors = [bar_colors[cols.index(c)] for c in available]

        ax.bar(positions, means, yerr=sems, color=col_colors, alpha=0.7, capsize=5)

        # Individual subjects as jittered dots
        rng = np.random.default_rng(42)
        for i, c in enumerate(available):
            vals = subj[c].values
            jitter = rng.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color=col_colors[i], edgecolor='k', linewidth=0.5,
                       s=20, zorder=5, alpha=0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_ylabel(label)

        # Paired t-test: retrieved vs control
        if ret_col in available and ctrl_col in available:
            t, p = ttest_rel(subj[ret_col], subj[ctrl_col])
            ax.set_title(f"{label}\nRet vs Ctrl: t={t:.2f}, p={p:.3f}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_brain_behavior_correlation(
    metrics_df: pd.DataFrame,
    metric_col: str = "sin_amplitude",
    ctrl_col: str = "ctrl_sin_amplitude",
    save_path: str = None,
):
    """
    Correlate subject-level sinusoidal amplitude with behavioral accuracy.
    Use the *difference* (retrieved - control) as a cleaner neural measure.
    """
    df = metrics_df.dropna(subset=[metric_col, "behav_correct"]).copy()
    if df.empty:
        return

    # Subject-level: mean amplitude and mean accuracy
    subj = df.groupby("subject").agg(
        mean_amp=(metric_col, "mean"),
        mean_ctrl_amp=(ctrl_col, "mean") if ctrl_col in df.columns else (metric_col, "mean"),
        mean_accuracy=("behav_correct", "mean"),
        n_trials=("behav_correct", "count"),
    ).reset_index()

    subj["amp_delta"] = subj["mean_amp"] - subj["mean_ctrl_amp"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, xcol, xlabel in zip(
        axes,
        ["mean_amp", "amp_delta"],
        [f"Mean {metric_col}", f"Mean {metric_col} - control"],
    ):
        x = subj[xcol].values
        y = subj["mean_accuracy"].values

        ax.scatter(x, y, s=60, alpha=0.7, color="#2166ac")

        if len(x) >= 5:
            r, p = pearsonr(x, y)
            # Regression line
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, "r--", lw=2)
            ax.set_title(f"r={r:.3f}, p={p:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean behavioral accuracy")

    fig.suptitle("Brain-behavior correlation: sinusoidal amplitude vs accuracy", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# 5. ENTRY POINT
# =====================================================================

def main():
    # ── Configuration ──
    Pre_post = 'postsleep'  # 'postsleep' or 'presleep'

    # ── Paths (mirroring 4.7) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir,
                  f'4.6_SODA_based_on_singlesubj_4.4_{Pre_post}', 'SODA_results_csv')

    csv_files = sorted(glob.glob(opj(csv_dir, '*_soda_results_per_combination.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    # ── Load all ROI CSVs ──
    all_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if 'ROI' not in df.columns:
            roi_name = os.path.basename(fp).replace('_soda_results_per_combination.csv', '')
            df['ROI'] = roi_name
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure numeric
    for col in ["tr", "retrieved_soda", "control_soda", "behav_correct"]:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    rois = sorted(master_df['ROI'].unique())
    print(f"Found {len(rois)} ROIs: {rois}")
    print(f"Total rows: {len(master_df)}")

    # ── Loop over ROIs ──
    all_roi_summaries = []

    for roi in rois:
        print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

        df = master_df[master_df['ROI'] == roi].copy()

        # --- Output directory ---
        out_dir = create_script_named_folder(roi)
        print(f"Output directory: {out_dir}")

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

        # --- Compute trial-level metrics ---
        print("Fitting sinusoidal models per trial...")
        metrics_df = compute_trial_level_metrics(df)
        print(f"Computed metrics for {len(metrics_df)} trials")

        n_success = metrics_df["sin_fit_success"].sum()
        n_total = len(metrics_df)
        print(f"Successful fits: {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")

        # Summary stats
        successful = metrics_df[metrics_df["sin_fit_success"]]
        if len(successful) > 0:
            print(f"Amplitude:  mean={successful['sin_amplitude'].mean():.4f}, "
                  f"median={successful['sin_amplitude'].median():.4f}")
            print(f"R²:         mean={successful['sin_r_squared'].mean():.4f}, "
                  f"median={successful['sin_r_squared'].median():.4f}")
            print(f"Frequency:  mean={successful['sin_frequency'].mean():.4f}, "
                  f"median={successful['sin_frequency'].median():.4f}")

        # --- Save trial-level results ---
        out_csv = opj(out_dir, f"{roi}_sinusoidal_fit_results.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"Saved trial-level metrics to: {out_csv}")

        # --- Plots ---
        print("Generating plots...")

        plot_example_trial_fits(
            df, metrics_df,
            save_path=opj(out_dir, f"{roi}_example_sinusoidal_fits.png"),
        )

        plot_metrics_summary(
            metrics_df,
            save_path=opj(out_dir, f"{roi}_sinusoidal_metrics_summary.png"),
        )

        plot_amplitude_by_accuracy(
            metrics_df,
            title_suffix=f"(ROI={roi})",
            save_path=opj(out_dir, f"{roi}_amplitude_by_accuracy.png"),
        )

        plot_brain_behavior_correlation(
            metrics_df,
            save_path=opj(out_dir, f"{roi}_brain_behavior_correlation.png"),
        )

        # --- Subject-level 1-sample t-test: amplitude > 0 ---
        subj_amp = metrics_df.groupby("subject")["sin_amplitude"].mean().dropna()
        if len(subj_amp) >= 3:
            t, p = ttest_1samp(subj_amp, 0)
            print(f"\nSubject-level amplitude > 0: t={t:.3f}, p={p:.4f} (n={len(subj_amp)})")

        # --- Subject-level paired t-test: retrieved vs control amplitude ---
        subj_paired = metrics_df.groupby("subject")[["sin_amplitude", "ctrl_sin_amplitude"]].mean().dropna()
        if len(subj_paired) >= 3:
            t, p = ttest_rel(subj_paired["sin_amplitude"], subj_paired["ctrl_sin_amplitude"])
            print(f"Retrieved vs control amplitude: t={t:.3f}, p={p:.4f} (n={len(subj_paired)})")

        # --- Subject-level paired t-tests: retrieved vs shuffled controls ---
        shuf_amp_cols = ["sin_amplitude", "shuf_ret_sin_amplitude", "ctrl_sin_amplitude", "shuf_ctrl_sin_amplitude"]
        shuf_amp_available = [c for c in shuf_amp_cols if c in metrics_df.columns]
        subj_shuf = metrics_df.groupby("subject")[shuf_amp_available].mean().dropna()

        if len(subj_shuf) >= 3 and "shuf_ret_sin_amplitude" in subj_shuf.columns:
            t, p = ttest_rel(subj_shuf["sin_amplitude"], subj_shuf["shuf_ret_sin_amplitude"])
            print(f"Retrieved vs retrieved(shuf) amplitude: t={t:.3f}, p={p:.4f} (n={len(subj_shuf)})")

        if len(subj_shuf) >= 3 and "shuf_ctrl_sin_amplitude" in subj_shuf.columns and "ctrl_sin_amplitude" in subj_shuf.columns:
            t, p = ttest_rel(subj_shuf["ctrl_sin_amplitude"], subj_shuf["shuf_ctrl_sin_amplitude"])
            print(f"Control vs control(shuf) amplitude: t={t:.3f}, p={p:.4f} (n={len(subj_shuf)})")

        # Collect summary for this ROI
        roi_summary = {"ROI": roi, "n_subjects": df["subject"].nunique(), "n_trials": n_total, "n_fits_success": int(n_success)}
        if len(subj_amp) >= 3:
            t_1s, p_1s = ttest_1samp(subj_amp, 0)
            roi_summary["t_amp_gt0"] = t_1s
            roi_summary["p_amp_gt0"] = p_1s
        if len(subj_paired) >= 3:
            t_paired, p_paired = ttest_rel(subj_paired["sin_amplitude"], subj_paired["ctrl_sin_amplitude"])
            roi_summary["t_ret_vs_ctrl"] = t_paired
            roi_summary["p_ret_vs_ctrl"] = p_paired
        if len(subj_shuf) >= 3 and "shuf_ret_sin_amplitude" in subj_shuf.columns:
            t_rs, p_rs = ttest_rel(subj_shuf["sin_amplitude"], subj_shuf["shuf_ret_sin_amplitude"])
            roi_summary["t_ret_vs_shuf"] = t_rs
            roi_summary["p_ret_vs_shuf"] = p_rs
        if len(subj_shuf) >= 3 and "shuf_ctrl_sin_amplitude" in subj_shuf.columns and "ctrl_sin_amplitude" in subj_shuf.columns:
            t_cs, p_cs = ttest_rel(subj_shuf["ctrl_sin_amplitude"], subj_shuf["shuf_ctrl_sin_amplitude"])
            roi_summary["t_ctrl_vs_shuf"] = t_cs
            roi_summary["p_ctrl_vs_shuf"] = p_cs
        all_roi_summaries.append(roi_summary)

        print(f"\nDone with ROI {roi}.")

    # ── Save cross-ROI summary ──
    summary_dir = create_script_named_folder('summary')
    summary_df = pd.DataFrame(all_roi_summaries)
    summary_path = opj(summary_dir, 'sinusoidal_fit_summary_all_rois.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()