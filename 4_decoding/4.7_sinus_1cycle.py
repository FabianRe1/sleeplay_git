# -*- coding: utf-8 -*-
"""
Windowed single-cycle sinusoidal fitting of SODA timecourses.

Extends sinusoidal_soda_fit.py with a windowed model following the logic
of Wittkuhn & Schuck (2021, Nature Communications).

The key insight: during retrieval you expect ONE cycle of a sinusoidal
SODA pattern (forward → backward sequentiality), not a continuous
oscillation. The signal is flat (zero) before retrieval onset and after
the retrieval sweep completes.

Model
-----
Continuous sinusoid (original):
    y(t) = A * sin(2π*f*t + φ) + c

Windowed single-cycle (new, following Wittkuhn & Schuck 2021, Eq. 4-5):
    y(t) = A * sin(2π/λ * (t - d)) * W(t; d, λ) + c

    where:
      A     = amplitude (retrieval strength)
      λ     = duration of the single cycle (in TRs)
      d     = onset delay (when retrieval starts, in TRs)
      c     = baseline offset
      W     = window function: 1 inside [d, d+λ], 0 outside
              (with optional smooth edges via raised-cosine taper)

    The frequency of the single cycle is f = 1/λ.

Comparison with continuous sinusoid:
    - Continuous: assumes retrieval oscillates across the entire window
    - Windowed:   assumes exactly one up-down cycle, flat elsewhere
    - Windowed is more realistic but has similar number of parameters (4)
    - Both can be fit and compared via R² or AIC/BIC

Usage
-----
    This script can be used standalone or its functions can be imported
    into sinusoidal_soda_fit.py or fft_soda_analysis.py.

    python windowed_soda_fit.py
"""

import os
import sys
import glob
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import ttest_rel, ttest_1samp, pearsonr
from scipy.signal import periodogram
from matplotlib import pyplot as plt


def create_script_named_folder(folder_name=None):
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
# 1. WINDOWED SINGLE-CYCLE SINUSOIDAL MODEL
# =====================================================================

def windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
    """
    Single-cycle sinusoidal model with smooth on/off ramps.

    y(t) = A * sin(2π/λ * (t - d)) * W(t; d, λ, taper) + c

    Parameters
    ----------
    t : array
        Time points (TR indices).
    amplitude : float
        Peak amplitude of the sinusoidal cycle.
    duration : float
        Duration λ of the single cycle (in TRs). The frequency is 1/λ.
    onset : float
        Onset delay d (in TRs). The cycle begins at t=d.
    offset : float
        Baseline offset c.
    taper_frac : float
        Fraction of the window duration used for raised-cosine tapering
        at the edges (0 = hard rectangular, 0.5 = full Tukey window).
        Default 0.15 provides gentle edges to avoid discontinuities.

    Returns
    -------
    y : array, same shape as t
    """
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)

    # Sinusoidal part: one full cycle over duration λ
    phase = 2 * np.pi / duration * (t - onset)
    sine_part = amplitude * np.sin(phase)

    # Window: active in [onset, onset + duration]
    active = (t >= onset) & (t <= onset + duration)

    # Taper edges with raised cosine (Tukey-style)
    taper_len = taper_frac * duration
    window = np.zeros_like(t)
    window[active] = 1.0

    if taper_len > 0:
        # Rising edge: onset to onset + taper_len
        rise = active & (t < onset + taper_len)
        if rise.any():
            window[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - onset) / taper_len))

        # Falling edge: onset + duration - taper_len to onset + duration
        fall = active & (t > onset + duration - taper_len)
        if fall.any():
            window[fall] = 0.5 * (1 - np.cos(np.pi * (onset + duration - t[fall]) / taper_len))

    y = sine_part * window + offset
    return y


def _windowed_sinusoid_for_curvefit(t, amplitude, duration, onset, offset):
    """Wrapper for curve_fit (fixed taper_frac)."""
    return windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15)


def fit_windowed_sinusoid(
    trs: np.ndarray,
    soda_values: np.ndarray,
    duration_min: float = 3.0,
    duration_max: float = 12.0,
    n_restarts: int = 10,
) -> dict:
    """
    Fit the windowed single-cycle sinusoidal model to a trial's SODA timecourse.

    Parameters
    ----------
    trs : array, shape (n_trs,)
        TR indices (e.g., 0, 1, 2, ..., 9).
    soda_values : array, shape (n_trs,)
        SODA slope values at each TR.
    duration_min, duration_max : float
        Bounds on cycle duration λ (in TRs).
        duration_min=3 → fastest: 1 cycle over 3 TRs
        duration_max=12 → slowest: 1 cycle over 12 TRs
    n_restarts : int
        Number of random restarts.

    Returns
    -------
    dict with keys:
        amplitude, duration, onset, offset : fitted parameters
        frequency : 1/duration (derived)
        r_squared : variance explained
        fit_success : bool
        y_hat : fitted values
    """
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)

    valid = ~np.isnan(soda_values)
    if valid.sum() < 4:
        return _empty_windowed_result(len(trs))

    t = trs[valid]
    y = soda_values[valid]

    t_min, t_max = t.min(), t.max()
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    # Bounds: [amplitude, duration, onset, offset]
    # Amplitude can be negative (sign encodes direction)
    lower = [-np.inf, duration_min, t_min - 1.0, -np.inf]
    upper = [np.inf,  duration_max, t_max,        np.inf]

    best_result = None
    best_residual = np.inf
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
                rng.uniform(t_min - 0.5, t_min + (t_max - t_min) * 0.3),
                y_mean + rng.normal(0, a0 * 0.1),
            ]

        try:
            popt, _ = curve_fit(
                _windowed_sinusoid_for_curvefit, t, y,
                p0=p0, bounds=(lower, upper), maxfev=5000,
            )
            y_hat_valid = _windowed_sinusoid_for_curvefit(t, *popt)
            residual = np.sum((y - y_hat_valid) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_windowed_result(len(trs))

    amplitude, duration, onset, offset = best_result

    y_hat_valid = _windowed_sinusoid_for_curvefit(t, *best_result)
    ss_res = np.sum((y - y_hat_valid) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    y_hat_full = np.full(len(trs), np.nan)
    y_hat_full[valid] = y_hat_valid

    return {
        "amplitude": amplitude,
        "duration": duration,
        "onset": onset,
        "offset": offset,
        "frequency": 1.0 / duration,
        "r_squared": r_squared,
        "fit_success": True,
        "y_hat": y_hat_full,
    }


def _empty_windowed_result(n_trs):
    return {
        "amplitude": np.nan,
        "duration": np.nan,
        "onset": np.nan,
        "offset": np.nan,
        "frequency": np.nan,
        "r_squared": np.nan,
        "fit_success": False,
        "y_hat": np.full(n_trs, np.nan),
    }


# =====================================================================
# 2. CONTINUOUS SINUSOIDAL MODEL (from sinusoidal_soda_fit.py)
# =====================================================================

def continuous_sinusoid(t, amplitude, frequency, phase, offset):
    """Standard continuous sinusoidal model: A * sin(2π*f*t + φ) + c"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def fit_continuous_sinusoid(
    trs: np.ndarray,
    soda_values: np.ndarray,
    f_min: float = 0.03,
    f_max: float = 0.45,
    n_restarts: int = 5,
) -> dict:
    """Fit continuous sinusoidal model (from sinusoidal_soda_fit.py)."""
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)

    valid = ~np.isnan(soda_values)
    if valid.sum() < 4:
        return _empty_continuous_result(len(trs))

    t = trs[valid]
    y = soda_values[valid]

    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    lower = [0.0, f_min, -2 * np.pi, -np.inf]
    upper = [np.inf, f_max, 2 * np.pi, np.inf]

    best_result = None
    best_residual = np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, 0.15, 0.0, y_mean]
        else:
            p0 = [
                rng.uniform(0, a0 * 2),
                rng.uniform(f_min, f_max),
                rng.uniform(-np.pi, np.pi),
                y_mean + rng.normal(0, a0 * 0.1),
            ]

        try:
            popt, _ = curve_fit(
                continuous_sinusoid, t, y,
                p0=p0, bounds=(lower, upper), maxfev=5000,
            )
            y_hat_valid = continuous_sinusoid(t, *popt)
            residual = np.sum((y - y_hat_valid) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_continuous_result(len(trs))

    amplitude, frequency, phase, offset = best_result
    y_hat_valid = continuous_sinusoid(t, *best_result)
    ss_res = np.sum((y - y_hat_valid) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

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


def _empty_continuous_result(n_trs):
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
# 3. MODEL COMPARISON (BIC)
# =====================================================================

def bic(n, k, ss_res):
    """Bayesian Information Criterion.
    n = number of data points, k = number of parameters,
    ss_res = residual sum of squares."""
    if ss_res <= 0 or n <= k:
        return np.inf
    return n * np.log(ss_res / n) + k * np.log(n)


def compare_models(trs, soda_values, windowed_fit, continuous_fit):
    """
    Compare windowed vs continuous model using BIC.

    Returns
    -------
    dict with delta_bic (negative = windowed better), preferred model name.
    """
    valid = ~np.isnan(soda_values)
    n = valid.sum()

    k_windowed = 4    # amplitude, duration, onset, offset
    k_continuous = 4  # amplitude, frequency, phase, offset

    y = soda_values[valid]

    if windowed_fit["fit_success"] and continuous_fit["fit_success"]:
        yh_w = windowed_fit["y_hat"][valid]
        yh_c = continuous_fit["y_hat"][valid]

        ss_w = np.sum((y - yh_w) ** 2)
        ss_c = np.sum((y - yh_c) ** 2)

        bic_w = bic(n, k_windowed, ss_w)
        bic_c = bic(n, k_continuous, ss_c)

        delta_bic = bic_w - bic_c  # negative → windowed better

        return {
            "bic_windowed": bic_w,
            "bic_continuous": bic_c,
            "delta_bic": delta_bic,
            "preferred": "windowed" if delta_bic < 0 else "continuous",
        }

    return {"bic_windowed": np.nan, "bic_continuous": np.nan,
            "delta_bic": np.nan, "preferred": "neither"}


# =====================================================================
# 4. TRIAL-LEVEL COMPUTATION
# =====================================================================

def compute_trial_level_fits(
    df: pd.DataFrame,
    soda_cols: dict = None,
) -> pd.DataFrame:
    """
    For each trial, fit both windowed and continuous sinusoidal models
    and compute model comparison metrics.
    """
    if soda_cols is None:
        soda_cols = {
            "retrieved": "retrieved_soda",
            "control": "control_soda",
            "shuf_retrieved": "retrieved_soda_shuffled",
            "shuf_control": "control_soda_shuffled",
        }

    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in df.columns]

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

            # Fit both models
            w_fit = fit_windowed_sinusoid(trs, vals)
            c_fit = fit_continuous_sinusoid(trs, vals)

            # Model comparison
            comp = compare_models(trs, vals, w_fit, c_fit)

            prefix = cond_label

            # Windowed model results
            trial_info[f"{prefix}_w_amplitude"] = w_fit["amplitude"]
            trial_info[f"{prefix}_w_duration"] = w_fit["duration"]
            trial_info[f"{prefix}_w_onset"] = w_fit["onset"]
            trial_info[f"{prefix}_w_frequency"] = w_fit["frequency"]
            trial_info[f"{prefix}_w_r_squared"] = w_fit["r_squared"]
            trial_info[f"{prefix}_w_fit_success"] = w_fit["fit_success"]

            # Continuous model results
            trial_info[f"{prefix}_c_amplitude"] = c_fit["amplitude"]
            trial_info[f"{prefix}_c_frequency"] = c_fit["frequency"]
            trial_info[f"{prefix}_c_r_squared"] = c_fit["r_squared"]
            trial_info[f"{prefix}_c_fit_success"] = c_fit["fit_success"]

            # Model comparison
            trial_info[f"{prefix}_delta_bic"] = comp["delta_bic"]
            trial_info[f"{prefix}_preferred"] = comp["preferred"]

        # Behavior
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

def plot_example_fits_comparison(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    soda_col: str = "retrieved_soda",
    n_examples: int = 8,
    save_path: str = None,
):
    """
    Show example trials with both windowed and continuous fits overlaid.
    Select trials where the two models disagree most (largest |delta_bic|).
    """
    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in metrics_df.columns]

    # Get trials with successful fits for both models
    mask = (
        metrics_df["retrieved_w_fit_success"].fillna(False) &
        metrics_df["retrieved_c_fit_success"].fillna(False)
    )
    valid = metrics_df[mask].copy()

    if valid.empty:
        print("No trials with both successful fits.")
        return

    # Top N where windowed wins, top N where continuous wins
    windowed_wins = valid.nsmallest(n_examples // 2, "retrieved_delta_bic")
    continuous_wins = valid.nlargest(n_examples // 2, "retrieved_delta_bic")
    examples = pd.concat([windowed_wins, continuous_wins])

    n_plots = len(examples)
    if n_plots == 0:
        return

    ncols = min(4, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for i, (_, row) in enumerate(examples.iterrows()):
        ax = axes[i // ncols, i % ncols]

        # Get trial data
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

        # Data
        ax.plot(trs, soda_vals, "ko", ms=5, zorder=5, label="Data")

        # Windowed fit
        if row["retrieved_w_fit_success"]:
            y_w = windowed_sinusoid(
                t_fine, row["retrieved_w_amplitude"],
                row["retrieved_w_duration"], row["retrieved_w_onset"],
                row.get("retrieved_w_offset", 0),  # offset might be named differently
            )
            # We don't store offset separately — refit quickly
            w_fit = fit_windowed_sinusoid(trs, soda_vals)
            y_w = windowed_sinusoid(t_fine, w_fit["amplitude"], w_fit["duration"],
                                     w_fit["onset"], w_fit["offset"])
            ax.plot(t_fine, y_w, "r-", lw=2, alpha=0.8, label="Windowed")

        # Continuous fit
        if row["retrieved_c_fit_success"]:
            c_fit = fit_continuous_sinusoid(trs, soda_vals)
            y_c = continuous_sinusoid(t_fine, c_fit["amplitude"], c_fit["frequency"],
                                      c_fit["phase"], c_fit["offset"])
            ax.plot(t_fine, y_c, "b--", lw=1.5, alpha=0.7, label="Continuous")

        ax.axhline(0, color="grey", ls=":", lw=0.7)

        dbic = row["retrieved_delta_bic"]
        pref = "W" if dbic < 0 else "C"
        ax.set_title(
            f"ΔBIC={dbic:.1f} → {pref}\n"
            f"R²_w={row['retrieved_w_r_squared']:.2f}, "
            f"R²_c={row['retrieved_c_r_squared']:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("TR", fontsize=8)
        ax.set_ylabel("SODA", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)

    for j in range(n_plots, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Model comparison: windowed (red) vs continuous (blue) sinusoid", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison_summary(
    metrics_df: pd.DataFrame,
    save_path: str = None,
):
    """
    Summary of model comparison across trials and conditions.
    """
    conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]
    cond_labels = ["Retrieved", "Control", "Ret (shuf)", "Ctrl (shuf)"]
    colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: ΔBIC distribution (retrieved condition)
    ax = axes[0]
    dbic_col = "retrieved_delta_bic"
    if dbic_col in metrics_df.columns:
        vals = metrics_df[dbic_col].dropna()
        ax.hist(vals, bins=40, color="#3274A1", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", ls="--", lw=1.5)
        n_w = (vals < 0).sum()
        n_c = (vals > 0).sum()
        ax.set_xlabel("ΔBIC (windowed − continuous)")
        ax.set_ylabel("Count")
        ax.set_title(f"Model preference (retrieved)\nWindowed: {n_w} | Continuous: {n_c}")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: R² comparison windowed vs continuous
    ax = axes[1]
    r2_w = "retrieved_w_r_squared"
    r2_c = "retrieved_c_r_squared"
    if r2_w in metrics_df.columns and r2_c in metrics_df.columns:
        valid = metrics_df.dropna(subset=[r2_w, r2_c])
        ax.scatter(valid[r2_c], valid[r2_w], s=8, alpha=0.3, color="#3274A1")
        lim = max(valid[r2_c].max(), valid[r2_w].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("R² (continuous)")
        ax.set_ylabel("R² (windowed)")
        ax.set_title("R² comparison per trial")
        ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: Amplitude boxplot (windowed) across 4 conditions
    ax = axes[2]
    amp_cols = [f"{c}_w_amplitude" for c in conditions]
    available = [c for c in amp_cols if c in metrics_df.columns]
    if len(available) >= 2 and "subject" in metrics_df.columns:
        # Use absolute amplitude for comparison
        subj = metrics_df.groupby("subject")[available].apply(
            lambda x: x.abs().mean()
        ).dropna()

        if len(subj) >= 3:
            plot_data = [subj[c].values for c in available]
            plot_labels = [cond_labels[amp_cols.index(c)] for c in available]
            plot_colors = [colors[amp_cols.index(c)] for c in available]

            bp = ax.boxplot(plot_data, positions=range(len(plot_data)),
                           widths=0.5, patch_artist=True, showfliers=False)
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
                          color=color, edgecolor="k", linewidth=0.5,
                          s=25, zorder=5, alpha=0.8)

            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, fontsize=9)
            ax.set_ylabel("|Amplitude| (windowed model)")
            ax.set_title("Windowed amplitude across conditions")

            # --- Paired t-test: retrieved vs control ---
            ret_col = "retrieved_w_amplitude"
            ctrl_col = "control_w_amplitude"
            if ret_col in available and ctrl_col in available:
                ret_vals = subj[ret_col].values
                ctrl_vals = subj[ctrl_col].values
                t_stat, p_val = ttest_rel(ret_vals, ctrl_vals)

                # Significance stars
                if p_val < 0.001:
                    sig_str = f"p = {p_val:.1e} ***"
                elif p_val < 0.01:
                    sig_str = f"p = {p_val:.3f} **"
                elif p_val < 0.05:
                    sig_str = f"p = {p_val:.3f} *"
                else:
                    sig_str = f"p = {p_val:.3f} n.s."

                # Draw bracket between retrieved (pos 0) and control (pos 1)
                idx_ret = available.index(ret_col)
                idx_ctrl = available.index(ctrl_col)
                y_max = max(np.max(plot_data[idx_ret]), np.max(plot_data[idx_ctrl]))
                bracket_y = y_max * 1.08
                bracket_h = y_max * 0.03
                ax.plot([idx_ret, idx_ret, idx_ctrl, idx_ctrl],
                        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                        lw=1.2, color="black")
                ax.text((idx_ret + idx_ctrl) / 2, bracket_y + bracket_h * 1.5,
                        sig_str, ha="center", va="bottom", fontsize=8)

                print(f"  Ret vs Ctrl |amplitude| (windowed): "
                      f"t({len(ret_vals)-1})={t_stat:.3f}, {sig_str}")

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_windowed_parameters(
    metrics_df: pd.DataFrame,
    save_path: str = None,
):
    """
    Distribution of windowed model parameters (duration, onset) for
    retrieved vs control.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Duration
    ax = axes[0]
    for cond, color, label in [
        ("retrieved", "#3274A1", "Retrieved"),
        ("control", "#E1812C", "Control"),
    ]:
        col = f"{cond}_w_duration"
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            ax.hist(vals, bins=30, color=color, alpha=0.5, edgecolor="white", label=label)
    ax.set_xlabel("Duration λ (TRs)")
    ax.set_ylabel("Count")
    ax.set_title("Cycle duration")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Onset
    ax = axes[1]
    for cond, color, label in [
        ("retrieved", "#3274A1", "Retrieved"),
        ("control", "#E1812C", "Control"),
    ]:
        col = f"{cond}_w_onset"
        if col in metrics_df.columns:
            vals = metrics_df[col].dropna()
            ax.hist(vals, bins=30, color=color, alpha=0.5, edgecolor="white", label=label)
    ax.set_xlabel("Onset d (TRs)")
    ax.set_ylabel("Count")
    ax.set_title("Retrieval onset")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Onset vs duration scatter (retrieved only)
    ax = axes[2]
    d_col = "retrieved_w_duration"
    o_col = "retrieved_w_onset"
    if d_col in metrics_df.columns and o_col in metrics_df.columns:
        valid = metrics_df.dropna(subset=[d_col, o_col])
        scatter = ax.scatter(
            valid[o_col], valid[d_col],
            c=valid.get("behav_correct", pd.Series(np.nan)),
            cmap="RdBu", s=10, alpha=0.4, edgecolors="none",
        )
        ax.set_xlabel("Onset d (TRs)")
        ax.set_ylabel("Duration λ (TRs)")
        ax.set_title("Onset vs duration (retrieved)")
        if "behav_correct" in valid.columns:
            plt.colorbar(scatter, ax=ax, label="Accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# 6. ENTRY POINT
# =====================================================================

def main():
    # ---- CLI: optional ROI filter and pre/post flag ----
    # Usage:  python 4.7_sinus_1cycle.py [ROI] [presleep|postsleep]
    roi_filter = sys.argv[1] if len(sys.argv) > 1 else None
    Pre_post   = sys.argv[2] if len(sys.argv) > 2 else 'presleep'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir,
                  f'4.6_SODA_based_on_singlesubj_4.4_{Pre_post}', 'SODA_results_csv')

    csv_files = sorted(glob.glob(opj(csv_dir, '*_soda_results_per_combination.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    all_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if 'ROI' not in df.columns:
            roi_name = os.path.basename(fp).replace('_soda_results_per_combination.csv', '')
            df['ROI'] = roi_name
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    for col in ["tr", "retrieved_soda", "control_soda", "behav_correct",
                 "retrieved_soda_shuffled", "control_soda_shuffled"]:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    # If an ROI was specified on the command line, filter to just that one
    if roi_filter:
        master_df = master_df[master_df['ROI'] == roi_filter]
        if master_df.empty:
            raise ValueError(f"ROI '{roi_filter}' not found in data. "
                             f"Available: {sorted(pd.concat(all_dfs, ignore_index=True)['ROI'].unique())}")

    rois = sorted(master_df['ROI'].unique())
    print(f"Found {len(rois)} ROIs: {rois}")

    all_roi_summaries = []

    for roi in rois:
        print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

        df = master_df[master_df['ROI'] == roi].copy()
        out_dir = create_script_named_folder(roi)

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Fitting windowed + continuous models per trial...")

        metrics_df = compute_trial_level_fits(df)
        print(f"Computed fits for {len(metrics_df)} trials")

        # Save
        out_csv = opj(out_dir, f"{roi}_windowed_fit_results.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"Saved to: {out_csv}")

        # Summary stats
        for cond in ["retrieved", "control"]:
            dbic_col = f"{cond}_delta_bic"
            if dbic_col in metrics_df.columns:
                vals = metrics_df[dbic_col].dropna()
                n_w = (vals < 0).sum()
                n_c = (vals > 0).sum()
                print(f"  {cond}: windowed preferred in {n_w}/{len(vals)} trials "
                      f"({100*n_w/len(vals):.1f}%)")

        # Plots
        print("Generating plots...")

        plot_example_fits_comparison(
            df, metrics_df,
            save_path=opj(out_dir, f"{roi}_example_fits_comparison.png"),
        )

        plot_model_comparison_summary(
            metrics_df,
            save_path=opj(out_dir, f"{roi}_model_comparison_summary.png"),
        )

        plot_windowed_parameters(
            metrics_df,
            save_path=opj(out_dir, f"{roi}_windowed_parameters.png"),
        )

        # Group stats
        roi_summary = {
            "ROI": roi,
            "n_subjects": df["subject"].nunique(),
            "n_trials": len(metrics_df),
        }

        # Subject-level amplitude comparison (windowed model)
        for cond_a, cond_b, label in [
            ("retrieved_w_amplitude", "control_w_amplitude", "ret_vs_ctrl_w"),
            ("retrieved_w_amplitude", "shuf_retrieved_w_amplitude", "ret_vs_shuf_w"),
        ]:
            if cond_a in metrics_df.columns and cond_b in metrics_df.columns:
                subj = metrics_df.groupby("subject")[[cond_a, cond_b]].apply(
                    lambda x: x.abs().mean()
                ).dropna()
                if len(subj) >= 3:
                    t, p = ttest_rel(subj[cond_a], subj[cond_b])
                    print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj)})")
                    roi_summary[f"t_{label}"] = t
                    roi_summary[f"p_{label}"] = p

        all_roi_summaries.append(roi_summary)
        print(f"\nDone with ROI {roi}.")

    summary_dir = create_script_named_folder('summary')
    summary_df = pd.DataFrame(all_roi_summaries)
    summary_path = opj(summary_dir, 'windowed_fit_summary_all_rois.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()