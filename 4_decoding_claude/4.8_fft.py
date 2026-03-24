# -*- coding: utf-8 -*-
"""FFT-based spectral analysis of SODA timecourses.

Following Wittkuhn & Schuck (2021): power spectral density of the
sequentiality metric is phase-invariant, so trials with different
retrieval onsets still contribute to the same spectral peak.

For each trial's SODA timecourse across TRs:
  1. Remove the mean (DC component)
  2. Compute one-sided power spectrum via periodogram
  3. Extract mean power in target frequency band [f_min, f_max]
  4. Compare across conditions

With TR=1.5s and ~10 TRs, a retrieval spanning 5-12 TRs corresponds
to roughly 0.08-0.20 cycles/TR.

Usage
-----
    python 4.8_fft.py [ROI] [presleep|postsleep]
"""

import os
import sys
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.signal import periodogram
from matplotlib import pyplot as plt


# =====================================================================
# 1. SPECTRAL ANALYSIS FUNCTIONS
# =====================================================================

def compute_power_spectrum(
    soda_values: np.ndarray,
    tr_duration: float = 1.0,
    nfft_factor: int = 4,
    window: str = "hann",
    detrend: str = "constant",
) -> dict:
    """Compute one-sided power spectral density of a SODA timecourse.

    Parameters
    ----------
    soda_values : array (n_trs,)
    tr_duration : float
        Sampling interval. 1.0 gives frequencies in cycles/TR.
    nfft_factor : int
        Zero-pad factor for finer frequency resolution.
    window : str
        Window function ('hann' reduces spectral leakage).
    detrend : str
        'constant' removes mean, 'linear' removes trend.

    Returns
    -------
    dict with freqs, power, total_power, success
    """
    v = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(v)
    v = v[valid]

    if len(v) < 4:
        return {"freqs": np.array([]), "power": np.array([]),
                "total_power": np.nan, "success": False}

    nfft = len(v) * nfft_factor

    freqs, psd = periodogram(
        v,
        fs=1.0 / tr_duration,
        window=window,
        nfft=nfft,
        detrend=detrend,
        return_onesided=True,
        scaling="spectrum",
    )

    # Exclude DC (index 0)
    freqs = freqs[1:]
    psd = psd[1:]

    return {
        "freqs": freqs,
        "power": psd,
        "total_power": float(np.sum(psd)),
        "success": True,
    }


def band_power(freqs, power, f_min, f_max):
    """Mean power within a frequency band."""
    band = (freqs >= f_min) & (freqs <= f_max)
    if not band.any():
        return np.nan
    return float(np.mean(power[band]))


def peak_frequency(freqs, power, f_min=0.0, f_max=np.inf):
    """Frequency with maximum power (optionally within a band)."""
    band = (freqs >= f_min) & (freqs <= f_max)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(power[band])])


def relative_band_power(freqs, power, f_min, f_max):
    """Band power as fraction of total power (excluding DC)."""
    total = np.sum(power)
    if total == 0:
        return np.nan
    bp = band_power(freqs, power, f_min, f_max)
    return bp / (total / len(power)) if not np.isnan(bp) else np.nan


# =====================================================================
# 2. TRIAL-LEVEL COMPUTATION
# =====================================================================

SODA_COLS = {
    "retrieved": "retrieved_soda",
    "control": "control_soda",
    "shuf_retrieved": "retrieved_soda_shuffled",
    "shuf_control": "control_soda_shuffled",
    "shuf_sel_retrieved": "retrieved_soda_shuffled_selected",
    "shuf_sel_control": "control_soda_shuffled_selected",
}


def compute_trial_level_fft(
    df: pd.DataFrame,
    soda_cols: dict | None = None,
    f_min: float = 0.08,
    f_max: float = 0.20,
    tr_duration: float = 1.0,
) -> pd.DataFrame:
    """Compute FFT-based spectral metrics per trial for all conditions.

    Returns DataFrame with one row per trial, columns for band_power,
    peak_freq, relative_band_power per condition.
    """
    if soda_cols is None:
        soda_cols = SODA_COLS

    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
                  if c in df.columns]
    results = []

    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))
        trial_df = trial_df.sort_values("tr")

        for cond_label, col_name in soda_cols.items():
            if col_name not in trial_df.columns:
                continue

            vals = trial_df[col_name].values
            spec = compute_power_spectrum(vals, tr_duration=tr_duration)

            if spec["success"]:
                trial_info[f"{cond_label}_band_power"] = band_power(
                    spec["freqs"], spec["power"], f_min, f_max)
                trial_info[f"{cond_label}_total_power"] = spec["total_power"]
                trial_info[f"{cond_label}_peak_freq"] = peak_frequency(
                    spec["freqs"], spec["power"], f_min, f_max)
                trial_info[f"{cond_label}_relative_band_power"] = relative_band_power(
                    spec["freqs"], spec["power"], f_min, f_max)
            else:
                trial_info[f"{cond_label}_band_power"] = np.nan
                trial_info[f"{cond_label}_total_power"] = np.nan
                trial_info[f"{cond_label}_peak_freq"] = np.nan
                trial_info[f"{cond_label}_relative_band_power"] = np.nan

        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 3. PLOTTING
# =====================================================================

COND_STYLES = {
    "retrieved":          {"color": "#3274A1", "lw": 2.5, "label": "Retrieved"},
    "control":            {"color": "#E1812C", "lw": 2.5, "label": "Control"},
    "shuf_retrieved":     {"color": "#8CB8D0", "lw": 1.5, "label": "Shuf Ret", "ls": "--"},
    "shuf_control":       {"color": "#F0C08A", "lw": 1.5, "label": "Shuf Ctrl", "ls": "--"},
    "shuf_sel_retrieved": {"color": "#A0D0A0", "lw": 1.5, "label": "Shuf Sel Ret", "ls": ":"},
    "shuf_sel_control":   {"color": "#D0A0D0", "lw": 1.5, "label": "Shuf Sel Ctrl", "ls": ":"},
}


def plot_band_power_boxplots(
    metrics_df: pd.DataFrame,
    metric: str = "band_power",
    save_path: str | None = None,
):
    """Boxplot of FFT band power across all 6 conditions."""
    conditions = list(SODA_COLS.keys())
    col_names = [f"{c}_{metric}" for c in conditions]
    available = [c for c in col_names if c in metrics_df.columns]

    if len(available) < 2 or "subject" not in metrics_df.columns:
        print(f"Not enough data for boxplot (found {len(available)} conditions)")
        return

    subj = metrics_df.groupby("subject")[available].mean().dropna()
    if len(subj) < 3:
        print("Too few subjects for boxplot")
        return

    plot_data = [subj[c].values for c in available]
    plot_labels = [COND_STYLES.get(conditions[col_names.index(c)], {}).get("label", c)
                   for c in available]
    plot_colors = [COND_STYLES.get(conditions[col_names.index(c)], {}).get("color", "grey")
                   for c in available]

    fig, ax = plt.subplots(figsize=(9, 5))

    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
                    patch_artist=True, showfliers=False)
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
                   s=30, zorder=5, alpha=0.8)

    # Paired t-tests
    stat_lines = []
    ret_col = f"retrieved_{metric}"
    for other, label in [
        (f"control_{metric}", "Ret vs Ctrl"),
        (f"shuf_retrieved_{metric}", "Ret vs Shuf"),
        (f"shuf_sel_retrieved_{metric}", "Ret vs ShufSel"),
    ]:
        if ret_col in available and other in available:
            t, p = ttest_rel(subj[ret_col], subj[other])
            sig = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
            stat_lines.append(f"{label}: t={t:.2f}, {sig}")
            print(f"  {label}: t={t:.2f}, {sig}")

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(f"Mean {metric}")
    title = f"FFT {metric} across conditions (n={len(subj)})"
    if stat_lines:
        title += "\n" + " | ".join(stat_lines)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 4. ENTRY POINT
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

    # Ensure numeric columns
    numeric_cols = list(SODA_COLS.values()) + ["tr", "behav_correct"]
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
        out_dir = opj(script_dir, "4.8_fft", roi)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Computing FFT per trial...")

        metrics_df = compute_trial_level_fft(df)
        print(f"Computed FFT for {len(metrics_df)} trials")

        metrics_df.to_csv(opj(out_dir, f"{roi}_fft_results_{pre_post}.csv"), index=False)

        print("Generating plots...")
        plot_band_power_boxplots(
            metrics_df, metric="band_power",
            save_path=opj(out_dir, f"{roi}_band_power_boxplots_{pre_post}.png"),
        )
        plot_band_power_boxplots(
            metrics_df, metric="relative_band_power",
            save_path=opj(out_dir, f"{roi}_relative_band_power_boxplots_{pre_post}.png"),
        )

        roi_summary = {"ROI": roi, "n_subjects": df["subject"].nunique(), "n_trials": len(metrics_df)}

        for cond_a, cond_b, label in [
            ("retrieved_band_power", "control_band_power", "ret_vs_ctrl"),
            ("retrieved_band_power", "shuf_retrieved_band_power", "ret_vs_shuf"),
            ("retrieved_band_power", "shuf_sel_retrieved_band_power", "ret_vs_shuf_sel"),
        ]:
            if cond_a in metrics_df.columns and cond_b in metrics_df.columns:
                subj = metrics_df.groupby("subject")[[cond_a, cond_b]].mean().dropna()
                if len(subj) >= 3:
                    t, p = ttest_rel(subj[cond_a], subj[cond_b])
                    print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj)})")
                    roi_summary[f"t_{label}"] = t
                    roi_summary[f"p_{label}"] = p

        all_roi_summaries.append(roi_summary)
        print(f"Done with ROI {roi}.")

    summary_dir = opj(script_dir, "4.8_fft", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = opj(summary_dir, f"fft_summary_{pre_post}.csv")
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
