# # -*- coding: utf-8 -*-
# """
# FFT-based analysis of SODA (sequentiality) timecourses.

# Rationale
# ---------
# Following Wittkuhn & Schuck (2021), the power spectral density of the
# sequentiality metric can distinguish sequential retrieval from noise.
# If a trial contains a sinusoidal SODA pattern (negative→positive slope),
# the FFT will show elevated power at the corresponding frequency. This
# power is phase-invariant: trials with different retrieval onsets will
# still show the same spectral peak, solving the phase cancellation problem.

# Approach
# --------
# For each trial's SODA timecourse across TRs:
#   1. Remove the mean (DC component)
#   2. Compute the one-sided power spectrum via FFT
#   3. Extract mean power in a target frequency band [f_min, f_max]
#   4. Compare across conditions: retrieved vs control vs shuffled

# The target band should correspond to plausible retrieval durations.
# With TR=1.5s and 10 TRs (15s window), frequencies range from ~0.07
# to 0.5 cycles/TR (Nyquist). A retrieval spanning 5-12 TRs corresponds
# to roughly 0.08-0.20 cycles/TR.

# Usage
# -----
#     python fft_soda_analysis.py
# """

# import os
# import sys
# import glob
# import warnings
# from os.path import join as opj

# import numpy as np
# import pandas as pd
# from scipy.stats import ttest_rel, ttest_1samp, pearsonr
# from scipy.signal import periodogram
# from matplotlib import pyplot as plt


# def create_script_named_folder(folder_name=None):
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     script_name = os.path.basename(__file__)
#     script_name_without_ext = os.path.splitext(script_name)[0]
#     if folder_name is None:
#         new_dir_path = os.path.join(script_dir, script_name_without_ext)
#     else:
#         new_dir_path = os.path.join(script_dir, script_name_without_ext, folder_name)
#     if not os.path.exists(new_dir_path):
#         os.makedirs(new_dir_path)
#     return new_dir_path


# # =====================================================================
# # 1. SPECTRAL ANALYSIS FUNCTIONS
# # =====================================================================

# def compute_power_spectrum(soda_values: np.ndarray, tr_duration: float = 1.0,
#                            nfft_factor: int = 4, window: str = "hann",
#                            detrend: str = "constant"):
#     """
#     Compute the one-sided power spectral density of a SODA timecourse
#     using scipy.signal.periodogram.

#     Parameters
#     ----------
#     soda_values : array, shape (n_trs,)
#         SODA slope values at each TR.
#     tr_duration : float
#         Sampling interval. Default 1.0 gives frequencies in cycles/TR.
#         Set to e.g. 1.5 to get frequencies in Hz.
#     nfft_factor : int
#         Zero-pad to nfft = len(signal) * nfft_factor for finer frequency
#         resolution. Default 4.
#     window : str
#         Window function applied before FFT. 'hann' reduces spectral leakage.
#         Use 'boxcar' for no windowing (equivalent to a rectangular window).
#     detrend : str
#         Detrending mode passed to periodogram.
#         'constant' removes mean (DC). 'linear' removes linear trend. 
#         False for no detrending.

#     Returns
#     -------
#     dict with keys:
#         freqs : array, one-sided frequencies (excluding DC)
#         power : array, PSD at each frequency (excluding DC)
#         total_power : float, total power (sum of PSD)
#         success : bool
#     """
#     v = np.asarray(soda_values, dtype=float)
#     valid = ~np.isnan(v)
#     v = v[valid]

#     if len(v) < 4:
#         return {"freqs": np.array([]), "power": np.array([]),
#                 "total_power": np.nan, "success": False}

#     nfft = len(v) * nfft_factor

#     freqs, psd = periodogram(
#         v,
#         fs=1.0 / tr_duration,     # sampling frequency
#         window=window,
#         nfft=nfft,
#         detrend=detrend,
#         return_onesided=True,
#         scaling="spectrum",        # power spectrum (V²) not density (V²/Hz)
#     )

#     # Exclude DC (index 0)
#     freqs = freqs[1:]
#     psd = psd[1:]

#     return {
#         "freqs": freqs,
#         "power": psd,
#         "total_power": float(np.sum(psd)),
#         "success": True,
#     }


# def band_power(freqs, power, f_min, f_max):
#     """Mean power within a frequency band."""
#     band = (freqs >= f_min) & (freqs <= f_max)
#     if not band.any():
#         return np.nan
#     return float(np.mean(power[band]))


# def peak_frequency(freqs, power, f_min=0.0, f_max=np.inf):
#     """Frequency with maximum power (optionally within a band)."""
#     band = (freqs >= f_min) & (freqs <= f_max)
#     if not band.any():
#         return np.nan
#     band_freqs = freqs[band]
#     band_power_vals = power[band]
#     return float(band_freqs[np.argmax(band_power_vals)])


# def relative_band_power(freqs, power, f_min, f_max):
#     """Band power as fraction of total power (excluding DC)."""
#     total = np.sum(power)
#     if total == 0:
#         return np.nan
#     bp = band_power(freqs, power, f_min, f_max)
#     return bp / (total / len(power)) if not np.isnan(bp) else np.nan


# # =====================================================================
# # 2. TRIAL-LEVEL COMPUTATION
# # =====================================================================

# def compute_trial_level_fft(
#     df: pd.DataFrame,
#     soda_cols: dict = None,
#     f_min: float = 0.08,
#     f_max: float = 0.20,
#     tr_duration: float = 1.0,
# ) -> pd.DataFrame:
#     """
#     For each trial, compute FFT-based spectral metrics for all SODA conditions.

#     Parameters
#     ----------
#     df : DataFrame
#         SODA results with columns: subject, trial, tr, retrieved_soda, etc.
#     soda_cols : dict
#         Mapping from condition label to column name.
#         Default: retrieved, control, shuffled_retrieved, shuffled_control.
#     f_min, f_max : float
#         Target frequency band (cycles/TR).
#     tr_duration : float
#         TR duration for frequency axis (1.0 = cycles/TR).

#     Returns
#     -------
#     DataFrame with one row per trial, columns for each condition's spectral metrics.
#     """
#     if soda_cols is None:
#         soda_cols = {
#             "retrieved": "retrieved_soda",
#             "control": "control_soda",
#             "shuf_retrieved": "retrieved_soda_shuffled",
#             "shuf_control": "control_soda_shuffled",
#         }

#     group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
#     group_cols = [c for c in group_cols if c in df.columns]

#     results = []

#     for keys, trial_df in df.groupby(group_cols, sort=False):
#         if not isinstance(keys, tuple):
#             keys = (keys,)
#         trial_info = dict(zip(group_cols, keys))

#         trial_df = trial_df.sort_values("tr")
#         trs = trial_df["tr"].values

#         # Compute spectral metrics for each condition
#         for cond_label, col_name in soda_cols.items():
#             if col_name not in trial_df.columns:
#                 continue

#             vals = trial_df[col_name].values
#             spec = compute_power_spectrum(vals, tr_duration=tr_duration)

#             prefix = cond_label

#             if spec["success"]:
#                 trial_info[f"{prefix}_band_power"] = band_power(
#                     spec["freqs"], spec["power"], f_min, f_max
#                 )
#                 trial_info[f"{prefix}_total_power"] = spec["total_power"]
#                 trial_info[f"{prefix}_peak_freq"] = peak_frequency(
#                     spec["freqs"], spec["power"], f_min, f_max
#                 )
#                 trial_info[f"{prefix}_relative_band_power"] = relative_band_power(
#                     spec["freqs"], spec["power"], f_min, f_max
#                 )
#                 # Store full spectrum for averaging later
#                 trial_info[f"{prefix}_freqs"] = spec["freqs"]
#                 trial_info[f"{prefix}_power"] = spec["power"]
#             else:
#                 trial_info[f"{prefix}_band_power"] = np.nan
#                 trial_info[f"{prefix}_total_power"] = np.nan
#                 trial_info[f"{prefix}_peak_freq"] = np.nan
#                 trial_info[f"{prefix}_relative_band_power"] = np.nan
#                 trial_info[f"{prefix}_freqs"] = np.array([])
#                 trial_info[f"{prefix}_power"] = np.array([])

#         # Behavior
#         if "behav_correct" in trial_df.columns:
#             bc = trial_df["behav_correct"].dropna()
#             trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

#         # Retrieval position
#         if "retrieved_position" in trial_df.columns:
#             rp = trial_df["retrieved_position"].dropna()
#             trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

#         results.append(trial_info)

#     return pd.DataFrame(results)


# # =====================================================================
# # 3. PLOTTING
# # =====================================================================

# def plot_average_power_spectra(
#     metrics_df: pd.DataFrame,
#     conditions: list = None,
#     f_min: float = 0.08,
#     f_max: float = 0.20,
#     save_path: str = None,
# ):
#     """
#     Plot the average power spectrum across trials for each condition.
#     Shows the target frequency band as a shaded region.
#     """
#     if conditions is None:
#         conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]

#     cond_styles = {
#         "retrieved":      {"color": "#3274A1", "lw": 2.5, "label": "Retrieved"},
#         "control":        {"color": "#E1812C", "lw": 2.5, "label": "Control"},
#         "shuf_retrieved": {"color": "#8CB8D0", "lw": 1.5, "label": "Retrieved (shuffled)", "ls": "--"},
#         "shuf_control":   {"color": "#F0C08A", "lw": 1.5, "label": "Control (shuffled)", "ls": "--"},
#     }

#     fig, ax = plt.subplots(figsize=(8, 5))

#     for cond in conditions:
#         freq_col = f"{cond}_freqs"
#         power_col = f"{cond}_power"

#         if freq_col not in metrics_df.columns:
#             continue

#         # Collect all spectra that have data
#         all_freqs = []
#         all_power = []
#         for _, row in metrics_df.iterrows():
#             f = row[freq_col]
#             p = row[power_col]
#             if isinstance(f, np.ndarray) and len(f) > 0:
#                 all_freqs.append(f)
#                 all_power.append(p)

#         if not all_power:
#             continue

#         # All trials should have same frequency axis if same number of TRs
#         # Group by frequency length and average within groups
#         # (handles potential variable-length trials)
#         ref_freqs = all_freqs[0]
#         matching = [(f, p) for f, p in zip(all_freqs, all_power) if len(f) == len(ref_freqs)]

#         if not matching:
#             continue

#         power_matrix = np.array([p for _, p in matching])
#         mean_power = np.mean(power_matrix, axis=0)
#         sem_power = np.std(power_matrix, axis=0, ddof=1) / np.sqrt(power_matrix.shape[0])

#         style = cond_styles.get(cond, {"color": "grey", "lw": 1.5, "label": cond})
#         ls = style.get("ls", "-")

#         ax.plot(ref_freqs, mean_power, color=style["color"], lw=style["lw"],
#                 ls=ls, label=style["label"])
#         ax.fill_between(ref_freqs, mean_power - sem_power, mean_power + sem_power,
#                         color=style["color"], alpha=0.15, linewidth=0)

#     # Shade target band
#     ax.axvspan(f_min, f_max, color="grey", alpha=0.1, label=f"Target band [{f_min}–{f_max}]")

#     ax.set_xlabel("Frequency (cycles/TR)")
#     ax.set_ylabel("Power")
#     ax.set_title("Average power spectrum of SODA timecourse")
#     ax.legend(frameon=False, fontsize=9)
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_band_power_boxplots(
#     metrics_df: pd.DataFrame,
#     metric: str = "band_power",
#     save_path: str = None,
# ):
#     """
#     Boxplot of band power across 4 conditions, mirroring the 4.7 layout.
#     Subject-level averages with individual dots.
#     """
#     conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]
#     col_names = [f"{c}_{metric}" for c in conditions]
#     available = [c for c in col_names if c in metrics_df.columns]

#     if len(available) < 2:
#         print(f"Not enough conditions for boxplot (found {len(available)})")
#         return

#     # Subject-level means
#     subj = metrics_df.groupby("subject")[available].mean().dropna()
#     if len(subj) < 3:
#         print("Too few subjects for boxplot")
#         return

#     labels = ["Retrieved", "Control", "Ret (shuf)", "Ctrl (shuf)"]
#     colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A"]
#     # Only keep those that are available
#     plot_data = []
#     plot_labels = []
#     plot_colors = []
#     for col, lab, clr in zip(col_names, labels, colors):
#         if col in available:
#             plot_data.append(subj[col].values)
#             plot_labels.append(lab)
#             plot_colors.append(clr)

#     fig, ax = plt.subplots(figsize=(7, 5))

#     bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
#                     patch_artist=True, showfliers=False)
#     for patch, color in zip(bp["boxes"], plot_colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.5)
#     for median in bp["medians"]:
#         median.set_color("black")
#         median.set_linewidth(1.5)

#     # Individual subjects as jittered dots
#     rng = np.random.default_rng(42)
#     for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
#         jitter = rng.normal(0, 0.06, size=len(vals))
#         ax.scatter(np.full_like(vals, i) + jitter, vals,
#                    color=color, edgecolor="k", linewidth=0.5,
#                    s=30, zorder=5, alpha=0.8)

#     ax.axhline(0, color="grey", ls="--", lw=0.8)

#     # Paired t-tests
#     stat_lines = []
#     if f"retrieved_{metric}" in available and f"control_{metric}" in available:
#         t, p = ttest_rel(subj[f"retrieved_{metric}"], subj[f"control_{metric}"])
#         stat_lines.append(f"Ret vs Ctrl: t={t:.2f}, p={p:.3f}")
#     if f"retrieved_{metric}" in available and f"shuf_retrieved_{metric}" in available:
#         t, p = ttest_rel(subj[f"retrieved_{metric}"], subj[f"shuf_retrieved_{metric}"])
#         stat_lines.append(f"Ret vs Ret(shuf): t={t:.2f}, p={p:.3f}")

#     ax.set_xticks(range(len(plot_labels)))
#     ax.set_xticklabels(plot_labels)
#     ax.set_ylabel(f"Mean {metric}")
#     title = f"FFT {metric} across conditions (n={len(subj)})"
#     if stat_lines:
#         title += "\n" + " | ".join(stat_lines)
#     ax.set_title(title, fontsize=10)
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_band_power_by_accuracy(
#     metrics_df: pd.DataFrame,
#     metric: str = "band_power",
#     save_path: str = None,
# ):
#     """
#     Compare band power for correct vs incorrect trials.
#     """
#     ret_col = f"retrieved_{metric}"
#     ctrl_col = f"control_{metric}"

#     if ret_col not in metrics_df.columns:
#         print(f"Column {ret_col} not found")
#         return

#     df = metrics_df.dropna(subset=[ret_col, "behav_correct"]).copy()
#     if df.empty:
#         return

#     # Subject-level averages split by accuracy
#     subj = df.groupby(["subject", "behav_correct"]).agg(
#         retrieved_power=(ret_col, "mean"),
#         control_power=(ctrl_col, "mean") if ctrl_col in df.columns else (ret_col, "mean"),
#     ).reset_index()

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Panel 1: Correct vs Incorrect
#     ax = axes[0]
#     for acc_val, color, label in [(1, "#3274A1", "Correct"), (0, "#E1812C", "Incorrect")]:
#         vals = subj[subj["behav_correct"] == acc_val]["retrieved_power"]
#         if len(vals) > 0:
#             ax.bar(label, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0,
#                    color=color, alpha=0.7, capsize=5)
#             ax.scatter([label] * len(vals), vals, color=color, alpha=0.4, zorder=3, s=20)
#     ax.set_ylabel(f"Mean retrieved {metric}")
#     ax.set_title(f"Retrieved {metric} by accuracy")
#     ax.spines[["top", "right"]].set_visible(False)

#     # Panel 2: Retrieved vs Control for correct trials
#     ax = axes[1]
#     correct = subj[subj["behav_correct"] == 1]
#     if len(correct) > 1:
#         for col, color, label in [
#             ("retrieved_power", "#3274A1", "Retrieved"),
#             ("control_power", "#E1812C", "Control"),
#         ]:
#             vals = correct[col]
#             ax.bar(label, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0,
#                    color=color, alpha=0.7, capsize=5)
#             ax.scatter([label] * len(vals), vals, color=color, alpha=0.4, zorder=3, s=20)

#         if len(correct) >= 3:
#             t, p = ttest_rel(correct["retrieved_power"], correct["control_power"])
#             ax.set_title(f"Correct trials: Ret vs Ctrl\nt={t:.2f}, p={p:.3f}")
#     ax.set_ylabel(f"Mean {metric}")
#     ax.spines[["top", "right"]].set_visible(False)

#     fig.suptitle(f"FFT {metric} analysis", fontsize=13)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_brain_behavior_correlation_fft(
#     metrics_df: pd.DataFrame,
#     metric: str = "band_power",
#     save_path: str = None,
# ):
#     """
#     Subject-level correlation between band power and behavioral accuracy.
#     """
#     ret_col = f"retrieved_{metric}"
#     ctrl_col = f"control_{metric}"

#     cols_needed = [ret_col, "behav_correct"]
#     if not all(c in metrics_df.columns for c in cols_needed):
#         return

#     df = metrics_df.dropna(subset=cols_needed).copy()
#     if df.empty:
#         return

#     agg_dict = {
#         "mean_power": (ret_col, "mean"),
#         "mean_accuracy": ("behav_correct", "mean"),
#     }
#     if ctrl_col in df.columns:
#         agg_dict["mean_ctrl_power"] = (ctrl_col, "mean")

#     subj = df.groupby("subject").agg(**agg_dict).reset_index()

#     if ctrl_col in df.columns:
#         subj["power_delta"] = subj["mean_power"] - subj["mean_ctrl_power"]
#     else:
#         subj["power_delta"] = subj["mean_power"]

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     for ax, xcol, xlabel in zip(
#         axes,
#         ["mean_power", "power_delta"],
#         [f"Mean {ret_col}", f"Mean {ret_col} − control"],
#     ):
#         x = subj[xcol].values
#         y = subj["mean_accuracy"].values

#         ax.scatter(x, y, s=60, alpha=0.7, color="#3274A1")

#         if len(x) >= 5:
#             r, p = pearsonr(x, y)
#             m, b = np.polyfit(x, y, 1)
#             x_line = np.linspace(x.min(), x.max(), 100)
#             ax.plot(x_line, m * x_line + b, "r--", lw=2)
#             ax.set_title(f"r={r:.3f}, p={p:.3f}")

#         ax.set_xlabel(xlabel)
#         ax.set_ylabel("Mean behavioral accuracy")
#         ax.spines[["top", "right"]].set_visible(False)

#     fig.suptitle(f"Brain-behavior: FFT {metric} vs accuracy", fontsize=13)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_spectra_by_accuracy(
#     metrics_df: pd.DataFrame,
#     f_min: float = 0.08,
#     f_max: float = 0.20,
#     save_path: str = None,
# ):
#     """
#     Plot average power spectra split by behavioral accuracy (correct vs incorrect).
#     Only for the retrieved condition.
#     """
#     freq_col = "retrieved_freqs"
#     power_col = "retrieved_power"

#     if freq_col not in metrics_df.columns or "behav_correct" not in metrics_df.columns:
#         return

#     fig, ax = plt.subplots(figsize=(8, 5))

#     for acc_val, color, label in [(1, "#3274A1", "Correct"), (0, "#E1812C", "Incorrect")]:
#         subset = metrics_df[metrics_df["behav_correct"] == acc_val]

#         all_power = []
#         ref_freqs = None
#         for _, row in subset.iterrows():
#             f = row[freq_col]
#             p = row[power_col]
#             if isinstance(f, np.ndarray) and len(f) > 0:
#                 if ref_freqs is None:
#                     ref_freqs = f
#                 if len(f) == len(ref_freqs):
#                     all_power.append(p)

#         if not all_power or ref_freqs is None:
#             continue

#         power_matrix = np.array(all_power)
#         mean_power = np.mean(power_matrix, axis=0)
#         sem_power = np.std(power_matrix, axis=0, ddof=1) / np.sqrt(len(all_power))

#         ax.plot(ref_freqs, mean_power, color=color, lw=2, label=f"{label} (n={len(all_power)})")
#         ax.fill_between(ref_freqs, mean_power - sem_power, mean_power + sem_power,
#                         color=color, alpha=0.15, linewidth=0)

#     ax.axvspan(f_min, f_max, color="grey", alpha=0.1, label=f"Target band")
#     ax.set_xlabel("Frequency (cycles/TR)")
#     ax.set_ylabel("Power")
#     ax.set_title("Retrieved SODA power spectrum: correct vs incorrect")
#     ax.legend(frameon=False, fontsize=9)
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()


# # =====================================================================
# # 4. ENTRY POINT
# # =====================================================================

# def main():
#     # ── Configuration ──
#     Pre_post = 'presleep'
#     f_min = 0.08   # lower bound of target band (cycles/TR)
#     f_max = 0.20   # upper bound of target band (cycles/TR)
#     tr_duration = 1.0  # 1.0 = frequencies in cycles/TR; set to 1.5 for Hz

#     # ── Paths ──
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     csv_dir = opj(script_dir,
#                   f'4.6_SODA_based_on_singlesubj_4.4_{Pre_post}', 'SODA_results_csv')

#     csv_files = sorted(glob.glob(opj(csv_dir, '*_soda_results_per_combination.csv')))
#     if not csv_files:
#         raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

#     # ── Load all ROI CSVs ──
#     all_dfs = []
#     for fp in csv_files:
#         df = pd.read_csv(fp)
#         if 'ROI' not in df.columns:
#             roi_name = os.path.basename(fp).replace('_soda_results_per_combination.csv', '')
#             df['ROI'] = roi_name
#         all_dfs.append(df)

#     master_df = pd.concat(all_dfs, ignore_index=True)

#     for col in ["tr", "retrieved_soda", "control_soda", "behav_correct",
#                  "retrieved_soda_shuffled", "control_soda_shuffled"]:
#         if col in master_df.columns:
#             master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

#     rois = sorted(master_df['ROI'].unique())
#     print(f"Found {len(rois)} ROIs: {rois}")
#     print(f"Total rows: {len(master_df)}")
#     print(f"Target frequency band: [{f_min}, {f_max}] cycles/TR")

#     # ── Loop over ROIs ──
#     all_roi_summaries = []

#     for roi in rois:
#         print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

#         df = master_df[master_df['ROI'] == roi].copy()

#         out_dir = create_script_named_folder(roi)
#         print(f"Output directory: {out_dir}")
#         print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

#         # --- Compute trial-level FFT metrics ---
#         print("Computing FFT per trial...")
#         metrics_df = compute_trial_level_fft(
#             df, f_min=f_min, f_max=f_max, tr_duration=tr_duration
#         )
#         print(f"Computed metrics for {len(metrics_df)} trials")

#         # --- Save trial-level results ---
#         # Drop the array columns (freqs/power) for CSV — they're variable length
#         save_cols = [c for c in metrics_df.columns
#                      if not c.endswith("_freqs") and not c.endswith("_power")]
#         out_csv = opj(out_dir, f"{roi}_fft_results.csv")
#         metrics_df[save_cols].to_csv(out_csv, index=False)
#         print(f"Saved to: {out_csv}")

#         # --- Summary stats ---
#         conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]
#         for cond in conditions:
#             col = f"{cond}_band_power"
#             if col in metrics_df.columns:
#                 vals = metrics_df[col].dropna()
#                 print(f"  {cond} band_power: mean={vals.mean():.6f}, median={vals.median():.6f}")

#         # --- Plots ---
#         print("Generating plots...")

#         plot_average_power_spectra(
#             metrics_df, f_min=f_min, f_max=f_max,
#             save_path=opj(out_dir, f"{roi}_average_power_spectra.png"),
#         )

#         plot_band_power_boxplots(
#             metrics_df, metric="band_power",
#             save_path=opj(out_dir, f"{roi}_band_power_boxplot.png"),
#         )

#         plot_band_power_boxplots(
#             metrics_df, metric="total_power",
#             save_path=opj(out_dir, f"{roi}_total_power_boxplot.png"),
#         )

#         plot_band_power_by_accuracy(
#             metrics_df, metric="band_power",
#             save_path=opj(out_dir, f"{roi}_band_power_by_accuracy.png"),
#         )

#         plot_spectra_by_accuracy(
#             metrics_df, f_min=f_min, f_max=f_max,
#             save_path=opj(out_dir, f"{roi}_spectra_by_accuracy.png"),
#         )

#         plot_brain_behavior_correlation_fft(
#             metrics_df, metric="band_power",
#             save_path=opj(out_dir, f"{roi}_brain_behavior_correlation_fft.png"),
#         )

#         # --- Group stats ---
#         roi_summary = {
#             "ROI": roi,
#             "n_subjects": df["subject"].nunique(),
#             "n_trials": len(metrics_df),
#             "f_min": f_min,
#             "f_max": f_max,
#         }

#         bp_cols = {
#             "retrieved": "retrieved_band_power",
#             "control": "control_band_power",
#             "shuf_retrieved": "shuf_retrieved_band_power",
#             "shuf_control": "shuf_control_band_power",
#         }
#         available_bp = {k: v for k, v in bp_cols.items() if v in metrics_df.columns}

#         subj_means = metrics_df.groupby("subject")[list(available_bp.values())].mean().dropna()

#         # Paired t-tests
#         def _safe_ttest(col_a, col_b, label):
#             if col_a in subj_means.columns and col_b in subj_means.columns and len(subj_means) >= 3:
#                 t, p = ttest_rel(subj_means[col_a], subj_means[col_b])
#                 print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj_means)})")
#                 roi_summary[f"t_{label}"] = t
#                 roi_summary[f"p_{label}"] = p

#         _safe_ttest("retrieved_band_power", "control_band_power", "ret_vs_ctrl")
#         _safe_ttest("retrieved_band_power", "shuf_retrieved_band_power", "ret_vs_shuf_ret")
#         _safe_ttest("control_band_power", "shuf_control_band_power", "ctrl_vs_shuf_ctrl")

#         all_roi_summaries.append(roi_summary)
#         print(f"\nDone with ROI {roi}.")

#     # ── Save cross-ROI summary ──
#     summary_dir = create_script_named_folder('summary')
#     summary_df = pd.DataFrame(all_roi_summaries)
#     summary_path = opj(summary_dir, 'fft_summary_all_rois.csv')
#     summary_df.to_csv(summary_path, index=False)
#     print(f"\nCross-ROI summary saved to: {summary_path}")
#     print(summary_df.to_string(index=False))


# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
FFT-based analysis of SODA (sequentiality) timecourses.

Rationale
---------
Following Wittkuhn & Schuck (2021), the power spectral density of the
sequentiality metric can distinguish sequential retrieval from noise.
If a trial contains a sinusoidal SODA pattern (negative→positive slope),
the FFT will show elevated power at the corresponding frequency. This
power is phase-invariant: trials with different retrieval onsets will
still show the same spectral peak, solving the phase cancellation problem.

Approach
--------
For each trial's SODA timecourse across TRs:
  1. Remove the mean (DC component)
  2. Compute the one-sided power spectrum via FFT
  3. Extract mean power in a target frequency band [f_min, f_max]
  4. Compare across conditions: retrieved vs control vs shuffled

The target band should correspond to plausible retrieval durations.
With TR=1.5s and 10 TRs (15s window), frequencies range from ~0.07
to 0.5 cycles/TR (Nyquist). A retrieval spanning 5-12 TRs corresponds
to roughly 0.08-0.20 cycles/TR.

Usage
-----
    python fft_soda_analysis.py
"""

import os
import sys
import glob
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
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
# 1. SPECTRAL ANALYSIS FUNCTIONS
# =====================================================================

def compute_power_spectrum(soda_values: np.ndarray, tr_duration: float = 1.0,
                           nfft_factor: int = 4, window: str = "hann",
                           detrend: str = "constant"):
    """
    Compute the one-sided power spectral density of a SODA timecourse
    using scipy.signal.periodogram.

    Parameters
    ----------
    soda_values : array, shape (n_trs,)
        SODA slope values at each TR.
    tr_duration : float
        Sampling interval. Default 1.0 gives frequencies in cycles/TR.
        Set to e.g. 1.5 to get frequencies in Hz.
    nfft_factor : int
        Zero-pad to nfft = len(signal) * nfft_factor for finer frequency
        resolution. Default 4.
    window : str
        Window function applied before FFT. 'hann' reduces spectral leakage.
        Use 'boxcar' for no windowing (equivalent to a rectangular window).
    detrend : str
        Detrending mode passed to periodogram.
        'constant' removes mean (DC). 'linear' removes linear trend. 
        False for no detrending.

    Returns
    -------
    dict with keys:
        freqs : array, one-sided frequencies (excluding DC)
        power : array, PSD at each frequency (excluding DC)
        total_power : float, total power (sum of PSD)
        success : bool
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
        fs=1.0 / tr_duration,     # sampling frequency
        window=window,
        nfft=nfft,
        detrend=detrend,
        return_onesided=True,
        scaling="spectrum",        # power spectrum (V²) not density (V²/Hz)
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
    band_freqs = freqs[band]
    band_power_vals = power[band]
    return float(band_freqs[np.argmax(band_power_vals)])


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

def compute_trial_level_fft(
    df: pd.DataFrame,
    soda_cols: dict = None,
    f_min: float = 0.08,
    f_max: float = 0.20,
    tr_duration: float = 1.0,
) -> pd.DataFrame:
    """
    For each trial, compute FFT-based spectral metrics for all SODA conditions.

    Parameters
    ----------
    df : DataFrame
        SODA results with columns: subject, trial, tr, retrieved_soda, etc.
    soda_cols : dict
        Mapping from condition label to column name.
        Default: retrieved, control, shuffled_retrieved, shuffled_control.
    f_min, f_max : float
        Target frequency band (cycles/TR).
    tr_duration : float
        TR duration for frequency axis (1.0 = cycles/TR).

    Returns
    -------
    DataFrame with one row per trial, columns for each condition's spectral metrics.
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

        # Compute spectral metrics for each condition
        for cond_label, col_name in soda_cols.items():
            if col_name not in trial_df.columns:
                continue

            vals = trial_df[col_name].values
            spec = compute_power_spectrum(vals, tr_duration=tr_duration)

            prefix = cond_label

            if spec["success"]:
                trial_info[f"{prefix}_band_power"] = band_power(
                    spec["freqs"], spec["power"], f_min, f_max
                )
                trial_info[f"{prefix}_total_power"] = spec["total_power"]
                trial_info[f"{prefix}_peak_freq"] = peak_frequency(
                    spec["freqs"], spec["power"], f_min, f_max
                )
                trial_info[f"{prefix}_relative_band_power"] = relative_band_power(
                    spec["freqs"], spec["power"], f_min, f_max
                )
                # Store full spectrum for averaging later
                trial_info[f"{prefix}_freqs"] = spec["freqs"]
                trial_info[f"{prefix}_power"] = spec["power"]
            else:
                trial_info[f"{prefix}_band_power"] = np.nan
                trial_info[f"{prefix}_total_power"] = np.nan
                trial_info[f"{prefix}_peak_freq"] = np.nan
                trial_info[f"{prefix}_relative_band_power"] = np.nan
                trial_info[f"{prefix}_freqs"] = np.array([])
                trial_info[f"{prefix}_power"] = np.array([])

        # Behavior
        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        # Retrieval position
        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 2b. CLUSTER-BASED PERMUTATION TEST OVER FREQUENCIES
# =====================================================================

def _build_subject_freq_matrix(metrics_df, cond_label):
    """
    Build a (n_subjects x n_freqs) matrix of power spectra, averaged
    across trials within each subject.

    Returns
    -------
    freqs : array (n_freqs,)
    matrix : array (n_subjects, n_freqs)
    subjects : list of subject IDs
    """
    freq_col = f"{cond_label}_freqs"
    power_col = f"{cond_label}_power"

    if freq_col not in metrics_df.columns:
        return None, None, None

    # Group by subject, collect all spectra, average
    subject_spectra = {}
    ref_freqs = None

    for subj, subj_df in metrics_df.groupby("subject"):
        spectra = []
        for _, row in subj_df.iterrows():
            f = row[freq_col]
            p = row[power_col]
            if isinstance(f, np.ndarray) and len(f) > 0:
                if ref_freqs is None:
                    ref_freqs = f
                if len(f) == len(ref_freqs):
                    spectra.append(p)
        if spectra:
            subject_spectra[subj] = np.mean(spectra, axis=0)

    if ref_freqs is None or len(subject_spectra) < 3:
        return None, None, None

    subjects = sorted(subject_spectra.keys())
    matrix = np.array([subject_spectra[s] for s in subjects])

    return ref_freqs, matrix, subjects


def _find_clusters(t_vals, threshold):
    """
    Find contiguous clusters where |t| > threshold.

    Returns list of (cluster_indices, cluster_mass) tuples,
    where cluster_mass = sum of t-values within the cluster.
    """
    above = np.abs(t_vals) > threshold
    clusters = []
    in_cluster = False
    start = 0

    for i in range(len(above)):
        if above[i] and not in_cluster:
            start = i
            in_cluster = True
        elif not above[i] and in_cluster:
            indices = np.arange(start, i)
            mass = np.sum(t_vals[indices])
            clusters.append((indices, mass))
            in_cluster = False

    # Handle cluster extending to end
    if in_cluster:
        indices = np.arange(start, len(above))
        mass = np.sum(t_vals[indices])
        clusters.append((indices, mass))

    return clusters


def cluster_permutation_test(
    metrics_df: pd.DataFrame,
    cond_a: str = "retrieved",
    cond_b: str = "shuf_retrieved",
    n_permutations: int = 5000,
    cluster_threshold: float = 2.0,
    seed: int = 42,
):
    """
    Cluster-based permutation test comparing power spectra between two
    conditions across frequency bins.

    The test is performed at the subject level: for each subject, power
    spectra are first averaged across trials, then paired comparisons
    are made at each frequency bin.

    Procedure (Maris & Oostenveld, 2007):
      1. At each frequency bin, compute a paired t-test between conditions.
      2. Threshold the t-map: find clusters of contiguous bins where |t| > threshold.
      3. For each cluster, compute cluster mass = sum of t-values.
      4. Build null distribution: permute condition labels within subjects
         (swap A and B), recompute t-map, find max cluster mass. Repeat N times.
      5. p-value for each observed cluster = proportion of null max-masses
         that exceed the observed cluster mass.

    Parameters
    ----------
    metrics_df : DataFrame
        Output of compute_trial_level_fft (must contain _freqs and _power columns).
    cond_a, cond_b : str
        Condition labels to compare (e.g., 'retrieved' vs 'shuf_retrieved').
    n_permutations : int
        Number of permutations for null distribution.
    cluster_threshold : float
        T-value threshold for cluster formation. Default 2.0 corresponds
        roughly to p<0.05 uncorrected for moderate sample sizes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        freqs : array of frequency bins
        t_values : array of observed t-values per bin
        clusters : list of dicts, each with 'indices', 'mass', 'p_value'
        null_distribution : array of max cluster masses from permutations
        n_subjects : int
    """
    # Build subject-level frequency x power matrices
    freqs_a, matrix_a, subjects_a = _build_subject_freq_matrix(metrics_df, cond_a)
    freqs_b, matrix_b, subjects_b = _build_subject_freq_matrix(metrics_df, cond_b)

    if freqs_a is None or freqs_b is None:
        print(f"Cannot run cluster test: missing data for {cond_a} or {cond_b}")
        return None

    # Ensure same subjects in both conditions
    shared_subjects = sorted(set(subjects_a) & set(subjects_b))
    if len(shared_subjects) < 3:
        print(f"Too few shared subjects ({len(shared_subjects)}) for cluster test")
        return None

    idx_a = [subjects_a.index(s) for s in shared_subjects]
    idx_b = [subjects_b.index(s) for s in shared_subjects]
    A = matrix_a[idx_a]  # (n_subjects, n_freqs)
    B = matrix_b[idx_b]

    n_subjects, n_freqs = A.shape
    diff = A - B  # (n_subjects, n_freqs)

    # Step 1: Observed t-values at each frequency
    t_obs = np.zeros(n_freqs)
    for f in range(n_freqs):
        t_obs[f], _ = ttest_rel(A[:, f], B[:, f])

    # Step 2-3: Find observed clusters
    obs_clusters = _find_clusters(t_obs, cluster_threshold)
    obs_masses = [abs(mass) for _, mass in obs_clusters]

    print(f"  Cluster test ({cond_a} vs {cond_b}): {n_subjects} subjects, "
          f"{n_freqs} freq bins, {len(obs_clusters)} cluster(s) found")

    if not obs_clusters:
        return {
            "freqs": freqs_a,
            "t_values": t_obs,
            "clusters": [],
            "null_distribution": np.array([]),
            "n_subjects": n_subjects,
            "cond_a": cond_a,
            "cond_b": cond_b,
        }

    # Step 4: Permutation null distribution
    rng = np.random.default_rng(seed)
    null_max_masses = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # For each subject, randomly swap A and B (flip sign of difference)
        flip = rng.choice([-1, 1], size=n_subjects)
        perm_diff = diff * flip[:, np.newaxis]

        # Paired t-test on permuted data
        t_perm = np.zeros(n_freqs)
        for f in range(n_freqs):
            mean_d = np.mean(perm_diff[:, f])
            se_d = np.std(perm_diff[:, f], ddof=1) / np.sqrt(n_subjects)
            t_perm[f] = mean_d / se_d if se_d > 0 else 0.0

        # Find clusters in permuted t-map
        perm_clusters = _find_clusters(t_perm, cluster_threshold)
        if perm_clusters:
            null_max_masses[perm] = max(abs(mass) for _, mass in perm_clusters)
        else:
            null_max_masses[perm] = 0.0

    # Step 5: p-values for observed clusters
    cluster_results = []
    for indices, mass in obs_clusters:
        p_value = np.mean(null_max_masses >= abs(mass))
        cluster_results.append({
            "indices": indices,
            "freqs": freqs_a[indices],
            "mass": mass,
            "p_value": p_value,
            "freq_range": (float(freqs_a[indices[0]]), float(freqs_a[indices[-1]])),
        })
        print(f"    Cluster: freq=[{freqs_a[indices[0]]:.3f}–{freqs_a[indices[-1]]:.3f}], "
              f"mass={mass:.3f}, p={p_value:.4f}")

    return {
        "freqs": freqs_a,
        "t_values": t_obs,
        "clusters": cluster_results,
        "null_distribution": null_max_masses,
        "n_subjects": n_subjects,
        "cond_a": cond_a,
        "cond_b": cond_b,
    }


def plot_cluster_permutation_result(
    result: dict,
    cluster_alpha: float = 0.05,
    save_path: str = None,
):
    """
    Plot the cluster-based permutation test result.

    Top panel: t-values across frequencies with significant clusters highlighted.
    Bottom panel: null distribution of max cluster masses with observed values marked.
    """
    if result is None:
        return

    freqs = result["freqs"]
    t_vals = result["t_values"]
    clusters = result["clusters"]
    null_dist = result["null_distribution"]
    cond_a = result["cond_a"]
    cond_b = result["cond_b"]
    n_sub = result["n_subjects"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])

    # --- Top panel: t-values across frequencies ---
    ax = axes[0]
    ax.plot(freqs, t_vals, "k-", lw=1.5, label="Observed t-values")
    ax.axhline(0, color="grey", ls="-", lw=0.5)

    # Shade significant clusters
    sig_found = False
    for cl in clusters:
        color = "#cc3333" if cl["p_value"] < cluster_alpha else "#cccccc"
        alpha = 0.35 if cl["p_value"] < cluster_alpha else 0.15
        label_text = f"p={cl['p_value']:.3f}" if cl["p_value"] < cluster_alpha else None

        ax.axvspan(
            cl["freqs"][0], cl["freqs"][-1],
            color=color, alpha=alpha, label=label_text,
        )
        if cl["p_value"] < cluster_alpha:
            sig_found = True

    ax.set_xlabel("Frequency (cycles/TR)")
    ax.set_ylabel("t-value (paired)")
    ax.set_title(
        f"Cluster permutation test: {cond_a} vs {cond_b} (n={n_sub})\n"
        f"{len(clusters)} cluster(s), "
        f"{'significant' if sig_found else 'none significant'} at α={cluster_alpha}"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Bottom panel: null distribution ---
    ax = axes[1]
    if len(null_dist) > 0:
        ax.hist(null_dist, bins=50, color="#888888", alpha=0.6, edgecolor="white",
                label="Null distribution (max cluster mass)")

        # Mark observed cluster masses
        for cl in clusters:
            color = "#cc3333" if cl["p_value"] < cluster_alpha else "#444444"
            ax.axvline(abs(cl["mass"]), color=color, ls="--", lw=2,
                       label=f"|mass|={abs(cl['mass']):.2f}, p={cl['p_value']:.3f}")

        ax.set_xlabel("|Max cluster mass|")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No clusters found — no null distribution to show",
                ha="center", va="center", transform=ax.transAxes, color="#888")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# 3. PLOTTING
# =====================================================================

def plot_average_power_spectra(
    metrics_df: pd.DataFrame,
    conditions: list = None,
    f_min: float = 0.08,
    f_max: float = 0.20,
    save_path: str = None,
):
    """
    Plot the average power spectrum across trials for each condition.
    Shows the target frequency band as a shaded region.
    """
    if conditions is None:
        conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]

    cond_styles = {
        "retrieved":      {"color": "#3274A1", "lw": 2.5, "label": "Retrieved"},
        "control":        {"color": "#E1812C", "lw": 2.5, "label": "Control"},
        "shuf_retrieved": {"color": "#8CB8D0", "lw": 1.5, "label": "Retrieved (shuffled)", "ls": "--"},
        "shuf_control":   {"color": "#F0C08A", "lw": 1.5, "label": "Control (shuffled)", "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in conditions:
        freq_col = f"{cond}_freqs"
        power_col = f"{cond}_power"

        if freq_col not in metrics_df.columns:
            continue

        # Collect all spectra that have data
        all_freqs = []
        all_power = []
        for _, row in metrics_df.iterrows():
            f = row[freq_col]
            p = row[power_col]
            if isinstance(f, np.ndarray) and len(f) > 0:
                all_freqs.append(f)
                all_power.append(p)

        if not all_power:
            continue

        # All trials should have same frequency axis if same number of TRs
        # Group by frequency length and average within groups
        # (handles potential variable-length trials)
        ref_freqs = all_freqs[0]
        matching = [(f, p) for f, p in zip(all_freqs, all_power) if len(f) == len(ref_freqs)]

        if not matching:
            continue

        power_matrix = np.array([p for _, p in matching])
        mean_power = np.mean(power_matrix, axis=0)
        sem_power = np.std(power_matrix, axis=0, ddof=1) / np.sqrt(power_matrix.shape[0])

        style = cond_styles.get(cond, {"color": "grey", "lw": 1.5, "label": cond})
        ls = style.get("ls", "-")

        ax.plot(ref_freqs, mean_power, color=style["color"], lw=style["lw"],
                ls=ls, label=style["label"])
        ax.fill_between(ref_freqs, mean_power - sem_power, mean_power + sem_power,
                        color=style["color"], alpha=0.15, linewidth=0)

    # Shade target band
    ax.axvspan(f_min, f_max, color="grey", alpha=0.1, label=f"Target band [{f_min}–{f_max}]")

    ax.set_xlabel("Frequency (cycles/TR)")
    ax.set_ylabel("Power")
    ax.set_title("Average power spectrum of SODA timecourse")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_band_power_boxplots(
    metrics_df: pd.DataFrame,
    metric: str = "band_power",
    save_path: str = None,
):
    """
    Boxplot of band power across 4 conditions, mirroring the 4.7 layout.
    Subject-level averages with individual dots.
    """
    conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]
    col_names = [f"{c}_{metric}" for c in conditions]
    available = [c for c in col_names if c in metrics_df.columns]

    if len(available) < 2:
        print(f"Not enough conditions for boxplot (found {len(available)})")
        return

    # Subject-level means
    subj = metrics_df.groupby("subject")[available].mean().dropna()
    if len(subj) < 3:
        print("Too few subjects for boxplot")
        return

    labels = ["Retrieved", "Control", "Ret (shuf)", "Ctrl (shuf)"]
    colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A"]
    # Only keep those that are available
    plot_data = []
    plot_labels = []
    plot_colors = []
    for col, lab, clr in zip(col_names, labels, colors):
        if col in available:
            plot_data.append(subj[col].values)
            plot_labels.append(lab)
            plot_colors.append(clr)

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Individual subjects as jittered dots
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   color=color, edgecolor="k", linewidth=0.5,
                   s=30, zorder=5, alpha=0.8)

    ax.axhline(0, color="grey", ls="--", lw=0.8)

    # Paired t-tests
    stat_lines = []
    if f"retrieved_{metric}" in available and f"control_{metric}" in available:
        t, p = ttest_rel(subj[f"retrieved_{metric}"], subj[f"control_{metric}"])
        stat_lines.append(f"Ret vs Ctrl: t={t:.2f}, p={p:.3f}")
    if f"retrieved_{metric}" in available and f"shuf_retrieved_{metric}" in available:
        t, p = ttest_rel(subj[f"retrieved_{metric}"], subj[f"shuf_retrieved_{metric}"])
        stat_lines.append(f"Ret vs Ret(shuf): t={t:.2f}, p={p:.3f}")

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel(f"Mean {metric}")
    title = f"FFT {metric} across conditions (n={len(subj)})"
    if stat_lines:
        title += "\n" + " | ".join(stat_lines)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_band_power_by_accuracy(
    metrics_df: pd.DataFrame,
    metric: str = "band_power",
    save_path: str = None,
):
    """
    Compare band power for correct vs incorrect trials.
    """
    ret_col = f"retrieved_{metric}"
    ctrl_col = f"control_{metric}"

    if ret_col not in metrics_df.columns:
        print(f"Column {ret_col} not found")
        return

    df = metrics_df.dropna(subset=[ret_col, "behav_correct"]).copy()
    if df.empty:
        return

    # Subject-level averages split by accuracy
    subj = df.groupby(["subject", "behav_correct"]).agg(
        retrieved_power=(ret_col, "mean"),
        control_power=(ctrl_col, "mean") if ctrl_col in df.columns else (ret_col, "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Correct vs Incorrect
    ax = axes[0]
    for acc_val, color, label in [(1, "#3274A1", "Correct"), (0, "#E1812C", "Incorrect")]:
        vals = subj[subj["behav_correct"] == acc_val]["retrieved_power"]
        if len(vals) > 0:
            ax.bar(label, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0,
                   color=color, alpha=0.7, capsize=5)
            ax.scatter([label] * len(vals), vals, color=color, alpha=0.4, zorder=3, s=20)
    ax.set_ylabel(f"Mean retrieved {metric}")
    ax.set_title(f"Retrieved {metric} by accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Retrieved vs Control for correct trials
    ax = axes[1]
    correct = subj[subj["behav_correct"] == 1]
    if len(correct) > 1:
        for col, color, label in [
            ("retrieved_power", "#3274A1", "Retrieved"),
            ("control_power", "#E1812C", "Control"),
        ]:
            vals = correct[col]
            ax.bar(label, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0,
                   color=color, alpha=0.7, capsize=5)
            ax.scatter([label] * len(vals), vals, color=color, alpha=0.4, zorder=3, s=20)

        if len(correct) >= 3:
            t, p = ttest_rel(correct["retrieved_power"], correct["control_power"])
            ax.set_title(f"Correct trials: Ret vs Ctrl\nt={t:.2f}, p={p:.3f}")
    ax.set_ylabel(f"Mean {metric}")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"FFT {metric} analysis", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_brain_behavior_correlation_fft(
    metrics_df: pd.DataFrame,
    metric: str = "band_power",
    save_path: str = None,
):
    """
    Subject-level correlation between band power and behavioral accuracy.
    """
    ret_col = f"retrieved_{metric}"
    ctrl_col = f"control_{metric}"

    cols_needed = [ret_col, "behav_correct"]
    if not all(c in metrics_df.columns for c in cols_needed):
        return

    df = metrics_df.dropna(subset=cols_needed).copy()
    if df.empty:
        return

    agg_dict = {
        "mean_power": (ret_col, "mean"),
        "mean_accuracy": ("behav_correct", "mean"),
    }
    if ctrl_col in df.columns:
        agg_dict["mean_ctrl_power"] = (ctrl_col, "mean")

    subj = df.groupby("subject").agg(**agg_dict).reset_index()

    if ctrl_col in df.columns:
        subj["power_delta"] = subj["mean_power"] - subj["mean_ctrl_power"]
    else:
        subj["power_delta"] = subj["mean_power"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, xcol, xlabel in zip(
        axes,
        ["mean_power", "power_delta"],
        [f"Mean {ret_col}", f"Mean {ret_col} − control"],
    ):
        x = subj[xcol].values
        y = subj["mean_accuracy"].values

        ax.scatter(x, y, s=60, alpha=0.7, color="#3274A1")

        if len(x) >= 5:
            r, p = pearsonr(x, y)
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, "r--", lw=2)
            ax.set_title(f"r={r:.3f}, p={p:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean behavioral accuracy")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Brain-behavior: FFT {metric} vs accuracy", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectra_by_accuracy(
    metrics_df: pd.DataFrame,
    f_min: float = 0.08,
    f_max: float = 0.20,
    save_path: str = None,
):
    """
    Plot average power spectra split by behavioral accuracy (correct vs incorrect).
    Only for the retrieved condition.
    """
    freq_col = "retrieved_freqs"
    power_col = "retrieved_power"

    if freq_col not in metrics_df.columns or "behav_correct" not in metrics_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for acc_val, color, label in [(1, "#3274A1", "Correct"), (0, "#E1812C", "Incorrect")]:
        subset = metrics_df[metrics_df["behav_correct"] == acc_val]

        all_power = []
        ref_freqs = None
        for _, row in subset.iterrows():
            f = row[freq_col]
            p = row[power_col]
            if isinstance(f, np.ndarray) and len(f) > 0:
                if ref_freqs is None:
                    ref_freqs = f
                if len(f) == len(ref_freqs):
                    all_power.append(p)

        if not all_power or ref_freqs is None:
            continue

        power_matrix = np.array(all_power)
        mean_power = np.mean(power_matrix, axis=0)
        sem_power = np.std(power_matrix, axis=0, ddof=1) / np.sqrt(len(all_power))

        ax.plot(ref_freqs, mean_power, color=color, lw=2, label=f"{label} (n={len(all_power)})")
        ax.fill_between(ref_freqs, mean_power - sem_power, mean_power + sem_power,
                        color=color, alpha=0.15, linewidth=0)

    ax.axvspan(f_min, f_max, color="grey", alpha=0.1, label=f"Target band")
    ax.set_xlabel("Frequency (cycles/TR)")
    ax.set_ylabel("Power")
    ax.set_title("Retrieved SODA power spectrum: correct vs incorrect")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# 4. ENTRY POINT
# =====================================================================

def main():
    # ── Configuration ──
    Pre_post = 'postsleep'
    f_min = 0.08   # lower bound of target band (cycles/TR)
    f_max = 0.20   # upper bound of target band (cycles/TR)
    tr_duration = 1.0  # 1.0 = frequencies in cycles/TR; set to 1.5 for Hz

    # ── Paths ──
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

    for col in ["tr", "retrieved_soda", "control_soda", "behav_correct",
                 "retrieved_soda_shuffled", "control_soda_shuffled"]:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    rois = sorted(master_df['ROI'].unique())
    print(f"Found {len(rois)} ROIs: {rois}")
    print(f"Total rows: {len(master_df)}")
    print(f"Target frequency band: [{f_min}, {f_max}] cycles/TR")

    # ── Loop over ROIs ──
    all_roi_summaries = []

    for roi in rois:
        print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

        df = master_df[master_df['ROI'] == roi].copy()

        out_dir = create_script_named_folder(roi)
        print(f"Output directory: {out_dir}")
        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

        # --- Compute trial-level FFT metrics ---
        print("Computing FFT per trial...")
        metrics_df = compute_trial_level_fft(
            df, f_min=f_min, f_max=f_max, tr_duration=tr_duration
        )
        print(f"Computed metrics for {len(metrics_df)} trials")

        # --- Save trial-level results ---
        # Drop the array columns (freqs/power) for CSV — they're variable length
        save_cols = [c for c in metrics_df.columns
                     if not c.endswith("_freqs") and not c.endswith("_power")]
        out_csv = opj(out_dir, f"{roi}_fft_results.csv")
        metrics_df[save_cols].to_csv(out_csv, index=False)
        print(f"Saved to: {out_csv}")

        # --- Summary stats ---
        conditions = ["retrieved", "control", "shuf_retrieved", "shuf_control"]
        for cond in conditions:
            col = f"{cond}_band_power"
            if col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                print(f"  {cond} band_power: mean={vals.mean():.6f}, median={vals.median():.6f}")

        # --- Plots ---
        print("Generating plots...")

        plot_average_power_spectra(
            metrics_df, f_min=f_min, f_max=f_max,
            save_path=opj(out_dir, f"{roi}_average_power_spectra.png"),
        )

        plot_band_power_boxplots(
            metrics_df, metric="band_power",
            save_path=opj(out_dir, f"{roi}_band_power_boxplot.png"),
        )

        plot_band_power_boxplots(
            metrics_df, metric="total_power",
            save_path=opj(out_dir, f"{roi}_total_power_boxplot.png"),
        )

        plot_band_power_by_accuracy(
            metrics_df, metric="band_power",
            save_path=opj(out_dir, f"{roi}_band_power_by_accuracy.png"),
        )

        plot_spectra_by_accuracy(
            metrics_df, f_min=f_min, f_max=f_max,
            save_path=opj(out_dir, f"{roi}_spectra_by_accuracy.png"),
        )

        plot_brain_behavior_correlation_fft(
            metrics_df, metric="band_power",
            save_path=opj(out_dir, f"{roi}_brain_behavior_correlation_fft.png"),
        )

        # --- Cluster-based permutation tests ---
        print("Running cluster-based permutation tests...")

        roi_summary = {
            "ROI": roi,
            "n_subjects": df["subject"].nunique(),
            "n_trials": len(metrics_df),
            "f_min": f_min,
            "f_max": f_max,
        }

        comparisons = [
            ("retrieved", "control",        "ret_vs_ctrl"),
            ("retrieved", "shuf_retrieved", "ret_vs_shuf_ret"),
            ("control",   "shuf_control",   "ctrl_vs_shuf_ctrl"),
        ]

        cluster_results_all = {}
        for cond_a, cond_b, label in comparisons:
            result = cluster_permutation_test(
                metrics_df, cond_a=cond_a, cond_b=cond_b,
                n_permutations=5000, cluster_threshold=2.0,
            )
            cluster_results_all[label] = result

            plot_cluster_permutation_result(
                result, cluster_alpha=0.05,
                save_path=opj(out_dir, f"{roi}_cluster_perm_{label}.png"),
            )

            # Store significant cluster info in roi_summary
            if result is not None and result["clusters"]:
                sig_clusters = [c for c in result["clusters"] if c["p_value"] < 0.05]
                roi_summary[f"cluster_{label}_n_sig"] = len(sig_clusters)
                if sig_clusters:
                    best = min(sig_clusters, key=lambda c: c["p_value"])
                    roi_summary[f"cluster_{label}_best_p"] = best["p_value"]
                    roi_summary[f"cluster_{label}_best_freq"] = f"{best['freq_range'][0]:.3f}-{best['freq_range'][1]:.3f}"

        # --- Group stats (band power t-tests) ---
        bp_cols = {
            "retrieved": "retrieved_band_power",
            "control": "control_band_power",
            "shuf_retrieved": "shuf_retrieved_band_power",
            "shuf_control": "shuf_control_band_power",
        }
        available_bp = {k: v for k, v in bp_cols.items() if v in metrics_df.columns}

        subj_means = metrics_df.groupby("subject")[list(available_bp.values())].mean().dropna()

        # Paired t-tests
        def _safe_ttest(col_a, col_b, label):
            if col_a in subj_means.columns and col_b in subj_means.columns and len(subj_means) >= 3:
                t, p = ttest_rel(subj_means[col_a], subj_means[col_b])
                print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj_means)})")
                roi_summary[f"t_{label}"] = t
                roi_summary[f"p_{label}"] = p

        _safe_ttest("retrieved_band_power", "control_band_power", "ret_vs_ctrl")
        _safe_ttest("retrieved_band_power", "shuf_retrieved_band_power", "ret_vs_shuf_ret")
        _safe_ttest("control_band_power", "shuf_control_band_power", "ctrl_vs_shuf_ctrl")

        all_roi_summaries.append(roi_summary)
        print(f"\nDone with ROI {roi}.")

    # ── Save cross-ROI summary ──
    summary_dir = create_script_named_folder('summary')
    summary_df = pd.DataFrame(all_roi_summaries)
    summary_path = opj(summary_dir, 'fft_summary_all_rois.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()