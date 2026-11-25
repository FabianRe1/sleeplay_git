#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B6b - Encoding-session decoding, group level.

Collects the per-participant decoding time courses of B6a and plots them per
ROI, separately for scenes and objects: one line per participant (mean over
categories, each averaged over its trials) and the grand average.

Input   outputs/B6a_apply_classifiers_encoding/decoding_timecourses_data/*.npy
Output  outputs/B6b_encoding_decoding_group/group_timecourses_data/*.npy
            (the plotted per-participant time courses)
        outputs/B6b_encoding_decoding_group/encoding_decoding_summary.csv
            per ROI and classifier: peak inside the peak window and mean over
            the window, each tested against chance (10 percent), with Holm
            correction across ROIs
Plots   plots/B6b_encoding_decoding_group/*.png, *.pdf
        plots/B6b_encoding_decoding_group/fig_encoding_time_course_{obj,scene}.{png,pdf}
            paper figure, ROIs of MAIN_ROIS: one thin line per participant (mean over
            categories, each averaged over its trials), the group mean, and the 95 %
            confidence interval across participants as shading. Time is in seconds
            from the real picture onset. The peak inside the peak window carries
            stars (Holm-corrected across these ROIs: * < .05, ** < .01,
            *** < .001) and the test is written next to it. Added 2026-11-25.

Ported from 4_decoding_claude_retrieval/4.3_transfer_visual_classifier_encoding_grouplevel.py;
with two changes. (1) The peak that is tested against chance is searched only
2 to 6 TRs after the real stimulus onset (stored TR -1 to +2, because the
stored time courses are relative to the onset plus 5 s). The original took
the highest of all nine TRs, which could fall before the stimulus. (2) The
summary table is new. One difference in file handling: the original also copied
each plot into `1_thesis_chapter_plots/`. This step writes only to its own
output folder.

Usage:  python B6b_encoding_decoding_group.py
"""

import os
import re
import glob
from os.path import join as opj
from collections import defaultdict

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HRF_DELAY, step_input_dir, step_output_dir, step_plot_dir

TR = 1.5
# The classifier peak is searched 2 to 6 TRs after the real stimulus onset.
# The time courses are stored relative to the onset shifted by HRF_DELAY
# (5 s = 3.33 TRs), so this is stored TR -1 to +2.
PEAK_WINDOW_TRS_AFTER_ONSET = (2, 6)


def _peak_window_mask(tr_window: np.ndarray) -> np.ndarray:
	trs_after_onset = tr_window + HRF_DELAY / TR
	lo, hi = PEAK_WINDOW_TRS_AFTER_ONSET
	return (trs_after_onset >= lo) & (trs_after_onset <= hi)


def _peak_stats(data: np.ndarray, tr_window: np.ndarray, chance_level: float) -> dict:
	"""Peak of the group mean inside the peak window, tested against chance.

	Also tests the mean over the whole window, which involves no selection
	of a time point.
	"""
	from scipy.stats import ttest_1samp
	in_window = _peak_window_mask(tr_window)
	mean_tc = np.nanmean(data, axis=0)
	peak_idx = int(np.flatnonzero(in_window)[np.nanargmax(mean_tc[in_window])])
	peak_values = data[:, peak_idx]
	peak_values = peak_values[~np.isnan(peak_values)]
	t_peak, p_peak = ttest_1samp(peak_values, chance_level)
	window_values = np.nanmean(data[:, in_window], axis=1)
	window_values = window_values[~np.isnan(window_values)]
	t_win, p_win = ttest_1samp(window_values, chance_level)
	return dict(
		n=data.shape[0], peak_idx=peak_idx, peak_tr=int(tr_window[peak_idx]),
		peak_tr_after_onset=round(float(tr_window[peak_idx] + HRF_DELAY / TR), 2),
		peak_mean=float(mean_tc[peak_idx]), t_peak=float(t_peak), p_peak=float(p_peak),
		window_mean=float(window_values.mean()), t_window=float(t_win), p_window=float(p_win),
	)


MAIN_ROIS = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL",
			 "hpc": "Hippocampus", "somato": "Somatosensory (control)", "caudal_acc": "Caudal ACC (control)"}


def _stars(p: float) -> str:
	return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _p_text(p: float) -> str:
	return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def plot_main_figure(tcs: dict, tr_window: np.ndarray, modality: str, summary, out_dir: str, chance: float = 0.1) -> None:
	"""Paper figure for one classifier: participants, group mean and 95 % CI per main ROI."""
	from scipy import stats as sstats
	rois = [r for r in MAIN_ROIS if (r, modality) in tcs]
	if not rois:
		return
	t_s = tr_window * TR + HRF_DELAY                       # seconds from the real onset
	lo, hi = PEAK_WINDOW_TRS_AFTER_ONSET
	colour = "#2a78d6" if modality == "obj" else "#1baf7a"
	ncol = 3 if len(rois) > 4 else len(rois)
	nrow = int(np.ceil(len(rois) / ncol))
	fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow), squeeze=False)
	for ax in axes.ravel()[len(rois):]:
		ax.set_visible(False)
	for ax, roi in zip(axes.ravel(), rois):
		data = np.vstack(list(tcs[(roi, modality)].values())) * 100.0
		n = data.shape[0]
		mean = np.nanmean(data, axis=0)
		ci = sstats.t.ppf(0.975, n - 1) * np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
		ax.axvspan(lo * TR, hi * TR, color="#e4e3df", alpha=0.5, lw=0, zorder=0)
		for tc in data:
			ax.plot(t_s, tc, color="#52514e", alpha=0.22, lw=0.7, zorder=1)
		ax.fill_between(t_s, mean - ci, mean + ci, color=colour, alpha=0.35, lw=0, zorder=2)
		ax.plot(t_s, mean, color=colour, lw=2.6, zorder=3)
		ax.axhline(chance * 100, color="#0b0b0b", ls=":", lw=1, zorder=1)
		ax.axvline(0, color="#0b0b0b", lw=0.6, zorder=1)
		row = summary[(summary.roi == roi) & (summary.classifier == modality)].iloc[0]
		pk = int(np.flatnonzero(tr_window == row.peak_tr)[0])
		top = np.nanmax(data)
		st = _stars(row.p_peak_holm_main)
		ax.text(t_s[pk], mean[pk] + ci[pk] + 0.02 * (top - chance * 100), st, ha="center", va="bottom",
				fontsize=13 if st != "n.s." else 8, fontweight="bold", color="#0b0b0b", zorder=5)
		ax.text(0.98, 0.97, f"peak {row.peak_mean * 100:.1f} % at {row.peak_tr_after_onset * TR:.0f} s\n"
				f"t({n - 1}) = {row.t_peak:.1f}, {_p_text(row.p_peak_holm_main)}\n"
				f"window mean {row.window_mean * 100:.1f} %\nt({n - 1}) = {row.t_window:.1f}, {_p_text(row.p_window_holm_main)}",
				transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#0b0b0b")
		ax.set_title(f"{MAIN_ROIS[roi]} (N = {n})", loc="left", fontsize=10, fontweight="bold")
		ax.spines[["top", "right"]].set_visible(False)
	for ax in axes[-1]:
		ax.set_xlabel("Time from picture onset (s)")
	fig.supylabel("Probability of the shown " + ("object category" if modality == "obj" else "scene") + " (%)", fontsize=11)
	fig.tight_layout()
	for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
		fig.savefig(opj(out_dir, f"fig_encoding_time_course_{modality}.{ext}"), **kw)
	plt.close(fig)


def _ensure_dir(path: str) -> str:
	if not os.path.exists(path):
		os.makedirs(path)
	return path


def _load_decoding_dict(npy_path: str) -> dict:
	"""Loads the dict saved by np.save(..., decoding_results)."""
	data = np.load(npy_path, allow_pickle=True)
	# np.save of a dict yields a 0-d array with dtype=object
	if hasattr(data, "item"):
		return data.item()
	raise ValueError(f"Unexpected npy format for {npy_path}")


def _subject_mean_timecourse(decoding_dict: dict) -> np.ndarray:
	"""Mean across categories, where each category is averaged across its trials."""
	category_means = []
	for _, trial_timecourses in decoding_dict.items():
		if trial_timecourses is None:
			continue
		if len(trial_timecourses) == 0:
			continue
		category_means.append(np.nanmean(np.asarray(trial_timecourses, dtype=float), axis=0))
	if len(category_means) == 0:
		raise ValueError("No trials found in decoding dict")
	return np.nanmean(np.vstack(category_means), axis=0)


def _plot_subject_lines(
	subject_to_tc: dict,
	tr_window: np.ndarray,
	title: str,
	out_png: str,
	chance_level: float = None,
) -> None:
	from scipy.stats import sem as scipy_sem, ttest_1samp
	import matplotlib as mpl

	mpl.rcParams.update({
		"font.family": "sans-serif",
		"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
		"font.size": 11,
		"axes.titlesize": 12,
		"axes.titleweight": "bold",
		"axes.labelsize": 12,
		"xtick.labelsize": 11,
		"ytick.labelsize": 11,
		"legend.fontsize": 10,
		"axes.linewidth": 0.8,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})

	data = np.vstack(list(subject_to_tc.values()))  # (n_subjects, n_trs)
	grand_avg = np.nanmean(data, axis=0)
	sem_tc = scipy_sem(data, axis=0, nan_policy="omit")

	fig, ax = plt.subplots(figsize=(5.0, 4.0))

	# individual subjects (alpha=0.3)
	for subj, tc in sorted(subject_to_tc.items()):
		ax.plot(tr_window, tc, color="gray", alpha=0.3, linewidth=0.8)

	# SEM shading around grand average
	ax.fill_between(
		tr_window, grand_avg - sem_tc, grand_avg + sem_tc,
		alpha=0.25, color="black",
	)
	# grand average
	ax.plot(tr_window, grand_avg, color="black", linewidth=3, label="Grand Average")

	# Chance level
	if chance_level is not None:
		ax.axhline(chance_level, color="gray", linestyle=":", linewidth=1, label=f"Chance ({chance_level:.2f})")

	# One-sample t-test at peak TR vs chance — place star above highest subject line
	if chance_level is not None:
		st = _peak_stats(data, tr_window, chance_level)
		peak_idx, peak_tr, p_val = st["peak_idx"], st["peak_tr"], st["p_peak"]
		if p_val < 0.05:
			# Place star above the highest data point at peak TR
			max_at_peak = np.nanmax(data[:, peak_idx])
			star_y = max(max_at_peak, grand_avg[peak_idx] + sem_tc[peak_idx]) + sem_tc[peak_idx] * 0.5
			ax.text(
				peak_tr, star_y,
				"*", fontsize=24, fontweight="bold", color="darkred",
				ha="center", va="bottom",
			)

	ax.set_xlabel("Cue onset + 5 sec (TRs)")
	ax.set_ylabel("Probability target class")
	ax.set_title(f"{title}\n(N = {len(subject_to_tc)})")
	ax.legend(loc="upper right", frameon=False)
	fig.tight_layout()
	fig.savefig(out_png, dpi=300)
	# Also save as PDF
	fig.savefig(out_png.replace(".png", ".pdf"))

	plt.close(fig)


def _plot_aggregate(
	subject_to_tc: dict,
	tr_window: np.ndarray,
	title: str,
	out_png: str,
	chance_level: float = None,
) -> None:
	"""Aggregate plot: mean +/- SEM across subjects with shaded error band."""
	from scipy.stats import sem as scipy_sem, ttest_1samp

	data = np.vstack(list(subject_to_tc.values()))  # (n_subjects, n_trs)
	n_subj = data.shape[0]
	mean_tc = np.nanmean(data, axis=0)
	sem_tc = scipy_sem(data, axis=0, nan_policy="omit")

	fig, ax = plt.subplots(figsize=(8, 5))

	# SEM shading
	ax.fill_between(
		tr_window, mean_tc - sem_tc, mean_tc + sem_tc,
		alpha=0.3, color="steelblue",
	)
	# Mean line
	ax.plot(tr_window, mean_tc, color="steelblue", linewidth=3, label="Mean")

	# Chance level
	if chance_level is not None:
		ax.axhline(chance_level, color="gray", linestyle=":", linewidth=1, label=f"Chance ({chance_level:.2f})")

		# One-sample t-test at peak TR vs chance
		st = _peak_stats(data, tr_window, chance_level)
		peak_idx, peak_tr, p_val = st["peak_idx"], st["peak_tr"], st["p_peak"]
		if p_val < 0.05:
			star_y = mean_tc[peak_idx] + sem_tc[peak_idx] * 1.5
			ax.text(
				peak_tr, star_y,
				"*", fontsize=28, fontweight="bold", color="darkred",
				ha="center", va="bottom",
			)

	ax.set_xlabel("TRs from Onset (+5 s HRF delay)")
	ax.set_ylabel("Decoding Probability (Target Class)")
	ax.set_title(f"{title}\n(N = {n_subj})")
	ax.legend(loc="upper right")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_png, dpi=150)
	plt.close(fig)


def main() -> None:
	# Match how the subject-level script named the files
	use_overall_classifier1_specific2 = 2

	data_dir = str(step_input_dir("B6a_apply_classifiers_encoding", "decoding_timecourses_data"))
	out_dir = str(step_plot_dir(__file__))
	data_out_dir = str(step_output_dir(__file__, "group_timecourses_data"))

	# ----------------------------------------------------------------
	# Regex: capture subject, optional ROI name, modality, classifier
	# Filenames are either:
	#   s13_decoding_timecourses_scene_use_overall_classifier1_specific2_2.npy
	#   s13_vis_decoding_timecourses_scene_use_overall_classifier1_specific2_2.npy
	# ----------------------------------------------------------------
	pat = re.compile(
		r"^(s\d+)_(?:(.+?)_)?decoding_timecourses_(scene|obj)_use_overall_classifier1_specific2_(\d+)\.npy$"
	)

	# Collect files keyed by (roi, modality)
	# roi = None for the bare (no-ROI) files
	files_by_roi_modality = defaultdict(list)

	for fpath in glob.glob(opj(data_dir, "*.npy")):
		fname = os.path.basename(fpath)
		m = pat.match(fname)
		if not m:
			continue
		subj, roi, modality, clf_type = m.group(1), m.group(2), m.group(3), int(m.group(4))
		if clf_type != use_overall_classifier1_specific2:
			continue
		if roi is None:
			roi = "no_roi"
		files_by_roi_modality[(roi, modality)].append((subj, fpath))

	if len(files_by_roi_modality) == 0:
		raise FileNotFoundError(
			"No matching decoding npy files found in "
			f"{data_dir} for use_overall_classifier1_specific2={use_overall_classifier1_specific2}"
		)

	# Report what was found
	all_rois = sorted(set(roi for roi, _ in files_by_roi_modality.keys()))
	print(f"[grouplevel] Found ROIs: {all_rois}")
	for key, flist in sorted(files_by_roi_modality.items()):
		print(f"  {key}: {len(flist)} subjects")

	# Infer TR window from saved timecourse length (should be 9 for -4..+4)
	tr_window = np.arange(-4, 5)

	stats_rows = []
	all_tcs = {}

	# Process each ROI x modality combination
	for (roi, modality), file_list in sorted(files_by_roi_modality.items()):
		roi_label = roi.replace("_", " ").title() if roi != "no_roi" else "Overall"
		modality_label = "Scenes" if modality == "scene" else "Objects"

		subject_to_tc = {}
		for subj, fpath in sorted(file_list):
			try:
				decoding_dict = _load_decoding_dict(fpath)
				tc = _subject_mean_timecourse(decoding_dict)
			except Exception as e:
				print(f"  [WARNING] Skipping {os.path.basename(fpath)}: {e}")
				continue
			if tc.shape[0] != tr_window.shape[0]:
				print(
					f"  [WARNING] Unexpected timecourse length in {os.path.basename(fpath)}: "
					f"got {tc.shape[0]}, expected {tr_window.shape[0]}. Skipping."
				)
				continue
			subject_to_tc[subj] = tc

		if len(subject_to_tc) == 0:
			print(f"  [WARNING] No valid subjects for roi={roi}, modality={modality}. Skipping.")
			continue

		print(f"  [roi={roi}, modality={modality}] {len(subject_to_tc)} subjects loaded.")

		safe_roi = roi  # already filesystem-safe
		png_name = (
			f"grouplevel_{safe_roi}_decoding_timecourses_{modality}s"
			f"_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png"
		)
		npy_name = (
			f"grouplevel_{safe_roi}_subject_mean_timecourses_{modality}s"
			f"_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy"
		)

		# Chance = 1/n_classes: 10 scenes, 10 object categories
		chance = 1.0 / 10.0

		_plot_subject_lines(
			subject_to_tc=subject_to_tc,
			tr_window=tr_window,
			title=(
				f"{roi_label} – {modality_label}: Mean across categories per subject\n"
				f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
			),
			out_png=opj(out_dir, png_name),
			chance_level=chance,
		)

		# Aggregate plot (mean +/- SEM)
		agg_png_name = (
			f"aggregate_{safe_roi}_decoding_timecourses_{modality}s"
			f"_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png"
		)
		_plot_aggregate(
			subject_to_tc=subject_to_tc,
			tr_window=tr_window,
			title=(
				f"{roi_label} – {modality_label}: Aggregate across subjects\n"
				f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
			),
			out_png=opj(out_dir, agg_png_name),
			chance_level=chance,
		)

		np.save(opj(data_out_dir, npy_name), subject_to_tc)
		all_tcs[(roi, modality)] = subject_to_tc

		row = dict(roi=roi, classifier=modality, chance=chance)
		row.update(_peak_stats(np.vstack(list(subject_to_tc.values())), tr_window, chance))
		row.pop("peak_idx")
		stats_rows.append(row)

	# Summary table; Holm correction across ROIs within classifier
	import pandas as pd
	from statsmodels.stats.multitest import multipletests
	summary = pd.DataFrame(stats_rows)
	for col in ("p_peak", "p_window"):
		summary[col + "_holm"] = np.nan
		for _, idx in summary.groupby("classifier").groups.items():
			summary.loc[idx, col + "_holm"] = multipletests(summary.loc[idx, col], method="holm")[1]
	# the same within the ROIs of MAIN_ROIS (the paper figure)
	main = summary[summary.roi.isin(MAIN_ROIS)]
	for col in ("p_peak", "p_window"):
		summary[col + "_holm_main"] = np.nan
		for _, idx in main.groupby("classifier").groups.items():
			summary.loc[idx, col + "_holm_main"] = multipletests(summary.loc[idx, col], method="holm")[1]
	for modality in ("obj", "scene"):
		plot_main_figure(all_tcs, tr_window, modality, summary, out_dir)
	summary_path = opj(str(step_output_dir(__file__)), "encoding_decoding_summary.csv")
	summary.to_csv(summary_path, index=False)
	print(f"[grouplevel] Saved {summary_path}")


if __name__ == "__main__":
	main()

