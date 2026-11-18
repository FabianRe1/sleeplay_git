
# -*- coding: utf-8 -*-

"""Group-level aggregation for 4.3_transfer_visual_classifier_encoding outputs.

Loads the saved per-subject decoding timecourses (scene + object) and creates
aggregated plots **per ROI**:
  - one line per subject: mean across categories (after averaging trials)
  - one bold line: grand average across subjects
"""

import os
import re
import glob
from os.path import join as opj
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt


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
		peak_idx = np.nanargmax(grand_avg)
		peak_tr = tr_window[peak_idx]
		peak_values = data[:, peak_idx]
		peak_values_clean = peak_values[~np.isnan(peak_values)]
		t_stat, p_val = ttest_1samp(peak_values_clean, chance_level)
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
	ax.set_ylabel("Probability target class (%)")
	ax.set_title(f"{title}\n(N = {len(subject_to_tc)})")
	ax.legend(loc="upper right", frameon=False)
	fig.tight_layout()
	fig.savefig(out_png, dpi=300)
	# Also save PDF for thesis
	fig.savefig(out_png.replace(".png", ".pdf"))

	# Save to thesis chapter plots Fig3 directory + png/pdf folders
	import shutil
	thesis_base = os.path.normpath(os.path.join(
		os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
		"1_thesis_chapter_plots",
	))
	thesis_fig3_dir = opj(thesis_base, "Fig3")
	thesis_png_dir = opj(thesis_base, "png")
	thesis_pdf_dir = opj(thesis_base, "pdf")
	basename = os.path.basename(out_png)
	pdf_name = basename.replace(".png", ".pdf")
	for d, fname in [(thesis_fig3_dir, basename), (thesis_fig3_dir, pdf_name),
	                  (thesis_png_dir, basename), (thesis_pdf_dir, pdf_name)]:
		if os.path.isdir(d):
			src = out_png if fname.endswith(".png") else out_png.replace(".png", ".pdf")
			shutil.copy2(src, opj(d, fname))
	print(f"  Copied to thesis Fig3 + png/pdf: {basename}")

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
		peak_idx = np.nanargmax(mean_tc)
		peak_tr = tr_window[peak_idx]
		peak_values = data[:, peak_idx]
		peak_values_clean = peak_values[~np.isnan(peak_values)]
		t_stat, p_val = ttest_1samp(peak_values_clean, chance_level)
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

	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = opj(
		script_dir,
		"4.3_transfer_visual_classifier_encoding",
		"decoding_timecourses_data",
	)
	out_dir = _ensure_dir(
		opj(
			script_dir,
			"4.3_transfer_visual_classifier_encoding_grouplevel",
			"decoding_timecourses_plots",
		)
	)

	if not os.path.exists(data_dir):
		raise FileNotFoundError(
			"Could not find saved decoding outputs. Expected folder: "
			f"{data_dir}"
		)

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

		np.save(opj(out_dir, npy_name), subject_to_tc)


if __name__ == "__main__":
	main()

