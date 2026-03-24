
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
		category_means.append(np.mean(np.asarray(trial_timecourses), axis=0))
	if len(category_means) == 0:
		raise ValueError("No trials found in decoding dict")
	return np.mean(np.vstack(category_means), axis=0)


def _plot_subject_lines(
	subject_to_tc: dict,
	tr_window: np.ndarray,
	title: str,
	out_png: str,
) -> None:
	plt.figure(figsize=(10, 6))

	# individual subjects
	for subj, tc in sorted(subject_to_tc.items()):
		plt.plot(tr_window, tc, color="gray", alpha=0.6, linewidth=2, label=subj)

	# grand average
	grand_avg = np.mean(np.vstack(list(subject_to_tc.values())), axis=0)
	plt.plot(tr_window, grand_avg, color="black", linewidth=5, label="Grand Average")

	plt.axvline(0, color="red", linestyle="--")
	plt.xlabel("Onset +5seconds (TRs)")
	plt.ylabel("Decoding Probability (Target Class)")
	plt.title(f"{title}\n(N = {len(subject_to_tc)})")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()


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

		_plot_subject_lines(
			subject_to_tc=subject_to_tc,
			tr_window=tr_window,
			title=(
				f"{roi_label} – {modality_label}: Mean across categories per subject\n"
				f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
			),
			out_png=opj(out_dir, png_name),
		)

		np.save(opj(out_dir, npy_name), subject_to_tc)


if __name__ == "__main__":
	main()

