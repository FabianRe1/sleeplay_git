# -*- coding: utf-8 -*-

"""Group-level aggregation for 4.4_transfer_visual_classifier_retrieval_presleep outputs.

Loads the saved per-subject decoding timecourses and creates aggregated plots:
  - one line per subject: mean across categories (after averaging trials)
  - one bold line: grand average across subjects

Expected subject-level files (saved by 4.4):
  - sXX_decoding_timecourses_retrievalpresleep_scene_presented_use_overall_classifier1_specific2_Y.npy
  - sXX_decoding_timecourses_retrievalpresleep_object_retrieved_use_overall_classifier1_specific2_Y.npy
  - sXX_decoding_timecourses_retrievalpresleep_scene_retrieved_use_overall_classifier1_specific2_Y.npy
  - sXX_decoding_timecourses_retrievalpresleep_tr_window.npy
"""

import os
import re
import glob
from os.path import join as opj

import numpy as np
from matplotlib import pyplot as plt

try:
	from scipy.stats import ttest_1samp
except Exception as e:
	ttest_1samp = None
	_scipy_import_error = e


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

	# grand average + SEM band
	all_subj = np.vstack(list(subject_to_tc.values()))
	grand_avg = np.mean(all_subj, axis=0)
	if all_subj.shape[0] > 1:
		sem = np.std(all_subj, axis=0, ddof=1) / np.sqrt(all_subj.shape[0])
		plt.fill_between(
			tr_window,
			grand_avg - sem,
			grand_avg + sem,
			color="black",
			alpha=0.15,
			linewidth=0,
			label="±SEM",
		)
	plt.plot(tr_window, grand_avg, color="black", linewidth=5, label="Grand Average")

	plt.axvline(0, color="red", linestyle="--")
	plt.xlabel("TRs from onset + 5 secs HRF lag")
	plt.ylabel("Decoding Probability (Target Class)")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()


def _load_tr_window(data_dir: str) -> np.ndarray:
	"""Try to load a tr_window saved by 4.4; otherwise fall back to -4..+4."""
	tr_files = sorted(glob.glob(opj(data_dir, "*_decoding_timecourses_retrievalpresleep_tr_window.npy")))
	if len(tr_files) == 0:
		return np.arange(-4, 5)
	tr = np.load(tr_files[0], allow_pickle=True)
	# handle both np.ndarray and 0-d object array
	if hasattr(tr, "item") and tr.shape == ():
		tr = tr.item()
	tr = np.asarray(tr)
	if tr.ndim != 1:
		raise ValueError(f"Unexpected tr_window shape in {tr_files[0]}: {tr.shape}")
	return tr


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
	"""Benjamini–Hochberg FDR correction (returns q-values)."""
	p = np.asarray(p_values, dtype=float)
	if p.ndim != 1:
		p = p.ravel()
	n = p.size
	order = np.argsort(p)
	ranked = p[order]
	q = ranked * n / (np.arange(1, n + 1))
	# enforce monotonicity
	q = np.minimum.accumulate(q[::-1])[::-1]
	q = np.clip(q, 0.0, 1.0)
	out = np.empty(n, dtype=float)
	out[order] = q
	return out


def _pvals_vs_chance_greater(all_subj: np.ndarray, chance: float) -> np.ndarray:
	"""One-sample t-test at each timepoint (H1: mean > chance)."""
	if ttest_1samp is None:
		raise ImportError(
			"scipy is required for per-timepoint tests (ttest_1samp). "
			f"Import error: {_scipy_import_error}"
		)

	# Newer SciPy supports alternative='greater'
	try:
		res = ttest_1samp(all_subj, popmean=chance, axis=0, alternative="greater")
		return np.asarray(res.pvalue, dtype=float)
	except TypeError:
		# Fallback: convert two-sided p to one-sided (greater)
		res = ttest_1samp(all_subj, popmean=chance, axis=0)
		t = np.asarray(res.statistic, dtype=float)
		p_two = np.asarray(res.pvalue, dtype=float)
		p_one = np.where(t > 0, p_two / 2.0, 1.0 - (p_two / 2.0))
		return p_one


def _print_timepoint_stats(label: str, tr_window: np.ndarray, p_raw: np.ndarray, q_fdr: np.ndarray) -> None:
	print("\n" + "=" * 80)
	print(f"{label}: per-TR test vs chance=0.10 (H1: mean > chance)")
	print("TR\tp_raw\tq_fdr")
	for tr, p, q in zip(tr_window, p_raw, q_fdr):
		try:
			tr_int = int(tr)
			tr_str = f"{tr_int:+d}"
		except Exception:
			tr_str = str(tr)
		print(f"{tr_str}\t{p:.6g}\t{q:.6g}")


def main() -> None:
	# Match how the subject-level script named the files
	use_overall_classifier1_specific2 = 2

	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = opj(
		script_dir,
		"4.4_transfer_visual_classifier_retrieval_presleep",
		"decoding_timecourses_data",
	)
	out_dir = _ensure_dir(
		opj(
			script_dir,
			"4.4_transfer_visual_classifier_retrieval_presleep_grouplevel",
			"decoding_timecourses_plots",
		)
	)

	if not os.path.exists(data_dir):
		raise FileNotFoundError(
			"Could not find saved decoding outputs. Expected folder: "
			f"{data_dir}"
		)

	tr_window = _load_tr_window(data_dir)

	# Regex parse for subject + modality
	pat = re.compile(
		r"^(s\d+)_decoding_timecourses_retrievalpresleep_"
		r"(scene_presented|object_retrieved|scene_retrieved)_"
		r"use_overall_classifier1_specific2_(\d+)\.npy$"
	)

	files_scene_presented = []
	files_object_retrieved = []
	files_scene_retrieved = []
	for fpath in glob.glob(opj(data_dir, "*.npy")):
		fname = os.path.basename(fpath)
		m = pat.match(fname)
		if not m:
			continue
		modality, clf_type = m.group(2), int(m.group(3))
		if clf_type != use_overall_classifier1_specific2:
			continue
		if modality == "scene_presented":
			files_scene_presented.append(fpath)
		elif modality == "object_retrieved":
			files_object_retrieved.append(fpath)
		elif modality == "scene_retrieved":
			files_scene_retrieved.append(fpath)

	if (
		len(files_scene_presented) == 0
		and len(files_object_retrieved) == 0
		and len(files_scene_retrieved) == 0
	):
		raise FileNotFoundError(
			"No matching decoding npy files found in "
			f"{data_dir} for use_overall_classifier1_specific2={use_overall_classifier1_specific2}"
		)

	def _run_one_modality(files: list[str], title: str, out_prefix: str) -> None:
		if len(files) == 0:
			return
		subject_to_tc = {}
		for fpath in sorted(files):
			subj = pat.match(os.path.basename(fpath)).group(1)
			decoding_dict = _load_decoding_dict(fpath)
			tc = _subject_mean_timecourse(decoding_dict)
			if tc.shape[0] != tr_window.shape[0]:
				raise ValueError(
					f"Unexpected timecourse length in {fpath}: got {tc.shape[0]}, expected {tr_window.shape[0]}"
				)
			subject_to_tc[subj] = tc

		# Per-timepoint stats vs chance (across subjects)
		chance = 0.10
		all_subj = np.vstack(list(subject_to_tc.values()))
		if all_subj.shape[0] < 2:
			print(f"\n{out_prefix}: only {all_subj.shape[0]} subject(s); skipping t-tests.")
		else:
			p_raw = _pvals_vs_chance_greater(all_subj, chance=chance)
			q_fdr = _fdr_bh(p_raw)
			_print_timepoint_stats(out_prefix, tr_window, p_raw, q_fdr)

		_plot_subject_lines(
			subject_to_tc=subject_to_tc,
			tr_window=tr_window,
			title=title,
			out_png=opj(
				out_dir,
				f"grouplevel_{out_prefix}_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.png",
			),
		)

		np.save(
			opj(
				out_dir,
				f"grouplevel_subject_mean_timecourses_{out_prefix}_use_overall_classifier1_specific2_{use_overall_classifier1_specific2}.npy",
			),
			subject_to_tc,
		)

	_run_one_modality(
		files=files_scene_presented,
		title=(
			"Retrieval presleep: Scene presented\n"
			f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
		),
		out_prefix="decoding_timecourses_retrievalpresleep_scene_presented",
	)

	_run_one_modality(
		files=files_object_retrieved,
		title=(
			"Retrieval presleep: Object retrieved\n"
			f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
		),
		out_prefix="decoding_timecourses_retrievalpresleep_object_retrieved",
	)

	_run_one_modality(
		files=files_scene_retrieved,
		title=(
			"Retrieval presleep: Scene retrieved\n"
			f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}"
		),
		out_prefix="decoding_timecourses_retrievalpresleep_scene_retrieved",
	)

	# Save the TR window used for plotting
	np.save(
		opj(
			out_dir,
			"grouplevel_decoding_timecourses_retrievalpresleep_tr_window.npy",
		),
		tr_window,
	)


if __name__ == "__main__":
	main()
