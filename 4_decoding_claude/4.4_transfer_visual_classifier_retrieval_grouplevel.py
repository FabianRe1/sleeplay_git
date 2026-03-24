# -*- coding: utf-8 -*-
"""Group-level aggregation for 4.4 retrieval decoding outputs (presleep & postsleep).

Unifies the original 4.4_..._presleep_grouplevel.py (and the missing postsleep
grouplevel) into a single parametrised script.

Usage
-----
    python 4.4_transfer_visual_classifier_retrieval_grouplevel.py [presleep|postsleep]
"""

import os
import re
import sys
import glob
from os.path import join as opj

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

from utils import ensure_dir, fdr_bh


# =====================================================================
# HELPERS
# =====================================================================

def _load_decoding_dict(npy_path: str) -> dict:
    """Load a dict saved by ``np.save(..., decoding_results)``."""
    data = np.load(npy_path, allow_pickle=True)
    if hasattr(data, "item"):
        return data.item()
    raise ValueError(f"Unexpected npy format for {npy_path}")


def _subject_mean_timecourse(decoding_dict: dict) -> np.ndarray:
    """Mean across categories, where each category is averaged across its trials."""
    category_means = []
    for trial_timecourses in decoding_dict.values():
        if trial_timecourses is None or len(trial_timecourses) == 0:
            continue
        category_means.append(np.nanmean(np.asarray(trial_timecourses), axis=0))
    if len(category_means) == 0:
        raise ValueError("No trials found in decoding dict")
    return np.nanmean(np.vstack(category_means), axis=0)


def _pvals_vs_chance_greater(all_subj: np.ndarray, chance: float) -> np.ndarray:
    """One-sample t-test at each timepoint (H1: mean > chance)."""
    try:
        res = ttest_1samp(all_subj, popmean=chance, axis=0, alternative="greater")
        return np.asarray(res.pvalue, dtype=float)
    except TypeError:
        res = ttest_1samp(all_subj, popmean=chance, axis=0)
        t = np.asarray(res.statistic, dtype=float)
        p_two = np.asarray(res.pvalue, dtype=float)
        return np.where(t > 0, p_two / 2.0, 1.0 - (p_two / 2.0))


def _print_timepoint_stats(
    label: str, tr_window: np.ndarray, p_raw: np.ndarray, q_fdr: np.ndarray
) -> None:
    print("\n" + "=" * 80)
    print(f"{label}: per-TR test vs chance=0.10 (H1: mean > chance)")
    print("TR\tp_raw\tq_fdr")
    for tr, p, q in zip(tr_window, p_raw, q_fdr):
        try:
            tr_str = f"{int(tr):+d}"
        except Exception:
            tr_str = str(tr)
        print(f"{tr_str}\t{p:.6g}\t{q:.6g}")


def _plot_subject_lines(
    subject_to_tc: dict,
    tr_window: np.ndarray,
    title: str,
    out_png: str,
) -> None:
    plt.figure(figsize=(10, 6))

    for subj, tc in sorted(subject_to_tc.items()):
        plt.plot(tr_window, tc, color="gray", alpha=0.6, linewidth=2, label=subj)

    all_subj = np.vstack(list(subject_to_tc.values()))
    grand_avg = np.nanmean(all_subj, axis=0)
    if all_subj.shape[0] > 1:
        sem = np.nanstd(all_subj, axis=0, ddof=1) / np.sqrt(all_subj.shape[0])
        plt.fill_between(
            tr_window, grand_avg - sem, grand_avg + sem,
            color="black", alpha=0.15, linewidth=0, label="±SEM",
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


def _load_tr_window(data_dir: str, pre_post: str) -> np.ndarray:
    """Try to load a tr_window saved by 4.4; otherwise fall back to defaults."""
    tag = "retrievalpresleep" if pre_post == "presleep" else "retrievalpostsleep"
    tr_files = sorted(glob.glob(opj(data_dir, f"*_decoding_timecourses_{tag}_tr_window.npy")))
    if len(tr_files) == 0:
        raise FileNotFoundError(
            f"No tr_window file found in {data_dir} for {pre_post}. "
            f"Run 4.4_transfer_visual_classifier_retrieval.py first."
        )
    tr = np.load(tr_files[0], allow_pickle=True)
    if hasattr(tr, "item") and tr.shape == ():
        tr = tr.item()
    tr = np.asarray(tr)
    if tr.ndim != 1:
        raise ValueError(f"Unexpected tr_window shape: {tr.shape}")
    return tr


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    pre_post = sys.argv[1] if len(sys.argv) > 1 else "presleep"
    if pre_post not in ("presleep", "postsleep"):
        raise ValueError(f"Expected 'presleep' or 'postsleep', got '{pre_post}'")

    use_overall_classifier1_specific2 = 2

    if pre_post == "presleep":
        data_parent = "4.4_transfer_visual_classifier_retrieval_presleep"
        out_parent = "4.4_transfer_visual_classifier_retrieval_presleep_grouplevel"
        tag = "retrievalpresleep"
    else:
        data_parent = "4.4_transfer_visual_classifier_retrieval_postsleep"
        out_parent = "4.4_transfer_visual_classifier_retrieval_postsleep_grouplevel"
        tag = "retrievalpostsleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = opj(script_dir, data_parent, "decoding_timecourses_data")
    out_dir = ensure_dir(opj(script_dir, out_parent, "decoding_timecourses_plots"))

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    tr_window = _load_tr_window(data_dir, pre_post)

    pat = re.compile(
        rf"^(s\d+)_(?:ROI_([a-z_]+)_)?decoding_timecourses_{tag}_"
        r"(scene_presented|object_retrieved|scene_retrieved)_"
        r"use_overall_classifier1_specific2_(\d+)\.npy$"
    )

    modality_files: dict[str, list[str]] = {
        "scene_presented": [],
        "object_retrieved": [],
        "scene_retrieved": [],
    }

    for fpath in glob.glob(opj(data_dir, "*.npy")):
        fname = os.path.basename(fpath)
        m = pat.match(fname)
        if not m:
            continue
        modality, clf_type = m.group(3), int(m.group(4))
        if clf_type != use_overall_classifier1_specific2:
            continue
        modality_files[modality].append(fpath)

    if all(len(v) == 0 for v in modality_files.values()):
        raise FileNotFoundError(f"No matching npy files in {data_dir}")

    def _run_one_modality(files: list[str], title: str, out_prefix: str) -> None:
        if not files:
            return
        subject_to_tc = {}
        for fpath in sorted(files):
            fname = os.path.basename(fpath)
            subj = pat.match(fname).group(1)
            decoding_dict = _load_decoding_dict(fpath)
            try:
                tc = _subject_mean_timecourse(decoding_dict)
            except ValueError:
                print(f"WARNING: Skipping {fname} — no valid trials in decoding dict")
                continue
            if tc.shape[0] != tr_window.shape[0]:
                raise ValueError(
                    f"Timecourse length mismatch in {fname}: "
                    f"got {tc.shape[0]}, expected {tr_window.shape[0]}"
                )
            subject_to_tc[subj] = tc

        if not subject_to_tc:
            print(f"WARNING: No subjects with valid trials for {out_prefix} — skipping.")
            return

        chance = 0.10
        all_subj = np.vstack(list(subject_to_tc.values()))
        if all_subj.shape[0] >= 2:
            p_raw = _pvals_vs_chance_greater(all_subj, chance=chance)
            _, q_fdr = fdr_bh(p_raw)
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

    pre_post_label = "presleep" if pre_post == "presleep" else "postsleep"

    _run_one_modality(
        modality_files["scene_presented"],
        title=f"Retrieval {pre_post_label}: Scene presented\n"
              f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}",
        out_prefix=f"decoding_timecourses_{tag}_scene_presented",
    )
    _run_one_modality(
        modality_files["object_retrieved"],
        title=f"Retrieval {pre_post_label}: Object retrieved\n"
              f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}",
        out_prefix=f"decoding_timecourses_{tag}_object_retrieved",
    )
    _run_one_modality(
        modality_files["scene_retrieved"],
        title=f"Retrieval {pre_post_label}: Scene retrieved\n"
              f"trained on use_overall_classifier1_specific2_{use_overall_classifier1_specific2}",
        out_prefix=f"decoding_timecourses_{tag}_scene_retrieved",
    )

    np.save(
        opj(out_dir, f"grouplevel_decoding_timecourses_{tag}_tr_window.npy"),
        tr_window,
    )


if __name__ == "__main__":
    main()
