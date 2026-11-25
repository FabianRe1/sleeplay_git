#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B5 - Cross-validated classifier accuracy, group level.

Collects the leave-one-run-out results of B4 over all participants and plots
accuracy per participant, per stimulus and per ROI, for the scene and the
object classifiers. Chance is 10 percent.

The classifiers are trained on trials in which the image is shown ("encoding")
and trials in which it is recalled ("retrieval"), because this transfers better
to retrieval in the main task. Accuracy is therefore reported for all trials
and separately for the two trial types.

Input   outputs/B4_train_classifiers/df_classifier_results_csv/*.csv
Output  outputs/B5_classifier_accuracy/accuracy_summary.csv
            mean, SEM, N, t-test against chance per ROI, classifier and trial
            type; p_holm is corrected across all ROIs within classifier and trial
            type, p_holm_main across the ROIs of the paper figure only (MAIN_ROIS)
        outputs/B5_classifier_accuracy/accuracy_per_participant.csv
Plots   plots/B5_classifier_accuracy/*.png, *.pdf
        plots/B5_classifier_accuracy/fig_classifier_accuracy.{png,pdf}
            paper figure: object and scene classifier in the ROIs of MAIN_ROIS,
            image shown against image recalled, one dot per participant, mean
            with 95 % confidence interval, stars from p_holm_main
            (* < .05, ** < .01, *** < .001; n.s. otherwise). Added 2026-11-25.

Ported from 4_decoding_claude_retrieval/4.2_analyse_best_classifier_performance.py.
Changes to the original: accuracy split by trial type and the summary tables
are new; the mean accuracy is no longer part of the file names (old plots
stayed behind whenever a number changed); the dot jitter is reproducible.

Usage:  python B5_classifier_accuracy.py
"""

import os
import sys
import glob
import re
import zlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

# ---------------------------------------------------------------------
# Parameter grids (mirrors 4.1_train_visual_classifier.py)
# ---------------------------------------------------------------------
mask_labels = ['mtl', 'parietal_precuneus', 'parietal', 'precuneus', 'medial_ofc', 'rostral_acc', 'caudal_acc', 'superior_frontal', 'rsc', 'posterior_cingulate', 'vis', 'cortical_memory', 'hpc', 'somato']
smoothing_levels = [4]
f_thresholds = [30]
reliability_thresholds = [70]
temporal_smoothing_levels = [3]
ls_only_scenes1_objects2_both3 = [1, 2]


def get_results_dir():
    """Cross-validation tables written by B4."""
    return str(step_input_dir("B4_train_classifiers", "df_classifier_results_csv"))


def load_combined_results(only_scenes1_objects2_both3, mask, smoothing, fstat_percentile,
                          reliability_threshold, temporal_smoothing):
    """
    Load all subject CSVs for a given parameter combination.
    Adds 'subj' column extracted from filename.
    """
    results_dir = get_results_dir()
    pattern = (
        f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_*_"
        f"Mask_{mask}_Smooth_{smoothing}_Threshold_{fstat_percentile}"
        f"_reliability_threshold_{reliability_threshold}"
        f"_tmpsmoothing_{temporal_smoothing}.csv"
    )
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))

    if not paths:
        return None  # caller handles "no files for this combo"

    dfs = []
    for p in paths:
        df = pd.read_csv(p, index_col=0)
        fname = os.path.basename(p)
        tokens = fname.split("_")

        # Extract subject ID: first token matching pattern like "s13"
        subj = None
        for t in tokens:
            if re.match(r"^s\d+$", t):
                subj = t
                break
        if subj is None:
            raise ValueError(f"Could not infer subject ID from filename: {fname}")
        df["subj"] = subj

        if "label_match" not in df.columns or "encoding_category" not in df.columns:
            raise ValueError(f"Required columns missing in file: {p}")

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def plot_overall_per_subject(df, title_suffix=""):
    """
    Bar plot of overall performance (mean label_match) per subject,
    with per-stimulus scatter points for each subject.
    """
    subj_perf = df.groupby("subj")["label_match"].mean().sort_index()
    n_classes = df["encoding_category"].nunique()
    chance_level = 100.0 / n_classes

    subj_list = subj_perf.index.tolist()
    x = np.arange(len(subj_list))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Bars: overall subject accuracy
    ax.bar(x, subj_perf.values * 100.0, color="steelblue", zorder=1)

    # Per-stimulus accuracy per subject as scatter
    subj_stim = (
        df.groupby(["subj", "encoding_category"])["label_match"]
        .mean()
        .reset_index()
    )

    rng = np.random.default_rng(42)
    for i, subj in enumerate(subj_list):
        sub_df = subj_stim[subj_stim["subj"] == subj]
        if sub_df.empty:
            continue

        # Small horizontal jitter so points don't perfectly overlap
        jitter = (rng.random(len(sub_df)) - 0.5) * 0.3
        ax.scatter(
            np.full(len(sub_df), x[i]) + jitter,
            sub_df["label_match"].values * 100.0,
            color="black",
            alpha=0.4,
            s=15,
            zorder=2,
        )

    overall_mean = subj_perf.mean() * 100.0

    ax.axhline(
        chance_level,
        color="red",
        linestyle="--",
        label=f"Chance ({chance_level:.1f}%)",
        zorder=0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subj_list)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Subject")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Overall classification performance per subject"
        f"{title_suffix}\nGroup mean = {overall_mean:.2f}%"
    )

    ax.legend()
    fig.tight_layout()
    return fig, ax, overall_mean


def plot_per_stimulus_across_subjects(df, title_suffix=""):
    """
    Bar plot of performance per stimulus, averaged across subjects,
    plus one line per subject with dots connected across stimuli.
    """
    # Mean across subjects per stimulus (for bars)
    stim_perf = (
        df.groupby("encoding_category")["label_match"]
        .mean()
        .sort_values(ascending=False)
    )

    stim_order = stim_perf.index.tolist()
    n_classes = len(stim_order)
    chance_level = 100.0 / n_classes

    width = max(10.0, 0.4 * n_classes)
    fig, ax = plt.subplots(figsize=(width, 5))

    x = np.arange(n_classes)

    # Bars: group mean per stimulus
    bars = ax.bar(
        x,
        stim_perf[stim_order].values * 100.0,
        color="slateblue",
        zorder=1,
    )

    # Per-subject lines: accuracy across stimuli
    subj_stim = (
        df.groupby(["subj", "encoding_category"])["label_match"]
        .mean()
        .unstack("encoding_category")
    )

    # Reorder columns to match stim_order and drop stimuli not present
    subj_stim = subj_stim.reindex(columns=stim_order)

    for subj, row in subj_stim.iterrows():
        if row.isna().all():
            continue
        y = row.values * 100.0
        ax.plot(
            x,
            y,
            color="black",
            alpha=0.3,
            linewidth=0.8,
            marker="o",
            markersize=3,
            zorder=2,
        )

    ax.axhline(
        chance_level,
        color="red",
        linestyle="--",
        label=f"Chance ({chance_level:.1f}%)",
        zorder=0,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Stimulus (encoding_category)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Performance per stimulus across subjects{title_suffix}")

    ax.set_xticks(x)
    ax.set_xticklabels(stim_order, rotation=90, ha="center")

    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_decoding_per_mask(combo_results, condition, out_dir):
    """
    Bar plot of decoding accuracy per mask (group mean across subjects)
    with individual subject scatter dots connected across masks.

    Parameters
    ----------
    combo_results : list[dict]
        Each entry has keys: mask, overall_mean, and per-subject data.
    condition : int
        1=scenes, 2=objects, 3=both.
    out_dir : str
        Directory to save the figure.
    """
    label_map = {1: "Scenes only", 2: "Objects only", 3: "Scenes & Objects"}
    cond_label = label_map.get(condition, f"Condition {condition}")

    # Filter results for this condition
    subset = [r for r in combo_results if r["only_scenes1_objects2_both3"] == condition]
    if not subset:
        return

    # Get unique masks in original order
    masks_in_results = []
    for r in subset:
        if r["mask"] not in masks_in_results:
            masks_in_results.append(r["mask"])

    # For each mask, compute subject-level mean accuracy
    mask_subj_data = {}  # mask -> {subj: mean_acc}
    mask_group_mean = {}
    mask_group_sem = {}
    mask_n = {}

    for mask in masks_in_results:
        r = next((r for r in subset if r["mask"] == mask), None)
        if r is None or r.get("subj_means") is None:
            continue
        subj_means = r["subj_means"]  # dict {subj: accuracy %}
        vals = np.array(list(subj_means.values()))
        mask_subj_data[mask] = subj_means
        mask_group_mean[mask] = vals.mean()
        mask_group_sem[mask] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        mask_n[mask] = len(vals)

    masks = [m for m in masks_in_results if m in mask_subj_data]
    if not masks:
        return

    x = np.arange(len(masks))
    means = [mask_group_mean[m] for m in masks]
    sems = [mask_group_sem[m] for m in masks]

    # Determine chance level from first result
    r0 = next(r for r in subset if r["mask"] == masks[0])
    chance_level = r0.get("chance_level", 10.0)

    fig, ax = plt.subplots(figsize=(max(8, len(masks) * 0.8), 5))

    ax.bar(x, means, yerr=sems, width=0.6,
           color="steelblue", edgecolor="#333333", linewidth=0.8,
           capsize=4, error_kw={"linewidth": 1.2, "capthick": 1.2}, zorder=1)

    # Collect all subjects across masks
    all_subjs = set()
    for subj_means in mask_subj_data.values():
        all_subjs.update(subj_means.keys())

    # Plot subject dots with connecting lines
    for sid in sorted(all_subjs):
        jit = np.random.default_rng(zlib.crc32(sid.encode())).uniform(-0.12, 0.12)
        xs_s, ys_s = [], []
        for i, mask in enumerate(masks):
            if sid in mask_subj_data[mask]:
                xs_s.append(x[i] + jit)
                ys_s.append(mask_subj_data[mask][sid])
        ax.plot(xs_s, ys_s, color="black", lw=0.5, alpha=0.25, zorder=2)
        ax.scatter(xs_s, ys_s, color="black", s=18, alpha=0.45, zorder=3, linewidths=0)

    ax.axhline(chance_level, color="red", ls="--", lw=1,
               label=f"Chance ({chance_level:.1f}%)", zorder=0)

    # One-sample t-tests against chance level
    for i, mask in enumerate(masks):
        vals = np.array(list(mask_subj_data[mask].values()))
        if len(vals) > 1:
            t_stat, p_val = stats.ttest_1samp(vals, chance_level)
            print(f'mask: {mask} - T: {t_stat} - p: {p_val}')
            if p_val < 0.05:
                ax.text(i, 65, "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(masks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title(f"Decoding per mask — {cond_label}", fontweight="bold")

    for i in range(len(masks)):
        ax.text(i, means[i] + sems[i] + 1.5,
                f"N={mask_n[masks[i]]}", ha="center", fontsize=8, color="#555")

    ax.legend(frameon=False)
    fig.tight_layout()

    fname = os.path.join(out_dir, f"decoding_per_mask_cond{condition}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fname_pdf = os.path.join(out_dir, f"decoding_per_mask_cond{condition}.pdf")
    fig.savefig(fname_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decoding-per-mask plot for {cond_label} → {fname}")


TRIAL_TYPES = {"all": "All trials", "encoding": "Image shown", "retrieval": "Image recalled"}
# the ROIs of the paper figure; hippocampus and the two control regions (somatosensory cortex,
# caudal ACC) were added to visual, parietal + precuneus and MTL on 2026-11-25
MAIN_ROIS = {"vis": "Visual", "parietal_precuneus": "Parietal +\nprecuneus", "mtl": "MTL",
             "hpc": "Hippo-\ncampus", "somato": "Somatosensory\n(control)", "caudal_acc": "Caudal ACC\n(control)"}
CLASSIFIER_NAMES = {1: "scene", 2: "object", 3: "both"}


def accuracy_by_trial_type(df):
    """Accuracy (%) per participant for all trials and per trial type."""
    out = {"all": (df.groupby("subj")["label_match"].mean() * 100.0).to_dict()}
    for phase in ("encoding", "retrieval"):
        d = df[df["trial_phase"] == phase]
        out[phase] = (d.groupby("subj")["label_match"].mean() * 100.0).to_dict()
    return out


def write_accuracy_tables(combo_results, out_dir):
    """Per-participant accuracies and the group summary with t-tests against chance."""
    per_subj, summary = [], []
    for r in combo_results:
        clf = CLASSIFIER_NAMES[r["only_scenes1_objects2_both3"]]
        for trial_type, subj_means in r["subj_means_by_type"].items():
            for subj, acc in sorted(subj_means.items()):
                per_subj.append(dict(classifier=clf, roi=r["mask"], trial_type=trial_type,
                                     subj=subj, accuracy=acc))
            vals = np.array(list(subj_means.values()))
            t_stat, p_val = stats.ttest_1samp(vals, r["chance_level"])
            summary.append(dict(classifier=clf, roi=r["mask"], trial_type=trial_type,
                                n=len(vals), mean=vals.mean(),
                                sem=vals.std(ddof=1) / np.sqrt(len(vals)),
                                chance=r["chance_level"], t=t_stat, df=len(vals) - 1, p=p_val))
    per_subj, summary = pd.DataFrame(per_subj), pd.DataFrame(summary)
    # Holm correction across ROIs, within classifier and trial type
    summary["p_holm"] = np.nan
    for _, idx in summary.groupby(["classifier", "trial_type"]).groups.items():
        summary.loc[idx, "p_holm"] = multipletests(summary.loc[idx, "p"], method="holm")[1]
    summary["p_holm_main"] = np.nan
    main = summary[summary.roi.isin(MAIN_ROIS)]
    for _, idx in main.groupby(["classifier", "trial_type"]).groups.items():
        summary.loc[idx, "p_holm_main"] = multipletests(summary.loc[idx, "p"], method="holm")[1]
    per_subj.to_csv(os.path.join(out_dir, "accuracy_per_participant.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "accuracy_summary.csv"), index=False)
    print(f"Saved accuracy tables to {out_dir}")
    return summary, per_subj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_main_figure(summary, per_subj, out_dir):
    """Paper figure: both classifiers, four main ROIs, image shown against image recalled."""
    colors = {"encoding": "#2a78d6", "retrieval": "#eb6834"}
    rois = [r for r in MAIN_ROIS if r in set(summary.roi)]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), sharey=True)
    for ax, clf in zip(axes, ("object", "scene")):
        for i, roi in enumerate(rois):
            for k, trial_type in enumerate(("encoding", "retrieval")):
                x0 = i + (k - 0.5) * 0.36
                vals = per_subj[(per_subj.classifier == clf) & (per_subj.roi == roi)
                                & (per_subj.trial_type == trial_type)].accuracy.values
                if len(vals) < 2:
                    continue
                jit = np.random.default_rng(zlib.crc32(f"{clf}{roi}{trial_type}".encode())).uniform(-0.11, 0.11, len(vals))
                ax.scatter(x0 + jit, vals, s=9, color=colors[trial_type], alpha=0.35, linewidths=0, zorder=2)
                ci = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
                ax.errorbar(x0, vals.mean(), yerr=ci, fmt="o", color="#0b0b0b", markerfacecolor=colors[trial_type],
                            markersize=8, capsize=3, lw=1.3, zorder=4,
                            label=TRIAL_TYPES[trial_type] if i == 0 else None)
                row = summary[(summary.classifier == clf) & (summary.roi == roi)
                              & (summary.trial_type == trial_type)].iloc[0]
                txt = stars(row.p_holm_main)
                ax.text(x0, vals.max() + 1.5, txt, ha="center", va="bottom",
                        fontsize=12 if txt != "n.s." else 8, fontweight="bold" if txt != "n.s." else "normal",
                        color="#0b0b0b" if txt != "n.s." else "#52514e")
        ax.axhline(10, color="#52514e", ls="--", lw=1, zorder=1, label="Chance (10 %)")
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels([MAIN_ROIS[r] for r in rois])
        ax.set_title(f"{clf.capitalize()} classifier", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 92)
    axes[0].set_ylabel("Cross-validated accuracy (%)")
    axes[0].legend(frameon=False, loc="upper right")
    n = per_subj.subj.nunique()
    fig.suptitle(f"Classifiers trained on shown and recalled images, tested by trial type (N = {n})", fontsize=11)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(out_dir, f"fig_classifier_accuracy.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


def plot_decoding_per_mask_by_trial_type(combo_results, summary, condition, out_dir):
    """Accuracy per ROI, separately for trials with the image shown and recalled."""
    subset = [r for r in combo_results if r["only_scenes1_objects2_both3"] == condition]
    if not subset:
        return
    clf = CLASSIFIER_NAMES[condition]
    masks = [r["mask"] for r in subset]
    x = np.arange(len(masks))
    colors = {"encoding": "steelblue", "retrieval": "darkorange"}
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(masks) * 1.0), 5))
    for k, trial_type in enumerate(["encoding", "retrieval"]):
        offset = (k - 0.5) * width
        for i, r in enumerate(subset):
            vals = np.array(list(r["subj_means_by_type"][trial_type].values()))
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.bar(x[i] + offset, vals.mean(), yerr=sem, width=width, color=colors[trial_type],
                   edgecolor="#333333", linewidth=0.8, capsize=3, zorder=1,
                   label=TRIAL_TYPES[trial_type] if i == 0 else None)
            jit = np.random.default_rng(zlib.crc32(f"{r['mask']}{trial_type}".encode())).uniform(
                -0.1, 0.1, len(vals))
            ax.scatter(x[i] + offset + jit, vals, color="black", s=10, alpha=0.35,
                       zorder=3, linewidths=0)
            row = summary[(summary.classifier == clf) & (summary.roi == r["mask"])
                          & (summary.trial_type == trial_type)].iloc[0]
            if row.p_holm < 0.05:
                ax.text(x[i] + offset, vals.mean() + sem + 1, "*", ha="center", va="bottom",
                        fontsize=12, fontweight="bold")
    ax.axhline(subset[0]["chance_level"], color="red", ls="--", lw=1, zorder=0,
               label=f"Chance ({subset[0]['chance_level']:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(masks, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title(f"Decoding per mask by trial type, {clf} classifier\n"
                 "(* p < .05 against chance, Holm-corrected across masks)", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(out_dir, f"decoding_per_mask_cond{condition}_by_trial_type.{ext}"),
                    bbox_inches="tight", **kw)
    plt.close(fig)


def build_param_suffix(mask, smoothing, fstat_percentile,
                       reliability_threshold, temporal_smoothing):
    return (f"\nmask={mask}, smooth={smoothing}, "
            f"fstat={fstat_percentile}, rel={reliability_threshold}, "
            f"tmp_smooth={temporal_smoothing}")


def main():
    results_dir = get_results_dir()
    out_dir = str(step_plot_dir(__file__))

    print(f"Reading CSVs from: {results_dir}")
    print(f"Saving group plots to: {out_dir}")

    n_done = 0
    combo_results = []  # store (condition, params, mean) per parameter combination
    for only_scenes1_objects2_both3 in ls_only_scenes1_objects2_both3:
        for mask in mask_labels:
            for reliability_threshold in reliability_thresholds:
                for smoothing in smoothing_levels:
                    for temporal_smoothing in temporal_smoothing_levels:
                        for fstat_percentile in f_thresholds:

                            df = load_combined_results(
                                only_scenes1_objects2_both3=only_scenes1_objects2_both3,
                                mask=mask,
                                smoothing=smoothing,
                                fstat_percentile=fstat_percentile,
                                reliability_threshold=reliability_threshold,
                                temporal_smoothing=temporal_smoothing,
                            )

                            if df is None or df.empty:
                                # No subjects / no files for this combo
                                continue

                            suffix = build_param_suffix(
                                mask,
                                smoothing,
                                fstat_percentile,
                                reliability_threshold,
                                temporal_smoothing,
                            )

                            # --- overall performance per subject ---
                            fig1, ax1, overall_mean = plot_overall_per_subject(df, title_suffix=suffix)

                            base = (f"group_mask-{mask}_smooth-{smoothing}"
                                    f"_fstat-{fstat_percentile}"
                                    f"_rel-{reliability_threshold}"
                                    f"_tmpsmooth-{temporal_smoothing}")

                            f1_path = os.path.join(out_dir, f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{base}_overall_per_subject.png")
                            fig1.savefig(f1_path, dpi=300, bbox_inches="tight")
                            plt.close(fig1)

                            # --- per-stimulus performance across subjects ---
                            fig2, ax2 = plot_per_stimulus_across_subjects(df, title_suffix=suffix)
                            f2_path = os.path.join(out_dir, f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{base}_per_stimulus.png")
                            fig2.savefig(f2_path, dpi=300, bbox_inches="tight")
                            plt.close(fig2)

                            n_done += 1
                            print(f"Saved plots for combo #{n_done}: {base}")

                            # Subject-level means for per-mask plot
                            subj_means = (df.groupby("subj")["label_match"].mean() * 100.0).to_dict()
                            n_classes = df["encoding_category"].nunique()
                            chance_level = 100.0 / n_classes

                            combo_results.append({
                                "only_scenes1_objects2_both3": only_scenes1_objects2_both3,
                                "mask": mask,
                                "smoothing": smoothing,
                                "fstat_percentile": fstat_percentile,
                                "reliability_threshold": reliability_threshold,
                                "temporal_smoothing": temporal_smoothing,
                                "overall_mean": overall_mean,
                                "subj_means": subj_means,
                                "subj_means_by_type": accuracy_by_trial_type(df),
                                "chance_level": chance_level,
                            })

    if n_done == 0:
        print("No matching CSV files found for any parameter combination.")
    else:
        print(f"Finished. Generated {n_done * 2} figures.")

        # For each condition (1=scenes, 2=objects, 3=both), print top 3 combos
        label_map = {1: "scenes only", 2: "objects only", 3: "scenes and objects"}
        for cond in ls_only_scenes1_objects2_both3:
            subset = [r for r in combo_results if r["only_scenes1_objects2_both3"] == cond]
            if not subset:
                continue

            subset = sorted(subset, key=lambda r: r["overall_mean"], reverse=True)
            print("\nTop 3 parameter combinations for"
                  f" {label_map.get(cond, cond)} (only_scenes1_objects2_both3={cond}):")

            for rank, r in enumerate(subset[:3], 1):
                print(
                    f"  {rank}. {r['overall_mean']:.2f}% - "
                    f"mask={r['mask']}, smooth={r['smoothing']}, "
                    f"fstat={r['fstat_percentile']}, "
                    f"rel={r['reliability_threshold']}, "
                    f"tmp_smooth={r['temporal_smoothing']}"
                )

        # --- decoding per mask summary plot (one per condition) ---
        for cond in ls_only_scenes1_objects2_both3:
            plot_decoding_per_mask(combo_results, cond, out_dir)

        # --- accuracy tables and the split by trial type ---
        summary, per_subj = write_accuracy_tables(combo_results, str(step_output_dir(__file__)))
        plot_main_figure(summary, per_subj, out_dir)
        for cond in ls_only_scenes1_objects2_both3:
            plot_decoding_per_mask_by_trial_type(combo_results, summary, cond, out_dir)


if __name__ == "__main__":
    main()
