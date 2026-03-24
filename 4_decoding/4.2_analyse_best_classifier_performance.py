# -*- coding: utf-8 -*-

import os
import glob
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------
# Parameter grids (mirrors 4.1_train_visual_classifier.py)
# ---------------------------------------------------------------------
mask_labels = ['mtl', 'parietal_precuneus', 'parietal', 'precuneus', 'medial_ofc', 'rostral_acc', 'caudal_acc', 'superior_frontal', 'rsc', 'posterior_cingulate', 'vis', 'cortical_memory']  # extend if you add more masks in 4.1
smoothing_levels = [4]
f_thresholds = [30]
reliability_thresholds = [70]
temporal_smoothing_levels = [3]
ls_only_scenes1_objects2_both3 = [1,2]  # must match 4.1

def get_results_dir():
    """
    Points to the CSV output directory created by 4.1_train_visual_classifier.py:
        4.1_train_visual_classifier/df_classifier_results_csv
    located in the same directory as this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_stem = "4.1_train_visual_classifier"
    results_dir = os.path.join(script_dir, train_script_stem, "df_classifier_results_csv")

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Run 4.1_train_visual_classifier.py first to generate CSVs."
        )
    return results_dir


def load_combined_results(only_scenes1_objects2_both3, mask, smoothing, fstat_percentile,
                          reliability_threshold, temporal_smoothing):
    """
    Load all subject CSVs for a given parameter combination.
    Adds 'subj' column extracted from filename.
    """
    results_dir = get_results_dir()
    # filename_df = opj(path_df, f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing}.csv")
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

    for i, subj in enumerate(subj_list):
        sub_df = subj_stim[subj_stim["subj"] == subj]
        if sub_df.empty:
            continue

        # Small horizontal jitter so points don't perfectly overlap
        jitter = (np.random.rand(len(sub_df)) - 0.5) * 0.3
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
        jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.12, 0.12)
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
    plt.show()
    print(f"Saved decoding-per-mask plot for {cond_label} → {fname}")


def build_param_suffix(mask, smoothing, fstat_percentile,
                       reliability_threshold, temporal_smoothing):
    return (f"\nmask={mask}, smooth={smoothing}, "
            f"fstat={fstat_percentile}, rel={reliability_threshold}, "
            f"tmp_smooth={temporal_smoothing}")


def main():
    results_dir = get_results_dir()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "4.2_analyse_best_classifier_performance", "group_plots")
    os.makedirs(out_dir, exist_ok=True)

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

                            f1_path = os.path.join(out_dir, f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{overall_mean:.2f}_{base}_overall_per_subject.png")
                            fig1.savefig(f1_path, dpi=300, bbox_inches="tight")
                            plt.close(fig1)

                            # --- per-stimulus performance across subjects ---
                            fig2, ax2 = plot_per_stimulus_across_subjects(df, title_suffix=suffix)
                            f2_path = os.path.join(out_dir, f"only_scenes1_objects2_both3_{only_scenes1_objects2_both3}_{overall_mean:.2f}_{base}_per_stimulus.png")
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


if __name__ == "__main__":
    main()
    
    