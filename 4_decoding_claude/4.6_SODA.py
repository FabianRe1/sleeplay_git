# -*- coding: utf-8 -*-
"""SODA (Sequential Ordering of Decoding Analyses) for retrieval data.

Unifies the original 4.6_SODA_based_on_singlesubj_4.4_presleep.py and
4.6_SODA_based_on_singlesubj_4.4_postsleep.py into a single parametrised
script.

Usage
-----
    python 4.6_SODA.py <ROI> [presleep|postsleep]

Examples
--------
    python 4.6_SODA.py vis presleep
    python 4.6_SODA.py cortical_memory postsleep
"""

import os
import sys
import re
import ast
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from utils import (
    seq_fitter, lookup_5, control_5, script_output_dir, fdr_bh,
    apply_selected_shuffle, get_selected_shuffles,
)


# =====================================================================
# PARAMETERS
# =====================================================================

ROI_to_use = sys.argv[1]
pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

if pre_post not in ("presleep", "postsleep"):
    raise ValueError(f"Expected 'presleep' or 'postsleep', got '{pre_post}'")

shuffle_control = True
RNG = np.random.default_rng(42)  # reproducible shuffling

# Map pre/post to the correct input directory
if pre_post == "presleep":
    INPUT_DIR = "./4.4_transfer_visual_classifier_retrieval_presleep/decoding_results_tables_for_SODA"
    OUTPUT_STEM = "4.6_SODA_based_on_singlesubj_4.4_presleep"
else:
    INPUT_DIR = "./4.4_transfer_visual_classifier_retrieval_postsleep/decoding_results_tables_for_SODA"
    OUTPUT_STEM = "4.6_SODA_based_on_singlesubj_4.4_postsleep"


# =====================================================================
# LOAD DATA
# =====================================================================

csv_files = sorted(glob.glob(opj(INPUT_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(INPUT_DIR)}")

# Filter by ROI using simple string matching (replaces 24-line if/elif chain)
csv_files = [f for f in csv_files if f"ROI_{ROI_to_use}" in f]
if not csv_files:
    raise ValueError(f"No CSV files found for ROI '{ROI_to_use}' in {INPUT_DIR}")

dfs = []
for fpath in tqdm(csv_files, desc="Loading subject CSVs"):
    dfs.append(pd.read_csv(fpath))

script_dir = os.path.dirname(os.path.abspath(__file__))
plt_path = os.path.join(script_dir, OUTPUT_STEM, "SODA_plots")
os.makedirs(plt_path, exist_ok=True)


# =====================================================================
# PLOT 1: Matched vs Unmatched classifier probabilities
# =====================================================================

stacked_df = pd.concat(dfs, ignore_index=True, sort=False)
scenes1_objects2_both3 = 3

if scenes1_objects2_both3 == 1:
    stacked_df = stacked_df[stacked_df["object_scene_retrieval"] == "scene"].copy()
elif scenes1_objects2_both3 == 2:
    stacked_df = stacked_df[stacked_df["object_scene_retrieval"] == "object"].copy()

d = stacked_df[["subject", "tr", "target_classifier_matched", "probability_from_classifier"]].copy()
for col in ["tr", "probability_from_classifier", "target_classifier_matched"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")
d = d.dropna(subset=["subject", "tr", "target_classifier_matched", "probability_from_classifier"])
d["target_classifier_matched"] = d["target_classifier_matched"].astype(int)

subj_tr_mean = (
    d.groupby(["subject", "target_classifier_matched", "tr"], as_index=False)
    ["probability_from_classifier"].mean()
)


def _pivot_condition(df: pd.DataFrame, matched_value: int) -> pd.DataFrame:
    tmp = df[df["target_classifier_matched"] == matched_value]
    return tmp.pivot(index="tr", columns="subject", values="probability_from_classifier").sort_index()


wide_unmatched = _pivot_condition(subj_tr_mean, 0)
wide_matched = _pivot_condition(subj_tr_mean, 1)


def _mean_and_sem(wide: pd.DataFrame):
    mean = wide.mean(axis=1, skipna=True)
    n = wide.count(axis=1)
    sd = wide.std(axis=1, ddof=1)
    sem = sd / np.sqrt(n)
    return mean, sem, n


mean_unmatched, sem_unmatched, _ = _mean_and_sem(wide_unmatched)
mean_matched, sem_matched, _ = _mean_and_sem(wide_matched)

# Paired t-test per TR + FDR
all_trs = sorted(set(wide_unmatched.index) | set(wide_matched.index))
pvals = np.full(len(all_trs), np.nan)

for i, tr in enumerate(all_trs):
    if tr not in wide_unmatched.index or tr not in wide_matched.index:
        continue
    a = wide_matched.loc[tr]
    b = wide_unmatched.loc[tr]
    ok = ~a.isna() & ~b.isna()
    if ok.sum() < 3:
        continue
    _, p = ttest_rel(a[ok].values, b[ok].values)
    pvals[i] = p

rejected_fdr, qvals = fdr_bh(pvals)

# Plot
plt.figure(figsize=(10, 5))

for sub in wide_unmatched.columns:
    plt.plot(wide_unmatched.index, wide_unmatched[sub].values, color="0.6", alpha=0.3, lw=1)
for sub in wide_matched.columns:
    plt.plot(wide_matched.index, wide_matched[sub].values, color="#034e87", alpha=0.3, lw=1)

plt.fill_between(
    mean_unmatched.index,
    (mean_unmatched - sem_unmatched).values,
    (mean_unmatched + sem_unmatched).values,
    color="0.5", alpha=0.20, linewidth=0,
)
plt.plot(mean_unmatched.index, mean_unmatched.values, color="0.35", lw=3,
         label=f"unmatched (n_sub={wide_unmatched.shape[1]})")

plt.fill_between(
    mean_matched.index,
    (mean_matched - sem_matched).values,
    (mean_matched + sem_matched).values,
    color="#034e87", alpha=0.25, linewidth=0,
)
plt.plot(mean_matched.index, mean_matched.values, color="#034e87", lw=3,
         label=f"matched (n_sub={wide_matched.shape[1]})")

sig_trs = [tr for tr, rej in zip(all_trs, rejected_fdr) if rej]
if sig_trs:
    plt.scatter(sig_trs, [0.18] * len(sig_trs), s=24, color="k", marker="*", zorder=10, label="FDR q<0.05")

plt.xlabel("TR")
plt.ylabel("probability_from_classifier")
plt.title(
    f"Matched vs unmatched: per subject + mean ± SEM; paired t-test per TR (FDR)\n"
    f"scenes1_objects2_both3={scenes1_objects2_both3}\n"
    f"ROI={ROI_to_use}, shuffle_control={shuffle_control}, {pre_post}"
)
plt.ylim(0.05, 0.2)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(opj(plt_path, f"01_matched_vs_unmatched_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# =====================================================================
# COMPUTE SODA VALUES PER UNIQUE COMBINATION
# =====================================================================

HierarchyType = dict[str, dict[str, list[str]]]

group_columns = ["subject", "trial", "tr", "object_scene_retrieval", "retrieval_target"]
unique_combinations = stacked_df[group_columns].drop_duplicates()

# --- Pre-compute per-subject caches ---
# 1. Hierarchy structures (constant per subject)
_hier_cache: dict[str, tuple[HierarchyType, HierarchyType]] = {}

def _get_hierarchies(subj: str, subj_df: pd.DataFrame) -> tuple[HierarchyType, HierarchyType]:
    if subj not in _hier_cache:
        cat_col = subj_df["full_structured_sequence_category"].dropna()
        obj_col = subj_df["full_structured_sequence_object"].dropna()
        if cat_col.empty or obj_col.empty:
            raise ValueError(f"No sequence structure found for subject {subj}")
        _hier_cache[subj] = (
            ast.literal_eval(cat_col.iloc[0]),
            ast.literal_eval(obj_col.iloc[0]),
        )
    return _hier_cache[subj]

# 2. Group stacked_df by subject once (avoid repeated full-DF filtering)
_subj_groups = {subj: grp for subj, grp in stacked_df.groupby("subject")}

# 3. Build per-subject lookup: (trial, tr, obj_scene, target) -> subset rows
#    Index by stimulus_class_from_classifier for fast probability lookup
print("Building per-subject indexes...")
_subj_indexed: dict[str, dict[tuple, dict[str, float]]] = {}
for subj, subj_df in _subj_groups.items():
    idx = {}
    for _, r in subj_df.iterrows():
        key = (r["trial"], r["tr"], r["object_scene_retrieval"], r["retrieval_target"])
        if key not in idx:
            idx[key] = {
                "_meta": {
                    "run": r["run"],
                    "retrieval_target_precise": r["retrieval_target_precise"],
                    "retrieval_cue": r["retrieval_cue"],
                    "retrieval_cue_position": r["retrieval_cue_position"],
                    "behav_correct": r["behav_correct"],
                },
            }
        idx[key][r["stimulus_class_from_classifier"]] = r["probability_from_classifier"]
    _subj_indexed[subj] = idx
print(f"Indexed {len(_subj_indexed)} subjects.")

results = []

# --- Per-trial shuffle cache: use the SAME permutation for all TRs within a trial ---
# This avoids injecting artificial variance from changing permutations across TRs.
# Trial key = (subject, trial, object_scene_retrieval, retrieval_target)
_trial_perm_cache: dict[tuple, dict] = {}


def _get_trial_perms(trial_key: tuple, n_items: int, rng: np.random.Generator) -> dict:
    """Return cached permutations for a trial (fixed across TRs)."""
    if trial_key not in _trial_perm_cache:
        _trial_perm_cache[trial_key] = {
            "perm_ret": rng.permutation(n_items),
            "perm_ctrl": rng.permutation(n_items),
            "perm_sel_ret": get_selected_shuffles(n_items)[rng.integers(len(get_selected_shuffles(n_items)))],
            "perm_sel_ctrl": get_selected_shuffles(n_items)[rng.integers(len(get_selected_shuffles(n_items)))],
        }
    return _trial_perm_cache[trial_key]


for _, row in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc="Computing SODA"):
    subj = row["subject"]
    subj_df = _subj_groups[subj]
    hier_structure, hier_structure_obj = _get_hierarchies(subj, subj_df)

    combo_key = (row["trial"], row["tr"], row["object_scene_retrieval"], row["retrieval_target"])
    prob_lookup = _subj_indexed[subj].get(combo_key)
    if prob_lookup is None:
        continue
    meta = prob_lookup["_meta"]

    item_to_be_retrieved = row["retrieval_target"]
    object_scene_retrieval = row["object_scene_retrieval"]

    retrieved_sequence = lookup_5(hier_structure, item_to_be_retrieved, kind=object_scene_retrieval)
    probs = [prob_lookup.get(item, np.nan) for item in retrieved_sequence]

    control_sequence = control_5(hier_structure, item_to_be_retrieved, kind=object_scene_retrieval)
    control_probs = [prob_lookup.get(item, np.nan) for item in control_sequence]

    # Get fixed per-trial permutations (same for all TRs within this trial)
    trial_key = (subj, row["trial"], row["object_scene_retrieval"], row["retrieval_target"])
    trial_perms = _get_trial_perms(trial_key, len(probs), RNG)

    # Random shuffle: fixed permutation per trial
    probs_arr = np.array(probs, dtype=float)
    control_arr = np.array(control_probs, dtype=float)
    probs_shuffle = list(probs_arr[trial_perms["perm_ret"]])
    control_probs_shuffle = list(control_arr[trial_perms["perm_ctrl"]])

    # Selected shuffle: fixed strict permutation per trial
    probs_shuffle_selected = list(probs_arr[list(trial_perms["perm_sel_ret"])])
    control_probs_shuffle_selected = list(control_arr[list(trial_perms["perm_sel_ctrl"])])

    # SODA slopes (signed)
    retrieved_soda = seq_fitter(np.array(probs))
    control_soda = seq_fitter(np.array(control_probs))
    retrieved_soda_shuffle = seq_fitter(np.array(probs_shuffle))
    control_soda_shuffle = seq_fitter(np.array(control_probs_shuffle))
    retrieved_soda_shuffle_selected = seq_fitter(np.array(probs_shuffle_selected))
    control_soda_shuffle_selected = seq_fitter(np.array(control_probs_shuffle_selected))

    results.append({
        "ROI": ROI_to_use,
        "subject": subj,
        "trial": row["trial"],
        "run": meta["run"],
        "tr": row["tr"],
        "object_scene_retrieval": object_scene_retrieval,
        "retrieval_target": row["retrieval_target"],
        "retrieval_target_precise": meta["retrieval_target_precise"],
        "retrieval_cue": meta["retrieval_cue"],
        "retrieved_position": meta["retrieval_cue_position"],
        # Signed SODA
        "retrieved_soda": retrieved_soda,
        "control_soda": control_soda,
        "retrieved_soda_shuffled": retrieved_soda_shuffle,
        "control_soda_shuffled": control_soda_shuffle,
        "retrieved_soda_shuffled_selected": retrieved_soda_shuffle_selected,
        "control_soda_shuffled_selected": control_soda_shuffle_selected,
        # Absolute SODA (unsigned slope)
        "retrieved_soda_abs": abs(retrieved_soda) if not np.isnan(retrieved_soda) else np.nan,
        "control_soda_abs": abs(control_soda) if not np.isnan(control_soda) else np.nan,
        "retrieved_soda_shuffled_abs": abs(retrieved_soda_shuffle) if not np.isnan(retrieved_soda_shuffle) else np.nan,
        "control_soda_shuffled_abs": abs(control_soda_shuffle) if not np.isnan(control_soda_shuffle) else np.nan,
        "retrieved_soda_shuffled_selected_abs": abs(retrieved_soda_shuffle_selected) if not np.isnan(retrieved_soda_shuffle_selected) else np.nan,
        "control_soda_shuffled_selected_abs": abs(control_soda_shuffle_selected) if not np.isnan(control_soda_shuffle_selected) else np.nan,
        # Probabilities
        "retrieved_probs": probs,
        "control_probs": control_probs,
        "retrieved_probs_shuffle": probs_shuffle,
        "control_probs_shuffle": control_probs_shuffle,
        "retrieved_probs_shuffle_selected": probs_shuffle_selected,
        "control_probs_shuffle_selected": control_probs_shuffle_selected,
        "behav_correct": meta["behav_correct"],
        "full_structured_sequence_category": hier_structure,
        "full_structured_sequence_object": hier_structure_obj,
    })

results_df = pd.DataFrame(results)

csv_out_dir = os.path.join(script_dir, OUTPUT_STEM, "SODA_results_csv")
os.makedirs(csv_out_dir, exist_ok=True)
results_df.to_csv(opj(csv_out_dir, f"{ROI_to_use}_soda_results_per_combination.csv"), index=False)


# =====================================================================
# PLOT 2: Retrieved vs Control SODA per TR
# =====================================================================

subj_tr = results_df.groupby(["subject", "tr"], as_index=False)[["retrieved_soda", "control_soda"]].mean()
agg = subj_tr.groupby("tr")[["retrieved_soda", "control_soda"]].agg(["mean", "std", "count"])

mean_ret = agg[("retrieved_soda", "mean")]
mean_ctrl = agg[("control_soda", "mean")]
sem_ret = agg[("retrieved_soda", "std")] / np.sqrt(agg[("retrieved_soda", "count")])
sem_ctrl = agg[("control_soda", "std")] / np.sqrt(agg[("control_soda", "count")])

plt.figure(figsize=(10, 5))
plt.plot(mean_ret.index, mean_ret.values, label="Retrieved SODA", color="blue", lw=2)
plt.fill_between(mean_ret.index, (mean_ret - sem_ret).values, (mean_ret + sem_ret).values,
                 color="blue", alpha=0.2, linewidth=0)
plt.plot(mean_ctrl.index, mean_ctrl.values, label="Control SODA", color="orange", lw=2)
plt.fill_between(mean_ctrl.index, (mean_ctrl - sem_ctrl).values, (mean_ctrl + sem_ctrl).values,
                 color="orange", alpha=0.2, linewidth=0)
plt.ylim(-0.02, 0.02)
plt.xlabel("TR")
plt.ylabel("Mean SODA")
plt.title(f"Mean Retrieved SODA vs Control SODA per TR (± SEM)\nROI={ROI_to_use}, {pre_post}")
plt.legend()
plt.tight_layout()
plt.savefig(opj(plt_path, f"02_retrieved_vs_control_soda_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# =====================================================================
# PLOT 3: SODA by retrieved position (5 panels)
# =====================================================================

results_df["retrieved_position"] = pd.to_numeric(results_df["retrieved_position"], errors="coerce")
results_df["tr"] = pd.to_numeric(results_df["tr"], errors="coerce")

positions = [1, 2, 3, 4, 5]

fig, axes = plt.subplots(nrows=1, ncols=len(positions), figsize=(24, 4.5), sharey=True, sharex=True)
if len(positions) == 1:
    axes = [axes]

pos_summaries = []

for ax, pos in zip(axes, positions):
    df_pos = results_df[results_df["retrieved_position"] == pos].copy()

    if df_pos.empty:
        ax.set_title(f"pos={pos} (no data)")
        ax.set_xlabel("TR")
        continue

    subj_tr = df_pos.groupby(["subject", "tr"], as_index=False)[["retrieved_soda", "control_soda"]].mean()

    subj_meanTR = subj_tr.groupby("subject")[["retrieved_soda", "control_soda"]].mean()
    subj_absmeanTR = subj_tr.groupby("subject")[["retrieved_soda", "control_soda"]].apply(lambda x: x.abs().mean())

    meanTR_retrieved = float(subj_meanTR["retrieved_soda"].mean())
    meanTR_control = float(subj_meanTR["control_soda"].mean())
    absmeanTR_retrieved = float(subj_absmeanTR["retrieved_soda"].mean())
    absmeanTR_control = float(subj_absmeanTR["control_soda"].mean())
    n_sub = int(subj_meanTR.shape[0])

    tmp = subj_meanTR.reset_index()
    tmp["retrieved_position"] = pos
    tmp["mean_delta"] = tmp["retrieved_soda"] - tmp["control_soda"]
    pos_summaries.append(tmp[["subject", "retrieved_position", "retrieved_soda", "control_soda", "mean_delta"]])

    agg = subj_tr.groupby("tr")[["retrieved_soda", "control_soda"]].agg(["mean", "std", "count"])
    m_ret = agg[("retrieved_soda", "mean")]
    m_ctrl = agg[("control_soda", "mean")]
    s_ret = agg[("retrieved_soda", "std")] / np.sqrt(agg[("retrieved_soda", "count")])
    s_ctrl = agg[("control_soda", "std")] / np.sqrt(agg[("control_soda", "count")])

    ax.plot(m_ret.index, m_ret.values, label="Retrieved", color="blue", lw=2)
    ax.fill_between(m_ret.index, (m_ret - s_ret).values, (m_ret + s_ret).values,
                    color="blue", alpha=0.2, linewidth=0)
    ax.plot(m_ctrl.index, m_ctrl.values, label="Control", color="orange", lw=2)
    ax.fill_between(m_ctrl.index, (m_ctrl - s_ctrl).values, (m_ctrl + s_ctrl).values,
                    color="orange", alpha=0.2, linewidth=0)

    ax.set_title(
        f"pos={pos} (n_sub={n_sub})\n"
        f"meanTR R={meanTR_retrieved:.3f}, C={meanTR_control:.3f} | "
        f"absmeanTR R={absmeanTR_retrieved:.3f}, C={absmeanTR_control:.3f}"
    )
    ax.set_xlabel("TR")

axes[0].set_ylabel("Mean SODA")
axes[-1].legend(frameon=False, loc="best")

fig.suptitle(
    f"Mean Retrieved SODA vs Control SODA per TR (± SEM), split by retrieved position\n"
    f"ROI={ROI_to_use}, shuffle_control={shuffle_control}, {pre_post}",
    y=1.10,
)
fig.tight_layout()
plt.savefig(opj(plt_path, f"03_soda_by_retrieved_position_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png"),
            dpi=150, bbox_inches="tight")
plt.close()
