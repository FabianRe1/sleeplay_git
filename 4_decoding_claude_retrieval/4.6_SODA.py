# -*- coding: utf-8 -*-
"""SODA (Sequential Ordering of Decoding Analyses) for retrieval data.

Unifies the original 4.6_SODA_based_on_singlesubj_4.4_presleep.py and
4.6_SODA_based_on_singlesubj_4.4_postsleep.py into a single parametrised
script.

Three nested fit windows are computed per trial and TR:

  full   : all 5 sequence elements (the original analysis; unchanged)
  trunc  : elements 1 .. p, where p is the retrieved position
           (a retrieval of element p is assumed to involve replaying the
           sequence from its start up to and including element p, so
           elements after p should not enter the regression)
  runup  : elements 1 .. p-1, i.e. the same window with the retrieved
           element itself removed. Diagnostic: the retrieved element has an
           elevated classifier probability by construction, so contrasting
           `trunc` against `runup` separates that endpoint from the
           sequential ramp leading up to it.

`trunc` and `runup` require at least MIN_ELEMENTS elements, so `trunc` is
defined for p >= 3 and `runup` for p >= 4; all other trials get NaN.
Control and shuffle arms use the same window length as the retrieved arm.

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

# Import utils from the original pipeline directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "4_decoding_claude"))
import re
import ast
import glob
import hashlib
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
RNG = np.random.default_rng(42)  # reproducible shuffling (full 5-element window)

# --- Position-truncated windows -------------------------------------------
# Minimum number of sequence elements required for a truncated regression.
MIN_ELEMENTS = 3
# Base seed for the truncated arms. Each trial × window gets its own generator
# derived from this seed and the trial key (see _rng_for), so the truncated
# shuffles are independent of the full-window random stream and stable under
# unrelated code changes.
RNG_TRUNC_SEED = 43

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
# NOTE: match the full ROI token, not a prefix — "ROI_parietal" is also a
# substring of "ROI_parietal_precuneus", which would silently load both ROIs
# (and let the second one overwrite the first in the per-subject index).
csv_files = [f for f in csv_files if f"ROI_{ROI_to_use}_decoding" in f]
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


# =====================================================================
# SUBJECT-LEVEL SEQUENCE VALIDATION
# =====================================================================
# A subject's sequence structure comes from their encoding behaviour, while the
# retrieved position comes from the `seq_pos` cue of the retrieval trial. If the
# two disagree, the retrieval session was run against a different sequence
# structure than the one encoded (this is the case for s41 postsleep, whose
# session was recorded under subject 42's number and therefore used subject 42's
# stimulus randomisation). Every metric for such a subject — the full 5-element
# SODA included — is computed over the wrong sequence, and trials that happen to
# agree do so by coincidence. Such subjects are dropped here, before anything is
# computed, so they cannot contribute to any output of this script.

def _validate_subject_sequences(df: pd.DataFrame) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Return (subjects to drop, {subject: (n_ok, n_mismatch)})."""
    trials = df.drop_duplicates(
        subset=["subject", "trial", "object_scene_retrieval",
                "retrieval_target", "retrieval_cue_position"]
    )
    report: dict[str, tuple[int, int]] = {}
    for subj, grp in trials.groupby("subject"):
        hier_col = df.loc[df["subject"] == subj, "full_structured_sequence_category"].dropna()
        if hier_col.empty:
            report[subj] = (0, len(grp))
            continue
        hier = ast.literal_eval(hier_col.iloc[0])
        n_ok = n_bad = 0
        for _, r in grp.iterrows():
            pos = pd.to_numeric(r["retrieval_cue_position"], errors="coerce")
            try:
                seq = lookup_5(hier, r["retrieval_target"], kind=r["object_scene_retrieval"])
                idx = seq.index(r["retrieval_target"]) + 1
            except (KeyError, ValueError):
                idx = np.nan
            if np.isnan(pos) or np.isnan(idx) or int(pos) != int(idx):
                n_bad += 1
            else:
                n_ok += 1
        report[subj] = (n_ok, n_bad)
    return [s for s, (_, bad) in report.items() if bad > 0], report


_drop_subjects, _seq_report = _validate_subject_sequences(stacked_df)
if _drop_subjects:
    print("\n" + "=" * 70)
    print("EXCLUDED SUBJECTS — encoded sequence structure does not match the")
    print(f"retrieval trials of this session ({pre_post}). No output of this")
    print("script contains any data from them.")
    for s in sorted(_drop_subjects):
        ok, bad = _seq_report[s]
        print(f"  {s}: {bad}/{ok + bad} trials fail the position cross-check")
    print("=" * 70 + "\n")
    stacked_df = stacked_df[~stacked_df["subject"].isin(_drop_subjects)].copy()
    if stacked_df.empty:
        raise ValueError("All subjects were excluded by the sequence validation.")
print(f"[subjects] {stacked_df['subject'].nunique()} retained, {len(_drop_subjects)} excluded")

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


# =====================================================================
# POSITION-TRUNCATED WINDOWS
# =====================================================================

# The "selected" shuffle of the full 5-element window (utils.get_selected_shuffles)
# additionally requires that no originally-adjacent items stay adjacent. That
# constraint is unsatisfiable for n=3 (no permutation of 3 items avoids the
# identity, the reverse and all adjacent pairs), so for the truncated windows it
# is relaxed: a selected shuffle is any permutation that is neither the identity
# nor the exact reverse. The reverse stays excluded because it yields a slope of
# identical magnitude with flipped sign, i.e. it is indistinguishable from the
# true order for the *_abs metrics. The same rule is applied at every window
# length so the null is comparable across retrieved positions.
_RELAXED_SHUFFLE_CACHE: dict[int, list[tuple[int, ...]]] = {}


def get_relaxed_shuffles(n: int) -> list[tuple[int, ...]]:
    """All permutations of range(n) except the identity and the exact reverse."""
    if n not in _RELAXED_SHUFFLE_CACHE:
        from itertools import permutations
        identity = tuple(range(n))
        reverse = tuple(range(n - 1, -1, -1))
        _RELAXED_SHUFFLE_CACHE[n] = [
            p for p in permutations(range(n)) if p != identity and p != reverse
        ]
    return _RELAXED_SHUFFLE_CACHE[n]


_trunc_perm_cache: dict[tuple, dict] = {}


def _rng_for(cache_key: tuple) -> np.random.Generator:
    """Deterministic generator for one trial × window.

    Seeding per window (rather than drawing from one shared stream) keeps the
    drawn permutations stable when unrelated parts of the script change: with a
    single shared stream, adding or removing any other random draw shifts every
    subsequent shuffle and can move a test statistic substantially.
    """
    digest = hashlib.blake2b(repr(cache_key).encode(), digest_size=8).digest()
    return np.random.default_rng(RNG_TRUNC_SEED + int.from_bytes(digest, "big"))


def _get_window_perms(cache_key: tuple, n_items: int) -> dict:
    """Cached permutations for one trial × window length (fixed across TRs)."""
    if cache_key not in _trunc_perm_cache:
        rng = _rng_for(cache_key)
        relaxed = get_relaxed_shuffles(n_items)
        _trunc_perm_cache[cache_key] = {
            "perm_ret": rng.permutation(n_items),
            "perm_ctrl": rng.permutation(n_items),
            "perm_sel_ret": relaxed[rng.integers(len(relaxed))],
            "perm_sel_ctrl": relaxed[rng.integers(len(relaxed))],
        }
    return _trunc_perm_cache[cache_key]


# --- Exhaustive permutation null -------------------------------------------
# A single drawn permutation is a noisy baseline: the same test moved from
# p = .06 to p = .0002 between two runs of identical data, purely because of
# which permutation each trial happened to draw. The relaxed permutation set is
# small enough to enumerate completely (4 for n=3, 22 for n=4, 118 for n=5), so
# instead of sampling 100 shuffles the *exact* null over all valid permutations
# is computed. It is deterministic, has no sampling error, and is cheaper.
#
# The slope is linear in the probabilities:
#     slope = sum_i w_i * x_i        with w_i = (i - mean(i)) / sum_j (j - mean(j))^2
# so the slope of every permutation is one matrix-vector product: for the
# permutation matrix row k built from perm_k, slopes = M @ x.
_NULL_MATRIX_CACHE: dict[int, np.ndarray] = {}


def get_null_matrix(n: int) -> np.ndarray:
    """(n_perms × n) matrix giving the slope of every valid permutation of x."""
    if n not in _NULL_MATRIX_CACHE:
        perms = get_relaxed_shuffles(n)
        pos = np.arange(1, n + 1, dtype=float)
        w = (pos - pos.mean()) / np.sum((pos - pos.mean()) ** 2)
        M = np.zeros((len(perms), n))
        for k, perm in enumerate(perms):
            for i, j in enumerate(perm):
                M[k, j] = w[i]
        _NULL_MATRIX_CACHE[n] = M
    return _NULL_MATRIX_CACHE[n]


def null_distribution(values: np.ndarray) -> np.ndarray:
    """Slopes of all valid permutations of *values* (NaN array if unusable)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < MIN_ELEMENTS or np.isnan(values).any():
        # NaNs would propagate through the matrix product; fall back to the
        # element-wise fitter, which drops them while preserving positions.
        if n < MIN_ELEMENTS or np.sum(~np.isnan(values)) < MIN_ELEMENTS:
            return np.array([np.nan])
        return np.array([seq_fitter(values[list(p)]) for p in get_relaxed_shuffles(n)])
    return get_null_matrix(n) @ values


def _null_summary(values: np.ndarray, prefix: str) -> dict:
    """Summarise the permutation null and locate the observed slope in it.

    Reported for the signed slope and for |slope| separately: the |slope| null
    is not the absolute value of the signed null's mean, so it has to be
    summarised over the permutations directly.

    The observed slope is located by its RANK in the null (`_null_pct`, the
    fraction of permutations below it) rather than by a z-score. A z-score is
    unusable here: with only 4 permutations at n=3 the null SD is sometimes
    ~0 and z explodes. The rank is scale-free, always defined, and is the
    natural permutation statistic — its expectation is 0.5 under the null.
    """
    slopes = null_distribution(values)
    observed = _soda(values)
    keys = ["_null_mean", "_null_sd", "_null_pct",
            "_abs_null_mean", "_abs_null_sd", "_abs_null_pct", "_n_null"]
    if np.all(np.isnan(slopes)) or np.isnan(observed):
        return {f"{prefix}{k}": np.nan for k in keys}
    a = np.abs(slopes)
    return {
        f"{prefix}_null_mean": float(slopes.mean()),
        f"{prefix}_null_sd": float(slopes.std(ddof=1)),
        f"{prefix}_null_pct": float(np.mean(slopes < observed)),
        f"{prefix}_abs_null_mean": float(a.mean()),
        f"{prefix}_abs_null_sd": float(a.std(ddof=1)),
        f"{prefix}_abs_null_pct": float(np.mean(a < abs(observed))),
        f"{prefix}_n_null": len(slopes),
    }


def _soda(values: np.ndarray) -> float:
    """SODA slope, but only if enough elements survived as non-NaN."""
    values = np.asarray(values, dtype=float)
    if len(values) < MIN_ELEMENTS or np.sum(~np.isnan(values)) < MIN_ELEMENTS:
        return np.nan
    return seq_fitter(values)


def _abs_or_nan(x: float) -> float:
    return abs(x) if not np.isnan(x) else np.nan


def window_metrics(probs_arr: np.ndarray, control_arr: np.ndarray,
                   n_items, perms: dict | None, suffix: str) -> dict:
    """SODA metrics for the first *n_items* elements of a trial.

    Returns the full column block (all NaN when the window is too short),
    so every row of the output CSV has the same schema.
    """
    keys = [
        f"retrieved_soda_{suffix}", f"control_soda_{suffix}",
        f"retrieved_soda_{suffix}_shuffled", f"control_soda_{suffix}_shuffled",
        f"retrieved_soda_{suffix}_shuffled_selected", f"control_soda_{suffix}_shuffled_selected",
    ]

    null_keys = ["_null_mean", "_null_sd", "_null_pct",
                 "_abs_null_mean", "_abs_null_sd", "_abs_null_pct", "_n_null"]

    if perms is None or n_items is None or n_items < MIN_ELEMENTS:
        out = {k: np.nan for k in keys}
        out.update({f"{k}_abs": np.nan for k in keys})
        for arm in ("retrieved", "control"):
            out.update({f"{arm}_soda_{suffix}{k}": np.nan for k in null_keys})
        out[f"n_elements_{suffix}"] = np.nan
        out[f"retrieved_probs_{suffix}"] = np.nan
        out[f"control_probs_{suffix}"] = np.nan
        return out

    ret_w = probs_arr[:n_items]
    ctrl_w = control_arr[:n_items]

    values = {
        f"retrieved_soda_{suffix}": _soda(ret_w),
        f"control_soda_{suffix}": _soda(ctrl_w),
        f"retrieved_soda_{suffix}_shuffled": _soda(ret_w[perms["perm_ret"]]),
        f"control_soda_{suffix}_shuffled": _soda(ctrl_w[perms["perm_ctrl"]]),
        f"retrieved_soda_{suffix}_shuffled_selected": _soda(ret_w[list(perms["perm_sel_ret"])]),
        f"control_soda_{suffix}_shuffled_selected": _soda(ctrl_w[list(perms["perm_sel_ctrl"])]),
    }
    out = dict(values)
    out.update({f"{k}_abs": _abs_or_nan(v) for k, v in values.items()})
    # exact permutation null over all valid orderings of the same elements
    out.update(_null_summary(ret_w, f"retrieved_soda_{suffix}"))
    out.update(_null_summary(ctrl_w, f"control_soda_{suffix}"))
    out[f"n_elements_{suffix}"] = n_items
    # plain floats: np.float64 would be written to CSV as "np.float64(...)",
    # which ast.literal_eval cannot parse downstream
    out[f"retrieved_probs_{suffix}"] = [float(x) for x in ret_w]
    out[f"control_probs_{suffix}"] = [float(x) for x in ctrl_w]
    return out


# Counters for the end-of-run report
_position_check = {"ok": 0, "mismatch": 0, "missing": 0}
_mismatch_examples: list[tuple] = []


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

    # --- Retrieved position: cross-check the behavioural position cue against
    #     the position of the target within the sequence used for the fit.
    #     These must agree; if they do not, the stored sequence order is wrong
    #     (see the sort_keys bug fixed in 4.4) and the truncation is meaningless.
    retrieved_position = pd.to_numeric(meta["retrieval_cue_position"], errors="coerce")
    index_in_sequence = (
        retrieved_sequence.index(item_to_be_retrieved) + 1
        if item_to_be_retrieved in retrieved_sequence else np.nan
    )
    if np.isnan(retrieved_position) or np.isnan(index_in_sequence):
        position_consistent = False
        _position_check["missing"] += 1
    elif int(retrieved_position) == int(index_in_sequence):
        position_consistent = True
        _position_check["ok"] += 1
    else:
        position_consistent = False
        _position_check["mismatch"] += 1
        example = (subj, row["trial"], object_scene_retrieval, item_to_be_retrieved,
                   int(retrieved_position), int(index_in_sequence))
        if len(_mismatch_examples) < 10 and example not in _mismatch_examples:
            _mismatch_examples.append(example)

    # Window lengths: 1..p (trunc) and 1..p-1 (runup, target excluded).
    # Only defined when the behavioural position and the sequence agree.
    if position_consistent:
        p = int(retrieved_position)
        n_trunc = p if p >= MIN_ELEMENTS else None
        n_runup = p - 1 if (p - 1) >= MIN_ELEMENTS else None
    else:
        n_trunc = n_runup = None

    trunc_perms = (
        _get_window_perms(trial_key + ("trunc",), n_trunc)
        if n_trunc is not None else None
    )
    runup_perms = (
        _get_window_perms(trial_key + ("runup",), n_runup)
        if n_runup is not None else None
    )

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

    row_out = {
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
        "position_in_sequence": index_in_sequence,
        "position_consistent": position_consistent,
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
    }

    # Position-truncated windows: 1..p and 1..p-1
    row_out.update(window_metrics(probs_arr, control_arr, n_trunc, trunc_perms, "trunc"))
    row_out.update(window_metrics(probs_arr, control_arr, n_runup, runup_perms, "runup"))

    results.append(row_out)

results_df = pd.DataFrame(results)

# --- Report on the position cross-check and the truncated windows ---
print(
    f"\n[position check] consistent: {_position_check['ok']}, "
    f"mismatched: {_position_check['mismatch']}, missing: {_position_check['missing']}"
)
if _position_check["mismatch"]:
    print(
        "  WARNING: the behavioural position cue disagrees with the position of the\n"
        "  target inside the sequence used for the fit. This means the stored sequence\n"
        "  order is wrong (4.4 must be re-run after the json sort_keys fix).\n"
        "  Truncated windows are NOT computed for those trials.\n"
        "  (subject, trial, kind, target, position_cue, position_in_sequence):"
    )
    for ex in _mismatch_examples:
        print(f"    {ex}")

_trials = results_df.drop_duplicates(subset=["subject", "trial", "object_scene_retrieval", "retrieval_target"])
print(
    f"[truncated windows] trials with a trunc fit: "
    f"{int(_trials['n_elements_trunc'].notna().sum())}/{len(_trials)}, "
    f"with a runup fit: {int(_trials['n_elements_runup'].notna().sum())}/{len(_trials)} "
    f"(MIN_ELEMENTS={MIN_ELEMENTS})"
)

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


# =====================================================================
# PLOT 4: Truncated-window SODA (elements 1..p) by retrieved position
# =====================================================================

trunc_positions = [p for p in positions if p >= MIN_ELEMENTS]
trunc_arms = [
    ("retrieved_soda_trunc", "Retrieved (1..p)", "blue"),
    ("control_soda_trunc", "Control (1..p)", "orange"),
    ("retrieved_soda_trunc_shuffled_selected", "Retrieved, shuffled", "green"),
    ("retrieved_soda_runup", "Retrieved (1..p-1)", "purple"),
]

if results_df["n_elements_trunc"].notna().any():
    fig, axes = plt.subplots(nrows=1, ncols=len(trunc_positions), figsize=(6 * len(trunc_positions), 4.5),
                             sharey=True, sharex=True)
    if len(trunc_positions) == 1:
        axes = [axes]

    for ax, pos in zip(axes, trunc_positions):
        df_pos = results_df[(results_df["retrieved_position"] == pos)
                            & results_df["n_elements_trunc"].notna()]
        if df_pos.empty:
            ax.set_title(f"pos={pos} (no data)")
            ax.set_xlabel("TR")
            continue

        cols = [c for c, _, _ in trunc_arms]
        subj_tr = df_pos.groupby(["subject", "tr"], as_index=False)[cols].mean()
        agg = subj_tr.groupby("tr")[cols].agg(["mean", "std", "count"])
        n_sub = subj_tr["subject"].nunique()

        for col, label, color in trunc_arms:
            if agg[(col, "count")].sum() == 0:
                continue
            m = agg[(col, "mean")]
            s = agg[(col, "std")] / np.sqrt(agg[(col, "count")])
            ax.plot(m.index, m.values, label=label, color=color, lw=2)
            ax.fill_between(m.index, (m - s).values, (m + s).values, color=color, alpha=0.15, linewidth=0)

        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        n_trials = df_pos.drop_duplicates(
            subset=["subject", "trial", "object_scene_retrieval", "retrieval_target"]
        ).shape[0]
        ax.set_title(f"pos={pos} (n_sub={n_sub}, n_trials={n_trials})")
        ax.set_xlabel("TR")

    axes[0].set_ylabel("Mean SODA (truncated window)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Truncated-window SODA: elements 1..p (p = retrieved position), ± SEM\n"
        f"ROI={ROI_to_use}, {pre_post}, MIN_ELEMENTS={MIN_ELEMENTS}",
        y=1.10,
    )
    fig.tight_layout()
    plt.savefig(opj(plt_path, f"04_truncated_soda_by_retrieved_position_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("[plot 4] skipped — no truncated fits available "
          "(re-run 4.4 after the json sort_keys fix if the position check reported mismatches).")
