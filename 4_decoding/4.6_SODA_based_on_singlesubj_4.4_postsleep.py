# -*- coding: utf-8 -*-


'''
========================================================================
IMPORT PACKAGES:
========================================================================
'''
import glob
import os
from os.path import join as opj
import pandas as pd
import copy
import sys
#import scipy
import numpy as np
#import datalad.api as dl
#from scipy.stats import sem, spearmanr, pearsonr
from nilearn import image, masking, signal, plotting
#from nilearn.decoding import Decoder
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel
#from sklearn import metrics
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from groo.groo import get_root
from nilearn.maskers import NiftiMasker
from joblib import dump, load
from tqdm import tqdm
import re
import ast
import json
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from typing import Literal, Any
import sys

##################
# SET PARAMETERS
##################
ROI_to_use = sys.argv[1]  # options: 'vis', cortical_memory''
shuffle_control = True  # whether to shuffle control probs before fitting
##################



def seq_fitter(x, method='lm'):
    """
    Fit a sequentiality slope to classifier probabilities ordered by sequence position.

    Uses ascending predictor [1, 2, 3, ...] following Wittkuhn & Schuck (2021) convention:
      - Negative slope at peak of early items (item 1 high, item N low)
      - Positive slope at peak of late items (item 1 low, item N high)

    NaN handling: NaNs are dropped but original position indices are preserved,
    so the slope correctly reflects the spacing of surviving items.

    Parameters
    ----------
    x : array-like
        Classifier probabilities ordered by sequence position (item 1 first).
    method : str
        'lm' for linear regression slope, 'spearman' for Spearman correlation.

    Returns
    -------
    float or np.nan
    """
    x = np.asarray(x, dtype=float)

    # Ascending position predictor [1, 2, 3, ..., N] matching original positions
    positions = np.arange(1, len(x) + 1, dtype=float)

    # Keep position mapping intact when dropping NaNs
    valid = ~np.isnan(x)
    x_valid = x[valid]
    pos_valid = positions[valid]

    # Need at least 3 data points for a meaningful slope
    if len(x_valid) < 3:
        return np.nan

    if method == 'lm':
        slope = np.polyfit(pos_valid, x_valid, 1)[0]
        return slope
    elif method == 'spearman':
        corr, _ = spearmanr(pos_valid, x_valid)
        return corr
    else:
        raise ValueError(f"Unknown method: {method}")




HierarchyType = dict[str, dict[str, list[str]]]
def lookup_5(
    hier: HierarchyType,
    value: str,
    kind: Literal["object", "scene"],
    *,
    return_all_for_object: bool = False,
) -> Any:
    """
    kind='scene': value is a scene filename (e.g., 'autumn03.jpg')
        -> returns list of 5 scene keys from the corresponding top-level block ('day'/'night').

    kind='object': value is an object/category string (e.g., 'fruit')
        -> returns the list-of-5 strings for the first scene that contains it,
           or (if return_all_for_object=True) returns all matching lists.
    """
    if kind == "scene":
        for top_key, scenes in hier.items():          # top_key: 'day'/'night'
            if value in scenes:                       # value is a scene key
                return list(scenes.keys())            # the 5 scene keys in that block
        raise KeyError(f"Scene '{value}' not found in any top-level group.")

    if kind == "object":
        matches = []
        for top_key, scenes in hier.items():
            for scene_key, obj_list in scenes.items():
                if value in obj_list:
                    if return_all_for_object:
                        matches.append(obj_list)
                    else:
                        return obj_list
        if return_all_for_object and matches:
            # de-duplicate identical lists (optional)
            uniq = []
            for m in matches:
                if m not in uniq:
                    uniq.append(m)
            return uniq
        raise KeyError(f"Object '{value}' not found in any scene list.")

    raise ValueError("kind must be 'object' or 'scene'.")

def control_5(
        hier: HierarchyType,
        value: str,
        kind: Literal["object", "scene"],
    ) -> list[str]:
        """
        Returns the *other* 5-item sequence at the same hierarchy level as `lookup_5(...)`.

        - kind='scene': value is a scene filename -> returns the 5 scene keys from the *other* top-level block
        - kind='object': value is an object/category string -> returns the 5-object list from the *other* object-set
        """
        if kind == "scene":
            # find whether scene is in 'day' or 'night'
            scene_group = None
            for top_key, scenes in hier.items():
                if value in scenes:
                    scene_group = top_key
                    break
            if scene_group is None:
                raise KeyError(f"Scene '{value}' not found in any top-level group.")
            other_group = [k for k in hier.keys() if k != scene_group]
            if len(other_group) != 1:
                raise ValueError(f"Expected exactly 2 top-level groups, got: {list(hier.keys())}")
            return list(hier[other_group[0]].keys())

        if kind == "object":
            retrieved = lookup_5(hier, value, kind="object")  # the 5 objects for the set containing `value`

            # collect the unique object-sets present in the hierarchy
            uniq_sets: list[list[str]] = []
            for _, scenes in hier.items():
                for _, obj_list in scenes.items():
                    if obj_list not in uniq_sets:
                        uniq_sets.append(obj_list)

            # choose the other set (different from retrieved)
            other_sets = [s for s in uniq_sets if s != retrieved]
            if len(other_sets) == 0:
                raise ValueError("Could not find a complementary object-set in the hierarchy.")
            if len(other_sets) > 1:
                # if you ever add more than two object-sets, decide how to pick a control set here
                raise ValueError(f"Found >1 complementary object-set. Need a rule. Sets={other_sets}")

            return other_sets[0]

        raise ValueError("kind must be 'object' or 'scene'.")


def create_script_named_folder(folder_name=None):
    """
    Creates a new directory with the name of the script file (without extension) or with the provided folder name.
    
    Args:
        folder_name (str, optional): The name of the folder to be created inside the script directory. 
            If not provided, the folder will have the same name as the script file (without extension).
    
    Returns:
        str: The path of the newly created directory.
    """
    
    # Assuming __file__ represents the current file name in a global scope
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


path_to_csvs = './4.5_transfer_visual_classifier_retrieval_postsleep/decoding_results_tables_for_SODA'

# Collect all CSV files
csv_files = sorted(glob.glob(opj(path_to_csvs, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(path_to_csvs)}")




# filter csv files based on ROI
if ROI_to_use == 'vis':
    csv_files = [f for f in csv_files if 'ROI_vis' in f]
elif ROI_to_use == 'cortical_memory':
    csv_files = [f for f in csv_files if 'ROI_cortical_memory' in f]
elif ROI_to_use == 'mtl':
    csv_files = [f for f in csv_files if 'ROI_mtl' in f]
elif ROI_to_use == 'parietal_precuneus':
    csv_files = [f for f in csv_files if 'ROI_parietal_precuneus' in f]
elif ROI_to_use == 'parietal':
    csv_files = [f for f in csv_files if 'ROI_parietal_decoding' in f]
elif ROI_to_use == 'precuneus':
    csv_files = [f for f in csv_files if 'ROI_precuneus' in f]
elif ROI_to_use == 'medial_ofc':
    csv_files = [f for f in csv_files if 'ROI_medial_ofc' in f]
elif ROI_to_use == 'rostral_acc':
    csv_files = [f for f in csv_files if 'ROI_rostral_acc' in f]
elif ROI_to_use == 'caudal_acc':
    csv_files = [f for f in csv_files if 'ROI_caudal_acc' in f]
elif ROI_to_use == 'superior_frontal':
    csv_files = [f for f in csv_files if 'ROI_superior_frontal' in f]
elif ROI_to_use == 'rsc':
    csv_files = [f for f in csv_files if 'ROI_rsc' in f]
elif ROI_to_use == 'posterior_cingulate':
    csv_files = [f for f in csv_files if 'ROI_posterior_cingulate' in f]
else:
    raise ValueError(f"Unknown ROI_to_use: {ROI_to_use}")

dfs = []
sub_re = re.compile(r"(sub-[A-Za-z0-9]+)")

for fpath in tqdm(csv_files, desc="Loading subject CSVs"):
    df = pd.read_csv(fpath)
    dfs.append(df)

plt_path = create_script_named_folder('SODA_plots')  # create folder named after script

########################################################################
#%% Plot prob of retrieved vs the other conditions
########################################################################
stacked_df = pd.concat(dfs, ignore_index=True, sort=False)
scenes1_objects2_both3 = 3 # 

if scenes1_objects2_both3 == 1:
    stacked_df = stacked_df[stacked_df["object_scene_retrieval"] == "scene"].copy()
elif scenes1_objects2_both3 == 2:
    stacked_df = stacked_df[stacked_df["object_scene_retrieval"] == "object"].copy()
# else: keep both


# --- build subject-averaged timecourses per TR and condition ---
d = stacked_df[["subject", "tr", "target_classifier_matched", "probability_from_classifier"]].copy()
d["tr"] = pd.to_numeric(d["tr"], errors="coerce")
d["probability_from_classifier"] = pd.to_numeric(d["probability_from_classifier"], errors="coerce")
d["target_classifier_matched"] = pd.to_numeric(d["target_classifier_matched"], errors="coerce")

d = d.dropna(subset=["subject", "tr", "target_classifier_matched", "probability_from_classifier"])
d["target_classifier_matched"] = d["target_classifier_matched"].astype(int)

# Average within subject first -> SEM/test are across subjects (not across trials)
subj_tr_mean = (
    d.groupby(["subject", "target_classifier_matched", "tr"], as_index=False)["probability_from_classifier"]
    .mean()
)

def _pivot_condition(df: pd.DataFrame, matched_value: int) -> pd.DataFrame:
    tmp = df[df["target_classifier_matched"] == matched_value]
    # rows = TRs, columns = subjects
    wide = tmp.pivot(index="tr", columns="subject", values="probability_from_classifier").sort_index()
    return wide

wide_unmatched = _pivot_condition(subj_tr_mean, 0)  # tr x subject
wide_matched = _pivot_condition(subj_tr_mean, 1)    # tr x subject

def _mean_and_sem(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    mean = wide.mean(axis=1, skipna=True)
    n = wide.count(axis=1)
    sd = wide.std(axis=1, ddof=1)
    sem = sd / np.sqrt(n)
    return mean, sem, n

mean_unmatched, sem_unmatched, n_unmatched = _mean_and_sem(wide_unmatched)
mean_matched, sem_matched, n_matched = _mean_and_sem(wide_matched)

# --- paired t-test per TR across subjects + FDR(BH) across TRs ---
all_trs = sorted(set(wide_unmatched.index).union(set(wide_matched.index)))

pvals = np.full(len(all_trs), np.nan, dtype=float)
n_pair = np.zeros(len(all_trs), dtype=int)

for i, tr in enumerate(all_trs):
    if (tr not in wide_unmatched.index) or (tr not in wide_matched.index):
        continue

    a = wide_matched.loc[tr]
    b = wide_unmatched.loc[tr]
    ok = (~a.isna()) & (~b.isna())
    n = int(ok.sum())
    n_pair[i] = n

    if n < 3:
        continue

    _, p = ttest_rel(a[ok].values, b[ok].values)
    pvals[i] = p

valid = ~np.isnan(pvals)
rejected_fdr = np.zeros_like(valid, dtype=bool)
qvals = np.full_like(pvals, np.nan, dtype=float)

if np.any(valid):
    rejected_valid, qvals_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
    rejected_fdr[valid] = rejected_valid
    qvals[valid] = qvals_valid

# --- plot (per-subject lines + mean ± SEM) ---
plt.figure(figsize=(10, 5))

# per-subject lines (low alpha)
for sub in wide_unmatched.columns:
    plt.plot(wide_unmatched.index, wide_unmatched[sub].values, color="0.6", alpha=0.3, lw=1)
for sub in wide_matched.columns:
    plt.plot(wide_matched.index, wide_matched[sub].values, color="#034e87", alpha=0.3, lw=1)

# mean + SEM
plt.fill_between(
    mean_unmatched.index,
    (mean_unmatched - sem_unmatched).values,
    (mean_unmatched + sem_unmatched).values,
    color="0.5",
    alpha=0.20,
    linewidth=0,
)
plt.plot(
    mean_unmatched.index,
    mean_unmatched.values,
    color="0.35",
    lw=3,
    label=f"unmatched (n_sub={wide_unmatched.shape[1]})",
)

plt.fill_between(
    mean_matched.index,
    (mean_matched - sem_matched).values,
    (mean_matched + sem_matched).values,
    color="#034e87",
    alpha=0.25,
    linewidth=0,
)
plt.plot(
    mean_matched.index,
    mean_matched.values,
    color="#034e87",
    lw=3,
    label=f"matched (n_sub={wide_matched.shape[1]})",
)

# mark FDR-significant TRs (q<0.05) at top of plot
sig_trs = [tr for tr, rej in zip(all_trs, rejected_fdr) if rej]
if sig_trs:
    y_sig = 0.18  # within ylim
    plt.scatter(sig_trs, [y_sig] * len(sig_trs), s=24, color="k", marker="*", zorder=10, label="FDR q<0.05")

plt.xlabel("TR")
plt.ylabel("probability_from_classifier")
plt.title(f"Matched vs unmatched: per subject + mean ± SEM; paired t-test per TR (FDR)\n scenes1_objects2_both3={scenes1_objects2_both3}\nROI={ROI_to_use}, shuffle_control={shuffle_control}")
plt.ylim(0.05, 0.2)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(opj(plt_path, f'01_matched_vs_unmatched_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png'), dpi=150, bbox_inches='tight')
plt.show()

########################################################################
#%% Now compute SODA values per unique combination
########################################################################


# get unique combinations of:
group_columns = ['subject', 'trial', 'tr', 'object_scene_retrieval', 'retrieval_target']
unique_combinations = stacked_df[group_columns].drop_duplicates()

# Create an empty list to store the results
results = []

# Loop over each unique combination
for _, row in tqdm(unique_combinations.iterrows()):
    
    stacked_df_subj = stacked_df[stacked_df['subject'] == row['subject']]
    
    # extract the hier sturcture
    hierarcical_structure = ast.literal_eval(stacked_df_subj['full_structured_sequence_category'].iloc[0])
    hierarcical_structure_obj = ast.literal_eval(stacked_df_subj['full_structured_sequence_object'].iloc[0])
    
    # Filter the DataFrame for the current combination, handling NaN comparisons properly.
    subset = stacked_df_subj[
        (stacked_df_subj['subject'] == row['subject']) &
        (stacked_df_subj['tr'] == row['tr']) &
        (stacked_df_subj['trial'] == row['trial']) &
        (stacked_df_subj['object_scene_retrieval'] == row['object_scene_retrieval']) &
        (stacked_df_subj['retrieval_target'] == row['retrieval_target']) 
    ]
    
    item_to_be_retrieved = row['retrieval_target']
    retrieval_cue = subset['retrieval_cue'].iloc[0]
    object_scene_retrieval = subset['object_scene_retrieval'].iloc[0]
    
    # Get the full sequence to be retrieved:
    retrieved_sequence = lookup_5(hierarcical_structure, item_to_be_retrieved, kind=object_scene_retrieval)
    
    # Loop over retrieved sequence and extract probabilities from subset
    probs = []
    for item in retrieved_sequence:
        prob_row = subset[subset['stimulus_class_from_classifier'] == item]
        if not prob_row.empty:
            prob = prob_row['probability_from_classifier'].values[0]
        else:
            prob = np.nan  # or some default value if not found
        probs.append(prob)
        

    control_sequence = control_5(hierarcical_structure, item_to_be_retrieved, kind=object_scene_retrieval)

    # keep ONLY those 5 controls (and in that control-sequence order)
    control_probs = []
    for item in control_sequence:
        prob_row = subset[subset["stimulus_class_from_classifier"] == item]
        control_probs.append(prob_row["probability_from_classifier"].values[0] if not prob_row.empty else np.nan)

    control_probs_shuffle = np.random.permutation(control_probs).tolist()
    probs_shuffle = np.random.permutation(probs).tolist()

    # Fit sequence to retrieved probs
    retrieved_soda = seq_fitter(np.array(probs), method='lm')
    control_soda = seq_fitter(np.array(control_probs), method='lm')
    
    # Fit sequence to shuffled control probs
    retrieved_soda_shuffle = seq_fitter(np.array(probs_shuffle), method='lm')
    control_soda_shuffle = seq_fitter(np.array(control_probs_shuffle), method='lm')
    
    # Store the results
    result = {
        'ROI': ROI_to_use,
        'subject': row['subject'],
        'trial': row['trial'],
        'run': subset['run'].iloc[0],
        'tr': row['tr'],
        'object_scene_retrieval': row['object_scene_retrieval'],
        'retrieval_target': row['retrieval_target'], # item to be retrieved (category level)
        'retrieval_target_precise': subset['retrieval_target_precise'].iloc[0], # precise item to be retrieved (image level)
        'retrieval_cue': retrieval_cue, # item that served as cue during retrieval
        'retrieved_position': subset['retrieval_cue_position'].iloc[0],     # position of the cue in the sequence
        'retrieved_soda': retrieved_soda, # SODA value for the retrieved sequence
        'control_soda': control_soda, # SODA value for the control items
        'retrieved_soda_shuffled': retrieved_soda_shuffle, # SODA value for the shuffled retrieved sequence
        'control_soda_shuffled': control_soda_shuffle, # SODA value for the shuffled control items
        'retrieved_probs': probs, # probabilities for the retrieved sequence (items 1-5)
        'control_probs': control_probs, # probabilities for the control items
        'retrieved_probs_shuffle': probs_shuffle, # shuffled probabilities for the retrieved sequence (items 1-5)
        'control_probs_shuffle': control_probs_shuffle, # shuffled probabilities for the control items
        'behav_correct': subset['behav_correct'].iloc[0], # whether the retrieval was correct (1) or incorrect (0)
        'full_structured_sequence_category': hierarcical_structure, 
        'full_structured_sequence_object': hierarcical_structure_obj
    }
    results.append(result)
    
# Convert results list to DataFrame
results_df = pd.DataFrame(results)


# save results_df to csv
output_csv_path = create_script_named_folder('SODA_results_csv')  # create folder named after script
output_csv_file = opj(output_csv_path, f'{ROI_to_use}_soda_results_per_combination.csv')
results_df.to_csv(output_csv_file, index=False)

########################################################################
# Plotting
########################################################################

# --- compute mean ± SEM across subjects (average within subject first) ---
subj_tr = (
    results_df.groupby(["subject", "tr"], as_index=False)[["retrieved_soda", "control_soda"]]
    .mean()
)

agg = subj_tr.groupby("tr")[["retrieved_soda", "control_soda"]].agg(["mean", "std", "count"])
mean_retrieved_soda = agg[("retrieved_soda", "mean")]
mean_control_soda = agg[("control_soda", "mean")]

sem_retrieved_soda = agg[("retrieved_soda", "std")] / np.sqrt(agg[("retrieved_soda", "count")])
sem_control_soda = agg[("control_soda", "std")] / np.sqrt(agg[("control_soda", "count")])

# --- plot mean ± SEM per TR ---
plt.figure(figsize=(10, 5))

plt.plot(mean_retrieved_soda.index, mean_retrieved_soda.values, label="Retrieved SODA", color="blue", lw=2)
plt.fill_between(
    mean_retrieved_soda.index,
    (mean_retrieved_soda - sem_retrieved_soda).values,
    (mean_retrieved_soda + sem_retrieved_soda).values,
    color="blue",
    alpha=0.2,
    linewidth=0,
)

plt.plot(mean_control_soda.index, mean_control_soda.values, label="Control SODA", color="orange", lw=2)
plt.fill_between(
    mean_control_soda.index,
    (mean_control_soda - sem_control_soda).values,
    (mean_control_soda + sem_control_soda).values,
    color="orange",
    alpha=0.2,
    linewidth=0,
)

# ylilm -0.02 to 0.02 
plt.ylim(-0.02, 0.02)

plt.xlabel("TR")
plt.ylabel("Mean SODA")
plt.title(f"Mean Retrieved SODA vs Control SODA per TR (± SEM across subjects)\nROI={ROI_to_use}, shuffle_control={shuffle_control}")
plt.legend()
plt.tight_layout()
plt.savefig(opj(plt_path, f'02_retrieved_vs_control_soda_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png'), dpi=150, bbox_inches='tight')
plt.show()


########################################################################
# Plotting: 5 panels by retrieved_position + titles w/ meanTR and abs-meanTR
########################################################################

results_df["retrieved_position"] = pd.to_numeric(results_df["retrieved_position"], errors="coerce")
results_df["tr"] = pd.to_numeric(results_df["tr"], errors="coerce")

positions = [1, 2, 3, 4, 5]  # adjust if needed

fig, axes = plt.subplots(
    nrows=1, ncols=len(positions),
    figsize=(24, 4.5),
    sharey=True,
    sharex=True,
)

if len(positions) == 1:
    axes = [axes]

# collect per-subject summaries for the trend test
pos_summaries = []

for ax, pos in zip(axes, positions):
    df_pos = results_df[results_df["retrieved_position"] == pos].copy()

    if df_pos.empty:
        ax.set_title(f"pos={pos} (no data)")
        ax.set_xlabel("TR")
        continue

    # average within subject first (across trials etc.)
    subj_tr = (
        df_pos.groupby(["subject", "tr"], as_index=False)[["retrieved_soda", "control_soda"]]
        .mean()
    )

    # ---- per-panel summary numbers (computed per subject, then averaged across subjects) ----
    subj_meanTR = subj_tr.groupby("subject")[["retrieved_soda", "control_soda"]].mean()
    subj_absmeanTR = subj_tr.groupby("subject")[["retrieved_soda", "control_soda"]].apply(lambda x: x.abs().mean())

    meanTR_retrieved = float(subj_meanTR["retrieved_soda"].mean())
    meanTR_control = float(subj_meanTR["control_soda"].mean())
    absmeanTR_retrieved = float(subj_absmeanTR["retrieved_soda"].mean())
    absmeanTR_control = float(subj_absmeanTR["control_soda"].mean())
    n_sub = int(subj_meanTR.shape[0])

    # store for the across-position test (use delta; change to retrieved_soda if you prefer)
    tmp = subj_meanTR.reset_index()
    tmp["retrieved_position"] = pos
    tmp["mean_delta"] = tmp["retrieved_soda"] - tmp["control_soda"]
    pos_summaries.append(tmp[["subject", "retrieved_position", "retrieved_soda", "control_soda", "mean_delta"]])

    # ---- timecourse mean ± SEM across subjects ----
    agg = subj_tr.groupby("tr")[["retrieved_soda", "control_soda"]].agg(["mean", "std", "count"])
    mean_retrieved_soda = agg[("retrieved_soda", "mean")]
    mean_control_soda = agg[("control_soda", "mean")]

    sem_retrieved_soda = agg[("retrieved_soda", "std")] / np.sqrt(agg[("retrieved_soda", "count")])
    sem_control_soda = agg[("control_soda", "std")] / np.sqrt(agg[("control_soda", "count")])

    ax.plot(mean_retrieved_soda.index, mean_retrieved_soda.values, label="Retrieved", color="blue", lw=2)
    ax.fill_between(
        mean_retrieved_soda.index,
        (mean_retrieved_soda - sem_retrieved_soda).values,
        (mean_retrieved_soda + sem_retrieved_soda).values,
        color="blue",
        alpha=0.2,
        linewidth=0,
    )

    ax.plot(mean_control_soda.index, mean_control_soda.values, label="Control", color="orange", lw=2)
    ax.fill_between(
        mean_control_soda.index,
        (mean_control_soda - sem_control_soda).values,
        (mean_control_soda + sem_control_soda).values,
        color="orange",
        alpha=0.2,
        linewidth=0,
    )

    ax.set_title(
        f"pos={pos} (n_sub={n_sub})\n"
        f"meanTR R={meanTR_retrieved:.3f}, C={meanTR_control:.3f} | "
        f"absmeanTR R={absmeanTR_retrieved:.3f}, C={absmeanTR_control:.3f}"
    )
    ax.set_xlabel("TR")

axes[0].set_ylabel("Mean SODA")
axes[-1].legend(frameon=False, loc="best")

fig.suptitle(f"Mean Retrieved SODA vs Control SODA per TR (± SEM), split by retrieved position\nROI={ROI_to_use}, shuffle_control={shuffle_control}", y=1.10)
fig.tight_layout()
plt.savefig(opj(plt_path, f'03_soda_by_retrieved_position_ROI-{ROI_to_use}_shuffle-{shuffle_control}.png'), dpi=150, bbox_inches='tight')
plt.show()