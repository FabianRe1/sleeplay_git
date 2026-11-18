# -*- coding: utf-8 -*-
"""
4.7 – LME stats on SODA values across ROIs

Loads per-ROI SODA CSVs from 4.6, reshapes to long format
(one row per subject × trial × TR × condition), then fits:

    soda ~ condition + (1 | subject)

with post-hoc pairwise comparisons between the 4 conditions:
  retrieved, control, retrieved_shuffled, control_shuffled
"""

import glob
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import combinations

##################
# TOGGLE: aggregation mode across TRs
#   'raw'            → use raw signed SODA values (positive/negative may cancel)
#   'absolute'       → abs(soda) per TR before fitting
#   'peak_detection' → per trial: mean of |max across TRs| and |min across TRs|
##################
AGGREGATION_MODE = 'peak_detection'  # 'raw', 'absolute', or 'peak_detection'

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = opj(script_dir,
              '4.6_SODA_based_on_singlesubj_4.4', 'SODA_results_csv')

csv_files = sorted(glob.glob(opj(csv_dir, '*_soda_results_per_combination.csv')))
if not csv_files:
    raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

output_dir = opj(script_dir, '4.7_SODA_aggregate_boxplots')
os.makedirs(output_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Load all ROI CSVs
# ──────────────────────────────────────────────────────────────────────

all_dfs = []
for fp in csv_files:
    df = pd.read_csv(fp)
    if 'ROI' not in df.columns:
        roi_name = os.path.basename(fp).replace('_soda_results_per_combination.csv', '')
        df['ROI'] = roi_name
    all_dfs.append(df)

master_df = pd.concat(all_dfs, ignore_index=True)

rois = sorted(master_df['ROI'].unique())
print(f"Found {len(rois)} ROIs: {rois}")

# ──────────────────────────────────────────────────────────────────────
# Reshape wide → long: 4 SODA values per row → 4 rows
# ──────────────────────────────────────────────────────────────────────

soda_wide_cols = {
    'retrieved_soda':          'retrieved',
    'control_soda':            'control',
    'retrieved_soda_shuffled': 'retrieved_shuffled',
    'control_soda_shuffled':   'control_shuffled',
}

soda_col_list = list(soda_wide_cols.keys())

if AGGREGATION_MODE == 'peak_detection':
    # per trial: mean(|positive peak|, |negative peak|) across TRs
    trial_df = master_df.groupby(['ROI', 'subject', 'trial'])[soda_col_list].agg(
        lambda x: (np.abs(x.max()) + np.abs(x.min())) / 2.0
    ).reset_index()
    id_cols = ['ROI', 'subject', 'trial']
    source_df = trial_df
elif AGGREGATION_MODE == 'absolute':
    source_df = master_df.copy()
    for c in soda_col_list:
        source_df[c] = pd.to_numeric(source_df[c], errors='coerce').abs()
    id_cols = ['ROI', 'subject', 'trial', 'tr']
else:  # 'raw'
    source_df = master_df
    id_cols = ['ROI', 'subject', 'trial', 'tr']

long_frames = []
for wide_col, cond_label in soda_wide_cols.items():
    tmp = source_df[id_cols + [wide_col]].copy()
    tmp = tmp.rename(columns={wide_col: 'soda'})
    tmp['condition'] = cond_label
    long_frames.append(tmp)

long_df = pd.concat(long_frames, ignore_index=True)
long_df['soda'] = pd.to_numeric(long_df['soda'], errors='coerce')
long_df = long_df.dropna(subset=['soda'])

# make condition a categorical with 'retrieved' as reference
long_df['condition'] = pd.Categorical(
    long_df['condition'],
    categories=['retrieved', 'control', 'retrieved_shuffled', 'control_shuffled'],
    ordered=False,
)

print(f"Long-format rows: {len(long_df)}")

# ──────────────────────────────────────────────────────────────────────
# Fit LME per ROI + pairwise post-hoc contrasts
# ──────────────────────────────────────────────────────────────────────

all_omnibus = []
all_posthoc = []

conditions = ['retrieved', 'control', 'retrieved_shuffled', 'control_shuffled']
pairs = list(combinations(conditions, 2))

for roi in rois:
    print(f"\n{'='*60}\nROI: {roi}\n{'='*60}")

    roi_long = long_df[long_df['ROI'] == roi].copy()
    n_sub = roi_long['subject'].nunique()
    print(f"  {n_sub} subjects, {len(roi_long)} observations")

    # ── fit omnibus model: soda ~ condition + (1 | subject) ──
    md = smf.mixedlm('soda ~ condition', roi_long, groups=roi_long['subject'])
    mdf = md.fit(reml=True)
    print(mdf.summary())

    all_omnibus.append({
        'ROI': roi, 'n_sub': n_sub,
        # fixed effects for each condition relative to reference (retrieved)
        **{f'beta_{k}': v for k, v in mdf.fe_params.items()},
        **{f'pval_{k}': v for k, v in mdf.pvalues.items()},
    })

    # ── pairwise post-hoc: refit with each pair's first condition as reference ──
    for cond_a, cond_b in pairs:
        pair_df = roi_long[roi_long['condition'].isin([cond_a, cond_b])].copy()
        pair_df['condition'] = pd.Categorical(
            pair_df['condition'], categories=[cond_a, cond_b], ordered=False,
        )

        md_pair = smf.mixedlm('soda ~ condition', pair_df, groups=pair_df['subject'])
        mdf_pair = md_pair.fit(reml=True)

        # the second level's coefficient is the contrast
        contrast_name = f'condition[T.{cond_b}]'
        beta = mdf_pair.fe_params.get(contrast_name, np.nan)
        pval = mdf_pair.pvalues.get(contrast_name, np.nan)

        print(f"  {cond_a} vs {cond_b}: beta={beta:.5f}, p={pval:.4f}")

        all_posthoc.append({
            'ROI': roi,
            'condition_a': cond_a,
            'condition_b': cond_b,
            'beta': beta,
            'pval': pval,
        })

# ──────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────

omnibus_df = pd.DataFrame(all_omnibus)
posthoc_df = pd.DataFrame(all_posthoc)

omnibus_file = opj(output_dir, f'lme_omnibus_all_rois_{AGGREGATION_MODE}.csv')
posthoc_file = opj(output_dir, f'lme_posthoc_pairwise_all_rois_{AGGREGATION_MODE}.csv')

omnibus_df.to_csv(omnibus_file, index=False)
posthoc_df.to_csv(posthoc_file, index=False)

print(f"\nOmnibus results saved to: {omnibus_file}")
print(f"Post-hoc results saved to: {posthoc_file}")
print("\n── Post-hoc summary ──")
print(posthoc_df.to_string(index=False))
