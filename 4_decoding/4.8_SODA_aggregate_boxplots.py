# -*- coding: utf-8 -*-
"""
4.7 – Aggregate SODA boxplots across ROIs

Loads per-ROI SODA CSVs from 4.6, then for each ROI:
  1. Average SODA across TRs and trials → one value per subject
  2. Boxplot: Retrieved | Control | Retrieved(shuffled) | Control(shuffled)
  3. Paired t-tests
"""

import glob
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

##################
# TOGGLE: aggregation mode across TRs
#   'raw'            → average signed SODA across TRs (positive/negative may cancel)
#   'absolute'       → abs(soda) per TR, then average
#   'peak_detection' → per trial: mean of |max across TRs| and |min across TRs|
#   'variance'       → variance of SODA across TRs per trial, then average per subject
##################
AGGREGATION_MODE = 'variance'  # 'raw', 'absolute', 'peak_detection', or 'variance'
Pre_post = 'postsleep' # postsleep or presleep (for file paths and plot titles)
# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = opj(script_dir,
              f'4.6_SODA_based_on_singlesubj_4.4_{Pre_post}', 'SODA_results_csv')

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
print(f"Total rows: {len(master_df)}")

# ──────────────────────────────────────────────────────────────────────
# Loop over ROIs: aggregate per subject, boxplot, stats
# ──────────────────────────────────────────────────────────────────────

soda_cols = ['retrieved_soda', 'control_soda',
             'retrieved_soda_shuffled', 'control_soda_shuffled']

summary_stats = []

for roi in rois:
    print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

    roi_df = master_df[master_df['ROI'] == roi].copy()

    if AGGREGATION_MODE == 'absolute':
        # abs per TR, then average across TRs and trials per subject
        for c in soda_cols:
            roi_df[c] = roi_df[c].abs()
        subj_means = roi_df.groupby('subject')[soda_cols].mean().reset_index()

    elif AGGREGATION_MODE == 'peak_detection':
        # per trial: mean(|positive peak|, |negative peak|) across TRs, then average trials per subject
        trial_groups = roi_df.groupby(['subject', 'trial'])
        trial_peaks = trial_groups[soda_cols].agg(
            lambda x: (np.abs(x.max()) + np.abs(x.min())) / 2.0
        ).reset_index()
        subj_means = trial_peaks.groupby('subject')[soda_cols].mean().reset_index()

    elif AGGREGATION_MODE == 'variance':
        # variance of SODA across TRs per trial, then average variance per subject
        trial_groups = roi_df.groupby(['subject', 'trial'])
        trial_var = trial_groups[soda_cols].var().reset_index()
        subj_means = trial_var.groupby('subject')[soda_cols].mean().reset_index()

    else:  # 'raw'
        # plain average across TRs and trials per subject
        subj_means = roi_df.groupby('subject')[soda_cols].mean().reset_index()
    n_sub = len(subj_means)
    print(f"  {n_sub} subjects")

    # ── paired t-tests ──
    def _paired_t(col_a, col_b):
        a = subj_means[col_a].values
        b = subj_means[col_b].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 3:
            return np.nan, np.nan
        t, p = ttest_rel(a[mask], b[mask])
        return t, p

    t_rc, p_rc = _paired_t('retrieved_soda', 'control_soda')
    t_rs, p_rs = _paired_t('retrieved_soda', 'retrieved_soda_shuffled')
    t_cs, p_cs = _paired_t('control_soda', 'control_soda_shuffled')

    print(f"  Retrieved vs Control:          t={t_rc:.3f}, p={p_rc:.4f}")
    print(f"  Retrieved vs Retrieved(shuf):  t={t_rs:.3f}, p={p_rs:.4f}")
    print(f"  Control   vs Control(shuf):     t={t_cs:.3f}, p={p_cs:.4f}")

    summary_stats.append({
        'ROI': roi, 'n_sub': n_sub,
        't_ret_vs_ctrl': t_rc, 'p_ret_vs_ctrl': p_rc,
        't_ret_vs_shuf': t_rs, 'p_ret_vs_shuf': p_rs,
        't_ctrl_vs_shuf': t_cs, 'p_ctrl_vs_shuf': p_cs,
    })

    # ── boxplot ──
    labels = ['Retrieved', 'Control', 'Retrieved\n(shuffled)', 'Control\n(shuffled)']
    colors = ['#3274A1', '#E1812C', '#8CB8D0', '#F0C08A']
    data = [subj_means[c].dropna().values for c in soda_cols]

    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(data, positions=range(4), widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # individual subjects as jittered dots
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   color=color, edgecolor='k', linewidth=0.5,
                   s=30, zorder=5, alpha=0.8)

    ax.axhline(0, color='grey', ls='--', lw=0.8)

    # annotate retrieved vs control p-value
    ax.text(0.5, ax.get_ylim()[1] * 0.95, f'p={p_rc:.3f}',
            ha='center', fontsize=9, fontstyle='italic')

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    abs_tag = AGGREGATION_MODE
    ax.set_ylabel(f'Mean SODA (slope, {abs_tag})')
    ax.set_title(f'ROI: {roi}  (n={n_sub}, {abs_tag})')
    fig.tight_layout()

    fname = opj(output_dir, f'boxplot_SODA_{roi}_{abs_tag}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")

# ──────────────────────────────────────────────────────────────────────
# Save summary stats table
# ──────────────────────────────────────────────────────────────────────

stats_df = pd.DataFrame(summary_stats)
stats_file = opj(output_dir, f'summary_stats_all_rois_{AGGREGATION_MODE}.csv')
stats_df.to_csv(stats_file, index=False)
print(f"\nSummary stats saved to: {stats_file}")
print(stats_df.to_string(index=False))
