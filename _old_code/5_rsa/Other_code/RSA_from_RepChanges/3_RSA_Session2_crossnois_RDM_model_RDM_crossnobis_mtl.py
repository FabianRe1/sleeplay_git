#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import glob
import os
import tqdm
from os.path import join as opj
import pandas as pd
import copy
import sys
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from groo.groo import get_root
import os
import itertools
import seaborn as sns
from scipy.stats import spearmanr
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import rsatoolbox
import rsatoolbox.rdm as rsr
import pickle
import matplotlib.cm as cm
from scipy.stats import spearmanr # Ensure this is imported
import statsmodels.formula.api as smf

plt.rcParams.update({
    'font.size': 11,           # Increased by 2
    'axes.titlesize': 12,      # Increased by 2
    'axes.labelsize': 11,      # Increased by 2
    'xtick.labelsize': 8,     # Increased by 2
    'ytick.labelsize': 8,     # Increased by 2
    'legend.fontsize': 10,     # Increased by 2
    'lines.linewidth': 1,
    'axes.linewidth': 0.8
})
############
# THIS FILE NEEDS TO BE SUBMITTED ONCE PER ROI BUT PROCESSES ALL SUBJECTS
############

'''
========================================================================
# 1. Load basic functions
========================================================================
'''

# Function to autolabel bars (optional, for displaying the mean values on the bars)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
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

#%%
'''
========================================================================
# 2 Setup dirs, correlation measure and load model RDMs
========================================================================
'''


# Load the mask name
try:
    mask_name = sys.argv[2]
    print('This should be the mask name:', mask_name)
except:
    mask_name = 'mtl'
    print('cave this is run locally, using mtl as mask name')

# List of subjects
subs = [f'sub-{i}' for i in range(1, 53)]

# Get the root folder
root_folder = get_root(".my_hidden_root_file")
# Get the root folder
path_root = opj(root_folder, 'Code', '7_RSA')

corr_measure = 'spearman' # 'spearman' or 'tau'

models_to_analyse = [# 'random_rsm_binary_inverted_to_RDM',

                    # 'same_branch_rsm_inverted_to_RDM_dissimilar1',
                    # 'sequence_position_rsm_inverted_to_RDM_dissimilar1',
                    # 'object_category_rsm_inverted_to_RDM_dissimilar1',
                    # 'diagonal_rsm_inverted_to_RDM_dissimilar1',


                    # 'shared_reward_rsm_inverted_to_RDM',
                    # 'shared_reward_rsm_past_structure_change_inverted_to_RDM',
                    # 'RSM_shared_reward_minus_object_category_inverted_to_RDM',
                    # 'shared_reward_MINUS_same_branch_rsm_inverted_to_RDM',
                    
                    # 'shared_reward_rsm_minus_same_branch_past_structure_change_inverted_to_RDM_nan',
                    # 'RSM_shared_reward_minus_object_category_inverted_to_RDM_nan',
                    # 'shared_reward_rsm_inverted_to_RDM_nan_dissimilar1',
                    
                    # 'ortho_diagonal_rsm_dissimilar1',
                    'ortho_object_category_rsm_dissimilar1',
                    'ortho_sequence_position_rsm_dissimilar1',
                    'ortho_shared_branch_rsm_dissimilar1',
                    'ortho_shared_reward_rsm_dissimilar1',
                    
                    ]

model_rsm_files = []

# Iterate through each model name and glob the corresponding files
for model_name in models_to_analyse:
    file_pattern = os.path.join(path_root, 'Rewardlearning_reliability', '0_model_rsm_setup_RDM', f'{model_name}.csv')
    model_rsm_files.extend(glob.glob(file_pattern))

model_rsm_dfs = {}
for file in model_rsm_files:
    
    print(file)
    if 'inverted_to_RDM' in file:
        pass
    elif 'ortho' in file:
        pass
    else:
        continue
    df_name = os.path.splitext(os.path.basename(file))[0]
    print(df_name)
    
    model_rsm_dfs[df_name] = pd.read_csv(file, index_col=0)
    
    # invert the model RSM to match the empirical RSM
    model_rsm_dfs[df_name] = 1 - model_rsm_dfs[df_name]
    
    # plot model RSM with increased font and tick sizes
    plt.figure(figsize=(3.57, 3))
    import matplotlib.colors as mcolors
    # Create a ListedColormap for two discrete values: 0 will be darkblue, 1 will be grey.
    cmap = mcolors.ListedColormap(["#023858", "#d3d3d3"])
    # Set nan values to be white.
    cmap.set_bad("white")
    # Define boundaries such that values near 0 and 1 are correctly mapped.
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    ax = sns.heatmap(model_rsm_dfs[df_name], cmap=cmap, cbar=True, norm=norm)
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['0', '1'])
    
    x_labels = [label.replace('star', '*') for label in model_rsm_dfs[df_name].index]
    y_labels = [label.replace('star', '*') for label in model_rsm_dfs[df_name].index]
    

    # Set tick labels without specifying font sizes
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels, rotation=0)
    # ax.tick_params(axis='both', which='major')
    # Set title mapping to the model name with "Orthogonalized" style
    if 'ortho_object_category_rsm_dissimilar1' in file:
        plt.title('Orthogonalized object category model')
    elif 'ortho_sequence_position_rsm_dissimilar1' in file:
        plt.title('Orthogonalized sequence position model')
    elif 'ortho_shared_branch_rsm_dissimilar1' in file:
        plt.title('Orthogonalized shared branch model')
    elif 'ortho_shared_reward_rsm_dissimilar1' in file:
        plt.title('Orthogonalized shared reward model')
    elif 'ortho_diagonal_rsm_dissimilar1' in file:
        plt.title('Orthogonalized Diagonal model')
    save_rdm = create_script_named_folder('model_rdm_plots')
    # Save the plot
    plt.savefig(opj(save_rdm, f'{df_name}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()



#%%
'''
========================================================================
# 3 loop over subejcts and load RDMS
========================================================================
'''

correlation_dict = {}
correlation_dict_session4 = {}
lme_matrices_per_mask = {}
lme_matrices_per_mask[mask_name] = []

for session in ['2', '4']:
    for sub in subs:
        
        if session == '2':
            if sub not in correlation_dict:
                correlation_dict[sub] = {}
            if mask_name not in correlation_dict[sub]:
                correlation_dict[sub][mask_name] = {}
        elif session == '4':
            if sub not in correlation_dict_session4:
                correlation_dict_session4[sub] = {}
            if mask_name not in correlation_dict_session4[sub]:
                correlation_dict_session4[sub][mask_name] = {}

        paths_to_check_run1_day1 = opj(path_root, 'Rewardlearning_reliability', 'RSA_toolbox_Runs_1_2_vs_4_5_crossnobis', f'2_RSA_Session2_createRDM_RSA_toolbox_crossnobis', 'rdms_from_rsatoolbox_with_residual_crossnobis', f'{sub}_mask_{mask_name}_rdm_cv_run_2_and_run_3.pkl')
        paths_to_check_run4_day1 = opj(path_root, 'Rewardlearning_reliability', 'RSA_toolbox_Runs_1_2_vs_4_5_crossnobis', f'2_RSA_Session2_createRDM_RSA_toolbox_crossnobis', 'rdms_from_rsatoolbox_with_residual_crossnobis', f'{sub}_mask_{mask_name}_rdm_cv_run_5_and_run_6.pkl')
        paths_to_check_run1_day2 = opj(path_root, 'Rewardlearning_reliability', 'RSA_toolbox_Runs_1_2_vs_4_5_crossnobis', f'2_RSA_Session4_createRDM_RSA_toolbox_crossnobis', 'rdms_from_rsatoolbox_with_residual_crossnobis', f'{sub}_mask_{mask_name}_rdm_cv_run_2_and_run_3.pkl')
        paths_to_check_run4_day2 = opj(path_root, 'Rewardlearning_reliability', 'RSA_toolbox_Runs_1_2_vs_4_5_crossnobis', f'2_RSA_Session4_createRDM_RSA_toolbox_crossnobis', 'rdms_from_rsatoolbox_with_residual_crossnobis', f'{sub}_mask_{mask_name}_rdm_cv_run_5_and_run_6.pkl')
        paths_to_check = [paths_to_check_run1_day1, paths_to_check_run4_day1, paths_to_check_run1_day2, paths_to_check_run4_day2]

        ##################################
        # check if all files exist    
        # ##################################
        missing_files = []
        for path in paths_to_check:
            if not os.path.exists(path):
                missing_files.append(path)

        if len(missing_files) == 0:
            print(f'Everything is there - starting with sub {sub}')
            if session == '2':
                paths_to_check_run1 = paths_to_check_run1_day1
                paths_to_check_run4 = paths_to_check_run4_day1
            elif session == '4':
                paths_to_check_run1 = paths_to_check_run1_day2
                paths_to_check_run4 = paths_to_check_run4_day2
            else:
                raise ValueError('Session must be either 2 or 4')
        else:
            print(f'{sub} {mask_name} is missing the following files:')
            for file in missing_files:
                print(file)
            continue

        # Load the correlation matrices
        rdm_curr_subj_curr_mask_run1= rsr.load_rdm(paths_to_check_run1)
        rdm_curr_subj_curr_mask_run4= rsr.load_rdm(paths_to_check_run4)

        # Extract matrices
        rdm_curr_subj_curr_mask_run1_matrix = rdm_curr_subj_curr_mask_run1.get_matrices()[0,:,:]
        rdm_curr_subj_curr_mask_run4_matrix = rdm_curr_subj_curr_mask_run4.get_matrices()[0,:,:]

        '''
        ========================================================================
        # 4 loop over model rdm and compare to empirical RDMs
        ========================================================================
        '''

        # Generate a df with data RSM and model RSM next to pre and post 
        df = pd.DataFrame()
        
        # Loop over model RSMs and compare to empirical RSMs
        for model_rsm_name, model_rsm_df in model_rsm_dfs.items():
            
            if session == '2':
                if model_rsm_name not in correlation_dict[sub][mask_name]:
                    correlation_dict[sub][mask_name][model_rsm_name] = {}
            elif session == '4':
                if model_rsm_name not in correlation_dict_session4[sub][mask_name]:
                    correlation_dict_session4[sub][mask_name][model_rsm_name] = {}

            # copy empiricla RDMs
            rdm_matrix_run1 = rdm_curr_subj_curr_mask_run1_matrix.copy()
            rdm_matrix_run4 = rdm_curr_subj_curr_mask_run4_matrix.copy()

            # create mask to only keep upper triangle
            mask = np.triu(np.ones(rdm_matrix_run1.shape), k=1).astype(bool)

            # Flatten the correlation matrix and apply the mask
            rdm_matrix_run1_uppertriangle = rdm_matrix_run1[mask]
            rdm_matrix_run4_uppertriangle = rdm_matrix_run4[mask]
            model_rdm = np.array(model_rsm_df)[mask]
            
            # # Save model RSMs and empirical RSMs in df WITHOUT DIAGONAL
            df['empirical_RDM'] = np.hstack((rdm_matrix_run1_uppertriangle, rdm_matrix_run4_uppertriangle))
            df['subject'] = sub
            df['session'] = session
            df['mask'] = mask_name
            df[model_rsm_name] = np.hstack((model_rdm, model_rdm))            
            df['phase_early1_late2'] = np.hstack((
                np.ones_like(rdm_matrix_run1_uppertriangle),
                np.ones_like(rdm_matrix_run1_uppertriangle) * 2,
            ))
            
            # set all entries where model_rdm is nan to nan in the empirical RDMs
            rdm_matrix_run1_uppertriangle[np.isnan(model_rdm)] = np.nan
            rdm_matrix_run4_uppertriangle[np.isnan(model_rdm)] = np.nan
            
            rdm_matrix_run1_uppertriangle = rdm_matrix_run1_uppertriangle[~np.isnan(model_rdm)]
            rdm_matrix_run4_uppertriangle = rdm_matrix_run4_uppertriangle[~np.isnan(model_rdm)]
            model_rdm = model_rdm[~np.isnan(model_rdm)]

            # True correlation calculation (keep the original model_rdm intact)
            spearman_corr_run1, _ = spearmanr(rdm_matrix_run1_uppertriangle, model_rdm)
            spearman_corr_run4, _ = spearmanr(rdm_matrix_run4_uppertriangle, model_rdm)
            
            if session == '2':
                correlation_dict[sub][mask_name][model_rsm_name]['true_corr'] = {'run1': spearman_corr_run1, 'run4': spearman_corr_run4}
            elif session == '4':
                correlation_dict_session4[sub][mask_name][model_rsm_name]['true_corr'] = {'run1': spearman_corr_run1, 'run4': spearman_corr_run4}

        # append the df to the list of dfs
        lme_matrices_per_mask[mask_name].append(df)
        
# stack the lme matrices and save after processing all sessions
lme_matrices_per_mask[mask_name] = pd.concat(lme_matrices_per_mask[mask_name])
path_for_lme_matrices = create_script_named_folder('lme_matrices')
lme_matrices_per_mask[mask_name].to_csv(opj(path_for_lme_matrices, f'lme_matrices_{mask_name}_{corr_measure}.csv'))


print(correlation_dict)
# save correlation_dict as pkl after processing all sessions
path_for_correlation_dict = create_script_named_folder('correlation_dict')
with open(opj(path_for_correlation_dict, f'{mask_name}_correlation_dict_correleation_measure_{corr_measure}.pkl'), 'wb') as f:
    pickle.dump(correlation_dict, f)
with open(opj(path_for_correlation_dict, f'{mask_name}_correlation_dict_session4_correleation_measure_{corr_measure}.pkl'), 'wb') as f:
    pickle.dump(correlation_dict_session4, f)
    


# load the lme_matrices for hpc
mask_name= 'mtl'
path_for_lme_matrices = create_script_named_folder('lme_matrices')
lme_hpc = pd.read_csv(opj(path_for_lme_matrices, f'lme_matrices_{mask_name}_{corr_measure}.csv'))

###############################################################
# check squares for which paths simialrities changes teh most
###############################################################

def reconstruct_half_rdms(
    lme_df: pd.DataFrame,
    value_col: str = 'empirical_RDM',
    triangle: str = 'upper',          # 'upper' or 'lower'
    diag_value: float | None = 0.0,   # set to None to leave diagonal NaN
    i_col: str | None = None,
    j_col: str | None = None,
):
    """
    Rebuild per-subject×session×phase matrices from lme_hpc values without mirroring.
    Leaves the opposite triangle as NaN and restores original positions.

    If i_col/j_col (0- or 1-based) are present, use them; otherwise use the mask order
    of np.triu_indices(..., 1) or np.tril_indices(..., -1).
    Returns: dict[subject][session][phase] -> (n x n) float array.
    """
    rdms = {}

    # Try to auto-detect index columns if not provided
    if i_col is None or j_col is None:
        cand_i = ['i', 'row', 'idx_i', 'cond_i']
        cand_j = ['j', 'col', 'idx_j', 'cond_j']
        i_col = next((c for c in cand_i if c in lme_df.columns), None)
        j_col = next((c for c in cand_j if c in lme_df.columns), None)

    for (sub, ses, ph), grp in lme_df.groupby(['subject', 'session', 'phase_early1_late2'], sort=False):
        grp = grp.reset_index(drop=True)

        # Place by mask order consistent with vectorization
        k = len(grp)
        # infer n from k = n(n-1)/2
        n = int((1 + np.sqrt(1 + 8 * k)) // 2)

        mat = np.full((n, n), np.nan, dtype=float)
        if triangle == 'upper':
            idx = np.triu_indices(n, k=1)
        elif triangle == 'lower':
            idx = np.tril_indices(n, k=-1)
        else:
            raise ValueError("triangle must be 'upper' or 'lower'")
        mat[idx] = grp[value_col].to_numpy()

        if diag_value is not None:
            np.fill_diagonal(mat, diag_value)

        rdms.setdefault(sub, {}).setdefault(str(ses), {})[int(float(ph))] = mat
        

        # Optional sanity check: re-vectorize and compare order
        if i_col is None:
            vec = mat[idx]
            assert np.allclose(vec, grp[value_col].to_numpy(), equal_nan=True)
            print('===')
            print(vec[0:5])
            
            print(grp[value_col].to_numpy()[0:5])
            print('===')

    return rdms


emp_rdms_half = reconstruct_half_rdms(lme_hpc, value_col='empirical_RDM', triangle='upper', diag_value=None)
labels = []
for context in ['A', 'A_star', 'B', 'C']:
    for pos in range(1, 5):
        labels.append(f'{context}{pos}')

def compute_category_block_changes(
    rdms_half: dict,
    sessions=('2','4'),
    phase_early=1,
    phase_late=2,
    categories=('A','A_star','B','C'),
    elems_per_cat=4,
):
    """
    For each subject and session, compute early vs late mean dissimilarity change
    per unordered category pair. Assumes upper-triangle storage and order
    [A, A_star, B, C] each with 4 consecutive rows/cols.
    Returns DataFrame with columns:
    subject, session, day, pair, early_mean, late_mean, diff (late - early)
    """
    import numpy as np
    import pandas as pd

    # Precompute index ranges for each category
    cat_idx = {}
    start = 0
    for cat in categories:
        cat_idx[cat] = np.arange(start, start + elems_per_cat)
        start += elems_per_cat

    pairs = []
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            pairs.append((categories[i], categories[j]))  # A-A_star, A-B, ...

    rows = []
    for sub in rdms_half:
        for ses in sessions:
            if ses not in rdms_half[sub]:
                continue
            # Early & Late matrices
            if phase_early not in rdms_half[sub][ses] or phase_late not in rdms_half[sub][ses]:
                continue
            mat_early = rdms_half[sub][ses][phase_early]
            mat_late  = rdms_half[sub][ses][phase_late]

            for (c1, c2) in pairs:
                i = cat_idx[c1]
                j = cat_idx[c2]

                # Block (c1,c2) lies in upper triangle (since i < j given ordering)
                block_early = mat_early[np.ix_(i, j)]
                block_late  = mat_late[np.ix_(i, j)]

                early_mean = np.nanmean(block_early)
                late_mean  = np.nanmean(block_late)
                diff = late_mean - early_mean

                rows.append({
                    'subject': sub,
                    'session': ses,
                    'day': 'Day 1' if ses == '2' else 'Day 2',
                    'pair': f'{c1}-{c2}',
                    'early_mean': early_mean,
                    'late_mean': late_mean,
                    'diff_late_minus_early': diff
                })

    return pd.DataFrame(rows)

block_change_df = compute_category_block_changes(emp_rdms_half)

def plot_block_change_with_stats(block_change_df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    import matplotlib.cm as cm

    # Aggregate for bars
    summary_df = (block_change_df
                  .groupby(['day','pair'], as_index=False)
                  .agg(diff_mean=('diff_late_minus_early','mean')))

    # Order
    pair_order = list(summary_df['pair'].unique())
    day_order = ['Day 1','Day 2']
    summary_df['pair'] = pd.Categorical(summary_df['pair'], categories=pair_order, ordered=True)
    summary_df['day'] = pd.Categorical(summary_df['day'], categories=day_order, ordered=True)
    summary_df = summary_df.sort_values(['pair','day']).reset_index(drop=True)

    # One-sample t-tests vs 0 per bar
    tests = []
    for pair in pair_order:
        for day in day_order:
            vals = block_change_df[(block_change_df['pair']==pair) & (block_change_df['day']==day)]['diff_late_minus_early'].dropna()
            if len(vals) >= 2:
                t, p = ttest_1samp(vals, 0.0, nan_policy='omit')
            else:
                t, p = np.nan, np.nan
            tests.append({'pair': pair, 'day': day, 'n': len(vals), 't': t, 'p': p})
    tests_df = pd.DataFrame(tests)

    # FDR correction across all bars
    valid = tests_df['p'].notna()
    pvals = tests_df.loc[valid, 'p'].values
    if pvals.size:
        rej, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        tests_df.loc[valid, 'p_fdr'] = p_fdr
        tests_df.loc[valid, 'reject_fdr_0.05'] = rej
    else:
        tests_df['p_fdr'] = np.nan
        tests_df['reject_fdr_0.05'] = False

    # Merge stats into summary for convenience
    summary_stats = summary_df.merge(tests_df, on=['pair','day'], how='left')

    # Plot bars with subject scatters and significance stars
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # Colors to match earlier style
    cmap_plot = cm.get_cmap('PuBu', 30)
    pu_bu_extended = [cmap_plot(i) for i in range(cmap_plot.N)]
    color_day1 = pu_bu_extended[12]
    color_day2 = pu_bu_extended[28]

    x = np.arange(len(pair_order))
    width = 0.36

    # Heights per day
    y_day1 = summary_stats[summary_stats['day']=='Day 1']['diff_mean'].to_numpy()
    y_day2 = summary_stats[summary_stats['day']=='Day 2']['diff_mean'].to_numpy()

    bars1 = ax.bar(x - width/2, y_day1, width, label='Day 1', color=color_day1, alpha=0.9)
    bars2 = ax.bar(x + width/2, y_day2, width, label='Day 2', color=color_day2, alpha=0.9)

    # Scatter per subject with jitter
    rng = np.random.default_rng(42)
    jitter_scale = 0.06
    for i, pair in enumerate(pair_order):
        for j, day in enumerate(day_order):
            x_center = (x[i] - width/2) if day == 'Day 1' else (x[i] + width/2)
            vals = block_change_df[(block_change_df['pair']==pair) & (block_change_df['day']==day)]['diff_late_minus_early'].dropna()
            if len(vals) == 0:
                continue
            x_jitter = x_center + rng.normal(0, jitter_scale, size=len(vals))
            ax.scatter(x_jitter, vals, s=12, color='grey', alpha=0.6, zorder=3)

    # Significance stars (FDR-corrected)
    def p_to_stars(p):
        if p is None or np.isnan(p):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    for i, pair in enumerate(pair_order):
        for day, bar in [('Day 1', bars1[i]), ('Day 2', bars2[i])]:
            p_adj = summary_stats[(summary_stats['pair']==pair) & (summary_stats['day']==day)]['p_fdr'].iloc[0]
            stars = p_to_stars(p_adj)
            if stars:
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, y + 0.01, stars, ha='center', va='bottom', fontsize=10)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_order, rotation=0)
    ax.set_ylabel('Late − Early (mean dissimilarity)')
    ax.set_title('Category-pair changes with subject scatter (FDR-corrected)\nnegative values indicate increased similarity')
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Print stats table concise
    out = summary_stats[['day','pair','n','diff_mean','t','p','p_fdr','reject_fdr_0.05']]
    print('\nPer-bar one-sample t-tests (late − early vs 0), FDR across bars:')
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return summary_stats

# Run and plot
stats_summary = plot_block_change_with_stats(block_change_df)


##########################################################################
# Run lme predicting empirical_rdm with model_rsm and phase_early1_late2
###########################################################################

# Ensure 'session' and 'phase_early1_late2' are treated as categorical variables
# 'session' might be like 2, 4 or '2', '4'. 'phase_early1_late2' might be 1, 2.
lme_hpc['session'] = pd.Categorical(lme_hpc['session'])
lme_hpc['phase_early1_late2'] = pd.Categorical(lme_hpc['phase_early1_late2'])
lme_hpc['subject'] = pd.Categorical(lme_hpc['subject'])

for model_rsm in model_rsm_dfs.keys():
    # Join all terms for the fixed effects part of the formula
    fixed_effects_formula = f"session * phase_early1_late2 * {model_rsm}"
    full_formula = f"empirical_RDM ~ {fixed_effects_formula}"

    print(f"\nGenerated LME formula:\n{full_formula}\n")
    
    lme_data = lme_hpc.copy()
    # Remove NaN values from the data and keep only included cols
    included_cols = ['empirical_RDM', 'session', 'phase_early1_late2', model_rsm, 'subject']
    lme_data = lme_data[included_cols].dropna()
    

    # Fit the LME model
    # The 'groups' argument specifies the random effects structure. Here, random intercepts for 'subject'.
    print("Fitting LME model... (this may take some time)")

    md = smf.mixedlm(full_formula, lme_data, groups=lme_data["subject"])
    mdf = md.fit(reml=False) # Use REML=False for likelihood ratio tests if comparing models

    # Print the summary of the fitted model
    print("\nLME Model Summary:")
    print(mdf.summary())
    
# Follow-up analyses to decompose the 3-way interaction
# Test simple effects within each session-phase combination

# Create subset datasets
day1_early = lme_data[(lme_data['session'] == 2) & (lme_data['phase_early1_late2'] == 1.0)]
day1_late = lme_data[(lme_data['session'] == 2) & (lme_data['phase_early1_late2'] == 2.0)]
day2_early = lme_data[(lme_data['session'] == 4) & (lme_data['phase_early1_late2'] == 1.0)]
day2_late = lme_data[(lme_data['session'] == 4) & (lme_data['phase_early1_late2'] == 2.0)]

# Test model RDM effect in each condition
subsets = [('day1_early', day1_early), ('day1_late', day1_late), 
           ('day2_early', day2_early), ('day2_late', day2_late)]

for name, subset in subsets:
    if len(subset) > 0:
        formula = f"empirical_RDM ~ {model_rsm}"
        md_subset = smf.mixedlm(formula, subset, groups=subset["subject"])
        mdf_subset = md_subset.fit(reml=False)
        print(f"\n{name}:")
        print(mdf_subset.summary())
        
# Generate LaTeX tables for each model

def generate_statsmodels_style_latex(model_name, mdf, lme_data):
    """Generate LaTeX table in statsmodels summary style"""
    
    # Get model information
    n_obs = int(mdf.nobs)
    n_groups = len(lme_data['subject'].unique())
    group_sizes = lme_data.groupby('subject').size()
    min_group_size = int(group_sizes.min())
    max_group_size = int(group_sizes.max())
    mean_group_size = group_sizes.mean()
    
    # Get model fit statistics
    log_likelihood = mdf.llf
    scale = mdf.scale
    converged = "Yes" if mdf.converged else "No"
    
    latex_lines = []
    
    # Start table
    latex_lines.append("\\begin{table}")
    latex_lines.append(f"\\caption{{Mixed Linear Model Regression Results - {model_name.replace('_', ' ').title()}}}")
    latex_lines.append("\\label{}")
    latex_lines.append("\\begin{center}")
    
    # First table - Model info
    latex_lines.append("\\begin{tabular}{llll}")
    latex_lines.append("\\hline")
    latex_lines.append("Model:            & MixedLM & Dependent Variable: & empirical\\_RDM  \\\\")
    latex_lines.append(f"No. Observations: & {n_obs}   & Method:             & ML                   \\\\")
    latex_lines.append(f"No. Groups:       & {n_groups}      & Scale:              & {scale:.4f}               \\\\")
    latex_lines.append(f"Min. group size:  & {min_group_size}    & Log-Likelihood:     & {log_likelihood:.4f}           \\\\")
    latex_lines.append(f"Max. group size:  & {max_group_size}    & Converged:          & {converged}                  \\\\")
    latex_lines.append(f"Mean group size:  & {mean_group_size:.1f}  &                     &                      \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    
    # Second table - Coefficients
    latex_lines.append("\\begin{center}")
    latex_lines.append("\\begin{tabular}{lrrrrrr}")
    latex_lines.append("\\hline")
    latex_lines.append("                                        &  Coef. & Std.Err. &      z & P$> |$z$|$ & [0.025 & 0.975]  \\\\")
    latex_lines.append("\\hline")
    
    # Get coefficient results
    params = mdf.params
    se = mdf.bse
    tvalues = mdf.tvalues
    pvalues = mdf.pvalues
    conf_int = mdf.conf_int()
    
    # Format each parameter row
    for param in params.index:
        coef = params[param]
        std_err = se[param]
        z_val = tvalues[param]
        p_val = pvalues[param]
        ci_lower = conf_int.loc[param, 0]
        ci_upper = conf_int.loc[param, 1]
        
        # Clean parameter name for LaTeX (replace underscores, handle special characters)
        clean_param = param.replace('_', '\\_')
        clean_param = clean_param.replace('[', '[')
        clean_param = clean_param.replace(']', ']')
        
        # Format the row with proper spacing to match the example
        param_str = f"{clean_param:<40}"  # Left align parameter name
        row = f"{param_str} & {coef:6.3f} & {std_err:8.3f} & {z_val:6.3f} & {p_val:11.3f} & {ci_lower:6.3f} & {ci_upper:7.3f}  \\\\"
        latex_lines.append(row)
    
    # Add Group Var row (random effects variance)
    try:
        group_var = mdf.cov_re.iloc[0, 0] if hasattr(mdf, 'cov_re') and mdf.cov_re is not None else 0.000
        latex_lines.append(f"Group Var{'':<32} & {group_var:6.3f} & {0.000:8.3f} &        &             &        &         \\\\")
    except:
        latex_lines.append(f"Group Var{'':<32} & {0.000:6.3f} & {0.000:8.3f} &        &             &        &         \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{center}")
    latex_lines.append("\\end{table}")
    latex_lines.append("\\bigskip")
    
    return "\n".join(latex_lines)



# Dictionary to store all LaTeX tables
all_latex_tables = []

# Modified model fitting loop to generate LaTeX tables
for model_rsm in model_rsm_dfs.keys():
    # Your existing model fitting code here...
    fixed_effects_formula = f"session * phase_early1_late2 * {model_rsm}"
    full_formula = f"empirical_RDM ~ {fixed_effects_formula}"

    print(f"\nGenerated LME formula:\n{full_formula}\n")
    
    lme_data = lme_hpc.copy()
    included_cols = ['empirical_RDM', 'session', 'phase_early1_late2', model_rsm, 'subject']
    lme_data = lme_data[included_cols].dropna()
    
    print("Fitting LME model... (this may take some time)")
    md = smf.mixedlm(full_formula, lme_data, groups=lme_data["subject"])
    mdf = md.fit(reml=False)
    
    print("\nLME Model Summary:")
    print(mdf.summary())
    
    # Generate LaTeX table for this model
    latex_table = generate_statsmodels_style_latex(model_rsm, mdf, lme_data)
    all_latex_tables.append(latex_table)

# Print all LaTeX tables
print("\n" + "="*80)
print("LATEX TABLES FOR OVERLEAF (STATSMODELS STYLE):")
print("="*80)

for i, latex_table in enumerate(all_latex_tables):
    print(f"\n% Model {i+1}")
    print(latex_table)

print("="*80)

####################################################################
# create correlation barplot
####################################################################


path_for_correlation_dict = create_script_named_folder('correlation_dict')
model_rsm_name = 'ortho_shared_reward_rsm_dissimilar1'
mask_name = 'hpc'
# load correlation_dict
pkl_to_load = opj(path_for_correlation_dict, f'{mask_name}_correlation_dict_correleation_measure_{corr_measure}.pkl')
with open(pkl_to_load, "rb") as file:
    correlation_dict = pickle.load(file)
pkl_to_load = opj(path_for_correlation_dict, f'{mask_name}_correlation_dict_session4_correleation_measure_{corr_measure}.pkl')
with open(pkl_to_load, "rb") as file:
    correlation_dict_session4 = pickle.load(file)

cmap_plot = cm.get_cmap('PuBu', 30)
pu_bu_extended = [cmap_plot(i) for i in range(cmap_plot.N)]
bar_colors = [pu_bu_extended[12], pu_bu_extended[28], pu_bu_extended[12], pu_bu_extended[28]]

data = {'Day 1': {}, 'Day 2': {}}

# Load data for both days
for day in ['Day 1', 'Day 2']:
    
    if day == 'Day 1':
        correlation_dict = correlation_dict
    elif day == 'Day 2':
        correlation_dict = correlation_dict_session4


    filtered_dict = {subject: masks for subject, masks in correlation_dict.items() if mask_name in masks and masks[mask_name]}
    subs = list(filtered_dict.keys())

    run1_values = [correlation_dict[sub][mask_name][model_rsm_name]['true_corr']['run1'] for sub in subs]
    run4_values = [correlation_dict[sub][mask_name][model_rsm_name]['true_corr']['run4'] for sub in subs]

    data[day]['Early'] = run1_values
    data[day]['Late'] = run4_values

# Compute means and errors
means = [np.mean(data['Day 1']['Early']), np.mean(data['Day 1']['Late']),
            np.mean(data['Day 2']['Early']), np.mean(data['Day 2']['Late'])]
errors = [np.std(data['Day 1']['Early']) / np.sqrt(len(data['Day 1']['Early'])),
            np.std(data['Day 1']['Late']) / np.sqrt(len(data['Day 1']['Late'])),
            np.std(data['Day 2']['Early']) / np.sqrt(len(data['Day 2']['Early'])),
            np.std(data['Day 2']['Late']) / np.sqrt(len(data['Day 2']['Late']))]

# Plotting
fig, ax = plt.subplots()
# Adjust the positions of the bars to increase the space between the second and third bar
bar_positions = [0, 1, 2.5, 3.5]
bars = ax.bar(bar_positions, means, yerr=errors, color=bar_colors, capsize=10, width=0.6)

# Adjust x-ticks to match the new bar positions
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Day 1 Early', 'Day 1 Late', 'Day 2 Early', 'Day 2 Late\n(new structure)'], fontsize=14)

# Scatter plot for individual values
for i, (day, phase) in enumerate([('Day 1', 'Early'), ('Day 1', 'Late'), ('Day 2', 'Early'), ('Day 2', 'Late')]):
    x = np.random.normal(bar_positions[i], 0.05, size=len(data[day][phase]))  # Add jitter
    ax.scatter(x, data[day][phase], color='grey', alpha=0.5, s=15)

# Labels and title
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Day 1 Early', 'Day 1 Late', 'Day 2 Early', 'Day 2 Late\n(new structure)'], fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylabel('Correlation', fontsize=20)

if mask_name == 'hpc':
    mask_title = 'Hippocampus'
elif mask_name == 'mtl':
    mask_title = 'Medial Temporal Lobe'
else:
    mask_title = mask_name

ax.set_title(f'{mask_title} - Latent structure model', fontsize=20, pad=20)

# Gridlines and layout
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
sns.despine()

# Adjust figure size and padding
fig.set_size_inches(8, 6)
plt.tight_layout(pad=2.0)

plt.ylim(-0.2, 0.2)

# Save and show the plot
path = create_script_named_folder()
plt.savefig(opj(path, f'{mask_name}_combined_correlation.png'), bbox_inches='tight', dpi=300)
plt.show()

####################################################################
#%% create coefficient barplot
####################################################################

def plot_lme_coefficients(lme_hpc, model_rsm_name, mask_name, corr_measure, save_path=None):
    """
    Plot regression coefficients from LME model across the 4 timepoints
    
    Parameters:
    -----------
    lme_hpc : DataFrame
        The LME data containing empirical_RDM, session, phase_early1_late2, model RSMs, and subject
    model_rsm_name : str
        Name of the model RSM to analyze (e.g., 'ortho_shared_reward_rsm_dissimilar1')
    mask_name : str
        Brain region name (e.g., 'hpc', 'mtl')
    corr_measure : str
        Correlation measure used
    save_path : str, optional
        Path to save the figure
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from os.path import join as opj
    
    # Prepare data
    lme_data = lme_hpc.copy()
    lme_data['session'] = pd.Categorical(lme_data['session'])
    lme_data['phase_early1_late2'] = pd.Categorical(lme_data['phase_early1_late2'])
    lme_data['subject'] = pd.Categorical(lme_data['subject'])
    
    # Remove NaN values
    included_cols = ['empirical_RDM', 'session', 'phase_early1_late2', model_rsm_name, 'subject']
    lme_data = lme_data[included_cols].dropna()
    
    # Fit separate models for each timepoint to get coefficients and confidence intervals
    timepoints = [
        ('Day 1 Early', (lme_data['session'] == 2) & (lme_data['phase_early1_late2'] == 1.0)),
        ('Day 1 Late', (lme_data['session'] == 2) & (lme_data['phase_early1_late2'] == 2.0)),
        ('Day 2 Early', (lme_data['session'] == 4) & (lme_data['phase_early1_late2'] == 1.0)),
        ('Day 2 Late', (lme_data['session'] == 4) & (lme_data['phase_early1_late2'] == 2.0))
    ]
    
    coefficients = []
    ci_lower = []
    ci_upper = []
    std_errors = []
    p_values = []
    
    for name, condition in timepoints:
        subset = lme_data[condition]
        
        if len(subset) > 0:
            # Fit model for this timepoint
            formula = f"empirical_RDM ~ {model_rsm_name}"
            md_subset = smf.mixedlm(formula, subset, groups=subset["subject"])
            mdf_subset = md_subset.fit(reml=False)
            
            # Get coefficient for the model RSM
            coef = mdf_subset.params[model_rsm_name]
            coefficients.append(coef)
            
            # Get confidence intervals
            conf_int = mdf_subset.conf_int()
            ci_lower.append(conf_int.loc[model_rsm_name, 0])
            ci_upper.append(conf_int.loc[model_rsm_name, 1])
            
            # Get standard error and p-value
            std_errors.append(mdf_subset.bse[model_rsm_name])
            p_values.append(mdf_subset.pvalues[model_rsm_name])
        else:
            coefficients.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            std_errors.append(np.nan)
            p_values.append(np.nan)
    
    # Calculate error bars (CI width)
    errors = [(ci_upper[i] - ci_lower[i]) / 2 for i in range(len(coefficients))]
    
    # Set up colors (matching your correlation plot)
    cmap_plot = cm.get_cmap('PuBu', 30)
    pu_bu_extended = [cmap_plot(i) for i in range(cmap_plot.N)]
    bar_colors = [pu_bu_extended[12], pu_bu_extended[28], pu_bu_extended[12], pu_bu_extended[28]]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(3.57, 3))
    
    # Adjust bar positions to match your correlation plot
    bar_positions = [0, 1, 2.5, 3.5]
    
    # Plot bars with error bars
    bars = ax.bar(bar_positions, coefficients, yerr=errors, 
                   color=bar_colors, capsize=4, width=0.8, alpha=0.8)
    
    # Add significance stars
    for i, (pos, p_val) in enumerate(zip(bar_positions, p_values)):
        if p_val < 0.001:
            significance = '***'
        elif p_val < 0.01:
            significance = '**'
        elif p_val < 0.05:
            significance = '*'
        else:
            significance = ''
        
        if significance:
            y_pos = coefficients[i] + errors[i] + 0.01
            ax.text(pos, y_pos, significance, ha='center', va='bottom')
            
    # Add "new structure" text below the last bar
    last_bar_pos = bar_positions[-1]  # Position of the last bar
    last_bar_height = coefficients[-1] - errors[-1]  # Height including error bar, subtracted to move below
    ax.text(last_bar_pos, last_bar_height-0.01, 'new\nstructure', 
            ha='center', va='top', color='darkgrey', fontsize=10)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Labels and title
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Day 1\nEarly', 'Day 1\nLate', 'Day 2\nEarly', 'Day 2\nLate'])
    ax.set_ylabel('Regression coefficient')
    
    # Set title based on mask name
    if mask_name == 'hpc':
        mask_title = 'Hippocampus'
    elif mask_name == 'mtl':
        mask_title = 'Medial Temporal Lobe'
    else:
        mask_title = mask_name
    
    ax.set_title(f'{mask_title}\nregression coefficients over time')
    
    # Removed background horizontal grid lines by not calling ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.tick_params(axis='y')
    sns.despine()
    
    # Adjust y-axis limits based on data
    y_max = max(coefficients) + max(errors) + 0.05
    y_min = min(coefficients) - max(errors) - 0.05
    ax.set_ylim(y_min, y_max)
    
    #plt.tight_layout(pad=2.0)
    
    # Save if path provided
    if save_path:
        plt.savefig(opj(save_path, f'{mask_name}_lme_coefficients.svg'), 
                    bbox_inches='tight', dpi=300)
    
    plt.show()
    
    # Print coefficient summary
    print("\nLME Coefficient Summary:")
    print("-" * 60)
    for i, name in enumerate(['Day 1 Early', 'Day 1 Late', 'Day 2 Early', 'Day 2 Late']):
        print(f"{name}: β = {coefficients[i]:.4f}, "
              f"95% CI = [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}], "
              f"p = {p_values[i]:.4f}")
    
    return {
        'coefficients': coefficients,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_values': p_values,
        'timepoints': ['Day 1 Early', 'Day 1 Late', 'Day 2 Early', 'Day 2 Late']
    }

# # Alternative function that extracts coefficients from the full interaction model
# def plot_interaction_coefficients(mdf, model_rsm_name, mask_name, save_path=None):
#     """
#     Plot coefficients from the full interaction model to show the effect 
#     of the model RSM at each timepoint combination
    
#     Parameters:
#     -----------
#     mdf : fitted mixed effects model
#         The fitted model with session * phase * model_rsm interaction
#     model_rsm_name : str
#         Name of the model RSM
#     mask_name : str
#         Brain region name
#     save_path : str, optional
#         Path to save the figure
#     """
    
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm
#     import seaborn as sns
#     import numpy as np
#     from os.path import join as opj
    
    
    
#     # Extract relevant coefficients from the full model
#     params = mdf.params
#     conf_int = mdf.conf_int()
    
#     # Calculate the effect at each timepoint combination
#     # Base effect (Day 1 Early - reference level)
#     day1_early_coef = params[model_rsm_name]
    
#     # Day 1 Late: base + phase effect + phase:model interaction
#     day1_late_coef = (params[model_rsm_name] + 
#                       params.get(f'phase_early1_late2[T.2.0]:{model_rsm_name}', 0))
    
#     # Day 2 Early: base + session effect + session:model interaction
#     day2_early_coef = (params[model_rsm_name] + 
#                        params.get(f'session[T.4]:{model_rsm_name}', 0))
    
#     # Day 2 Late: all effects combined
#     day2_late_coef = (params[model_rsm_name] + 
#                       params.get(f'phase_early1_late2[T.2.0]:{model_rsm_name}', 0) +
#                       params.get(f'session[T.4]:{model_rsm_name}', 0) +
#                       params.get(f'session[T.4]:phase_early1_late2[T.2.0]:{model_rsm_name}', 0))
    
#     coefficients = [day1_early_coef, day1_late_coef, day2_early_coef, day2_late_coef]
    
#     # Set up colors
#     cmap_plot = cm.get_cmap('PuBu', 30)
#     pu_bu_extended = [cmap_plot(i) for i in range(cmap_plot.N)]
#     bar_colors = [pu_bu_extended[12], pu_bu_extended[28], pu_bu_extended[12], pu_bu_extended[28]]
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(3.57, 3))
    
#     # Adjust bar positions
#     bar_positions = [0, 1, 2.5, 3.5]
    
#     # Plot bars
#     bars = ax.bar(bar_positions, coefficients, color=bar_colors, width=0.6, alpha=0.8)
    
#     # Add horizontal line at y=0
#     ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
#     # Labels and title
#     ax.set_xticks(bar_positions)
#     ax.set_xticklabels(['Day 1 Early', 'Day 1 Late', 'Day 2 Early', 'Day 2 Late\n(new structure)'], 
#                        fontsize=14)
#     ax.set_ylabel('Model Effect (from interaction model)', fontsize=20)
    
#     # Set title based on mask name
#     if mask_name == 'hpc':
#         mask_title = 'Hippocampus'
#     elif mask_name == 'mtl':
#         mask_title = 'Medial Temporal Lobe'
#     else:
#         mask_title = mask_name
    
#     ax.set_title(f'{mask_title} - Decomposed Effects from Interaction Model', 
#                  fontsize=20, pad=20)
    
#     # Grid and styling
#     ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#     ax.set_axisbelow(True)
#     ax.tick_params(axis='y', labelsize=15)
#     sns.despine()
    
#     plt.tight_layout(pad=2.0)
    
#     # Save if path provided
#     if save_path:
#         plt.savefig(opj(save_path, f'{mask_name}_interaction_coefficients.png'), 
#                     bbox_inches='tight', dpi=300)
    
#     plt.show()
    
#     return coefficients

mask_name = 'mtl'  # Change to 'hpc' or other mask as needed
# Example usage:
for model_rsm_name in model_rsm_dfs.keys():
    print(f"Plotting LME coefficients for model: {model_rsm_name}")
    plot_lme_coefficients(lme_hpc, model_rsm_name, mask_name, corr_measure, save_path=path)
    
#plot_interaction_coefficients(mdf, model_rsm_name, mask_name, save_path=path)
    # %%
