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
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from groo.groo import get_root
from nilearn.maskers import NiftiMasker
from joblib import dump
from tqdm import tqdm

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


# # create encoding stimulus column and encoding category column
# def _exclusive_merge(df, col_a, col_b):
#     if col_a not in df.columns:
#         raise KeyError(f"Missing required column: {col_a}")
#     if col_b not in df.columns:
#         raise KeyError(f"Missing required column: {col_b}")

#     s1 = df[col_a]
#     s2 = df[col_b]

#     # base non-null masks
#     v1_base = ~pd.isna(s1)
#     v2_base = ~pd.isna(s2)

#     # consider empty/whitespace and common NA tokens as null (but keep original values when chosen)
#     def invalid_text_mask(s, base_mask):
#         txt = s.astype(str).str.strip().str.lower()
#         return base_mask & (
#             (txt == '') |
#             (txt == 'na') | (txt == 'nan') | (txt == 'none') | (txt == 'null')
#         )

#     inv1 = invalid_text_mask(s1, v1_base)
#     inv2 = invalid_text_mask(s2, v2_base)

#     valid1 = v1_base & ~inv1
#     valid2 = v2_base & ~inv2

#     overlap = valid1 & valid2
#     if overlap.any():
#         idx = df.index[overlap].tolist()
#         raise ValueError(f"Overlap detected between {col_a} and {col_b} at rows {idx[:20]} (showing up to 20); total overlaps: {overlap.sum()}")

#     merged = pd.Series(np.nan, index=df.index, dtype=object)
#     pick_a = valid1 & ~valid2
#     pick_b = valid2 & ~valid1
#     merged.loc[pick_a] = s1.loc[pick_a]
#     merged.loc[pick_b] = s2.loc[pick_b]
#     return merged

# # encoding_stimulus: obj_enc_stim + sc_enc_stim
# df_behavior['encoding_stimulus'] = _exclusive_merge(df_behavior, 'obj_enc_stim', 'sc_enc_stim')
# df_behavior['encoding_category'] = _exclusive_merge(df_behavior, 'obj_cat', 'sc_enc_stim')

# # Check encoding_stimulus and encoding_category if only encoding_category has an entry set it nan as it comes from obj retrieval
# _only_category = df_behavior['encoding_stimulus'].isna() & df_behavior['encoding_category'].notna()
# if _only_category.any():
#     df_behavior.loc[_only_category, 'encoding_category'] = np.nan
#     print(f"Cleared encoding_category for {int(_only_category.sum())} rows where encoding_stimulus is NaN.")
    



'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''

root_folder = get_root(".my_hidden_root_file")
path_fmriprep = opj(root_folder, 'mri', '1_derivatives')
path_masks = opj(root_folder, 'mri', '3_reliability_voxel_masks_localizers_intersected')


TR=1.5
smoothing_levels=[4] #[4,6]
temporal_smoothing_levels = [3]
f_thresholds=[50]
reliability_thresholds = [50]


# Get subject info from sys.argv if not it chooses subj 5 for development
try:
    sub = 'sub-%s' % sys.argv[1]
    if len(sub) > 10: raise ValueError('Subject name too long --> locally run sys.args[1]')

except:
    print('ACHTUNG - DEVELOPER SUBJ CHOSEN')
    sub = 'sub-13'

# Get the number of the subject
parts = sub.split('-')
if len(parts[1]) == 1:
    mri_sub = f"{parts[0]}-0{parts[1]}"
else:
    mri_sub = sub

# Define the sessions 
mri_sessions_loc1 = 'ses-02'
mri_sessions_loc2 = 'ses-03' 
behavior_sessions_loc1 = 'loc1' 
behavior_sessions_loc2 = 'loc2' 

only_scenes1_objects2_both3 = 3 # 1=scenes only, 2=objects only, 3=both



'''
========================================================================
LOAD BEHAVIORAL DATA FOR THE LOCALISER 1
========================================================================
'''

# Load df of localizer - this is the behavioral data for the pre task localizer session
sub_num = sub[4:]
path_behavior = opj(root_folder, 'behavior', sub, f's{sub_num}_df_{behavior_sessions_loc1}_w_HRF.csv')
file_list = glob.glob(path_behavior)
df_behavior_loc1 = pd.read_csv(file_list[0], sep=';')

df_behavior_loc1 = df_behavior_loc1.dropna(subset=['encoding_category'])
df_behavior_loc1 = df_behavior_loc1.reset_index(drop=True)

# Blocks in localizer
runs_localizer_1 = df_behavior_loc1.run_nr.unique()
runs_localizer_1 = runs_localizer_1[~pd.isna(runs_localizer_1)].astype(int)

# Get the number of runs in the loclaizer
nruns_localizer_1 = len(runs_localizer_1)

# The TRs of the presented stimuli across blocks
grouped_localizer_1 = df_behavior_loc1.groupby('run_nr')
# create a list of series, where each series contains the 'tr_idx' values for the corresponding block
trs_vis_scan1_test_localizer_loc_1 = [group['TR_time_onset_w_HRF'] for _, group in sorted(grouped_localizer_1, key=lambda x: x[0])]

# The stimuli label presented across blocks
labels_scan1_video_localizer_loc_1 = df_behavior_loc1.encoding_category

# Block information across blocks
runvector_scan1_localizer = df_behavior_loc1.run_nr.copy()

'''
========================================================================
LOAD BEHAVIORAL DATA FOR THE LOCALISER 2
========================================================================
'''

path_behavior_loc2 = opj(root_folder, 'behavior', sub, f's{sub_num}_df_{behavior_sessions_loc2}_w_HRF.csv')
file_list = glob.glob(path_behavior_loc2)
df_behavior_loc2 = pd.read_csv(file_list[0], sep=';')

# drop rows where encoding_category is nan
df_behavior_loc2 = df_behavior_loc2.dropna(subset=['encoding_category'])
df_behavior_loc2 = df_behavior_loc2.reset_index(drop=True)

# Blocks in localizer
runs_localizer_2 = df_behavior_loc2.run_nr.unique() + 13
runs_localizer_2 = runs_localizer_2[~pd.isna(runs_localizer_2)].astype(int)

# Get the number of runs in the loclaizer
nruns_localizer_2 = len(np.unique(df_behavior_loc2.run_nr))

# The TRs of the presented stimuli across blocks
grouped_localizer_2 = df_behavior_loc2.groupby('run_nr')
# create a list of series, where each series contains the 'tr_ idx' values for the corresponding blockf
trs_vis_scan1_test_localizer_loc_2 = [group['TR_time_onset_w_HRF'] for _, group in sorted(grouped_localizer_2, key=lambda x: x[0])]

# The stimuli label presented across blocks
labels_scan1_video_localizer_loc_2 = df_behavior_loc2.encoding_category

# Block information across blocks
runvector_scan2_localizer = df_behavior_loc2.run_nr.copy() + 13

'''
========================================================================
Combine the two localizer runs for later training
========================================================================
'''

def adjust_series_indices(series_list):
    for i in range(1, len(series_list)):
        last_index = series_list[i - 1].index[-1] + 1
        current_length = len(series_list[i])
        series_list[i].index = pd.RangeIndex(start=last_index, stop=last_index + current_length)
    return series_list

trs_vis_scan1_test_localizer = trs_vis_scan1_test_localizer_loc_1 + trs_vis_scan1_test_localizer_loc_2
TR_video_onset = adjust_series_indices(trs_vis_scan1_test_localizer) # list of TRs 1 per run with the relevant TRs corresponding to the labels

# stack the df_behavior_loc1 and df_behavior_loc2
df_behavior = pd.concat([df_behavior_loc1, df_behavior_loc2])
df_behavior = df_behavior.reset_index(drop=True)

# combine the two localizer runs labels
labels_scan1_video_localizer = pd.concat([labels_scan1_video_localizer_loc_1, labels_scan1_video_localizer_loc_2], ignore_index=True).reset_index(drop=True)

# combine the two localizer runs block information
runvector_localizer = pd.concat((runvector_scan1_localizer, runvector_scan2_localizer)).reset_index(drop=True)

nruns_localizer = nruns_localizer_1 + nruns_localizer_2 # number of runs in both localizer

runs_localizer = np.concatenate((runs_localizer_1, runs_localizer_2)) # array with run IDs

'''
========================================================================
LOAD localizer 1 func data, anatomical data and confounds
========================================================================
'''

path_func_task_scan1 = opj(path_fmriprep, sub, mri_sessions_loc1, 'func', f'{sub}_{mri_sessions_loc1}_task-loc*_space-T1w_desc-preproc_bold.nii.gz')
path_func_task_scan1 = sorted(glob.glob(path_func_task_scan1), key=lambda f: os.path.basename(f))
data_task_scan_loc1 = [image.load_img(i) for i in path_func_task_scan1]


# load the confounds files
path_confs_task_scan1 = opj(path_fmriprep, sub, mri_sessions_loc1, 'func', f'{sub}_{mri_sessions_loc1}_task-loc*_run*_desc-confounds_timeseries.tsv') # sub-13_ses-02_task-loc1_run-02_desc-confounds_timeseries.tsv
path_confs_task_loc1 = sorted(glob.glob(path_confs_task_scan1),key=lambda f: os.path.basename(f))
confs_scan_loc1 = [pd.read_csv(path_confs_task_loc1[cr], sep='\t') for cr in range(nruns_localizer_1)]

# set confound variables of interest
conf_vars_localizer = ['global_signal', 'framewise_displacement',
                'trans_x', 'trans_y', 'trans_z',
                'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                'rot_x', 'rot_y', 'rot_z',
                'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# load the anatomical mri file - we do not need it here, but might want it for plotting. 
path_anat_scan1 = opj(path_fmriprep, sub, 'anat', f'{mri_sub}_desc-preproc_T1w.nii.gz')
path_anat_scan_loc1 = sorted(glob.glob(path_anat_scan1), key=lambda f: os.path.basename(f))


'''
========================================================================
LOAD localizer 2 func data, anatomical data and confounds
========================================================================
'''

path_func_task_scan2 = opj(path_fmriprep, sub, mri_sessions_loc2, 'func', f'{sub}_{mri_sessions_loc2}_task-loc*_space-T1w_desc-preproc_bold.nii.gz')
path_func_task_scan2 = sorted(glob.glob(path_func_task_scan2), key=lambda f: os.path.basename(f))
data_task_scan_loc2 = [image.load_img(i) for i in path_func_task_scan2]

# load the confounds files
path_confs_task_scan2 = opj(path_fmriprep, sub, mri_sessions_loc2, 'func', f'{sub}_{mri_sessions_loc2}_task-loc*_run*_desc-confounds_timeseries.tsv') # sub-13_ses-02_task-loc1_run-02_desc-confounds_timeseries.tsv
path_confs_task_loc2 = sorted(glob.glob(path_confs_task_scan2),key=lambda f: os.path.basename(f))
confs_scan_loc2 = [pd.read_csv(path_confs_task_loc2[cr], sep='\t') for cr in range(nruns_localizer_2)]


'''
========================================================================
DEFINE THE CLASSIFIERS
========================================================================
'''

# set mask labels you want to test
mask_labels =['vis'] # ['lpfc', 'vis', 'hpc', 'mtl', 'wholebrain', 'erh', 'ofc']

# create empty dataframes to store data. Use it to store the results of the classifier parameters
df_out_scan1 = pd.DataFrame(columns=['subj', 'mask', 'smoothing', 'fstat_percentile', 'reliability_threshold', 'temporal_smoothing', 'mean_performance'])

# begin looping over masks
for cmask in mask_labels:
    
    print('Running ' + cmask + ' mask')
    
    for reliability_threshold in reliability_thresholds:

        # Load vis mask
        path_vis_mask = f'{sub}_mask_vis_threshold_{reliability_threshold}_combined.nii'
        mask_vis_scan1 = image.load_img(opj(path_masks, path_vis_mask))

        # Load hpc mask
        path_hpc_mask = f'{sub}_mask_hpc_threshold_{reliability_threshold}_combined.nii'
        mask_hpc_scan1 = image.load_img(opj(path_masks, path_hpc_mask))

        # Load EC mask
        path_erh_mask = f'{sub}_mask_erh_threshold_{reliability_threshold}_combined.nii'
        mask_erh_scan1 = image.load_img(opj(path_masks, path_erh_mask))

        # Load PFC mask
        path_lpfc_mask = f'{sub}_mask_lpfc_threshold_{reliability_threshold}_combined.nii'
        mask_lpfc_scan1 = image.load_img(opj(path_masks, path_lpfc_mask))

        # Load OFC mask
        path_ofc_mask = f'{sub}_mask_ofc_threshold_{reliability_threshold}_combined.nii'
        mask_ofc_scan1 = image.load_img(opj(path_masks, path_ofc_mask))

        # Load MTL mask
        path_mtl_mask = f'{sub}_mask_mtl_threshold_{reliability_threshold}_combined.nii'
        mask_mtl_scan1 = image.load_img(opj(path_masks, path_mtl_mask))

        # Load whole brain mask
        path_whole_brain_mask = f'{sub}_mask_wholebrain_threshold_{reliability_threshold}_combined.nii'
        mask_wholebrain_scan1 = image.load_img(opj(path_masks, path_whole_brain_mask))

        masks_scan1 = {'lpfc': mask_lpfc_scan1, 'vis': mask_vis_scan1, 'hpc': mask_hpc_scan1, 'mtl': mask_mtl_scan1,'wholebrain': mask_wholebrain_scan1, 'erh': mask_erh_scan1, 'ofc': mask_ofc_scan1}



        # loop over smoothing conditions
        for csmooth in smoothing_levels:
            
            print('Smoothing ' + str(csmooth) + 'fwhm')  
            
            
            # =====================================================================
            # Masking, smooth and detrend data
            # =====================================================================
            # Localizer 1
            print('Running wholebrain smoothing and applying anatomical ' + cmask + ' mask on loc 1 data')
            data_masked_scan1_localizer=[]
            data_masked_scan1_localizer = [masking.apply_mask(i, masks_scan1[cmask], smoothing_fwhm=csmooth) for i in data_task_scan_loc1]
            data_masked_detrend_scan1_localizer = [signal.clean(signals=data_masked_scan1_localizer[cr],
                                                t_r=TR, detrend=True,
                                                standardize='zscore_sample',
                                                confounds=confs_scan_loc1[cr][conf_vars_localizer].fillna(0))
                                                for cr in range(nruns_localizer_1)]
            
            # Localizer 2
            print('Running wholebrain smoothing and applying anatomical ' + cmask + ' mask on loc 2 data')
            data_masked_scan2_localizer=[]
            data_masked_scan2_localizer = [masking.apply_mask(i, masks_scan1[cmask], smoothing_fwhm=csmooth) for i in data_task_scan_loc2]
            data_masked_detrend_scan2_localizer = [signal.clean(signals=data_masked_scan2_localizer[cr],
                                                t_r=TR, detrend=True,
                                                standardize='zscore_sample',
                                                confounds=confs_scan_loc2[cr][conf_vars_localizer].fillna(0))
                                                for cr in range(nruns_localizer_2)]
            
            
            # combine the two localizer runs
            data_masked_detrend_scan1_localizer = data_masked_detrend_scan1_localizer + data_masked_detrend_scan2_localizer
            
            
            # ========================================================================
            # Check if TRs from behavior and scanning TRs match if not drop the TRs that do not
            # if the scanner was stopped too early there will be TRs in the behavior that are not in the MRI (ideally this should not be necessary)
            # ========================================================================

            # drop those from localizer
            new_trs_scan1_train_localizer = []
            for j in range(len(TR_video_onset)):
                print(f'loclaizer: TRs from behavior {TR_video_onset[j].iloc[-1]}; TRs recorded in MRI {data_masked_detrend_scan1_localizer[j].shape[0]}')
                if TR_video_onset[j].iloc[-1] >= data_masked_detrend_scan1_localizer[j].shape[0]:
                    print(f'Too few TRs in {sub} at run {j} in the loclaizer')
                # filter the trs this returns a boolean vactor with false for the entries that are not in the MRI but in the log data
                new_series = TR_video_onset[j][TR_video_onset[j] < data_masked_detrend_scan1_localizer[j].shape[0]]
                new_trs_scan1_train_localizer.append(new_series)
            trs_scan1_train_localizer = new_trs_scan1_train_localizer
            
            # Get the remaining indices that are present in the logs and the MRI data and filter other variables
            tmp_trs_scan1_train_localizer_combined = [n.index.values for n in trs_scan1_train_localizer]
            tmp_trs_scan1_train_localizer_combined = np.concatenate( tmp_trs_scan1_train_localizer_combined, axis=0)

            # filter the variables if all TRs have been recorded len(tmp_trs_scan1_train_localizer_combined) should be length of all other variables
            runvector_localizer = runvector_localizer[tmp_trs_scan1_train_localizer_combined]
            labels_scan1_video_localizer = labels_scan1_video_localizer[tmp_trs_scan1_train_localizer_combined]
            df_scan1_localizer = df_behavior.loc[df_behavior.index.isin(tmp_trs_scan1_train_localizer_combined)]
            
            
            # =====================================================================
            # Based on the selected TRs from the log file pick the timepoints from the MRI
            # =====================================================================

            # get relevant trs from detrended data
            data_masked_detrend_sel_scan1_train_localizer = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer[cr].astype(int), ]
                                                        for cr in range(nruns_localizer)]
            
            data_masked_detrend_sel_scan1_localizer = np.vstack(data_masked_detrend_sel_scan1_train_localizer) # number of TRs x number of voxels


            # =====================================================================
            # Temporal smoothing
            # In case tmp smoothing is > 0 averaging with neighboring TRs to reduce noise
            # not very helpful in visual cortex but might be beneficial in other areas (own experience)
            # =====================================================================

            for temporal_smoothing in temporal_smoothing_levels:
                if temporal_smoothing == 0:
                    data_masked_detrend_sel_scan1_localizer_t = data_masked_detrend_sel_scan1_localizer
                    print('No temporal smoothing')
                elif temporal_smoothing == 1:
                    # Add 1 to each element of each series in trs_scan1_train_localizer
                    trs_scan1_train_localizer_1_delayed = [series.apply(lambda x: x - 1) for series in trs_scan1_train_localizer]

                    data_masked_detrend_sel_scan1_train_localizer_1_delayed = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_1_delayed[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
            
                    data_masked_detrend_sel_scan1_localizer_1_delayed = np.vstack(data_masked_detrend_sel_scan1_train_localizer_1_delayed)

                    # Average delayed and non-delayed data
                    data_masked_detrend_sel_scan1_localizer = (data_masked_detrend_sel_scan1_localizer + data_masked_detrend_sel_scan1_localizer_1_delayed) / 2

                elif temporal_smoothing == 2:
                    # Add 1 to each element of each series in trs_scan1_train_localizer
                    trs_scan1_train_localizer_1_delayed = [series.apply(lambda x: x + 1) for series in trs_scan1_train_localizer]
                    data_masked_detrend_sel_scan1_train_localizer_1_delayed = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_1_delayed[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
                    data_masked_detrend_sel_scan1_localizer_1_delayed = np.vstack(data_masked_detrend_sel_scan1_train_localizer_1_delayed)

                    # Subtract 1 from each element of each series in trs_scan1_train_localizer
                    trs_scan1_train_localizer_1_prior = [series.apply(lambda x: x - 1) for series in trs_scan1_train_localizer]
                    data_masked_detrend_sel_scan1_train_localizer_1_prior = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_1_prior[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
                    data_masked_detrend_sel_scan1_localizer_1_prior = np.vstack(data_masked_detrend_sel_scan1_train_localizer_1_prior)
                    
                    # Average delayed and non-delayed data
                    data_masked_detrend_sel_scan1_localizer = (data_masked_detrend_sel_scan1_localizer + data_masked_detrend_sel_scan1_localizer_1_delayed + data_masked_detrend_sel_scan1_localizer_1_prior) / 3

                elif temporal_smoothing == 3:
                    # Add 1 to each element of each series in trs_scan1_train_localizer
                    trs_scan1_train_localizer_1_delayed = [series.apply(lambda x: x + 1) for series in trs_scan1_train_localizer]
                    data_masked_detrend_sel_scan1_train_localizer_1_delayed = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_1_delayed[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
                    data_masked_detrend_sel_scan1_localizer_1_delayed = np.vstack(data_masked_detrend_sel_scan1_train_localizer_1_delayed)

                    # Add 2
                    trs_scan1_train_localizer_2_delayed = [series.apply(lambda x: x + 2) for series in trs_scan1_train_localizer]
                    data_masked_detrend_sel_scan1_train_localizer_2_delayed = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_2_delayed[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
                    data_masked_detrend_sel_scan1_localizer_2_delayed = np.vstack(data_masked_detrend_sel_scan1_train_localizer_2_delayed)

                    # Subtract 1 from each element of each series in trs_scan1_train_localizer
                    trs_scan1_train_localizer_1_prior = [series.apply(lambda x: x - 1) for series in trs_scan1_train_localizer]
                    data_masked_detrend_sel_scan1_train_localizer_1_prior = [data_masked_detrend_scan1_localizer[cr][trs_scan1_train_localizer_1_prior[cr].astype(int), ]
                                for cr in range(nruns_localizer)]
                    data_masked_detrend_sel_scan1_localizer_1_prior = np.vstack(data_masked_detrend_sel_scan1_train_localizer_1_prior)
                    
                    # Average delayed and non-delayed data
                    data_masked_detrend_sel_scan1_localizer = (data_masked_detrend_sel_scan1_localizer + 
                                                                data_masked_detrend_sel_scan1_localizer_1_delayed + 
                                                                data_masked_detrend_sel_scan1_localizer_1_prior + 
                                                                data_masked_detrend_sel_scan1_localizer_2_delayed) / 4

                # =============================================================================
                # FILTER CATEGORIES (Moved outside f_thresholds loop)
                # =============================================================================
                
                # Create copies to avoid modifying the original data for the next temporal smoothing iteration
                # (Though currently temporal smoothing overwrites 'data_masked_detrend_sel_scan1_localizer' anyway, 
                #  so strictly speaking we are already committed to that branch, but good practice).
                
                curr_data = data_masked_detrend_sel_scan1_localizer.copy()
                curr_labels = labels_scan1_video_localizer.copy()
                curr_runvector = runvector_localizer.copy()
                curr_df = df_scan1_localizer.copy()

                if only_scenes1_objects2_both3 == 1:
                    # select only scene categories
                    sel_idx = curr_df['encoding_category'].isin(['canyons01.jpg', 'road01.jpg', 'forest03.jpg', 'desert03.jpg',
                                                                    'mountain02.jpg', 'winter01.jpg', 'lake01.jpg', 'beach02.jpg',
                                                                    'waterfall04.jpg', 'autumn03.jpg'])
                elif only_scenes1_objects2_both3 == 2:
                    # select only object categories
                    sel_idx = curr_df['encoding_category'].isin(['vehicle', 'furniture', 'fruit', 'clothing', 'kitchen', 'sports', 'music',
                                                                    'animal', 'tool', 'body'])
                else:
                    sel_idx = slice(None) # Select all

                # Apply filter
                if only_scenes1_objects2_both3 in [1, 2]:
                    curr_data = curr_data[sel_idx.values, ] # Use .values for safety with numpy
                    curr_labels = curr_labels[sel_idx].reset_index(drop=True)
                    curr_runvector = curr_runvector[sel_idx].reset_index(drop=True)
                    curr_df = curr_df.loc[sel_idx].reset_index(drop=True)

                # --- SANITY CHECKS START ---
                print(f"DEBUG: Original data shape: {data_masked_detrend_sel_scan1_localizer.shape}")
                print(f"DEBUG: Filtered data shape: {curr_data.shape}")
                print(f"DEBUG: Unique classes remaining: {curr_labels.unique()}")
                print(f"DEBUG: Number of runs remaining: {len(np.unique(curr_runvector))}")

                # Check for runs with dangerously low sample counts
                for r in np.unique(curr_runvector):
                    n_samples = sum(curr_runvector == r)
                    if n_samples < 5:
                        print(f"WARNING: Run {r} has only {n_samples} samples after filtering!")
                # --- SANITY CHECKS END ---
                
                # UPDATE RUNS LIST: Only iterate over runs that actually exist in the filtered data
                curr_runs_localizer = np.unique(curr_runvector)

                # =============================================================================
                # Start classifier bit           
                # =============================================================================

                for cthreshold in f_thresholds:
                    print('Feature selection at ' + str(cthreshold) + 'th percentile')
                    
                    clas_c=0
                    clas_c+=1
                    # clear all variables
                    if clas_c>=2:
                        del train_runs
                        del train_data
                        del train_labels
                        del train_data_re,
                        del train_labels_re
                        del clf
                        del df_temp_scan1

                    
                    # initialise lists 
                    probs=[]
                    labels=[]  

                    # create a copy of the behavioral data where we can add the classifier info
                    df_temp_scan1 = curr_df.copy()
                    
                    #========================================================================
                    # Choose to have train test split or to train on all data for replay classifier
                    use_everything_for_training = 1
                    print('Use train test split - 1 uses everything for training and saves the classifier: ' + str(use_everything_for_training))
                    #========================================================================

                    # train classifier to use on replay rest data
                    if use_everything_for_training: 
                        
                        # get training data - contains 600 TRs from the blocks for training
                        train_data = curr_data
                        
                        # get training labels
                        train_labels = curr_labels
                        
                        # If there is NA in training labels this would exclude it
                        excl_train = np.where(pd.Series(train_labels).isna())
                        if len(excl_train[0]) > 0:
                                train_data = np.delete(train_data, excl_train, 0)
                                train_labels = np.delete(train_labels, excl_train, 0)
                                
                        # balance classes    
                        train_data_re, train_labels_re = SMOTE().fit_resample(train_data, train_labels) 
                        # train_data_re, train_labels_re = train_data, train_labels # Bypass SMOTE
                            
                        # select features    
                        feature_selection = SelectPercentile(f_classif, percentile=cthreshold)
                        
                        # create multi-class logistic regression classifier
                        clf = make_pipeline(StandardScaler(), 
                                            feature_selection,
                                            LogisticRegression(C=1., solver='lbfgs', penalty='l2',
                                                                multi_class='ovr', max_iter=10000,
                                                                class_weight='balanced',
                                                                random_state=42))
                        
                        # fit the model using the training data
                        clf.fit(train_data_re, train_labels_re)
                        
                        # Save classifier
                        path_clf = create_script_named_folder('classifiers')
                        filename_clf = opj(path_clf, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing}_only_scenes1_objects2_both3_{only_scenes1_objects2_both3}.joblib")
                        dump(clf, filename_clf)

                    else:
                        

                        #========================================================================
                        # Here a leave one run out cross validation is performed and the results plotted and stored
                        #========================================================================
                        
                        # FIX: Iterate over the updated runs list
                        for iteration, irun in enumerate(tqdm(curr_runs_localizer, desc='Leave-one-run-out', unit='run')):
        
                            #========================================================================
                            # ORGANISE TRAINING AND TEST DATA FOR CLASSIFICATION
                            #========================================================================
                            # define the run indices for the test set:
                            test_runs = [irun]
                            # get runs not in the test set
                            # FIX: Use curr_runs_localizer for negation logic
                            train_runs = curr_runs_localizer[~pd.Series(curr_runs_localizer).isin(test_runs)]
                            
                            # get training data
                            # FIX: Use curr_data and curr_runvector
                            train_data = curr_data[curr_runvector.isin(train_runs), ]
                            # get test data
                            test_data = curr_data[curr_runvector.isin(test_runs), ]
                            
                            # get training and test labels
                            train_labels = curr_labels[curr_runvector.isin(train_runs)]
                            test_labels = curr_labels[curr_runvector.isin(test_runs)]
                            
                            
                            # If there is NA in training labels this would exclude it
                            excl_train = np.where(pd.Series(train_labels).isna())
                            if len(excl_train[0]) > 0:
                                train_data = np.delete(train_data, excl_train, 0)
                                train_labels = np.delete(train_labels, excl_train, 0)
                                    
                            train_data_re, train_labels_re = SMOTE().fit_resample(train_data, train_labels)
                            # train_data_re, train_labels_re = train_data, train_labels # Bypass SMOTE

                            #========================================================================
                            #BEGIN CLASSIFICATION
                            #========================================================================
                            
                            # Delete classifier on every new round
                            if irun>runs_localizer[0]:
                                del clf
                                
                            # select features    
                            feature_selection = SelectPercentile(f_classif, percentile=cthreshold)
                            
                            # create multi-class logistic regression classifier
                            clf = make_pipeline(StandardScaler(), 
                                                feature_selection,
                                                LogisticRegression(C=1., solver='lbfgs', penalty='l2',
                                                                    multi_class='ovr', max_iter=10000,
                                                                    class_weight='balanced',
                                                                    random_state=42))
                            
                            # fit the model using the training data
                            clf.fit(train_data_re, train_labels_re)
                            
                            preds_scan1 = clf.predict(test_data)
                            preds_scan1_prob = clf.predict_proba(test_data)
                            
                            if iteration == 0:
                                predictions = preds_scan1
                                predictions_prob = preds_scan1_prob
            
                            else:
                                predictions=np.concatenate((predictions,preds_scan1))
                                predictions_prob=np.concatenate((predictions_prob,preds_scan1_prob))
                        
                        
                        # add data to the array
                        size_df=len(train_labels) + len(test_labels)
                        
                        
                        lb_len=len(np.unique(train_labels)) # number of different classes
                        # df_temp_scan1['training_labels']=list(np.repeat(np.reshape(np.unique(train_labels),[1,lb_len]),size_df,axis=0))
                        # df_temp_scan1['class_labels']=list(np.repeat(np.reshape(clf.classes_,[1,lb_len]),size_df,axis=0))
                        df_temp_scan1['predicted_labels']=list(predictions)
                        df_temp_scan1['predicted_probabilities'] = predictions_prob.tolist()
                        df_temp_scan1['mask']=cmask

                        # Hyperparams
                        df_temp_scan1['smoothing']=np.repeat(csmooth,size_df)
                        df_temp_scan1['fstat_percentile']=np.repeat(cthreshold,size_df)
                        df_temp_scan1['reliability_threshold'] = reliability_threshold
                        df_temp_scan1['temporal_smoothing'] = temporal_smoothing
        
                        # Create a new column that indicates whether the predicted label matches the stimulus
                        df_temp_scan1['label_match'] = (df_temp_scan1['encoding_category'] == df_temp_scan1['predicted_labels'])
                        
                        
                        # =============================================================================
                        # Save df
                        # =============================================================================

                        path_df = create_script_named_folder('df_classifier_results_csv')
                        filename_df = opj(path_df, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing}.csv")
                        df_temp_scan1.to_csv(filename_df)
                        
                        # =============================================================================
                        # Plot performance    
                        # =============================================================================

                        # path_plots = create_script_named_folder('df_classifier_results_plots')
                        
                        # Get the percentage of label matches for all stimuli
                        match_percentages = df_temp_scan1.groupby('encoding_category')['label_match'].mean() * 100
                        mean_performance = match_percentages.mean()

                        # Create a bar plot of the percentage of label matches for all stimuli
                        fig, ax = plt.subplots(figsize=(10, 8))
                        bars = ax.bar(match_percentages.index, match_percentages.values)
                        # Set the colors of the bars based on their height on the y-axis
                        colors = plt.cm.PuBu(match_percentages.values/100)
                        for i, bar in enumerate(bars):
                            bar.set_color(colors[i])
                        # Set the plot title and axis labels
                        ax.set_title(f"{sub} Session:loc1loc3 combined Percentage of label matches by stimulus:\nMask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold} reliability_threshold_{reliability_threshold} _tmpsmoothing_{temporal_smoothing}\nAvg_{mean_performance:.2f}")
                        ax.set_xlabel('Stimulus')
                        ax.set_ylabel('Percentage')
                        
                        # Add text showing scene/object (or both) performance
                        scene_cats = ['canyons01.jpg', 'road01.jpg', 'forest03.jpg', 'desert03.jpg',
                                      'mountain02.jpg', 'winter01.jpg', 'lake01.jpg', 'beach02.jpg',
                                      'waterfall04.jpg', 'autumn03.jpg']
                        object_cats = ['vehicle', 'furniture', 'fruit', 'clothing', 'kitchen', 'sports',
                                       'music', 'animal', 'tool', 'body']

                        # Determine which category groups are present in the results
                        present_scene = [c for c in scene_cats if c in match_percentages.index]
                        present_object = [c for c in object_cats if c in match_percentages.index]

                        text_lines = []
                        if len(present_scene) > 0:
                            scene_perf = match_percentages.loc[present_scene].mean()
                            text_lines.append(f"Scene mean: {scene_perf:.2f}% ({len(present_scene)} items)")
                        if len(present_object) > 0:
                            object_perf = match_percentages.loc[present_object].mean()
                            text_lines.append(f"Object mean: {object_perf:.2f}% ({len(present_object)} items)")

                        if text_lines:
                            text_block = "\n".join(text_lines)
                            # place text in upper-left corner of the axes
                            ax.text(0.01, 0.98, text_block, transform=ax.transAxes,
                                    fontsize=10, va='top', ha='left',
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        
                        # Add a red line at y-axis level of 1/16
                        ax.axhline(y=1/len(df_temp_scan1.encoding_category.unique()) * 100, color='red', linestyle='--', label='Chance level')
                        
                        # Rotate x-axis labels by 90 degrees
                        plt.xticks(rotation=90)
                        
                        # Add percentage values on top of the bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%', ha='center', va='bottom')
                        
                        # Set legend
                        ax.legend()
                        
                        # save the plot
                        path_plots = create_script_named_folder('df_classifier_results_plots')
                        filename_plot = opj(path_plots, f"{sub}_Mask_{cmask}_Smooth_{csmooth}_Threshold_{cthreshold}_reliability_threshold_{reliability_threshold}_tmpsmoothing_{temporal_smoothing}.png")
                        plt.savefig(filename_plot, bbox_inches='tight')
                        
                        # Show the plot
                        plt.show()

                        ############################################################
                        # Calculate mean performance across all stimuli

                        new_row = pd.DataFrame({
                            'subj': [sub],
                            'mask': [cmask],
                            'smoothing': [csmooth],
                            'fstat_percentile': [cthreshold],
                            'reliability_threshold': [reliability_threshold],
                            'temporal_smoothing': [temporal_smoothing],
                            'mean_performance': [mean_performance]
                        })

                        # Append the new row to the df_out_scan1 DataFrame
                        df_out_scan1 = pd.concat([df_out_scan1, new_row], ignore_index=True)

