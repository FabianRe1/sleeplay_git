# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:45:11 2020

@author: Moneta, adapted by F. Renz

Here we extract ROIs
ROIs: mOFC, lOFC, OFC, HP_all, HP_gm, EC, HP_all_EC, HP_gm_EC
Thresholds: 0.1,0.3,0.5 (relative to global signal)
smoothed: 4mm (for now, can do 8mm later)
Output: 
    1. ROI map per subject
    2. Signal loss per subject
    3. save data from ROI

background: defaults.mask.thresh = 0.8 will only estimate the model in voxels whose mean value is at least 80% of the global signal


Steps: 
    1. load the freesurfer in native space (can also do MNI later)
    2. take ROIs
    3. load the functional smoothed images and calculate the global mean signal for all data in block
    5. Get the mask of the brain that is at threshold
    6. See how much signal lose we have in the ROIs for each subject
    - it could be that we need to run signal clean for optimal? 
    
    labels i think : https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    OFC
    1012 2012    ctx-rh-lateralorbitofrontal
    1014 2014    ctx-rh-medialorbitofrontal
    HP
    17  Left-Hippocampus                        220 216 20  0
    53  Right-Hippocampus                       220 216 20  0
    1016    ctx-lh-parahippocampal              20  220 60  0
    2016    ctx-rh-parahippocampal              20  220 60  0
    EC
    1006    ctx-lh-entorhinal                   220 20  10  0
    2006    ctx-rh-entorhinal                   220 20  10  0
    MOTOR control
    1024    ctx-lh-precentral                   60  20  220 0
    2024    ctx-rh-precentral                   60  20  220 0
    
    1022    ctx-lh-postcentral                  220 20  20  0
    2022    ctx-lh-postcentral                  220 20  20  0
    
   primary auditory 
    1030    ctx-lh-superiortemporal             140 220 220 0
    2030    ctx-lh-superiortemporal             140 220 220 0


    ALSO - make brain mask as the intercept of all runs in session brain masks
"""

import os
from os.path import join as opj
import sys
import numpy as np
from nilearn.image import load_img,new_img_like
from pathlib import Path
import glob
from tqdm import tqdm
import re
from groo.groo import get_root

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

'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
runcomp = 2  # 1 = server, 2 = Fabi's comp
do_rs = 0

# Root for outputs (root file is also on the HPC, so this should work)
root_folder = get_root(".my_hidden_root_file")

# MRI data (fMRIPrep derivatives) are on the shared HPC folder
path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

path_output_files_base = create_script_named_folder('thresholded_roi_output')
path_output_info_base = create_script_named_folder('thresholded_roi_info')

# ---------------------------------------------------------------------
# Subject handling:
#   - If called with an argument, use it as subject
#   - Accept either "13" or "sub-13"
#   - On failure or no argument: fall back to "sub-13" and print warning
# ---------------------------------------------------------------------
default_subj = "sub-13"

if len(sys.argv) > 1:
    subj_arg = sys.argv[1]
    # If user passes just a number, prepend "sub-"
    if re.match(r'^\d+$', subj_arg):
        subj = f"sub-{subj_arg}"
    elif re.match(r'^sub-\d+$', subj_arg):
        subj = subj_arg
    else:
        print(
            f"WARNING: Unexpected subject format '{subj_arg}'. "
            f"Falling back to default subject {default_subj}."
        )
        subj = default_subj
else:
    print(
        f"WARNING: No subject specified on command line. "
        f"Using default subject {default_subj}."
    )
    subj = default_subj

path_fmriprep_subj = opj(path_fmriprep, subj)


sessions = ['ses-02', 'ses-03', 'ses-04', 'ses-06']
spaces = ['T1w', 'MNI152NLin2009cAsym']
thresholds = [.0, .3, .5, 0.7, .8, .9]
roinames = [
    'VIS', 'MTL', 'HPC', 'HPPHC', 'PHPC', 'ERH', 'OFC', 'LPFC',
    'RACING', 'CACING', 'mOFC', 'ventricles', 'AUD', 'SOMATO',
    'CORTICAL_MEMORY', 'PARIETAL_PRECUNEUS', 'PARIETAL', 'PRECUNEUS',
    'MEDIAL_OFC', 'ROSTRAL_ACC', 'CAUDAL_ACC', 'SUPERIOR_FRONTAL',
    'RSC', 'POSTERIOR_CINGULATE',
]

# Loop over session and make sure it is a dir
print(f"[INFO] Subject: {subj}")
print(f"[INFO] fMRIPrep path: {path_fmriprep_subj}")
print(f"[INFO] Sessions to process: {sessions}")
for session in sessions:
    sess_path = opj(path_fmriprep, subj, session)
    if not os.path.isdir(sess_path):
        print(f"[SKIP] {session} — directory does not exist: {sess_path}")
        continue

    print('Currently processing session', session)

    path_output_files = opj(path_output_files_base, subj, session)
    path_output_info = opj(path_output_info_base, subj, session)

    Path(path_output_files).mkdir(parents=True, exist_ok=True)
    Path(path_output_info).mkdir(parents=True, exist_ok=True)
    outF=path_output_files
    outFsl=path_output_info

    subdata = list()
    print('starting '+ subj)
    print('session: '+session)

    # save nifti
    savenifti = True

    for space in spaces:

            # brain mask
            data_BM = opj(
                path_fmriprep, subj, session, 'func',
                f'{subj}_{session}_task-*_run-*_space-{space}_desc-brain_mask.nii.gz'
            )
            # If resting-state:
            if do_rs == 1:
                data_BM = opj(
                    path_fmriprep, subj, session, 'func',
                    f'{subj}_{session}_task-rest*_run-*_space-{space}_desc-brain_mask.nii.gz'
                )
            data_BM = sorted(glob.glob(data_BM), key=lambda f: os.path.basename(f))
            RUNS=len(data_BM)


            BM = load_img(data_BM) # concatenate all runs
            BMd = np.array(BM.dataobj).astype(int) # 4D array
            BMs = np.sum(BMd,axis=3) # 3d array, indicating how often the voxels were considered brain

            # Loop over the 9 runs
            for run in np.arange(1,RUNS+1):
                print('starting run ' +str(run))
                if run<10:
                    crun='0'
                else:
                    crun=''
                # --- Check if ALL output files for this (space, run) already exist ---
                expected_files = []
                for _T in thresholds:
                    _Tstr = str(int(_T * 100))
                    if do_rs == 1:
                        expected_files.append(opj(outF, f'wholebrain_{subj}_{session}_task-rest_space-{space}_T-{_Tstr}_run-{crun}{run}nii'))
                    else:
                        expected_files.append(opj(outF, f'wholebrain_{subj}_{session}_space-{space}_T-{_Tstr}_run-{crun}{run}.nii'))
                    for _roiname in roinames:
                        if do_rs == 1:
                            expected_files.append(opj(outF, f'{_roiname}_{subj}_{session}_task-rest_space-{space}_T-{_Tstr}_run-{crun}{run}.nii'))
                        else:
                            expected_files.append(opj(outF, f'{_roiname}_{subj}_{session}_space-{space}_T-{_Tstr}_run-{crun}{run}.nii'))
                # if all(os.path.isfile(f) for f in expected_files):
                #     print(f'[SKIP] {session} space-{space} run-{crun}{run} — all {len(expected_files)} files already exist')
                #     continue

                # ROI masks
                # contains the segmented brain --> each voxel receives number indicating the identified brain area
                data_FS = opj(path_fmriprep,subj,session,'func', f'{subj}_{session}_task-*_run-{crun}{run}_space-{space}_desc-aparcaseg_dseg.nii.gz')
                print(data_FS)
                data_FS = sorted(glob.glob(data_FS), key=lambda f: os.path.basename(f))


                print('data_FS:')
                print(data_FS)
                img1 = load_img(data_FS)
                FS = np.array(img1.dataobj).astype(int)
                # 1000 is left 2--- is right
                OFC = (FS==1012) | (FS==2012) | (FS==1014) | (FS==2014)
                lOFC = (FS==1012) | (FS==2012)
                mOFC = (FS==1014) | (FS==2014)
                HPC =  (FS==17) | (FS==53) # 17 is left hippo 53 is right
                HPPHC =  (FS==17) | (FS==53) | (FS==1016) | (FS==2016) # HPC + paraHPC
                PHPC = (FS==1016) | (FS==2016) # paraHPC
                ERH = (FS==1006) | (FS==2006)
                MTL =  (FS==17) | (FS==53) | (FS==1016) | (FS==2016) | (FS==1006) | (FS==2006)
                MOT1 = (FS==1022) | (FS==2022)
                MOT2 = (FS==1024) | (FS==1024)
                AUD = (FS==1030) | (FS==2030)
                VIS = (FS==1005) | (FS==2005) | (FS==1011) | (FS==2011) | (FS==1021) | (FS==2021) | (FS==1029) | (FS==2029) | (FS==1013) | (FS==2013) | (FS==1008) | (FS==2008) | (FS==1007) | (FS==2007) | (FS==1009) | (FS==2009) | (FS==1016) | (FS==2016) | (FS==1015) | (FS==2015)
                mask_visual_labels = [
                1005, 2005,  # cuneus
                1011, 2011,  # lateral occipital
                1021, 2021,  # pericalcarine
                1029, 2029,  # superioparietal
                1013, 2013,  # lingual
                1008, 2008,  # inferioparietal
                1007, 2007,  # fusiform
                1009, 2009,  # inferiotemporal
                1016, 2016,  # parahippocampal
                1015, 2015,  # middle temporal
                ]
                #PFC = (FS==9000) | (FS==9500) # prefrontal cortex  mask - not actually correct in freesurfer, need to piece together PFC from other frontal  masks
                RMFG = (FS==1027) | (FS==2027) # rostral middle frontal gyrus

                mask_LPFC_labels=[
                1003,2003, # caudal middle frontal
                1027,2027, # rostral middle frontal
                1018,2018, # parsopercularis
                1020,2020, # parstriangularis
                1019,2019, # parsorbitalis
                ]

                LPFC = (FS==1003) | (FS==2003) | (FS==1027) | (FS==2027) | (FS==1018) | (FS==2018) | (FS==1020) | (FS==2020) | (FS==1019) | (FS==2019)

                left_OFC = (FS==1012) | (FS==1014)
                left_HPC =  (FS==17) # 17 is left hippo 53 is right
                left_HPPHC =  (FS==17) | (FS==1016) # left HPC + paraHPC
                left_PHPC = (FS==1016)
                left_MTL =(FS==17) | (FS==1016) | (FS==1006)
                left_ERH = (FS==1006)
                left_MOT1 = (FS==1022)
                left_MOT2 = (FS==1024)
                left_AUD = (FS==1030)

                right_OFC = (FS==2012) | (FS==2014)
                right_HPC =  (FS==53)
                right_HPPHC =  (FS==53) | (FS==2016) # right HPC and paraHPC
                right_PHPC = (FS==2016)
                right_ERH = (FS==2006)
                right_MTL = (FS==53) | (FS==2016) | (FS==2006)
                right_MOT1 = (FS==2022)
                right_MOT2 = (FS==2024)
                right_AUD = (FS==2030)

                RACING = (FS==1026) | (FS==2026) # rostral anterior cingulate
                CACING = (FS==1002) | (FS==2002) # caudal anterior cingulate
                mOFC = (FS==1014) | (FS==2014) # medial OFC

                ventricles = (FS==4) | (FS==43) | (FS==213)

                Somato = (FS==1022) | (FS==2022)

                cortical_memory = \
                    (FS==1014) | (FS==2014) | \
                    (FS==1026) | (FS==2026) | \
                    (FS==1002) | (FS==2002) | \
                    (FS==1028) | (FS==2028) | \
                    (FS==1010) | (FS==2010) | \
                    (FS==1023) | (FS==2023) | \
                    (FS==1025) | (FS==2025) | \
                    (FS==1008) | (FS==2008)
                # Breakdown of regions included above:
                # mPFC: Medial OFC (1014/2014), Rostral ACC (1026/2026), Caudal ACC (1002/2002), Superior Frontal (1028/2028)
                # Precuneus: Precuneus (1025/2025)
                # Parietal:  Inf Parietal (1008/2008)

                parietal_precuneus = (FS==1008) | (FS==2008) | (FS==1025) | (FS==2025)
                parietal = (FS==1008) | (FS==2008)
                precuneus = (FS==1025) | (FS==2025)
                medial_OFC = (FS==1014) | (FS==2014)
                rostral_ACC = (FS==1026) | (FS==2026)
                caudal_ACC = (FS==1002) | (FS==2002)
                superior_frontal = (FS==1028) | (FS==2028)
                RSC = (FS==1010) | (FS==2010)
                posterior_cingulate = (FS==1023) | (FS==2023)


                [print(np.sum(roi)) for roi in [VIS, MTL, HPC, HPPHC, PHPC, ERH, OFC, LPFC]]
                # load data and get global mean
                data_task_path = opj(path_fmriprep,subj,session,'func', f'{subj}_{session}_task-*_run-{crun}{run}_space-{space}_desc-preproc_bold.nii.gz')

                data_task = sorted(glob.glob(data_task_path), key=lambda f: os.path.basename(f))
                print('Path', data_task_path)
                print('Glob', data_task)


                img2 = load_img(data_task)
                TASK = np.array(img2.dataobj)
                GlobMean = np.mean(TASK)
                # make voxel mean
                Voxmean = np.mean(TASK,axis=3)

                print('Voxmean', Voxmean)
                # loop through thresholds
                for T in thresholds:
                    # The higher the T, the more restrictive, only takes voxels higher than T percent of global mean
                    VOX = Voxmean > (T * GlobMean)
                    # save brain mask
                    masked_image = new_img_like(img1, VOX)
                    if savenifti:
                        outname = opj(outF, 'wholebrain_' + subj + '_' + session + '_space-' + space + '_T-' + str(int(T * 100)) + '_run-' + str(crun) + str(run) + '.nii')
                        if do_rs == 1:
                            outname = opj(outF, 'wholebrain_' + subj + '_' + session + '_task-rest_space-' + space + '_T-' + str(int(T * 100)) + '_run-' + str(crun) + str(run) + 'nii')
                        masked_image.to_filename(outname)
                    # save number of voxels in whole brain
                    subdata.append([subj, session, run, space, 'WholeBrain', str(int(T * 100)), np.sum(VOX), 1, T])

                    rois = [VIS, MTL, HPC, HPPHC, PHPC, ERH, OFC, LPFC, RACING, CACING, mOFC, ventricles, AUD, Somato, cortical_memory, parietal_precuneus, parietal, precuneus, medial_OFC, rostral_ACC, caudal_ACC, superior_frontal, RSC, posterior_cingulate]
                    for roi, roiname in zip(rois, roinames):

                        # save ROI mask
                        masked_image = new_img_like(img1,VOX & np.squeeze(roi))
                        if savenifti:
                            #if roiname=='PFC' or roiname=='RMFG':
                            if do_rs==1:
                                masked_image.to_filename(opj(outF,roiname+'_'+subj+'_'+session+'_task-rest_space-'+space+'_T-'+str(int(T*100))+'_run-'+str(crun)+str(run)+'.nii'))
                            else:
                                print('Saving nifti: ', opj(outF,roiname+'_'+subj+'_'+session+'_space-'+space+'_T-'+str(int(T*100))+'_run-'+str(crun)+str(run)+'.nii'))
                                masked_image.to_filename(opj(outF,roiname+'_'+subj+'_'+session+'_space-'+space+'_T-'+str(int(T*100))+'_run-'+str(crun)+str(run)+'.nii'))
                        # save signal lose in ROI
                        subdata.append([subj,session,run,space,roiname,str(int(T*100)),np.sum(VOX & np.squeeze(roi)),np.sum(VOX & np.squeeze(roi))/np.sum(roi),T]) # 100%of anat as this is anat
    if do_rs==1:
        csv_path = opj(outFsl,"ROIsDropout"+subj+'_'+session+"_resting-state.csv")
    else:
        csv_path = opj(outFsl,"ROIsDropout"+subj+'_'+session+".csv")
    if os.path.isfile(csv_path) and not subdata:
        print(f'[SKIP] {session} — CSV already exists: {csv_path}')
    elif subdata:
        np.savetxt(csv_path, np.vstack(subdata), delimiter=',', fmt='%s')

# %%
