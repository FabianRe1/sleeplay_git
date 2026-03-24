#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:38:57 2023


@author: fabi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from groo.groo import get_root
from os.path import join as opj
import glob
from groo.groo import get_root


root_folder = get_root(".my_hidden_root_file")
# all_fmriprep_subjs = sorted(os.listdir(opj(root_folder, 'mri', 'derivatives')))
# all_fmriprep_subjs = [subj for subj in all_fmriprep_subjs if subj.startswith('sub-')]

all_behavior_subjs = sorted(os.listdir(opj(root_folder, 'behavior')))
all_behavior_subjs = [subj for subj in all_behavior_subjs if subj.startswith('s')]


# Session to analyse
sessions = [1,2]
HRF_delay = 5

for subj in all_behavior_subjs:
        
    # get subj number everything after sub-
    subj_number = subj[1:]    
        # =========================================
    print('=========================================')
    print(f"start checking subject {subj}")
    print('=========================================')
    
    # =========================================
    # 1. Check localizer files
    # =========================================
    
    # CHeck localizers are there
    dir_behavior_loc1 = opj(root_folder, 'behavior', subj, 'behav', 'loc', f's{subj_number}_df_loc1.csv')
    dir_behavior_loc2 = opj(root_folder, 'behavior', subj, 'behav', 'loc', f's{subj_number}_df_loc2.csv')
    
    # load both and check that the highest number in block_nr is 50
    if os.path.exists(dir_behavior_loc1):
        df_loc1 = pd.read_csv(dir_behavior_loc1, sep=None, engine='python')
        max_block_loc1 = df_loc1['block_nr'].max()
        if max_block_loc1 < 50:
            print(f"Localizer 1 for subject {subj} incomplete, max block_nr: {max_block_loc1}")
            
    if os.path.exists(dir_behavior_loc2):
        df_loc2 = pd.read_csv(dir_behavior_loc2, sep=None, engine='python')
        max_block_loc2 = df_loc2['block_nr'].max()
        if max_block_loc2 < 50:
            print(f"Localizer 2 for subject {subj} incomplete, max block_nr: {max_block_loc2}")
    
    # Check if files exist
    if not os.path.exists(dir_behavior_loc1):
        print(f"Missing localizer 1 for subject {subj}")
    if not os.path.exists(dir_behavior_loc2):
        print(f"Missing localizer 2 for subject {subj}")
        
    # =========================================
    # 2. Check main files
    # =========================================    
    
    dir_behavior_main1 = opj(root_folder, 'behavior', subj, 'behav', 'main', f'sub{subj_number}_main_task_nses_0_step0')
    # dir_behavior_main2 = opj(root_folder, 'behavior', subj, 'behav', 'main', f'sub{subj_number}_main_task_nses_1_step0')
    dir_behavior_main3 = opj(root_folder, 'behavior', subj, 'behav', 'main', f'sub{subj_number}_main_task_nses_2_step2')
    
    # check if these directories exist
    if not os.path.isdir(dir_behavior_main1):
        print(f"Missing main task session 1 for subject {subj}")
    # if not os.path.isdir(dir_behavior_main2):
    #     print(f"Missing main task session 2 for subject {subj}")
    if not os.path.isdir(dir_behavior_main3):
        print(f"Missing main task session 3 for subject {subj}")
        
    # load ses 0
    if os.path.isdir(dir_behavior_main1):
        csv_file_main1 = opj(dir_behavior_main1, f'output_df_sub{subj_number}_w_HRF.csv')
        if os.path.exists(csv_file_main1):
            df_main1 = pd.read_csv(csv_file_main1, sep=None, engine='python')
            max_trial_main1 = df_main1['trl_nr'].max()
            if max_trial_main1 < 130:
                print(f"Main task session 1 for subject {subj} incomplete, max block_nr: {max_trial_main1}")
        else:
            print(f"Missing CSV file for main task session 1 for subject {subj}")
        
    if os.path.isdir(dir_behavior_main3):
        csv_file_main3 = opj(dir_behavior_main3, f'output_df_sub{subj_number}_w_HRF.csv')
        if os.path.exists(csv_file_main3):
            df_main3 = pd.read_csv(csv_file_main3, sep=None, engine='python')
            max_trial_main3 = df_main3['trl_nr'].max()
            if max_trial_main3 < 130:
                print(f"Main task session 2 for subject {subj} incomplete, max block_nr: {max_trial_main3}")
        else:
            print(f"Missing CSV file for main task session 2 for subject {subj}")
        
    # check if any other main task directories are there (should not be)
    main_task_dirs = glob.glob(opj(root_folder, 'behavior', subj, 'behav', 'main', f'sub{subj_number}_main_task_nses_*'))
    for directory in main_task_dirs:
        if os.path.isdir(directory) and not (directory.endswith('nses_0_step0') or directory.endswith('nses_2_step2') or directory.endswith('nses_1_step0')or directory.endswith('nses_1_step1')):
            print(f"Unexpected main task directory for subject {subj}: {directory}")
