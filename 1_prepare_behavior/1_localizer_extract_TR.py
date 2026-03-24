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

def list_files(directory):
    for filepath in glob.glob(f'{directory}/*'):
        print(filepath)    

def find_closest_TR(target, start=0, end = 1000, tr=1.5):
    # Generate number stream
    numbers = []
    while start <= end:
        numbers.append(start)
        start += tr
    # find closest TR index
    closest_index = None
    closest_difference = float("inf")
    for i, num in enumerate(numbers):
        difference = abs(num - target)
        if difference < closest_difference:
            closest_index = i
            closest_difference = difference
    return closest_index

root_folder = get_root(".my_hidden_root_file")
# all_fmriprep_subjs = sorted(os.listdir(opj(root_folder, 'mri', 'derivatives')))
# all_fmriprep_subjs = [subj for subj in all_fmriprep_subjs if subj.startswith('sub-')]

all_behavior_subjs = sorted(os.listdir(opj(root_folder, 'behavior')))
all_behavior_subjs = [subj for subj in all_behavior_subjs if subj.startswith('s')]


# Session to analyse
sessions = [1,2]
HRF_delay = 5

for subj in all_behavior_subjs:
    for session in sessions:
        
        # get subj number everything after sub-
        subj_number = subj[1:]
        
        dir_behavior_logs = opj(root_folder, 'behavior', subj, 'behav', 'loc', f's{subj_number}_df_loc{session}.csv')
        
        try:
            curr_subj_data = pd.read_csv(dir_behavior_logs, sep=None, engine='python')
        except:
            print(f'Could not read file for subject {subj}, session {session}. Skipping.')
            continue

        # Initialise HRF shifted columns
        curr_subj_data['time_onset_w_HRF'] = None
        curr_subj_data['TR_time_onset'] = None
        curr_subj_data['TR_time_onset_w_HRF'] = None
        
        for trial in range(curr_subj_data.shape[0]):
            
            curr_subj_data.loc[trial, 'time_onset_w_HRF'] = curr_subj_data.loc[trial, 'time_onset'] + HRF_delay
            curr_subj_data.loc[trial, 'TR_time_onset'] = find_closest_TR(curr_subj_data.loc[trial, 'time_onset'])
            curr_subj_data.loc[trial, 'TR_time_onset_w_HRF'] = find_closest_TR(curr_subj_data.loc[trial, 'time_onset_w_HRF'])
            
        # Save updated DataFrame to CSV
        output_filename = opj(root_folder, 'behavior', subj, 'behav', 'loc', f's{subj_number}_df_loc{session}_w_HRF.csv')
        curr_subj_data.to_csv(output_filename, sep=';', index=False)
        
    

    


        
        





