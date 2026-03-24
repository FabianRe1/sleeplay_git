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


all_behavior_subjs = sorted(os.listdir(opj(root_folder, 'behavior')))
all_behavior_subjs = [subj for subj in all_behavior_subjs if subj.startswith('s')]

HRF_delay = 5

for subj in all_behavior_subjs:
    dir_behavior_logs = opj(root_folder, 'behavior', subj, 'behav')
    
    # Find all csv files recursively
    csv_files = glob.glob(opj(dir_behavior_logs, 'main', '**', '*.csv'), recursive=True)
    
    for csv_file in csv_files:
        # Skip files that are already processed
        if '_w_HRF' in csv_file:
            continue
        
        if 'loc' in csv_file:
            continue  # Skip localizer files
        
        if 'output' not in csv_file:
            continue  # Process only output files
            
        print(f"Processing {csv_file}")
        try:
            curr_subj_data = pd.read_csv(csv_file, sep=',')
        except Exception as e:
            print(f"Could not read {csv_file}: {e}")
            continue
        
        # convert t_onset to float, coerce errors to NaN
        curr_subj_data['t_onset'] = pd.to_numeric(curr_subj_data['t_onset'], errors='coerce')
        
        if 't_onset' in curr_subj_data.columns:
            # Initialise HRF shifted columns
            curr_subj_data['time_onset_w_HRF'] = None
            curr_subj_data['TR_time_onset_w_HRF'] = None
            
            for trial in range(curr_subj_data.shape[0]):
                
                # if t_onset is not a float (i.e., NaN), skip
                if not isinstance(curr_subj_data.loc[trial, 't_onset'], (float, int)):
                    print(f"Skipping trial {trial} due to invalid t_onset")
                    continue
                
                curr_subj_data.loc[trial, 'time_onset_w_HRF'] = curr_subj_data.loc[trial, 't_onset'] + HRF_delay
                curr_subj_data.loc[trial, 'TR_time_onset'] = find_closest_TR(curr_subj_data.loc[trial, 't_onset'])
                curr_subj_data.loc[trial, 'TR_time_onset_w_HRF'] = find_closest_TR(curr_subj_data.loc[trial, 'time_onset_w_HRF'])
            
            # Save updated DataFrame to CSV
            save_path = csv_file.replace('.csv', '_w_HRF.csv')
            curr_subj_data.to_csv(save_path, index=False)
            print(f"Saved to {save_path}")

    

    


        
        





