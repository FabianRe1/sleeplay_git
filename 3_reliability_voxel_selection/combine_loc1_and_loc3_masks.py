# -*- coding: utf-8 -*-

import glob
import os
from os.path import join as opj
import numpy as np
from nilearn import image, masking
from groo.groo import get_root
import re

def combine_masks(mask_loc_1, mask_loc_2):
    """
    Combine two masks using nilearn's intersect_masks function.
    This function will keep only the voxels that are non-zero in both masks.
    Args:
    - mask_loc_1: NIfTI image object for the ROI mask
    - mask_loc_2: NIfTI image object for the correlation mask

    threshold 1 means that the voxels should be non-zero in both masks
    threshold 0 means that the voxels should be non-zero in at least one mask
    """
    combined_mask = masking.intersect_masks([mask_loc_1, mask_loc_2], threshold=0, connected=False)
    return combined_mask

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

def process_and_save_masks(input_folder, paths_locs_intersectd_output, ROIS):
    output_folder = paths_locs_intersectd_output
    
    # Retrieve all NIfTI files for loc1
    # Pattern: sub-XX_mask_YYY_threshold_ZZ_localizer_loc1.nii
    mask_files_1 = glob.glob(os.path.join(input_folder, '*_localizer_loc1.nii'))
    
    print(f"Found {len(mask_files_1)} files for loc1 in {input_folder}")
    
    for roi in ROIS:

        for mask_file_1 in mask_files_1:
            
            if roi not in mask_file_1:
                continue
            
            base_name = os.path.basename(mask_file_1)
            
            # Construct corresponding loc2 filename
            corresponding_name = base_name.replace('localizer_loc1', 'localizer_loc2')
            mask_file_2 = os.path.join(input_folder, corresponding_name)
            
            # Check if the corresponding file exists
            if os.path.exists(mask_file_2):
                # Define the combined mask filename
                new_name = base_name.replace('localizer_loc1', 'combined')
                print(new_name)
                # Replace sXX with sub-XX
                new_name = re.sub(r's(\d+)', r'sub-\1', new_name)
                print(new_name)
                

                combined_mask_filename = os.path.join(output_folder, new_name)
                
                # check if combined_mask_filename already exists, if it does, skip the processing and print a message
                if os.path.exists(combined_mask_filename):
                    print(f"File already exists, skipping: {combined_mask_filename}")
                    continue
                
                # Check if the combined mask already exists
                # if os.path.exists(combined_mask_filename):
                #     print(f"File already exists, skipping: {combined_mask_filename}")
                # else:
                # Load the masks
                mask_loc_1 = image.load_img(mask_file_1)
                mask_loc_2 = image.load_img(mask_file_2)
                
                # Combine the masks
                combined_mask = combine_masks(mask_loc_1, mask_loc_2)

                # Save the combined mask in the output folder
                combined_mask.to_filename(combined_mask_filename)
                print(f"Saved combined mask to {combined_mask_filename}")
            else:
                print(f"Corresponding file not found for {base_name}: {corresponding_name}")

# Load project parameters
root_folder = get_root(".my_hidden_root_file")

out_path = create_script_named_folder()
path_masks_voxel_reliability = opj(root_folder, 'code', '3_reliability_voxel_selection', 'generate_voxel_reliability_mask_localizer_1', 'reliability_masks')
# Call the function to process the masks
process_and_save_masks(path_masks_voxel_reliability, out_path, ROIS=['vis', 'mtl', 'cortical_memory', 'parietal_precuneus', 'parietal', 'precuneus', 'medial_ofc', 'rostral_acc', 'caudal_acc', 'superior_frontal', 'rsc', 'posterior_cingulate'])

