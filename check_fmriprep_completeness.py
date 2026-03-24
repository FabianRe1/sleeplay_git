#!/usr/bin/env python3
"""
Check fmriprep derivatives directory for completeness.
Reports which subjects have/lack files ending in T1w_desc-aparcaseg_dseg.nii.gz
"""

import os
import glob
from os.path import join as opj

path_fmriprep = "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

# Find all subject directories
sub_dirs = sorted(glob.glob(opj(path_fmriprep, "sub-*")))
sub_dirs = [d for d in sub_dirs if os.path.isdir(d)]

target_suffix = "T1w_desc-aparcaseg_dseg.nii.gz"

found = []
missing = []

print(f"Searching in: {path_fmriprep}")
print(f"Looking for files ending in: {target_suffix}")
print(f"Total subject directories found: {len(sub_dirs)}\n")
print("=" * 60)

for sub_dir in sub_dirs:
    sub_id = os.path.basename(sub_dir)
    # Search recursively for the target file
    matches = glob.glob(opj(sub_dir, "**", f"*{target_suffix}"), recursive=True)
    if matches:
        found.append(sub_id)
        for m in matches:
            print(f"  FOUND  | {sub_id} | {os.path.relpath(m, path_fmriprep)}")
    else:
        missing.append(sub_id)
        print(f"  MISSING | {sub_id}")

print("=" * 60)
print(f"\nSummary:")
print(f"  Subjects with file:    {len(found)}/{len(sub_dirs)}")
print(f"  Subjects without file: {len(missing)}/{len(sub_dirs)}")

if missing:
    print(f"\nMissing subjects:")
    for s in missing:
        print(f"  - {s}")

if found:
    print(f"\nFound subjects:")
    for s in found:
        print(f"  - {s}")
