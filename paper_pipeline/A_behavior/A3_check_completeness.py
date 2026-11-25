#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3 - Check that every participant's behaviour files are complete.

Checks per participant:
  - both localizer logs exist and reach block 50
  - evening (nses_0) and morning (nses_2) main-task folders exist
  - their TR-annotated logs (from A1) exist and reach trial 130
  - no unexpected main-task session folders

Input   behavior/ (read-only) and outputs/A1_add_tr_columns_main/
Output  outputs/A3_check_completeness/completeness_report.csv
        (one row per problem found; an empty file means nothing is missing)

Ported from 1_prepare_behavior/2_check_missing.py; same checks, with the
messages collected in a table instead of only printed.

Usage:  python A3_check_completeness.py
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BEHAVIOR_DIR, behavior_subjects, step_input_dir, step_output_dir

A1_DIR = step_input_dir("A1_add_tr_columns_main")
OUT_DIR = step_output_dir(__file__)

EXPECTED_FOLDER_ENDINGS = ("nses_0_step0", "nses_2_step2", "nses_1_step0", "nses_1_step1")
problems = []


def report(subj, message):
    print(f"{subj}: {message}")
    problems.append({"subject": subj, "problem": message})


for subj in behavior_subjects():
    n = subj[1:]

    # 1. Localizers
    for loc in (1, 2):
        path = BEHAVIOR_DIR / subj / "behav" / "loc" / f"s{n}_df_loc{loc}.csv"
        if not path.exists():
            report(subj, f"Missing localizer {loc}")
            continue
        max_block = pd.read_csv(path, sep=None, engine="python")["block_nr"].max()
        if max_block < 50:
            report(subj, f"Localizer {loc} incomplete, max block_nr: {max_block}")

    # 2. Main task: evening (nses_0_step0) and morning (nses_2_step2)
    main_dir = BEHAVIOR_DIR / subj / "behav" / "main"
    for label, folder in (("evening", f"sub{n}_main_task_nses_0_step0"),
                          ("morning", f"sub{n}_main_task_nses_2_step2")):
        if not (main_dir / folder).is_dir():
            report(subj, f"Missing main task {label} session folder ({folder})")
            continue
        csv_file = A1_DIR / subj / folder / f"output_df_sub{n}_w_HRF.csv"
        if not csv_file.exists():
            report(subj, f"Missing TR-annotated CSV for main task {label} session")
            continue
        max_trial = pd.read_csv(csv_file, sep=None, engine="python")["trl_nr"].max()
        if max_trial < 130:
            report(subj, f"Main task {label} session incomplete, max trl_nr: {max_trial}")

    # 3. Unexpected session folders
    for directory in glob.glob(str(main_dir / f"sub{n}_main_task_nses_*")):
        if os.path.isdir(directory) and not directory.endswith(EXPECTED_FOLDER_ENDINGS):
            report(subj, f"Unexpected main task directory: {os.path.basename(directory)}")

out_file = OUT_DIR / "completeness_report.csv"
pd.DataFrame(problems, columns=["subject", "problem"]).to_csv(out_file, index=False)
print(f"\nA3 done: {len(problems)} problems listed in {out_file}")
