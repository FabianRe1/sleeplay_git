#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 - Add TR columns to the localizer behaviour files.

Same as A1, for the two localizer sessions (loc1, loc2). These files are the
labels for classifier training.

Input   behavior/sXX/behav/loc/sXX_df_loc{1,2}.csv            (read-only)
Output  outputs/A2_add_tr_columns_localizer/sXX/sXX_df_loc{1,2}_w_HRF.csv
        (semicolon-separated, as in the original)

Ported from 1_prepare_behavior/1_localizer_extract_TR.py; logic unchanged.

Usage:  python A2_add_tr_columns_localizer.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BEHAVIOR_DIR, HRF_DELAY, behavior_subjects, find_closest_TR, step_output_dir

OUT_DIR = step_output_dir(__file__)
LOCALIZER_SESSIONS = [1, 2]

n_written = 0
for subj in behavior_subjects():
    subj_number = subj[1:]
    for session in LOCALIZER_SESSIONS:
        src = BEHAVIOR_DIR / subj / "behav" / "loc" / f"s{subj_number}_df_loc{session}.csv"
        try:
            data = pd.read_csv(src, sep=None, engine="python")
        except Exception:
            print(f"Could not read localizer {session} for {subj}. Skipping.")
            continue

        # The raw localizer logs of some participants carry two extra columns that
        # the old reliability-mask script wrote back into them (March 2026). They are
        # derived, not raw, and are rebuilt by the classifier step, so drop them here.
        data = data.drop(columns=["encoding_stimulus", "encoding_category"], errors="ignore")

        data["time_onset_w_HRF"] = None
        data["TR_time_onset"] = None
        data["TR_time_onset_w_HRF"] = None

        for trial in range(data.shape[0]):
            data.loc[trial, "time_onset_w_HRF"] = data.loc[trial, "time_onset"] + HRF_DELAY
            data.loc[trial, "TR_time_onset"] = find_closest_TR(data.loc[trial, "time_onset"])
            data.loc[trial, "TR_time_onset_w_HRF"] = find_closest_TR(data.loc[trial, "time_onset_w_HRF"])

        save_path = OUT_DIR / subj / f"s{subj_number}_df_loc{session}_w_HRF.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, sep=";", index=False)
        n_written += 1
        print(f"Wrote {subj} loc{session}")

print(f"\nA2 done: {n_written} files written to {OUT_DIR}")
