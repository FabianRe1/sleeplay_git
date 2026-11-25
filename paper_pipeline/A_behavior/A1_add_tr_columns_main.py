#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1 - Add TR columns to the main-task behaviour files.

For every raw main-task log (behavior/sXX/behav/main/<session>/output*.csv)
this adds the HRF-shifted onset and the TR indices used by all fMRI steps:

    time_onset_w_HRF      t_onset + HRF_DELAY (s)
    TR_time_onset         index of the TR closest to t_onset
    TR_time_onset_w_HRF   index of the TR closest to time_onset_w_HRF

Input   behavior/sXX/behav/main/<session folder>/output*.csv   (read-only)
Output  outputs/A1_add_tr_columns_main/sXX/<session folder>/<name>_w_HRF.csv

Ported from 1_prepare_behavior/1_enc_ret_extract_TR.py; logic unchanged.
The original wrote the *_w_HRF.csv files next to the raw logs.

Added 2026-11-25: for the sessions in config.RELABELLED_SESSIONS (the morning
logs of s41, which were recorded with another participant's trial list; see
config.py) the retrieval targets are replaced by the participant's own learned
item at the cued scene and position. The targets as logged are kept in the
columns `obj_ret_targ_logged` and `sc_ret_targ_logged`. The raw logs are not
changed.

Usage:  python A1_add_tr_columns_main.py [subject ...]     e.g. s41; default: all
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
from config import BEHAVIOR_DIR, HRF_DELAY, behavior_subjects, find_closest_TR, is_relabelled, step_output_dir
from decoding_utils import relabel_targets_from_cue


def encoding_log(subj):
    """The raw evening log of a participant that contains the learning trials."""
    for f in sorted(glob.glob(str(BEHAVIOR_DIR / subj / "behav" / "main" / "*_nses_0_step*" / "output*.csv")), reverse=True):
        if "_w_HRF" in f:
            continue
        d = pd.read_csv(f, sep=",")
        if "obj_enc_stim" in d.columns and d["obj_enc_stim"].notna().any():
            return d
    raise FileNotFoundError(f"no evening log with learning trials for {subj}")


OUT_DIR = step_output_dir(__file__)

n_written = 0
only = set(sys.argv[1:])
for subj in [s for s in behavior_subjects() if not only or s in only]:
    main_dir = BEHAVIOR_DIR / subj / "behav" / "main"
    csv_files = sorted(glob.glob(str(main_dir / "**" / "*.csv"), recursive=True))

    for csv_file in csv_files:
        rel = os.path.relpath(csv_file, main_dir)

        if "_w_HRF" in rel:
            continue  # derived file left by the original script
        if "loc" in rel:
            continue  # localizer files are handled in A2
        if "output" not in rel:
            continue  # only the task output logs

        print(f"Processing {subj}/{rel}")
        try:
            data = pd.read_csv(csv_file, sep=",")
        except Exception as e:
            print(f"  Could not read {csv_file}: {e}")
            continue

        if "t_onset" not in data.columns:
            continue

        data["t_onset"] = pd.to_numeric(data["t_onset"], errors="coerce")
        data["time_onset_w_HRF"] = None
        data["TR_time_onset_w_HRF"] = None

        for trial in range(data.shape[0]):
            if not isinstance(data.loc[trial, "t_onset"], (float, int)):
                print(f"  Skipping row {trial} due to invalid t_onset")
                continue
            data.loc[trial, "time_onset_w_HRF"] = data.loc[trial, "t_onset"] + HRF_DELAY
            data.loc[trial, "TR_time_onset"] = find_closest_TR(data.loc[trial, "t_onset"])
            data.loc[trial, "TR_time_onset_w_HRF"] = find_closest_TR(data.loc[trial, "time_onset_w_HRF"])

        session = "presleep" if "_nses_0_" in rel else "postsleep"
        if is_relabelled(subj, session) and data["label_type"].isin(["sc_no4obj", "d_no4sc"]).any():
            data = relabel_targets_from_cue(encoding_log(subj), data, f"{subj} {rel}")

        save_path = OUT_DIR / subj / rel.replace(".csv", "_w_HRF.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        n_written += 1

print(f"\nA1 done: {n_written} files written to {OUT_DIR}")
