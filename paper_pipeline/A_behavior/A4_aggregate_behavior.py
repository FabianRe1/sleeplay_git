#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4 - Combine all participants' main-task behaviour into one table.

For each participant and each session (nses_0, nses_1, nses_2) the folder
with the highest step number is used, and its TR-annotated log from A1 is
loaded. Adds the columns sub_id, ses_nr and source_file.

Input   outputs/A1_add_tr_columns_main/sXX/<session folder>/output_df_*_w_HRF.csv
Output  outputs/A4_aggregate_behavior/all_subjects_behav_data.csv

Ported from 1_prepare_behavior/3_aggregate_behav.py; logic unchanged. The
original had a hard-coded laptop path as its data folder.

Usage:  python A4_aggregate_behavior.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir

A1_DIR = step_input_dir("A1_add_tr_columns_main")
OUT_DIR = step_output_dir(__file__)


def discover_subjects(base_dir: Path) -> list[Path]:
    """Subject folders named s<number>."""
    pattern = re.compile(r"^s(\d+)$")
    return [e for e in sorted(base_dir.iterdir()) if e.is_dir() and pattern.match(e.name)]


def get_session_folders(main_dir: Path) -> dict[int, Path]:
    """Session number -> folder with the highest step.

    Folder naming: sub<N>_main_task_nses_<ses>_step<step>
    """
    pattern = re.compile(r"_nses_(\d+)_step(\d+)$")
    sessions: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for entry in main_dir.iterdir():
        if entry.is_dir():
            match = pattern.search(entry.name)
            if match:
                sessions[int(match.group(1))].append((int(match.group(2)), entry))

    best: dict[int, Path] = {}
    for ses_num, entries in sorted(sessions.items()):
        entries.sort(key=lambda x: x[0], reverse=True)
        best[ses_num] = entries[0][1]
    return best


def find_csv_in_folder(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("output_df_*_w_HRF.csv"))
    return candidates[0] if candidates else None


def load_all_data(base_dir: Path, verbose: bool = True):
    subjects = discover_subjects(base_dir)
    if verbose:
        print(f"Found {len(subjects)} subject folders: {[s.name for s in subjects]}\n")

    all_dfs = []
    log = {"loaded": [], "missing": [], "warnings": []}

    for sub_dir in subjects:
        sub_id = sub_dir.name
        session_folders = get_session_folders(sub_dir)
        if not session_folders:
            log["missing"].append((sub_id, "No session folders found"))
            if verbose:
                print(f"  ! {sub_id}: no session folders found")
            continue

        found = sorted(session_folders.keys())
        missing_ses = {0, 1, 2} - set(found)
        if missing_ses:
            msg = f"Missing expected session(s): {sorted(missing_ses)}"
            log["warnings"].append((sub_id, None, msg))
            if verbose:
                print(f"  ! {sub_id}: {msg}")
        if verbose:
            print(f"  {sub_id}: found sessions {found}")

        for ses_nr, folder in sorted(session_folders.items()):
            csv_path = find_csv_in_folder(folder)
            if csv_path is None:
                log["missing"].append((sub_id, f"No output CSV found in {folder.name}"))
                if verbose:
                    print(f"    ! ses {ses_nr}: no output CSV in {folder.name}")
                continue
            try:
                df = pd.read_csv(csv_path)
                df["sub_id"] = sub_id
                df["ses_nr"] = ses_nr
                df["source_file"] = str(csv_path.relative_to(base_dir))
                all_dfs.append(df)
                log["loaded"].append((sub_id, ses_nr, str(csv_path)))
                if verbose:
                    print(f"    ok ses {ses_nr}: {len(df)} rows  ({folder.name})")
            except Exception as e:
                log["missing"].append((sub_id, f"Error loading {csv_path.name}: {e}"))
                if verbose:
                    print(f"    x ses {ses_nr}: error loading {csv_path.name}: {e}")

    all_data = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if verbose:
        print(f"\nSubjects found: {len(subjects)}   files loaded: {len(log['loaded'])}   "
              f"missing/errors: {len(log['missing'])}   warnings: {len(log['warnings'])}   "
              f"rows: {len(all_data)}")
    return all_data, log


if __name__ == "__main__":
    df, log = load_all_data(A1_DIR)
    if df.empty:
        raise SystemExit("No data loaded. Run A1_add_tr_columns_main.py first.")

    print(f"\nLoaded table: {df.shape}")
    print("Sessions per subject:")
    print(df.groupby("sub_id")["ses_nr"].nunique().to_string())

    out_file = OUT_DIR / "all_subjects_behav_data.csv"
    df.to_csv(out_file)
    print(f"\nA4 done: {out_file}")
