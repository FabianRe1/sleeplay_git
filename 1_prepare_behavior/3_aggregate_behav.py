"""
Load all Sleeplay behavioral CSV data across subjects and sessions.

For each subject folder (e.g., s1, s12, ...) in the behavior directory,
this script navigates to behav/main/, identifies all session folders
(nses_0, nses_1, nses_2, ...), selects the highest step version for each
session, and loads the output CSV.

Usage:
    Adjust BASE_DIR below, then run or import.
"""

import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from os.path import join as opj

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

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/renzadmin/MaxPlanck_PhD/PhD_Schuck/Hummel/Data/Sleeplay_Hummel/behavior")
# ─────────────────────────────────────────────────────────────────────────────


def discover_subjects(base_dir: Path) -> list[Path]:
    """Find all subject folders matching 's<number>' pattern."""
    pattern = re.compile(r"^s(\d+)$")
    subjects = []
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and pattern.match(entry.name):
            subjects.append(entry)
    return subjects


def get_session_folders(main_dir: Path) -> dict[int, Path]:
    """
    For a given subject's main/ directory, find all session folders and
    return a dict mapping session number -> folder path (highest step).

    Folder naming: sub<N>_main_task_nses_<ses>_step<step>
    """
    pattern = re.compile(r"_nses_(\d+)_step(\d+)$")

    # Collect all (session, step, path) tuples
    sessions: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for entry in main_dir.iterdir():
        if entry.is_dir():
            match = pattern.search(entry.name)
            if match:
                ses_num = int(match.group(1))
                step_num = int(match.group(2))
                sessions[ses_num].append((step_num, entry))

    # For each session, pick the folder with the highest step
    best: dict[int, Path] = {}
    for ses_num, entries in sorted(sessions.items()):
        entries.sort(key=lambda x: x[0], reverse=True)
        best[ses_num] = entries[0][1]  # highest step

    return best


def find_csv_in_folder(folder: Path) -> Path | None:
    """Find the output CSV file in a session folder."""
    candidates = list(folder.glob("output_df_*_w_HRF.csv"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Prefer the one matching the expected pattern
        return sorted(candidates)[0]
    return None


def load_all_data(base_dir: Path | None = None, verbose: bool = True):
    """
    Main loader. Returns:
        all_data : pd.DataFrame  — concatenated data with sub_id and ses_nr columns
        log      : dict          — loading log with warnings/errors per subject
    """
    if base_dir is None:
        base_dir = BASE_DIR

    subjects = discover_subjects(base_dir)
    if verbose:
        print(f"Found {len(subjects)} subject folders: {[s.name for s in subjects]}\n")

    all_dfs = []
    log = {
        "loaded": [],       # (sub_id, ses_nr, path)
        "missing": [],      # (sub_id, issue description)
        "warnings": [],     # (sub_id, ses_nr, warning)
    }

    for sub_dir in subjects:
        sub_id = sub_dir.name  # e.g. 's12'

        # Check expected directory structure
        main_dir = sub_dir / "behav" / "main"
        if not main_dir.exists():
            msg = f"Missing behav/main/ directory"
            log["missing"].append((sub_id, msg))
            if verbose:
                print(f"  ⚠ {sub_id}: {msg}")
            continue

        # Discover sessions
        session_folders = get_session_folders(main_dir)
        if not session_folders:
            msg = f"No session folders found in {main_dir}"
            log["missing"].append((sub_id, msg))
            if verbose:
                print(f"  ⚠ {sub_id}: {msg}")
            continue

        # Check for expected sessions (0, 1, 2)
        found_sessions = sorted(session_folders.keys())
        expected = {0, 1, 2}
        missing_ses = expected - set(found_sessions)
        if missing_ses:
            msg = f"Missing expected session(s): {sorted(missing_ses)}"
            log["warnings"].append((sub_id, None, msg))
            if verbose:
                print(f"  ⚠ {sub_id}: {msg}")

        if verbose:
            print(f"  {sub_id}: found sessions {found_sessions}")

        # Load each session
        for ses_nr, folder in sorted(session_folders.items()):
            csv_path = find_csv_in_folder(folder)
            if csv_path is None:
                msg = f"No output CSV found in {folder.name}"
                log["missing"].append((sub_id, msg))
                if verbose:
                    print(f"    ⚠ ses {ses_nr}: {msg}")
                continue

            try:
                df = pd.read_csv(csv_path)
                df["sub_id"] = sub_id
                df["ses_nr"] = ses_nr
                df["source_file"] = str(csv_path)
                all_dfs.append(df)
                log["loaded"].append((sub_id, ses_nr, str(csv_path)))
                if verbose:
                    print(f"    ✓ ses {ses_nr}: {len(df)} rows  ({folder.name})")
            except Exception as e:
                msg = f"Error loading {csv_path.name}: {e}"
                log["missing"].append((sub_id, msg))
                if verbose:
                    print(f"    ✗ ses {ses_nr}: {msg}")

    # Concatenate
    if all_dfs:
        all_data = pd.concat(all_dfs, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Subjects found:    {len(subjects)}")
        print(f"  Files loaded:      {len(log['loaded'])}")
        print(f"  Missing/errors:    {len(log['missing'])}")
        print(f"  Warnings:          {len(log['warnings'])}")
        print(f"  Total rows:        {len(all_data)}")
        print(f"{'='*60}")

        if log["missing"]:
            print(f"\nMISSING / ERRORS:")
            for sub_id, msg in log["missing"]:
                print(f"  {sub_id}: {msg}")

    return all_data, log


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, log = load_all_data()

    if not df.empty:
        print(f"\nLoaded DataFrame shape: {df.shape}")
        print(f"Subjects: {sorted(df['sub_id'].unique())}")
        print(f"Sessions per subject:")
        print(df.groupby("sub_id")["ses_nr"].nunique().to_string())
        
        # create dir
        save_path = create_script_named_folder('all_data_csv')
        
        # save df
        df.to_csv(opj(save_path, 'all_subjects_behav_data.csv'))
     