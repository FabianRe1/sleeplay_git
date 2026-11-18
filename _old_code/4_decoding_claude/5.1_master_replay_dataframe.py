# -*- coding: utf-8 -*-
"""Build a master trial-level dataframe combining all replay metrics.

Combines SODA (absolute slope), sinusoidal fit (amplitude), and FFT (band power)
across ROIs and sessions into a single long-format CSV suitable for LME modeling.

Usage
-----
    python 5.1_master_replay_dataframe.py

Output
------
    5.1_master_replay_dataframe/master_replay_metrics.csv
"""

import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd

def script_output_dir(script_file, subfolder=""):
    """Create and return an output directory named after the script."""
    script_name = os.path.splitext(os.path.basename(script_file))[0]
    base = os.path.dirname(os.path.abspath(script_file))
    out = opj(base, script_name, subfolder) if subfolder else opj(base, script_name)
    os.makedirs(out, exist_ok=True)
    return out

# =====================================================================
# CONFIGURATION
# =====================================================================

FOCUS_ROIS = ["vis", "mtl", "parietal"]
SESSIONS = ["presleep", "postsleep"]

# =====================================================================
# LOADING FUNCTIONS
# =====================================================================

def load_soda_trial_level(script_dir, roi, session):
    """Load SODA CSV, aggregate across TRs to get trial-level absolute slope."""
    csv_path = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{session}",
                   "SODA_results_csv", f"{roi}_soda_results_per_combination.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: SODA CSV not found: {csv_path}")
        return None

    # Only load columns we need (SODA CSVs are ~64MB each due to probability arrays)
    usecols = [
        "subject", "trial", "run", "object_scene_retrieval", "retrieval_target",
        "retrieved_position", "behav_correct", "tr",
        "retrieved_soda_abs", "control_soda_abs", "retrieved_soda", "control_soda",
        "retrieved_soda_shuffled_abs", "control_soda_shuffled_abs",
        "retrieved_soda_shuffled_selected_abs", "control_soda_shuffled_selected_abs",
    ]
    print(f"  Loading SODA CSV (selective columns for speed)...")
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)

    # Ensure numeric
    for col in df.columns:
        if col not in ["subject", "object_scene_retrieval", "retrieval_target"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate across TRs: mean per trial
    group_cols = ["subject", "trial", "run", "object_scene_retrieval",
                  "retrieval_target", "retrieved_position", "behav_correct"]
    # Keep only columns that exist
    group_cols = [c for c in group_cols if c in df.columns]

    agg_cols = {
        "retrieved_soda_abs": "mean",
        "control_soda_abs": "mean",
        "retrieved_soda": "mean",
        "control_soda": "mean",
    }
    # Add shuffled columns if present
    for col in ["retrieved_soda_shuffled_abs", "control_soda_shuffled_abs",
                "retrieved_soda_shuffled_selected_abs", "control_soda_shuffled_selected_abs"]:
        if col in df.columns:
            agg_cols[col] = "mean"

    trial_df = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    return trial_df


def load_sinusoidal_trial_level(script_dir, roi, session):
    """Load sinusoidal fit results (already trial-level)."""
    csv_path = opj(script_dir, "4.7_sinus_1cycle", roi, f"{roi}_windowed_fit_results_{session}.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: Sinusoidal CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Take absolute amplitude (sign is arbitrary for the sinusoidal fit)
    for col in ["retrieved_w_amplitude", "control_w_amplitude",
                "shuf_retrieved_w_amplitude", "shuf_control_w_amplitude",
                "shuf_sel_retrieved_w_amplitude", "shuf_sel_control_w_amplitude"]:
        if col in df.columns:
            df[col] = df[col].abs()

    keep_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target",
                 "behav_correct", "retrieved_position",
                 "retrieved_w_amplitude", "control_w_amplitude",
                 "retrieved_w_r_squared", "control_w_r_squared",
                 "retrieved_w_frequency", "control_w_frequency",
                 "shuf_retrieved_w_amplitude", "shuf_control_w_amplitude",
                 "shuf_sel_retrieved_w_amplitude", "shuf_sel_control_w_amplitude"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def load_fft_trial_level(script_dir, roi, session):
    """Load FFT band power results (already trial-level)."""
    csv_path = opj(script_dir, "4.8_fft", roi, f"{roi}_fft_results_{session}.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: FFT CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    keep_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target",
                 "behav_correct", "retrieved_position",
                 "retrieved_band_power", "control_band_power",
                 "retrieved_relative_band_power", "control_relative_band_power",
                 "shuf_retrieved_band_power", "shuf_control_band_power",
                 "shuf_sel_retrieved_band_power", "shuf_sel_control_band_power"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


# =====================================================================
# MAIN
# =====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_output_dir(__file__, "")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    for session in SESSIONS:
        for roi in FOCUS_ROIS:
            print(f"\n{'='*60}")
            print(f"Loading {roi} / {session}")
            print(f"{'='*60}")

            # --- SODA ---
            soda_df = load_soda_trial_level(script_dir, roi, session)

            # --- Sinusoidal ---
            sinus_df = load_sinusoidal_trial_level(script_dir, roi, session)

            # --- FFT ---
            fft_df = load_fft_trial_level(script_dir, roi, session)

            if soda_df is None:
                print(f"  Skipping {roi}/{session}: no SODA data")
                continue

            # Use SODA as the base (it has run info)
            merge_keys = ["subject", "trial"]

            # Merge sinusoidal
            if sinus_df is not None:
                soda_df = soda_df.merge(
                    sinus_df.drop(columns=[c for c in ["object_scene_retrieval", "retrieval_target",
                                                        "behav_correct", "retrieved_position"]
                                           if c in sinus_df.columns], errors="ignore"),
                    on=merge_keys, how="left",
                )
                n_sinus = sinus_df[merge_keys].drop_duplicates().shape[0]
                print(f"  Sinusoidal: {n_sinus} trials merged")
            else:
                print(f"  Sinusoidal: NOT AVAILABLE")

            # Merge FFT
            if fft_df is not None:
                soda_df = soda_df.merge(
                    fft_df.drop(columns=[c for c in ["object_scene_retrieval", "retrieval_target",
                                                      "behav_correct", "retrieved_position"]
                                         if c in fft_df.columns], errors="ignore"),
                    on=merge_keys, how="left",
                )
                n_fft = fft_df[merge_keys].drop_duplicates().shape[0]
                print(f"  FFT: {n_fft} trials merged")
            else:
                print(f"  FFT: NOT AVAILABLE")

            # Add metadata
            soda_df["roi"] = roi
            soda_df["session"] = session

            n_subj = soda_df["subject"].nunique()
            n_trials = len(soda_df)
            print(f"  Total: {n_subj} subjects, {n_trials} trials")

            all_rows.append(soda_df)

    if not all_rows:
        print("ERROR: No data loaded!")
        sys.exit(1)

    master = pd.concat(all_rows, ignore_index=True)

    # Ensure clean types
    master["behav_correct"] = pd.to_numeric(master["behav_correct"], errors="coerce")
    master["trial"] = pd.to_numeric(master["trial"], errors="coerce")
    if "run" in master.columns:
        master["run"] = pd.to_numeric(master["run"], errors="coerce")

    # Reorder columns for readability
    id_cols = ["subject", "session", "roi", "trial", "run",
               "object_scene_retrieval", "retrieval_target",
               "retrieved_position", "behav_correct"]
    id_cols = [c for c in id_cols if c in master.columns]
    other_cols = [c for c in master.columns if c not in id_cols]
    master = master[id_cols + sorted(other_cols)]

    out_path = opj(out_dir, "master_replay_metrics.csv")
    master.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Master dataframe saved: {out_path}")
    print(f"Shape: {master.shape}")
    print(f"Sessions: {master['session'].value_counts().to_dict()}")
    print(f"ROIs: {master['roi'].value_counts().to_dict()}")
    print(f"Subjects: {master['subject'].nunique()}")
    print(f"Columns: {list(master.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
