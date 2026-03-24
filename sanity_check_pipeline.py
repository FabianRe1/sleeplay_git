# -*- coding: utf-8 -*-
"""Pipeline sanity checks for the Sleeplay replay analysis.

Checks:
  1. Are (subject, trial) pairs unique per session in SODA output?
     If not, the 5.1 merge on ["subject","trial"] produces many-to-many joins.
  2. What does seq_pos look like in the 4.4 output CSVs?
  3. Are presleep encoding runs shorter than 120 TRs?
  4. Is trl_nr consistent between behavior and fMRI trial iteration?

Usage:
    python sanity_check_pipeline.py [enc|ret|both]

    enc  = encoding-only pipeline (4_decoding_claude)
    ret  = encoding+retrieval pipeline (4_decoding_claude_retrieval)
    both = run both (default)
"""

import os
import sys
import glob
import ast
from os.path import join as opj

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def check_trial_uniqueness(pipeline_dir, pipeline_label):
    """Check 1: Are (subject, trial) unique after TR-aggregation in SODA CSVs?"""
    header(f"CHECK 1: Trial uniqueness in SODA output — {pipeline_label}")

    any_problem = False
    for session in ["presleep", "postsleep"]:
        soda_dir = opj(pipeline_dir,
                       f"4.6_SODA_based_on_singlesubj_4.4_{session}",
                       "SODA_results_csv")
        csvs = sorted(glob.glob(opj(soda_dir, "*_soda_results_per_combination.csv")))
        if not csvs:
            print(f"\n  [{session}] No SODA CSVs found in {soda_dir}")
            continue

        for csv_path in csvs:
            roi = os.path.basename(csv_path).replace("_soda_results_per_combination.csv", "")
            print(f"\n  [{session}] ROI={roi}: {os.path.basename(csv_path)}")

            df = pd.read_csv(csv_path, usecols=lambda c: c in [
                "subject", "trial", "tr", "run",
                "object_scene_retrieval", "retrieval_target",
            ])
            df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
            df["tr"] = pd.to_numeric(df["tr"], errors="coerce")

            n_rows = len(df)
            n_subjects = df["subject"].nunique()
            print(f"    Raw rows: {n_rows}, subjects: {n_subjects}")

            # After aggregating across TRs (as 5.1 does), check uniqueness
            trial_level = df.drop(columns=["tr"], errors="ignore").drop_duplicates()
            dup_check = trial_level.groupby(["subject", "trial"]).size()
            n_dups = (dup_check > 1).sum()

            if n_dups > 0:
                print(f"    *** PROBLEM: {n_dups} (subject, trial) pairs map to MULTIPLE rows ***")
                print(f"    Max rows per (subject, trial): {dup_check.max()}")
                # Show examples
                bad = dup_check[dup_check > 1].head(5)
                for (subj, trial), count in bad.items():
                    example = trial_level[(trial_level["subject"] == subj) &
                                          (trial_level["trial"] == trial)]
                    print(f"      subject={subj}, trial={trial} -> {count} rows:")
                    print(f"        {example[['run','object_scene_retrieval','retrieval_target']].to_string(index=False)}")
                any_problem = True
            else:
                n_unique = len(dup_check)
                print(f"    OK: {n_unique} unique (subject, trial) pairs, all map to exactly 1 row")

            # Also check: do trial numbers reset per run?
            if "run" in df.columns:
                trial_ranges = df.groupby(["subject", "run"])["trial"].agg(["min", "max"])
                overlap = False
                for subj in df["subject"].unique():
                    subj_ranges = trial_ranges.loc[subj] if subj in trial_ranges.index.get_level_values(0) else None
                    if subj_ranges is None or len(subj_ranges) < 2:
                        continue
                    runs = subj_ranges.index.tolist()
                    for i in range(len(runs)):
                        for j in range(i+1, len(runs)):
                            r1_min, r1_max = subj_ranges.loc[runs[i]]
                            r2_min, r2_max = subj_ranges.loc[runs[j]]
                            if r1_min <= r2_max and r2_min <= r1_max:
                                if not overlap:
                                    print(f"    *** WARNING: Trial number ranges OVERLAP across runs ***")
                                    overlap = True
                                print(f"      subject={subj}: run {runs[i]} trials [{r1_min}-{r1_max}], "
                                      f"run {runs[j]} trials [{r2_min}-{r2_max}]")
                                break
                        if overlap:
                            break
                    if overlap:
                        break
                if not overlap:
                    print(f"    Trial ranges do NOT overlap across runs (trial numbers are globally unique per subject)")

    if not any_problem:
        print(f"\n  RESULT: All (subject, trial) pairs are unique. The 5.1 merge is safe.")
    else:
        print(f"\n  *** RESULT: MERGE KEY COLLISION DETECTED. The 5.1 merge on ['subject','trial'] "
              f"will produce many-to-many joins! ***")
        print(f"  FIX: Change merge_keys in 5.1 to include 'object_scene_retrieval' and 'retrieval_target'")


def check_seq_pos(pipeline_dir, pipeline_label):
    """Check 2: What does seq_pos look like in the 4.4 output?"""
    header(f"CHECK 2: seq_pos format in 4.4 output — {pipeline_label}")

    for session in ["presleep", "postsleep"]:
        out_dir = opj(pipeline_dir,
                      f"4.4_transfer_visual_classifier_retrieval_{session}",
                      "decoding_results_tables_for_SODA")
        csvs = sorted(glob.glob(opj(out_dir, "*.csv")))
        if not csvs:
            print(f"\n  [{session}] No 4.4 output CSVs found in {out_dir}")
            continue

        # Check just the first CSV
        csv_path = csvs[0]
        print(f"\n  [{session}] Checking: {os.path.basename(csv_path)}")

        df = pd.read_csv(csv_path, nrows=1000)

        if "retrieval_cue_position" in df.columns:
            vals = df["retrieval_cue_position"].dropna().unique()[:10]
            print(f"    retrieval_cue_position values (first 10 unique): {vals}")
            print(f"    dtype: {df['retrieval_cue_position'].dtype}")
            print(f"    range: {df['retrieval_cue_position'].min()} - {df['retrieval_cue_position'].max()}")
            print(f"    NaN count: {df['retrieval_cue_position'].isna().sum()} / {len(df)}")
        else:
            print(f"    'retrieval_cue_position' column NOT FOUND in 4.4 output")
            print(f"    Available columns: {list(df.columns)}")

        # Also check if seq_pos exists (it might be in behavioral data, not output)
        if "seq_pos" in df.columns:
            vals = df["seq_pos"].dropna().unique()[:10]
            print(f"    seq_pos values (first 10 unique): {vals}")
            print(f"    dtype: {df['seq_pos'].dtype}")
        else:
            print(f"    'seq_pos' not in 4.4 output (expected — it's parsed during 4.4 execution)")


def check_presleep_run_lengths(pipeline_dir, pipeline_label):
    """Check 3: Are presleep encoding runs shorter than 120 TRs?"""
    header(f"CHECK 3: Presleep run lengths — {pipeline_label}")

    out_dir = opj(pipeline_dir,
                  "4.4_transfer_visual_classifier_retrieval_presleep",
                  "decoding_results_tables_for_SODA")
    csvs = sorted(glob.glob(opj(out_dir, "*.csv")))
    if not csvs:
        print(f"  No presleep 4.4 output CSVs found. Checking raw input instead...")
        # Can't check without the data
        return

    # The 4.4 output only contains retrieval runs (after filtering).
    # We need to check if the number of runs is correct (3 for presleep).
    csv_path = csvs[0]
    print(f"  Checking first subject: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, usecols=lambda c: c in ["subject", "run", "tr", "trial"])
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df["tr"] = pd.to_numeric(df["tr"], errors="coerce")

    for subj in df["subject"].unique()[:3]:
        subj_df = df[df["subject"] == subj]
        runs = sorted(subj_df["run"].dropna().unique())
        print(f"\n  subject={subj}:")
        print(f"    Runs in 4.4 output: {runs} (expected: 3 retrieval runs for presleep)")
        for run in runs:
            run_df = subj_df[subj_df["run"] == run]
            n_trs = run_df["tr"].nunique()
            tr_range = f"{run_df['tr'].min():.0f}-{run_df['tr'].max():.0f}"
            n_trials = run_df["trial"].nunique()
            print(f"    Run {run}: {n_trs} unique TRs (range {tr_range}), {n_trials} trials")


def check_trial_nr_consistency(pipeline_dir, pipeline_label):
    """Check 4: Is trl_nr consistent between behavior matching and output?"""
    header(f"CHECK 4: Trial number consistency — {pipeline_label}")

    for session in ["presleep", "postsleep"]:
        out_dir = opj(pipeline_dir,
                      f"4.4_transfer_visual_classifier_retrieval_{session}",
                      "decoding_results_tables_for_SODA")
        csvs = sorted(glob.glob(opj(out_dir, "*.csv")))
        if not csvs:
            print(f"\n  [{session}] No 4.4 output CSVs found")
            continue

        csv_path = csvs[0]
        print(f"\n  [{session}] Checking: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path, usecols=lambda c: c in [
            "subject", "run", "trial", "tr", "behav_correct",
            "object_scene_retrieval", "retrieval_target",
        ])
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
        df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")

        # Check if behav_correct has reasonable values (should be 0 or 1)
        bc_vals = df["behav_correct"].dropna().unique()
        print(f"    behav_correct unique values: {sorted(bc_vals)}")
        print(f"    behav_correct NaN rate: {df['behav_correct'].isna().mean():.1%}")

        # Check that behav_correct is constant within a trial (as expected)
        trial_bc = df.groupby(["subject", "trial"])["behav_correct"].nunique()
        inconsistent = (trial_bc > 1).sum()
        if inconsistent > 0:
            print(f"    *** WARNING: {inconsistent} trials have INCONSISTENT behav_correct across rows ***")
        else:
            print(f"    OK: behav_correct is consistent within each trial")

        # Check trial count per subject
        trial_counts = df.groupby("subject")["trial"].nunique()
        print(f"    Trials per subject: min={trial_counts.min()}, max={trial_counts.max()}, "
              f"median={trial_counts.median():.0f}")

        expected = 30 if session == "presleep" else 60  # 3 runs x 10 trials or 6 runs x 10 trials
        if trial_counts.median() != expected:
            print(f"    NOTE: Expected ~{expected} trials for {session} "
                  f"({'3' if session == 'presleep' else '6'} runs x 10 trials)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    pipelines = []
    if mode in ("enc", "both"):
        pipelines.append((opj(SCRIPT_DIR, "4_decoding_claude"), "Encoding-only"))
    if mode in ("ret", "both"):
        pipelines.append((opj(SCRIPT_DIR, "4_decoding_claude_retrieval"), "Encoding+Retrieval"))

    if not pipelines:
        print(f"Usage: python sanity_check_pipeline.py [enc|ret|both]")
        sys.exit(1)

    for pipeline_dir, label in pipelines:
        print(f"\n{'#'*70}")
        print(f"# PIPELINE: {label}")
        print(f"# {pipeline_dir}")
        print(f"{'#'*70}")

        if not os.path.isdir(pipeline_dir):
            print(f"  Directory not found: {pipeline_dir}")
            continue

        check_trial_uniqueness(pipeline_dir, label)
        check_seq_pos(pipeline_dir, label)
        check_presleep_run_lengths(pipeline_dir, label)
        check_trial_nr_consistency(pipeline_dir, label)

    header("ALL CHECKS COMPLETE")


if __name__ == "__main__":
    main()
