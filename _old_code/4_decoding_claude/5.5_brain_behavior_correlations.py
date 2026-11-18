# -*- coding: utf-8 -*-
"""Brain-behavior correlations: does overall replay strength predict accuracy?

Uses raw retrieved metric (not retrieved-control difference) correlated with
behavioral accuracy at the subject level.

Usage
-----
    python 5.5_brain_behavior_correlations.py

Output
------
    5.5_brain_behavior_correlations/
        brain_behavior_collapsed.csv
        brain_behavior_per_session.csv
        results.txt
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = opj(SCRIPT_DIR, "5.5_brain_behavior_correlations")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

METRICS = [
    ("retrieved_soda_abs", "Abs SODA"),
    ("retrieved_w_amplitude", "Sinus Amp"),
    ("retrieved_band_power", "FFT Power"),
]
ROIS = ["vis", "mtl", "parietal"]


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "."
    return ""


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(MASTER_CSV)
    df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")
    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")

    log = []

    # ==================================================================
    # Collapsed across sessions
    # ==================================================================
    log.append("=" * 70)
    log.append("Brain-behavior: does overall replay strength predict accuracy?")
    log.append("  Subject-level Pearson r(mean_retrieved_metric, mean_accuracy)")
    log.append("=" * 70)

    rows_collapsed = []
    log.append("\n--- Collapsed across sessions ---")
    for ret_col, name in METRICS:
        log.append(f"\n  {name}:")
        for roi in ROIS:
            sub = df[df["roi"] == roi].dropna(subset=[ret_col, "behav_correct"])
            smeans = sub.groupby("subject").agg(
                replay=(ret_col, "mean"), acc=("behav_correct", "mean")
            ).dropna()
            r, p = pearsonr(smeans["replay"], smeans["acc"])
            s = sig_stars(p)
            log.append(f"    {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
            rows_collapsed.append(dict(
                metric=name, roi=roi, session="collapsed",
                r=r, p_value=p, n=len(smeans),
            ))

    tbl_collapsed = pd.DataFrame(rows_collapsed)
    tbl_collapsed.to_csv(opj(RESULTS_DIR, "brain_behavior_collapsed.csv"), index=False)

    # ==================================================================
    # Per session
    # ==================================================================
    rows_session = []
    for session in ["presleep", "postsleep"]:
        log.append(f"\n--- {session} ---")
        for ret_col, name in METRICS:
            log.append(f"\n  {name}:")
            for roi in ROIS:
                sub = df[(df["roi"] == roi) & (df["session"] == session)].dropna(
                    subset=[ret_col, "behav_correct"]
                )
                smeans = sub.groupby("subject").agg(
                    replay=(ret_col, "mean"), acc=("behav_correct", "mean")
                ).dropna()
                if len(smeans) < 5:
                    continue
                r, p = pearsonr(smeans["replay"], smeans["acc"])
                s = sig_stars(p)
                log.append(f"    {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
                rows_session.append(dict(
                    metric=name, roi=roi, session=session,
                    r=r, p_value=p, n=len(smeans),
                ))

    tbl_session = pd.DataFrame(rows_session)
    tbl_session.to_csv(opj(RESULTS_DIR, "brain_behavior_per_session.csv"), index=False)

    # Save log
    log_path = opj(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\n".join(log))


if __name__ == "__main__":
    main()
