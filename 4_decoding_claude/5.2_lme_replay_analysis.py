# -*- coding: utf-8 -*-
"""LME analysis of replay metrics: pre-sleep vs post-sleep comparison.

Uses the master trial-level dataframe to fit linear mixed-effects models
with subject as random intercept. Tests effects of session, ROI, and their
interaction on replay strength (absolute SODA slope, sinusoidal amplitude,
FFT band power).

Usage
-----
    python 5.2_lme_replay_analysis.py

Output
------
    Hummel_local/results/lme_results.txt       — summary of all models
    Hummel_local/results/lme_results.csv        — table of key statistics
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = opj(SCRIPT_DIR, "5.2_lme_replay_analysis")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

# Metrics to analyze: (column_retrieved, column_control, display_name)
METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "Absolute SODA Slope"),
    ("retrieved_w_amplitude", "control_w_amplitude", "Sinusoidal Amplitude"),
    ("retrieved_band_power", "control_band_power", "FFT Band Power"),
]


# =====================================================================
# HELPERS
# =====================================================================

def compute_replay_difference(df, ret_col, ctrl_col, out_col="replay_diff"):
    """Compute retrieved - control difference as the replay metric."""
    df = df.copy()
    df[out_col] = df[ret_col] - df[ctrl_col]
    return df


def fit_lme(df, formula, group_var="subject"):
    """Fit a linear mixed-effects model with subject random intercept."""
    # Drop rows with NaN in any column used by the formula
    df_clean = df.dropna(subset=[group_var]).copy()
    # Extract dependent variable from formula
    dep_var = formula.split("~")[0].strip()
    if dep_var in df_clean.columns:
        df_clean = df_clean.dropna(subset=[dep_var])
    model = smf.mixedlm(formula, df_clean, groups=df_clean[group_var])
    result = model.fit(reml=True, method="powell")
    return result


def extract_fixed_effects(result):
    """Extract fixed effects table from LME result."""
    fe = result.summary().tables[1]
    return fe


def subject_means(df, metric_col, groupby_cols):
    """Compute subject-level means for plotting/reporting."""
    return df.groupby(groupby_cols)[metric_col].mean().reset_index()


# =====================================================================
# ANALYSIS 1: Per-ROI, per-session replay tests
# =====================================================================

def analysis_per_roi_session(df, results_log):
    """For each ROI x session: is replay_diff > 0? (retrieved > control)"""
    results_log.append("\n" + "=" * 70)
    results_log.append("ANALYSIS 1: Replay effect per ROI x Session")
    results_log.append("  Test: Is retrieved > control for each combination?")
    results_log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, metric_name in METRICS:
        results_log.append(f"\n--- {metric_name} ---")
        for roi in sorted(df["roi"].unique()):
            for session in ["presleep", "postsleep"]:
                sub = df[(df["roi"] == roi) & (df["session"] == session)].copy()
                sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]

                # Subject-level means
                smeans = sub.groupby("subject")["replay_diff"].mean().dropna()
                n = len(smeans)
                mean_diff = smeans.mean()

                # One-sample t-test against 0
                from scipy.stats import ttest_1samp
                t, p = ttest_1samp(smeans, 0)

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                results_log.append(
                    f"  {roi:12s} {session:10s}: mean={mean_diff:+.5f}, t={t:+.3f}, p={p:.4f} {sig}  (N={n})"
                )
                rows.append({
                    "analysis": "per_roi_session",
                    "metric": metric_name,
                    "roi": roi,
                    "session": session,
                    "n_subjects": n,
                    "mean_diff": mean_diff,
                    "t_stat": t,
                    "p_value": p,
                })

    return pd.DataFrame(rows)


# =====================================================================
# ANALYSIS 2: Session effect (pre vs post) per ROI
# =====================================================================

def analysis_session_effect(df, results_log):
    """Per ROI: does replay_diff change from presleep to postsleep?"""
    results_log.append("\n" + "=" * 70)
    results_log.append("ANALYSIS 2: Session effect (pre vs post) per ROI")
    results_log.append("  LME: replay_diff ~ session + (1|subject)")
    results_log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, metric_name in METRICS:
        results_log.append(f"\n--- {metric_name} ---")
        for roi in sorted(df["roi"].unique()):
            sub = df[df["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["session_code"] = (sub["session"] == "postsleep").astype(int)

            try:
                result = fit_lme(sub, "replay_diff ~ session_code", "subject")
                coef = result.params.get("session_code", np.nan)
                se = result.bse.get("session_code", np.nan)
                z = result.tvalues.get("session_code", np.nan)
                p = result.pvalues.get("session_code", np.nan)
                n = sub["subject"].nunique()

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                results_log.append(
                    f"  {roi:12s}: coef={coef:+.5f}, SE={se:.5f}, z={z:+.3f}, p={p:.4f} {sig}  (N={n})"
                )
                rows.append({
                    "analysis": "session_effect",
                    "metric": metric_name,
                    "roi": roi,
                    "coef_session": coef,
                    "se": se,
                    "z_stat": z,
                    "p_value": p,
                    "n_subjects": n,
                })
            except Exception as e:
                results_log.append(f"  {roi:12s}: FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# ANALYSIS 3: ROI x Session interaction
# =====================================================================

def analysis_roi_session_interaction(df, results_log):
    """Full model: replay_diff ~ session * roi + (1|subject)"""
    results_log.append("\n" + "=" * 70)
    results_log.append("ANALYSIS 3: Session x ROI interaction")
    results_log.append("  LME: replay_diff ~ session * roi + (1|subject)")
    results_log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, metric_name in METRICS:
        results_log.append(f"\n--- {metric_name} ---")
        sub = df.copy()
        sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
        sub["session_code"] = (sub["session"] == "postsleep").astype(int)

        # Use C() for categorical ROI
        try:
            result = fit_lme(sub, "replay_diff ~ session_code * C(roi)", "subject")
            results_log.append(str(result.summary().tables[1]))

            for param in result.params.index:
                rows.append({
                    "analysis": "roi_session_interaction",
                    "metric": metric_name,
                    "parameter": param,
                    "coef": result.params[param],
                    "se": result.bse[param],
                    "z_stat": result.tvalues[param],
                    "p_value": result.pvalues[param],
                    "n_subjects": sub["subject"].nunique(),
                })
        except Exception as e:
            results_log.append(f"  FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# ANALYSIS 4: Replay predicts behavior (trial-level)
# =====================================================================

def analysis_replay_behavior(df, results_log):
    """Does raw replay strength predict behavioral accuracy?
    Subject-level Pearson r(mean_retrieved_metric, mean_accuracy), per ROI x session."""
    results_log.append("\n" + "=" * 70)
    results_log.append("ANALYSIS 4: Brain-behavior relationship (raw retrieved metric)")
    results_log.append("  Subject-level Pearson r(mean_retrieved, mean_accuracy), per ROI x session")
    results_log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, metric_name in METRICS:
        results_log.append(f"\n--- {metric_name} ---")
        for roi in sorted(df["roi"].unique()):
            for session in ["presleep", "postsleep"]:
                sub = df[(df["roi"] == roi) & (df["session"] == session)].copy()
                sub["behav_correct"] = pd.to_numeric(sub["behav_correct"], errors="coerce")
                sub = sub.dropna(subset=[ret_col, "behav_correct"])

                if len(sub) < 10:
                    continue

                # Subject-level correlation: mean raw retrieved vs mean accuracy
                smeans = sub.groupby("subject").agg(
                    replay=(ret_col, "mean"),
                    accuracy=("behav_correct", "mean"),
                ).dropna()

                if len(smeans) < 5:
                    continue

                from scipy.stats import pearsonr
                r, p = pearsonr(smeans["replay"], smeans["accuracy"])
                n = len(smeans)

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                results_log.append(
                    f"  {roi:12s} {session:10s}: r={r:+.3f}, p={p:.4f} {sig}  (N={n})"
                )
                rows.append({
                    "analysis": "brain_behavior",
                    "metric": metric_name,
                    "roi": roi,
                    "session": session,
                    "r": r,
                    "p_value": p,
                    "n_subjects": n,
                })

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"Loading master dataframe from {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    print(f"  Shape: {df.shape}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  Sessions: {df['session'].value_counts().to_dict()}")
    print(f"  ROIs: {df['roi'].value_counts().to_dict()}")

    results_log = []
    results_log.append("=" * 70)
    results_log.append("REPLAY METRICS LME ANALYSIS — Pre vs Post Sleep Comparison")
    results_log.append(f"  N subjects: {df['subject'].nunique()}")
    results_log.append(f"  ROIs: {sorted(df['roi'].unique())}")
    results_log.append(f"  Sessions: {sorted(df['session'].unique())}")
    results_log.append(f"  Total trials: {len(df)}")
    results_log.append("=" * 70)

    all_tables = []

    # Run analyses
    print("\nRunning Analysis 1: Per ROI x Session replay tests...")
    t1 = analysis_per_roi_session(df, results_log)
    all_tables.append(t1)

    print("Running Analysis 2: Session effect per ROI...")
    t2 = analysis_session_effect(df, results_log)
    all_tables.append(t2)

    print("Running Analysis 3: Session x ROI interaction...")
    t3 = analysis_roi_session_interaction(df, results_log)
    all_tables.append(t3)

    print("Running Analysis 4: Brain-behavior correlations...")
    t4 = analysis_replay_behavior(df, results_log)
    all_tables.append(t4)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Text log
    log_path = opj(RESULTS_DIR, "lme_results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(results_log))
    print(f"\nResults log saved: {log_path}")

    # CSV tables
    for i, (tbl, name) in enumerate(zip(all_tables, [
        "per_roi_session", "session_effect", "roi_session_interaction",
        "brain_behavior",
    ])):
        if tbl is not None and len(tbl) > 0:
            csv_path = opj(RESULTS_DIR, f"lme_{name}.csv")
            tbl.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

    # Print full log
    print("\n" + "\n".join(results_log))


if __name__ == "__main__":
    main()
