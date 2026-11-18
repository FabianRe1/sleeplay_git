# -*- coding: utf-8 -*-
"""LME analysis: object vs scene retrieval and position effects on replay.

Usage: python 5.3_lme_object_scene_position.py
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_rel, pearsonr

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = opj(SCRIPT_DIR, "5.3_lme_object_scene_position")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "Absolute SODA Slope"),
    ("retrieved_w_amplitude", "control_w_amplitude", "Sinusoidal Amplitude"),
    ("retrieved_band_power", "control_band_power", "FFT Band Power"),
]

ROIS = ["vis", "mtl", "parietal"]


def fit_lme(df, formula, group_var="subject"):
    df_clean = df.dropna(subset=[group_var]).copy()
    dep_var = formula.split("~")[0].strip()
    if dep_var in df_clean.columns:
        df_clean = df_clean.dropna(subset=[dep_var])
    model = smf.mixedlm(formula, df_clean, groups=df_clean[group_var])
    return model.fit(reml=True, method="powell")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(MASTER_CSV)
    df["replay_pos"] = df["retrieved_position"].astype(int)
    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")
    print(f"  Objects: {(df['object_scene_retrieval']=='object').sum()}, Scenes: {(df['object_scene_retrieval']=='scene').sum()}")

    log = []
    all_tables = []

    # ==================================================================
    # ANALYSIS A: Object vs Scene replay effect
    # ==================================================================
    log.append("=" * 70)
    log.append("ANALYSIS A: Replay by trial type (object vs scene)")
    log.append("  Per ROI x session x trial_type: is retrieved > control?")
    log.append("=" * 70)

    rows_a = []
    for ret_col, ctrl_col, metric_name in METRICS:
        log.append(f"\n--- {metric_name} ---")
        for roi in ROIS:
            for session in ["presleep", "postsleep"]:
                for ttype in ["object", "scene"]:
                    sub = df[(df["roi"] == roi) & (df["session"] == session) &
                             (df["object_scene_retrieval"] == ttype)].copy()
                    sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
                    smeans = sub.groupby("subject")["replay_diff"].mean().dropna()
                    if len(smeans) < 3:
                        continue
                    t, p = ttest_1samp(smeans, 0)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                    log.append(f"  {roi:10s} {session:10s} {ttype:7s}: mean={smeans.mean():+.5f}, t={t:+.3f}, p={p:.4f} {sig}  (N={len(smeans)})")
                    rows_a.append(dict(metric=metric_name, roi=roi, session=session,
                                       trial_type=ttype, mean_diff=smeans.mean(),
                                       t_stat=t, p_value=p, n=len(smeans)))
    tbl_a = pd.DataFrame(rows_a)

    # ==================================================================
    # ANALYSIS B: LME with trial_type as predictor
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("ANALYSIS B: LME — trial_type effect on replay")
    log.append("  replay_diff ~ trial_type * session + (1|subject), per ROI")
    log.append("=" * 70)

    rows_b = []
    for ret_col, ctrl_col, metric_name in METRICS:
        log.append(f"\n--- {metric_name} ---")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["is_object"] = (sub["object_scene_retrieval"] == "object").astype(int)
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            try:
                result = fit_lme(sub, "replay_diff ~ is_object * is_post", "subject")
                log.append(f"  {roi}:")
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows_b.append(dict(metric=metric_name, roi=roi, parameter=param,
                                        coef=result.params[param], se=result.bse[param],
                                        z=result.tvalues[param], p=result.pvalues[param]))
            except Exception as e:
                log.append(f"  {roi}: FAILED - {e}")
    tbl_b = pd.DataFrame(rows_b)

    # ==================================================================
    # ANALYSIS C: Position effect on replay
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("ANALYSIS C: Replay by retrieved position (1-5)")
    log.append("  Per ROI x session: subject-level mean replay_diff at each position")
    log.append("=" * 70)

    rows_c = []
    for ret_col, ctrl_col, metric_name in METRICS:
        log.append(f"\n--- {metric_name} ---")
        for roi in ROIS:
            for session in ["presleep", "postsleep"]:
                log_line = f"  {roi:10s} {session:10s}:"
                pos_means = []
                for pos in [1, 2, 3, 4, 5]:
                    sub = df[(df["roi"] == roi) & (df["session"] == session) &
                             (df["replay_pos"] == pos)].copy()
                    sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
                    smeans = sub.groupby("subject")["replay_diff"].mean().dropna()
                    m = smeans.mean()
                    pos_means.append(m)
                    rows_c.append(dict(metric=metric_name, roi=roi, session=session,
                                        position=pos, mean_diff=m, n=len(smeans)))
                log_line += "  ".join([f"P{i+1}={v:+.5f}" for i, v in enumerate(pos_means)])
                log.append(log_line)

    tbl_c = pd.DataFrame(rows_c)

    # ==================================================================
    # ANALYSIS D: LME with position as continuous predictor
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("ANALYSIS D: LME — position effect on replay")
    log.append("  replay_diff ~ position * session + (1|subject), per ROI")
    log.append("  (object trials only, since scene positions may differ)")
    log.append("=" * 70)

    rows_d = []
    for ret_col, ctrl_col, metric_name in METRICS:
        log.append(f"\n--- {metric_name} ---")
        for roi in ROIS:
            sub = df[(df["roi"] == roi) & (df["object_scene_retrieval"] == "object")].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["pos_centered"] = sub["replay_pos"] - 3  # center at position 3
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            try:
                result = fit_lme(sub, "replay_diff ~ pos_centered * is_post", "subject")
                log.append(f"  {roi}:")
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows_d.append(dict(metric=metric_name, roi=roi, parameter=param,
                                        coef=result.params[param], se=result.bse[param],
                                        z=result.tvalues[param], p=result.pvalues[param]))
            except Exception as e:
                log.append(f"  {roi}: FAILED - {e}")
    tbl_d = pd.DataFrame(rows_d)

    # ==================================================================
    # ANALYSIS E: Position effect — does replay track behavioral primacy?
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("ANALYSIS E: Does raw replay strength correlate with position-dependent accuracy?")
    log.append("  Per ROI x session: correlation of position-level raw retrieved vs accuracy")
    log.append("=" * 70)

    rows_e = []
    for ret_col, ctrl_col, metric_name in METRICS:
        log.append(f"\n--- {metric_name} ---")
        for roi in ROIS:
            for session in ["presleep", "postsleep"]:
                sub = df[(df["roi"] == roi) & (df["session"] == session) &
                         (df["object_scene_retrieval"] == "object")].copy()
                sub["behav_correct"] = pd.to_numeric(sub["behav_correct"], errors="coerce")

                # Subject-level: correlation between position-specific raw retrieved and accuracy
                smeans = sub.groupby(["subject", "replay_pos"]).agg(
                    replay=(ret_col, "mean"),
                    accuracy=("behav_correct", "mean"),
                ).reset_index()

                # Grand mean across subjects per position
                pos_agg = smeans.groupby("replay_pos")[["replay", "accuracy"]].mean()
                if len(pos_agg) >= 4:
                    r, p = pearsonr(pos_agg["replay"], pos_agg["accuracy"])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                    log.append(f"  {roi:10s} {session:10s}: r={r:+.3f}, p={p:.4f} {sig}  (across 5 positions)")
                    rows_e.append(dict(metric=metric_name, roi=roi, session=session,
                                        r=r, p_value=p))
    tbl_e = pd.DataFrame(rows_e)

    # ==================================================================
    # Save
    # ==================================================================
    log_path = opj(RESULTS_DIR, "lme_object_scene_position.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))
    print(f"\nResults saved: {log_path}")

    for tbl, name in [(tbl_a, "replay_by_trialtype"), (tbl_b, "lme_trialtype"),
                       (tbl_c, "replay_by_position"), (tbl_d, "lme_position"),
                       (tbl_e, "position_accuracy_correlation")]:
        if len(tbl) > 0:
            path = opj(RESULTS_DIR, f"{name}.csv")
            tbl.to_csv(path, index=False)
            print(f"  Saved: {path}")

    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
