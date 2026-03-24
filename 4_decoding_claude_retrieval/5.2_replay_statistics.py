# -*- coding: utf-8 -*-
"""Consolidated replay statistics: all analyses in one script.

Analyses
--------
A) Replay effect: is retrieved > control? (collapsed + per session + per ROI)
B) Session comparison: is replay bigger postsleep? (paired t-test + LME)
C) ROI x Session interaction (LME)
D) Object vs Scene replay (t-tests + LME)
E) Position effects on replay (descriptive + LME)
F) Brain-behavior correlations (raw retrieved + ret-ctrl diff vs accuracy)

Usage
-----
    python 5.2_replay_statistics.py

Output
------
    5.2_replay_statistics/
        results.txt              — full text log
        A_replay_effect.csv
        B_session_comparison.csv
        C_roi_session_interaction.csv
        D_object_scene.csv
        D_lme_trialtype.csv
        E_position_descriptive.csv
        E_lme_position.csv
        F_brain_behavior.csv
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
RESULTS_DIR = opj(SCRIPT_DIR, "5.2_replay_statistics")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

# (retrieved_col, control_col, display_name)
METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "Abs SODA"),
    ("retrieved_w_amplitude", "control_w_amplitude", "Sinus Amp"),
    ("retrieved_band_power", "control_band_power", "FFT Power"),
]
ROIS = ["vis", "mtl", "parietal"]


# =====================================================================
# HELPERS
# =====================================================================

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


def fit_lme(df, formula, group_var="subject"):
    """Fit LME with subject random intercept (Powell optimizer)."""
    df_clean = df.dropna(subset=[group_var]).copy()
    dep_var = formula.split("~")[0].strip()
    if dep_var in df_clean.columns:
        df_clean = df_clean.dropna(subset=[dep_var])
    model = smf.mixedlm(formula, df_clean, groups=df_clean[group_var])
    return model.fit(reml=True, method="powell")


# =====================================================================
# A) REPLAY EFFECT: is retrieved > control?
# =====================================================================

def analysis_A(df, log):
    log.append("\n" + "=" * 70)
    log.append("A) REPLAY EFFECT: is retrieved > control?")
    log.append("   One-sample t-test on subject-level mean(ret - ctrl) vs 0")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # A1: Collapsed across sessions
        log.append("\n  Collapsed across sessions:")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["diff"] = sub[ret_col] - sub[ctrl_col]
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            s = sig_stars(p)
            log.append(f"    {roi:10s}: mean={smeans.mean():+.6f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
            rows.append(dict(metric=name, roi=roi, session="collapsed",
                             mean_diff=smeans.mean(), t_stat=t, p_value=p, n=len(smeans)))

        # A2: Per session
        for session in ["presleep", "postsleep"]:
            log.append(f"\n  {session}:")
            for roi in ROIS:
                sub = df[(df["roi"] == roi) & (df["session"] == session)].copy()
                sub["diff"] = sub[ret_col] - sub[ctrl_col]
                smeans = sub.groupby("subject")["diff"].mean().dropna()
                t, p = ttest_1samp(smeans, 0)
                s = sig_stars(p)
                log.append(f"    {roi:10s}: mean={smeans.mean():+.6f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
                rows.append(dict(metric=name, roi=roi, session=session,
                                 mean_diff=smeans.mean(), t_stat=t, p_value=p, n=len(smeans)))

    return pd.DataFrame(rows)


# =====================================================================
# B) SESSION COMPARISON: is replay bigger postsleep?
# =====================================================================

def analysis_B(df, log):
    log.append("\n" + "=" * 70)
    log.append("B) SESSION COMPARISON: is replay (ret - ctrl) bigger postsleep?")
    log.append("   B1: Paired t-test on subject means (post vs pre)")
    log.append("   B2: LME: replay_diff ~ is_post + (1|subject)")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # B1: Paired t-test
        log.append("\n  Paired t-test (post - pre):")
        for roi in ROIS:
            pre = df[(df["roi"] == roi) & (df["session"] == "presleep")].copy()
            post = df[(df["roi"] == roi) & (df["session"] == "postsleep")].copy()
            pre["diff"] = pre[ret_col] - pre[ctrl_col]
            post["diff"] = post[ret_col] - post[ctrl_col]
            pre_m = pre.groupby("subject")["diff"].mean()
            post_m = post.groupby("subject")["diff"].mean()
            common = pre_m.index.intersection(post_m.index)
            if len(common) < 3:
                continue
            pre_v = pre_m.loc[common]
            post_v = post_m.loc[common]
            change = post_v - pre_v
            t, p = ttest_rel(post_v, pre_v)
            s = sig_stars(p)
            log.append(f"    {roi:10s}: pre={pre_v.mean():+.6f}, post={post_v.mean():+.6f}, "
                        f"diff={change.mean():+.6f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(common)})")
            rows.append(dict(metric=name, roi=roi, test="paired_ttest",
                             pre_mean=pre_v.mean(), post_mean=post_v.mean(),
                             diff=change.mean(), stat=t, p_value=p, n=len(common)))

        # B2: LME
        log.append("\n  LME: replay_diff ~ is_post + (1|subject):")

        # All ROIs combined
        sub = df.copy()
        sub["diff"] = sub[ret_col] - sub[ctrl_col]
        sub["is_post"] = (sub["session"] == "postsleep").astype(int)
        sub = sub.dropna(subset=["diff"])
        try:
            result = fit_lme(sub, "diff ~ is_post")
            coef = result.params["is_post"]
            p = result.pvalues["is_post"]
            z = result.tvalues["is_post"]
            s = sig_stars(p)
            log.append(f"    All ROIs : coef={coef:+.6f}, z={z:+.3f}, p={p:.4f} {s}")
            rows.append(dict(metric=name, roi="all", test="lme",
                             pre_mean=np.nan, post_mean=np.nan,
                             diff=coef, stat=z, p_value=p,
                             n=sub["subject"].nunique()))
        except Exception as e:
            log.append(f"    All ROIs : FAILED - {e}")

        # Per ROI
        for roi in ROIS:
            rsub = sub[sub["roi"] == roi].copy()
            try:
                result = fit_lme(rsub, "diff ~ is_post")
                coef = result.params["is_post"]
                p = result.pvalues["is_post"]
                z = result.tvalues["is_post"]
                s = sig_stars(p)
                log.append(f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, p={p:.4f} {s}")
                rows.append(dict(metric=name, roi=roi, test="lme",
                                 pre_mean=np.nan, post_mean=np.nan,
                                 diff=coef, stat=z, p_value=p,
                                 n=rsub["subject"].nunique()))
            except Exception as e:
                log.append(f"    {roi:10s}: FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# C) ROI x SESSION INTERACTION
# =====================================================================

def analysis_C(df, log):
    log.append("\n" + "=" * 70)
    log.append("C) ROI x SESSION INTERACTION")
    log.append("   LME: replay_diff ~ is_post * C(roi) + (1|subject)")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")
        sub = df.copy()
        sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
        sub["is_post"] = (sub["session"] == "postsleep").astype(int)
        try:
            result = fit_lme(sub, "replay_diff ~ is_post * C(roi)")
            log.append(str(result.summary().tables[1]))
            for param in result.params.index:
                rows.append(dict(metric=name, parameter=param,
                                 coef=result.params[param], se=result.bse[param],
                                 z=result.tvalues[param], p_value=result.pvalues[param]))
        except Exception as e:
            log.append(f"  FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# D) OBJECT vs SCENE
# =====================================================================

def analysis_D(df, log):
    log.append("\n" + "=" * 70)
    log.append("D) OBJECT vs SCENE REPLAY")
    log.append("   D1: Per trial_type x ROI x session: is ret > ctrl?")
    log.append("   D2: LME: replay_diff ~ is_object * is_post + (1|subject)")
    log.append("=" * 70)

    rows_d1 = []
    rows_d2 = []

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # D1: t-tests per cell
        log.append("\n  D1: One-sample t-tests (ret - ctrl vs 0):")
        for roi in ROIS:
            for session in ["presleep", "postsleep"]:
                for ttype in ["object", "scene"]:
                    sub = df[(df["roi"] == roi) & (df["session"] == session) &
                             (df["object_scene_retrieval"] == ttype)].copy()
                    sub["diff"] = sub[ret_col] - sub[ctrl_col]
                    smeans = sub.groupby("subject")["diff"].mean().dropna()
                    if len(smeans) < 3:
                        continue
                    t, p = ttest_1samp(smeans, 0)
                    s = sig_stars(p)
                    log.append(f"    {roi:10s} {session:10s} {ttype:7s}: "
                               f"mean={smeans.mean():+.5f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
                    rows_d1.append(dict(metric=name, roi=roi, session=session,
                                        trial_type=ttype, mean_diff=smeans.mean(),
                                        t_stat=t, p_value=p, n=len(smeans)))

        # D2: LME per ROI
        log.append("\n  D2: LME — replay_diff ~ is_object * is_post + (1|subject):")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["is_object"] = (sub["object_scene_retrieval"] == "object").astype(int)
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            try:
                result = fit_lme(sub, "replay_diff ~ is_object * is_post")
                log.append(f"    {roi}:")
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows_d2.append(dict(metric=name, roi=roi, parameter=param,
                                        coef=result.params[param], se=result.bse[param],
                                        z=result.tvalues[param], p_value=result.pvalues[param]))
            except Exception as e:
                log.append(f"    {roi}: FAILED - {e}")

    return pd.DataFrame(rows_d1), pd.DataFrame(rows_d2)


# =====================================================================
# E) POSITION EFFECTS
# =====================================================================

def analysis_E(df, log):
    log.append("\n" + "=" * 70)
    log.append("E) POSITION EFFECTS ON REPLAY")
    log.append("   E1: Descriptive — mean replay_diff per position (1-5)")
    log.append("   E2: LME: replay_diff ~ pos_centered * is_post + (1|subject)")
    log.append("   (object trials only)")
    log.append("=" * 70)

    df["replay_pos"] = df["retrieved_position"].astype(int)
    rows_e1 = []
    rows_e2 = []

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # E1: Descriptive
        log.append("\n  E1: Mean replay_diff by position:")
        for roi in ROIS:
            for session in ["presleep", "postsleep"]:
                log_line = f"    {roi:10s} {session:10s}:  "
                pos_vals = []
                for pos in [1, 2, 3, 4, 5]:
                    sub = df[(df["roi"] == roi) & (df["session"] == session) &
                             (df["replay_pos"] == pos)].copy()
                    sub["diff"] = sub[ret_col] - sub[ctrl_col]
                    smeans = sub.groupby("subject")["diff"].mean().dropna()
                    m = smeans.mean()
                    pos_vals.append(m)
                    rows_e1.append(dict(metric=name, roi=roi, session=session,
                                        position=pos, mean_diff=m, n=len(smeans)))
                log_line += "  ".join([f"P{i+1}={v:+.5f}" for i, v in enumerate(pos_vals)])
                log.append(log_line)

        # E2: LME (object trials only)
        log.append("\n  E2: LME — replay_diff ~ pos_centered * is_post + (1|subject):")
        for roi in ROIS:
            sub = df[(df["roi"] == roi) & (df["object_scene_retrieval"] == "object")].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["pos_centered"] = sub["replay_pos"] - 3
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            try:
                result = fit_lme(sub, "replay_diff ~ pos_centered * is_post")
                log.append(f"    {roi}:")
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows_e2.append(dict(metric=name, roi=roi, parameter=param,
                                        coef=result.params[param], se=result.bse[param],
                                        z=result.tvalues[param], p_value=result.pvalues[param]))
            except Exception as e:
                log.append(f"    {roi}: FAILED - {e}")

    return pd.DataFrame(rows_e1), pd.DataFrame(rows_e2)


# =====================================================================
# F) BRAIN-BEHAVIOR CORRELATIONS
# =====================================================================

def analysis_F(df, log):
    log.append("\n" + "=" * 70)
    log.append("F) BRAIN-BEHAVIOR CORRELATIONS")
    log.append("   F1: Raw retrieved metric vs accuracy")
    log.append("   F2: Replay diff (ret - ctrl) vs accuracy")
    log.append("   Subject-level Pearson r, per ROI x session + collapsed")
    log.append("=" * 70)

    df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")
    rows = []

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        for variant, variant_label in [("raw", "F1: Raw retrieved"), ("diff", "F2: Ret - Ctrl")]:
            log.append(f"\n  {variant_label}:")

            # Collapsed across sessions
            log.append("    Collapsed:")
            for roi in ROIS:
                sub = df[df["roi"] == roi].copy()
                if variant == "raw":
                    sub = sub.dropna(subset=[ret_col, "behav_correct"])
                    smeans = sub.groupby("subject").agg(
                        replay=(ret_col, "mean"), acc=("behav_correct", "mean")
                    ).dropna()
                else:
                    sub["diff"] = sub[ret_col] - sub[ctrl_col]
                    sub = sub.dropna(subset=["diff", "behav_correct"])
                    smeans = sub.groupby("subject").agg(
                        replay=("diff", "mean"), acc=("behav_correct", "mean")
                    ).dropna()

                if len(smeans) < 5:
                    continue
                r, p = pearsonr(smeans["replay"], smeans["acc"])
                s = sig_stars(p)
                log.append(f"      {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
                rows.append(dict(metric=name, variant=variant_label, roi=roi,
                                 session="collapsed", r=r, p_value=p, n=len(smeans)))

            # Per session
            for session in ["presleep", "postsleep"]:
                log.append(f"    {session}:")
                for roi in ROIS:
                    sub = df[(df["roi"] == roi) & (df["session"] == session)].copy()
                    if variant == "raw":
                        sub = sub.dropna(subset=[ret_col, "behav_correct"])
                        smeans = sub.groupby("subject").agg(
                            replay=(ret_col, "mean"), acc=("behav_correct", "mean")
                        ).dropna()
                    else:
                        sub["diff"] = sub[ret_col] - sub[ctrl_col]
                        sub = sub.dropna(subset=["diff", "behav_correct"])
                        smeans = sub.groupby("subject").agg(
                            replay=("diff", "mean"), acc=("behav_correct", "mean")
                        ).dropna()

                    if len(smeans) < 5:
                        continue
                    r, p = pearsonr(smeans["replay"], smeans["acc"])
                    s = sig_stars(p)
                    log.append(f"      {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
                    rows.append(dict(metric=name, variant=variant_label, roi=roi,
                                     session=session, r=r, p_value=p, n=len(smeans)))

    return pd.DataFrame(rows)


# =====================================================================
# G) REPLAY ~ IS_RETRIEVED * IS_POST  (long-format LME)
# =====================================================================

def analysis_G(df, log):
    """Predict replay metric from is_retrieved * is_post + (1|subject).

    Reshapes data to long format so each row has one replay value and a
    binary indicator for whether it comes from the retrieved or control
    condition.  Fits per ROI and collapsed across ROIs.
    """
    log.append("\n" + "=" * 70)
    log.append("G) LME ON REPLAY METRIC: is_retrieved * is_post")
    log.append("   Model: replay_metric ~ is_retrieved * is_post + (1|subject)")
    log.append("   Data reshaped to long format (retrieved & control stacked)")
    log.append("   G1: All trials   G2: Object trials only")
    log.append("=" * 70)

    rows = []

    for trial_filter, filter_label in [("all", "G1: All trials"),
                                        ("object", "G2: Object trials only")]:
        log.append(f"\n{'='*40}")
        log.append(f"  {filter_label}")
        log.append(f"{'='*40}")

        for ret_col, ctrl_col, name in METRICS:
            log.append(f"\n--- {name} ({filter_label}) ---")

            # Build long-format dataframe: stack retrieved and control
            keep_cols = ["subject", "session", "roi", "object_scene_retrieval",
                         ret_col, ctrl_col]
            sub = df[[c for c in keep_cols if c in df.columns]].dropna(
                subset=[ret_col, ctrl_col]).copy()

            if trial_filter == "object":
                sub = sub[sub["object_scene_retrieval"] == "object"].copy()

            df_ret = sub[["subject", "session", "roi", ret_col]].copy()
            df_ret = df_ret.rename(columns={ret_col: "replay_metric"})
            df_ret["is_retrieved"] = 1

            df_ctrl = sub[["subject", "session", "roi", ctrl_col]].copy()
            df_ctrl = df_ctrl.rename(columns={ctrl_col: "replay_metric"})
            df_ctrl["is_retrieved"] = 0

            df_long = pd.concat([df_ret, df_ctrl], ignore_index=True)
            df_long["is_post"] = (df_long["session"] == "postsleep").astype(int)

            # --- All ROIs combined ---
            log.append("\n  All ROIs combined:")
            try:
                result = fit_lme(df_long, "replay_metric ~ is_retrieved * is_post")
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows.append(dict(metric=name, roi="all", trial_filter=trial_filter,
                                     parameter=param,
                                     coef=result.params[param], se=result.bse[param],
                                     z=result.tvalues[param], p_value=result.pvalues[param],
                                     n_subjects=df_long["subject"].nunique()))
            except Exception as e:
                log.append(f"    FAILED - {e}")

            # --- Per ROI ---
            for roi in ROIS:
                log.append(f"\n  {roi}:")
                rsub = df_long[df_long["roi"] == roi].copy()
                try:
                    result = fit_lme(rsub, "replay_metric ~ is_retrieved * is_post")
                    log.append(str(result.summary().tables[1]))
                    for param in result.params.index:
                        rows.append(dict(metric=name, roi=roi, trial_filter=trial_filter,
                                         parameter=param,
                                         coef=result.params[param], se=result.bse[param],
                                         z=result.tvalues[param], p_value=result.pvalues[param],
                                         n_subjects=rsub["subject"].nunique()))
                except Exception as e:
                    log.append(f"    FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# H) REPLAY ~ IS_RETRIEVED * IS_OBJECT  (scene vs object LME)
# =====================================================================

def analysis_H(df, log):
    """Predict replay metric from is_retrieved * is_object + (1|subject).

    Tests whether the replay effect (retrieved > control) differs between
    object and scene trials.  Reshapes data to long format.
    Fits per ROI and collapsed across ROIs.
    """
    log.append("\n" + "=" * 70)
    log.append("H) LME ON REPLAY METRIC: is_retrieved * is_object")
    log.append("   Model: replay_metric ~ is_retrieved * is_object + (1|subject)")
    log.append("   Data reshaped to long format (retrieved & control stacked)")
    log.append("=" * 70)

    rows = []

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # Build long-format dataframe: stack retrieved and control
        keep_cols = ["subject", "session", "roi", "object_scene_retrieval",
                     ret_col, ctrl_col]
        sub = df[[c for c in keep_cols if c in df.columns]].dropna(
            subset=[ret_col, ctrl_col]).copy()

        # Drop trials without object/scene label
        sub = sub.dropna(subset=["object_scene_retrieval"]).copy()
        sub = sub[sub["object_scene_retrieval"].isin(["object", "scene"])].copy()

        df_ret = sub[["subject", "session", "roi", "object_scene_retrieval", ret_col]].copy()
        df_ret = df_ret.rename(columns={ret_col: "replay_metric"})
        df_ret["is_retrieved"] = 1

        df_ctrl = sub[["subject", "session", "roi", "object_scene_retrieval", ctrl_col]].copy()
        df_ctrl = df_ctrl.rename(columns={ctrl_col: "replay_metric"})
        df_ctrl["is_retrieved"] = 0

        df_long = pd.concat([df_ret, df_ctrl], ignore_index=True)
        df_long["is_object"] = (df_long["object_scene_retrieval"] == "object").astype(int)

        formula = "replay_metric ~ is_retrieved * is_object"

        for roi in ROIS:
            log.append(f"\n  {roi}:")
            rsub = df_long[df_long["roi"] == roi].copy()
            try:
                result = fit_lme(rsub, formula)
                log.append(str(result.summary().tables[1]))
                for param in result.params.index:
                    rows.append(dict(metric=name, roi=roi, parameter=param,
                                     coef=result.params[param], se=result.bse[param],
                                     z=result.tvalues[param], p_value=result.pvalues[param],
                                     n_subjects=rsub["subject"].nunique()))
            except Exception as e:
                log.append(f"    FAILED - {e}")

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")
    print(f"  Sessions: {df['session'].value_counts().to_dict()}")
    print(f"  ROIs: {df['roi'].value_counts().to_dict()}")
    print(f"  Objects: {(df['object_scene_retrieval']=='object').sum()}, "
          f"Scenes: {(df['object_scene_retrieval']=='scene').sum()}")

    log = []
    log.append("=" * 70)
    log.append("REPLAY STATISTICS — Consolidated Analysis")
    log.append(f"  N subjects: {df['subject'].nunique()}")
    log.append(f"  ROIs: {sorted(df['roi'].unique())}")
    log.append(f"  Sessions: {sorted(df['session'].unique())}")
    log.append(f"  Total trials: {len(df)}")
    log.append("=" * 70)

    csvs = {}

    print("\nA) Replay effect (ret > ctrl)...")
    csvs["A_replay_effect"] = analysis_A(df, log)

    print("B) Session comparison (post > pre)...")
    csvs["B_session_comparison"] = analysis_B(df, log)

    print("C) ROI x Session interaction...")
    csvs["C_roi_session_interaction"] = analysis_C(df, log)

    print("D) Object vs Scene...")
    d1, d2 = analysis_D(df, log)
    csvs["D_object_scene"] = d1
    csvs["D_lme_trialtype"] = d2

    print("E) Position effects...")
    e1, e2 = analysis_E(df, log)
    csvs["E_position_descriptive"] = e1
    csvs["E_lme_position"] = e2

    print("F) Brain-behavior correlations...")
    csvs["F_brain_behavior"] = analysis_F(df, log)

    print("G) LME: replay ~ is_retrieved * is_post...")
    csvs["G_lme_retrieved_x_session"] = analysis_G(df, log)

    print("H) LME: replay ~ is_retrieved * is_object...")
    csvs["H_lme_retrieved_x_category"] = analysis_H(df, log)

    # Save text log
    log_path = opj(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))
    print(f"\nResults log: {log_path}")

    # Save CSV tables
    for name, tbl in csvs.items():
        if tbl is not None and len(tbl) > 0:
            path = opj(RESULTS_DIR, f"{name}.csv")
            tbl.to_csv(path, index=False)
            print(f"  {path}")

    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main()
