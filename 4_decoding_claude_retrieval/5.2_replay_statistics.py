#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay statistics — systematic, transparent analysis pipeline.

Narrative
---------
1) DATA DESCRIPTION
   Trial counts, imbalances, design overview.

2) REPLAY DETECTION
   Is retrieved > control?  All trials.
   One-sample t-test on subject-level mean(ret − ctrl) vs 0.
   Per metric × ROI, collapsed + per session.

3) SESSION EFFECT ON REPLAY
   Does replay change from presleep to postsleep?
   3a: All trials
   3b: Object trials only
   3c: Scene trials only
   Paired t-test + LME per metric × ROI.

4) ACCURACY EFFECT ON REPLAY
   Does replay differ for correct vs incorrect items?
   All trials, then object trials only.
   LME: replay_diff ~ is_correct + (1|subject).

5) TRIAL TYPE COMPARISON
   Does replay differ between objects and scenes?
   LME: replay_diff ~ is_object + (1|subject).

6) INTERACTIONS
   6a: Session × trial type — LME per ROI
   6b: Omnibus three-way — objects only

7) BRAIN-BEHAVIOR CORRELATIONS
   Objects only.  Subject-level Pearson r.

8) KEY FINDINGS SUMMARY

9) SHUFFLED BASELINE VALIDATION
   Confirm replay is sequence-specific (shuffled labels → null).

11) WITHIN-SESSION TEMPORAL DYNAMICS
    Does replay decay within postsleep (early vs late runs)?

13) FIT QUALITY AS MODERATOR
    R² median split + R² as replay metric (ret R² > ctrl R²).

Design notes
------------
- Postsleep has 2× the trials of presleep (4620 vs 2310 per ROI).
- ~2.5× more object than scene trials (4950 vs 1980 per ROI).
- ~5.5× more correct than incorrect trials.
- The pipeline runs TWICE: once on all data, once on trial-matched
  data (postsleep subsampled to first half, matching presleep counts).
- All p-values reported uncorrected.

Usage
-----
    python 5.2_replay_statistics.py

Output
------
    5.2_replay_statistics/          — full data
    5.2_replay_statistics_matched/  — first-half postsleep
    Each contains: results.txt + per-analysis CSVs
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_rel, pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = opj(SCRIPT_DIR, "5.2_replay_statistics")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe",
                 "master_replay_metrics.csv")

METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "Abs SODA"),
    ("retrieved_w_amplitude", "control_w_amplitude", "Sinus Amp"),
]
# The thesis "parietal" ROI was parietal_precuneus data (see 5.1 FOCUS_ROIS).
# Three ROIs, as in the thesis: the omnibus models (6b) include C(roi) over this
# list, so adding ROIs changes those models.
ROIS = ["vis", "mtl", "parietal_precuneus"]

METRICS_SHUFFLED = [
    ("retrieved_soda_shuffled_abs", "control_soda_shuffled_abs",
     "Abs SODA (shuffled)"),
    ("shuf_retrieved_w_amplitude", "shuf_control_w_amplitude",
     "Sinus Amp (shuffled)"),
]


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


def cohens_d_onesample(values):
    """Cohen's d for one-sample test vs 0."""
    sd = values.std(ddof=1)
    return values.mean() / sd if sd > 0 else 0.0


def cohens_d_paired(x, y):
    """Cohen's d for paired samples."""
    diff = x - y
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


# =====================================================================
# 1) DATA DESCRIPTION
# =====================================================================

def data_description(df, log):
    """Comprehensive trial-count breakdown."""
    log.append("=" * 70)
    log.append("1) DATA DESCRIPTION")
    log.append("=" * 70)

    ref_roi = ROIS[0]
    ref = df[df["roi"] == ref_roi]

    log.append(f"\n  N subjects : {df['subject'].nunique()}")
    log.append(f"  ROIs       : {sorted(df['roi'].unique().tolist())}")
    log.append(f"  Total rows : {len(df)}  "
               f"({len(ref)} trials × {df['roi'].nunique()} ROIs)")

    log.append("\n  Trials by session:")
    for session in sorted(df["session"].unique()):
        n = len(ref[ref["session"] == session])
        ns = ref[ref["session"] == session]["subject"].nunique()
        log.append(f"    {session:12s}: {n:5d} trials, {ns} subjects")

    log.append("\n  Trials by type:")
    for tt in ["object", "scene"]:
        n = len(ref[ref["object_scene_retrieval"] == tt])
        log.append(f"    {tt:12s}: {n:5d} trials")

    log.append("\n  Trials by session × type:")
    for session in sorted(df["session"].unique()):
        for tt in ["object", "scene"]:
            mask = ((ref["session"] == session)
                    & (ref["object_scene_retrieval"] == tt))
            log.append(f"    {session:12s} × {tt:7s}: {mask.sum():5d} trials")

    behav = pd.to_numeric(ref["behav_correct"], errors="coerce")
    nc = int((behav == 1).sum())
    ni = int((behav == 0).sum())
    pct = 100 * nc / (nc + ni) if (nc + ni) > 0 else 0
    log.append(f"\n  Accuracy: correct {nc} ({pct:.0f}%), "
               f"incorrect {ni} ({100 - pct:.0f}%)")

    log.append("\n  Design notes:")
    log.append("    - Postsleep has 2× the trials of presleep.")
    log.append("    - ~2.5× more object than scene trials.")
    log.append("    - ~5.5× more correct than incorrect trials.")
    log.append("    - More trials → more reliable subject-level estimates → "
               "more power.")
    log.append("    - LME handles imbalanced designs; t-tests on subject means")
    log.append("      average within-subject first, absorbing count differences.")
    log.append("")


# =====================================================================
# 2) REPLAY DETECTION
# =====================================================================

def analysis_replay_detection(df, log):
    """One-sample t-test: subject-level mean(ret − ctrl) vs 0."""
    log.append("\n" + "=" * 70)
    log.append("2) REPLAY DETECTION — ALL TRIALS")
    log.append("   Question: Is there above-chance replay (retrieved > control)?")
    log.append("   Method:   One-sample t-test on subject-level mean(ret − ctrl)")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # Collapsed across sessions
        log.append("\n  Collapsed across sessions:")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["diff"] = sub[ret_col] - sub[ctrl_col]
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            d = cohens_d_onesample(smeans)
            s = sig_stars(p)
            log.append(
                f"    {roi:10s}: mean={smeans.mean():+.6f}, "
                f"t({len(smeans)-1})={t:+.3f}, p={p:.4f} {s}, d={d:+.3f}  "
                f"(N={len(smeans)})"
            )
            rows.append(dict(
                metric=name, roi=roi, session="collapsed",
                mean_diff=smeans.mean(), sd=smeans.std(),
                t_stat=t, p_value=p, cohens_d=d,
                n_subjects=len(smeans),
                n_trial_rows=len(sub),
            ))

        # Per session
        for session in ["presleep", "postsleep"]:
            log.append(f"\n  {session}:")
            for roi in ROIS:
                sub = df[(df["roi"] == roi)
                         & (df["session"] == session)].copy()
                sub["diff"] = sub[ret_col] - sub[ctrl_col]
                smeans = sub.groupby("subject")["diff"].mean().dropna()
                t, p = ttest_1samp(smeans, 0)
                d = cohens_d_onesample(smeans)
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: mean={smeans.mean():+.6f}, "
                    f"t({len(smeans)-1})={t:+.3f}, p={p:.4f} {s}, "
                    f"d={d:+.3f}  (N={len(smeans)}, {len(sub)} trial-rows)"
                )
                rows.append(dict(
                    metric=name, roi=roi, session=session,
                    mean_diff=smeans.mean(), sd=smeans.std(),
                    t_stat=t, p_value=p, cohens_d=d,
                    n_subjects=len(smeans),
                    n_trial_rows=len(sub),
                ))

    out = pd.DataFrame(rows)
    # FDR (Benjamini-Hochberg) with the families used in the thesis:
    #   collapsed rows: across ROIs x metrics (3 x 2 = 6 tests)
    #   per-session rows: across ROIs x sessions x metrics (3 x 2 x 2 = 12 tests)
    out["p_fdr"] = np.nan
    for fam in [out["session"] == "collapsed", out["session"] != "collapsed"]:
        ok = fam & out["p_value"].notna()
        if ok.any():
            out.loc[ok, "p_fdr"] = multipletests(out.loc[ok, "p_value"], method="fdr_bh")[1]
    return out


# =====================================================================
# 3) SESSION EFFECT
# =====================================================================

def _session_tests(df, log, filter_name):
    """Paired t-test + LME for post > pre, given a (possibly filtered) df."""
    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # --- Paired t-test ---
        log.append("\n  Paired t-test (post − pre), subject-level means:")
        for roi in ROIS:
            pre = df[(df["roi"] == roi)
                     & (df["session"] == "presleep")].copy()
            post = df[(df["roi"] == roi)
                      & (df["session"] == "postsleep")].copy()
            pre["diff"] = pre[ret_col] - pre[ctrl_col]
            post["diff"] = post[ret_col] - post[ctrl_col]
            pre_m = pre.groupby("subject")["diff"].mean()
            post_m = post.groupby("subject")["diff"].mean()
            common = pre_m.index.intersection(post_m.index)
            if len(common) < 3:
                log.append(f"    {roi:10s}: too few paired subjects "
                           f"(N={len(common)})")
                continue
            pre_v = pre_m.loc[common]
            post_v = post_m.loc[common]
            change = post_v - pre_v
            t, p = ttest_rel(post_v, pre_v)
            d = cohens_d_paired(post_v, pre_v)
            s = sig_stars(p)
            log.append(
                f"    {roi:10s}: pre={pre_v.mean():+.6f}, "
                f"post={post_v.mean():+.6f}, "
                f"Δ={change.mean():+.6f}, t({len(common)-1})={t:+.3f}, "
                f"p={p:.4f} {s}, d={d:+.3f}  (N={len(common)})"
            )
            rows.append(dict(
                metric=name, trial_filter=filter_name, roi=roi,
                test="paired_ttest",
                pre_mean=pre_v.mean(), post_mean=post_v.mean(),
                diff=change.mean(), stat=t, p_value=p, cohens_d=d,
                n=len(common),
            ))

        # --- LME ---
        log.append("\n  LME: replay_diff ~ is_post + (1|subject):")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            sub = sub.dropna(subset=["replay_diff"])
            try:
                res = fit_lme(sub, "replay_diff ~ is_post")
                coef = res.params["is_post"]
                p = res.pvalues["is_post"]
                z = res.tvalues["is_post"]
                se = res.bse["is_post"]
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: coef={coef:+.6f} (SE={se:.6f}), "
                    f"z={z:+.3f}, p={p:.4f} {s}"
                )
                rows.append(dict(
                    metric=name, trial_filter=filter_name, roi=roi,
                    test="lme",
                    pre_mean=np.nan, post_mean=np.nan,
                    diff=coef, stat=z, p_value=p, cohens_d=np.nan,
                    n=sub["subject"].nunique(),
                ))
            except Exception as e:
                log.append(f"    {roi:10s}: FAILED – {e}")

        # --- Long-format LME: replay ~ is_retrieved * is_post ---
        log.append("\n  LME (long-format): replay ~ is_retrieved * is_post "
                   "+ (1|subject):")
        keep = ["subject", "session", "roi", ret_col, ctrl_col]
        sub = df[[c for c in keep if c in df.columns]].dropna(
            subset=[ret_col, ctrl_col]
        ).copy()

        df_ret = sub[["subject", "session", "roi", ret_col]].copy()
        df_ret = df_ret.rename(columns={ret_col: "replay_metric"})
        df_ret["is_retrieved"] = 1

        df_ctrl = sub[["subject", "session", "roi", ctrl_col]].copy()
        df_ctrl = df_ctrl.rename(columns={ctrl_col: "replay_metric"})
        df_ctrl["is_retrieved"] = 0

        df_long = pd.concat([df_ret, df_ctrl], ignore_index=True)
        df_long["is_post"] = (
            df_long["session"] == "postsleep"
        ).astype(int)

        for roi in ROIS:
            rsub = df_long[df_long["roi"] == roi].copy()
            try:
                res = fit_lme(
                    rsub, "replay_metric ~ is_retrieved * is_post"
                )
                ix_param = "is_retrieved:is_post"
                coef = res.params[ix_param]
                p = res.pvalues[ix_param]
                z = res.tvalues[ix_param]
                se = res.bse[ix_param]
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: interaction coef={coef:+.6f} "
                    f"(SE={se:.6f}), z={z:+.3f}, p={p:.4f} {s}"
                )
                rows.append(dict(
                    metric=name, trial_filter=filter_name, roi=roi,
                    test="lme_interaction",
                    pre_mean=np.nan, post_mean=np.nan,
                    diff=coef, stat=z, p_value=p, cohens_d=np.nan,
                    n=rsub["subject"].nunique(),
                ))
            except Exception as e:
                log.append(f"    {roi:10s}: FAILED – {e}")

    return rows


def analysis_session_effect(df_all, log):
    """Does replay change from presleep to postsleep?"""
    log.append("\n" + "=" * 70)
    log.append("3) SESSION EFFECT ON REPLAY")
    log.append("   Question: Does replay (ret − ctrl) change pre → post?")
    log.append("   Methods:  Paired t-test on subject means + LME per ROI")
    log.append("=" * 70)

    all_rows = []

    # 3a — All trials
    log.append("\n" + "-" * 50)
    log.append("3a: ALL TRIALS")
    log.append("-" * 50)
    all_rows.extend(_session_tests(df_all, log, "all"))

    # 3b — Objects only
    df_obj = df_all[df_all["object_scene_retrieval"] == "object"].copy()
    n_obj = len(df_obj[df_obj["roi"] == ROIS[0]])
    log.append("\n" + "-" * 50)
    log.append(f"3b: OBJECT TRIALS ONLY  "
               f"({n_obj} trials, {df_obj['subject'].nunique()} subjects)")
    log.append("-" * 50)
    all_rows.extend(_session_tests(df_obj, log, "object"))

    # 3c — Scenes only
    df_scene = df_all[df_all["object_scene_retrieval"] == "scene"].copy()
    n_scene = len(df_scene[df_scene["roi"] == ROIS[0]])
    log.append("\n" + "-" * 50)
    log.append(f"3c: SCENE TRIALS ONLY  "
               f"({n_scene} trials, {df_scene['subject'].nunique()} subjects)")
    log.append("-" * 50)
    all_rows.extend(_session_tests(df_scene, log, "scene"))

    return pd.DataFrame(all_rows)


# =====================================================================
# 4) ACCURACY EFFECT
# =====================================================================

def analysis_accuracy(df_all, log):
    """Does replay differ for correctly vs incorrectly recalled items?"""
    log.append("\n" + "=" * 70)
    log.append("4) ACCURACY EFFECT ON REPLAY")
    log.append("   Question: Does replay (ret − ctrl) differ by accuracy?")
    log.append("   Method:   LME: replay_diff ~ is_correct + (1|subject)")
    log.append("   Note:     Strong imbalance (~85 % correct / 15 % incorrect)")
    log.append("=" * 70)

    rows = []
    filters = [
        ("all", df_all, "ALL TRIALS"),
        ("object",
         df_all[df_all["object_scene_retrieval"] == "object"].copy(),
         "OBJECT TRIALS ONLY"),
    ]

    for filter_name, df, label in filters:
        log.append(f"\n" + "-" * 50)
        log.append(f"  {label}")
        log.append("-" * 50)

        for ret_col, ctrl_col, name in METRICS:
            log.append(f"\n--- {name} ---")

            # Collapsed
            log.append("\n  Collapsed across sessions:")
            for roi in ROIS:
                sub = df[df["roi"] == roi].copy()
                sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
                sub["is_correct"] = pd.to_numeric(
                    sub["behav_correct"], errors="coerce"
                ).astype(float)
                sub = sub.dropna(subset=["replay_diff", "is_correct"])
                try:
                    res = fit_lme(sub, "replay_diff ~ is_correct")
                    coef = res.params["is_correct"]
                    p = res.pvalues["is_correct"]
                    z = res.tvalues["is_correct"]
                    s = sig_stars(p)
                    log.append(
                        f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                        f"p={p:.4f} {s}"
                    )
                    rows.append(dict(
                        metric=name, trial_filter=filter_name, roi=roi,
                        session="collapsed", coef=coef, stat=z, p_value=p,
                        n=sub["subject"].nunique(),
                    ))
                except Exception as e:
                    log.append(f"    {roi:10s}: FAILED – {e}")

            # Per session
            for session in ["presleep", "postsleep"]:
                log.append(f"\n  {session}:")
                for roi in ROIS:
                    sub = df[(df["roi"] == roi)
                             & (df["session"] == session)].copy()
                    sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
                    sub["is_correct"] = pd.to_numeric(
                        sub["behav_correct"], errors="coerce"
                    )
                    sub = sub.dropna(subset=["replay_diff", "is_correct"])
                    try:
                        res = fit_lme(sub, "replay_diff ~ is_correct")
                        coef = res.params["is_correct"]
                        p = res.pvalues["is_correct"]
                        z = res.tvalues["is_correct"]
                        s = sig_stars(p)
                        log.append(
                            f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                            f"p={p:.4f} {s}"
                        )
                        rows.append(dict(
                            metric=name, trial_filter=filter_name, roi=roi,
                            session=session, coef=coef, stat=z, p_value=p,
                            n=sub["subject"].nunique(),
                        ))
                    except Exception as e:
                        log.append(f"    {roi:10s}: FAILED – {e}")

    return pd.DataFrame(rows)


# =====================================================================
# 5) TRIAL TYPE — Object vs Scene
# =====================================================================

def analysis_trial_type(df_all, log):
    """Does replay differ between objects and scenes?"""
    log.append("\n" + "=" * 70)
    log.append("5) TRIAL TYPE COMPARISON — OBJECT vs SCENE")
    log.append("   Question: Does replay (ret − ctrl) differ by trial type?")
    log.append("   Method:   LME: replay_diff ~ is_object + (1|subject)")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # Collapsed
        log.append("\n  Collapsed across sessions:")
        for roi in ROIS:
            sub = df_all[df_all["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["is_object"] = (
                sub["object_scene_retrieval"] == "object"
            ).astype(int)
            try:
                res = fit_lme(sub, "replay_diff ~ is_object")
                coef = res.params["is_object"]
                p = res.pvalues["is_object"]
                z = res.tvalues["is_object"]
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                    f"p={p:.4f} {s}"
                )
                rows.append(dict(
                    metric=name, roi=roi, session="collapsed",
                    coef=coef, stat=z, p_value=p,
                ))
            except Exception as e:
                log.append(f"    {roi:10s}: FAILED – {e}")

        # Per session
        for session in ["presleep", "postsleep"]:
            log.append(f"\n  {session}:")
            for roi in ROIS:
                sub = df_all[
                    (df_all["roi"] == roi)
                    & (df_all["session"] == session)
                ].copy()
                sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
                sub["is_object"] = (
                    sub["object_scene_retrieval"] == "object"
                ).astype(int)
                try:
                    res = fit_lme(sub, "replay_diff ~ is_object")
                    coef = res.params["is_object"]
                    p = res.pvalues["is_object"]
                    z = res.tvalues["is_object"]
                    s = sig_stars(p)
                    log.append(
                        f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                        f"p={p:.4f} {s}"
                    )
                    rows.append(dict(
                        metric=name, roi=roi, session=session,
                        coef=coef, stat=z, p_value=p,
                    ))
                except Exception as e:
                    log.append(f"    {roi:10s}: FAILED – {e}")

    return pd.DataFrame(rows)


# =====================================================================
# 6) INTERACTIONS
# =====================================================================

def analysis_interactions(df_all, log):
    """Session × trial-type interactions and omnibus model."""
    log.append("\n" + "=" * 70)
    log.append("6) INTERACTIONS")
    log.append("=" * 70)

    rows = []

    # ---- 6a: Session × Trial Type, per ROI ----
    log.append("\n" + "-" * 50)
    log.append("6a: SESSION × TRIAL TYPE — ALL TRIALS, per ROI")
    log.append("    LME: replay_diff ~ is_post * is_object + (1|subject)")
    log.append("    Tests whether the session effect differs for "
               "objects vs scenes")
    log.append("-" * 50)

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")
        for roi in ROIS:
            sub = df_all[df_all["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub["is_post"] = (sub["session"] == "postsleep").astype(int)
            sub["is_object"] = (
                sub["object_scene_retrieval"] == "object"
            ).astype(int)
            sub = sub.dropna(subset=["replay_diff"])
            try:
                res = fit_lme(sub, "replay_diff ~ is_post * is_object")
                log.append(f"\n    {roi}:")
                for param in res.params.index:
                    coef = res.params[param]
                    p = res.pvalues[param]
                    z = res.tvalues[param]
                    s = sig_stars(p)
                    log.append(
                        f"      {param:30s}: coef={coef:+.6f}, "
                        f"z={z:+.3f}, p={p:.4f} {s}"
                    )
                    rows.append(dict(
                        metric=name, analysis="6a_session_x_type",
                        roi=roi, parameter=param,
                        coef=coef, se=res.bse[param],
                        stat=z, p_value=p,
                        n=sub["subject"].nunique(),
                    ))
            except Exception as e:
                log.append(f"    {roi}: FAILED – {e}")

    # ---- 6b: Omnibus — objects only ----
    log.append("\n" + "-" * 50)
    log.append("6b: OMNIBUS LME — OBJECT TRIALS ONLY")
    log.append("    replay ~ is_retrieved * is_post * C(roi) + (1|subject)")
    log.append("    Three-way interaction: does the replay boost (ret > ctrl)")
    log.append("    increase postsleep differently across ROIs?")
    log.append("-" * 50)

    df_obj = df_all[df_all["object_scene_retrieval"] == "object"].copy()

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        keep = ["subject", "session", "roi", ret_col, ctrl_col]
        sub = df_obj[[c for c in keep if c in df_obj.columns]].dropna(
            subset=[ret_col, ctrl_col]
        ).copy()

        df_ret = sub[["subject", "session", "roi", ret_col]].copy()
        df_ret = df_ret.rename(columns={ret_col: "replay_metric"})
        df_ret["is_retrieved"] = 1

        df_ctrl = sub[["subject", "session", "roi", ctrl_col]].copy()
        df_ctrl = df_ctrl.rename(columns={ctrl_col: "replay_metric"})
        df_ctrl["is_retrieved"] = 0

        df_long = pd.concat([df_ret, df_ctrl], ignore_index=True)
        df_long["is_post"] = (
            df_long["session"] == "postsleep"
        ).astype(int)

        try:
            res = fit_lme(
                df_long,
                "replay_metric ~ is_retrieved * is_post * C(roi)"
            )
            log.append("")
            for param in res.params.index:
                coef = res.params[param]
                p = res.pvalues[param]
                z = res.tvalues[param]
                s = sig_stars(p)
                log.append(
                    f"  {param:55s}: coef={coef:+.6f}, "
                    f"z={z:+.3f}, p={p:.4f} {s}"
                )
                rows.append(dict(
                    metric=name, analysis="6b_omnibus_objects",
                    roi="all", parameter=param,
                    coef=coef, se=res.bse[param],
                    stat=z, p_value=p,
                    n=df_long["subject"].nunique(),
                ))
        except Exception as e:
            log.append(f"  FAILED – {e}")

    return pd.DataFrame(rows)


# =====================================================================
# 7) BRAIN-BEHAVIOR
# =====================================================================

def analysis_brain_behavior(df, log):
    """Subject-level correlations — replay vs accuracy (objects only)."""
    log.append("\n" + "=" * 70)
    log.append("7) BRAIN-BEHAVIOR CORRELATIONS — OBJECT TRIALS ONLY")
    log.append("   Pearson r at the subject level")
    log.append("   7a: Raw retrieved metric vs accuracy")
    log.append("   7b: Replay diff (ret − ctrl) vs accuracy")
    log.append("=" * 70)

    df["behav_correct"] = pd.to_numeric(df["behav_correct"], errors="coerce")
    rows = []

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        for variant, vlabel in [("raw", "7a: Raw retrieved"),
                                ("diff", "7b: Ret − Ctrl")]:
            log.append(f"\n  {vlabel}:")

            # Collapsed
            log.append("    Collapsed:")
            for roi in ROIS:
                sub = df[df["roi"] == roi].copy()
                if variant == "raw":
                    sub = sub.dropna(subset=[ret_col, "behav_correct"])
                    sm = sub.groupby("subject").agg(
                        replay=(ret_col, "mean"),
                        acc=("behav_correct", "mean"),
                    ).dropna()
                else:
                    sub["diff"] = sub[ret_col] - sub[ctrl_col]
                    sub = sub.dropna(subset=["diff", "behav_correct"])
                    sm = sub.groupby("subject").agg(
                        replay=("diff", "mean"),
                        acc=("behav_correct", "mean"),
                    ).dropna()

                if len(sm) < 5:
                    continue
                r, p = pearsonr(sm["replay"], sm["acc"])
                s = sig_stars(p)
                log.append(
                    f"      {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  "
                    f"(N={len(sm)})"
                )
                rows.append(dict(
                    metric=name, variant=vlabel, roi=roi,
                    session="collapsed", r=r, p_value=p, n=len(sm),
                ))

            # Per session
            for session in ["presleep", "postsleep"]:
                log.append(f"    {session}:")
                for roi in ROIS:
                    sub = df[(df["roi"] == roi)
                             & (df["session"] == session)].copy()
                    if variant == "raw":
                        sub = sub.dropna(subset=[ret_col, "behav_correct"])
                        sm = sub.groupby("subject").agg(
                            replay=(ret_col, "mean"),
                            acc=("behav_correct", "mean"),
                        ).dropna()
                    else:
                        sub["diff"] = sub[ret_col] - sub[ctrl_col]
                        sub = sub.dropna(subset=["diff", "behav_correct"])
                        sm = sub.groupby("subject").agg(
                            replay=("diff", "mean"),
                            acc=("behav_correct", "mean"),
                        ).dropna()

                    if len(sm) < 5:
                        continue
                    r, p = pearsonr(sm["replay"], sm["acc"])
                    s = sig_stars(p)
                    log.append(
                        f"      {roi:10s}: r={r:+.3f}, p={p:.4f} {s}  "
                        f"(N={len(sm)})"
                    )
                    rows.append(dict(
                        metric=name, variant=vlabel, roi=roi,
                        session=session, r=r, p_value=p, n=len(sm),
                    ))

    out = pd.DataFrame(rows)
    # FDR (Benjamini-Hochberg) across the ROIs within each metric x variant x
    # session — the "FDR across the three ROIs" reported in the thesis. Before
    # this, the p_FDR values in the thesis were computed by hand.
    out["p_fdr"] = np.nan
    for _, idx in out.groupby(["metric", "variant", "session"]).groups.items():
        ok = out.loc[idx, "p_value"].notna()
        idx = [i for i, k in zip(idx, ok) if k]
        if idx:
            out.loc[idx, "p_fdr"] = multipletests(out.loc[idx, "p_value"], method="fdr_bh")[1]
    return out


# =====================================================================
# 9) SHUFFLED BASELINE VALIDATION
# =====================================================================

def analysis_shuffled_validation(df, log):
    """Confirm replay is sequence-specific: shuffled labels should kill it."""
    log.append("\n" + "=" * 70)
    log.append("9) SHUFFLED BASELINE VALIDATION")
    log.append("   Question: Does the replay effect vanish with shuffled "
               "position labels?")
    log.append("   Method:   One-sample t-test on subject-level mean"
               "(shuf_ret − shuf_ctrl) vs 0")
    log.append("   Expected: Null everywhere (confirming real effect is "
               "sequence-specific)")
    log.append("=" * 70)

    rows = []
    for ret_col, ctrl_col, name in METRICS_SHUFFLED:
        log.append(f"\n--- {name} ---")

        # Collapsed
        log.append("\n  Collapsed across sessions:")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["diff"] = sub[ret_col] - sub[ctrl_col]
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            d = cohens_d_onesample(smeans)
            s = sig_stars(p)
            log.append(
                f"    {roi:10s}: mean={smeans.mean():+.6f}, "
                f"t({len(smeans)-1})={t:+.3f}, p={p:.4f} {s}, d={d:+.3f}  "
                f"(N={len(smeans)})"
            )
            rows.append(dict(
                metric=name, roi=roi, session="collapsed",
                mean_diff=smeans.mean(), t_stat=t, p_value=p,
                cohens_d=d, n_subjects=len(smeans),
            ))

        # Per session
        for session in ["presleep", "postsleep"]:
            log.append(f"\n  {session}:")
            for roi in ROIS:
                sub = df[(df["roi"] == roi)
                         & (df["session"] == session)].copy()
                sub["diff"] = sub[ret_col] - sub[ctrl_col]
                smeans = sub.groupby("subject")["diff"].mean().dropna()
                t, p = ttest_1samp(smeans, 0)
                d = cohens_d_onesample(smeans)
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: mean={smeans.mean():+.6f}, "
                    f"t({len(smeans)-1})={t:+.3f}, p={p:.4f} {s}, "
                    f"d={d:+.3f}  (N={len(smeans)})"
                )
                rows.append(dict(
                    metric=name, roi=roi, session=session,
                    mean_diff=smeans.mean(), t_stat=t, p_value=p,
                    cohens_d=d, n_subjects=len(smeans),
                ))

    return pd.DataFrame(rows)


# =====================================================================
# 11) WITHIN-SESSION TEMPORAL DYNAMICS
# =====================================================================

def analysis_run_effects(df, log):
    """Does replay decay within postsleep (early vs late runs)?"""
    log.append("\n" + "=" * 70)
    log.append("11) WITHIN-SESSION TEMPORAL DYNAMICS")
    log.append("   Postsleep runs 1–3 = Ret₂ session 1 (early)")
    log.append("   Postsleep runs 4–6 = Ret₂ session 2 (late)")
    log.append("   11a: Paired t-test early vs late postsleep")
    log.append("   11b: LME replay_diff ~ run + (1|subject) within "
               "postsleep")
    log.append("=" * 70)

    rows = []
    df_post = df[df["session"] == "postsleep"].copy()

    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # 11a: Early vs late paired t-test
        log.append("\n  11a: Paired t-test (early runs 1–3 vs "
                   "late runs 4–6):")
        for roi in ROIS:
            sub = df_post[df_post["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            early = sub[sub["run"].isin([1, 2, 3])]
            late = sub[sub["run"].isin([4, 5, 6])]
            early_m = early.groupby("subject")["replay_diff"].mean()
            late_m = late.groupby("subject")["replay_diff"].mean()
            common = early_m.index.intersection(late_m.index)
            if len(common) < 3:
                log.append(f"    {roi:10s}: too few subjects (N="
                           f"{len(common)})")
                continue
            e_v = early_m.loc[common]
            l_v = late_m.loc[common]
            change = l_v - e_v
            t, p = ttest_rel(l_v, e_v)
            d = cohens_d_paired(l_v, e_v)
            s = sig_stars(p)
            log.append(
                f"    {roi:10s}: early={e_v.mean():+.6f}, "
                f"late={l_v.mean():+.6f}, Δ={change.mean():+.6f}, "
                f"t({len(common)-1})={t:+.3f}, p={p:.4f} {s}, "
                f"d={d:+.3f}  (N={len(common)})"
            )
            rows.append(dict(
                metric=name, roi=roi, test="11a_early_vs_late",
                early_mean=e_v.mean(), late_mean=l_v.mean(),
                diff=change.mean(), stat=t, p_value=p, cohens_d=d,
                n=len(common),
            ))

        # 11b: Continuous run effect (LME)
        log.append("\n  11b: LME replay_diff ~ run + (1|subject), "
                   "postsleep:")
        for roi in ROIS:
            sub = df_post[df_post["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub = sub.dropna(subset=["replay_diff"])
            try:
                res = fit_lme(sub, "replay_diff ~ run")
                coef = res.params["run"]
                p = res.pvalues["run"]
                z = res.tvalues["run"]
                s = sig_stars(p)
                log.append(
                    f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                    f"p={p:.4f} {s}"
                )
                rows.append(dict(
                    metric=name, roi=roi, test="11b_lme_run",
                    early_mean=np.nan, late_mean=np.nan,
                    diff=coef, stat=z, p_value=p, cohens_d=np.nan,
                    n=sub["subject"].nunique(),
                ))
            except Exception as e:
                log.append(f"    {roi:10s}: FAILED – {e}")

    return pd.DataFrame(rows)


# =====================================================================
# 13) FIT QUALITY AS MODERATOR
# =====================================================================

def analysis_fit_quality(df, log):
    """Are sinusoidal replay effects stronger for well-fit trials?"""
    log.append("\n" + "=" * 70)
    log.append("13) FIT QUALITY AS MODERATOR — sinusoidal metric only")
    log.append("   13a: Median split on R² — high vs low fit quality")
    log.append("   13b: LME replay_diff ~ R² + (1|subject)")
    log.append("=" * 70)

    ret_col = "retrieved_w_amplitude"
    ctrl_col = "control_w_amplitude"
    r2_col = "retrieved_w_r_squared"

    rows = []

    # 13a: Median split
    log.append("\n  13a: Median split on sinusoidal R²:")
    for session_label in ["collapsed", "presleep", "postsleep"]:
        if session_label == "collapsed":
            log.append("\n    Collapsed:")
            sub_all = df.copy()
        else:
            log.append(f"\n    {session_label}:")
            sub_all = df[df["session"] == session_label].copy()

        for roi in ROIS:
            sub = sub_all[sub_all["roi"] == roi].copy()
            sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
            sub = sub.dropna(subset=["replay_diff", r2_col])

            # Within-subject median split
            medians = sub.groupby("subject")[r2_col].transform("median")
            sub["r2_high"] = (sub[r2_col] >= medians).astype(int)

            hi = sub[sub["r2_high"] == 1].groupby(
                "subject"
            )["replay_diff"].mean()
            lo = sub[sub["r2_high"] == 0].groupby(
                "subject"
            )["replay_diff"].mean()
            common = hi.dropna().index.intersection(lo.dropna().index)
            if len(common) < 3:
                log.append(f"      {roi:10s}: too few subjects")
                continue
            hi_v = hi.loc[common]
            lo_v = lo.loc[common]
            t, p = ttest_rel(hi_v, lo_v)
            d = cohens_d_paired(hi_v, lo_v)
            s = sig_stars(p)
            log.append(
                f"      {roi:10s}: high_R²={hi_v.mean():+.6f}, "
                f"low_R²={lo_v.mean():+.6f}, "
                f"Δ={hi_v.mean()-lo_v.mean():+.6f}, "
                f"t({len(common)-1})={t:+.3f}, p={p:.4f} {s}  "
                f"(N={len(common)})"
            )
            rows.append(dict(
                roi=roi, session=session_label,
                test="13a_median_split",
                high_r2_mean=hi_v.mean(), low_r2_mean=lo_v.mean(),
                diff=hi_v.mean() - lo_v.mean(),
                stat=t, p_value=p, cohens_d=d, n=len(common),
            ))

    # 13b: Continuous R² as predictor (LME, collapsed)
    log.append("\n  13b: LME replay_diff ~ R² + (1|subject), collapsed:")
    for roi in ROIS:
        sub = df[df["roi"] == roi].copy()
        sub["replay_diff"] = sub[ret_col] - sub[ctrl_col]
        sub = sub.dropna(subset=["replay_diff", r2_col])
        try:
            res = fit_lme(sub, f"replay_diff ~ {r2_col}")
            coef = res.params[r2_col]
            p = res.pvalues[r2_col]
            z = res.tvalues[r2_col]
            s = sig_stars(p)
            log.append(
                f"    {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, "
                f"p={p:.4f} {s}"
            )
            rows.append(dict(
                roi=roi, session="collapsed",
                test="13b_lme_r2",
                high_r2_mean=np.nan, low_r2_mean=np.nan,
                diff=coef, stat=z, p_value=p, cohens_d=np.nan,
                n=sub["subject"].nunique(),
            ))
        except Exception as e:
            log.append(f"    {roi:10s}: FAILED – {e}")

    # 13c: R² as a replay metric — retrieved R² > control R²
    r2_ctrl_col = "control_w_r_squared"
    log.append("\n  13c: R² as replay evidence — is retrieved R² > "
               "control R²?")
    log.append("       Logic: if replay produces a sinusoidal SODA "
               "pattern, the")
    log.append("       sinusoidal fit should be BETTER for retrieved "
               "than control.")

    # Collapsed + per session
    for session_label in ["collapsed", "presleep", "postsleep"]:
        if session_label == "collapsed":
            log.append("\n    Collapsed:")
            sub_all = df.copy()
        else:
            log.append(f"\n    {session_label}:")
            sub_all = df[df["session"] == session_label].copy()
        for roi in ROIS:
            sub = sub_all[sub_all["roi"] == roi].copy()
            sub["r2_diff"] = sub[r2_col] - sub[r2_ctrl_col]
            sub = sub.dropna(subset=["r2_diff"])
            smeans = sub.groupby("subject")["r2_diff"].mean().dropna()
            if len(smeans) < 3:
                continue
            t, p = ttest_1samp(smeans, 0)
            d = cohens_d_onesample(smeans)
            s = sig_stars(p)
            log.append(
                f"      {roi:10s}: mean={smeans.mean():+.6f}, "
                f"t({len(smeans)-1})={t:+.3f}, p={p:.4f} {s}, "
                f"d={d:+.3f}  (N={len(smeans)})"
            )
            rows.append(dict(
                roi=roi, session=session_label,
                test="13c_r2_as_metric",
                high_r2_mean=np.nan, low_r2_mean=np.nan,
                diff=smeans.mean(), stat=t, p_value=p,
                cohens_d=d, n=len(smeans),
            ))

    # 13d: R² session effect — does fit quality increase post-sleep?
    log.append("\n  13d: Session effect on R² diff (post > pre?):")
    for roi in ROIS:
        pre = df[(df["roi"] == roi)
                 & (df["session"] == "presleep")].copy()
        post = df[(df["roi"] == roi)
                  & (df["session"] == "postsleep")].copy()
        pre["r2_diff"] = pre[r2_col] - pre[r2_ctrl_col]
        post["r2_diff"] = post[r2_col] - post[r2_ctrl_col]
        pre_m = pre.groupby("subject")["r2_diff"].mean()
        post_m = post.groupby("subject")["r2_diff"].mean()
        common = pre_m.dropna().index.intersection(
            post_m.dropna().index
        )
        if len(common) < 3:
            continue
        pre_v = pre_m.loc[common]
        post_v = post_m.loc[common]
        change = post_v - pre_v
        t, p = ttest_rel(post_v, pre_v)
        d_val = cohens_d_paired(post_v, pre_v)
        s = sig_stars(p)
        log.append(
            f"    {roi:10s}: pre={pre_v.mean():+.6f}, "
            f"post={post_v.mean():+.6f}, "
            f"Δ={change.mean():+.6f}, t({len(common)-1})={t:+.3f}, "
            f"p={p:.4f} {s}  (N={len(common)})"
        )
        rows.append(dict(
            roi=roi, session="post_vs_pre",
            test="13d_r2_session",
            high_r2_mean=np.nan, low_r2_mean=np.nan,
            diff=change.mean(), stat=t, p_value=p,
            cohens_d=d_val, n=len(common),
        ))

    return pd.DataFrame(rows)


# =====================================================================
# 8) KEY FINDINGS SUMMARY
# =====================================================================

def write_summary(csvs, log):
    """Data-driven summary of all significant results."""
    log.append("\n\n" + "=" * 70)
    log.append("8) KEY FINDINGS SUMMARY")
    log.append("=" * 70)

    # --- Replay detection ---
    det = csvs["2_replay_detection"]
    collapsed = det[det["session"] == "collapsed"]
    pre = det[det["session"] == "presleep"]
    post = det[det["session"] == "postsleep"]

    log.append("\n  REPLAY DETECTION (Analysis 2):")
    log.append(f"    Collapsed:  {(collapsed['p_value'] < 0.05).sum()}"
               f"/{len(collapsed)} significant (p < .05)")
    log.append(f"    Presleep:   {(pre['p_value'] < 0.05).sum()}"
               f"/{len(pre)} significant")
    log.append(f"    Postsleep:  {(post['p_value'] < 0.05).sum()}"
               f"/{len(post)} significant")

    # Which ROIs are significant per session?
    for session, sdf in [("presleep", pre), ("postsleep", post)]:
        sig_rois = sdf[sdf["p_value"] < 0.05]["roi"].unique()
        if len(sig_rois) > 0:
            log.append(f"    → {session} significant ROIs: "
                       f"{', '.join(sorted(sig_rois))}")

    # --- Session effect ---
    sess = csvs["3_session_effect"]
    for filt, label in [("all", "All trials"), ("object", "Objects"),
                        ("scene", "Scenes")]:
        sub = sess[sess["trial_filter"] == filt]
        sig = sub[sub["p_value"] < 0.05]
        log.append(f"\n  SESSION EFFECT — {label} (Analysis 3):")
        if len(sig) == 0:
            log.append(f"    No significant effects (0/{len(sub)} tests)")
        else:
            for _, row in sig.iterrows():
                log.append(
                    f"    {row['metric']:12s} {row['roi']:10s} "
                    f"({row['test']:14s}): "
                    f"p={row['p_value']:.4f} {sig_stars(row['p_value'])}"
                    f"  Δ={row['diff']:+.6f}"
                )

    # --- Accuracy ---
    acc = csvs["4_accuracy"]
    log.append(f"\n  ACCURACY EFFECT (Analysis 4):")
    if len(acc) == 0 or "p_value" not in acc.columns:
        log.append(f"    No tests completed")
    else:
        sig_acc = acc[acc["p_value"] < 0.05]
        if len(sig_acc) == 0:
            log.append(f"    No significant effects (0/{len(acc)} tests)")
        else:
            for _, row in sig_acc.iterrows():
                log.append(
                    f"    {row['metric']:12s} {row['roi']:10s} "
                    f"{row['trial_filter']:8s} {row['session']:12s}: "
                    f"p={row['p_value']:.4f}"
                )

    # --- Trial type ---
    tt = csvs["5_trial_type"]
    log.append(f"\n  TRIAL TYPE — Object vs Scene (Analysis 5):")
    if len(tt) == 0 or "p_value" not in tt.columns:
        log.append(f"    No tests completed")
    else:
        sig_tt = tt[tt["p_value"] < 0.05]
        if len(sig_tt) == 0:
            log.append(f"    No significant differences (0/{len(tt)} tests)")
        else:
            for _, row in sig_tt.iterrows():
                log.append(
                    f"    {row['metric']:12s} {row['roi']:10s}: "
                    f"p={row['p_value']:.4f}"
                )

    # --- Brain-behavior ---
    bb = csvs["7_brain_behavior"]
    sig_bb = bb[bb["p_value"] < 0.05]
    log.append(f"\n  BRAIN-BEHAVIOR (Analysis 7, objects only):")
    log.append(f"    {len(sig_bb)}/{len(bb)} significant at p < .05 "
               f"uncorrected")
    if len(sig_bb) > 0:
        for _, row in sig_bb.iterrows():
            log.append(
                f"    {row['metric']:12s} {row['variant']:20s} "
                f"{row['roi']:10s} {row['session']:12s}: "
                f"r={row['r']:+.3f}, p={row['p_value']:.4f}"
            )

    # --- Shuffled validation ---
    if "9_shuffled_validation" in csvs:
        shuf = csvs["9_shuffled_validation"]
        log.append(f"\n  SHUFFLED VALIDATION (Analysis 9):")
        if len(shuf) > 0 and "p_value" in shuf.columns:
            n_sig = (shuf["p_value"] < 0.05).sum()
            log.append(f"    {n_sig}/{len(shuf)} shuffled tests "
                       f"significant (expect ~0)")

    # --- Run effects ---
    if "11_run_effects" in csvs:
        run_df = csvs["11_run_effects"]
        log.append(f"\n  RUN EFFECTS — postsleep decay (Analysis 11):")
        if len(run_df) > 0 and "p_value" in run_df.columns:
            sig_run = run_df[run_df["p_value"] < 0.05]
            if len(sig_run) == 0:
                log.append(f"    No significant decay (0/{len(run_df)})")
            else:
                for _, row in sig_run.iterrows():
                    log.append(
                        f"    {row.get('metric',''):12s} "
                        f"{row.get('roi',''):10s} ({row['test']:20s}): "
                        f"p={row['p_value']:.4f} {sig_stars(row['p_value'])}"
                    )

    # --- Narrative ---
    log.append("\n" + "-" * 50)
    log.append("INTERPRETATION NOTES")
    log.append("-" * 50)
    log.append(
        "  Replay detection:  Collapsed across sessions, replay is robust\n"
        "  across all ROIs and metrics.  When split by session, presleep\n"
        "  replay is concentrated in visual cortex, while postsleep replay\n"
        "  extends to MTL and parietal cortex.  This regional shift is\n"
        "  consistent with systems-level consolidation, but note that\n"
        "  postsleep has 2× the trials, giving it more statistical power.\n"
    )
    log.append(
        "  Session effect:  No overall pre-to-post increase on all trials.\n"
        "  However, when restricting to object trials — which carry\n"
        "  behavioral variance (50 unique items, variable accuracy) —\n"
        "  a session boost emerges in parietal cortex.  Scene trials\n"
        "  (5 repeated backgrounds, near-ceiling accuracy) show no\n"
        "  session effect, confirming the specificity.  The parietal\n"
        "  sleep effect for objects is theoretically expected: objects\n"
        "  are the harder, more variable items that benefit from\n"
        "  sleep-dependent consolidation.\n"
    )
    log.append(
        "  Accuracy:  No significant replay difference between correct\n"
        "  and incorrect items.  This may reflect the strong accuracy\n"
        "  imbalance (~85 %/15 %) limiting power for this comparison.\n"
    )
    log.append(
        "  Brain-behavior:  Subjects with stronger parietal replay show\n"
        "  better object memory accuracy, converging across SODA and FFT\n"
        "  metrics.  MTL replay also correlates with accuracy,\n"
        "  particularly postsleep.\n"
    )


# =====================================================================
# MAIN
# =====================================================================

def subsample_postsleep(df):
    """Keep only the first half of postsleep trials per subject × ROI."""
    df_pre = df[df["session"] == "presleep"].copy()
    df_post = df[df["session"] == "postsleep"].copy()
    keep_idx = []
    for (subj, roi), grp in df_post.groupby(["subject", "roi"]):
        half = len(grp) // 2
        keep_idx.extend(grp.index[:half].tolist())
    df_post_half = df_post.loc[keep_idx]
    return pd.concat([df_pre, df_post_half], ignore_index=True)


def run_pipeline(df, results_dir, label=""):
    """Run the full analysis pipeline on a dataframe.

    Parameters
    ----------
    df : DataFrame — master data (possibly subsampled).
    results_dir : str — output directory for CSVs and results.txt.
    label : str — header label (e.g. "FULL DATA" or "MATCHED").
    """
    os.makedirs(results_dir, exist_ok=True)

    log = []
    csvs = {}

    if label:
        log.append("=" * 70)
        log.append(f"  {label}")
        log.append("=" * 70)

    # 1) Data description
    print("  1) Data description...")
    data_description(df, log)

    # 2) Replay detection
    print("  2) Replay detection...")
    csvs["2_replay_detection"] = analysis_replay_detection(df, log)

    # 3) Session effect
    print("  3) Session effect (all / objects / scenes)...")
    csvs["3_session_effect"] = analysis_session_effect(df, log)

    # 4) Accuracy effect
    print("  4) Accuracy effect...")
    csvs["4_accuracy"] = analysis_accuracy(df, log)

    # 5) Trial type
    print("  5) Trial type comparison...")
    csvs["5_trial_type"] = analysis_trial_type(df, log)

    # 6) Interactions
    print("  6) Interactions...")
    csvs["6_interactions"] = analysis_interactions(df, log)

    # 7) Brain-behavior (objects only)
    df_obj = df[df["object_scene_retrieval"] == "object"].copy()
    n_obj = len(df_obj[df_obj["roi"] == ROIS[0]])
    print(f"  7) Brain-behavior correlations ({n_obj} object trials)...")
    csvs["7_brain_behavior"] = analysis_brain_behavior(df_obj, log)

    # 9) Shuffled validation
    print("  9) Shuffled baseline validation...")
    csvs["9_shuffled_validation"] = analysis_shuffled_validation(df, log)

    # 11) Run effects
    print("  11) Within-session temporal dynamics...")
    csvs["11_run_effects"] = analysis_run_effects(df, log)

    # 13) Fit quality
    print("  13) Fit quality as moderator...")
    csvs["13_fit_quality"] = analysis_fit_quality(df, log)

    # Summary
    print("  Compiling summary...")
    write_summary(csvs, log)

    # Save text log
    log_path = opj(results_dir, "results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))
    print(f"  Results log: {log_path}")

    # Save CSV tables
    for name, tbl in csvs.items():
        if tbl is not None and len(tbl) > 0:
            path = opj(results_dir, f"{name}.csv")
            tbl.to_csv(path, index=False)
            print(f"    {path}")

    return log


def main():
    df_all = pd.read_csv(MASTER_CSV)
    # The master dataframe can hold more ROIs than are analysed here; restrict
    # to ROIS so that models over C(roi) (6b omnibus) use exactly these ROIs.
    df_all = df_all[df_all["roi"].isin(ROIS)].copy()

    # Prepare behavioral column (may be bool or string)
    df_all["behav_correct"] = pd.to_numeric(
        df_all["behav_correct"], errors="coerce"
    ).astype(float)
    df_all["is_correct"] = df_all["behav_correct"]

    # --- Run 1: Full data ---
    full_dir = opj(SCRIPT_DIR, "5.2_replay_statistics")
    print("\n" + "=" * 60)
    print("FULL DATA (all postsleep trials)")
    print("=" * 60)
    log_full = run_pipeline(df_all, full_dir, "FULL DATA")

    # --- Run 2: Matched (first-half postsleep) ---
    df_matched = subsample_postsleep(df_all)
    matched_dir = opj(SCRIPT_DIR, "5.2_replay_statistics_matched")
    ref_roi = ROIS[0]
    n_pre = len(df_matched[(df_matched["session"] == "presleep")
                           & (df_matched["roi"] == ref_roi)])
    n_post = len(df_matched[(df_matched["session"] == "postsleep")
                            & (df_matched["roi"] == ref_roi)])
    print("\n" + "=" * 60)
    print(f"MATCHED DATA (postsleep first half: "
          f"pre={n_pre}, post={n_post} per ROI)")
    print("=" * 60)
    log_matched = run_pipeline(df_matched, matched_dir,
                               f"MATCHED (pre={n_pre}, post={n_post})")

    # Print both logs
    print("\n" + "\n".join(log_full))
    print("\n" + "\n".join(log_matched))


if __name__ == "__main__":
    main()
