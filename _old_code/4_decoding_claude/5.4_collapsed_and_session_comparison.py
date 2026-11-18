# -*- coding: utf-8 -*-
"""Collapsed replay effects and pre-vs-post session comparison.

A) Collapsed across sessions: is retrieved > control? (one-sample t-test)
B) Is the effect bigger postsleep? (paired t-test on subject means)
C) LME: replay_diff ~ session + (1|subject)

Usage
-----
    python 5.4_collapsed_and_session_comparison.py

Output
------
    5.4_collapsed_and_session_comparison/
        collapsed_replay_effect.csv
        session_comparison.csv
        lme_session_effect.csv
        results.txt
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_rel

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = opj(SCRIPT_DIR, "5.4_collapsed_and_session_comparison")
MASTER_CSV = opj(SCRIPT_DIR, "5.1_master_replay_dataframe", "master_replay_metrics.csv")

METRICS = [
    ("retrieved_soda_abs", "control_soda_abs", "Abs SODA"),
    ("retrieved_w_amplitude", "control_w_amplitude", "Sinus Amp"),
    ("retrieved_band_power", "control_band_power", "FFT Power"),
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
    print(f"Loaded {len(df)} trials, {df['subject'].nunique()} subjects")

    log = []

    # ==================================================================
    # A) Collapsed across sessions: is retrieved > control?
    # ==================================================================
    log.append("=" * 70)
    log.append("A) Collapsed across sessions: is retrieved > control?")
    log.append("=" * 70)

    rows_a = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")
        for roi in ROIS:
            sub = df[df["roi"] == roi].copy()
            sub["diff"] = sub[ret_col] - sub[ctrl_col]
            smeans = sub.groupby("subject")["diff"].mean().dropna()
            t, p = ttest_1samp(smeans, 0)
            s = sig_stars(p)
            log.append(f"  {roi:10s}: mean={smeans.mean():+.6f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(smeans)})")
            rows_a.append(dict(metric=name, roi=roi, mean_diff=smeans.mean(),
                               t_stat=t, p_value=p, n=len(smeans)))

    tbl_a = pd.DataFrame(rows_a)
    tbl_a.to_csv(opj(RESULTS_DIR, "collapsed_replay_effect.csv"), index=False)

    # ==================================================================
    # B) Is the effect bigger postsleep than presleep?
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("B) Is the effect BIGGER postsleep than presleep?")
    log.append("   Paired t-test on subject-level (post - pre) replay_diff")
    log.append("=" * 70)

    rows_b = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")
        for roi in ROIS:
            pre = df[(df["roi"] == roi) & (df["session"] == "presleep")].copy()
            post = df[(df["roi"] == roi) & (df["session"] == "postsleep")].copy()
            pre["diff"] = pre[ret_col] - pre[ctrl_col]
            post["diff"] = post[ret_col] - post[ctrl_col]
            pre_means = pre.groupby("subject")["diff"].mean()
            post_means = post.groupby("subject")["diff"].mean()
            common = pre_means.index.intersection(post_means.index)
            if len(common) < 3:
                continue
            pre_m = pre_means.loc[common]
            post_m = post_means.loc[common]
            change = post_m - pre_m
            t, p = ttest_rel(post_m, pre_m)
            s = sig_stars(p)
            log.append(f"  {roi:10s}: pre={pre_m.mean():+.6f}, post={post_m.mean():+.6f}, "
                        f"diff={change.mean():+.6f}, t={t:+.3f}, p={p:.4f} {s}  (N={len(common)})")
            rows_b.append(dict(metric=name, roi=roi, pre_mean=pre_m.mean(),
                               post_mean=post_m.mean(), diff=change.mean(),
                               t_stat=t, p_value=p, n=len(common)))

    tbl_b = pd.DataFrame(rows_b)
    tbl_b.to_csv(opj(RESULTS_DIR, "session_comparison.csv"), index=False)

    # ==================================================================
    # C) LME: replay_diff ~ session + (1|subject)
    # ==================================================================
    log.append("\n" + "=" * 70)
    log.append("C) LME: replay_diff ~ is_post + (1|subject)")
    log.append("=" * 70)

    rows_c = []
    for ret_col, ctrl_col, name in METRICS:
        log.append(f"\n--- {name} ---")

        # All ROIs combined
        sub = df.copy()
        sub["diff"] = sub[ret_col] - sub[ctrl_col]
        sub["is_post"] = (sub["session"] == "postsleep").astype(int)
        sub = sub.dropna(subset=["diff"])
        model = smf.mixedlm("diff ~ is_post", sub, groups=sub["subject"])
        result = model.fit(reml=True, method="powell")
        coef = result.params["is_post"]
        p = result.pvalues["is_post"]
        z = result.tvalues["is_post"]
        s = sig_stars(p)
        log.append(f"  All ROIs : coef={coef:+.6f}, z={z:+.3f}, p={p:.4f} {s}")
        rows_c.append(dict(metric=name, roi="all", coef=coef, z_stat=z, p_value=p))

        # Per ROI
        for roi in ROIS:
            rsub = sub[sub["roi"] == roi].copy()
            model = smf.mixedlm("diff ~ is_post", rsub, groups=rsub["subject"])
            result = model.fit(reml=True, method="powell")
            coef = result.params["is_post"]
            p = result.pvalues["is_post"]
            z = result.tvalues["is_post"]
            s = sig_stars(p)
            log.append(f"  {roi:10s}: coef={coef:+.6f}, z={z:+.3f}, p={p:.4f} {s}")
            rows_c.append(dict(metric=name, roi=roi, coef=coef, z_stat=z, p_value=p))

    tbl_c = pd.DataFrame(rows_c)
    tbl_c.to_csv(opj(RESULTS_DIR, "lme_session_effect.csv"), index=False)

    # Save log
    log_path = opj(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\n".join(log))


if __name__ == "__main__":
    main()
