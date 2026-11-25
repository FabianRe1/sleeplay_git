#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C7 - Which reactivation measures relate to memory performance?

Relates every reactivation measure of C5 and C6 to how well objects were
retrieved, in the four main ROIs (vis, parietal_precuneus, mtl and the control
region caudal_acc), object trials only.

Measures (each: retrieved sequence minus control sequence, per trial)
    retrieved item            evidence for the retrieved item             (TRs 3-8)
    predecessor / successor   evidence for the item before / after it     (TRs 3-8, positions 2-4)
    predecessor - successor                                               (TRs 3-8, positions 2-4)
    sequence evidence         mean evidence of the five elements          (TRs 3-8)
    |slope| five items        absolute SODA slope, all TRs (thesis metric)
    |slope| up to retrieved   absolute slope over elements 1..p, p >= 3, all TRs
    sinusoid five items       fitted sinusoid amplitude (thesis SinAmp)
    sinusoid up to retrieved  the same for the slope up to the retrieved item

Analyses
    1  Between participants: measure (mean over a participant's object trials)
       against the participant's object-retrieval accuracy, in the same
       condition (pooled, pre, post1, post2, post). Pearson r, Spearman rho, and
       Pearson r without the participant with the lowest pooled accuracy
       (flagged in the original analysis as driving part of the correlation).
    2  Within participants: measure on correct trials minus measure on incorrect
       trials (participants with at least MIN_TRIALS of each), tested against zero.
       Sessions pooled and per session.
    3  Change: (post - pre) of the measure against (post - pre) of accuracy,
       between participants; also post1 - pre.
`p_fdr_main` corrects across the four main ROIs within a measure and condition.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
        outputs/B9_sinusoid_summary/{thesis,trunc_tapered}/<roi>/<roi>_windowed_fit_results_<session>.csv
Output  outputs/C7_brain_behaviour/*.csv
Plots   plots/C7_brain_behaviour/*.pdf and *.png

Usage:  python C7_brain_behaviour.py

Not a port: added on 2026-11-19. The original brain-behaviour analysis (C1,
section 7) covers the slope and the sinusoid amplitude over five items.
"""

import sys
import warnings
from os.path import join as opj
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

OUT_DIR = str(step_output_dir(__file__))
PLOT_DIR = str(step_plot_dir(__file__))
SODA_DIR = opj(str(step_input_dir("B8_sequentiality_soda")), "{session}", "SODA_results_csv")
SINUS_DIR = str(step_input_dir("B9_sinusoid_summary"))

MAIN_ROIS = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]
ROI_LABELS = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL",
              "caudal_acc": "Caudal ACC (control)"}
MEASURES = ["retrieved item", "predecessor", "successor", "predecessor - successor", "sequence evidence",
            "|slope| five items", "|slope| up to retrieved", "sinusoid five items", "sinusoid up to retrieved"]
SIGNAL_TRS = (3, 8)
POST1_MAX_TRIAL = 70
MIN_TRIALS = 5
CONDS = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post1": ["post1"], "post2": ["post2"],
         "post": ["post1", "post2"]}
SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
BLUE, MUTED, GRID = "#2a78d6", "#52514e", "#e4e3df"


def _fdr(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def load_trials(roi: str) -> pd.DataFrame:
    """One row per object trial with all measures."""
    parts = []
    for session in ["presleep", "postsleep"]:
        d = pd.read_csv(opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv"),
                        usecols=["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                                 "retrieved_probs", "control_probs", "behav_correct",
                                 "retrieved_soda_abs", "control_soda_abs",
                                 "retrieved_soda_trunc_abs", "control_soda_trunc_abs"], low_memory=False)
        d = d[d["object_scene_retrieval"] == "object"].reset_index(drop=True)
        E = 100.0 * (np.array([eval(x, SAFE_EVAL) for x in d["retrieved_probs"]], dtype=float)
                     - np.array([eval(x, SAFE_EVAL) for x in d["control_probs"]], dtype=float))
        p = d["retrieved_position"].astype(int).values
        k = np.arange(len(d))
        mid = np.isin(p, [2, 3, 4])
        row = pd.DataFrame(dict(subject=d["subject"].values, trial=d["trial"].values, tr=d["tr"].values,
                                correct=d["behav_correct"].astype(bool).values))
        win = (row["tr"] >= SIGNAL_TRS[0]) & (row["tr"] <= SIGNAL_TRS[1])
        row["retrieved item"] = np.where(win, E[k, p - 1], np.nan)
        row["predecessor"] = np.where(win & mid, E[k, np.clip(p - 2, 0, 4)], np.nan)
        row["successor"] = np.where(win & mid, E[k, np.clip(p, 0, 4)], np.nan)
        row["predecessor - successor"] = row["predecessor"] - row["successor"]
        row["sequence evidence"] = np.where(win, E.mean(1), np.nan)
        row["|slope| five items"] = 100.0 * (d["retrieved_soda_abs"].values - d["control_soda_abs"].values)
        row["|slope| up to retrieved"] = 100.0 * (d["retrieved_soda_trunc_abs"].values
                                                  - d["control_soda_trunc_abs"].values)
        tri = row.groupby(["subject", "trial"], as_index=False).mean(numeric_only=True).drop(columns="tr")
        tri["correct"] = tri["correct"] > 0.5
        tri["session"] = session
        for measure, preset in (("sinusoid five items", "thesis"), ("sinusoid up to retrieved", "trunc_tapered")):
            f = pd.read_csv(opj(SINUS_DIR, preset, roi, f"{roi}_windowed_fit_results_{session}.csv"),
                            usecols=["subject", "trial", "retrieved_w_amplitude", "control_w_amplitude"])
            f[measure] = f["retrieved_w_amplitude"].abs() - f["control_w_amplitude"].abs()
            tri = tri.merge(f[["subject", "trial", measure]], on=["subject", "trial"], how="left")
        tri["cond"] = np.where(session == "presleep", "pre",
                               np.where(tri["trial"] <= POST1_MAX_TRIAL, "post1", "post2"))
        parts.append(tri)
    x = pd.concat(parts, ignore_index=True)
    x["roi"] = roi
    return x


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 10:
        return dict(n=len(x), r=np.nan, p=np.nan, rho=np.nan, p_rho=np.nan)
    r, p = pearsonr(x, y)
    rho, pr = spearmanr(x, y)
    return dict(n=len(x), r=r, p=p, rho=rho, p_rho=pr)


def main():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300})
    print("Loading...")
    tri = pd.concat([load_trials(r) for r in MAIN_ROIS], ignore_index=True)

    # accuracy per participant and condition (the same for every ROI)
    one = tri[tri["roi"] == MAIN_ROIS[0]]
    acc = {c: one[one["cond"].isin(cs)].groupby("subject")["correct"].mean() for c, cs in CONDS.items()}
    lowest = acc["pooled"].idxmin()
    print(f"  {tri['subject'].nunique()} participants; lowest pooled accuracy: {lowest} "
          f"({acc['pooled'].min():.2f})")

    means = {c: tri[tri["cond"].isin(cs)].groupby(["roi", "subject"])[MEASURES].mean() for c, cs in CONDS.items()}

    # 1 between participants
    rows = []
    for c in CONDS:
        for roi in MAIN_ROIS:
            m = means[c].loc[roi]
            a = acc[c].reindex(m.index)
            for meas in MEASURES:
                res = _corr(m[meas].values, a.values)
                keep = m.index != lowest
                res2 = _corr(m[meas].values[keep], a.values[keep])
                rows.append(dict(cond=c, roi=roi, measure=meas, **res,
                                 r_without_lowest=res2["r"], p_without_lowest=res2["p"]))
    between = pd.DataFrame(rows)
    between["p_fdr_main"] = between.groupby(["cond", "measure"])["p"].transform(_fdr)

    # 2 within participants: correct minus incorrect
    rows, per_subj = [], []
    for c, cs in CONDS.items():
        d = tri[tri["cond"].isin(cs)]
        n = d[d["roi"] == MAIN_ROIS[0]].groupby(["subject", "correct"]).size().unstack(fill_value=0)
        ok = n[(n.get(True, 0) >= MIN_TRIALS) & (n.get(False, 0) >= MIN_TRIALS)].index
        g = d[d["subject"].isin(ok)].groupby(["roi", "subject", "correct"])[MEASURES].mean()
        diff = g.xs(True, level="correct") - g.xs(False, level="correct")
        if c == "pooled":
            per_subj = diff.reset_index()
        for roi in MAIN_ROIS:
            for meas in MEASURES:
                v = diff.loc[roi][meas].dropna().values
                t, p = ttest_1samp(v, 0) if len(v) >= 10 else (np.nan, np.nan)
                rows.append(dict(cond=c, roi=roi, measure=meas, n=len(v), mean_diff=np.mean(v) if len(v) else np.nan,
                                 t=t, p=p))
    within = pd.DataFrame(rows)
    within["p_fdr_main"] = within.groupby(["cond", "measure"])["p"].transform(_fdr)

    # 3 change against change
    rows = []
    for a_, b_ in (("post", "pre"), ("post1", "pre")):
        dacc = acc[a_] - acc[b_]
        for roi in MAIN_ROIS:
            dm = means[a_].loc[roi] - means[b_].loc[roi]
            for meas in MEASURES:
                rows.append(dict(contrast=f"{a_} - {b_}", roi=roi, measure=meas,
                                 **_corr(dm[meas].values, dacc.reindex(dm.index).values)))
    change = pd.DataFrame(rows)
    change["p_fdr_main"] = change.groupby(["contrast", "measure"])["p"].transform(_fdr)

    subj = means["pooled"].reset_index()
    subj["accuracy"] = subj["subject"].map(acc["pooled"])
    for name, t in {"1_between_participants": between, "2_within_participants_correct_vs_incorrect": within,
                    "3_change_vs_change": change, "subject_values_pooled": subj,
                    "subject_correct_minus_incorrect": per_subj}.items():
        t.to_csv(opj(OUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}.csv")

    # figure: scatter of every measure against accuracy, sessions pooled
    fig, axes = plt.subplots(len(MEASURES), len(MAIN_ROIS), figsize=(1.9 * len(MAIN_ROIS) + 0.9, 1.45 * len(MEASURES)),
                             sharex=True)
    for i, meas in enumerate(MEASURES):
        for j, roi in enumerate(MAIN_ROIS):
            ax = axes[i, j]
            g = subj[subj["roi"] == roi]
            ax.axhline(0, color=MUTED, lw=0.5, zorder=0)
            ax.scatter(g["accuracy"], g[meas], s=9, color=BLUE, alpha=0.7, lw=0)
            low = g[g["subject"] == lowest]
            ax.scatter(low["accuracy"], low[meas], s=16, facecolors="none", edgecolors=MUTED, lw=0.8)
            b = between[(between["cond"] == "pooled") & (between["roi"] == roi) & (between["measure"] == meas)].iloc[0]
            ax.text(0.03, 0.97, f"r = {b['r']:.2f}, rho = {b['rho']:.2f}", transform=ax.transAxes, va="top",
                    fontsize=6.5, color="#0b0b0b")
            if i == 0:
                ax.set_title(ROI_LABELS[roi])
            if j == 0:
                ax.set_ylabel(meas, fontsize=6.5)
            ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    fig.supxlabel("Object retrieval accuracy (sessions pooled); circle = lowest participant", fontsize=8, y=0.005)
    fig.savefig(opj(PLOT_DIR, "1_measures_vs_accuracy_pooled.pdf")); fig.savefig(opj(PLOT_DIR, "1_measures_vs_accuracy_pooled.png"))
    plt.close(fig)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
