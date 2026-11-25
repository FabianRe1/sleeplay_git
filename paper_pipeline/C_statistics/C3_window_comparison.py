#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 - Comparison of metric versions.

Repeats the three main analyses (detection, sleep effect, brain-behaviour
relation) for six versions of the replay metric (full or position-truncated
window, slope or sinusoid amplitude) and shows side by side which results
hold under which version.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
        outputs/B9_sinusoid_summary/thesis/<roi>/<roi>_windowed_fit_results_<session>.csv
        outputs/B9_sinusoid_summary/trunc_tapered/<roi>/<roi>_windowed_fit_results_<session>.csv
Output  outputs/C3_window_comparison/*.csv
Plots   plots/C3_window_comparison/*.pdf

Usage:  python C3_window_comparison.py

Ported from 4_decoding_claude_retrieval/5.4_window_comparison.py; logic unchanged.
The description below is the original one. Its step numbers map to this
pipeline as 4.4 = B7, 4.6 = B8, 4.7 = B9, 5.1 = B10, 5.2 = C1, 5.3 = C2,
5.4 = C3, and its folder names are replaced by the Input/Output lines here.

--- original description ---

Full vs truncated SODA window — which results hold under which version.

Recomputes the three thesis analyses (5.2) for six versions of the replay
metric and shows them side by side:

    |SODA| full     : slope over all 5 elements, every trial          (thesis)
    |SODA| full p>=3: same, only trials retrieving position 3, 4 or 5
    |SODA| trunc    : slope over elements 1..p, p >= 3
    SinAmp full     : thesis tapered sinusoid on the full-window slope (thesis)
    SinAmp full p>=3: same, positions 3-5 only
    SinAmp trunc    : same tapered model on the truncated slope (p >= 3)

"full" vs "full p>=3" isolates the trial selection; "full p>=3" vs "trunc"
isolates the window (same trials). All metrics are absolute (|slope| per TR
averaged over the trial; |amplitude|) and contrasted retrieved − non-retrieved.

Inputs
------
4.6_SODA_based_on_singlesubj_4.4_<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
4.7_sinus_1cycle/<roi>/<roi>_windowed_fit_results_<session>.csv                (preset "thesis")
4.7_sinus_1cycle_trunc_tapered/<roi>/<roi>_windowed_fit_results_<session>.csv  (preset "trunc_tapered")

Analyses (same definitions as 5.2)
----------------------------------
detection : one-sample t on subject means of (retrieved − control); all trials
sleep     : objects; per-ROI LME diff ~ is_post + (1|subject), all post trials and
            trial-matched (post-sleep pass 1); omnibus
            replay ~ is_retrieved * is_post * C(roi) with ROIs vis, mtl, X (X vs MTL)
brain-beh.: objects; Pearson and Spearman r(subject mean diff, accuracy)

Outputs: 5.4_window_comparison/  (CSVs + heatmap PDFs)

Usage
-----
    python 5.4_window_comparison.py
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, pearsonr, spearmanr

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

OUT_DIR = str(step_output_dir(__file__))
PLOT_DIR = str(step_plot_dir(__file__))
SODA_ROOT = str(step_input_dir("B8_sequentiality_soda"))
SINUS_ROOT = str(step_input_dir("B9_sinusoid_summary"))

ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl"]
ROI_LABELS = {"vis": "Visual", "cortical_memory": "Cort. memory", "parietal_precuneus": "Par.+precun.",
              "parietal": "Parietal", "precuneus": "Precuneus", "mtl": "MTL"}
POST1_MAX_TRIAL = 70

# version -> (retrieved column, control column, positions 3-5 only?)
VERSIONS = {
    "|SODA| full":      ("soda_full_ret",  "soda_full_ctl",  False),
    "|SODA| full p>=3": ("soda_full_ret",  "soda_full_ctl",  True),
    "|SODA| trunc":     ("soda_trunc_ret", "soda_trunc_ctl", True),
    "SinAmp full":      ("sin_full_ret",   "sin_full_ctl",   False),
    "SinAmp full p>=3": ("sin_full_ret",   "sin_full_ctl",   True),
    "SinAmp trunc":     ("sin_trunc_ret",  "sin_trunc_ctl",  True),
}
KEY = ["subject", "trial", "object_scene_retrieval"]


# =====================================================================
# DATA
# =====================================================================

def _sin(folder, roi, session):
    path = opj(SINUS_ROOT, folder, roi, f"{roi}_windowed_fit_results_{session}.csv")
    d = pd.read_csv(path, usecols=lambda c: c in KEY + ["retrieved_w_amplitude", "control_w_amplitude"])
    d["retrieved_w_amplitude"] = d["retrieved_w_amplitude"].abs()
    d["control_w_amplitude"] = d["control_w_amplitude"].abs()
    return d


def load(roi):
    parts = []
    for session in ["presleep", "postsleep"]:
        s = pd.read_csv(opj(SODA_ROOT, session, "SODA_results_csv",
                            f"{roi}_soda_results_per_combination.csv"),
                        usecols=KEY + ["retrieved_position", "behav_correct", "retrieved_soda_abs",
                                       "control_soda_abs", "retrieved_soda_trunc_abs", "control_soda_trunc_abs"],
                        low_memory=False)
        s = s.groupby(KEY + ["retrieved_position"], as_index=False).agg(
            behav_correct=("behav_correct", "first"),
            soda_full_ret=("retrieved_soda_abs", "mean"), soda_full_ctl=("control_soda_abs", "mean"),
            soda_trunc_ret=("retrieved_soda_trunc_abs", "mean"), soda_trunc_ctl=("control_soda_trunc_abs", "mean"))
        f = _sin("thesis", roi, session).rename(
            columns={"retrieved_w_amplitude": "sin_full_ret", "control_w_amplitude": "sin_full_ctl"})
        t = _sin("trunc_tapered", roi, session).rename(
            columns={"retrieved_w_amplitude": "sin_trunc_ret", "control_w_amplitude": "sin_trunc_ctl"})
        s = s.merge(f, on=KEY, how="left").merge(t, on=KEY, how="left")
        s["session"] = session
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    d["behav_correct"] = pd.to_numeric(d["behav_correct"].replace({"True": 1, "False": 0, True: 1, False: 0}),
                                       errors="coerce")
    d["cond"] = np.where(d["session"] == "presleep", "pre",
                         np.where(d["trial"] <= POST1_MAX_TRIAL, "post1", "post2"))
    d["roi"] = roi
    return d


def version_frame(d, version, objects_only=False):
    ret, ctl, p3 = VERSIONS[version]
    x = d.copy()
    if p3:
        x = x[x["retrieved_position"] >= 3]
    if objects_only:
        x = x[x["object_scene_retrieval"] == "object"]
    x = x.assign(ret=x[ret], ctl=x[ctl], diff=x[ret] - x[ctl])
    return x.dropna(subset=["diff"])


# =====================================================================
# ANALYSES
# =====================================================================

def detection(data):
    rows = []
    conds = {"collapsed": ["pre", "post1", "post2"], "pre": ["pre"], "post": ["post1", "post2"],
             "post1": ["post1"], "post2": ["post2"]}
    for roi, d in data.items():
        for v in VERSIONS:
            x = version_frame(d, v)
            for c, cs in conds.items():
                m = x[x["cond"].isin(cs)].groupby("subject")["diff"].mean()
                t, p = ttest_1samp(m, 0)
                rows.append(dict(roi=roi, version=v, cond=c, t=t, p=p, n=len(m)))
    return pd.DataFrame(rows)


def _lme(df, formula):
    return smf.mixedlm(formula, df, groups=df["subject"]).fit(reml=True, method="powell")


def sleep(data):
    rows = []
    for v in VERSIONS:
        for post in ["all post", "matched"]:
            frames = {}
            for roi, d in data.items():
                x = version_frame(d, v, objects_only=True)
                if post == "matched":
                    x = x[x["cond"].isin(["pre", "post1"])]
                x = x.assign(is_post=(x["session"] == "postsleep").astype(int))
                frames[roi] = x
                r = _lme(x, "diff ~ is_post")
                rows.append(dict(version=v, post=post, test="per-ROI LME", roi=roi,
                                 z=r.tvalues["is_post"], p=r.pvalues["is_post"]))
            for X in [r for r in data if r not in ("vis", "mtl")] + ["vis"]:
                parts = []
                for roi in dict.fromkeys(["vis", "mtl", X]):
                    x = frames[roi]
                    for col, flag in [("ret", 1), ("ctl", 0)]:
                        q = x[["subject", "is_post", col]].rename(columns={col: "y"})
                        q["is_retrieved"], q["roi"] = flag, roi
                        parts.append(q)
                L = pd.concat(parts)
                L["roi"] = pd.Categorical(L["roi"], categories=list(dict.fromkeys(["mtl", "vis", X])))
                r = _lme(L, "y ~ is_retrieved * is_post * C(roi)")
                k = [i for i in r.params.index if "is_retrieved:is_post:C(roi)" in i and f"[T.{X}]" in i][0]
                rows.append(dict(version=v, post=post, test="3-way vs MTL", roi=X, z=r.tvalues[k], p=r.pvalues[k]))
    return pd.DataFrame(rows)


def brain_behaviour(data):
    rows = []
    for roi, d in data.items():
        for v in VERSIONS:
            x = version_frame(d, v, objects_only=True)
            for c, cs in {"collapsed": ["pre", "post1", "post2"], "pre": ["pre"], "post": ["post1", "post2"]}.items():
                s = x[x["cond"].isin(cs)].groupby("subject").agg(r=("diff", "mean"), acc=("behav_correct", "mean")).dropna()
                r, p = pearsonr(s["r"], s["acc"])
                rho, prho = spearmanr(s["r"], s["acc"])
                rows.append(dict(roi=roi, version=v, cond=c, r=r, p=p, rho=rho, p_rho=prho, n=len(s)))
    return pd.DataFrame(rows)


# =====================================================================
# FIGURES
# =====================================================================

def _heat(ax, M, P, vmax, fmt, cmap="RdBu_r"):
    im = ax.imshow(M.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val, p = M.values[i, j], P.values[i, j]
            if np.isfinite(val):
                star = "**" if p < .01 else "*" if p < .05 else ""
                ax.text(j, i, f"{val:{fmt}}{star}", ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if p < .05 else "normal",
                        color="white" if abs(val) > .6 * vmax else "black")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=6.5, rotation=45, ha="right")
    return im


def fig_detection(det):
    conds = ["pre", "post1", "post2", "collapsed"]
    cols = [(r, c) for r in ROIS for c in conds]
    M = det.pivot_table(index="version", columns=["roi", "cond"], values="t").reindex(index=list(VERSIONS), columns=cols)
    P = det.pivot_table(index="version", columns=["roi", "cond"], values="p").reindex(index=list(VERSIONS), columns=cols)
    M.columns = P.columns = [f"{ROI_LABELS[r]} · {c}" for r, c in cols]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    im = _heat(ax, M, P, vmax=5, fmt="+.1f")
    for k in range(1, len(ROIS)):
        ax.axvline(k * len(conds) - .5, color="k", lw=1)
    ax.axhline(2.5, color="k", lw=1)
    fig.colorbar(im, ax=ax, label="t (retrieved − control > 0)", fraction=.02)
    ax.set_title("Detection: retrieved > non-retrieved sequence, all trials (bold/* p < .05)")
    fig.savefig(opj(PLOT_DIR, "1_detection_heatmap.pdf"))
    plt.close(fig)


def fig_sleep(sl):
    rois_order = ["parietal_precuneus", "precuneus", "parietal", "cortical_memory", "vis", "mtl"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [6, 4]})
    for ax, test, rois in [(axes[0], "per-ROI LME", rois_order),
                           (axes[1], "3-way vs MTL", [r for r in rois_order if r != "mtl"])]:
        x = sl[sl["test"] == test]
        cols = [(r, p) for r in rois for p in ["all post", "matched"]]
        M = x.pivot_table(index="version", columns=["roi", "post"], values="z").reindex(index=list(VERSIONS), columns=cols)
        P = x.pivot_table(index="version", columns=["roi", "post"], values="p").reindex(index=list(VERSIONS), columns=cols)
        M.columns = P.columns = [f"{ROI_LABELS[r]} · {p}" for r, p in cols]
        im = _heat(ax, M, P, vmax=4, fmt="+.1f")
        ax.axhline(2.5, color="k", lw=1)
        ax.set_title(f"Sleep effect (objects): {test}  — z, positive = post > pre", fontsize=8)
        if ax is axes[1]:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes, label="z", fraction=.02)
    fig.savefig(opj(PLOT_DIR, "2_sleep_effect_heatmap.pdf"))
    plt.close(fig)


def fig_brain_behaviour(bb):
    for stat, pcol, name in [("r", "p", "Pearson"), ("rho", "p_rho", "Spearman")]:
        cols = [(r, c) for r in ROIS for c in ["pre", "post", "collapsed"]]
        M = bb.pivot_table(index="version", columns=["roi", "cond"], values=stat).reindex(index=list(VERSIONS), columns=cols)
        P = bb.pivot_table(index="version", columns=["roi", "cond"], values=pcol).reindex(index=list(VERSIONS), columns=cols)
        M.columns = P.columns = [f"{ROI_LABELS[r]} · {c}" for r, c in cols]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        im = _heat(ax, M, P, vmax=.7, fmt="+.2f")
        for k in range(1, len(ROIS)):
            ax.axvline(k * 3 - .5, color="k", lw=1)
        ax.axhline(2.5, color="k", lw=1)
        fig.colorbar(im, ax=ax, label=f"{name} r", fraction=.02)
        ax.set_title(f"Brain–behaviour (objects): {name} r(retrieved − control, accuracy)")
        fig.savefig(opj(PLOT_DIR, f"3_brain_behaviour_heatmap_{name.lower()}.pdf"))
        plt.close(fig)


def fig_parietal_precuneus(data):
    """The version dependence of the thesis sleep effect, in one ROI."""
    d = data["parietal_precuneus"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
    colors = {"pre": "#7BAFD4", "post1": "#D4534E", "post2": "#E8A09C"}
    for ax, fam in zip(axes, ["|SODA|", "SinAmp"]):
        vs = [v for v in VERSIONS if v.startswith(fam)]
        for i, v in enumerate(vs):
            x = version_frame(d, v, objects_only=True)
            for j, c in enumerate(["pre", "post1", "post2"]):
                m = x[x["cond"] == c].groupby("subject")["diff"].mean()
                scale = x.groupby("subject")["ctl"].mean().mean()      # express as % of control
                y, e = 100 * m.mean() / scale, 100 * m.std(ddof=1) / np.sqrt(len(m)) / scale
                ax.bar(i + (j - 1) * .26, y, .26, yerr=e, color=colors[c], capsize=2,
                       label=c if i == 0 else None, edgecolor="0.3")
                t, p = ttest_1samp(m, 0)
                if p < .05:
                    ax.text(i + (j - 1) * .26, y + e + .5, "*", ha="center")
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([v.replace(fam + " ", "") for v in vs])
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"Parietal + precuneus, objects — {fam}")
        ax.set_ylabel("retrieved − control [% of control]")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(opj(PLOT_DIR, "4_parietal_precuneus_by_version.pdf"))
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main():
    mpl.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading...")
    data = {roi: load(roi) for roi in ROIS}
    missing = {roi: int(d["sin_trunc_ret"].notna().sum()) for roi, d in data.items()}
    print("  truncated SinAmp trials per ROI:", missing)

    print("Detection");        det = detection(data);         det.to_csv(opj(OUT_DIR, "1_detection.csv"), index=False)
    print("Sleep effect");     sl = sleep(data);              sl.to_csv(opj(OUT_DIR, "2_sleep_effect.csv"), index=False)
    print("Brain-behaviour");  bb = brain_behaviour(data);    bb.to_csv(opj(OUT_DIR, "3_brain_behaviour.csv"), index=False)
    print("Figures")
    fig_detection(det)
    fig_sleep(sl)
    fig_brain_behaviour(bb)
    fig_parietal_precuneus(data)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
