#!/usr/bin/env python3
"""
Figures of the univariate retrieval results (reads the tables of 15_univariate_session.py).

fig9   activity and response time: change in activity per second of response time,
       per region, for events of a fixed 3 s and for events from cue to response
fig10  session differences per region (pass 1 - evening, pass 2 - pass 1), both event models
fig11  precuneus and hippocampus activity by response-time quartile, per session

Object trials. Error bars: 95 % confidence intervals across participants.
Usage:  python 17_univariate_figures.py
Output: outputs/08_paper_figures/fig9_..., fig10_..., fig11_... (.png, .pdf)
"""

import os
from os.path import join as opj

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import rsa_config as cfg

OUT = opj(cfg.OUT_ROOT, "08_paper_figures")
FIXED = opj(cfg.OUT_ROOT, "15_univariate_session")
RESP = opj(cfg.OUT_ROOT, "15_univariate_session_lss_rt")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REGIONS = {"PRECUNEUS": "Precuneus", "RSC": "Retrosplenial", "PARIETAL_PRECUNEUS": "Parietal + precuneus",
           "CAUDAL_ACC": "Caudal ACC", "PHPC": "Parahippocampal", "VIS": "Visual", "MTL": "MTL",
           "HPC": "Hippocampus", "AUD": "Auditory", "SOMATO": "Somatosensory"}
MODELS = ((FIXED, "Fixed 3 s events", BLUE), (RESP, "Events from cue to response", ORANGE))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left"})


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)


def dci(v):
    """Standardised mean (mean / SD) with the 95 % interval of the mean in the same units."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1)
    return v.mean() / sd, stats.t.ppf(0.975, len(v) - 1) / np.sqrt(len(v))


def save(fig, name):
    fig.savefig(opj(OUT, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(opj(OUT, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig9():
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    rois = list(REGIONS)
    for (folder, lab, col), off in zip(MODELS, (-0.17, 0.17)):
        d = pd.read_csv(opj(folder, "subject_rt_adjusted.csv"))
        d = d[d.kind == "object"]
        for k, r in enumerate(rois):
            mu, h = dci(d[d.roi == r].per_second_rt)
            ax.errorbar(mu, k + off, xerr=h, fmt="o", ms=4.8, color=col, mec="white", mew=0.8, lw=1.3,
                        label=lab if k == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(rois)), [REGIONS[r] for r in rois]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("Change in activity per second of response time (standardised)")
    ax.legend(loc="upper left")
    save(fig, "fig9_activity_and_response_time")


def fig10():
    rois = ["PRECUNEUS", "PARIETAL_PRECUNEUS", "HPC", "PHPC", "VIS", "CAUDAL_ACC"]
    cons = ["post1 - pre", "post2 - post1"]
    fig, axes = plt.subplots(1, len(rois), figsize=(1.95 * len(rois) + 0.9, 2.6), sharey=True)
    for (folder, lab, col), off in zip(MODELS, (-0.14, 0.14)):
        d = pd.read_csv(opj(folder, "subject_item_matched.csv"))
        d = d[(d.kind == "object") & (d.variant == "all trials")]
        for ax, r in zip(axes, rois):
            for k, c in enumerate(cons):
                mu, h = dci(d[d.contrast == c][r])
                ax.errorbar(k + off, mu, yerr=h, fmt="o", ms=4.8, color=col, mec="white", mew=0.8, lw=1.3,
                            label=lab if k == 0 else None)
    for ax, r in zip(axes, rois):
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        ax.set_title(REGIONS[r])
        ax.set_xticks(range(2), ["Pass 1\n− evening", "Pass 2\n− pass 1"], fontsize=7)
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Difference in activity, same items\n(standardised)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig10_session_differences")


def fig11():
    t = pd.read_csv(opj(FIXED, "trial_roi_means.csv"))
    t = t[(t.kind == "object") & t.rt.notna()].copy()
    t["q"] = t.groupby(["subject", "cond"]).rt.transform(lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    s = t.dropna(subset=["q"]).groupby(["subject", "cond", "q"], as_index=False)[["rt", "PRECUNEUS", "HPC"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    fig.subplots_adjust(wspace=0.3)
    for ax, r, title in zip(axes, ("PRECUNEUS", "HPC"), ("Precuneus", "Hippocampus")):
        for cond, lab, col in (("pre", "Evening", BLUE), ("post1", "Morning pass 1", ORANGE), ("post2", "Morning pass 2", AQUA)):
            g = s[s.cond == cond]
            x = [g[g.q == k].rt.mean() for k in range(4)]
            st = [ci(g[g.q == k][r]) for k in range(4)]
            ax.errorbar(x, [a[0] for a in st], yerr=[a[1] for a in st], fmt="-o", ms=4.2, lw=1.7, color=col,
                        mec="white", mew=0.8, elinewidth=1, label=lab)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Activity (percent signal change),\nfixed 3 s events")
    axes[0].legend(loc="lower right")
    fig.supxlabel("Response time (s); points are quartiles within participant and session", fontsize=8, y=-0.03)
    save(fig, "fig11_precuneus_by_response_time")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig9(); fig10(); fig11()
