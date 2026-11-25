#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D4 - Figures of the similarity analyses (retrieval only).

fig1  the four model matrices and the measured dissimilarity (parietal cortex, morning pass 1)
fig2  the four models fitted alone and together, eight regions, mean of the three sessions
fig3  the four models per session (visual, parietal, cortical memory)
fig4  scene effect when the scene picture is shown in both trials (object x object)
      and when it is retrieved but not shown in one of them (object x scene)
fig5  dissimilarity by distance between positions, same scene and different scenes
fig6  extra neighbour advantage within a scene, evening and morning

Error bars are 95 % confidence intervals across participants.

Input   outputs/D2_binary_rsa/subject_betas.csv, group_mean_rdms.npz
        outputs/D3_position_and_neighbours/A_subject_distance_curve.csv,
                                           C_subject_neighbour_scene.csv
Plots   plots/D4_figures/fig<N>_<name>.png (300 dpi) and .pdf

Usage:  python D4_figures.py

Not a port of the original pipeline: added 2026-11-19, ported from
7_rsa_exploratory/08_paper_figures.py (its figures 1-5 and 7; figure 7 there
compared three estimators and shows the single estimator of this pipeline here,
as fig6). Figures 6 and 8 of that script need other estimators and are not ported.
"""

from __future__ import annotations

import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_plot_dir

OUT = str(step_plot_dir(__file__))
D2 = str(step_input_dir("D2_binary_rsa"))
D3 = str(step_input_dir("D3_position_and_neighbours"))

ROI = {"vis": "Visual", "parietal": "Parietal", "precuneus": "Precuneus",
       "cortical_memory": "Cortical memory", "superior_frontal": "Superior frontal",
       "rsc": "Retrosplenial", "posterior_cingulate": "Posterior cingulate", "mtl": "Medial temporal"}
MODELS = {"context": "Same context", "scene": "Same scene",
          "position": "Same position", "category": "Same category"}
SESS = {"pre": "Evening", "post1": "Morning 1", "post2": "Morning 2"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a07c", "#b8431a"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f4f1", "#86b6ef", "#184f95"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), stats.sem(v) * stats.t.ppf(0.975, len(v) - 1), len(v)


def save(fig, name):
    fig.savefig(opj(OUT, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(opj(OUT, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=MUTED, lw=0.6, zorder=0)


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="right", va="bottom")


def betas():
    b = pd.read_csv(opj(D2, "subject_betas.csv"))
    return b[(b.window == "win") & b.roi.isin(ROI)]


def fig1_models():
    i = np.arange(50)
    ctx, scene, pos = i // 25, i // 5, i % 5
    seq = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])[scene]            # illustrative assignment
    cat = seq * 5 + pos
    rd = {"Same context": ctx[:, None] != ctx[None], "Same scene": scene[:, None] != scene[None],
          "Same position": pos[:, None] != pos[None], "Same category": cat[:, None] != cat[None]}
    z = np.load(opj(D2, "group_mean_rdms.npz"))
    emp = z["parietal__post1"][:50, :50].copy()
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.12, 2.15, 0.08], wspace=0.25, hspace=0.35)
    for k, (name, m) in enumerate(rd.items()):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(m.astype(float), cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontweight="normal")
        ax.set_xticks([]); ax.set_yticks([12, 37], ["Day", "Night"] if k % 2 == 0 else ["", ""])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(24.5, color="white", lw=0.8); ax.axvline(24.5, color="white", lw=0.8)
        if k == 0:
            panel(ax, "a")
    ax = fig.add_subplot(gs[:, 3])
    v = np.nanpercentile(np.abs(emp), 98)
    im = ax.imshow(emp, cmap=DIV, vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title("Measured dissimilarity\nparietal cortex, morning pass 1", fontweight="normal")
    ax.set_xticks(np.arange(2, 50, 5), [f"{s + 1}" for s in range(5)] * 2)
    ax.set_yticks([12, 37], ["Day", "Night"]); ax.tick_params(length=0)
    ax.set_xlabel("Scene (five objects each), day then night")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(24.5, color=INK, lw=0.6); ax.axvline(24.5, color=INK, lw=0.6)
    panel(ax, "b")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 4]))
    cb.set_label("Dissimilarity (z)"); cb.outline.set_visible(False)
    fig.text(0.125, 0.02, "Light = same (predicted similar), dark = different", fontsize=7.5, color=MUTED)
    save(fig, "fig1_models_and_data")


def fig2_models():
    b = betas()
    b = b[(b.variant == "four") & (b.pairtype == "oo") & b.pair_set.isin(SESS) & b.model.isin(MODELS)]
    s = b.groupby(["subject", "roi", "model"], as_index=False)[["alone", "beta"]].mean()
    rois = list(ROI)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True, sharex=True)
    for ax, (m, title) in zip(axes, MODELS.items()):
        for off, col, mfc, lab in ((-0.16, "alone", "white", "Fitted alone"), (0.16, "beta", BLUE, "Fitted together")):
            for k, r in enumerate(rois):
                mu, h, _ = ci(s[(s.model == m) & (s.roi == r)][col])
                ax.errorbar(mu, k + off, xerr=h, fmt="o", ms=4.2, color=BLUE, mfc=mfc, mew=1.1, lw=1.1,
                            capsize=0, label=lab if k == 0 else None)
        zero(ax, False)
        ax.set_title(title)
        ax.set_yticks(range(len(rois)), [ROI[r] for r in rois])
        ax.invert_yaxis() if ax is axes[0] else None
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.9, -0.33), ncol=2, handletextpad=0.3)
    fig.supxlabel("Effect on similarity (z units; positive = more similar when shared)", fontsize=8, y=-0.02)
    save(fig, "fig2_four_models")


def fig3_sessions():
    b = betas()
    b = b[(b.variant == "four") & (b.pairtype == "oo") & b.pair_set.isin(SESS) & b.model.isin(MODELS)]
    rois = {"vis": BLUE, "parietal": ORANGE, "cortical_memory": AQUA}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
    for ax, (m, title) in zip(axes, MODELS.items()):
        for j, (r, c) in enumerate(rois.items()):
            mu, hh = [], []
            for ps in SESS:
                a, h, _ = ci(b[(b.model == m) & (b.roi == r) & (b.pair_set == ps)].beta)
                mu.append(a); hh.append(h)
            x = np.arange(3) + (j - 1) * 0.12
            ax.errorbar(x, mu, yerr=hh, fmt="-o", ms=4, lw=1.6, color=c, mec="white", mew=0.8,
                        elinewidth=1, label=ROI[r])
        zero(ax)
        ax.set_title(title)
        ax.set_xticks(range(3), ["Evening", "Morning\npass 1", "Morning\npass 2"])
        ax.set_xlim(-0.4, 2.4)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Unique effect on similarity (z)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.12), handlelength=1.4)
    save(fig, "fig3_sessions")


def fig4_cue():
    b = betas()
    b = b[(b.variant == "four") & b.pair_set.isin(SESS) & (b.model == "scene") & b.pairtype.isin(["oo", "os"])]
    s = b.groupby(["subject", "roi", "pairtype"], as_index=False).beta.mean()
    rois = list(ROI)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for off, pt, c, lab in ((-0.17, "oo", BLUE, "Object trial × object trial (scene picture shown in both)"),
                            (0.17, "os", ORANGE, "Object trial × scene trial (scene retrieved, not shown)")):
        for k, r in enumerate(rois):
            mu, h, _ = ci(s[(s.roi == r) & (s.pairtype == pt)].beta)
            ax.errorbar(k + off, mu, yerr=h, fmt="o", ms=5, color=c, mec="white", mew=0.8, lw=1.3,
                        label=lab if k == 0 else None)
    zero(ax)
    ax.set_xticks(range(len(rois)), [ROI[r].replace(" ", "\n") for r in rois])
    ax.set_ylabel("Same-scene effect on similarity (z)")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", handletextpad=0.2)
    save(fig, "fig4_scene_cue_control")


def fig5_distance():
    c = pd.read_csv(opj(D3, "A_subject_distance_curve.csv"))
    c = c[c.pair_set.isin(SESS) & c["where"].isin(["within_scene", "across_scenes"])]
    s = c.groupby(["subject", "roi", "where", "distance"], as_index=False).z_dissim.mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, r in zip(axes, ["vis", "parietal", "cortical_memory"]):
        for where, col, lab in (("across_scenes", BLUE, "Different scenes"), ("within_scene", ORANGE, "Same scene")):
            g = s[(s.roi == r) & (s["where"] == where)]
            d = sorted(g.distance.unique())
            st = [ci(g[g.distance == k].z_dissim) for k in d]
            ax.errorbar(d, [a[0] for a in st], yerr=[a[1] for a in st], fmt="-o", ms=4.5, lw=1.8, color=col,
                        mec="white", mew=0.8, elinewidth=1, label=lab)
        zero(ax)
        ax.set_title(ROI[r]); ax.set_xticks(range(5)); ax.set_xlabel("Distance between positions")
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Dissimilarity (z; lower = more similar)")
    axes[0].legend(loc="upper left")
    save(fig, "fig5_position_distance")


def fig6_within_scene():
    rois = ["parietal", "rsc", "cortical_memory", "superior_frontal", "vis", "precuneus"]
    d = pd.read_csv(opj(D3, "C_subject_neighbour_scene.csv"))
    d["session"] = d.pair_set.map({"pre": "Evening", "post1": "Morning", "post2": "Morning"})
    s = d.dropna(subset=["session"]).groupby(["subject", "roi", "session"], as_index=False)["diff"].mean()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for off, ses, c in ((-0.17, "Evening", BLUE), (0.17, "Morning", ORANGE)):
        for k, r in enumerate(rois):
            mu, h, _ = ci(s[(s.roi == r) & (s.session == ses)]["diff"])
            ax.errorbar(mu, k + off, xerr=h, fmt="o", ms=4.8, color=c, mec="white", mew=0.8, lw=1.3,
                        label=ses if k == 0 else None)
    zero(ax, False)
    ax.set_yticks(range(len(rois)), [ROI[r] for r in rois])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_xlabel("Extra neighbour advantage within a scene (z)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", ncol=2, bbox_to_anchor=(0.0, -0.1))
    save(fig, "fig6_within_scene_neighbours")


if __name__ == "__main__":
    for f in (fig1_models, fig2_models, fig3_sessions, fig4_cue, fig5_distance, fig6_within_scene):
        f()
