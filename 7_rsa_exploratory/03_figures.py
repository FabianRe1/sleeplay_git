# -*- coding: utf-8 -*-
"""Figures for the binary-model RSA (reads outputs/02_binary_rsa/).

  tmap_<pairtype>.png        t-values of the regression betas, ROI x model, one panel per pair set
  timecourse_<pairtype>.png  beta per TR after the cue, key ROIs x models, lines = pair sets
  group_rdm_<roi>.png        group-mean z-scored dissimilarity in structure order

Usage:  python 03_figures.py <source>     (raw, lss, glmsingle, glmsingle_hrf)
"""

from __future__ import annotations

import os
import sys
from os.path import join as opj

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsa_config as cfg

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "raw"
IN_DIR = opj(cfg.OUT_ROOT, "02_binary_rsa", SOURCE)
OUT_DIR = opj(cfg.OUT_ROOT, "03_figures", SOURCE)
SETS = ["enc", "pre", "post1", "post2", "post1xpost2", "prexpost1", "prexpost2"]
WINDOW_TEXT = "TRs 4-6 after the cue" if SOURCE == "raw" else f"{SOURCE} betas"
SET_LABEL = {"pre": "evening", "post1": "morning pass 1", "post2": "morning pass 2",
             "post1xpost2": "morning 1 x morning 2", "prexpost1": "evening x morning 1",
             "prexpost2": "evening x morning 2", "enc": "encoding"}
MODELS = {"oo": ["object", "category", "position", "catseq", "scene", "context"],
          "ss": ["scene", "position", "context"],
          "os": ["scene", "position", "context"]}
PAIR_LABEL = {"oo": "object trials x object trials", "ss": "scene trials x scene trials",
              "os": "object trials x scene trials"}
KEY_ROIS = ["vis", "mtl", "hpc", "parietal_precuneus"]
SET_COLOR = {"pre": "#2a78d6", "post1": "#eb6834", "post2": "#1baf71"}   # blue / orange / green
INK, MUTED = "#222222", "#8a8a8a"

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.dpi": 150})


def tmaps(stats: pd.DataFrame, variant="main", tag="", what="unique effect of sharing a feature",
          cbar="t (beta vs 0)"):
    s = stats[(stats.window == "win") & (stats.variant == variant)]
    for ptype, models in MODELS.items():
        models = [m for m in models if (s.model == m).any()]
        sets = [x for x in SETS if ((s.pairtype == ptype) & (s.pair_set == x)).any()]
        fig, axes = plt.subplots(1, len(sets), figsize=(2.6 * len(sets) + 1.6, 0.33 * len(cfg.ROIS) + 1.6),
                                 sharey=True, squeeze=False)
        for ax, pset in zip(axes[0], sets):
            g = s[(s.pairtype == ptype) & (s.pair_set == pset)]
            t = g.pivot(index="roi", columns="model", values="t").reindex(index=cfg.ROIS, columns=models)
            q = g.pivot(index="roi", columns="model", values="p_fdr_rois").reindex(index=cfg.ROIS, columns=models)
            p = g.pivot(index="roi", columns="model", values="p").reindex(index=cfg.ROIS, columns=models)
            im = ax.imshow(t.values, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    v = t.values[i, j]
                    if np.isfinite(v):
                        star = "**" if q.values[i, j] < .05 else ("*" if p.values[i, j] < .05 else "")
                        ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=6,
                                color="white" if abs(v) > 4 else INK)
            ax.set_xticks(range(len(models)), models, rotation=45, ha="right")
            ax.set_yticks(range(len(cfg.ROIS)), cfg.ROIS)
            n = int(g.n.max()) if len(g) else 0
            ax.set_title(f"{SET_LABEL[pset]} (n={n})", fontsize=8)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.colorbar(im, ax=axes, shrink=0.7, label=cbar)
        fig.suptitle(f"{PAIR_LABEL[ptype]}: {what}, {WINDOW_TEXT}\n"
                     "positive = more similar when shared;  * p<.05 uncorrected, ** p<.05 FDR across ROIs",
                     fontsize=8, x=0.02, ha="left")
        fig.savefig(opj(OUT_DIR, f"tmap{tag}_{ptype}.png"), bbox_inches="tight")
        plt.close(fig)


def timecourses(betas: pd.DataFrame):
    b = betas[betas.window.str.startswith("tr") & betas.pair_set.isin(SET_COLOR)].copy()
    if b.empty:
        return
    b["tr"] = b.window.str[2:].astype(int)
    for ptype, models in MODELS.items():
        models = [m for m in models if m != "object"]
        fig, axes = plt.subplots(len(KEY_ROIS), len(models), sharex=True, sharey="row", squeeze=False,
                                 figsize=(2.2 * len(models) + 0.6, 1.6 * len(KEY_ROIS) + 0.8))
        for i, roi in enumerate(KEY_ROIS):
            for j, model in enumerate(models):
                ax = axes[i, j]
                ax.axhline(0, color=MUTED, lw=0.6)
                ax.axvspan(3.5, 6.5, color="#000000", alpha=0.05, lw=0)
                for pset, col in SET_COLOR.items():
                    g = b[(b.pairtype == ptype) & (b.roi == roi) & (b.model == model) & (b.pair_set == pset)]
                    m = g.groupby("tr").beta.agg(["mean", "sem"])
                    ax.fill_between(m.index * cfg.TR, m["mean"] - m["sem"], m["mean"] + m["sem"],
                                    color=col, alpha=0.18, lw=0)
                    ax.plot(m.index * cfg.TR, m["mean"], color=col, lw=1.6, label=SET_LABEL[pset])
                if i == 0:
                    ax.set_title(f"same {model}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{roi}\nbeta")
                if i == len(KEY_ROIS) - 1:
                    ax.set_xlabel("time after cue (s)")
        axes[0, -1].legend(frameon=False, fontsize=7, loc="upper right")
        fig.suptitle(f"{PAIR_LABEL[ptype]}: beta per TR (mean, SEM band); grey = main window",
                     fontsize=8, x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(opj(OUT_DIR, f"timecourse_{ptype}.png"), bbox_inches="tight")
        plt.close(fig)


def group_rdms(rdms):
    for roi in KEY_ROIS:
        sets = [x for x in ("pre", "post1", "post1xpost2") if f"{roi}__{x}" in rdms]
        fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets) + 0.8, 3.9), squeeze=False)
        for ax, pset in zip(axes[0], sets):
            im = ax.imshow(rdms[f"{roi}__{pset}"], cmap="RdBu_r", vmin=-0.4, vmax=0.4, interpolation="nearest")
            for k in list(range(5, 50, 5)) + [50, 60]:
                lw = 1.0 if k in (25, 50, 60) else 0.3
                ax.axhline(k - 0.5, color=INK, lw=lw)
                ax.axvline(k - 0.5, color=INK, lw=lw)
            ticks, labels = [12, 37, 55, 65], ["objects\nday", "objects\nnight", "scenes\nday", "scenes\nnight"]
            ax.set_xticks(ticks, [l.replace('\n', ' ') for l in labels], rotation=45, ha='right')
            ax.set_yticks(ticks, labels)
            ax.tick_params(length=0)
            ax.set_title(SET_LABEL[pset], fontsize=8)
        fig.colorbar(im, ax=axes, shrink=0.7, label="z-scored 1 - r (blue = more similar)")
        fig.suptitle(f"{roi}: group-mean dissimilarity, trials ordered context > scene > position "
                     "(thin lines = scenes); between-run pairs only", fontsize=8, x=0.02, ha="left")
        fig.savefig(opj(OUT_DIR, f"group_rdm_{roi}.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = pd.read_csv(opj(IN_DIR, "group_beta_vs_zero.csv"))
    tmaps(stats)                                           # all predictors in one model
    if (stats.variant == "four").any():
        tmaps(stats, "four", "_four_joint",
              "context, scene, position, category fitted together (unique effects)")
        tmaps(pd.read_csv(opj(IN_DIR, "group_alone_vs_zero.csv")), "four", "_four_individual",
              "each predictor fitted on its own", "t (different - same vs 0)")
    timecourses(pd.read_csv(opj(IN_DIR, "subject_betas.csv")))
    group_rdms(np.load(opj(IN_DIR, "group_mean_rdms.npz")))
    print("figures in", OUT_DIR)


if __name__ == "__main__":
    main()
