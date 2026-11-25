#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E16 - The univariate figures of the paper, in the shared figure style
      (figure_style.py; same look and sizes as the decoding figures of C13).

No model is fitted here. The figures are drawn from the outputs of E4, E7, E9,
E14 and E15; the tables the text quotes stay those of E14 and E15.

  fig_univariate_1_wholebrain     group t maps of retrieval against baseline (E4, model ret_press,
                                  object trials), pooled and per time point, TFCE-corrected p < .05
  fig_univariate_2_time_points    retrieval activity per region at the three time points, object and
                                  scene trials (E7). Star: first morning against evening, object trials
  fig_univariate_3_learning       activity per region at the three showings during learning, whole
                                  sequence and objects only (E9). Star: second against first showing,
                                  whole sequence
  fig_univariate_4_across         change during learning (whole sequence, second minus first showing)
                                  against evening retrieval success and reaction time, one point per
                                  participant (E9, E14)
  fig_univariate_5_within         the same relation within participants, across each participant's
                                  sequences (E15)

Participants are shown as thin lines or dots, the group mean with the 95 %
confidence interval across participants. Stars are uncorrected paired t tests.

Input   outputs/E4_wholebrain_group/ret_press/object/
        outputs/E7_native_regions/participants_ret_press/
        outputs/E9_encoding_regions/participants/
        outputs/E14_region_summary/behaviour_per_participant.csv
        outputs/E15_learning_by_sequence/sequence_table.csv, group_within_participant.csv
Output  outputs/E16_paper_figures_univariate/figure_statistics.csv   (every test drawn in a figure)
Plots   plots/E16_paper_figures_univariate/fig_univariate_*.{png,pdf}
Usage:  python E16_paper_figures_univariate.py [1 2 3 4 5]

Not a port of the original pipeline: added 2026-11-25.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir
from figure_style import (BLUE, COND_LABEL, CONDS, GREY, GRID, INK, MUTED, ORANGE, WIDTH, apply_style, ci95, p_text,
                          save, stars, title)

OUT, PLOTS = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
LABELS = {"VIS": "Visual", "PARIETAL_PRECUNEUS": "Parietal + precuneus", "MTL": "MTL", "CAUDAL_ACC": "Caudal ACC (control)",
          "HPC": "Hippocampus", "PHPC": "Parahippocampal", "ERH": "Entorhinal", "PRECUNEUS": "Precuneus",
          "PARIETAL": "Inferior parietal", "RSC": "Retrosplenial", "POSTERIOR_CINGULATE": "Posterior cingulate"}
TIME_REGIONS = ["VIS", "PARIETAL_PRECUNEUS", "MTL", "CAUDAL_ACC", "HPC", "PHPC", "PRECUNEUS", "PARIETAL", "RSC",
                "POSTERIOR_CINGULATE", "ERH"]
LEARN_REGIONS = ["HPC", "PHPC", "ERH", "MTL", "VIS", "PRECUNEUS", "POSTERIOR_CINGULATE", "RSC", "PARIETAL", "CAUDAL_ACC"]
SCATTER_REGIONS = {"PHPC": "#2a78d6", "HPC": "#eb6834", "MTL": "#1baf7a", "PRECUNEUS": "#eda100", "VIS": "#e87ba4"}   # as E14, E15
MAP_TITLES = {"pooled": "All three time points", "pre": "Evening", "post1": "First morning", "post2": "Repeated morning"}
STATS = []


def _read(step, sub):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(str(step_input_dir(step, sub)), "*_region_estimates.csv")))])


def _mean_ci(ax, x, w, colour, label=None, lines=True):
    """Participants as thin lines, group mean with the 95 % confidence interval."""
    v = w.to_numpy(float)
    if lines:
        ax.plot(x, v.T, color=colour, lw=0.35, alpha=0.16, zorder=1)
    ax.errorbar(x, v.mean(0), yerr=ci95(v), color=colour, marker="o", ms=3.2, lw=1.4, capsize=1.8, capthick=0.8,
                elinewidth=0.9, zorder=4, label=label)


def _bracket(ax, x0, x1, p, y=0.93):
    """Star for a paired test between two positions, placed in axes height."""
    if p >= 0.05:
        return
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x0, x1, x1], [y - 0.015, y, y, y - 0.015], color=INK, lw=0.7, transform=tr, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.012, stars(p), ha="center", va="bottom", fontsize=8, transform=tr)


def _robust_ylim(ax, values, pad=0.12):
    """Limits from the 2nd to 98th percentile of the participant values, so that single extreme
    participants do not flatten the group course (their lines run out of the panel)."""
    lo, hi = np.nanpercentile(values, [2, 98])
    lo, hi = min(lo, 0), max(hi, 0)
    r = hi - lo
    ax.set_ylim(lo - pad * r, hi + (pad + 0.12) * r)


def figure_1():
    import nibabel as nib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn import plotting
    src = opj(str(step_input_dir("E4_wholebrain_group")), "ret_press", "object")
    rows = [c for c in MAP_TITLES if os.path.exists(opj(src, f"{c}_t_thresholded.nii.gz"))]
    vmax = 12
    fig = plt.figure(figsize=(WIDTH, 2.12 * len(rows) + 0.45))
    gs = fig.add_gridspec(2 * len(rows) + 1, 1, height_ratios=[1.0, 0.9] * len(rows) + [0.16], hspace=0.08,
                          left=0.01, right=0.99, top=0.975, bottom=0.035)
    for i, c in enumerate(rows):
        img = nib.load(opj(src, f"{c}_t_thresholded.nii.gz"))
        n = len(pd.read_csv(opj(src, f"{c}_participants.csv")))
        for j, (mode, cuts) in enumerate((("z", (-18, -8, 2, 12, 22, 32, 42, 52, 62)), ("x", (-50, -36, -22, -8, 0, 8, 22, 36, 50)))):
            ax = fig.add_subplot(gs[2 * i + j])
            disp = plotting.plot_stat_map(img, display_mode=mode, cut_coords=cuts, axes=ax, vmax=vmax, threshold=1e-6,
                                          cmap="RdBu_r", colorbar=False, annotate=False, draw_cross=False, black_bg=False)
            disp.annotate(size=6.5, left_right=(mode == "z"))
            if j == 0:
                ax.set_title(f"{MAP_TITLES[c]}  (n = {n})", loc="left", fontweight="bold", fontsize=8.5, color=INK, pad=1)
    cax = fig.add_axes([0.36, 0.022, 0.28, 0.009])
    cb = fig.colorbar(ScalarMappable(Normalize(-vmax, vmax), "RdBu_r"), cax=cax, orientation="horizontal", ticks=[-12, -6, 0, 6, 12])
    cb.outline.set_linewidth(0.5); cax.tick_params(length=2, labelsize=7)
    fig.text(0.35, 0.026, "below baseline", ha="right", va="center", fontsize=7.5, color=MUTED)
    fig.text(0.65, 0.026, "above baseline   (t)", ha="left", va="center", fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(opj(PLOTS, f"fig_univariate_1_wholebrain.{ext}"), dpi=400)
    plt.close(fig)


def figure_2():
    d = _read("E7_native_regions", "participants_ret_press")
    d = d[(d.model == "ret_press") & d.cond.isin(CONDS)]
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 6.5))
    for ax, reg in zip(axes.ravel(), TIME_REGIONS):
        both = []
        for kind, colour, dx in (("scene", ORANGE, 0.07), ("object", BLUE, -0.07)):
            w = d[(d.region == reg) & (d.regressor == kind)].pivot(index="subject", columns="cond", values="beta")[CONDS].dropna()
            _mean_ci(ax, np.arange(3) + dx, w, colour, label=f"{kind.capitalize()} trials")
            both.append(w.to_numpy())
            if kind == "object":
                r = stats.ttest_rel(w.post1, w.pre)
                STATS.append(dict(figure=2, region=reg, test="object trials, first morning minus evening", n=len(w),
                                  mean=(w.post1 - w.pre).mean(), t=r.statistic, p=r.pvalue))
                p_obj = r.pvalue
        _robust_ylim(ax, np.concatenate(both))
        _bracket(ax, -0.07, 0.93, p_obj)
        ax.axhline(0, color=INK, lw=0.5, zorder=0)
        ax.set_xticks(range(3)); ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=6.3); ax.set_xlim(-0.45, 2.45)
        title(ax, LABELS[reg]); ax.title.set_fontsize(8)
    ax = axes.ravel()[-1]
    ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="center left", title=f"n = {len(w)}", alignment="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Signal change\nagainst baseline (%)")
    for ax in axes[:-1].ravel()[:-1]:
        ax.set_xticklabels([])
    for ax in (axes[1, 3],):
        ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=6.3)
    fig.tight_layout(h_pad=1.0, w_pad=1.6)
    save(fig, opj(PLOTS, "fig_univariate_2_time_points"))


def figure_3():
    d = _read("E9_encoding_regions", "participants")
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 3.9), sharex=True)
    for ax, reg in zip(axes.ravel(), LEARN_REGIONS):
        both = []
        for what, model, pref, colour, dx, lines in (("Objects only", "events", "object_rep", MUTED, 0.07, False),
                                                     ("Whole sequence", "block", "sequence_rep", BLUE, -0.07, True)):
            w = d[(d.model == model) & d.regressor.str.startswith(pref) & (d.region == reg)].pivot(
                index="subject", columns="regressor", values="beta")[[f"{pref}{i}" for i in (1, 2, 3)]].dropna()
            _mean_ci(ax, np.arange(3) + dx, w, colour, label=what, lines=lines)
            if lines:
                both.append(w.to_numpy())
                r = stats.ttest_rel(w.iloc[:, 1], w.iloc[:, 0])
                STATS.append(dict(figure=3, region=reg, test="whole sequence, second minus first showing", n=len(w),
                                  mean=(w.iloc[:, 1] - w.iloc[:, 0]).mean(), t=r.statistic, p=r.pvalue))
                p_whole = r.pvalue
        _robust_ylim(ax, np.concatenate(both))
        _bracket(ax, -0.07, 0.93, p_whole)
        ax.axhline(0, color=INK, lw=0.5, zorder=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(["1st", "2nd", "3rd"]); ax.set_xlim(-0.4, 2.4)
        title(ax, LABELS[reg]); ax.title.set_fontsize(7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.045), title=None)
    fig.text(0.008, 1.0, f"n = {len(w)}", fontsize=7.5, color=MUTED, va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("Signal change\nagainst fixation (%)")
    fig.supxlabel("Showing of the sequence", fontsize=8, y=0.0)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, opj(PLOTS, "fig_univariate_3_learning"))


def _stat_label(ax, text, top):
    ax.text(0.04, 0.95 if top else 0.05, text, transform=ax.transAxes, fontsize=7, color=INK, zorder=6,
            va="top" if top else "bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))


def figure_4():
    b = pd.read_csv(opj(str(step_input_dir("E14_region_summary")), "behaviour_per_participant.csv"), index_col=0)
    enc = _read("E9_encoding_regions", "participants")
    blk = enc[(enc.model == "block") & enc.regressor.isin(["sequence_rep1", "sequence_rep2"])].pivot_table(
        index=["region", "subject"], columns="regressor", values="beta").dropna()
    x = (blk.sequence_rep2 - blk.sequence_rep1).unstack("region")
    fig, axes = plt.subplots(2, len(SCATTER_REGIONS), figsize=(WIDTH, 3.9), sharey="row", sharex="col")
    for i, (beh, ylab) in enumerate((("success_evening", "Retrieval success\n(evening)"), ("rt_evening", "Reaction time\n(s, evening)"))):
        for ax, (reg, colour) in zip(axes[i], SCATTER_REGIONS.items()):
            d = pd.concat([x[reg], b[beh]], axis=1, keys=["x", "y"]).dropna()
            ax.scatter(d.x, d.y, s=13, color=colour, edgecolor="white", linewidth=0.4, zorder=3)
            k, c0 = np.polyfit(d.x, d.y, 1)
            xs = np.array([d.x.min(), d.x.max()])
            ax.plot(xs, k * xs + c0, color=colour, lw=1.4, zorder=2)
            r, pv = stats.pearsonr(d.x, d.y)
            STATS.append(dict(figure=4, region=reg, test=f"Pearson r, change second minus first showing against {beh}", n=len(d),
                              mean=r, t=np.nan, p=pv))
            _stat_label(ax, f"r = {r:.2f}".replace("0.", ".") + f", {p_text(pv)}", top=(i == 1))
            ax.axvline(0, color=GREY, lw=0.6, ls=":", zorder=1)
            ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
            if i == 0:
                title(ax, LABELS[reg], colour)
        axes[i, 0].set_ylabel(ylab)
        axes[i, 0].set_ylim((-0.12, 1.04) if i == 0 else (0.5, 4.9))       # room for the statistics, clear of the points
    fig.supxlabel("Change in activity during learning, 2nd minus 1st showing (%)", fontsize=8, y=0.0)
    fig.text(0.008, 1.0, f"n = {len(d)}", fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, opj(PLOTS, "fig_univariate_4_across"))


def figure_5(model="block"):
    src = str(step_input_dir("E15_learning_by_sequence"))
    t, res = pd.read_csv(opj(src, "sequence_table.csv")), pd.read_csv(opj(src, "group_within_participant.csv"))
    fig, axes = plt.subplots(2, len(SCATTER_REGIONS), figsize=(WIDTH, 3.9), sharey="row")
    for i, (beh, ylab, lim) in enumerate((("success_pre", "Retrieval success\n(evening, centred)", 0.25),
                                          ("rt_pre", "Reaction time\n(s, evening, centred)", 1.2))):
        for ax, (reg, colour) in zip(axes[i], SCATTER_REGIONS.items()):
            d = t[(t.model == model) & (t.region == reg)].dropna(subset=["change", beh])
            slopes, xmax = [], 0.0
            for _, s_ in d.groupby("subject"):                      # selection of participants as in E15
                if len(s_) < 6 or s_[beh].std() == 0 or s_.change.std() == 0:
                    continue
                x, y = s_.change - s_.change.mean(), s_[beh] - s_[beh].mean()
                k = np.polyfit(x, y, 1)[0]
                slopes.append(k)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, k * xs, color=colour, lw=0.6, alpha=0.45, zorder=2)
                xmax = max(xmax, np.abs(x).quantile(0.9))
            xs = np.array([-xmax, xmax])
            ax.plot(xs, np.mean(slopes) * xs, color=INK, lw=1.8, zorder=4)
            r = res[(res.model == model) & (res.region == reg) & (res.behaviour == beh)].iloc[0]
            STATS.append(dict(figure=5, region=reg, test=f"slope of {beh} on change across sequences, against zero", n=len(slopes),
                              mean=np.mean(slopes), t=r.slope_t, p=r.slope_p))
            _stat_label(ax, f"t = {r.slope_t:.1f}, {p_text(r.slope_p)}\nn = {len(slopes)}", top=False)
            ax.axhline(0, color=GREY, lw=0.6, ls=":", zorder=1); ax.axvline(0, color=GREY, lw=0.6, ls=":", zorder=1)
            ax.set_xlim(-1.6 * xmax, 1.6 * xmax)
            if i == 0:
                title(ax, LABELS[reg], colour)
        axes[i, 0].set_ylabel(ylab)
        axes[i, 0].set_ylim(-lim, lim)
    fig.supxlabel("Change in activity during learning, 2nd minus 1st showing (%, centred within participant)", fontsize=8, y=0.0)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, opj(PLOTS, "fig_univariate_5_within"))


if __name__ == "__main__":
    apply_style()
    which = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for k in which:
        globals()[f"figure_{k}"]()
        print("figure", k, "done")
    if STATS:
        s = pd.DataFrame(STATS)
        s.to_csv(opj(OUT, "figure_statistics.csv"), index=False)
        print(s.round(4).to_string(index=False))
    print(f"-> {PLOTS}")
