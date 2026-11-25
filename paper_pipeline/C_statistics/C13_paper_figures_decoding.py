#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C13 - The decoding figures of the paper.

Three main figures, drawn from the tables of the earlier steps with the shared
look of figure_style.py (colours and names as in the univariate figures):

    fig_decoding_1_classifiers      A  cross-validated accuracy for objects, image shown
                                       and image recalled, six regions (B5)
                                    B  probability of the object on screen while the
                                       sequences are learned (B6b)
    fig_decoding_2_items            A  retrieved item minus matched control over the
                                       trial (C9)
                                    B  evidence by position relative to the retrieved
                                       item (C9)
    fig_decoding_3_sequentiality    A  absolute slope and sinusoid amplitude in
                                       parietal + precuneus at the three time points,
                                       five items and without the retrieved item (C11)
                                    B  change from the evening to the first morning
                                       retrieval in the six regions (C11)
                                    C  the same change with items left out (C11)

Every test shown in a figure is written to outputs/C13_paper_figures_decoding/
figure_statistics.csv, so that the numbers in the text can be checked against
the figures. Participant data are thin lines or dots, the group mean is drawn
with its 95 % confidence interval.

Input   outputs/B5_classifier_accuracy/accuracy_per_participant.csv
        outputs/B6b_encoding_decoding_group/group_timecourses_data/*.npy
        outputs/C9_item_time_courses/participant_time_courses.csv, press_times.csv
        outputs/C11_slope_amplitude_by_position/participants/*.csv, abs_slopes_v2.csv
Output  outputs/C13_paper_figures_decoding/figure_statistics.csv
Plots   plots/C13_paper_figures_decoding/fig_decoding_{1,2,3}_*.{png,pdf}

Not a port of the original pipeline: added 2026-11-25.

Usage:  python C13_paper_figures_decoding.py [1|2|3]
"""

from __future__ import annotations

import glob
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
from config import HRF_DELAY, TR, step_input_dir, step_output_dir, step_plot_dir
from figure_style import (BLUE, COND_LABEL, CONDS, GREY, GRID, INK, LIGHT, MUTED, ORANGE, REGION_COLOUR, REGION_LABEL,
                          REGION_SHORT, REGIONS, WIDTH, apply_style, ci95, holm, p_text, panel_letter, save, stars, title)

OUT, PLOTS = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
MAIN = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]          # regions with their own panel
WINDOW_S = (4.5, 12.0)
STATS = []


def region_title(ax, r):
    ax.set_title(REGION_LABEL[r], loc="left", color=REGION_COLOUR[r], fontweight="bold", fontsize=7.5, pad=4)


def shared_xlabel(fig, axes, text, pad=0.055):
    """One x label under a row of panels."""
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((x0 + x1) / 2, axes[0].get_position().y0 - pad, text, ha="center", va="top", fontsize=8)


def test(figure, panel, what, values, **extra):
    """One-sample t test across participants; kept for figure_statistics.csv."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    r = stats.ttest_1samp(v, 0)
    row = dict(figure=figure, panel=panel, test=what, n=len(v), mean=v.mean(), ci95=ci95(v), t=r.statistic, p=r.pvalue, **extra)
    STATS.append(row)
    return row


# =====================================================================
# Figure 1: classifiers
# =====================================================================

def figure_1():
    acc = pd.read_csv(opj(str(step_input_dir("B5_classifier_accuracy")), "accuracy_per_participant.csv"))
    acc = acc[(acc.classifier == "object") & acc.roi.isin(REGIONS)]
    fig = plt.figure(figsize=(WIDTH, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35)

    # --- A: accuracy -------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(0)
    rows = {}
    for kind in ("encoding", "retrieval"):
        rr = [test(1, "A", f"accuracy minus chance, image {'shown' if kind == 'encoding' else 'recalled'}",
                   acc[(acc.roi == r) & (acc.trial_type == kind)].accuracy - 10.0, roi=r) for r in REGIONS]
        for row, ph in zip(rr, holm([x["p"] for x in rr])):
            row["p_holm"] = ph
        rows[kind] = rr
    for i, r in enumerate(REGIONS):
        for kind, dx, colour, lab in (("encoding", -0.19, BLUE, "Image shown"), ("retrieval", 0.19, ORANGE, "Image recalled")):
            v = acc[(acc.roi == r) & (acc.trial_type == kind)].accuracy.values
            ax.bar(i + dx, v.mean(), width=0.34, color=colour, alpha=0.28, zorder=1)
            ax.scatter(i + dx + rng.uniform(-0.11, 0.11, len(v)), v, s=5, color=colour, alpha=0.55, lw=0, zorder=2)
            ax.errorbar(i + dx, v.mean(), yerr=ci95(v), color=INK, capsize=2, lw=1, zorder=3, marker="o", ms=3)
            if i == 0:
                ax.bar([np.nan], [np.nan], color=colour, alpha=0.6, label=lab)
            s = stars(rows[kind][i]["p_holm"])
            ax.text(i + dx, max(v.max(), v.mean()) + 1.2, s if s else "n.s.", ha="center", va="bottom",
                    fontsize=7 if s else 6, color=INK if s else MUTED)
    ax.axhline(10, color=INK, ls="--", lw=0.8)
    ax.text(-0.52, 10.8, "chance", ha="left", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels([REGION_LABEL[r].replace(" (control)", "\n(control)") for r in REGIONS])
    ax.set_ylabel("Cross-validated accuracy (%)"); ax.set_xlim(-0.55, len(REGIONS) - 0.45); ax.set_ylim(0, 84)
    ax.legend(loc="upper right", ncol=2)
    title(ax, "Object classifiers on the localizers")
    panel_letter(ax, "A", dx=-0.045)

    # --- B: learning session -------------------------------------------------------------
    folder = opj(str(step_input_dir("B6b_encoding_decoding_group")), "group_timecourses_data")
    t = np.arange(-4, 5) * TR + HRF_DELAY
    axes = [fig.add_subplot(gs[1, k]) for k in range(4)]
    for ax, r in zip(axes, MAIN):
        d = np.load(opj(folder, f"grouplevel_{r}_subject_mean_timecourses_objs_use_overall_classifier1_specific2_2.npy"),
                    allow_pickle=True).item()
        X = np.array([np.asarray(v, float) for v in d.values()]) * 100.0
        ax.plot(t, X.T, color=GREY, lw=0.4, alpha=0.6)
        m, c = X.mean(0), ci95(X)
        ax.fill_between(t, m - c, m + c, color=REGION_COLOUR[r], alpha=0.3, lw=0)
        ax.plot(t, m, color=REGION_COLOUR[r], lw=1.8)
        ax.axhline(10, color=INK, ls="--", lw=0.8); ax.axvline(0, color=MUTED, ls=":", lw=0.7)
        k = int(np.argmax(m))
        row = test(1, "B", "probability of the object on screen minus chance, at the group peak", X[:, k] - 10.0, roi=r,
                   time_s=float(t[k]))
        ax.text(0.97, 0.95, f"peak {m[k]:.1f} %\nt = {row['t']:.1f}{stars(row['p'])}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7)
        region_title(ax, r)
        ax.set_xticks([0, 5, 10])
    axes[0].set_ylabel("Probability of the\nobject on screen (%)")
    for ax in axes[1:]:
        ax.set_ylim(6, 20)
    for ax in axes[2:]:
        ax.set_yticklabels([])
    shared_xlabel(fig, axes, "Time from picture onset (s)")
    panel_letter(axes[0], "B", dx=-0.2)
    save(fig, opj(PLOTS, "fig_decoding_1_classifiers"))


# =====================================================================
# Figure 2: the retrieved item and its neighbours
# =====================================================================

RELATIVE = {"two_before": -2, "predecessor": -1, "retrieved": 0, "successor": 1, "two_after": 2}


def figure_2():
    src = str(step_input_dir("C9_item_time_courses"))
    d = pd.read_csv(opj(src, "participant_time_courses.csv"))
    d = d[(d.cond == "pooled") & d.roi.isin(REGIONS)]
    press = pd.read_csv(opj(src, "press_times.csv")).set_index("cond").median_press_s_from_cue["pooled"]
    win = d[(d.time_s >= WINDOW_S[0]) & (d.time_s <= WINDOW_S[1])].groupby(["subject", "roi", "item"]).difference.mean().unstack("item")

    fig = plt.figure(figsize=(WIDTH, 5.4))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.32)

    # --- A: time course of the retrieved item ---------------------------------------------
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    rows = [test(2, "A", "retrieved item minus control, 4.5-12 s", win.xs(r, level="roi")["retrieved"], roi=r) for r in REGIONS]
    for row, ph in zip(rows, holm([x["p"] for x in rows])):
        row["p_holm"] = ph
    by_roi = {x["roi"]: x for x in rows}
    for ax, r in zip(axes, MAIN):
        X = d[(d.roi == r) & (d.item == "retrieved")].pivot(index="subject", columns="time_s", values="difference")
        t = X.columns.values
        ax.axvspan(*WINDOW_S, color=LIGHT, lw=0, zorder=0)
        ax.plot(t, X.values.T, color=GREY, lw=0.4, alpha=0.55)
        m, c = X.mean(0).values, ci95(X.values)
        ax.fill_between(t, m - c, m + c, color=REGION_COLOUR[r], alpha=0.3, lw=0)
        ax.plot(t, m, color=REGION_COLOUR[r], lw=1.8)
        ax.axhline(0, color=INK, lw=0.6); ax.axvline(press, color=MUTED, ls=":", lw=0.8)
        row = by_roi[r]
        ax.text(0.97, 0.96, f"{row['mean']:+.1f} points\nt = {row['t']:.1f}{stars(row['p_holm'])}".replace("+0.0", "0.0").replace("-0.0", "0.0"), transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
        region_title(ax, r)
        ax.set_xticks([0, 4.5, 9, 13.5]); ax.set_xticklabels(["0", "4.5", "9", "13.5"])
        ax.set_ylim(-9, 13)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    shared_xlabel(fig, axes, "Time from retrieval cue (s)")
    axes[0].set_ylabel("Retrieved item minus\ncontrol (points)")
    axes[0].text(press + 0.3, -8.4, "median\nbutton press", fontsize=6, color=MUTED, va="bottom")
    panel_letter(axes[0], "A", dx=-0.2)

    # --- B: evidence by relative position ----------------------------------------------------
    axes = [fig.add_subplot(gs[1, k]) for k in range(4)]
    x = np.array(list(RELATIVE.values()))
    all_rows = {r: [test(2, "B", f"{item} minus control, 4.5-12 s", win.xs(r, level="roi")[item], roi=r) for item in RELATIVE]
                for r in REGIONS}
    for k in range(len(RELATIVE)):                                   # Holm across the six regions, per item
        for r, ph in zip(REGIONS, holm([all_rows[r][k]["p"] for r in REGIONS])):
            all_rows[r][k]["p_holm"] = ph
    for ax, r in zip(axes, MAIN):
        W = win.xs(r, level="roi")[list(RELATIVE)]
        rows = all_rows[r]
        m, c = W.mean(0).values, ci95(W.values)
        ax.axhline(0, color=INK, lw=0.6)
        ax.bar(x, m, width=0.62, color=[REGION_COLOUR[r] if k == 0 else GREY for k in x], alpha=0.85, zorder=1)
        ax.errorbar(x, m, yerr=c, fmt="none", ecolor=INK, capsize=2, lw=1, zorder=3)
        for xi, mi, ci_, row in zip(x, m, c, rows):
            s = stars(row["p_holm"])
            if s:
                ax.text(xi, mi + ci_ + 0.08 if mi > 0 else mi - ci_ - 0.08, s, ha="center", va="bottom" if mi > 0 else "top", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(["−2", "−1\nbefore", "0", "+1\nafter", "+2"], fontsize=7)
        ax.set_ylim(-1.3, 3.6)
        region_title(ax, r)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    shared_xlabel(fig, axes, "Position relative to the retrieved item (0)", pad=0.075)
    axes[0].set_ylabel("Item minus control (points)")
    panel_letter(axes[0], "B", dx=-0.2)
    save(fig, opj(PLOTS, "fig_decoding_2_items"))


# =====================================================================
# Figure 3: sequentiality at the three time points
# =====================================================================

MEASURE_LABEL = {"abs_slope": "Absolute slope", "amplitude": "Sinusoid amplitude"}
ITEMS = {"five": "All five items", "without": "Without the retrieved item"}
LEAVE_OUT = [("All five items", "all", None), ("Without the item before", "pred", None), ("Without the item after", "succ", None),
             ("Without the retrieved item", "without", None), ("Without retrieved item\nand item before", "without_retrieved_and_predecessor", None),
             ("Without retrieved item\nand item after", "without_retrieved_and_successor", None),
             ("Without retrieved item\nand both neighbours", "without_retrieved_and_neighbours", None)]


def _select(d, key):
    """Trials x item set, as ITEM_SETS of C11."""
    if key in ("all", "five"):
        return d[d.variant == "all"]
    if key == "without":
        return d[(d.variant == "without") & (d.q == d.position)]
    if key == "pred":
        return pd.concat([d[(d.variant == "without") & (d.q == d.position - 1)], d[(d.variant == "all") & (d.position == 1)]])
    if key == "succ":
        return pd.concat([d[(d.variant == "without") & (d.q == d.position + 1)], d[(d.variant == "all") & (d.position == 5)]])
    return d[d.variant == key]


def _load_sequentiality():
    src = str(step_input_dir("C11_slope_amplitude_by_position"))
    amp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(src, "participants", "*_amplitudes*.csv")))], ignore_index=True)
    ab = pd.read_csv(opj(src, "abs_slopes_v2.csv"))
    out = {}
    for name, d in (("amplitude", amp), ("abs_slope", ab)):
        d = d[d.roi.isin(REGIONS)].copy()
        d["difference"] = d.retrieved - d.control
        out[name] = d
    return out


def _by_cond(d, key):
    """Participants x time points (per region) for one item set."""
    return _select(d, key).groupby(["roi", "subject", "cond"]).difference.mean().unstack("cond")[CONDS]


def figure_3():
    data = _load_sequentiality()
    fig = plt.figure(figsize=(WIDTH, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.42)
    top = gs[0].subgridspec(1, 4, wspace=0.5)
    bottom = gs[1].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.95)
    colour = REGION_COLOUR["parietal_precuneus"]

    # --- A: parietal + precuneus at the three time points ------------------------------------
    axes = [fig.add_subplot(top[k]) for k in range(4)]
    k = 0
    for measure in MEASURE_LABEL:
        for key, lab in ITEMS.items():
            ax = axes[k]; k += 1
            W = _by_cond(data[measure], key).xs("parietal_precuneus", level="roi")
            x = np.arange(3)
            ax.plot(x, W.values.T, color=GREY, lw=0.4, alpha=0.55)
            m, c = W.mean(0).values, ci95(W.values)
            ax.axhline(0, color=INK, lw=0.6)
            ax.errorbar(x, m, yerr=c, color=colour, marker="o", ms=4.5, lw=1.8, capsize=2.5, zorder=4, mec=INK, mew=0.5, ecolor=INK)
            lim = np.nanpercentile(np.abs(W.values), 97) * 1.25
            ax.set_ylim(-lim, lim)
            for xi, cnd in zip(x, CONDS):
                row = test(3, "A", f"{measure}, {lab}, {cnd}", W[cnd], roi="parietal_precuneus")
                ax.text(xi, lim * 0.97, stars(row["p"]) or "n.s.", ha="center", va="top", fontsize=7.5 if stars(row["p"]) else 6,
                        color=INK if stars(row["p"]) else MUTED)
            ch = test(3, "A", f"{measure}, {lab}, post1 - pre", W.post1 - W.pre, roi="parietal_precuneus")
            test(3, "A", f"{measure}, {lab}, post2 - post1", W.post2 - W.post1, roi="parietal_precuneus")
            ax.plot([0, 0, 1, 1], [-lim * 0.78, -lim * 0.84, -lim * 0.84, -lim * 0.78], color=INK, lw=0.7)
            ax.text(0.5, -lim * 0.86, f"t = {ch['t']:.1f}{stars(ch['p'])}", ha="center", va="top", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(["Evening", "First\nmorning", "Repeat\nmorning"], fontsize=5.8)
            ax.set_xlim(-0.4, 2.4)
            ax.set_title(MEASURE_LABEL[measure], loc="left", fontsize=8, fontweight="bold", pad=14)
            ax.text(0, 1.02, {"five": "five items", "without": "without the retrieved item"}[key], transform=ax.transAxes,
                    fontsize=7, color=ORANGE if key == "without" else MUTED, va="bottom")
    axes[0].set_ylabel("Retrieved minus control sequence\n(points per position)")
    panel_letter(axes[0], "A", dx=-0.3, dy=1.2)
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.062, "Parietal + precuneus at the three time points",
             fontsize=8.5, fontweight="bold", color=colour, va="bottom")

    # --- B: change by region -----------------------------------------------------------------------
    ax = fig.add_subplot(bottom[0])
    y = np.arange(len(REGIONS))
    for measure, dy, mk in (("abs_slope", -0.17, "o"), ("amplitude", 0.17, "s")):
        W = _by_cond(data[measure], "five")
        for i, r in enumerate(REGIONS):
            w = W.xs(r, level="roi")
            row = test(3, "B", f"{measure}, five items, post1 - pre", w.post1 - w.pre, roi=r)
            ax.errorbar(row["mean"], i + dy, xerr=row["ci95"], color=REGION_COLOUR[r], marker=mk, ms=4.5, capsize=2, lw=1.2,
                        mec=INK, mew=0.5, ecolor=INK, mfc=REGION_COLOUR[r] if measure == "abs_slope" else "white")
            if stars(row["p"]):
                ax.text(row["mean"] + row["ci95"] + 0.03, i + dy, stars(row["p"]), va="center", fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([REGION_LABEL[r] for r in REGIONS]); ax.invert_yaxis()
    ax.set_xlabel("First morning minus evening\n(points per position)")
    title(ax, "Change by region (five items)")
    panel_letter(ax, "B", dx=-0.62)

    # --- C: items left out ------------------------------------------------------------------------------
    ax = fig.add_subplot(bottom[1])
    for measure, dy, mk in (("abs_slope", -0.17, "o"), ("amplitude", 0.17, "s")):
        for i, (lab, key, _) in enumerate(LEAVE_OUT):
            w = _by_cond(data[measure], key).xs("parietal_precuneus", level="roi")
            row = test(3, "C", f"{measure}, {lab.replace(chr(10), ' ')}, post1 - pre", w.post1 - w.pre, roi="parietal_precuneus")
            test(3, "C", f"{measure}, {lab.replace(chr(10), ' ')}, post1", w.post1, roi="parietal_precuneus")
            ax.errorbar(row["mean"], i + dy, xerr=row["ci95"], color=colour, marker=mk, ms=4.5, capsize=2, lw=1.2, mec=INK, mew=0.5,
                        ecolor=INK, mfc=colour if measure == "abs_slope" else "white")
            if stars(row["p"]):
                ax.text(row["mean"] + row["ci95"] + 0.04, i + dy, stars(row["p"]), va="center", fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.6); ax.axhline(2.5, color=GRID, lw=0.8)
    ax.set_yticks(range(len(LEAVE_OUT))); ax.set_yticklabels([l for l, _, _ in LEAVE_OUT], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("First morning minus evening\n(points per position)")
    ax.scatter([], [], marker="o", color=MUTED, edgecolors=INK, lw=0.5, s=22, label="Absolute slope")
    ax.scatter([], [], marker="s", color="white", edgecolors=INK, lw=0.7, s=22, label="Sinusoid amplitude")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2, fontsize=7.5,
               handletextpad=0.2, columnspacing=2)
    title(ax, "Items left out (parietal + precuneus)")
    panel_letter(ax, "C", dx=-0.66)
    save(fig, opj(PLOTS, "fig_decoding_3_sequentiality"))


if __name__ == "__main__":
    apply_style()
    which = sys.argv[1:] or ["1", "2", "3"]
    for k in which:
        {"1": figure_1, "2": figure_2, "3": figure_3}[k]()
    s = pd.DataFrame(STATS)
    tag = "" if len(which) == 3 else "_" + "".join(which)
    s.to_csv(opj(OUT, f"figure_statistics{tag}.csv"), index=False)
    print(s.round(3).to_string())
    print(f"-> {PLOTS}")
