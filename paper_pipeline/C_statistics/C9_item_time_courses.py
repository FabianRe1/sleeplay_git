#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C9 - Classifier probability over the retrieval trial for single items:
     the retrieved item, its predecessor and its successor, each next to its
     matched control item.

Object trials. Time 0 is the onset of the retrieval cue: B7 stores the
classifier probabilities for the ten volumes from the cue on (volume 0 is the
volume closest to the cue onset, so times are exact to +-0.75 s), 1.5 s apart.

Items and their control. The object classifier separates ten categories. Every
participant learned ten scenes of five objects each; the categories of a scene
follow one of two fixed category sequences (five categories each, e.g.
clothing - vehicle - music - body - animal), five scenes per sequence. For an
item at position p of the cued scene, the control item is the category at the
SAME position p of the OTHER category sequence. Every category is therefore the
item of interest on some trials and the control on the others, equally often,
so a general preference of the classifier for a category cancels.
(A control taken from "the same scene number in the other context" is not
usable throughout: in two or three of the five scene pairs both scenes follow
the same category sequence, and the control would be the same category.)

    retrieved    position p            (all positions 1-5)
    predecessor  position p - 1        (retrieved positions 2-5)
    successor    position p + 1        (retrieved positions 1-4)
    two_before   position p - 2        (retrieved positions 3-5)   tables and the by-position figure only
    two_after    position p + 2        (retrieved positions 1-3)   tables and the by-position figure only
    three_ / four_before and _after    (where they exist)          tables and the matrix only

Time points: pre = evening; post1 = morning trials 1-70; post2 = morning trials
71-140; pooled = all trials of the participant.

Sessions in config.EXCLUDED_SESSIONS are left out (none at present). The morning
session of s41 was recorded with another participant's trial list; its retrieved
items are taken from the cue (config.RELABELLED_SESSIONS, applied in A1 and B7).
Any table whose targets do not sit at the cued position of their category
sequence is skipped with a message.

Per participant, ROI, item and time point the trial probabilities are averaged
per volume; all figures and tests are across participants.

Input   outputs/B7_apply_classifiers_retrieval/<presleep|postsleep>/decoding_results_tables_for_SODA/*.csv
Output  outputs/C9_item_time_courses/participant_time_courses.csv
            subject, roi, item, cond, tr, time_s, item_prob, control_prob, difference, n_trials
        outputs/C9_item_time_courses/group_time_courses.csv
            per ROI, item, time point and volume: mean, 95 % CI, t-test of the
            difference against zero (p_holm: corrected across the ten volumes)
        outputs/C9_item_time_courses/group_window.csv
            the same for the mean over volumes 3-8 (4.5-12 s), the window of C2 / C5
            (p_holm: corrected across the ROIs of the figure), with the item and the
            control each tested against chance (10 %)
        outputs/C9_item_time_courses/group_contrasts.csv
            window means: predecessor minus successor on trials with retrieved position
            2-4 (both exist), per time point; and the change between time points
            (post1 - pre, post2 - post1, post2 - pre) of every item's difference
        outputs/C9_item_time_courses/press_times.csv   median time of the button press per time point
Plots   plots/C9_item_time_courses/<item>_probabilities_<cond>.{png,pdf}
            item and control probability: one thin line per participant, group mean,
            95 % confidence interval across participants as shading
        plots/C9_item_time_courses/<item>_difference_<cond>.{png,pdf}
            item minus control, same layout; stars per volume (Holm across volumes:
            * < .05, ** < .01, *** < .001) and the window test in the panel
        outputs/C9_item_time_courses/group_by_retrieved_position.csv
            window means of item minus control for every combination of the position of
            the retrieved item (1-5) and the position of the item in the sequence (1-5),
            with the number of trials per participant; `equal_trials` repeats the
            by-position summary on equal sets of trials (retrieved position 3 for all
            five relative positions; retrieved positions 2-4 for -1, 0, +1); and the
            linear trend of the retrieved item's evidence over its position
        plots/C9_item_time_courses/retrieved_position_matrix.{png,pdf}
            that table as one matrix per ROI (rows: retrieved position, columns: item
            position; the diagonal is the retrieved item)
        plots/C9_item_time_courses/by_position.{png,pdf}, by_position_main.{png,pdf} (visual, parietal + precuneus, MTL)
            item minus control (window mean) by position relative to the retrieved
            item, -2 to +2: rows = ROIs, columns = all time points and each time point;
            dots = participants, mean with 95 % CI, stars uncorrected (the corrected
            tests are in group_window.csv)

Usage:  python C9_item_time_courses.py

Not a port of the original pipeline: added 2026-11-25.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EXCLUDED_SESSIONS, TR, is_excluded, step_input_dir, step_output_dir, step_plot_dir

ROIS = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL", "hpc": "Hippocampus",
        "somato": "Somatosensory (control)", "caudal_acc": "Caudal ACC (control)"}
ITEMS = {"four_before": -4, "three_before": -3, "two_before": -2, "predecessor": -1, "retrieved": 0,
         "successor": 1, "two_after": 2, "three_after": 3, "four_after": 4}
BY_POSITION_ITEMS = {k: v for k, v in ITEMS.items() if abs(v) <= 2}     # the by-position figure shows -2 to +2
TIME_COURSE_ITEMS = ("retrieved", "predecessor", "successor")          # items with their own time-course figures
ITEM_LABELS = {"retrieved": "Retrieved item", "predecessor": "Item before the retrieved one",
               "successor": "Item after the retrieved one", "two_before": "Item two before", "two_after": "Item two after"}
CONDS = {"pooled": "all time points", "pre": "evening", "post1": "first morning retrieval",
         "post2": "repeated morning retrieval"}
POST1_MAX_TRIAL = 70
WINDOW = (3, 8)                       # volumes, as in C2 / C5
CUE_TO_PROMPT = 1.015                 # s from the retrieval cue to the question mark
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def p_text(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def category_sequences(d):
    """The two category sequences of a participant, from the hierarchy stored in the B7 table."""
    h = json.loads(d["full_structured_sequence_category"].dropna().iloc[0])
    seqs = []
    for scenes in h.values():
        for cats in scenes.values():
            if cats not in seqs:
                seqs.append(cats)
    if len(seqs) != 2 or any(len(s) != 5 for s in seqs) or set(seqs[0]) & set(seqs[1]):
        raise ValueError(f"expected two disjoint category sequences of five, got {seqs}")
    return seqs


def one_table(path, session):
    """Per trial, item and volume: probability of the item and of its control."""
    d = pd.read_csv(path)
    seqs = category_sequences(d)
    d = d[d.object_scene_retrieval == "object"]
    if d.empty:
        return None, None
    probs = d.pivot_table(index=["trial", "tr"], columns="stimulus_class_from_classifier",
                          values="probability_from_classifier")
    classes = list(probs.columns)
    trials = probs.index.get_level_values("trial").unique()
    trs = probs.index.get_level_values("tr").unique()
    P = probs.reindex(pd.MultiIndex.from_product([trials, trs])).values.reshape(len(trials), len(trs), len(classes))
    info = d.drop_duplicates("trial").set_index("trial").loc[trials]
    pos = info.retrieval_cue_position.astype(int).values
    in_first = info.retrieval_target.isin(seqs[0]).values
    for k, (t, p) in enumerate(zip(info.retrieval_target.values, pos)):
        if (seqs[0] if in_first[k] else seqs[1])[p - 1] != t:
            raise ValueError(f"trial {trials[k]}: target {t} is not at position {p} of its category sequence")
    cond = np.array(["pre"] * len(trials)) if session == "presleep" else np.where(trials <= POST1_MAX_TRIAL, "post1", "post2")
    code = [[classes.index(c) for c in s] for s in seqs]                 # class index per sequence and position
    rows = []
    for item, lag in ITEMS.items():
        q = pos + lag
        ok = (q >= 1) & (q <= 5)
        own = np.array([code[0 if f else 1][qq - 1] if o else 0 for f, qq, o in zip(in_first, q, ok)])
        ctl = np.array([code[1 if f else 0][qq - 1] if o else 0 for f, qq, o in zip(in_first, q, ok)])
        a = np.take_along_axis(P, own[:, None, None], axis=2)[:, :, 0][ok] * 100.0       # trials x volumes
        b = np.take_along_axis(P, ctl[:, None, None], axis=2)[:, :, 0][ok] * 100.0
        n = int(ok.sum())
        rows.append(pd.DataFrame(dict(trial=np.repeat(trials[ok], len(trs)), cond=np.repeat(cond[ok], len(trs)), item=item,
                                      position=np.repeat(pos[ok], len(trs)), tr=np.tile(trs, n),
                                      item_prob=a.ravel(), control_prob=b.ravel())))
    rt = info["ret_rt"].values if "ret_rt" in info else np.full(len(trials), np.nan)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(dict(cond=cond, rt=rt))


def load():
    parts, press = [], []
    for session in ("presleep", "postsleep"):
        folder = str(step_input_dir("B7_apply_classifiers_retrieval", opj(session, "decoding_results_tables_for_SODA")))
        for path in sorted(glob.glob(opj(folder, "*.csv"))):
            m = re.match(r"^(s\d+)_ROI_(.+)_decoding_results_retrieval", os.path.basename(path))
            if not m or m.group(2) not in ROIS:
                continue
            if is_excluded(m.group(1), session):                    # config.EXCLUDED_SESSIONS
                continue
            try:
                t, pr = one_table(path, session)
            except Exception as err:
                print(f"  skipped {os.path.basename(path)}: {err}")
                continue
            if t is None:
                continue
            parts.append(t.assign(subject=m.group(1), roi=m.group(2)))
            if m.group(2) == "vis":
                press.append(pr.assign(subject=m.group(1)))
    return pd.concat(parts, ignore_index=True), (pd.concat(press, ignore_index=True) if press else None)


def participant_means(t):
    keys = ["subject", "roi", "item", "tr"]
    by = t.groupby(keys + ["cond"]).agg(item_prob=("item_prob", "mean"), control_prob=("control_prob", "mean"),
                                        n_trials=("trial", "nunique")).reset_index()
    pooled = t.groupby(keys).agg(item_prob=("item_prob", "mean"), control_prob=("control_prob", "mean"),
                                 n_trials=("trial", "size")).reset_index().assign(cond="pooled")   # trial numbers repeat across sessions
    s = pd.concat([by, pooled], ignore_index=True)
    s["difference"] = s.item_prob - s.control_prob
    s["time_s"] = s.tr * TR
    return s


def group_tables(s, out):
    rows, win = [], []
    for (roi, item, cond), g in s.groupby(["roi", "item", "cond"]):
        w = {c: g.pivot(index="subject", columns="tr", values=c) for c in ("item_prob", "control_prob", "difference")}
        n = len(w["difference"])
        tt = stats.ttest_1samp(w["difference"], 0, axis=0, nan_policy="omit")
        ph = multipletests(tt.pvalue, method="holm")[1]
        k = stats.t.ppf(0.975, n - 1) / np.sqrt(n)
        for i, tr in enumerate(w["difference"].columns):
            rows.append(dict(roi=roi, item=item, cond=cond, tr=tr, time_s=tr * TR, n=n,
                             **{f"{c}_mean": w[c][tr].mean() for c in w},
                             **{f"{c}_ci95": w[c][tr].std(ddof=1) * k for c in w},
                             t=tt.statistic[i], p=tt.pvalue[i], p_holm=ph[i]))
        v = w["difference"].loc[:, WINDOW[0]:WINDOW[1]].mean(axis=1)
        r = stats.ttest_1samp(v, 0)
        ip = w["item_prob"].loc[:, WINDOW[0]:WINDOW[1]].mean(axis=1)
        cp = w["control_prob"].loc[:, WINDOW[0]:WINDOW[1]].mean(axis=1)
        ri, rc = stats.ttest_1samp(ip, 10), stats.ttest_1samp(cp, 10)
        win.append(dict(roi=roi, item=item, cond=cond, n=n, item_prob=ip.mean(), control_prob=cp.mean(),
                        difference=v.mean(), t=r.statistic, p=r.pvalue,
                        t_item_vs_chance=ri.statistic, p_item_vs_chance=ri.pvalue,
                        t_control_vs_chance=rc.statistic, p_control_vs_chance=rc.pvalue))
    g, w = pd.DataFrame(rows), pd.DataFrame(win)
    w["p_holm"] = w.groupby(["item", "cond"]).p.transform(lambda p: multipletests(p, method="holm")[1])
    g.to_csv(opj(out, "group_time_courses.csv"), index=False)
    w.to_csv(opj(out, "group_window.csv"), index=False)
    return g, w


def window_values(t):
    """Per participant, ROI, item and time point: item minus control over the window (also pooled)."""
    d = t[(t.tr >= WINDOW[0]) & (t.tr <= WINDOW[1])].assign(difference=lambda x: x.item_prob - x.control_prob)
    by = d.groupby(["roi", "item", "subject", "cond"]).difference.mean().unstack("cond")
    by["pooled"] = d.groupby(["roi", "item", "subject"]).difference.mean()
    return by


def contrasts(t, out):
    """Predecessor minus successor (retrieved positions 2-4) and the changes between time points."""
    rows = []
    mid = window_values(t[t.position.between(2, 4)])
    for roi in sorted(set(t.roi)):
        a, b = mid.loc[(roi, "predecessor")], mid.loc[(roi, "successor")]
        for cond in ["pooled", "pre", "post1", "post2"]:
            v = (a[cond] - b[cond]).dropna()
            r = stats.ttest_1samp(v, 0)
            rows.append(dict(roi=roi, test="predecessor - successor (positions 2-4)", cond=cond, n=len(v),
                             mean=v.mean(), t=r.statistic, p=r.pvalue))
    allv = window_values(t)
    for (roi, item), g in allv.groupby(level=["roi", "item"]):
        for name, (x, y) in {"post1 - pre": ("post1", "pre"), "post2 - post1": ("post2", "post1"), "post2 - pre": ("post2", "pre")}.items():
            v = (g[x] - g[y]).dropna()
            r = stats.ttest_1samp(v, 0)
            rows.append(dict(roi=roi, test=f"{item}: change between time points", cond=name, n=len(v),
                             mean=v.mean(), t=r.statistic, p=r.pvalue))
    c = pd.DataFrame(rows)
    c["p_holm"] = c.groupby(["test", "cond"]).p.transform(lambda p: multipletests(p, method="holm")[1])
    c.to_csv(opj(out, "group_contrasts.csv"), index=False)
    return c, allv


def retrieved_position_tables(t, out, plot_dir):
    """Does it matter which item is retrieved? Evidence for every item by position of the retrieved item."""
    d = t[(t.tr >= WINDOW[0]) & (t.tr <= WINDOW[1])].assign(difference=lambda x: x.item_prob - x.control_prob)
    d["item_position"] = d.position + d.item.map(ITEMS)
    n_tr = d.groupby(["roi", "subject", "position", "item_position"]).trial.count() / (WINDOW[1] - WINDOW[0] + 1)
    v = d.groupby(["roi", "subject", "position", "item_position"]).difference.mean()
    rows = []
    for (roi, p, q), g in v.groupby(level=["roi", "position", "item_position"]):
        r = stats.ttest_1samp(g, 0)
        rows.append(dict(table="matrix", roi=roi, retrieved_position=p, item_position=q, relative_position=q - p, n=len(g),
                         trials_per_participant=n_tr.loc[(roi, slice(None), p, q)].mean(), mean=g.mean(), t=r.statistic, p=r.pvalue))
    # the by-position summary on equal sets of trials
    for label, positions, lags in (("retrieved position 3", [3], (-2, -1, 0, 1, 2)), ("retrieved positions 2-4", [2, 3, 4], (-1, 0, 1))):
        e = d[d.position.isin(positions)]
        e = e.assign(lag=e.item.map(ITEMS))
        w = e[e.lag.isin(lags)].groupby(["roi", "subject", "lag"]).difference.mean()
        nt = e[e.lag.isin(lags)].groupby(["roi", "subject", "lag"]).trial.count() / (WINDOW[1] - WINDOW[0] + 1)
        for (roi, lag), g in w.groupby(level=["roi", "lag"]):
            r = stats.ttest_1samp(g, 0)
            rows.append(dict(table=f"equal_trials: {label}", roi=roi, relative_position=lag, n=len(g),
                             trials_per_participant=nt.loc[(roi, slice(None), lag)].mean(), mean=g.mean(), t=r.statistic, p=r.pvalue))
    # all trials, for the trial counts behind the by-position figure
    e = d.assign(lag=d.item.map(ITEMS))
    nt = e.groupby(["roi", "subject", "lag"]).trial.count() / (WINDOW[1] - WINDOW[0] + 1)
    for (roi, lag), g in e.groupby(["roi", "subject", "lag"]).difference.mean().groupby(level=["roi", "lag"]):
        r = stats.ttest_1samp(g, 0)
        rows.append(dict(table="all trials", roi=roi, relative_position=lag, n=len(g),
                         trials_per_participant=nt.loc[(roi, slice(None), lag)].mean(), mean=g.mean(), t=r.statistic, p=r.pvalue))
    # linear trend of the retrieved item's evidence over its position
    diag = v[v.index.get_level_values("position") == v.index.get_level_values("item_position")].unstack("position")
    diag.columns = diag.columns.get_level_values(0) if isinstance(diag.columns, pd.MultiIndex) else diag.columns
    for roi, g in diag.groupby(level="roi"):
        g = g.droplevel("item_position") if "item_position" in g.index.names else g
        g = g.groupby(level="subject").first().dropna()
        slope = g.apply(lambda r_: np.polyfit(np.arange(1, 6), r_.values.astype(float), 1)[0], axis=1)
        r = stats.ttest_1samp(slope, 0)
        rows.append(dict(table="trend of the retrieved item over its position (points per position)", roi=roi, n=len(slope),
                         mean=slope.mean(), t=r.statistic, p=r.pvalue))
    res = pd.DataFrame(rows)
    res.to_csv(opj(out, "group_by_retrieved_position.csv"), index=False)

    rois = [r for r in ("vis", "parietal_precuneus", "mtl") if r in set(res.roi)]
    fig, axes = plt.subplots(1, len(rois), figsize=(4.6 * len(rois), 4.7), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        m = res[(res.table == "matrix") & (res.roi == roi)]
        M = m.pivot(index="retrieved_position", columns="item_position", values="mean").reindex(index=range(1, 6), columns=range(1, 6))
        T = m.pivot(index="retrieved_position", columns="item_position", values="p").reindex(index=range(1, 6), columns=range(1, 6))
        lim = np.nanmax(np.abs(M.values))
        cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e4e3df")
        im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap, vmin=-lim, vmax=lim)
        for i in range(5):
            for j in range(5):
                if np.isfinite(M.values[i, j]):
                    ax.text(j, i, f"{M.values[i, j]:+.1f}{stars(T.values[i, j])}", ha="center", va="center", fontsize=8,
                            fontweight="bold" if i == j else "normal", color=INK)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.4))
        ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
        ax.set_xlabel("Position of the item in the sequence"); ax.set_title(ROIS[roi], loc="left", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0][0].set_ylabel("Position of the retrieved item")
    fig.suptitle("Item minus matched control (percentage points, 4.5-12 s) by which item is retrieved; "
                 "framed: the retrieved item; * p < .05 uncorrected", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"retrieved_position_matrix.{ext}"), **kw)
    plt.close(fig)
    return res


def by_position_figure(allv, plot_dir, only=None, name="by_position"):
    rois = [r for r in ROIS if r in set(allv.index.get_level_values("roi")) and (only is None or r in only)]
    conds = list(CONDS)
    fig, axes = plt.subplots(len(rois), len(conds), figsize=(3.3 * len(conds), 2.2 * len(rois)), squeeze=False, sharex=True)
    for i, roi in enumerate(rois):
        vals = {c: [allv.loc[(roi, it)][c].dropna().values for it in BY_POSITION_ITEMS] for c in conds}
        top = max(abs(v.mean()) + stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) for c in conds for v in vals[c])
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            for k, (item, lag) in enumerate(BY_POSITION_ITEMS.items()):
                v = vals[cond][k]
                colour = BLUE if lag == 0 else MUTED
                ax.scatter(lag + np.random.default_rng(k).uniform(-0.13, 0.13, len(v)), v, s=4, color=colour, alpha=0.18, linewidths=0, zorder=2)
                ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                ax.errorbar(lag, v.mean(), yerr=ci, fmt="o", color=INK, markerfacecolor=colour, markersize=6, capsize=2.5, lw=1.1, zorder=4)
                st = stars(stats.ttest_1samp(v, 0).pvalue)
                if st:
                    ax.text(lag, v.mean() + ci + 0.06 * top, st, ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.axhline(0, color=INK, ls=":", lw=0.8, zorder=1)
            ax.set_ylim(-1.9 * top, 1.9 * top)                      # single participants may leave the panel
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([-2, -1, 0, 1, 2])
            if i == 0:
                ax.set_title(CONDS[cond].capitalize(), fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ROIS[roi].replace(" (control)", "\n(control)").replace(" + ", " +\n"), fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Position relative to retrieved item")
    fig.suptitle("Item minus matched control, 4.5-12 s after the cue (percentage points); * p < .05 uncorrected", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


def _axes(n):
    ncol = 3 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow), squeeze=False, sharex=True)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes


def _decorate(ax, press_s, first):
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.axvline(CUE_TO_PROMPT, color=MUTED, lw=0.7, ls=":", zorder=1)
    if press_s is not None and np.isfinite(press_s):
        ax.axvline(press_s, color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    if first:
        y = ax.get_ylim()[1]
        ax.text(0.1, y, "cue", fontsize=7, va="top", color=INK)
        ax.text(CUE_TO_PROMPT + 0.1, y, "\n?", fontsize=7, va="top", color=MUTED)
        if press_s is not None and np.isfinite(press_s):
            ax.text(press_s + 0.1, y, "\n\nmedian press", fontsize=7, va="top", color=MUTED)


def figures(s, g, w, press_s, plot_dir):
    rois = [r for r in ROIS if r in set(s.roi)]
    for item in TIME_COURSE_ITEMS:
        for cond in CONDS:
            d = s[(s.item == item) & (s.cond == cond)]
            if d.empty:
                continue
            # 1. probabilities of the item and of its control
            fig, axes = _axes(len(rois))
            for i, (ax, roi) in enumerate(zip(axes.ravel(), rois)):
                for col, colour, lab in (("control_prob", ORANGE, "Matched control item"), ("item_prob", BLUE, ITEM_LABELS[item])):
                    x = d[d.roi == roi].pivot(index="subject", columns="tr", values=col)
                    t = x.columns.values * TR
                    for row in x.values:
                        ax.plot(t, row, color=colour, alpha=0.13, lw=0.6, zorder=2)
                    m = x.mean()
                    ci = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                    ax.fill_between(t, m - ci, m + ci, color=colour, alpha=0.3, lw=0, zorder=3)
                    ax.plot(t, m, color=colour, lw=2.4, zorder=4, label=lab)
                ax.axhline(10, color=INK, ls=":", lw=0.8, zorder=1)
                gm = g[(g.roi == roi) & (g.item == item) & (g.cond == cond)]
                lo = min(gm.item_prob_mean.min(), gm.control_prob_mean.min())
                hi = max(gm.item_prob_mean.max(), gm.control_prob_mean.max())
                ax.set_ylim(min(lo - 2.5, 7.5), max(hi + 2.5, 12.5))      # the means stay readable; single lines may leave the panel
                ax.set_title(f"{ROIS[roi]} (N = {x.shape[0]})", loc="left", fontsize=10, fontweight="bold")
                _decorate(ax, press_s.get(cond), i == 0)
            axes.ravel()[0].legend(frameon=False, fontsize=8, loc="lower right")
            for ax in axes[-1]:
                ax.set_xlabel("Time from retrieval cue (s)")
            fig.supylabel("Classifier probability (%)", fontsize=11)
            fig.suptitle(f"{ITEM_LABELS[item]} and its control, object trials, {CONDS[cond]}", fontsize=11)
            fig.tight_layout()
            for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
                fig.savefig(opj(plot_dir, f"{item}_probabilities_{cond}.{ext}"), **kw)
            plt.close(fig)

            # 2. item minus control
            fig, axes = _axes(len(rois))
            for i, (ax, roi) in enumerate(zip(axes.ravel(), rois)):
                x = d[d.roi == roi].pivot(index="subject", columns="tr", values="difference")
                t = x.columns.values * TR
                for row in x.values:
                    ax.plot(t, row, color=MUTED, alpha=0.2, lw=0.6, zorder=2)
                gm = g[(g.roi == roi) & (g.item == item) & (g.cond == cond)].sort_values("tr")
                m, ci = gm.difference_mean.values, gm.difference_ci95.values
                ax.axvspan(WINDOW[0] * TR, WINDOW[1] * TR, color=GRID, alpha=0.5, lw=0, zorder=0)
                ax.fill_between(t, m - ci, m + ci, color=BLUE, alpha=0.3, lw=0, zorder=3)
                ax.plot(t, m, color=BLUE, lw=2.4, zorder=4)
                ax.axhline(0, color=INK, ls=":", lw=0.8, zorder=1)
                span = max(np.abs(m - ci).max(), np.abs(m + ci).max(), 1.5) * 1.6
                ax.set_ylim(-span, span)
                for tj, mj, cj, pj in zip(t, m, ci, gm.p_holm.values):
                    if stars(pj):
                        ax.text(tj, mj + cj + 0.03 * span, stars(pj), ha="center", va="bottom", fontsize=11,
                                fontweight="bold", color=INK, zorder=5)
                r = w[(w.roi == roi) & (w.item == item) & (w.cond == cond)].iloc[0]
                ax.text(0.98, 0.04, f"4.5-12 s: {r.difference:+.2f} points\nt({int(r.n) - 1}) = {r.t:.1f}, {p_text(r.p_holm)}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
                ax.set_title(f"{ROIS[roi]} (N = {x.shape[0]})", loc="left", fontsize=10, fontweight="bold")
                _decorate(ax, press_s.get(cond), i == 0)
            for ax in axes[-1]:
                ax.set_xlabel("Time from retrieval cue (s)")
            fig.supylabel("Item minus control (percentage points)", fontsize=11)
            fig.suptitle(f"{ITEM_LABELS[item]} minus its control, object trials, {CONDS[cond]}", fontsize=11)
            fig.tight_layout()
            for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
                fig.savefig(opj(plot_dir, f"{item}_difference_{cond}.{ext}"), **kw)
            plt.close(fig)


def main():
    out, plot_dir = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
    for (sub, session), why in EXCLUDED_SESSIONS.items():
        print(f"excluded: {sub} {session} ({why})")
    t, press = load()
    s = participant_means(t)
    full = s[s.cond == "pooled"].groupby("subject").size()
    s.to_csv(opj(out, "participant_time_courses.csv"), index=False)
    press_s = {}
    if press is not None and press.rt.notna().any():
        med = press.groupby(["cond", "subject"]).rt.median().groupby("cond").median() + CUE_TO_PROMPT
        press_s = med.to_dict()
        press_s["pooled"] = float((press.groupby("subject").rt.median()).median() + CUE_TO_PROMPT)
        pd.Series(press_s, name="median_press_s_from_cue").rename_axis("cond").to_csv(opj(out, "press_times.csv"))
    g, w = group_tables(s, out)
    figures(s, g, w, press_s, plot_dir)
    c, allv = contrasts(t, out)
    retrieved_position_tables(t, out, plot_dir)
    by_position_figure(allv, plot_dir)
    by_position_figure(allv, plot_dir, only=("vis", "parietal_precuneus", "mtl"), name="by_position_main")
    print(f"{s.subject.nunique()} participants, ROIs {sorted(set(s.roi))} -> {out}")
    print(w[w.cond == "pooled"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
