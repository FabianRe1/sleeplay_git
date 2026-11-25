#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C10 - The slope over the retrieval trial: group time courses and one example trial.

B8 regresses, at every volume of a trial, the five classifier probabilities of a
sequence on their position (1-5). The slope is negative when early items have
the higher probability and positive when late items have it. B9 fits one
sinusoid cycle to a trial's ten slopes. This step shows what those slopes look
like when they are not picked by hand:

1. Group time course of the SIGNED slope, for the retrieved and for the control
   sequence: mean over a participant's object trials per volume, then across
   participants. All trials together, and separately by the position of the
   retrieved item (1-5). If retrieval steps through the sequence in order, the
   slope should first be negative and then positive within a trial.
   The same for the ABSOLUTE slope (the absolute value of each trial's slope,
   then averaged): a slope in either direction counts, so it does not cancel
   across trials, and the retrieved sequence is compared with the control.
2. One example trial for the three-panel illustration (probabilities -> slope ->
   sinusoid), chosen by a rule and not by eye (see `choose_example`).

Time 0 is the onset of the retrieval cue; volumes are 1.5 s apart.
Time points: pre = evening; post1 / post2 = morning trials 1-70 / 71-140;
pooled = all trials of a participant.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
        outputs/B9_sinusoid_summary/thesis/<roi>/<roi>_windowed_fit_results_<session>.csv   (example only)
Output  outputs/C10_slope_time_courses/participant_slopes.csv, group_slopes.csv
        outputs/C10_slope_time_courses/example_trial.csv        which trial was chosen and why
Plots   plots/C10_slope_time_courses/slope_time_course.{png,pdf}
            retrieved and control sequence, one thin line per participant, group
            mean, 95 % CI; stars: retrieved minus control per volume (Holm across volumes)
        plots/C10_slope_time_courses/abs_slope_time_course[_<time point>].{png,pdf}
            absolute slope, retrieved minus control: participants, group mean, 95 % CI,
            stars per volume (Holm) and the test of the mean over 4.5-12 s
        outputs/C10_slope_time_courses/group_abs_slope_window.csv   that window test per ROI and time point
        plots/C10_slope_time_courses/slope_by_retrieved_position.{png,pdf}
            the same means, split by the position of the retrieved item
        plots/C10_slope_time_courses/example_trial.{png,pdf}

Usage:  python C10_slope_time_courses.py

Not a port of the original pipeline: added 2026-11-25.
"""

from __future__ import annotations

import ast
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
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TR, is_excluded, step_input_dir, step_output_dir, step_plot_dir

ROIS = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL", "hpc": "Hippocampus",
        "somato": "Somatosensory (control)", "caudal_acc": "Caudal ACC (control)"}
EXAMPLE_ROI = "parietal_precuneus"
POST1_MAX_TRIAL = 70
CONDS = {"pooled": "all time points", "pre": "evening", "post1": "first morning retrieval",
         "post2": "repeated morning retrieval"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SAFE = {"nan": float("nan"), "inf": float("inf")}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
    """The tapered single cycle of B9 (preset `thesis`); copied so that B9 need not be imported."""
    t = np.asarray(t, float)
    y = np.full_like(t, offset)
    active = (t >= onset) & (t <= onset + duration)
    window = np.zeros_like(t)
    window[active] = 1.0
    taper = taper_frac * duration
    if taper > 0:
        rise = active & (t < onset + taper)
        window[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - onset) / taper))
        fall = active & (t > onset + duration - taper)
        window[fall] = 0.5 * (1 - np.cos(np.pi * (onset + duration - t[fall]) / taper))
    return y + amplitude * np.sin(2 * np.pi / duration * (t - onset)) * window


def load_roi(roi):
    parts = []
    for session in ("presleep", "postsleep"):
        try:
            path = opj(str(step_input_dir("B8_sequentiality_soda", opj(session, "SODA_results_csv"))),
                       f"{roi}_soda_results_per_combination.csv")
            d = pd.read_csv(path, usecols=["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                                           "retrieved_soda", "control_soda", "retrieved_probs", "control_probs"])
        except (FileNotFoundError, ValueError):
            continue
        d = d[d.object_scene_retrieval == "object"]
        d = d[[not is_excluded(s, session) for s in d.subject]]
        d["session"] = session
        d["cond"] = "pre" if session == "presleep" else np.where(d.trial <= POST1_MAX_TRIAL, "post1", "post2")
        parts.append(d)
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d[["retrieved_soda", "control_soda"]] *= 100.0                 # percentage points per position
    d["retrieved_abs"], d["control_abs"] = d.retrieved_soda.abs(), d.control_soda.abs()
    return d.assign(roi=roi)


def participant_means(d):
    keys = ["roi", "subject", "tr"]
    cols = dict(retrieved=("retrieved_soda", "mean"), control=("control_soda", "mean"),
                retrieved_abs=("retrieved_abs", "mean"), control_abs=("control_abs", "mean"))
    a = d.groupby(keys + ["cond"]).agg(**cols).reset_index().assign(position="all")
    b = d.groupby(keys).agg(**cols).reset_index().assign(cond="pooled", position="all")
    c = d.groupby(keys + ["retrieved_position"]).agg(**cols).reset_index().rename(columns={"retrieved_position": "position"})
    c = c.assign(cond="pooled", position=c.position.astype(int).astype(str))
    s = pd.concat([a, b, c], ignore_index=True)
    s["difference"] = s.retrieved - s.control
    s["difference_abs"] = s.retrieved_abs - s.control_abs
    return s


def group_table(s):
    rows = []
    for (roi, cond, pos), g in s.groupby(["roi", "cond", "position"]):
        w = {c: g.pivot(index="subject", columns="tr", values=c).dropna()
             for c in ("retrieved", "control", "difference", "retrieved_abs", "control_abs", "difference_abs")}
        n = len(w["difference"])
        k = stats.t.ppf(0.975, n - 1) / np.sqrt(n)
        tt = {c: stats.ttest_1samp(w[c], 0, axis=0) for c in w}
        ph = multipletests(tt["difference"].pvalue, method="holm")[1]
        pha = multipletests(tt["difference_abs"].pvalue, method="holm")[1]
        for i, tr in enumerate(w["difference"].columns):
            rows.append(dict(roi=roi, cond=cond, position=pos, tr=tr, time_s=tr * TR, n=n,
                             **{f"{c}_mean": w[c][tr].mean() for c in w}, **{f"{c}_ci95": w[c][tr].std(ddof=1) * k for c in w},
                             **{f"{c}_t": tt[c].statistic[i] for c in w}, **{f"{c}_p": tt[c].pvalue[i] for c in w},
                             difference_p_holm=ph[i], difference_abs_p_holm=pha[i]))
    return pd.DataFrame(rows)


def fig_time_course(s, g, plot_dir, cond="pooled"):
    rois = [r for r in ROIS if r in set(s.roi)]
    ncol = 3 if len(rois) > 4 else len(rois)
    nrow = int(np.ceil(len(rois) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow), squeeze=False, sharex=True)
    for ax in axes.ravel()[len(rois):]:
        ax.set_visible(False)
    for i, (ax, roi) in enumerate(zip(axes.ravel(), rois)):
        d = s[(s.roi == roi) & (s.cond == cond) & (s.position == "all")]
        gm = g[(g.roi == roi) & (g.cond == cond) & (g.position == "all")].sort_values("tr")
        t = gm.time_s.values
        for col, colour, lab in (("control", ORANGE, "Control sequence"), ("retrieved", BLUE, "Retrieved sequence")):
            x = d.pivot(index="subject", columns="tr", values=col)
            for row in x.values:
                ax.plot(t, row, color=colour, alpha=0.12, lw=0.6, zorder=2)
            m, ci = gm[f"{col}_mean"].values, gm[f"{col}_ci95"].values
            ax.fill_between(t, m - ci, m + ci, color=colour, alpha=0.3, lw=0, zorder=3)
            ax.plot(t, m, color=colour, lw=2.4, zorder=4, label=lab)
        span = max(np.abs(gm[[f"{c}_mean" for c in ("retrieved", "control")]].values).max()
                   + gm[[f"{c}_ci95" for c in ("retrieved", "control")]].values.max(), 0.2) * 1.8
        ax.set_ylim(-span, span)                                     # single participants may leave the panel
        for tj, pj in zip(t, gm.difference_p_holm.values):
            if stars(pj):
                ax.text(tj, 0.88 * span, stars(pj), ha="center", fontsize=11, fontweight="bold")
        ax.axhline(0, color=INK, ls=":", lw=0.8, zorder=1)
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        ax.set_title(f"{ROIS[roi]} (N = {int(gm.n.iloc[0])})", loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("Time from retrieval cue (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Slope (points per position)")
    fig.suptitle(f"Signed slope over the five items, object trials, {CONDS[cond]} "
                 "(negative: early items higher; positive: late items higher)", fontsize=11)
    fig.tight_layout()
    name = "slope_time_course" + ("" if cond == "pooled" else f"_{cond}")
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


WINDOW = (3, 8)                       # volumes, 4.5-12 s, as in C9


def abs_window_table(s, out):
    rows = []
    d = s[(s.position == "all") & (s.tr >= WINDOW[0]) & (s.tr <= WINDOW[1])]
    v = d.groupby(["roi", "cond", "subject"])[["retrieved_abs", "control_abs", "difference_abs"]].mean()
    for (roi, cond), g in v.groupby(level=["roi", "cond"]):
        r = stats.ttest_1samp(g.difference_abs, 0)
        rows.append(dict(roi=roi, cond=cond, n=len(g), retrieved_abs=g.retrieved_abs.mean(), control_abs=g.control_abs.mean(),
                         difference=g.difference_abs.mean(), t=r.statistic, p=r.pvalue))
    wide = v.difference_abs.unstack("cond")
    for a, b in (("post1", "pre"), ("post2", "post1"), ("post2", "pre")):
        for roi, g in (wide[a] - wide[b]).dropna().groupby(level="roi"):
            r = stats.ttest_1samp(g, 0)
            rows.append(dict(roi=roi, cond=f"{a} - {b}", n=len(g), difference=g.mean(), t=r.statistic, p=r.pvalue))
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("cond").p.transform(lambda p: multipletests(p, method="holm")[1])
    t.to_csv(opj(out, "group_abs_slope_window.csv"), index=False)
    return t


def fig_abs_time_course(s, g, win, plot_dir, cond="pooled"):
    rois = [r for r in ROIS if r in set(s.roi)]
    ncol = 3 if len(rois) > 4 else len(rois)
    nrow = int(np.ceil(len(rois) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow), squeeze=False, sharex=True)
    for ax in axes.ravel()[len(rois):]:
        ax.set_visible(False)
    for ax, roi in zip(axes.ravel(), rois):
        d = s[(s.roi == roi) & (s.cond == cond) & (s.position == "all")]
        gm = g[(g.roi == roi) & (g.cond == cond) & (g.position == "all")].sort_values("tr")
        t = gm.time_s.values
        x = d.pivot(index="subject", columns="tr", values="difference_abs")
        for row in x.values:
            ax.plot(t, row, color=MUTED, alpha=0.2, lw=0.6, zorder=2)
        m, ci = gm.difference_abs_mean.values, gm.difference_abs_ci95.values
        ax.axvspan(WINDOW[0] * TR, WINDOW[1] * TR, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.fill_between(t, m - ci, m + ci, color=BLUE, alpha=0.3, lw=0, zorder=3)
        ax.plot(t, m, color=BLUE, lw=2.4, zorder=4)
        span = max(np.abs(m - ci).max(), np.abs(m + ci).max(), 0.2) * 1.9
        ax.set_ylim(-span, span)
        for tj, mj, cj, pj in zip(t, m, ci, gm.difference_abs_p_holm.values):
            if stars(pj):
                ax.text(tj, mj + cj + 0.03 * span, stars(pj), ha="center", fontsize=11, fontweight="bold")
        r = win[(win.roi == roi) & (win.cond == cond)].iloc[0]
        ax.text(0.98, 0.04, f"4.5-12 s: {r.retrieved_abs:.2f} against {r.control_abs:.2f}\n"
                f"difference {r.difference:+.2f}, t({int(r.n) - 1}) = {r.t:.1f}, p = {r.p_holm:.3f}".replace("p = 0.", "p = ."),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.axhline(0, color=INK, ls=":", lw=0.8, zorder=1)
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        ax.set_title(f"{ROIS[roi]} (N = {int(gm.n.iloc[0])})", loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time from retrieval cue (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Absolute slope, retrieved minus control")
    fig.suptitle(f"Absolute slope over the five items (points per position), retrieved minus control sequence, object trials, {CONDS[cond]}", fontsize=11)
    fig.tight_layout()
    name = "abs_slope_time_course" + ("" if cond == "pooled" else f"_{cond}")
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


def fig_by_position(g, plot_dir, rois=("vis", "parietal_precuneus", "mtl")):
    rois = [r for r in rois if r in set(g.roi)]
    fig, axes = plt.subplots(len(rois), 5, figsize=(15, 2.7 * len(rois)), squeeze=False, sharex=True)
    for i, roi in enumerate(rois):
        gr = g[(g.roi == roi) & (g.cond == "pooled") & (g.position != "all")]
        span = (np.abs(gr[["retrieved_mean", "control_mean"]].values).max() + gr[["retrieved_ci95", "control_ci95"]].values.max()) * 1.15
        for j in range(5):
            ax, gm = axes[i, j], gr[gr.position == str(j + 1)].sort_values("tr")
            t = gm.time_s.values
            for col, colour in (("control", ORANGE), ("retrieved", BLUE)):
                m, ci = gm[f"{col}_mean"].values, gm[f"{col}_ci95"].values
                ax.fill_between(t, m - ci, m + ci, color=colour, alpha=0.3, lw=0)
                ax.plot(t, m, color=colour, lw=2.2, label={"control": "Control sequence", "retrieved": "Retrieved sequence"}[col])
            ax.axhline(0, color=INK, ls=":", lw=0.8)
            ax.set_ylim(-span, span)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"Retrieved item at position {j + 1}", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ROIS[roi].replace(" + ", " +\n"), fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time from retrieval cue (s)")
    fig.suptitle("Signed slope (percentage points per position) by the position of the retrieved item; mean and 95 % CI", fontsize=11)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"slope_by_retrieved_position.{ext}"), **kw)
    plt.close(fig)


def choose_example(d, out):
    """One trial for the illustration, by rule.

    Among the object trials of EXAMPLE_ROI whose sinusoid fit succeeded:
      - the fit explains at least 80 % of the slope's variance,
      - its absolute amplitude lies between the 60th and the 90th percentile (clear, not extreme).
    Of these, the trial is taken whose five probabilities at the volume with the steepest slope
    are ordered most monotonically by position (rank correlation), then with the lowest single
    highest probability, then with the best fit: the trial in which the slope is carried least
    by one item. example_trial.csv says how many candidates were fully monotonic.
    """
    fits = []
    for session in ("presleep", "postsleep"):
        try:
            f = pd.read_csv(opj(str(step_input_dir("B9_sinusoid_summary", opj("thesis", EXAMPLE_ROI))),
                                f"{EXAMPLE_ROI}_windowed_fit_results_{session}.csv"))
        except FileNotFoundError:
            continue
        fits.append(f.assign(session=session))
    if not fits:
        return None
    f = pd.concat(fits, ignore_index=True)
    f = f[(f.object_scene_retrieval == "object") & (f.retrieved_w_fit_success == True)]          # noqa: E712
    amp = f.retrieved_w_amplitude.abs()
    cand = f[(f.retrieved_w_r_squared >= 0.8) & (amp >= amp.quantile(0.6)) & (amp <= amp.quantile(0.9))]
    cand = cand.sort_values("retrieved_w_r_squared", ascending=False)
    rows = []
    for _, r in cand.iterrows():
        tr = d[(d.subject == r.subject) & (d.session == r.session) & (d.trial == r.trial)].sort_values("tr")
        if len(tr) != 10:
            continue
        k = int(tr.retrieved_soda.abs().values.argmax())
        probs = np.array(eval(tr.retrieved_probs.iloc[k], {"__builtins__": {}}, SAFE), float) * 100.0
        rows.append((abs(stats.spearmanr(np.arange(5), probs)[0]), -probs.max(), r.retrieved_w_r_squared, r.name))
    if not rows:
        return None
    rows.sort(reverse=True)                      # most monotonic first, then the least peaky, then the best fit
    rho, neg_max, _, idx = rows[0]
    r = cand.loc[idx]
    tr = d[(d.subject == r.subject) & (d.session == r.session) & (d.trial == r.trial)].sort_values("tr")
    k = int(tr.retrieved_soda.abs().values.argmax())
    probs = np.array(eval(tr.retrieved_probs.iloc[k], {"__builtins__": {}}, SAFE), float) * 100.0
    ctrl = np.array(eval(tr.control_probs.iloc[k], {"__builtins__": {}}, SAFE), float) * 100.0
    info = dict(roi=EXAMPLE_ROI, subject=r.subject, session=r.session, trial=int(r.trial),
                retrieved_position=int(r.retrieved_position), tr_of_steepest_slope=int(tr.tr.iloc[k]),
                r_squared=r.retrieved_w_r_squared, amplitude=r.retrieved_w_amplitude,
                rank_correlation_with_position=float(stats.spearmanr(np.arange(5), probs)[0]), highest_probability=float(probs.max()),
                n_candidates=len(rows), n_candidates_fully_monotonic=int(sum(x[0] > 0.999 for x in rows)),
                n_candidates_fully_monotonic_and_no_item_above_50=int(sum(x[0] > 0.999 and -x[1] < 50 for x in rows)))
    pd.DataFrame([info]).to_csv(opj(out, "example_trial.csv"), index=False)
    return dict(info=info, tr=tr, k=k, probs=probs, ctrl=ctrl, fit=r)


def peakiness(d, out):
    """How the slope at a trial's steepest volume comes about: share of trials with a monotonic order of the
    five probabilities, and how large the highest probability is. Retrieved sequence, all object trials."""
    rows = []
    for roi_d in d.values():
        k = roi_d.loc[roi_d.groupby(["subject", "session", "trial"]).retrieved_soda.apply(lambda x: x.abs().idxmax())]
        P = np.array([eval(x, {"__builtins__": {}}, SAFE) for x in k.retrieved_probs], float) * 100.0
        rho = np.array([abs(stats.spearmanr(np.arange(5), p)[0]) for p in P])
        srt = np.sort(P, axis=1)
        rows.append(dict(roi=roi_d.roi.iloc[0], n_trials=len(P), share_monotonic=float((rho > 0.999).mean()),
                         expected_by_chance=2 / 120, median_highest_probability=float(np.median(srt[:, -1])),
                         median_second_highest=float(np.median(srt[:, -2])),
                         share_highest_above_50=float((srt[:, -1] > 50).mean()),
                         share_highest_is_first_or_last_position=float(np.isin(P.argmax(axis=1), [0, 4]).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(opj(out, "slope_peakiness.csv"), index=False)
    return t


def fig_example(ex, plot_dir):
    tr, k, r = ex["tr"], ex["k"], ex["fit"]
    t = tr.tr.values * TR
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    pos = np.arange(1, 6)
    for vals, colour, lab in ((ex["ctrl"], ORANGE, "Control sequence"), (ex["probs"], BLUE, "Retrieved sequence")):
        b, a = np.polyfit(pos, vals, 1)
        ax.plot(pos, a + b * pos, color=colour, lw=1.6)
        ax.scatter(pos, vals, color=colour, s=45, zorder=3, label=f"{lab}: slope {b:+.1f}")
    ax.scatter([ex["info"]["retrieved_position"]], [ex["probs"][ex["info"]["retrieved_position"] - 1]], s=170,
               facecolors="none", edgecolors=INK, lw=1.2, zorder=4)
    ax.set_xticks(pos); ax.set_xlabel("Position in the sequence"); ax.set_ylabel("Classifier probability (%)")
    ax.set_title(f"1 · Five probabilities at {t[k]:.1f} s (circle: retrieved item)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(t, tr.control_soda, color=ORANGE, lw=2, marker="o", ms=4, label="Control sequence")
    ax.plot(t, tr.retrieved_soda, color=BLUE, lw=2, marker="o", ms=4, label="Retrieved sequence")
    ax.axvline(t[k], color=MUTED, ls=":", lw=0.8); ax.axhline(0, color=INK, ls=":", lw=0.8)
    ax.set_xlabel("Time from retrieval cue (s)"); ax.set_ylabel("Slope (points per position)")
    ax.set_title("2 · The slope at each of the ten volumes", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    tf = np.linspace(tr.tr.min(), tr.tr.max(), 300)
    fit = windowed_sinusoid(tf, r.retrieved_w_amplitude, r.retrieved_w_duration, r.retrieved_w_onset, r.retrieved_w_offset) * 100.0
    cfit = windowed_sinusoid(tf, r.control_w_amplitude, r.control_w_duration, r.control_w_onset, r.control_w_offset) * 100.0
    ax.plot(t, tr.control_soda, color=ORANGE, lw=0, marker="o", ms=5)
    ax.plot(tf * TR, cfit, color=ORANGE, lw=1.8,
            label=f"Control: amplitude {abs(r.control_w_amplitude) * 100:.1f}, R² = {r.control_w_r_squared:.2f}")
    ax.plot(t, tr.retrieved_soda, color=BLUE, lw=0, marker="o", ms=5)
    ax.plot(tf * TR, fit, color=BLUE, lw=1.8,
            label=f"Retrieved: amplitude {abs(r.retrieved_w_amplitude) * 100:.1f}, R² = {r.retrieved_w_r_squared:.2f}")
    ax.axhline(0, color=INK, ls=":", lw=0.8)
    ax.set_xlabel("Time from retrieval cue (s)"); ax.set_ylabel("Slope (points per position)")
    ax.set_title("3 · One sinusoid cycle per sequence", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    i = ex["info"]
    fig.suptitle(f"One trial ({i['subject']}, {i['session']}, trial {i['trial']}, retrieved position {i['retrieved_position']}), "
                 f"{ROIS[EXAMPLE_ROI]}, chosen by rule", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"example_trial.{ext}"), **kw)
    plt.close(fig)


def main():
    out, plot_dir = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
    data = {}
    for roi in ROIS:
        d = load_roi(roi)
        if d is not None:
            data[roi] = d
            print(f"{roi}: {d.subject.nunique()} participants, {d.groupby(['subject', 'session']).trial.nunique().sum()} trials", flush=True)
    s = pd.concat([participant_means(d) for d in data.values()], ignore_index=True)
    s.to_csv(opj(out, "participant_slopes.csv"), index=False)
    g = group_table(s)
    g.to_csv(opj(out, "group_slopes.csv"), index=False)
    win = abs_window_table(s, out)
    for cond in CONDS:
        fig_time_course(s, g, plot_dir, cond)
        fig_abs_time_course(s, g, win, plot_dir, cond)
    fig_by_position(g, plot_dir)
    if EXAMPLE_ROI in data:
        ex = choose_example(data[EXAMPLE_ROI], out)
        if ex is None:
            print("no trial met the rule for the example")
        else:
            fig_example(ex, plot_dir)
            print("example trial:", ex["info"])
    print(peakiness(data, out).round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
