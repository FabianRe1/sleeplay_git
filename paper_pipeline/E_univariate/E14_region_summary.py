#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E14 - Tables and figures of the region results reported for the univariate
      analyses (retrieval at the three time points, and learning by showing).

Collects the group results of E7, E10, E13 and E9 into the tables the text
quotes. No model is fitted here.

Retrieval (main model `ret_press`, each participant's own anatomy; participants
with all three time points: pre = evening, post1 = first morning retrieval,
post2 = repeated morning retrieval; session averages, no single runs):

  retrieval_three_time_points.csv
      per region and trial type: activity against the reference at each time
      point, the test over the three time points (repeated-measures F) and the
      three paired differences. `*_fdr` corrects across the four main regions
      (visual, parietal + precuneus, MTL, caudal ACC).
      Wording: a positive difference means the region is higher against its
      baseline (fixation and trial-type cue), not that retrieval activity is larger.
  retrieval_reference_check.csv
      the same differences for holding the answer and speaking, and for retrieval
      minus holding. Holding and speaking rise against the reference as much as
      retrieval does in precuneus and retrosplenial cortex.
  retrieval_matched_search_time.csv
      the paired differences on single trials matched for search time (E10), for
      object and scene trials.
  retrieval_time_course_windows.csv   (only if E13 was run with E13_MATCHED=1)
      response estimated without an assumed shape, object trials matched for
      search time over the three time points: windows of the course and their
      differences between time points.

Learning (E9, evening; objects only = model `events`, whole sequence = model `block`):

  learning_change_per_showing.csv
      second minus first showing (both in the same run, so the reference cancels),
      third minus second (across runs), and the slope over the three showings.

Behaviour:

  activity_vs_behaviour.csv
      across participants, Pearson and Spearman correlations of (a) the change in
      activity from the first to the second showing during learning, (b) retrieval
      activity pooled over the three time points and (c) its change from the evening
      to the first morning retrieval, with retrieval success (share of object trials
      with a button press: N_OBJECT_TRIALS per time point) and reaction time
      (response prompt to button press, trials with a press). `r_without_lowest`
      leaves out the participant with the lowest retrieval success in the evening (s41).
      Retrieval success is near ceiling for many participants (median above .9).

Input   outputs/E7_native_regions/participants_ret_press/     (E7 with E7_MODELS=ret_press)
        outputs/E10_matched_search_time/group_matched.csv, group_matched_scene.csv
        outputs/E13_response_time_course/matched/group_windows.csv
        outputs/E9_encoding_regions/participants/
        outputs/E10_matched_search_time/participants/     (trials with a press: behaviour)
Output  outputs/E14_region_summary/*.csv
Plots   plots/E14_region_summary/retrieval_by_time_point.{png,pdf}, learning_by_showing.{png,pdf},
        learning_vs_behaviour_across_participants[_third|_later|_objects|_objects_third].{png,pdf} (one point per participant;
        change as second minus first showing, as the slope over the three showings, which for three
        showings equals half of third minus first, or as the mean of the later showings minus the first)

Usage:  python E14_region_summary.py

Main version for learning (decided 2026-11-25): the WHOLE SEQUENCE (model `block`, scene onset to
rating), second minus first showing (within one run); third minus first (across runs) is the support.
The objects-only versions (model `events`) are kept for comparison.
Learning contrasts in activity_vs_behaviour.csv: rows "learning, whole sequence: ..." are the main ones;
rows "learning: ..." are the objects only. Also given: the slope over the three showings (half of third minus
first, across runs), later showings minus first, and for the whole sequence second minus first and
third minus first.

Participants. 36 have the evening retrieval, 34 all three time points. The two with the evening only
are s27 (no morning session) and s41. The morning session of s41 was run under the participant
number of s42, so its log holds the trial list and logged targets of s42. Whether that session is
used is set in config.py (EXCLUDED_SESSIONS / RELABELLED_SESSIONS; see README, Decisions); the steps
this one reads (E7, E10, E13) follow it through D1's retrieval_events().
Outputs up to the morning of 2026-11-25 are WITHOUT that session (34 participants with all three time
points; copies in logs/paper_pipeline/DE_check/out_before_s41_morning_2026-11-25/). On the afternoon of
2026-11-25 the session was taken in (retrieved items from the cue), E7, E10, E12, E13 and E15 were
rerun for s41 and this step was redone: 35 participants with all three time points. s41 has too few
trials with a button press for the matched comparisons, which are unchanged. Do not mix tables from
before and after.

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
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

OUT = str(step_output_dir(__file__))
PLOTS = str(step_plot_dir(__file__))
MAIN = ["VIS", "PARIETAL_PRECUNEUS", "MTL", "CAUDAL_ACC"]
OTHER = ["HPC", "PHPC", "ERH", "PRECUNEUS", "PARIETAL", "RSC", "POSTERIOR_CINGULATE"]
REGIONS = MAIN + OTHER
LABELS = {"VIS": "Visual", "PARIETAL_PRECUNEUS": "Parietal + precuneus", "MTL": "MTL", "CAUDAL_ACC": "Caudal ACC (control)",
          "HPC": "Hippocampus", "PHPC": "Parahippocampal", "ERH": "Entorhinal", "PRECUNEUS": "Precuneus",
          "PARIETAL": "Inferior parietal", "RSC": "Retrosplenial", "POSTERIOR_CINGULATE": "Posterior cingulate"}
CONDS = ["pre", "post1", "post2"]
COND_LABELS = ["Evening", "First\nmorning", "Repeated\nmorning"]
CONTRASTS = {"post1_minus_pre": ("post1", "pre"), "post2_minus_pre": ("post2", "pre"), "post2_minus_post1": ("post2", "post1")}
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
N_OBJECT_TRIALS = 50          # object retrieval trials per time point
# regions shown in the behaviour figures, each with a fixed colour (same in E15)
SCATTER_REGIONS = {"PHPC": "#2a78d6", "HPC": "#eb6834", "MTL": "#1baf7a", "PRECUNEUS": "#eda100", "VIS": "#e87ba4"}
BEHAV_REGIONS = ["PRECUNEUS", "POSTERIOR_CINGULATE", "RSC", "PARIETAL_PRECUNEUS", "HPC", "PHPC", "MTL", "VIS", "CAUDAL_ACC"]


def _paired(w):
    row = {}
    for name, (a, b) in CONTRASTS.items():
        d = w[a] - w[b]
        r = stats.ttest_1samp(d, 0)
        row[f"{name}_mean"], row[f"{name}_t"], row[f"{name}_p"] = d.mean(), r.statistic, r.pvalue
    return row


def _fdr_main(t, cols):
    k = t.region.isin(MAIN)
    for keys, g in t[k].groupby([c for c in ("kind", "measure") if c in t.columns]):
        for c in cols:
            t.loc[g.index, c + "_fdr"] = multipletests(g[c], method="fdr_bh")[1]
    return t


def retrieval():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(
        str(step_input_dir("E7_native_regions", "participants_ret_press")), "*_region_estimates.csv")))])
    d = d[(d.model == "ret_press") & d.cond.isin(CONDS) & d.region.isin(REGIONS)]
    wide = d.pivot_table(index=["region", "subject", "cond"], columns="regressor", values="beta").reset_index()
    wide["retrieval_minus_holding"] = wide["object"] - wide["phase_wait"]
    rows, ref, subj = [], [], []
    for reg in REGIONS:
        for kind in ("object", "scene"):
            w = wide[wide.region == reg].pivot(index="subject", columns="cond", values=kind)[CONDS].dropna()
            row = dict(region=reg, kind=kind, n=len(w))
            for c in CONDS:
                row[c], row[c + "_t"] = w[c].mean(), stats.ttest_1samp(w[c], 0).statistic
            long = w.reset_index().melt(id_vars="subject", var_name="cond", value_name="beta")
            a = AnovaRM(long, "beta", "subject", within=["cond"]).fit().anova_table
            row["F"], row["F_p"] = a["F Value"].iloc[0], a["Pr > F"].iloc[0]
            row.update(_paired(w))
            rows.append(row)
            subj.append(w.assign(region=reg, kind=kind).reset_index())
        for measure, col in (("retrieval (object trials)", "object"), ("holding the answer", "phase_wait"),
                             ("speaking", "phase_speak"), ("retrieval minus holding", "retrieval_minus_holding")):
            w = wide[wide.region == reg].pivot(index="subject", columns="cond", values=col)[CONDS].dropna()
            ref.append(dict(region=reg, measure=measure, n=len(w), **{c: w[c].mean() for c in CONDS}, **_paired(w)))
    t = _fdr_main(pd.DataFrame(rows), ["F_p"] + [f"{c}_p" for c in CONTRASTS])
    t.to_csv(opj(OUT, "retrieval_three_time_points.csv"), index=False)
    _fdr_main(pd.DataFrame(ref), [f"{c}_p" for c in CONTRASTS]).to_csv(opj(OUT, "retrieval_reference_check.csv"), index=False)

    parts = []
    for kind, tag in (("object", ""), ("scene", "_scene")):
        f = opj(str(step_input_dir("E10_matched_search_time")), f"group_matched{tag}.csv")
        if os.path.exists(f):
            g = pd.read_csv(f)
            g = g[(g.form == "press") & g.region.isin(REGIONS) & g.comparison.isin(CONTRASTS)]
            parts.append(g.assign(kind=kind)[["kind", "region", "comparison", "pairs_mean", "rt_diff_all", "rt_diff_matched",
                                                "all_trials_mean", "all_trials_t", "all_trials_p",
                                                "matched_n", "matched_mean", "matched_t", "matched_p"]])
    if parts:
        m = pd.concat(parts, ignore_index=True)
        k = m.region.isin(MAIN)
        for _, g in m[k].groupby(["kind", "comparison"]):
            m.loc[g.index, "matched_p_fdr"] = multipletests(g.matched_p, method="fdr_bh")[1]
        m.to_csv(opj(OUT, "retrieval_matched_search_time.csv"), index=False)
    try:
        w = pd.read_csv(opj(str(step_input_dir("E13_response_time_course", "matched")), "group_windows.csv"))
        w[(w.kind == "object_matched") & w.region.isin(REGIONS)].to_csv(opj(OUT, "retrieval_time_course_windows.csv"), index=False)
    except FileNotFoundError:
        print("E13 (matched) not found: retrieval_time_course_windows.csv not written")

    s = pd.concat(subj)
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.2))
    for ax, reg in zip(axes.ravel(), MAIN + ["HPC", "PHPC", "PRECUNEUS", "PARIETAL", "RSC", "POSTERIOR_CINGULATE", "ERH"]):
        for kind, colour, dx in (("object", BLUE, -0.08), ("scene", ORANGE, 0.08)):
            g = s[(s.region == reg) & (s.kind == kind)]
            ax.errorbar(np.arange(3) + dx, g[CONDS].mean(), yerr=g[CONDS].sem(), color=colour, marker="o", lw=1.8,
                        capsize=3, label=f"{kind} trials")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(COND_LABELS, fontsize=8)
        ax.set_title(LABELS[reg], fontsize=10, loc="left"); ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Signal change against baseline (%)")
    fig.suptitle("Retrieval (cue to button press) at the three time points; mean and SEM across participants", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(opj(PLOTS, f"retrieval_by_time_point.{ext}"), dpi=300)
    plt.close(fig)
    return t


def learning():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(
        str(step_input_dir("E9_encoding_regions", "participants")), "*_region_estimates.csv")))])
    rows, curves = [], []
    for what, model, pref in (("objects only", "events", "object_rep"), ("whole sequence", "block", "sequence_rep")):
        for reg in REGIONS:
            w = d[(d.model == model) & d.regressor.str.startswith(pref) & (d.region == reg)].pivot(
                index="subject", columns="regressor", values="beta").dropna()
            r1, r2, r3 = (w[f"{pref}{i}"] for i in (1, 2, 3))
            row = dict(what=what, region=reg, n=len(w), showing1=r1.mean(), showing2=r2.mean(), showing3=r3.mean())
            for name, v in (("second_minus_first", r2 - r1), ("third_minus_second", r3 - r2), ("slope_per_showing", (r3 - r1) / 2)):
                r = stats.ttest_1samp(v, 0)
                row[f"{name}_mean"], row[f"{name}_t"], row[f"{name}_p"] = v.mean(), r.statistic, r.pvalue
            rows.append(row)
            curves.append(pd.DataFrame(dict(what=what, region=reg, s1=r1, s2=r2, s3=r3)))
    t = pd.DataFrame(rows)
    t.to_csv(opj(OUT, "learning_change_per_showing.csv"), index=False)
    c = pd.concat(curves)
    show = ["HPC", "PHPC", "ERH", "MTL", "VIS", "PRECUNEUS", "POSTERIOR_CINGULATE", "RSC", "PARIETAL", "CAUDAL_ACC"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, reg in zip(axes.ravel(), show):
        for what, colour in (("whole sequence", BLUE), ("objects only", MUTED)):
            g = c[(c.region == reg) & (c.what == what)][["s1", "s2", "s3"]]
            ax.errorbar(range(3), g.mean(), yerr=g.sem(), color=colour, marker="o", lw=1.8, capsize=3, label=what)
        x = t[(t.region == reg) & (t.what == "whole sequence")].iloc[0]
        ax.set_title(f"{LABELS[reg]}\n2nd - 1st: t = {x.second_minus_first_t:.1f}", fontsize=10, loc="left")
        ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(["1st", "2nd", "3rd"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Signal change against fixation (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Showing of the sequence")
    fig.suptitle("Learning, by showing (1st and 2nd in one run, 3rd in the next); mean and SEM across participants", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(opj(PLOTS, f"learning_by_showing.{ext}"), dpi=300)
    plt.close(fig)
    return t


def behaviour():
    tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(
        str(step_input_dir("E10_matched_search_time", "participants")), "*_trials.csv")))])
    g = tr.groupby(["subject", "cond"]).rt.agg(["size", "mean"]).unstack("cond")
    b = pd.DataFrame({"success_evening": g[("size", "pre")] / N_OBJECT_TRIALS,
                      "success_morning": (g[("size", "post1")] + g[("size", "post2")]) / (2 * N_OBJECT_TRIALS),
                      "rt_evening": g[("mean", "pre")], "rt_morning": g[("mean", "post1")].add(g[("mean", "post2")]) / 2,
                      "success_change": (g[("size", "post1")] - g[("size", "pre")]) / N_OBJECT_TRIALS,
                      "rt_change": g[("mean", "post1")] - g[("mean", "pre")]})
    b["success_pooled"] = (g["size"][CONDS].sum(axis=1, min_count=3)) / (3 * N_OBJECT_TRIALS)
    b["rt_pooled"] = g["mean"][CONDS].mean(axis=1, skipna=False)
    enc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(
        str(step_input_dir("E9_encoding_regions", "participants")), "*_region_estimates.csv")))])
    blk = enc[(enc.model == "block") & enc.regressor.isin(["sequence_rep1", "sequence_rep2", "sequence_rep3"])].pivot_table(
        index=["region", "subject"], columns="regressor", values="beta").dropna()
    whole = {"second minus first showing": (blk.sequence_rep2 - blk.sequence_rep1).unstack("region"),
             "third minus first showing": (blk.sequence_rep3 - blk.sequence_rep1).unstack("region"),
             "later showings minus first": ((blk.sequence_rep2 + blk.sequence_rep3) / 2 - blk.sequence_rep1).unstack("region")}
    enc = enc[(enc.model == "events") & enc.regressor.isin(["object_rep1", "object_rep2", "object_rep3"])].pivot_table(
        index=["region", "subject"], columns="regressor", values="beta").dropna()
    # three ways to express the change over the showings (file tag, label, value)
    changes = {"": ("second minus first showing", "Change, 2nd minus 1st showing (%)", enc.object_rep2 - enc.object_rep1),
               "_slope": ("slope over the three showings", "Slope over the three showings (% per showing)",
                          (enc.object_rep3 - enc.object_rep1) / 2),
               "_later": ("later showings minus first", "Change, mean of 2nd and 3rd minus 1st showing (%)",
                          (enc.object_rep2 + enc.object_rep3) / 2 - enc.object_rep1)}
    changes = {k: (a, lab, v.unstack("region")) for k, (a, lab, v) in changes.items()}
    ret = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(
        str(step_input_dir("E7_native_regions", "participants_ret_press")), "*_region_estimates.csv")))])
    ret = ret[(ret.model == "ret_press") & (ret.regressor == "object") & ret.cond.isin(CONDS)].pivot_table(
        index=["region", "subject"], columns="cond", values="beta")
    sets = [(f"learning: {a}", v, ["success_evening", "success_morning", "rt_evening", "rt_morning"]) for a, _, v in changes.values()] + [
            (f"learning, whole sequence: {a}", v, ["success_evening", "rt_evening"]) for a, v in whole.items()] + [
            ("retrieval: pooled over time points", ret[CONDS].mean(axis=1, skipna=False).unstack("region"), ["success_pooled", "rt_pooled"]),
            ("retrieval: first morning minus evening", (ret.post1 - ret.pre).unstack("region"), ["success_change", "rt_change"])]
    lowest = b.success_evening.idxmin()          # s41 (14 % in the evening)
    rows = []
    for what, x, behs in sets:
        for reg in BEHAV_REGIONS:
            for beh in behs:
                d = pd.concat([x[reg], b[beh]], axis=1, keys=["x", "y"]).dropna()
                r, pv = stats.pearsonr(d.x, d.y)
                rho, prho = stats.spearmanr(d.x, d.y)
                d2 = d.drop(index=lowest, errors="ignore")
                rows.append(dict(activity=what, region=reg, behaviour=beh, n=len(d), r=r, p=pv, rho=rho, p_rho=prho,
                                 r_without_lowest=stats.pearsonr(d2.x, d2.y)[0]))
    t = pd.DataFrame(rows)
    for _, gq in t.groupby(["activity", "behaviour"]):
        t.loc[gq.index, "p_fdr"] = multipletests(gq.p, method="fdr_bh")[1]
    t.to_csv(opj(OUT, "activity_vs_behaviour.csv"), index=False)
    b.to_csv(opj(OUT, "behaviour_per_participant.csv"))

    # one point per participant: change during learning against evening retrieval, one figure per definition
    # main version: whole sequence (scene onset to rating); the objects-only versions carry the tag _objects
    figures = {"": ("whole sequence, second minus first showing (main contrast, within one run)",
                    "2nd minus 1st showing (%)", whole["second minus first showing"]),
               "_third": ("whole sequence, third minus first showing (across runs)",
                          "3rd minus 1st showing (%)", whole["third minus first showing"]),
               "_later": ("whole sequence, mean of the later showings minus the first",
                          "Mean of 2nd and 3rd minus 1st showing (%)", whole["later showings minus first"]),
               "_objects": ("objects only, second minus first showing", "2nd minus 1st showing (%)", changes[""][2]),
               "_objects_third": ("objects only, third minus first showing", "3rd minus 1st showing (%)", 2 * changes["_slope"][2])}
    for tag, (what, xlab, x) in figures.items():
        fig, axes = plt.subplots(2, len(SCATTER_REGIONS), figsize=(3.3 * len(SCATTER_REGIONS), 6.6), sharey="row")
        for i, (beh, ylab) in enumerate((("success_evening", "Retrieval success (evening)"), ("rt_evening", "Reaction time (s, evening)"))):
            for ax, (reg, colour) in zip(axes[i], SCATTER_REGIONS.items()):
                d = pd.concat([x[reg], b[beh]], axis=1, keys=["x", "y"]).dropna()
                ax.scatter(d.x, d.y, s=42, color=colour, edgecolor="white", linewidth=1.0, zorder=3)
                k, c0 = np.polyfit(d.x, d.y, 1)
                xs = np.array([d.x.min(), d.x.max()])
                ax.plot(xs, k * xs + c0, color=colour, lw=2, zorder=2)
                r, pv = stats.pearsonr(d.x, d.y)
                ax.text(0.03, 0.04 if i == 0 else 0.93, f"r = {r:.2f}, p = {pv:.3f}".replace("0.", "."), transform=ax.transAxes,
                        fontsize=9, color=INK, zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
                ax.axvline(0, color=MUTED, lw=0.6, ls=":", zorder=1)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(LABELS[reg], fontsize=11, loc="left", color=INK)
                else:
                    ax.set_xlabel(xlab, fontsize=9)
            axes[i, 0].set_ylabel(ylab)
        fig.suptitle(f"Across participants, {what}: one point per participant "
                     f"(n = {int(x[list(SCATTER_REGIONS)[0]].notna().sum())}); line: least-squares fit", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(opj(PLOTS, f"learning_vs_behaviour_across_participants{tag}.{ext}"), dpi=300)
        plt.close(fig)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    r = retrieval()
    print(r[r.kind == "object"][["region", "n", "pre", "post1", "post2", "F", "F_p", "post1_minus_pre_t", "post2_minus_pre_t",
                                 "post2_minus_post1_t"]].round(3).to_string(index=False))
    l = learning()
    print(l[l.what == "objects only"][["region", "n", "second_minus_first_t", "third_minus_second_t", "slope_per_showing_t"]]
          .round(2).to_string(index=False))
    bh = behaviour()
    print(bh.round(3).to_string(index=False))
    print(f"-> {OUT}\n-> {PLOTS}")
