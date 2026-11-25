#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C11 - Sinusoid amplitude of the slope, by which item is retrieved and by which
      items enter the slope.

The sequentiality measure is the amplitude of one sinusoid cycle fitted to the
ten slopes of a trial (B8 slope, B9 fit, preset `thesis`), for the retrieved
sequence minus the control sequence. C9 showed, item by item, where the
classifier evidence sits (retrieved item and its predecessor). This step asks
the same of the amplitude: for each position of the retrieved item, how does
the amplitude depend on the items that go into the slope?

Per object trial and volume, the five probabilities of the retrieved and of the
control sequence are regressed on position (as B8), for these sets of items:
    all five items                              (the measure of the thesis)
    items 1 .. q, for q = 2, 3, 4, 5            (slope up to item q; q = retrieved
                                                 position is the "truncated" measure)
    all items except item q, for q = 1 .. 5     (item q left out; q = retrieved
                                                 position is the "without target" control)
    sets relative to the retrieved item          without both neighbours; without the retrieved
    (run with C11_ONLY_RELATIVE=1)               item and its predecessor / successor / both
To each slope time course the tapered single cycle of B9 is fitted with B9's own
function. The value is the absolute amplitude, in percentage points per position.
Amplitudes of slopes over fewer items are larger and noisier in both sequences,
so columns of the "up to item q" matrix are not on one scale; the difference
retrieved minus control is what is tested.

Participant mode   python C11_slope_amplitude_by_position.py <subject>
    Output  outputs/C11_slope_amplitude_by_position/participants/<sub>_amplitudes.csv
            (subject, roi, session, cond, trial, position, variant, q, retrieved, control)
Group mode         python C11_slope_amplitude_by_position.py group
    Output  outputs/C11_slope_amplitude_by_position/group_matrix.csv
                retrieved minus control per retrieved position and variant, time points
                pooled, with trials per participant
            outputs/C11_slope_amplitude_by_position/group_summary.csv
                all retrieved positions together, per time point: all five items,
                up to the retrieved item (positions 3-5), without the retrieved item
    Plots   plots/C11_slope_amplitude_by_position/amplitude_matrix.{png,pdf}
            plots/C11_slope_amplitude_by_position/{amplitude,abs_slope}_by_time_point_<roi>.{png,pdf}
                retrieved and control sequence, all five items and without the retrieved
                item: all time points together next to the three time points
                (mean and 95 % CI across participants; thin lines: participants)

Input   outputs/B7_apply_classifiers_retrieval/<session>/decoding_results_tables_for_SODA/*.csv
Uses SLURM_CPUS_PER_TASK cores if set.

Absolute slope next to the amplitude (group mode). For the same trials and the
same sets of items, the absolute value of the slope is averaged over 4.5-12 s
after the cue (volumes 3-8), for the retrieved and the control sequence. A slope
in either direction counts. It needs no fit and is computed in group mode from
the B7 tables (cached in abs_slopes.csv). Every table has a `measure` column
(`amplitude` or `abs_slope`), and the figures show both:
    both_measures_by_time_point_<roi>  top row amplitude, bottom row absolute slope
    <measure>_by_time_point_<roi>      one measure only, at full size
    change_by_region                   change from the evening to the first morning
                                       retrieval of retrieved minus control, per region
    behaviour_<roi>                    retrieved minus control against the share of
                                       trials with a button press, across participants
    region_by_time_point_<items>       regions x time points (and all time points), each
                                       cell with the amplitude and the absolute slope
    carries_<roi>                      the first-morning effect and the change from the evening
                                       with single items or pairs of items left out
    behaviour_matrix_<items>           regions x four tests of the relation to reported
                                       retrieval and search time (group_behaviour.csv)
    abs_slope_matrix                   the matrix of amplitude_matrix for the absolute slope

Not a port of the original pipeline: added 2026-11-25. Exploratory.
"""

from __future__ import annotations

import glob
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
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from B9_sinusoid_summary import fit_windowed_sinusoid
from C9_item_time_courses import POST1_MAX_TRIAL, ROIS, category_sequences, stars
from config import is_excluded, step_input_dir, step_output_dir, step_plot_dir
from decoding_utils import parse_subject_arg

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
INK = "#0b0b0b"
POS = np.arange(1, 6, dtype=float)
# variant name, q, positions (1-based) that enter the slope
VARIANTS = [("all", 0, (1, 2, 3, 4, 5))] \
    + [("upto", q, tuple(range(1, q + 1))) for q in (2, 3, 4)] \
    + [("without", q, tuple(p for p in range(1, 6) if p != q)) for q in range(1, 6)]


# sets defined relative to the retrieved position p (items that do not exist are simply not dropped)
RELATIVE = {"without_neighbours": (-1, 1), "without_retrieved_and_predecessor": (0, -1),
            "without_retrieved_and_successor": (0, 1), "without_retrieved_and_neighbours": (0, -1, 1)}
ONLY_RELATIVE = os.environ.get("C11_ONLY_RELATIVE") == "1"     # participant mode: fit only the RELATIVE sets


def relative_keep(p, lags):
    return tuple(k for k in range(1, 6) if k not in {p + lag for lag in lags})


def _slopes(P, keep):
    """Slope of probability on position for every volume; P is volumes x 5, keep the positions used."""
    x = POS[[k - 1 for k in keep]]
    y = P[:, [k - 1 for k in keep]]
    xc = x - x.mean()
    return (y * xc).sum(axis=1) / (xc ** 2).sum()


def _trial(trs, R, C, p):
    out = []
    sets = [(name, 0, relative_keep(p, lags)) for name, lags in RELATIVE.items()] if ONLY_RELATIVE else VARIANTS
    for name, q, keep in sets:
        a = abs(fit_windowed_sinusoid(trs, _slopes(R, keep))["amplitude"])
        b = abs(fit_windowed_sinusoid(trs, _slopes(C, keep))["amplitude"])
        out.append((name, q, a, b))
    return out


def participant(sub):
    rows = []
    for session in ("presleep", "postsleep"):
        if is_excluded(sub, session):
            continue
        folder = str(step_input_dir("B7_apply_classifiers_retrieval", opj(session, "decoding_results_tables_for_SODA")))
        for path in sorted(glob.glob(opj(folder, f"{sub}_ROI_*.csv"))):
            roi = re.match(rf"^{sub}_ROI_(.+)_decoding_results_retrieval", os.path.basename(path)).group(1)
            if roi not in ROIS:
                continue
            d = pd.read_csv(path)
            seqs = category_sequences(d)
            d = d[d.object_scene_retrieval == "object"]
            if d.empty:
                continue
            probs = d.pivot_table(index=["trial", "tr"], columns="stimulus_class_from_classifier",
                                  values="probability_from_classifier") * 100.0
            info = d.drop_duplicates("trial").set_index("trial")
            jobs = []
            for t, r in info.iterrows():
                seq = seqs[0] if r.retrieval_target in seqs[0] else seqs[1]
                other = seqs[1] if seq is seqs[0] else seqs[0]
                p = int(r.retrieval_cue_position)
                if seq[p - 1] != r.retrieval_target:
                    raise ValueError(f"{path}: trial {t}: target is not at the cued position of its category sequence")
                tp = probs.loc[t]
                jobs.append((t, p, tp.index.values.astype(float), tp[seq].values, tp[other].values))
            fits = Parallel(n_jobs=N_JOBS)(delayed(_trial)(trs, R, C, p) for _, p, trs, R, C in jobs)
            for (t, p, _, _, _), f in zip(jobs, fits):
                cond = "pre" if session == "presleep" else ("post1" if t <= POST1_MAX_TRIAL else "post2")
                for name, q, a, b in f:
                    rows.append(dict(subject=sub, roi=roi, session=session, cond=cond, trial=t, position=p,
                                     variant=name, q=q, retrieved=a, control=b))
            print(f"{sub} {session} {roi}: {len(jobs)} trials", flush=True)
    if rows:
        name = f"{sub}_amplitudes_relative.csv" if ONLY_RELATIVE else f"{sub}_amplitudes.csv"
        pd.DataFrame(rows).to_csv(opj(str(step_output_dir(__file__, "participants")), name), index=False)
    print("done")


ABS_WINDOW = (3, 8)                   # volumes, 4.5-12 s
MEASURES = {"amplitude": "Sinusoid amplitude", "abs_slope": "Absolute slope"}
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"


def abs_slopes(out):
    """Absolute slope per trial and set of items (mean over ABS_WINDOW), and the button press per trial."""
    cache = opj(out, "abs_slopes_v2.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for session in ("presleep", "postsleep"):
        folder = str(step_input_dir("B7_apply_classifiers_retrieval", opj(session, "decoding_results_tables_for_SODA")))
        for path in sorted(glob.glob(opj(folder, "*.csv"))):
            m = re.match(r"^(s\d+)_ROI_(.+)_decoding_results_retrieval", os.path.basename(path))
            if not m or m.group(2) not in ROIS or is_excluded(m.group(1), session):
                continue
            d = pd.read_csv(path)
            seqs = category_sequences(d)
            d = d[(d.object_scene_retrieval == "object") & (d.tr >= ABS_WINDOW[0]) & (d.tr <= ABS_WINDOW[1])]
            if d.empty:
                continue
            probs = d.pivot_table(index=["trial", "tr"], columns="stimulus_class_from_classifier",
                                  values="probability_from_classifier") * 100.0
            info = d.drop_duplicates("trial").set_index("trial")
            for t, r in info.iterrows():
                seq = seqs[0] if r.retrieval_target in seqs[0] else seqs[1]
                other = seqs[1] if seq is seqs[0] else seqs[0]
                p = int(r.retrieval_cue_position)
                tp = probs.loc[t]
                R, C = tp[seq].values, tp[other].values
                cond = "pre" if session == "presleep" else ("post1" if t <= POST1_MAX_TRIAL else "post2")
                rt = float(r.ret_rt) if "ret_rt" in info.columns and pd.notna(r.ret_rt) else np.nan
                for name, q, keep in VARIANTS + [(n_, 0, relative_keep(p, lags)) for n_, lags in RELATIVE.items()]:
                    rows.append(dict(subject=m.group(1), roi=m.group(2), session=session, cond=cond, trial=t, position=p,
                                     variant=name, q=q, retrieved=np.abs(_slopes(R, keep)).mean(), control=np.abs(_slopes(C, keep)).mean(),
                                     press=bool(r.behav_correct), rt=rt))
    a = pd.DataFrame(rows)
    a.to_csv(cache, index=False)
    return a


def fig_time_points(dd, plot_dir, roi):
    fig, all_axes = plt.subplots(2, 4, figsize=(13.5, 7.6), gridspec_kw=dict(width_ratios=[1, 2.1, 1, 2.1]))
    for row, (measure, mlabel) in enumerate(MEASURES.items()):
        d = dd[dd.measure == measure]
        _time_point_row(d, all_axes[row], roi, mlabel, legend=(row == 0))
    for ax in all_axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    n = dd[dd.roi == roi].subject.nunique()
    fig.suptitle(f"{ROIS[roi]}: retrieved and control sequence, object trials (N = {n}; mean and 95 % CI; "
                 "t: retrieved minus control, * p < .05 uncorrected)", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"both_measures_by_time_point_{roi}.{ext}"), **kw)
    plt.close(fig)


def fig_time_points_single(dd, plot_dir, roi, measure):
    """One measure only, at full size: all time points next to the three time points."""
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), gridspec_kw=dict(width_ratios=[1, 2.1, 1, 2.1]))
    _time_point_row(dd[dd.measure == measure], axes, roi, MEASURES[measure], legend=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    n = dd[dd.roi == roi].subject.nunique()
    fig.suptitle(f"{ROIS[roi]}: {MEASURES[measure].lower()} of the retrieved and the control sequence, object trials "
                 f"(N = {n}; mean and 95 % CI; t: retrieved minus control, * p < .05 uncorrected)", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{measure}_by_time_point_{roi}.{ext}"), **kw)
    plt.close(fig)


def _time_point_row(d, axes, roi, mlabel, legend):
    sets = {"All five items": d[(d.roi == roi) & (d.variant == "all")],
            "Without the retrieved item": d[(d.roi == roi) & (d.variant == "without") & (d.q == d.position)]}
    for k, (name, e) in enumerate(sets.items()):
        pooled = e.groupby("subject")[["retrieved", "control"]].mean()
        bycond = e.groupby(["subject", "cond"])[["retrieved", "control"]].mean().unstack("cond")
        lo = min(pooled.values.min(), np.nanmin(bycond.values)); hi = max(pooled.values.max(), np.nanmax(bycond.values))
        ci = lambda v: stats.t.ppf(0.975, v.notna().sum() - 1) * v.std(ddof=1) / np.sqrt(v.notna().sum())
        ax = axes[2 * k]
        for _, r in pooled.iterrows():
            ax.plot([0, 1], [r.control, r.retrieved], color=MUTED, alpha=0.2, lw=0.6, zorder=1)
        for x, col, colour in ((0, "control", ORANGE), (1, "retrieved", BLUE)):
            ax.errorbar(x, pooled[col].mean(), yerr=ci(pooled[col]), fmt="o", color=INK, markerfacecolor=colour, markersize=8, capsize=3, zorder=4)
        t = _test(pooled.retrieved - pooled.control)
        ax.set_title(f"{name}\nall time points", loc="left", fontsize=9.5, fontweight="bold")
        ax.text(0.5, 0.97, f"t({t['n'] - 1}) = {t['t']:.1f}\np = {t['p']:.3f}".replace("0.", ".", 1) if t["p"] >= 0.001 else f"t({t['n'] - 1}) = {t['t']:.1f}\np < .001",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Retrieved"]); ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(f"{mlabel} (points per position)")
        ax = axes[2 * k + 1]
        conds = ["pre", "post1", "post2"]
        for col, colour, lab in (("control", ORANGE, "Control sequence"), ("retrieved", BLUE, "Retrieved sequence")):
            m = [bycond[(col, c)].mean() for c in conds]; e_ = [ci(bycond[(col, c)]) for c in conds]
            ax.errorbar(range(3), m, yerr=e_, fmt="-o", color=colour, markersize=7, capsize=3, lw=2, label=lab)
        mm = [bycond[(c_, c)].mean() for c_ in ("retrieved", "control") for c in conds]
        ee = [ci(bycond[(c_, c)]) for c_ in ("retrieved", "control") for c in conds]
        ylo, yhi = min(np.array(mm) - np.array(ee)), max(np.array(mm) + np.array(ee))
        for i, c in enumerate(conds):
            t = _test(bycond[("retrieved", c)] - bycond[("control", c)])
            ax.text(i, yhi + 0.06 * (yhi - ylo), f"t = {t['t']:.1f}{stars(t['p'])}", ha="center", fontsize=8.5)
        ax.set_ylim(ylo - 0.15 * (yhi - ylo), yhi + 0.22 * (yhi - ylo))
        ax.set_xticks(range(3)); ax.set_xticklabels(["Evening", "First\nmorning", "Repeated\nmorning"]); ax.set_xlim(-0.4, 2.4)
        ax.set_title(f"{name}\nby time point", loc="left", fontsize=9.5, fontweight="bold")
        if k == 0 and legend:
            ax.legend(frameon=False, fontsize=8, loc="lower left")


ITEM_SETS = {
    "all five items": lambda d: d[d.variant == "all"],
    "up to the retrieved item (positions 3-5)": lambda d: pd.concat(
        [d[(d.variant == "upto") & (d.q == d.position)], d[(d.variant == "all") & (d.position == 5)]]).query("position >= 3"),
    "without the retrieved item": lambda d: d[(d.variant == "without") & (d.q == d.position)],
    # one neighbour left out; where the retrieved item has no such neighbour, all five items stay
    "without the predecessor": lambda d: pd.concat([d[(d.variant == "without") & (d.q == d.position - 1)],
                                                    d[(d.variant == "all") & (d.position == 1)]]),
    "without the successor": lambda d: pd.concat([d[(d.variant == "without") & (d.q == d.position + 1)],
                                                  d[(d.variant == "all") & (d.position == 5)]]),
    "without both neighbours": lambda d: d[d.variant == "without_neighbours"],
    "without the retrieved item and its predecessor": lambda d: d[d.variant == "without_retrieved_and_predecessor"],
    "without the retrieved item and its successor": lambda d: d[d.variant == "without_retrieved_and_successor"],
    "without the retrieved item and both neighbours": lambda d: d[d.variant == "without_retrieved_and_neighbours"],
}
CARRIES_ORDER = ["all five items", "without the predecessor", "without the successor", "without both neighbours",
                 "without the retrieved item", "without the retrieved item and its predecessor",
                 "without the retrieved item and its successor", "without the retrieved item and both neighbours"]
TIME_POINTS = {"pooled": "All time\npoints", "pre": "Evening", "post1": "First\nmorning", "post2": "Repeated\nmorning"}


def summary_table(dd):
    """Retrieved minus control per measure, set of items, region and time point, and the changes between time points."""
    rows = []
    for measure in MEASURES:
        for items, pick in ITEM_SETS.items():
            e = pick(dd[dd.measure == measure])
            for cond in TIME_POINTS:
                x = e if cond == "pooled" else e[e.cond == cond]
                w = x.groupby(["roi", "subject"])
                for roi, g in w.difference.mean().groupby(level="roi"):
                    rows.append(dict(measure=measure, items=items, cond=cond, roi=roi,
                                     trials_per_participant=w.difference.count().loc[roi].mean(),
                                     retrieved=w.retrieved.mean().loc[roi].mean(), control=w.control.mean().loc[roi].mean(), **_test(g)))
            wide = e.groupby(["roi", "subject", "cond"]).difference.mean().unstack("cond")
            for a, b in (("post1", "pre"), ("post2", "post1"), ("post2", "pre")):
                for roi, g in (wide[a] - wide[b]).groupby(level="roi"):
                    rows.append(dict(measure=measure, items=items, cond=f"{a} - {b}", roi=roi, **_test(g)))
    return pd.DataFrame(rows)


def fig_region_time_matrix(summ, items, plot_dir):
    """Regions x time points; every cell holds retrieved minus control for the amplitude and for the absolute slope."""
    x = summ[summ["items"] == items]
    rois = [r for r in ROIS if r in set(x.roi)]
    conds = list(TIME_POINTS)
    T = np.full((len(rois), 2 * len(conds)), np.nan)
    fig, ax = plt.subplots(figsize=(11.5, 0.72 * len(rois) + 1.9))
    for i, roi in enumerate(rois):
        for j, cond in enumerate(conds):
            for k, measure in enumerate(MEASURES):
                r = x[(x.roi == roi) & (x.cond == cond) & (x.measure == measure)].iloc[0]
                T[i, 2 * j + k] = r.t
    ax.imshow(T, cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="auto")
    for i, roi in enumerate(rois):
        for j, cond in enumerate(conds):
            for k, measure in enumerate(MEASURES):
                r = x[(x.roi == roi) & (x.cond == cond) & (x.measure == measure)].iloc[0]
                ax.text(2 * j + k, i, f"{r['mean']:+.2f}{stars(r.p)}\nt = {r.t:.1f}", ha="center", va="center", fontsize=8.5, color=INK)
    for j in range(1, len(conds)):
        ax.axvline(2 * j - 0.5, color=INK, lw=1.6)
    ax.set_xticks(range(2 * len(conds)))
    ax.set_xticklabels(["Amplitude", "Absolute\nslope"] * len(conds), fontsize=8.5)
    ax.xaxis.tick_top()
    for j, cond in enumerate(conds):
        ax.text(2 * j + 0.5, -1.15, TIME_POINTS[cond].replace("\n", " "), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(rois))); ax.set_yticklabels([ROIS[r] for r in rois], fontsize=9.5)
    n = x[(x.cond == "pooled")].n.max(); nm = x[(x.cond == "post1")].n.max()
    ax.set_xlabel(f"Retrieved minus control sequence, {items}. Colour: t value. * p < .05, ** p < .01, *** p < .001, uncorrected. "
                  f"N = {int(n)} ({int(nm)} in the morning).", fontsize=8.5)
    fig.tight_layout()
    name = "region_by_time_point_" + items.split(" (")[0].replace(" ", "_")
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


def fig_carries(summ, plot_dir, roi):
    """Which items carry the effect: the first-morning effect and the change from the evening, with items left out."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (measure, mlabel) in zip(axes, MEASURES.items()):
        x = summ[(summ.roi == roi) & (summ.measure == measure)]
        for i, items in enumerate(CARRIES_ORDER):
            for cond, colour, dy, lab in (("post1", BLUE, 0.16, "First morning retrieval"), ("post1 - pre", INK, -0.16, "Change from the evening")):
                r = x[(x["items"] == items) & (x.cond == cond)]
                if r.empty or not np.isfinite(r.t.iloc[0]):
                    continue
                r = r.iloc[0]
                ci = abs(r["mean"] / r.t) * stats.t.ppf(0.975, r.n - 1) if r.t != 0 else 0
                ax.errorbar(r["mean"], i + dy, xerr=ci, fmt="o", color=colour, markersize=6, capsize=2.5, lw=1.3, label=lab if i == 0 else None)
                ax.text(r["mean"] + ci, i + dy, f"  t = {r.t:.1f}{stars(r.p)}", va="center", fontsize=8, color=colour)
        ax.axvline(0, color=INK, ls=":", lw=0.8)
        ax.axhline(3.5, color=MUTED, lw=0.6)
        ax.set_yticks(range(len(CARRIES_ORDER))); ax.set_yticklabels([c.capitalize() for c in CARRIES_ORDER], fontsize=9)
        ax.set_title(mlabel, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Retrieved minus control (points per position)")
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + 0.25 * (hi - lo))
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(f"{ROIS[roi]}: the effect with items left out of both sequences (mean and 95 % CI; * p < .05 uncorrected)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"carries_{roi}.{ext}"), **kw)
    plt.close(fig)


BEHAVIOUR_TESTS = {"between: press rate": "More reported\nretrieval (r)", "between: search time": "Longer search\ntime (r)",
                   "within: press vs none": "Trials with\nvs without press (t)", "within: per second": "Per second of\nsearch time (t)"}


def behaviour_tables(dd, ab, out):
    """Relation of retrieved minus control to reported retrieval (button press) and to search time."""
    trials = ab[ab.variant == "all"][["subject", "roi", "session", "trial", "press", "rt"]]
    one = trials[trials.roi == trials.roi.iloc[0]]
    press_rate = one.groupby("subject").press.mean() * 100.0
    med_rt = one.groupby("subject").rt.median()
    ch_press = (one[one.session == "postsleep"].query("trial <= @POST1_MAX_TRIAL").groupby("subject").press.mean()
                - one[one.session == "presleep"].groupby("subject").press.mean()) * 100.0
    ch_rt = (one[one.session == "postsleep"].query("trial <= @POST1_MAX_TRIAL").groupby("subject").rt.median()
             - one[one.session == "presleep"].groupby("subject").rt.median())
    rows = []
    for measure in MEASURES:
        for items in ("all five items", "without the retrieved item", "up to the retrieved item (positions 3-5)"):
            e = ITEM_SETS[items](dd[dd.measure == measure]).drop(columns=["press", "rt"], errors="ignore")
            e = e.merge(trials, on=["subject", "roi", "session", "trial"], how="left")
            for roi, g in e.groupby("roi"):
                tag = dict(measure=measure, items=items, roi=roi)
                v = g.groupby("subject").difference.mean()
                for name, b in (("between: press rate", press_rate), ("between: search time", med_rt)):
                    x = b.reindex(v.index)
                    ok = x.notna() & v.notna()
                    r = stats.pearsonr(x[ok], v[ok])
                    rows.append(dict(**tag, test=name, n=int(ok.sum()), value=r.statistic,
                                     t=r.statistic * np.sqrt((ok.sum() - 2) / (1 - r.statistic ** 2)), p=r.pvalue))
                w = g.groupby(["subject", "press"]).difference.agg(["mean", "count"]).unstack("press")
                okp = (w[("count", True)] >= 5) & (w[("count", False)] >= 5)
                dpress = (w[("mean", True)] - w[("mean", False)])[okp].dropna()
                if len(dpress) >= 10:
                    r = stats.ttest_1samp(dpress, 0)
                    rows.append(dict(**tag, test="within: press vs none", n=len(dpress), value=dpress.mean(), t=r.statistic, p=r.pvalue))
                sl = []
                for _, h in g[g.rt.notna() & g.difference.notna()].groupby("subject"):
                    if len(h) >= 10 and h.rt.std() > 0:
                        sl.append(np.polyfit(h.rt.values, h.difference.values, 1)[0])
                if len(sl) >= 10:
                    r = stats.ttest_1samp(sl, 0)
                    rows.append(dict(**tag, test="within: per second", n=len(sl), value=float(np.mean(sl)), t=r.statistic, p=r.pvalue))
                wide = g.groupby(["subject", "cond"]).difference.mean().unstack("cond")
                ch = (wide["post1"] - wide["pre"]).dropna()
                for name, b in (("change vs change in press rate", ch_press), ("change vs change in search time", ch_rt)):
                    x = b.reindex(ch.index)
                    ok = x.notna()
                    r = stats.pearsonr(x[ok], ch[ok])
                    rows.append(dict(**tag, test=name, n=int(ok.sum()), value=r.statistic,
                                     t=r.statistic * np.sqrt((ok.sum() - 2) / (1 - r.statistic ** 2)), p=r.pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(opj(out, "group_behaviour.csv"), index=False)
    return t


def fig_behaviour_matrix(beh, items, plot_dir):
    rois = [r for r in ROIS if r in set(beh.roi)]
    tests = list(BEHAVIOUR_TESTS)
    x = beh[beh["items"] == items]
    T = np.full((len(rois), 2 * len(tests)), np.nan)
    fig, ax = plt.subplots(figsize=(12, 0.72 * len(rois) + 2.0))
    for i, roi in enumerate(rois):
        for j, test in enumerate(tests):
            for k, measure in enumerate(MEASURES):
                r = x[(x.roi == roi) & (x.test == test) & (x.measure == measure)]
                if not r.empty:
                    T[i, 2 * j + k] = r.t.iloc[0]
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e4e3df")
    ax.imshow(np.ma.masked_invalid(T), cmap=cmap, vmin=-4.5, vmax=4.5, aspect="auto")
    for i, roi in enumerate(rois):
        for j, test in enumerate(tests):
            for k, measure in enumerate(MEASURES):
                r = x[(x.roi == roi) & (x.test == test) & (x.measure == measure)]
                if r.empty:
                    continue
                r = r.iloc[0]
                txt = f"r = {r.value:+.2f}" if test.startswith("between") else f"t = {r.t:+.1f}"
                ax.text(2 * j + k, i, f"{txt}{stars(r.p)}", ha="center", va="center", fontsize=8.5, color=INK)
    for j in range(1, len(tests)):
        ax.axvline(2 * j - 0.5, color=INK, lw=1.6)
    ax.axvline(3.5, color=INK, lw=3)
    ax.set_xticks(range(2 * len(tests))); ax.set_xticklabels(["Amplitude", "Absolute\nslope"] * len(tests), fontsize=8.5)
    ax.xaxis.tick_top()
    for j, test in enumerate(tests):
        ax.text(2 * j + 0.5, -1.15, BEHAVIOUR_TESTS[test].replace("\n", " "), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(rois))); ax.set_yticklabels([ROIS[r] for r in rois], fontsize=9.5)
    ax.set_xlabel(f"Retrieved minus control, {items}. Left half: across participants. Right half: across trials within participants. "
                  "Colour: t value. * p < .05, ** p < .01, uncorrected.", fontsize=8.5)
    fig.tight_layout()
    name = "behaviour_matrix_" + items.split(" (")[0].replace(" ", "_")
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


def _sets(d):
    return {"All five items": d[d.variant == "all"],
            "Without the retrieved item": d[(d.variant == "without") & (d.q == d.position)]}


def fig_change_by_region(dd, plot_dir):
    """Change of retrieved minus control from the evening to the first morning retrieval, per region."""
    rois = [r for r in ROIS if r in set(dd.roi)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    rows = []
    for i, (measure, mlabel) in enumerate(MEASURES.items()):
        for j, (name, e) in enumerate(_sets(dd[dd.measure == measure]).items()):
            ax = axes[i, j]
            w = e.groupby(["roi", "subject", "cond"]).difference.mean().unstack("cond")
            ch = (w["post1"] - w["pre"]).dropna()
            for x, roi in enumerate(rois):
                v = ch.loc[roi]
                t = _test(v)
                ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                colour = BLUE if roi in ("vis", "parietal_precuneus", "mtl", "hpc") else MUTED
                ax.scatter(x + np.random.default_rng(x).uniform(-0.12, 0.12, len(v)), v, s=5, color=colour, alpha=0.2, linewidths=0)
                ax.errorbar(x, v.mean(), yerr=ci, fmt="o", color=INK, markerfacecolor=colour, markersize=7, capsize=3, zorder=4)
                ax.text(x, v.mean() + ci, f" t = {t['t']:.1f}{stars(t['p'])}", ha="center", va="bottom", fontsize=8)
                rows.append(dict(measure=measure, items=name, roi=roi, **t))
            lim = max(abs(ch.groupby(level="roi").mean()).max() * 4.5, 0.3)
            ax.set_ylim(-lim, lim)                               # single participants may leave the panel
            ax.axhline(0, color=INK, ls=":", lw=0.8)
            ax.set_title(f"{mlabel}: {name.lower()}", loc="left", fontsize=10, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(range(len(rois)))
            ax.set_xticklabels([ROIS[r].replace(" (control)", "\n(control)").replace(" + ", " +\n") for r in rois], fontsize=8)
        axes[i, 0].set_ylabel("Change, first morning minus evening")
    fig.suptitle("Change of retrieved minus control from the evening to the first morning retrieval (mean and 95 % CI; * p < .05 uncorrected)", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"change_by_region.{ext}"), **kw)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_behaviour(dd, press, plot_dir, roi):
    """Across participants: retrieved minus control (time points pooled) against the share of trials with a button press."""
    panels = [("amplitude", "All five items"), ("abs_slope", "All five items"), ("amplitude", "Up to the retrieved item"), ("abs_slope", "Up to the retrieved item")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    rows = []
    for ax, (measure, name) in zip(axes, panels):
        d = dd[(dd.measure == measure) & (dd.roi == roi)]
        e = d[d.variant == "all"] if name.startswith("All") else pd.concat(
            [d[(d.variant == "upto") & (d.q == d.position)], d[(d.variant == "all") & (d.position == 5)]]).query("position >= 3")
        v = e.groupby("subject").difference.mean()
        x = press.reindex(v.index) * 100.0
        r = stats.pearsonr(x, v)
        ax.scatter(x, v, s=22, color=BLUE, alpha=0.7, linewidths=0)
        b, a = np.polyfit(x, v, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, color=INK, lw=1.2)
        ax.set_title(f"{MEASURES[measure]}\n{name.lower()}", loc="left", fontsize=9.5, fontweight="bold")
        ax.text(0.03, 0.97, f"r = {r.statistic:.2f}, p = {r.pvalue:.3f}".replace("0.", ".").replace("-.", "-0."), transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("Trials with a button press (%)")
        ax.spines[["top", "right"]].set_visible(False)
        rows.append(dict(roi=roi, measure=measure, items=name, n=len(v), r=r.statistic, p=r.pvalue))
    axes[0].set_ylabel("Retrieved minus control")
    fig.suptitle(f"{ROIS[roi]}: sequentiality and reported retrieval across participants (object trials, time points pooled)", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"behaviour_{roi}.{ext}"), **kw)
    plt.close(fig)
    return rows


def _test(g):
    g = g.dropna()
    r = stats.ttest_1samp(g, 0)
    return dict(n=len(g), mean=g.mean(), t=r.statistic, p=r.pvalue)


def group():
    out, plot_dir = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_amplitudes*.csv")))], ignore_index=True)
    d["difference"] = d.retrieved - d.control
    ab = abs_slopes(out)
    ab["difference"] = ab.retrieved - ab.control
    press = ab[(ab.variant == "all") & (ab.roi == ab.roi.iloc[0])].groupby("subject").press.mean()
    dd = pd.concat([d.assign(measure="amplitude"), ab.drop(columns=["press", "rt"]).assign(measure="abs_slope")], ignore_index=True)

    rows = []
    v = d.groupby(["roi", "variant", "q", "position", "subject"])
    m, n_tr, ra, ca = v.difference.mean(), v.difference.count(), v.retrieved.mean(), v.control.mean()
    for key, g in m.groupby(level=["roi", "variant", "q", "position"]):
        rows.append(dict(roi=key[0], variant=key[1], q=key[2], retrieved_position=key[3],
                         trials_per_participant=n_tr.loc[key].mean(), retrieved=ra.loc[key].mean(), control=ca.loc[key].mean(), **_test(g)))
    mat = pd.DataFrame(rows)
    mat.to_csv(opj(out, "group_matrix.csv"), index=False)

    summ = summary_table(dd)
    summ.to_csv(opj(out, "group_summary.csv"), index=False)
    for items in ("all five items", "without the retrieved item", "up to the retrieved item (positions 3-5)"):
        fig_region_time_matrix(summ, items, plot_dir)

    for roi in [r for r in ROIS if r in set(d.roi)]:
        fig_time_points(dd, plot_dir, roi)
        for measure in MEASURES:
            fig_time_points_single(dd, plot_dir, roi, measure)
    fig_change_by_region(dd, plot_dir).to_csv(opj(out, "group_change_by_region.csv"), index=False)
    for roi in [r for r in ROIS if r in set(d.roi)]:
        fig_behaviour(dd, press, plot_dir, roi)
        fig_carries(summ, plot_dir, roi)
    beh = behaviour_tables(dd, ab, out)
    for items in ("all five items", "without the retrieved item", "up to the retrieved item (positions 3-5)"):
        fig_behaviour_matrix(beh, items, plot_dir)

    rois = [r for r in ("vis", "parietal_precuneus", "mtl") if r in set(mat.roi)]
    mat = mat[mat.variant.isin(["all", "upto", "without"])]
    cols = [("all", 0, "all\nfive")] + [("upto", q, f"1 to {q}") for q in (2, 3, 4)] + [("without", q, f"without\n{q}") for q in range(1, 6)]
    fig, axes = plt.subplots(1, len(rois), figsize=(6.2 * len(rois), 4.6), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        M = np.full((5, len(cols)), np.nan); P = np.full((5, len(cols)), np.nan)
        for j, (name, q, _) in enumerate(cols):
            x = mat[(mat.roi == roi) & (mat.variant == name) & (mat.q == q)].set_index("retrieved_position")
            for p in range(1, 6):
                if p in x.index:
                    M[p - 1, j], P[p - 1, j] = x.loc[p, "mean"], x.loc[p, "p"]
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(5):
            for j, (name, q, _) in enumerate(cols):
                ax.text(j, i, f"{M[i, j]:+.2f}{stars(P[i, j])}", ha="center", va="center", fontsize=7.5, color=INK)
                if (name == "upto" and q == i + 1) or (name == "without" and q == i + 1) or (name == "all" and i == 4):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.4))
        ax.axvline(0.5, color=INK, lw=0.8); ax.axvline(3.5, color=INK, lw=0.8)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2] for c in cols], fontsize=8)
        ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
        ax.set_xlabel("Items that enter the slope"); ax.set_title(ROIS[roi], loc="left", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    axes[0][0].set_ylabel("Position of the retrieved item")
    fig.suptitle("Sinusoid amplitude of the slope, retrieved minus control sequence (points per position); framed: slope up to / "
                 "without the retrieved item; * p < .05 uncorrected", fontsize=10)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"amplitude_matrix.{ext}"), **kw)
    plt.close(fig)
    print(f"{d.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
