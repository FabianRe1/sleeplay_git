#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E13 - Time course of the signal over a retrieval trial, without an assumed
      response shape, per region and retrieval time point.

E7 / E12 assume one response shape (canonical response, scaled by the length of
the event) and express it against the fixation cross. This step estimates the
course of the signal over the trial freely: for each run, one regressor per
1.5 s step (one volume) from LAG_FIRST to LAG_LAST volumes around the onset of
the retrieval cue, separately for
    object       object trials with a button press
    scene        scene trials with a button press
    noretrieval  trials without a press
plus cosine drifts slower than 128 s and the 20 fMRIPrep confounds. No other
task regressors: waiting, speaking and the trial-type cue are part of the
course itself. Times outside the window (the fixation cross and the end of the
trial after the window) are the reference.

One trial: fixation 3-7 s -> trial-type cue, about 2 s -> retrieval cue (time 0)
-> question mark at 1.015 s -> button press -> waiting -> screen for the spoken
answer 11-13 s after the question mark, about 2.6 s -> next fixation.

Data and regions as E12 (T1w space, NOT smoothed, percent signal change, mean
over the voxels of each region, with the reliable-voxel versions), plus the B1
regions VIS, MTL, HPC, PHPC and ERH.

Matched for search time (E13_MATCHED=1; results in the subfolder matched/). The
object trials with a press are divided into `object_matched` and `object_rest`,
each with its own set of regressors. Matching is done within participant over
the three time points at once: the search times (question mark to press) are
sorted into bins of MATCH_BIN seconds, and from every bin the same number of
trials is taken at each time point (the smallest of the three counts; which
trials is drawn at random with a fixed seed). The matched trials therefore have
the same distribution of search times at all three time points. Participants
without all three time points are left out. <sub>_matching.csv lists the number
of matched trials and their mean search time per time point.

Participant output  outputs/E13_response_time_course/participants/<sub>_time_course.csv
                    (subject, cond, kind, lag, time_s, region, beta, n_runs);
                    cond = pre / post1 / post2 and, per run, preR1 ... post2R3
Group output        group_time_course.csv   mean and SEM per time point, kind, region, lag,
                                            and the paired tests between time points per lag
                    group_windows.csv       the same for windows of the course (see WINDOWS)
                                            and their differences
Plots               plots/E13_response_time_course/time_course_<kind>.{png,pdf}

Usage:  python E13_response_time_course.py <subject>
        python E13_response_time_course.py group

Not a port of the original pipeline: added 2026-11-25. Exploratory.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import E12_precuneus_parietal_subregions as e12
from config import TR, step_input_dir, step_output_dir, step_plot_dir
from decoding_utils import CONFOUND_VARS, build_sequence_dict_from_df_behavior_enc, parse_subject_arg

e3 = e12.e7.e3
d1, dec = e3.d1, e3.dec
LAG_FIRST, LAG_LAST = -2, 13                      # volumes relative to the retrieval cue: -3 s ... +19.5 s
LAGS = np.arange(LAG_FIRST, LAG_LAST + 1)
MATCHED = os.environ.get("E13_MATCHED") == "1"
MATCH_BIN = 0.25                                  # s, width of the search-time bins
KINDS = ("object_matched", "object_rest", "scene", "noretrieval") if MATCHED else ("object", "scene", "noretrieval")
SUB = "matched/" if MATCHED else ""
B1_EXTRA = ("VIS", "MTL", "HPC", "PHPC", "ERH")
CONDS = ("pre", "post1", "post2")
CONTRASTS = {"post1_minus_pre": ("post1", "pre"), "post2_minus_pre": ("post2", "pre"),
             "post2_minus_post1": ("post2", "post1")}
# windows of the course, in volumes after the cue (the signal lags the events by about 5 s)
WINDOWS = {"before": (-2, 0),      # -3 ... 0 s: trial-type cue, signal still at the level of the fixation
           "retrieval": (3, 5),    # 4.5 ... 7.5 s: cue and search
           "waiting": (8, 10),     # 12 ... 15 s: holding the answer
           "speaking": (12, 13)}   # 18 ... 19.5 s: spoken answer
PLOT_REGIONS = {"VIS": "Visual", "PARIETAL_PRECUNEUS": "Parietal + precuneus", "MTL": "MTL",
                "CAUDAL_ACC": "Caudal ACC (control)", "HPC": "Hippocampus",
                "PHPC": "Parahippocampal", "PRECUNEUS": "Precuneus", "PARIETAL": "Inferior parietal",
                "RSC": "Retrosplenial", "D_PARIETOOCC_S__r70": "Parieto-occipital sulcus (reliable voxels)"}


def match_three(trials):
    """Boolean per row of `trials` (columns cond, rt): in the set matched for search time over the three time points."""
    rng = np.random.default_rng(0)
    keep = np.zeros(len(trials), bool)
    bins = np.floor(trials.rt.values / MATCH_BIN).astype(int)
    for b in np.unique(bins):
        idx = {c: np.flatnonzero((bins == b) & (trials.cond.values == c)) for c in CONDS}
        k = min(len(v) for v in idx.values())
        for v in idx.values():
            keep[rng.choice(v, k, replace=False)] = True
    return keep


def fir(onsets_s, n):
    """One column per lag: 1 at the volume `lag` volumes after each onset."""
    X = np.zeros((n, len(LAGS)), dtype=np.float32)
    t0 = np.round(np.asarray(onsets_s, float) / TR).astype(int)
    for j, lag in enumerate(LAGS):
        idx = t0 + lag
        idx = idx[(idx >= 0) & (idx < n)]
        X[idx, j] = 1.0
    return X


def participant(sub):
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    path_loc = str(step_input_dir("A2_add_tr_columns_localizer"))
    path_main = str(step_input_dir("A1_add_tr_columns_main"))
    _, o2c = dec.load_localizer_behavior(path_loc, sub)
    df_enc, _ = dec.load_main_task_behavior(path_main, sub, dec.SESSION_CONFIG["presleep"])
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)
    phases = {}
    for session in d1.SESSIONS:
        try:
            scfg = dec.SESSION_CONFIG[session]
            ev, nuis = d1.retrieval_events(sub, session, hierarchy, o2c, path_main)
            f, c = d1.select_run_files(sub_mri, scfg["mri_session"], scfg)
            phases[session] = dict(funcs=f, confs=c, ev=ev)
        except Exception as err:
            print(f"{session} not usable: {err}")
    if not phases:
        raise RuntimeError("no usable session")
    all_funcs = [f for p in phases.values() for f in p["funcs"]]
    mask = masking.intersect_masks([f.replace("desc-preproc_bold", "desc-brain_mask") for f in all_funcs],
                                   threshold=1, connected=False)
    bm = np.asarray(mask.dataobj).astype(bool)
    regions = {k: v for k, v in e12.b1_regions(sub_mri, mask, bm).items() if k in B1_EXTRA}
    regions.update(e12.fine_regions(sub_mri, mask, bm))
    names = list(regions)
    loaded = {}
    for session, p in phases.items():
        ts, confs = [], []
        for f, c in zip(p["funcs"], p["confs"]):
            print(f"  loading {os.path.basename(f)}", flush=True)
            y = masking.apply_mask(f, mask).astype(np.float32)
            m = y.mean(axis=0); m[m <= 0] = np.inf
            y = (y / m - 1.0) * 100.0
            ts.append(np.column_stack([y[:, regions[r]].mean(axis=1) for r in names]))
            confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
            del y
        try:
            ev = d1.finalise_events(p["ev"], [y.shape[0] for y in ts], session, sub).copy()
        except ValueError as err:
            print(f"{session} not usable: {err}")
            continue
        ev["cond"] = "pre" if session == "presleep" else np.where(ev.run_idx < 3, "post1", "post2")
        ev["matched"] = False
        loaded[session] = dict(ts=ts, confs=confs, ev=ev)
    if MATCHED:
        sel = pd.concat([x["ev"][(x["ev"].kind == "object") & x["ev"].rt.notna()].assign(session=s_)
                         for s_, x in loaded.items()])
        if set(sel.cond) != set(CONDS):
            print("not all three time points: participant left out of the matched analysis")
            return
        keep = match_three(sel)
        for s_, x in loaded.items():
            x["ev"].loc[sel.index[keep & (sel.session.values == s_)], "matched"] = True
        m = sel[keep].groupby("cond").rt.agg(["size", "mean"]).rename(columns={"size": "n_matched", "mean": "rt_matched"})
        a = sel.groupby("cond").rt.agg(["size", "mean"]).rename(columns={"size": "n_all", "mean": "rt_all"})
        a.join(m).reset_index().assign(subject=sub).to_csv(
            opj(str(step_output_dir(__file__, SUB + "participants")), f"{sub}_matching.csv"), index=False)
        print(a.join(m).round(2).to_string(), flush=True)
    sums, counts = {}, {}
    for session, x in loaded.items():
        for ri, (y, conf) in enumerate(zip(x["ts"], x["confs"])):
            e = x["ev"][x["ev"].run_idx == ri]
            if e.empty:
                continue
            n = y.shape[0]
            cond = e.cond.iloc[0]
            obj = e[(e.kind == "object") & e.rt.notna()]
            sets = {"scene": e[(e.kind == "scene") & e.rt.notna()], "noretrieval": e[e.rt.isna()]}
            sets.update({"object_matched": obj[obj.matched], "object_rest": obj[~obj.matched]} if MATCHED else {"object": obj})
            used = [k for k in KINDS if len(sets[k]) >= 3]
            N = d1.nuisance_matrix(n, conf, pd.DataFrame(columns=["name", "onset_s", "duration"]))
            D = np.column_stack([fir(sets[k].onset_s.values, n) for k in used] + [N])
            b = np.linalg.lstsq(D, y, rcond=None)[0]
            for i, k in enumerate(used):
                est = b[i * len(LAGS):(i + 1) * len(LAGS)]              # lags x regions
                for key in ((cond, k), (f"{cond}R{ri % 3 + 1}", k)):
                    sums[key] = sums.get(key, 0) + est
                    counts[key] = counts.get(key, 0) + 1
            print(f"  {session} run {ri} ({cond}): " + ", ".join(f"{len(sets[k])} {k}" for k in KINDS), flush=True)
    rows = [dict(subject=sub, cond=c, kind=k, lag=int(lag), time_s=float(lag * TR), region=reg,
                 beta=float(v[j, i] / counts[(c, k)]), n_runs=counts[(c, k)])
            for (c, k), v in sums.items() for j, lag in enumerate(LAGS) for i, reg in enumerate(names)]
    pd.DataFrame(rows).to_csv(opj(str(step_output_dir(__file__, SUB + "participants")), f"{sub}_time_course.csv"), index=False)
    print("done")


def _tt(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    r = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue)


def group():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = str(step_output_dir(__file__, SUB.rstrip("/") or None))
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_time_course.csv")))])
    d = d[d.cond.isin(CONDS)]
    w = d.pivot_table(index=["kind", "region", "lag", "subject"], columns="cond", values="beta").reset_index()
    w = w.dropna(subset=list(CONDS))                                   # participants with all three time points
    rows = []
    for (k, reg, lag), g in w.groupby(["kind", "region", "lag"]):
        row = dict(kind=k, region=reg, lag=lag, time_s=lag * TR, n=len(g))
        for c in CONDS:
            row[f"{c}_mean"], row[f"{c}_sem"] = g[c].mean(), g[c].sem()
        for name, (a, b) in CONTRASTS.items():
            r = _tt(g[a] - g[b]); row[f"{name}_t"], row[f"{name}_p"] = r["t"], r["p"]
        rows.append(row)
    tc = pd.DataFrame(rows)
    tc.to_csv(opj(out, "group_time_course.csv"), index=False)

    win = {name: w[(w.lag >= a) & (w.lag <= b)].groupby(["kind", "region", "subject"])[list(CONDS)].mean()
           for name, (a, b) in WINDOWS.items()}
    win["retrieval_minus_before"] = win["retrieval"] - win["before"]
    win["retrieval_minus_waiting"] = win["retrieval"] - win["waiting"]
    rows = []
    for name, x in win.items():
        for (k, reg), g in x.groupby(["kind", "region"]):
            row = dict(window=name, kind=k, region=reg, n=len(g), **{c: g[c].mean() for c in CONDS})
            for cn, (a, b) in CONTRASTS.items():
                r = _tt(g[a] - g[b]); row[f"{cn}_t"], row[f"{cn}_p"] = r["t"], r["p"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(opj(out, "group_windows.csv"), index=False)

    colours = {"pre": "#52514e", "post1": "#2a78d6", "post2": "#eb6834"}
    labels = {"pre": "Evening", "post1": "First morning", "post2": "Repeated morning"}
    for kind in [k for k in ("object", "object_matched", "scene") if k in set(tc.kind)]:
        fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
        for ax, (reg, title) in zip(axes.ravel(), PLOT_REGIONS.items()):
            g = tc[(tc.kind == kind) & (tc.region == reg)].sort_values("lag")
            if g.empty:
                continue
            for c in CONDS:
                ax.plot(g.time_s, g[f"{c}_mean"], color=colours[c], lw=2, label=labels[c])
                ax.fill_between(g.time_s, g[f"{c}_mean"] - g[f"{c}_sem"], g[f"{c}_mean"] + g[f"{c}_sem"],
                                color=colours[c], alpha=0.15, lw=0)
            ax.axhline(0, color="#0b0b0b", lw=0.6); ax.axvline(0, color="#0b0b0b", lw=0.6, ls=":")
            ax.set_title(f"{title} (n = {int(g.n.iloc[0])})", fontsize=10, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Time from retrieval cue (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Signal change (%)")
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle({"object_matched": "Object trials matched for search time over the three retrievals"}.get(
            kind, f"{kind.capitalize()} trials with a button press") + ": signal over the trial (shading: SEM)", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(opj(str(step_plot_dir(__file__, SUB.rstrip("/") or None)), f"time_course_{kind}.{ext}"), dpi=300)
        plt.close(fig)
    print(f"{w.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
