#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 - Univariate retrieval activity, evening versus morning, for the same items.

Question: when the same item is retrieved in the evening and in the morning, does
the mean activity of a region change (for example more parietal / precuneus and
less hippocampal engagement after sleep)?

Data: the mean beta per retrieval trial and anatomical region from D1 (one model
per run with a regressor per trial; percent signal change against the run's
implicit baseline). Run for both event models of D1: events of a fixed 3 s
("fixed") and events from cue onset to the response ("cue_to_response").

  item-matched difference : per participant, the mean over items of (beta of the
                            item in session A - beta of the same item in session
                            B); items = objects (object trials) and scenes (scene
                            trials, two trials per session averaged)
  variant                 : all trials (the variants "item correct in both sessions"
                            and "relative to the whole-brain mean" agreed with it
                            throughout and were removed on 2026-11-22)
  RT-adjusted             : per participant, beta ~ session + response time; the
                            session coefficients and the coefficient per second
  level per session       : mean activity per session against baseline
  region contrast         : (parietal + precuneus) - hippocampus
  RT relation per session : per participant and session, slope of activity on
                            response time; tested per session and between sessions
Sessions: pre = evening, post1 / post2 = morning pass 1 / 2, post = mean of both.
Main family for correction (p_fdr_main): parietal + precuneus, hippocampus, MTL,
visual, caudal ACC (control).

Input   outputs/D1_trial_patterns/region_means/<events>/<sub>_<session>_region_means.csv
        outputs/D1_trial_patterns/region_voxel_counts/
Output  outputs/E1_session_comparison/<events>/
          trial_roi_means.csv, roi_voxel_counts.csv, subject_item_matched.csv,
          subject_rt_adjusted.csv, group_session_differences.csv,
          group_level_per_session.csv
        outputs/E1_session_comparison/rt_relation_per_session.csv   (both event models)

Usage:  python E1_session_comparison.py

Not a port of the original pipeline: added 2026-11-19, ported from
7_rsa_exploratory/15_univariate_session.py and part D of
18_supplementary_checks.py; logic unchanged, region means read from D1 instead of
whole-brain beta files.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir

EVENTS = {"fixed": "fixed 3 s", "cue_to_response": "cue to response"}
ROIS = {"PARIETAL_PRECUNEUS": "Parietal + precuneus", "PARIETAL": "Parietal", "PRECUNEUS": "Precuneus",
        "HPC": "Hippocampus", "PHPC": "Parahippocampal", "ERH": "Entorhinal", "MTL": "MTL",
        "VIS": "Visual", "CAUDAL_ACC": "Caudal ACC (control)", "RSC": "Retrosplenial",
        "POSTERIOR_CINGULATE": "Posterior cingulate", "SUPERIOR_FRONTAL": "Superior frontal",
        "MEDIAL_OFC": "Medial OFC", "AUD": "Auditory", "SOMATO": "Somatosensory",
        "ventricles": "Ventricles", "wholebrain": "Whole brain"}
MAIN = ["PARIETAL_PRECUNEUS", "HPC", "MTL", "VIS", "CAUDAL_ACC"]
CONTRASTS = [("post1", "pre"), ("post", "pre"), ("post2", "post1")]
RT_REGIONS = {"PRECUNEUS": "Precuneus", "PARIETAL_PRECUNEUS": "Parietal + precuneus", "RSC": "Retrosplenial",
              "CAUDAL_ACC": "Caudal ACC", "PHPC": "Parahippocampal", "HPC": "Hippocampus", "VIS": "Visual"}


def tt(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def fdr(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        order = np.argsort(p[ok])
        q = p[ok][order] * ok.sum() / (np.arange(ok.sum()) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        res = np.empty(ok.sum())
        res[order] = np.minimum(q, 1)
        out[ok] = res
    return out


def load_trials(events):
    """Trial table with the mean beta of every region, all participants (from D1)."""
    in_dir = str(step_input_dir("D1_trial_patterns", opj("region_means", events)))
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(in_dir, "*_presleep_region_means.csv"))})
    parts = []
    for sub in subs:
        for phase in ("presleep", "postsleep"):
            f = opj(in_dir, f"{sub}_{phase}_region_means.csv")
            if not os.path.exists(f):
                continue
            t = pd.read_csv(f)
            out = t[["subject", "kind", "object", "scene", "position", "correct", "rt", "run_idx"]].copy()
            out["cond"] = "pre" if phase == "presleep" else np.where(t.run_idx < 3, "post1", "post2")
            out["item"] = np.where(t.kind == "object", t.object, t.scene)
            for roi in ROIS:
                if roi in t.columns:
                    out[roi] = t[roi]
            parts.append(out)
    return pd.concat(parts, ignore_index=True)


def item_matched(tr, rois, both_correct=False):
    """Per participant and kind: mean over items of the between-session difference, per region."""
    rows = []
    for (sub, kind), g in tr.groupby(["subject", "kind"]):
        m = g.groupby(["item", "cond"])[rois].mean().unstack("cond")
        ok = g.groupby(["item", "cond"])["correct"].min().unstack("cond")
        for a, b in CONTRASTS:
            conds_a = ["post1", "post2"] if a == "post" else [a]
            if not all(c in ok.columns for c in conds_a + [b]):
                continue
            keep = ok[conds_a + [b]].notna().all(axis=1)
            if both_correct:
                keep &= (ok[conds_a + [b]] == True).all(axis=1)          # noqa: E712
            row = dict(subject=sub, kind=kind, contrast=f"{a} - {b}", n_items=int(keep.sum()))
            if keep.sum() < 5:
                continue
            for r in rois:
                va = m[r][conds_a].mean(axis=1)
                row[r] = float((va - m[r][b])[keep].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def rt_adjusted(tr, rois):
    rows = []
    for (sub, kind), g in tr.groupby(["subject", "kind"]):
        g = g.dropna(subset=["rt"])
        if g.cond.nunique() < 3 or len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g)), (g.cond == "post1").values, (g.cond == "post2").values,
                             g.rt.values - g.rt.mean()]).astype(float)
        for r in rois:
            b = np.linalg.lstsq(X, g[r].values, rcond=None)[0]
            rows.append(dict(subject=sub, kind=kind, roi=r, post1_minus_pre=b[1], post2_minus_pre=b[2],
                             post_minus_pre=(b[1] + b[2]) / 2, per_second_rt=b[3]))
    return pd.DataFrame(rows)


def group(df, rois, extra):
    rows = []
    for keys, g in df.groupby(["kind", "contrast"]):
        for r in rois:
            rows.append(dict(kind=keys[0], contrast=keys[1], roi=r, **extra, **tt(g[r])))
    return rows


def T(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.ttest_1samp(v, 0)[0] if len(v) > 9 else np.nan


def rt_relation(tr, label):
    """Slope of activity on response time per participant and session (object and scene trials)."""
    out = []
    t = tr.dropna(subset=["rt"])
    for kind in ("object", "scene"):
        rows = []
        for (sub, cond), g in t[t.kind == kind].groupby(["subject", "cond"]):
            if len(g) >= 10 and g.rt.std() > 0:
                for r in RT_REGIONS:
                    rows.append(dict(subject=sub, cond=cond, roi=r, slope=np.polyfit(g.rt - g.rt.mean(), g[r], 1)[0],
                                     r=np.corrcoef(g.rt, g[r])[0, 1]))
        d = pd.DataFrame(rows)
        for r, rl in RT_REGIONS.items():
            w = d[d.roi == r].pivot(index="subject", columns="cond", values="slope")
            wr = d[d.roi == r].pivot(index="subject", columns="cond", values="r")
            row = dict(events=label, kind=kind, region=rl)
            for c in ("pre", "post1", "post2"):
                row[f"t_{c}"], row[f"slope_{c}"], row[f"mean_r_{c}"] = T(w[c]), w[c].mean(), wr[c].mean()
            row["t_post1-pre"], row["t_post2-post1"] = T(w.post1 - w.pre), T(w.post2 - w.post1)
            out.append(row)
    return out


def run(events):
    out_dir = str(step_output_dir(__file__, events))
    tr = load_trials(events)
    rois = [r for r in ROIS if r in tr.columns]
    counts = sorted(glob.glob(opj(str(step_input_dir("D1_trial_patterns", "region_voxel_counts")), "*_region_voxel_counts.csv")))
    pd.concat([pd.read_csv(f) for f in counts], ignore_index=True).to_csv(opj(out_dir, "roi_voxel_counts.csv"), index=False)
    tr.to_csv(opj(out_dir, "trial_roi_means.csv"), index=False)
    rt_rows = rt_relation(tr, EVENTS[events])

    tr["PARPREC_minus_HPC"] = tr["PARIETAL_PRECUNEUS"] - tr["HPC"]
    rois_x = rois + ["PARPREC_minus_HPC"]

    # Only "all trials" (and, below, "RT-adjusted") are reported. The variants "both correct" and
    # "relative to whole brain" agreed with it throughout and were removed on 2026-11-22.
    rows, subj = [], []
    for vname, d, bc in (("all trials", tr, False),):
        im = item_matched(d, rois_x, bc)
        subj.append(im.assign(variant=vname))
        rows += group(im, rois_x, dict(variant=vname))
    g = pd.DataFrame(rows)

    ra = rt_adjusted(tr, rois_x)
    ra.to_csv(opj(out_dir, "subject_rt_adjusted.csv"), index=False)
    rrows = []
    for (kind, roi), gg in ra.groupby(["kind", "roi"]):
        for col, lab in (("post1_minus_pre", "post1 - pre"), ("post_minus_pre", "post - pre"), ("per_second_rt", "per second RT")):
            rrows.append(dict(kind=kind, contrast=lab, roi=roi, variant="RT-adjusted", **tt(gg[col])))
    g = pd.concat([g, pd.DataFrame(rrows)], ignore_index=True)
    g["p_fdr_main"] = np.nan
    m = g.roi.isin(MAIN)
    g.loc[m, "p_fdr_main"] = g[m].groupby(["kind", "contrast", "variant"])["p"].transform(fdr)
    g["region"] = g.roi.map(ROIS).fillna(g.roi)
    g.to_csv(opj(out_dir, "group_session_differences.csv"), index=False)
    pd.concat(subj, ignore_index=True).to_csv(opj(out_dir, "subject_item_matched.csv"), index=False)

    # level per session (activity against the implicit baseline)
    lev = tr.groupby(["subject", "kind", "cond"])[rois_x].mean().reset_index()
    lrows = []
    for (kind, cond), gg in lev.groupby(["kind", "cond"]):
        for r in rois_x:
            lrows.append(dict(kind=kind, cond=cond, roi=r, region=ROIS.get(r, r), **tt(gg[r])))
    pd.DataFrame(lrows).to_csv(opj(out_dir, "group_level_per_session.csv"), index=False)
    print("written to", out_dir, "| subjects:", tr.subject.nunique())
    return rt_rows


def main():
    rt_rows = []
    for events in EVENTS:
        rt_rows += run(events)
    pd.DataFrame(rt_rows).to_csv(opj(str(step_output_dir(__file__)), "rt_relation_per_session.csv"), index=False)


if __name__ == "__main__":
    main()
