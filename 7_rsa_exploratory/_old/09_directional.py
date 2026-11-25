#!/usr/bin/env python3
"""
Is the within-sequence position code directional (forward stepping)?

Idea: to retrieve item k, people step through items 1..k-1. Then the retrieval of
item k should carry its predecessors, but the retrieval of item k should not
carry its successors.

Part A  (indirect, RSA; run with a pattern source: raw | lss | glmsingle)
    Trial x trial similarity is symmetric, so direction cannot be read off it.
    But if trial k contains items 1..k, two trials of one scene share all items
    up to the earlier one. Prediction for pairs (i < j) of the same scene:
    dissimilarity = 1 - i/j ("prefix"), i.e. later neighbours are closer than
    early ones (4-5 closer than 1-2).
    Per subject and ROI: mean z-dissimilarity of same-scene pairs per cell (i, j),
    minus the same cell for pairs from other scenes (removes the general position
    code), averaged over the six retrieval pair sets. Then
      adjacent_slope : slope of the four adjacent cells over i (prefix predicts < 0)
      prefix, distance : cells ~ prefix + distance (both z-scored, fitted together)
    Output: outputs/09_directional/<source>/rsa_*.csv

Part B  (direct, classifier; run with source "decoding")
    Uses the category-classifier probabilities of the decoding pipeline
    (4.6 tables: probabilities of the 5 items of the retrieved sequence and of the
    5 items of the other sequence, per trial and TR; classifier trained on the
    localizer, so the template is independent of retrieval).
    Evidence for the item at lag l = (item position - retrieved position) is
    P(item of the retrieved sequence) - P(item at the same position of the other
    sequence), in percentage points, TRs 3-8 after the cue.
      lag profile           : all lags -4..+4
      predecessor/successor : lag -1 and lag +1 on trials at positions 2-4 (both exist)
      directional           : lag -1 minus lag +1; all preceding minus all following
      by TR                 : lag -1, lag +1 and target per TR
    Separately for object and scene trials, per session, all / correct trials.
    Output: outputs/09_directional/decoding/dec_*.csv

Usage:  python 09_directional.py <raw|lss|glmsingle|decoding> [n_jobs]
"""

import glob
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "lss"
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT_DIR = opj(cfg.OUT_ROOT, "09_directional", SOURCE)


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


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
        ("pre", "post1"), ("pre", "post2"), ("post1", "post2")]


def rsa_one(rsa, sub, roi):
    sets = rsa.load_sets(sub, roi)
    trs = rsa.MAIN_WINDOW if rsa.SOURCE == "raw" else (0, 0)
    rows = []
    for sa, sb in SETS:
        if sa not in sets or sb not in sets:
            continue
        (ta, pa), (tb, pb) = sets[sa], sets[sb]
        pairs = rsa.pair_table(ta, tb, sa == sb)
        d = rsa.corr_distance(rsa.trial_vectors(ta, pa, trs, True), rsa.trial_vectors(tb, pb, trs, True))
        y = d[pairs.ia.values, pairs.ib.values]
        sel = ((pairs.pairtype == "oo") & (pairs.same_run == 0) & np.isfinite(y)).values
        p, y = pairs[sel].reset_index(drop=True), y[sel]
        if len(y) < 50:
            continue
        y = (y - y.mean()) / y.std()
        a, b = ta.position.values[p.ia.values], tb.position.values[p.ib.values]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        name = sa if sa == sb else f"{sa}x{sb}"
        for i in range(1, 5):
            for j in range(i + 1, 6):
                m = (lo == i) & (hi == j)
                same, other = m & (p.scene.values == 0), m & (p.scene.values == 1)
                if same.sum() >= 1 and other.sum() >= 5:
                    rows.append(dict(subject=sub, roi=roi, pair_set=name, i=i, j=j,
                                     same_scene=y[same].mean(), other_scene=y[other].mean(),
                                     n_same=int(same.sum())))
    return pd.DataFrame(rows)


def part_a():
    import importlib
    rsa = importlib.import_module("02_binary_rsa")
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(rsa.IN_DIR, "*sleep", "*_trials.csv"))})
    rois = [r for r in cfg.ROIS if r != "hpc"]
    res = Parallel(n_jobs=N_JOBS)(delayed(rsa_one)(rsa, s, r) for s in subs for r in rois)
    cells = pd.concat(res, ignore_index=True)
    cells["diff"] = cells.same_scene - cells.other_scene
    cells.to_csv(opj(OUT_DIR, "rsa_subject_cells.csv"), index=False)

    groups = {"all_sets": None, "evening": ["pre"], "morning": ["post1", "post2"]}
    stat_rows, cell_rows = [], []
    for gname, members in groups.items():
        c = cells if members is None else cells[cells.pair_set.isin(members)]
        avg = c.groupby(["subject", "roi", "i", "j"], as_index=False)[["diff", "same_scene", "other_scene"]].mean()
        for (roi, i, j), g in avg.groupby(["roi", "i", "j"]):
            cell_rows.append(dict(group=gname, roi=roi, i=i, j=j, **tt(g["diff"]),
                                  same_scene=g.same_scene.mean(), other_scene=g.other_scene.mean()))
        per = []
        for (sub, roi), g in avg.groupby(["subject", "roi"]):
            if len(g) < 10:
                continue
            dist = (g.j - g.i).values.astype(float)
            prefix = 1.0 - g.i.values / g.j.values
            z = lambda v: (v - v.mean()) / v.std()
            X = np.column_stack([np.ones(len(g)), z(prefix), z(dist)])
            b = np.linalg.lstsq(X, g["diff"].values, rcond=None)[0]
            adj = g[g.j - g.i == 1].sort_values("i")
            per.append(dict(subject=sub, roi=roi, prefix=b[1], distance=b[2],
                            adjacent_slope=np.polyfit(adj.i.values, adj["diff"].values, 1)[0],
                            late_minus_early=adj["diff"].values[2:].mean() - adj["diff"].values[:2].mean()))
        per = pd.DataFrame(per)
        per.to_csv(opj(OUT_DIR, f"rsa_subject_models_{gname}.csv"), index=False)
        for roi, g in per.groupby("roi"):
            for m in ("prefix", "distance", "adjacent_slope", "late_minus_early"):
                stat_rows.append(dict(group=gname, roi=roi, measure=m, **tt(g[m])))
    st = pd.DataFrame(stat_rows)
    st["p_fdr_rois"] = st.groupby(["group", "measure"])["p"].transform(fdr)
    st.to_csv(opj(OUT_DIR, "rsa_group_models.csv"), index=False)
    pd.DataFrame(cell_rows).to_csv(opj(OUT_DIR, "rsa_group_cells.csv"), index=False)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
DEC_ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl",
            "superior_frontal", "rsc"]
SODA_DIR = opj(cfg.PATH_DECODING, "4.6_SODA_based_on_singlesubj_4.4_{session}", "SODA_results_csv")
SIGNAL_TRS = (3, 8)
SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}


def dec_roi(roi):
    longs = []
    for session in ("presleep", "postsleep"):
        f = opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                                    "retrieved_probs", "control_probs", "behav_correct"], low_memory=False)
        R = 100 * np.array([eval(x, SAFE_EVAL) for x in d.retrieved_probs], dtype=float)
        C = 100 * np.array([eval(x, SAFE_EVAL) for x in d.control_probs], dtype=float)
        p = d.retrieved_position.astype(int).values
        cond = np.where(session == "presleep", "pre", np.where(d.trial.values <= 70, "post1", "post2"))
        for j in range(1, 6):
            longs.append(pd.DataFrame(dict(
                subject=d.subject.values, cond=cond, kind=d.object_scene_retrieval.values, tr=d.tr.values,
                position=p, correct=d.behav_correct.astype(bool).values, lag=j - p,
                ev=R[:, j - 1] - C[:, j - 1], ret=R[:, j - 1], ctl=C[:, j - 1])))
    x = pd.concat(longs, ignore_index=True)
    x["roi"] = roi
    return x


def part_b():
    data = pd.concat(Parallel(n_jobs=min(N_JOBS, len(DEC_ROIS)))(delayed(dec_roi)(r) for r in DEC_ROIS),
                     ignore_index=True)
    conds = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post1": ["post1"], "post2": ["post2"]}
    win = data[(data.tr >= SIGNAL_TRS[0]) & (data.tr <= SIGNAL_TRS[1])]
    lag_rows, dir_rows, tr_rows = [], [], []
    for trials in ("all", "correct"):
        w0 = win if trials == "all" else win[win.correct]
        for cname, cs in conds.items():
            w = w0[w0.cond.isin(cs)]
            # lag profile
            s = w.groupby(["roi", "kind", "subject", "lag"], as_index=False)[["ev", "ret", "ctl"]].mean()
            for (roi, kind, lag), g in s.groupby(["roi", "kind", "lag"]):
                lag_rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, lag=lag, **tt(g.ev),
                                     ret=g.ret.mean(), ctl=g.ctl.mean()))
            # direct neighbours on trials that have both
            mid = w[w.position.isin([2, 3, 4]) & w.lag.isin([-1, 1])]
            s = mid.groupby(["roi", "kind", "subject", "lag"]).ev.mean().unstack("lag")
            s["directional"] = s[-1] - s[1]
            for (roi, kind), g in s.groupby(["roi", "kind"]):
                for col, lab in ((-1, "predecessor"), (1, "successor"), ("directional", "predecessor - successor")):
                    dir_rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, measure=lab, **tt(g[col])))
            # all preceding vs all following (positions 2-4)
            mid = w[w.position.isin([2, 3, 4]) & (w.lag != 0)].copy()
            mid["side"] = np.where(mid.lag < 0, "prec", "foll")
            s = mid.groupby(["roi", "kind", "subject", "side"]).ev.mean().unstack("side")
            for (roi, kind), g in s.groupby(["roi", "kind"]):
                dir_rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind,
                                     measure="all preceding - all following", **tt(g.prec - g.foll)))
    # per TR, pooled sessions, all trials
    mid = data[data.position.isin([2, 3, 4]) & data.lag.isin([-1, 0, 1])]
    s = mid.groupby(["roi", "kind", "subject", "tr", "lag"], as_index=False).ev.mean()
    for (roi, kind, tr, lag), g in s.groupby(["roi", "kind", "tr", "lag"]):
        tr_rows.append(dict(roi=roi, kind=kind, tr=tr, lag=lag, **tt(g.ev)))
    lag = pd.DataFrame(lag_rows)
    dr = pd.DataFrame(dir_rows)
    dr["p_fdr_rois"] = dr.groupby(["trials", "cond", "kind", "measure"])["p"].transform(fdr)
    lag.to_csv(opj(OUT_DIR, "dec_lag_profile.csv"), index=False)
    dr.to_csv(opj(OUT_DIR, "dec_directional.csv"), index=False)
    pd.DataFrame(tr_rows).to_csv(opj(OUT_DIR, "dec_by_tr.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    part_b() if SOURCE == "decoding" else part_a()
    print("written to", OUT_DIR)
