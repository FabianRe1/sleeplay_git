#!/usr/bin/env python3
"""
Two follow-up checks on the binary RSA.

A. Is the neighbour advantage of object position larger within a scene?
   (retrieval, object x object pairs from different runs, different positions)
   advantage = mean z-dissimilarity at distance >= 2 minus distance 1, for pairs
   in the same scene and for pairs in other scenes; "diff" = same scene - other.
   Versions: all trials / both trials correct. Also the regression interaction
       z(dissim) ~ context + scene + distance + distance x same_scene
   ("slope_x_scene" > 0 = dissimilarity rises faster with distance within a scene).
   Tested over all six pair sets, the three within-session sets, the three
   between-session sets, evening, morning, and morning - evening.

B. Why is context reversed at encoding?
   Object presentations come in a fixed order: run 1 = day, night, day, night
   (repetitions 1 and 2), run 2 = day, night (repetition 3). Pairs are taken
   between the two runs. Mean z-dissimilarity of different-scene pairs is given
   for every cell (repetition in run 1, context in run 1, context in run 2).

Usage:  python 06_followups.py <source> [n_jobs]
Output: outputs/06_followups/<source>/
"""

import glob
import importlib
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

rsa = importlib.import_module("02_binary_rsa")
OUT_DIR = opj(cfg.OUT_ROOT, "06_followups", rsa.SOURCE)
SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
        ("pre", "post1"), ("pre", "post2"), ("post1", "post2")]
GROUPS = {"all_sets": ["pre", "post1", "post2", "prexpost1", "prexpost2", "post1xpost2"],
          "within_session": ["pre", "post1", "post2"],
          "between_session": ["prexpost1", "prexpost2", "post1xpost2"],
          "evening": ["pre"], "morning": ["post1", "post2"]}


def one(sub, roi):
    sets = rsa.load_sets(sub, roi)
    trs = rsa.MAIN_WINDOW if rsa.SOURCE == "raw" else (0, 0)
    rows, enc = [], []
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
        dist = np.abs(ta.position.values[p.ia.values] - tb.position.values[p.ib.values]).astype(float)
        ok = (ta.correct.values[p.ia.values] == True) & (tb.correct.values[p.ib.values] == True)  # noqa: E712
        same_scene = p.scene.values == 0
        name = sa if sa == sb else f"{sa}x{sb}"
        out = dict(subject=sub, roi=roi, pair_set=name)
        for tag, keep in (("", np.ones(len(y), bool)), ("_correct", ok)):
            adv = {}
            for where, m in (("same_scene", same_scene), ("other_scene", ~same_scene)):
                near, far = keep & m & (dist == 1), keep & m & (dist >= 2)
                adv[where] = y[far].mean() - y[near].mean() if near.sum() >= 5 and far.sum() >= 5 else np.nan
            out[f"adv_same_scene{tag}"] = adv["same_scene"]
            out[f"adv_other_scene{tag}"] = adv["other_scene"]
            out[f"diff{tag}"] = adv["same_scene"] - adv["other_scene"]
        m = dist >= 1
        dc = dist[m] - dist[m].mean()
        X = np.column_stack([p.context.values[m], p.scene.values[m], dc, dc * same_scene[m]])
        b, _, _ = rsa.regress(y[m], X)
        out["slope"], out["slope_x_scene"] = b[2], b[3]
        rows.append(out)

    if "enc" in sets:                       # part B
        t, pat = sets["enc"]
        pairs = rsa.pair_table(t, t, True)
        for variant, centre in (("centred", True), ("not_centred", False)):
            d = rsa.corr_distance(rsa.trial_vectors(t, pat, (0, 0), centre),
                                  rsa.trial_vectors(t, pat, (0, 0), centre))
            y = d[pairs.ia.values, pairs.ib.values]
            sel = ((pairs.pairtype == "oo") & (pairs.same_run == 0) & np.isfinite(y)).values
            p, y = pairs[sel].reset_index(drop=True), y[sel]
            if len(y) < 50:
                continue
            y = (y - y.mean()) / y.std()
            a, b = t.iloc[p.ia.values].reset_index(drop=True), t.iloc[p.ib.values].reset_index(drop=True)
            first = a.run_idx.values < b.run_idx.values          # a = the earlier run
            rep1 = np.where(first, a.rep.values, b.rep.values)
            c1 = np.where(first, a.context.values, b.context.values)
            c2 = np.where(first, b.context.values, a.context.values)
            other = p.scene.values == 1
            for r in np.unique(rep1):
                for x1 in ("day", "night"):
                    for x2 in ("day", "night"):
                        mm = other & (rep1 == r) & (c1 == x1) & (c2 == x2)
                        if mm.sum() >= 5:
                            enc.append(dict(subject=sub, roi=roi, variant=variant, rep_run1=int(r),
                                            context_run1=x1, context_run2=x2, z_dissim=y[mm].mean()))
    return pd.DataFrame(rows), pd.DataFrame(enc)


def tt(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def main():
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(rsa.IN_DIR, "*sleep", "*_trials.csv"))})
    rois = [r for r in cfg.ROIS if r != "hpc"]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(s, r) for s in subs for r in rois)
    sub = pd.concat([r[0] for r in res], ignore_index=True)
    sub.to_csv(opj(OUT_DIR, "subject_neighbour_scene.csv"), index=False)
    measures = [c for c in sub.columns if c not in ("subject", "roi", "pair_set")]
    rows, per_group = [], {}
    for gname, members in GROUPS.items():
        avg = sub[sub.pair_set.isin(members)].groupby(["subject", "roi"], as_index=False)[measures].mean()
        per_group[gname] = avg.set_index(["subject", "roi"])
        for roi, g in avg.groupby("roi"):
            for m in measures:
                rows.append(dict(group=gname, roi=roi, measure=m, **tt(g[m])))
    delta = (per_group["morning"] - per_group["evening"]).reset_index()
    for roi, g in delta.groupby("roi"):
        for m in measures:
            rows.append(dict(group="morning_minus_evening", roi=roi, measure=m, **tt(g[m])))
    grp = pd.DataFrame(rows)
    grp["p_fdr_rois"] = grp.groupby(["group", "measure"])["p"].transform(rsa.fdr_bh)
    grp.to_csv(opj(OUT_DIR, "group_neighbour_scene.csv"), index=False)

    enc = [r[1] for r in res if len(r[1])]
    if enc:
        enc = pd.concat(enc, ignore_index=True)
        enc.to_csv(opj(OUT_DIR, "subject_encoding_context_cells.csv"), index=False)
        enc.groupby(["variant", "roi", "rep_run1", "context_run1", "context_run2"], as_index=False).agg(
            z_dissim=("z_dissim", "mean"), sem=("z_dissim", "sem"), n=("subject", "nunique")
        ).to_csv(opj(OUT_DIR, "group_encoding_context_cells.csv"), index=False)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
