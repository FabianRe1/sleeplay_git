#!/usr/bin/env python3
"""
Is the position effect more than a shared digit on the screen?

On object trials the cue shows the scene image and the position digit, so
"same position" pairs share the digit. Pairs at different positions never share
it, so a graded effect among them (neighbouring positions more similar than
distant ones) cannot come from the digit.

Per subject / ROI / pair set, object x object pairs from different runs:
    z(dissimilarity) ~ context + scene + category + same_position + distance
  same_position : 0 same digit, 1 different (absorbs the digit / identity effect)
  distance      : |pos_i - pos_j| centred among different-position pairs, 0 for
                  same-position pairs. Positive beta = further apart, less similar.
Also saved: mean z-dissimilarity per ordinal distance (0-4), for pairs within a
scene and pairs from different scenes.

Neighbour advantage (group_neighbour.csv): mean z-dissimilarity at distance >= 2
minus distance 1, separately for pairs in the same scene, in different scenes of
the same context, and in different contexts. A digit / number code predicts the
same advantage everywhere; a sequence code predicts more within a scene.

Scene order (model names so_*): the scene's position within its context is not
shown on object trials, so it is free of both the picture and the digit.
    z(dissimilarity) ~ context + scene + category + same_position + distance
                       + so_within + so_across
  so_within : |scene position_i - scene position_j| among different scenes of the
              same context (1-4), centred there, 0 elsewhere
  so_across : the same distance (0-4) among pairs from different contexts

Usage:  python 05_position_distance.py <source> [n_jobs]
Output: outputs/05_position_distance/<source>/
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

rsa = importlib.import_module("02_binary_rsa")          # reads the source from sys.argv[1]
OUT_DIR = opj(cfg.OUT_ROOT, "05_position_distance", rsa.SOURCE)
SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
        ("pre", "post1"), ("pre", "post2"), ("post1", "post2")]
COLS = ["context", "scene", "category", "same_position", "distance"]


def one(sub, roi):
    sets = rsa.load_sets(sub, roi)
    trs = rsa.MAIN_WINDOW if rsa.SOURCE == "raw" else (0, 0)
    betas, curve, neigh = [], [], []
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
        diff = dist > 0
        lin = np.where(diff, dist - dist[diff].mean(), 0.0)
        X = np.column_stack([p.context, p.scene, p.category, diff.astype(float), lin])
        b, _, n = rsa.regress(y, X)
        name = sa if sa == sb else f"{sa}x{sb}"
        betas += [dict(subject=sub, roi=roi, pair_set=name, model=c, beta=v) for c, v in zip(COLS, b)]
        # scene order
        sd = np.abs(ta.scene_position.values[p.ia.values] - tb.scene_position.values[p.ib.values]).astype(float)
        w = (p.context.values == 0) & (p.scene.values == 1)
        a = p.context.values == 1
        so_w = np.where(w, sd - sd[w].mean(), 0.0)
        so_a = np.where(a, sd - sd[a].mean(), 0.0)
        b2, _, _ = rsa.regress(y, np.column_stack([X, so_w, so_a]))
        betas += [dict(subject=sub, roi=roi, pair_set=name, model=c, beta=v)
                  for c, v in zip(["so_within", "so_across"], b2[-2:])]
        for where, m in (("same_context", w), ("other_context", a)):
            for k in range(5):
                mm = m & (sd == k)
                if mm.sum() >= 5:
                    curve.append(dict(subject=sub, roi=roi, pair_set=name, where=f"sceneorder_{where}",
                                      distance=k, z_dissim=y[mm].mean(), n_pairs=int(mm.sum())))
        # neighbour advantage of object position, by how the two objects are related
        for where, m in (("same_scene", p.scene.values == 0), ("same_context_other_scene", w),
                         ("other_context", a)):
            near, far = m & (dist == 1), m & (dist >= 2)
            if near.sum() >= 5 and far.sum() >= 5:
                neigh.append(dict(subject=sub, roi=roi, pair_set=name, where=where,
                                  advantage=y[far].mean() - y[near].mean()))
        for where, m in (("within_scene", p.scene.values == 0), ("across_scenes", p.scene.values == 1)):
            for k in range(5):
                mm = m & (dist == k)
                if mm.sum() >= 5:
                    curve.append(dict(subject=sub, roi=roi, pair_set=name, where=where, distance=k,
                                      z_dissim=y[mm].mean(), n_pairs=int(mm.sum())))
    return pd.DataFrame(betas), pd.DataFrame(curve), pd.DataFrame(neigh)


def ttests(df, value, keys):
    rows = []
    for k, g in df.groupby(keys):
        v = g[value].dropna().values
        if len(v) >= 10:
            t, p = stats.ttest_1samp(v, 0)
            rows.append(dict(zip(keys, k), n=len(v), mean=v.mean(), t=t, p=p))
    return pd.DataFrame(rows)


def main():
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(rsa.IN_DIR, "*sleep", "*_trials.csv"))})
    rois = [r for r in cfg.ROIS if r != "hpc"]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(s, r) for s in subs for r in rois)
    betas = pd.concat([r[0] for r in res], ignore_index=True)
    curve = pd.concat([r[1] for r in res], ignore_index=True)
    neigh = pd.concat([r[2] for r in res], ignore_index=True)
    neigh.to_csv(opj(OUT_DIR, "subject_neighbour.csv"), index=False)
    wide = neigh.pivot_table(index=["subject", "roi", "pair_set"], columns="where", values="advantage").reset_index()
    wide["scene_minus_other"] = wide["same_scene"] - wide[["same_context_other_scene", "other_context"]].mean(axis=1)
    wide["context_minus_other"] = wide["same_context_other_scene"] - wide["other_context"]
    long = wide.melt(id_vars=["subject", "roi", "pair_set"], var_name="where", value_name="advantage")
    gn = ttests(long, "advantage", ["pair_set", "where", "roi"])
    pn = long[long.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "where", "roi"], as_index=False).advantage.mean()
    gpn = ttests(pn, "advantage", ["where", "roi"])
    gpn.insert(0, "pair_set", "all_sessions")
    pd.concat([gn, gpn], ignore_index=True).to_csv(opj(OUT_DIR, "group_neighbour.csv"), index=False)
    betas.to_csv(opj(OUT_DIR, "subject_betas.csv"), index=False)
    curve.to_csv(opj(OUT_DIR, "subject_distance_curve.csv"), index=False)

    g = ttests(betas, "beta", ["pair_set", "model", "roi"])
    # pooled over the three within-session sets: one value per subject
    within = betas[betas.pair_set.isin(["pre", "post1", "post2"])]
    pooled = within.groupby(["subject", "model", "roi"], as_index=False).beta.mean()
    gp = ttests(pooled, "beta", ["model", "roi"])
    gp.insert(0, "pair_set", "all_sessions")
    pd.concat([g, gp], ignore_index=True).to_csv(opj(OUT_DIR, "group_beta_vs_zero.csv"), index=False)
    curve.groupby(["pair_set", "roi", "where", "distance"], as_index=False).agg(
        z_dissim=("z_dissim", "mean"), sem=("z_dissim", "sem"), n=("subject", "nunique")
    ).to_csv(opj(OUT_DIR, "group_distance_curve.csv"), index=False)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
