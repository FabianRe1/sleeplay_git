#!/usr/bin/env python3
"""
Does any region hold a successor-like (predictive) representation at retrieval?

Background: with the category classifier, visual cortex and precuneus carry the
item BEFORE the retrieved one, MTL the item AFTER it (09_directional.py, C5).
A successor representation (SR) predicts evidence for upcoming items that decays
with distance (weight gamma^(k-1) for the item k steps ahead) and none for past
items. Its mirror image (predecessor representation) predicts the reverse.

Part A  classifier evidence (decoding pipeline, 4.6 tables), all 12 ROIs
    Evidence of an item = P(item of the retrieved sequence) - P(item at the same
    position of the other sequence), percentage points, TRs 3-8 after the cue.
      forward_g  : discounted mean of the items after the retrieved one
      backward_g : discounted mean of the items before it
      asym_g     : forward - backward
    for gamma = 0.01 (immediate neighbour only), 0.5 and 1.0 (all items equal),
    on trials at positions 2-4 (both sides exist). Also the evidence at every
    forward and backward step, to see the decay.
    Per trial type (object / scene), session, all / correct trials. FDR across ROIs.

Part B  RSA on within-scene pairs (uses the cells written by 09_directional.py)
    For a 5-item chain with forward transitions, SR rows of two items overlap in
    their shared FUTURE, predecessor rows in their shared PAST. Dissimilarity
    predicted by each (1 - correlation of the rows, gamma = 0.5) is regressed
    together onto the cell dissimilarities (same scene minus other scenes).
    The two differ in where neighbours are close: SR early in the sequence,
    predecessor late.

Usage:  python 13_successor_test.py [n_jobs]
Output: outputs/13_successor_test/
"""

import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_DIR = opj(cfg.OUT_ROOT, "13_successor_test")
ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl",
        "superior_frontal", "rsc", "posterior_cingulate", "medial_ofc", "rostral_acc", "caudal_acc"]
SODA_DIR = opj(cfg.PATH_DECODING, "4.6_SODA_based_on_singlesubj_4.4_{session}", "SODA_results_csv")
SIGNAL_TRS = (3, 8)
GAMMAS = [0.01, 0.5, 1.0]
SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
CONDS = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post1": ["post1"], "post2": ["post2"],
         "post": ["post1", "post2"]}


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

def load_roi(roi):
    parts = []
    for session in ("presleep", "postsleep"):
        f = opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv")
        d = pd.read_csv(f, usecols=["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                                    "retrieved_probs", "control_probs", "behav_correct"], low_memory=False)
        d = d[(d.tr >= SIGNAL_TRS[0]) & (d.tr <= SIGNAL_TRS[1])].reset_index(drop=True)
        R = 100 * np.array([eval(x, SAFE_EVAL) for x in d.retrieved_probs], dtype=float)
        C = 100 * np.array([eval(x, SAFE_EVAL) for x in d.control_probs], dtype=float)
        E = R - C
        p = d.retrieved_position.astype(int).values
        out = pd.DataFrame(dict(subject=d.subject.values, kind=d.object_scene_retrieval.values, position=p,
                                correct=d.behav_correct.astype(bool).values,
                                cond=np.where(session == "presleep", "pre",
                                              np.where(d.trial.values <= 70, "post1", "post2"))))
        idx = np.arange(1, 6)[None, :]
        lag = idx - p[:, None]
        for k in range(1, 5):                                   # evidence k steps ahead / back
            for name, sign in (("fwd", 1), ("bwd", -1)):
                m = lag == sign * k
                out[f"{name}{k}"] = np.where(m.any(1), np.where(m, E, 0).sum(1), np.nan)
        for g in GAMMAS:
            for name, m in (("forward", lag > 0), ("backward", lag < 0)):
                w = np.where(m, g ** (np.abs(lag) - 1.0), 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{name}_{g}"] = np.where(w.sum(1) > 0, (w * E).sum(1) / w.sum(1), np.nan)
            out[f"asym_{g}"] = out[f"forward_{g}"] - out[f"backward_{g}"]
        parts.append(out)
    x = pd.concat(parts, ignore_index=True)
    x["roi"] = roi
    return x


def part_a():
    data = pd.concat(Parallel(n_jobs=min(N_JOBS, len(ROIS)))(delayed(load_roi)(r) for r in ROIS),
                     ignore_index=True)
    steps = [f"{n}{k}" for n in ("fwd", "bwd") for k in range(1, 5)]
    scores = [f"{n}_{g}" for g in GAMMAS for n in ("forward", "backward", "asym")]
    rows, subj = [], []
    for trials in ("all", "correct"):
        d0 = data if trials == "all" else data[data.correct]
        for cname, cs in CONDS.items():
            d = d0[d0.cond.isin(cs)]
            s1 = d.groupby(["roi", "kind", "subject"])[steps].mean()                    # every step, all positions
            s2 = d[d.position.isin([2, 3, 4])].groupby(["roi", "kind", "subject"])[scores].mean()
            s = s1.join(s2)
            subj.append(s.reset_index().assign(trials=trials, cond=cname))
            for (roi, kind), g in s.groupby(level=["roi", "kind"]):
                for m in steps + scores:
                    rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, measure=m, **tt(g[m])))
    res = pd.DataFrame(rows)
    res["p_fdr_rois"] = res.groupby(["trials", "cond", "kind", "measure"])["p"].transform(fdr)
    res.to_csv(opj(OUT_DIR, "decoding_group.csv"), index=False)
    pd.concat(subj, ignore_index=True).to_csv(opj(OUT_DIR, "decoding_subject.csv"), index=False)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def chain_models(gamma=0.5, n=5):
    T = np.zeros((n, n))
    for i in range(n - 1):
        T[i, i + 1] = 1.0
    M = np.linalg.inv(np.eye(n) - gamma * T)             # M[i, j] = gamma^(j-i) for j >= i
    sr = 1 - np.corrcoef(M)                              # rows: discounted future of each item
    pr = 1 - np.corrcoef(M.T)                            # columns: discounted past of each item
    return sr, pr


def part_b():
    sr, pr = chain_models()
    rows = []
    for src in ("lss", "raw", "glmsingle"):
        f = opj(cfg.OUT_ROOT, "09_directional", src, "rsa_subject_cells.csv")
        if not os.path.exists(f):
            continue
        cells = pd.read_csv(f)
        for gname, members in {"all_sets": None, "evening": ["pre"], "morning": ["post1", "post2"]}.items():
            c = cells if members is None else cells[cells.pair_set.isin(members)]
            avg = c.groupby(["subject", "roi", "i", "j"], as_index=False)["diff"].mean()
            per = []
            for (sub, roi), g in avg.groupby(["subject", "roi"]):
                if len(g) < 10:
                    continue
                z = lambda v: (v - v.mean()) / v.std()
                a = z(np.array([sr[i - 1, j - 1] for i, j in zip(g.i, g.j)]))
                b = z(np.array([pr[i - 1, j - 1] for i, j in zip(g.i, g.j)]))
                y = g["diff"].values
                joint = np.linalg.lstsq(np.column_stack([np.ones(len(y)), a, b]), y, rcond=None)[0]
                per.append(dict(subject=sub, roi=roi, successor_alone=np.polyfit(a, y, 1)[0],
                                predecessor_alone=np.polyfit(b, y, 1)[0],
                                successor_joint=joint[1], predecessor_joint=joint[2],
                                successor_minus_predecessor=joint[1] - joint[2]))
            per = pd.DataFrame(per)
            for roi, g in per.groupby("roi"):
                for m in per.columns[2:]:
                    rows.append(dict(source=src, group=gname, roi=roi, measure=m, **tt(g[m])))
    res = pd.DataFrame(rows)
    res["p_fdr_rois"] = res.groupby(["source", "group", "measure"])["p"].transform(fdr)
    res.to_csv(opj(OUT_DIR, "rsa_group.csv"), index=False)
    pd.DataFrame(dict(i=[i + 1 for i in range(5) for j in range(5)], j=[j + 1 for i in range(5) for j in range(5)],
                      successor=sr.ravel(), predecessor=pr.ravel())).to_csv(opj(OUT_DIR, "rsa_model_cells.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    part_b()
    part_a()
    print("written to", OUT_DIR)
