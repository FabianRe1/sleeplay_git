#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 - Same-or-different models regressed onto the trial-by-trial dissimilarity.

Measured dissimilarity
    One pattern per retrieval trial (D1, fixed 3 s events). Patterns are
    mean-centred per run and trial type; dissimilarity = 1 - Pearson r across the
    voxels of an ROI.

Trial sets
    pre    evening retrieval (70 trials: 50 objects once, 10 scenes twice)
    post1  morning, first pass (runs 1-3)      post2  morning, second pass (runs 4-6)

Pair sets
    within a set (pre, post1, post2): only pairs from DIFFERENT runs, so that
        temporal autocorrelation within a run cannot produce similarity;
    across sets (post1xpost2, prexpost1, prexpost2): every pair, always from
        different runs. These also allow a same-object predictor.
    Encoding trials are not part of this stage.

Regression, per participant / ROI / pair set / pair type
    z(dissimilarity) ~ 1 + binary predictors (0 = same, 1 = different).
    A positive beta means: trials that share the feature are more similar, over
    and above the other predictors. Because the predictors are nested (same scene
    implies same context; same category implies same position and same category
    sequence), the beta of the broader predictor is its effect for pairs that do
    not share the narrower one. "alone" is the plain difference
    different-minus-same without the other predictors.

Pair types and their predictors
    oo  object trial x object trial : context, scene, catseq, category, position (+ object across sets)
    ss  scene trial  x scene trial  : context, scene, position
    os  object trial x scene trial  : context, scene (object's scene = retrieved scene), position
Variants
    main     all predictors of the pair type
    four     only context, scene, position, category
    four_rt  as four, plus the absolute response-time difference of the two trials
             (z-scored; model `rt_diff`, positive = trials with similar response
             times are more similar). Only pairs in which both trials have a
             response, and only if at least MIN_PAIRS_SUBSET such pairs remain.
             Added 2026-11-23: response time shapes pattern similarity in every
             region, so the four models are also reported next to it.

Group level: one-sample t-tests of the betas (FDR across ROIs), paired session
contrasts, and a mixed model per ROI / pair type / predictor
    beta ~ session + (1 | participant)
giving the effect averaged over the three sessions (`*_overall`) and the test
of a session difference (`p_session`).

Input   outputs/D1_trial_patterns/patterns/fixed/<session>/
Output  outputs/D2_binary_rsa/subject_betas.csv, group_mean_rdms.npz,
        group_{beta,alone}_vs_zero.csv, group_{beta,alone}_session_contrasts.csv,
        group_{beta,alone}_session_lme.csv

Usage:  python D2_binary_rsa.py [n_jobs]

Not a port of the original pipeline: added 2026-11-19, ported from
7_rsa_exploratory/02_binary_rsa.py (variants main and four; logic unchanged).
Variant four_rt was added on 2026-11-23; the rows of main and four are unchanged.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir

EVENTS = "fixed"
ROIS = ["vis", "mtl", "parietal_precuneus", "hpc", "parietal", "precuneus", "cortical_memory",
        "superior_frontal", "posterior_cingulate", "rsc", "medial_ofc", "rostral_acc", "caudal_acc"]
SESSIONS = ["presleep", "postsleep"]
PAIR_SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
             ("post1", "post2"), ("pre", "post1"), ("pre", "post2")]
PREDICTORS = {
    "oo": ["context", "scene", "catseq", "category", "position", "object"],
    "ss": ["context", "scene", "position"],
    "os": ["context", "scene", "position"],
}
FOUR = ["context", "scene", "position", "category"]
VARIANTS = {"main": None, "four": FOUR, "four_rt": FOUR}     # predictors to keep (None = all)
RT_VARIANTS = {"four_rt"}            # variants with the response-time difference as covariate
MIN_PAIRS_SUBSET = 60                # fewest pairs for an analysis restricted to a subset of trials
N_CANON = 70          # 50 object trials + 20 scene trials in canonical order


def in_dir(events: str = EVENTS) -> str:
    return str(step_input_dir("D1_trial_patterns", opj("patterns", events)))


def subjects(events: str = EVENTS) -> list[str]:
    return sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(in_dir(events), "*", "*_trials.csv"))})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_sets(sub: str, roi: str, events: str = EVENTS) -> dict:
    """Return {set name: (trials, patterns[trials x voxels])} for one participant and ROI."""
    sets = {}
    for session in SESSIONS:
        f_tr = opj(in_dir(events), session, f"{sub}_{session}_trials.csv")
        f_pat = opj(in_dir(events), session, f"{sub}_{session}_{roi}_patterns.npy")
        if not (os.path.exists(f_tr) and os.path.exists(f_pat)):
            continue
        trials = pd.read_csv(f_tr)
        pat = np.load(f_pat).astype(np.float32)
        if session == "presleep":
            sets["pre"] = (trials.reset_index(drop=True), pat)
        else:
            for name, sel in (("post1", trials.run_idx < 3), ("post2", trials.run_idx >= 3)):
                sets[name] = (trials[sel].reset_index(drop=True), pat[sel.values])
    return sets


def canonical_index(trials: pd.DataFrame) -> np.ndarray:
    """Position of every trial in the structure-ordered 70-trial layout.

    objects 0-49: context (day, night) > scene position > object position
    scenes 50-69: context > scene position > repetition
    Stimuli differ between participants, the structure does not, so matrices in
    this order can be averaged across participants.
    """
    ctx = (trials.context == "night").astype(int).values
    idx = np.full(len(trials), -1)
    obj = (trials.kind == "object").values
    idx[obj] = ctx[obj] * 25 + (trials.scene_position.values[obj] - 1) * 5 + (trials.position.values[obj] - 1)
    rep = trials.groupby(["kind", "scene"]).cumcount().values
    idx[~obj] = 50 + ctx[~obj] * 10 + (trials.position.values[~obj] - 1) * 2 + np.minimum(rep[~obj], 1)
    return idx.astype(int)


# ---------------------------------------------------------------------------
# dissimilarity and regression
# ---------------------------------------------------------------------------

def trial_vectors(trials, pat, centre=True):
    x = pat
    if centre:
        x = x.copy()
        for _, g in trials.groupby(["run_idx", "kind"]):
            x[g.index] -= np.nanmean(x[g.index], axis=0)
    return x


def corr_distance(xa, xb):
    def usable(x):                      # voxels that are finite in every trial that has data
        rows = np.isfinite(x).any(axis=1)
        return np.isfinite(x[rows]).all(axis=0)
    keep = usable(xa) & usable(xb)
    xa, xb = xa[:, keep], xb[:, keep]

    def z(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return 1.0 - z(xa) @ z(xb).T


def pair_table(ta, tb, same_set):
    """Predictors for every trial pair (a from set A, b from set B)."""
    ia, ib = np.meshgrid(np.arange(len(ta)), np.arange(len(tb)), indexing="ij")
    if same_set:
        keep = ia < ib
        ia, ib = ia[keep], ib[keep]
    else:
        ia, ib = ia.ravel(), ib.ravel()
    a, b = ta.iloc[ia].reset_index(drop=True), tb.iloc[ib].reset_index(drop=True)
    ka, kb = (a.kind == "object").values, (b.kind == "object").values
    p = pd.DataFrame({"ia": ia, "ib": ib})
    p["pairtype"] = np.where(ka & kb, "oo", np.where(~ka & ~kb, "ss", "os"))
    for col in ("context", "scene", "catseq", "category", "position", "object"):
        p[col] = (a[col].values != b[col].values).astype(float)
    p["same_run"] = ((a.run_idx.values == b.run_idx.values) & same_set).astype(float)
    p["lag"] = np.where(p.same_run == 1, np.abs(a.onset_s.values - b.onset_s.values), 0.0)
    p["rt_diff"] = np.abs(a.rt.values.astype(float) - b.rt.values.astype(float))      # NaN without a response
    return p


def regress(y, X):
    """OLS betas of z-scored y on X (intercept added); zero-variance columns give NaN."""
    ok = np.isfinite(y)
    y, X = y[ok], X[ok]
    out = np.full(X.shape[1], np.nan)
    alone = np.full(X.shape[1], np.nan)
    if len(y) < 10 or np.std(y) == 0:
        return out, alone, len(y)
    y = (y - y.mean()) / y.std()
    use = X.std(axis=0) > 0
    A = np.column_stack([np.ones(len(y)), X[:, use]])
    out[use] = np.linalg.lstsq(A, y, rcond=None)[0][1:]
    for k in np.where(use)[0]:
        binary = set(np.unique(X[:, k])) <= {0.0, 1.0}
        alone[k] = (y[X[:, k] == 1].mean() - y[X[:, k] == 0].mean()) if binary else np.nan
    return out, alone, len(y)


def analyse_subject_roi(sub: str, roi: str):
    sets = load_sets(sub, roi)
    rows, mats = [], {}
    for sa, sb in PAIR_SETS:
        if sa not in sets or sb not in sets:
            continue
        (ta, pa), (tb, pb) = sets[sa], sets[sb]
        same_set = sa == sb
        pairs = pair_table(ta, tb, same_set)
        set_name = sa if same_set else f"{sa}x{sb}"
        d = corr_distance(trial_vectors(ta, pa), trial_vectors(tb, pb))
        y_all = d[pairs.ia.values, pairs.ib.values]
        if max(len(ta), len(tb)) <= N_CANON:                       # group RDM
            m = np.full((N_CANON, N_CANON), np.nan)
            ca, cb = canonical_index(ta), canonical_index(tb)
            sel = (pairs.same_run == 0).values
            yz = (y_all - np.nanmean(y_all[sel])) / np.nanstd(y_all[sel])
            m[ca[pairs.ia.values[sel]], cb[pairs.ib.values[sel]]] = yz[sel]
            if same_set:
                m[cb[pairs.ib.values[sel]], ca[pairs.ia.values[sel]]] = yz[sel]
            mats[set_name] = m
        for variant, keep in VARIANTS.items():
            for ptype, preds in PREDICTORS.items():
                if keep is not None:
                    preds = [m_ for m_ in preds if m_ in keep]
                sel = ((pairs.pairtype == ptype) & (pairs.same_run == 0)).values     # between-run pairs only
                if variant in RT_VARIANTS:
                    sel = sel & np.isfinite(pairs.rt_diff.values) & np.isfinite(y_all)
                    if sel.sum() < MIN_PAIRS_SUBSET:
                        continue
                    preds = preds + ["rt_diff"]
                y = y_all[sel]
                X = pairs.loc[sel, preds].values
                if variant in RT_VARIANTS:
                    X[:, -1] = (X[:, -1] - X[:, -1].mean()) / X[:, -1].std()
                beta, alone, n = regress(y, X)
                for k, name in enumerate(preds):
                    if np.isfinite(beta[k]):
                        rows.append(dict(
                            subject=sub, roi=roi, pair_set=set_name, pairtype=ptype,
                            window="win", variant=variant, model=name,
                            beta=beta[k], alone=alone[k], n_pairs=n,
                            n_same=int((X[np.isfinite(y), k] == 0).sum())))
    return pd.DataFrame(rows), mats


# ---------------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------------

def fdr_bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        order = np.argsort(p[ok])
        ranked = p[ok][order] * ok.sum() / np.arange(1, ok.sum() + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        res = np.empty(ok.sum())
        res[order] = np.minimum(adj, 1)
        out[ok] = res
    return out


def one_sample(df, value):
    keys = ["pair_set", "pairtype", "window", "variant", "model", "roi"]
    rows = []
    for k, g in df.groupby(keys):
        x = g[value].dropna().values
        if len(x) < 5:
            continue
        t, p = stats.ttest_1samp(x, 0)
        rows.append(dict(zip(keys, k), n=len(x), mean=x.mean(), sem=stats.sem(x),
                         t=t, p=p, d=x.mean() / x.std(ddof=1)))
    res = pd.DataFrame(rows)
    # FDR across ROIs within each pair set / pair type / variant / model
    res["p_fdr_rois"] = res.groupby(keys[:-1])["p"].transform(fdr_bh)
    return res


def session_contrasts(df, value):
    """Paired differences between pair sets (second minus first)."""
    keys = ["pairtype", "window", "variant", "model", "roi"]
    rows = []
    for a, b in (("pre", "post1"), ("pre", "post2"), ("post1", "post2")):
        wide = df[df.pair_set.isin([a, b])].pivot_table(
            index=keys + ["subject"], columns="pair_set", values=value).dropna()
        if wide.empty:
            continue
        for k, g in wide.groupby(keys):
            if len(g) < 5:
                continue
            diff = (g[b] - g[a]).values
            t, p = stats.ttest_1samp(diff, 0)
            rows.append(dict(zip(keys, k), contrast=f"{b}-{a}", n=len(diff), mean_diff=diff.mean(),
                             t=t, p=p, d=diff.mean() / diff.std(ddof=1)))
    res = pd.DataFrame(rows)
    res["p_fdr_rois"] = res.groupby(keys[:-1] + ["contrast"])["p"].transform(fdr_bh)
    return res


def session_lme(df, value):
    """Mixed model per ROI / pair type / predictor: value ~ session + (1 | participant).

    Session is evening (reference), morning pass 1, morning pass 2. Reported are
      - overall: the effect averaged over the three sessions (intercept of the same
        model with sum-coded session);
      - session: the two session coefficients against evening and the joint Wald
        test of both (does the effect depend on the time point).
    """
    import statsmodels.formula.api as smf
    keys = ["pairtype", "window", "variant", "model", "roi"]
    d = df[df.pair_set.isin(["pre", "post1", "post2"])].dropna(subset=[value])
    rows = []
    for k, g in d.groupby(keys):
        if g.subject.nunique() < 10 or g.pair_set.nunique() < 3:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(f"{value} ~ C(pair_set, Treatment('pre'))", g,
                                  groups=g["subject"]).fit(reml=True, method="powell")
            names = [n for n in fit.params.index if n.startswith("C(")]
            wald = fit.wald_test(np.eye(len(fit.params))[[list(fit.params.index).index(n) for n in names]],
                                 scalar=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg = smf.mixedlm(f"{value} ~ C(pair_set, Sum)", g,
                                  groups=g["subject"]).fit(reml=True, method="powell")
            row = dict(zip(keys, k), n_subjects=g.subject.nunique(),
                       b_overall=avg.params["Intercept"], z_overall=avg.tvalues["Intercept"],
                       p_overall=avg.pvalues["Intercept"], p_session=float(wald.pvalue))
            for n in names:
                lab = n.split("[T.")[1].rstrip("]")
                row.update({f"b_{lab}": fit.params[n], f"z_{lab}": fit.tvalues[n], f"p_{lab}": fit.pvalues[n]})
            rows.append(row)
        except Exception as err:
            print("LME failed for", k, err)
    res = pd.DataFrame(rows)
    if len(res):
        res["p_session_fdr_rois"] = res.groupby(keys[:-1])["p_session"].transform(fdr_bh)
        res["p_overall_fdr_rois"] = res.groupby(keys[:-1])["p_overall"].transform(fdr_bh)
    return res


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_dir = str(step_output_dir(__file__))
    subs = subjects()
    print(f"{len(subs)} participants, {len(ROIS)} ROIs, {n_jobs} jobs")
    jobs = [(s, r) for s in subs for r in ROIS]
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(analyse_subject_roi)(s, r) for s, r in jobs)

    betas = pd.concat([r[0] for r in results], ignore_index=True)
    betas.to_csv(opj(out_dir, "subject_betas.csv"), index=False)

    group_rdms = {}
    for roi in ROIS:
        for set_name in {k for (_, r), res in zip(jobs, results) if r == roi for k in res[1]}:
            stack = [res[1][set_name] for (_, r), res in zip(jobs, results)
                     if r == roi and set_name in res[1]]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                group_rdms[f"{roi}__{set_name}"] = np.nanmean(np.stack(stack), axis=0)
    np.savez_compressed(opj(out_dir, "group_mean_rdms.npz"), **group_rdms)

    for value in ("beta", "alone"):
        one_sample(betas, value).to_csv(opj(out_dir, f"group_{value}_vs_zero.csv"), index=False)
        session_contrasts(betas, value).to_csv(opj(out_dir, f"group_{value}_session_contrasts.csv"), index=False)
    for value in ("beta", "alone"):
        session_lme(betas, value).to_csv(opj(out_dir, f"group_{value}_session_lme.csv"), index=False)
    print("participants per pair set:", betas.groupby("pair_set").subject.nunique().to_dict())
    print("written to", out_dir)


if __name__ == "__main__":
    main()
