# -*- coding: utf-8 -*-
"""Binary model RDMs regressed onto single-trial neural RDMs.

Empirical RDM
    One pattern per retrieval trial (mean over a TR window after the cue, or a
    single TR for the time-resolved version). Dissimilarity = 1 - Pearson r
    across voxels. Patterns are mean-centred per run and trial type first
    (variant "nocenter" switches this off).

Trial sets
    pre    evening retrieval (70 trials: 50 objects once, 10 scenes twice)
    post1  morning, first pass (runs 1-3)      post2  morning, second pass (runs 4-6)

Pair sets
    within a set (pre, post1, post2): only pairs from DIFFERENT runs, so that
        within-run temporal autocorrelation cannot produce similarity
        (variant "allpairs" keeps them and adds same-run and time-lag regressors);
    across sets (post1xpost2, prexpost1, prexpost2): every pair, always from
        different runs. These also allow a same-object predictor.

Regression, per subject / ROI / pair set / pair type
    z(dissimilarity) ~ 1 + binary predictors (0 = same, 1 = different).
    A positive beta means: trials that share the feature are more similar,
    over and above the other predictors in the model. Because the predictors
    are nested (same scene implies same context; same category implies same
    position and same category sequence), the beta of the broader predictor
    is its effect for pairs that do NOT share the narrower one.
    "alone" is the plain difference different-minus-same without the others.

Pair types and their predictors
    oo  object trial x object trial : context, scene, catseq, category, position (+ object across sets)
    ss  scene trial  x scene trial  : context, scene, position
    os  object trial x scene trial  : context, scene (object's scene = retrieved scene), position

Pattern sources (first argument)
    raw            mean cleaned BOLD in a TR window (01); also gives the per-TR version
    lss            least-squares-separate betas (04)
    glmsingle      GLMsingle betas, full model (04)
    glmsingle_hrf  GLMsingle betas with fitted HRF only (04)
    glmsingle_sep, glmsingle_sep_hrf   as above, evening and morning fitted separately
The beta sources add the encoding session ("enc": object presentations and
scene presentations), analysed on its own. Encoding and retrieval trials are
never paired with each other.

Usage:  python 02_binary_rsa.py <source> [n_jobs]
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "raw"
IN_DIR = opj(cfg.OUT_ROOT, "01_trial_patterns" if SOURCE == "raw" else f"patterns_{SOURCE}")
OUT_DIR = opj(cfg.OUT_ROOT, "02_binary_rsa", SOURCE)

MAIN_WINDOW = (4, 6)          # TRs after cue, inclusive: 6-9 s (10.5 s incl. TR length)
PAIR_SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
             ("post1", "post2"), ("pre", "post1"), ("pre", "post2"),
             ("enc", "enc")]          # encoding is analysed on its own, never paired with retrieval
PREDICTORS = {
    "oo": ["context", "scene", "catseq", "category", "position", "object"],
    "ss": ["context", "scene", "position"],
    "os": ["context", "scene", "position"],
}
FOUR = ["context", "scene", "position", "category"]
VARIANTS = {          # name: (centre, between-run only, rank-transform, predictors to keep or None = all)
    "main": (True, True, False, None),
    "four": (True, True, False, FOUR),     # joint model with only context, scene, position, category
    "nocenter": (False, True, False, None),
    "allpairs": (True, False, False, None),
    "rank": (True, True, True, None),
}
N_CANON = 70          # 50 object trials + 20 scene trials in canonical order


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_sets(sub: str, roi: str) -> dict:
    """Return {set name: (trials, patterns[trials x TRs x voxels])} for one subject and ROI."""
    sets = {}
    for session in ["encoding"] + cfg.SESSIONS:
        # the separately fitted retrieval sources share the encoding betas of the joint fit
        in_dir = IN_DIR.replace("_sep", "") if session == "encoding" else IN_DIR
        f_tr = opj(in_dir, session, f"{sub}_{session}_trials.csv")
        f_pat = opj(in_dir, session, f"{sub}_{session}_{roi}_patterns.npy")
        if not (os.path.exists(f_tr) and os.path.exists(f_pat)):
            continue
        trials = pd.read_csv(f_tr)
        pat = np.load(f_pat).astype(np.float32)
        if session == "encoding":
            # scene presentations of both kinds count as scene events; day/night cues are left out
            trials["kind"] = trials["kind"].replace("scenecue", "scene")
            sel = trials.kind.isin(["object", "scene"])
            sets["enc"] = (trials[sel].reset_index(drop=True), pat[sel.values])
        elif session == "presleep":
            sets["pre"] = (trials.reset_index(drop=True), pat)
        else:
            for name, sel in (("post1", trials.run_idx < 3), ("post2", trials.run_idx >= 3)):
                sets[name] = (trials[sel].reset_index(drop=True), pat[sel.values])
    return sets


def canonical_index(trials: pd.DataFrame) -> np.ndarray:
    """Position of every trial in the structure-ordered 70-trial layout.

    objects 0-49: context (day, night) > scene position > object position
    scenes 50-69: context > scene position > repetition
    Stimuli differ between subjects, the structure does not, so matrices in
    this order can be averaged across subjects.
    """
    ctx = (trials.context == "night").astype(int).values
    idx = np.full(len(trials), -1)
    obj = (trials.kind == "object").values
    idx[obj] = ctx[obj] * 25 + (trials.scene_position.values[obj] - 1) * 5 + (trials.position.values[obj] - 1)
    rep = trials.groupby(["kind", "scene"]).cumcount().values
    idx[~obj] = 50 + ctx[~obj] * 10 + (trials.position.values[~obj] - 1) * 2 + np.minimum(rep[~obj], 1)
    return idx.astype(int)


# ---------------------------------------------------------------------------
# RDM and regression
# ---------------------------------------------------------------------------

def trial_vectors(trials, pat, trs, centre):
    x = pat[:, trs[0]:trs[1] + 1].mean(axis=1)            # NaN if any TR of the window is missing
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
    if SOURCE == "raw":
        windows = [("win", MAIN_WINDOW, v) for v in VARIANTS] + \
                  [(f"tr{t}", (t, t), "main") for t in cfg.TR_WINDOW]
    else:                                   # betas: one pattern per event
        windows = [("win", (0, 0), v) for v in VARIANTS]
    for sa, sb in PAIR_SETS:
        if sa not in sets or sb not in sets:
            continue
        (ta, pa), (tb, pb) = sets[sa], sets[sb]
        same_set = sa == sb
        pairs = pair_table(ta, tb, same_set)
        set_name = sa if same_set else f"{sa}x{sb}"
        for wname, trs, variant in windows:
            centre, between_only, rank, keep = VARIANTS[variant]
            d = corr_distance(trial_vectors(ta, pa, trs, centre), trial_vectors(tb, pb, trs, centre))
            y_all = d[pairs.ia.values, pairs.ib.values]
            if wname == "win" and variant == "main" and max(len(ta), len(tb)) <= N_CANON:  # group RDM
                m = np.full((N_CANON, N_CANON), np.nan)
                ca, cb = canonical_index(ta), canonical_index(tb)
                sel = (pairs.same_run == 0).values
                yz = (y_all - np.nanmean(y_all[sel])) / np.nanstd(y_all[sel])
                m[ca[pairs.ia.values[sel]], cb[pairs.ib.values[sel]]] = yz[sel]
                if same_set:
                    m[cb[pairs.ib.values[sel]], ca[pairs.ia.values[sel]]] = yz[sel]
                mats[set_name] = m
            for ptype, preds in PREDICTORS.items():
                if keep is not None:
                    preds = [m for m in preds if m in keep]
                sel = (pairs.pairtype == ptype).values
                if between_only:
                    sel &= (pairs.same_run == 0).values
                    cols = preds
                else:
                    cols = preds + ["same_run", "lag"]
                y = y_all[sel]
                if rank:
                    y = np.where(np.isfinite(y), stats.rankdata(y, nan_policy="omit"), np.nan)
                X = pairs.loc[sel, cols].values
                beta, alone, n = regress(y, X)
                for k, name in enumerate(cols):
                    if np.isfinite(beta[k]):
                        rows.append(dict(
                            subject=sub, roi=roi, pair_set=set_name, pairtype=ptype,
                            window=wname, variant=variant, model=name,
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
    # FDR across ROIs within each pair set / pair type / window / variant / model
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
    """Mixed model per ROI / pair type / model: value ~ session + (1 | subject).

    Session is evening (reference), morning pass 1, morning pass 2. Reported are
      - overall: the effect averaged over the three sessions (intercept of the same
        model with sum-coded session) = "is there an effect across sessions";
      - session: the two session coefficients against evening and the joint Wald
        test of both = "does the effect depend on the time point" (interaction of
        the model predictor with session).
    """
    import statsmodels.formula.api as smf
    import warnings
    keys = ["pairtype", "window", "variant", "model", "roi"]
    d = df[df.pair_set.isin(["pre", "post1", "post2"]) & (df.window == "win")].dropna(subset=[value])
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
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(opj(IN_DIR, "*", "*_trials.csv"))})
    print(f"{len(subs)} subjects, {len(cfg.ROIS)} ROIs, {n_jobs} jobs")
    jobs = [(s, r) for s in subs for r in cfg.ROIS]
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(analyse_subject_roi)(s, r) for s, r in jobs)

    betas = pd.concat([r[0] for r in results], ignore_index=True)
    betas.to_csv(opj(OUT_DIR, "subject_betas.csv"), index=False)

    group_rdms = {}
    for roi in cfg.ROIS:
        for set_name in {k for (_, r), res in zip(jobs, results) if r == roi for k in res[1]}:
            stack = [res[1][set_name] for (_, r), res in zip(jobs, results)
                     if r == roi and set_name in res[1]]
            with np.errstate(all="ignore"):
                group_rdms[f"{roi}__{set_name}"] = np.nanmean(np.stack(stack), axis=0)
    np.savez_compressed(opj(OUT_DIR, "group_mean_rdms.npz"), **group_rdms)

    for value in ("beta", "alone"):
        one_sample(betas, value).to_csv(opj(OUT_DIR, f"group_{value}_vs_zero.csv"), index=False)
        session_contrasts(betas, value).to_csv(opj(OUT_DIR, f"group_{value}_session_contrasts.csv"), index=False)
    for value in ("beta", "alone"):
        session_lme(betas, value).to_csv(opj(OUT_DIR, f"group_{value}_session_lme.csv"), index=False)
    n = betas.groupby("pair_set").subject.nunique().to_dict()
    print("subjects per pair set:", n)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
