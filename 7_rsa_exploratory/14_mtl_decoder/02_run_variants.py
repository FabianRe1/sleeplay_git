#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 - Leave-one-run-out cross-validation of decoder variants on the cached MTL data.

Localizer only. Same cross-validation as B4_train_classifiers.py (one of the 26
localizer runs left out per fold; SMOTE, scaling, feature selection and the
classifier are fitted on the training runs only). The classifier pipeline of the
current setting is B4.build_clf itself.

A variant is a dict with the keys
    roi     mtl | hpc | phc | erh          anatomical voxel set
    rel     10..90 | None                  intersect with the MTL reliability mask (None = anatomical only)
    fs      percent of voxels kept by F test inside each fold (100 = none)
    train   both | shown | recalled        trial types used for training
    smooth  0 | 2 | 4                      mm FWHM
    win     tuple of TR offsets around (onset + 5 s) that are averaged
    smote   True | False
    clf     lr | corr                      logistic regression (B4) | correlation with class centroids
    C       inverse L2 strength of the logistic regression
Test accuracy (mean of the per-category accuracies, percent) is always reported
for all, shown-only and recalled-only test trials.

Usage:  python 02_run_variants.py sub-13 stage1
Output: outputs/14_mtl_decoder/results/<stage>/sXX_<stage>.csv
        outputs/14_mtl_decoder/results/<stage>/sXX_<stage>_predictions_current.csv
"""

import os
import sys
import time
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import mtl_config as cfg

cfg.add_pipeline_to_path()
from decoding_utils import parse_subject_arg  # noqa: E402
import B4_train_classifiers as b4  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

CURRENT = dict(roi="mtl", rel=70, fs=30, train="both", smooth=4, win=(-1, 0, 1, 2),
               smote=True, clf="lr", C=1.0)
PHASE = {"shown": "encoding", "recalled": "retrieval"}


# ----------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------

def _v(name, base, **changes):
    d = dict(base)
    d.update(changes)
    d["name"] = name
    return d


def one_factor_variants(prefix, base):
    """One factor changed at a time around `base`."""
    out = [_v(f"{prefix}current", base)]
    for thr in (10, 30, 50, 70, 90):
        if thr != base["rel"]:
            out.append(_v(f"{prefix}rel{thr}", base, rel=thr))
    out.append(_v(f"{prefix}anat", base, rel=None))
    for fs in (100, 10, 50):
        out.append(_v(f"{prefix}fs{fs}", base, fs=fs))
    for tr in ("shown", "recalled"):
        out.append(_v(f"{prefix}train_{tr}", base, train=tr))
    for s in (0, 2):
        out.append(_v(f"{prefix}smooth{s}", base, smooth=s))
    for label, win in (("peak1", (0,)), ("3tr", (-1, 0, 1)), ("4tr_late1", (0, 1, 2, 3)),
                       ("4tr_late2", (1, 2, 3, 4)), ("4tr_late3", (2, 3, 4, 5)),
                       ("6tr", (-1, 0, 1, 2, 3, 4))):
        out.append(_v(f"{prefix}win_{label}", base, win=win))
    out.append(_v(f"{prefix}nosmote", base, smote=False))
    for C in (0.1, 0.01, 0.001):
        out.append(_v(f"{prefix}lrC{C}", base, C=C))
    out.append(_v(f"{prefix}corr", base, clf="corr"))
    return out


def stage_variants(stage):
    if stage == "validate":
        return [_v("mtl_current", CURRENT)]
    if stage == "stage1":
        v = one_factor_variants("mtl_", CURRENT)
        v += one_factor_variants("hpc_", dict(CURRENT, roi="hpc"))
        # other anatomical subregions, for orientation
        v += [_v("phc_anat", CURRENT, roi="phc", rel=None), _v("erh_anat", CURRENT, roi="erh", rel=None),
              _v("phc_rel70", CURRENT, roi="phc"), _v("erh_rel70", CURRENT, roi="erh")]
        return v
    if stage == "stage2":
        import stage2_variants
        return stage2_variants.build(CURRENT, _v)
    raise ValueError(stage)


# ----------------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------------

class CorrCentroid(ClassifierMixin, BaseEstimator):
    """Assign the class whose mean training pattern correlates best with the test pattern."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[np.asarray(y) == c].mean(0) for c in self.classes_])
        return self

    @staticmethod
    def _unit(a):
        a = a - a.mean(1, keepdims=True)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    def decision_function(self, X):
        return self._unit(X) @ self._unit(self.centroids_).T

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def build_pipeline(v):
    if v["clf"] == "lr":
        clf = b4.build_clf(v["fs"])                     # scaler, F-test selection, logistic regression
        clf.set_params(logisticregression__C=v["C"])
        return clf
    if v["clf"] == "corr":
        return make_pipeline(StandardScaler(), SelectPercentile(f_classif, percentile=v["fs"]),
                             CorrCentroid())
    raise ValueError(v["clf"])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class Cache:
    def __init__(self, sub):
        self.sub = sub
        self.meta = dict(np.load(opj(cfg.CACHE_DIR, f"{sub}_meta.npz")))
        tr = pd.read_csv(opj(cfg.CACHE_DIR, f"{sub}_trials.csv"))
        self.trials = tr[tr["valid"] & tr["kind"].isin(["scene", "object"])].reset_index(drop=True)
        self._ts, self._samples = {}, {}

    def samples(self, smooth, win):
        """Trial x voxel matrix: mean over the TR offsets (B4.apply_temporal_smoothing)."""
        key = (smooth, tuple(win))
        if key not in self._samples:
            if smooth not in self._ts:
                self._ts[smooth] = np.load(opj(cfg.CACHE_DIR, f"{self.sub}_ts_smooth{smooth}.npy"))
            ts = self._ts[smooth]
            start = self.meta["run_start"][self.trials["run_pos"].values]
            n = self.trials["n_trs_run"].values
            tr = self.trials["tr"].values
            acc = np.zeros((len(tr), ts.shape[1]), np.float64)
            cnt = np.zeros((len(tr), 1))
            for off in win:
                t = tr + off
                ok = (t >= 0) & (t < n)
                acc[ok] += ts[start[ok] + t[ok]]
                cnt[ok] += 1
            # A late window can lie completely after the end of the run (1-2 trials per
            # participant, latest window only): use the last TR of the run for these.
            empty = cnt[:, 0] == 0
            if empty.any():
                acc[empty] = ts[start[empty] + n[empty] - 1]
                cnt[empty] = 1
            self._samples[key] = acc / cnt
        return self._samples[key]

    def voxels(self, roi, rel):
        m = self.meta[f"anat_{roi}"].copy()
        if rel is not None:
            m &= self.meta[f"rel{rel}"]
        return m


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------

def run_cv(X, labels, runs, phase, v):
    """Leave-one-run-out. Returns predictions and the probability of the true class."""
    pred = np.empty(len(labels), dtype=object)
    p_true = np.full(len(labels), np.nan)
    train_ok = np.ones(len(labels), bool) if v["train"] == "both" else (phase == PHASE[v["train"]])
    for r in np.unique(runs):
        test = runs == r
        train = (~test) & train_ok
        Xtr, ytr = X[train], labels[train]
        if v["smote"]:
            Xtr, ytr = SMOTE(random_state=b4.SMOTE_SEED).fit_resample(Xtr, ytr)
        clf = build_pipeline(v)
        clf.fit(Xtr, ytr)
        pred[test] = clf.predict(X[test])
        if hasattr(clf, "predict_proba"):
            pr = clf.predict_proba(X[test])
            cls = list(clf.classes_)
            p_true[test] = [pr[i, cls.index(l)] for i, l in enumerate(labels[test])]
    return pred, p_true


def balanced_acc(labels, pred):
    """Mean of the per-category accuracies in percent (as in B4.save_cv_results)."""
    s = pd.Series(labels == pred).groupby(labels).mean()
    return float(s.mean() * 100)


def evaluate(sub, v, kind, X, trials):
    t0 = time.time()
    labels = trials["training_category"].values
    runs = trials["run"].values
    phase = trials["trial_phase"].values
    pred, p_true = run_cv(X, labels, runs, phase, v)
    rows = []
    for test_name, sel in (("all", np.ones(len(labels), bool)),
                           ("shown", phase == "encoding"), ("recalled", phase == "retrieval")):
        rows.append(dict(sub=sub, variant=v["name"], kind=kind, test=test_name,
                         acc=balanced_acc(labels[sel], pred[sel]),
                         p_true=float(np.nanmean(p_true[sel])) if np.isfinite(p_true[sel]).any() else np.nan,
                         n_test=int(sel.sum()), n_vox=X.shape[1], n_runs=len(np.unique(runs)),
                         **{k: (str(v[k]) if k == "win" else v[k]) for k in
                            ("roi", "rel", "fs", "train", "smooth", "win", "smote", "clf", "C")},
                         seconds=round(time.time() - t0, 1)))
    preds = None
    if v["name"].endswith("mtl_current"):
        preds = trials[["loc", "run", "run_in_ses", "trl_nr", "trial_phase", "training_category"]].copy()
        preds["kind"] = kind
        preds["predicted_labels"] = pred
    return rows, preds


def main():
    sub = parse_subject_arg(sys.argv[1])
    stage = sys.argv[2]
    n_jobs = int(os.environ.get("N_JOBS", 8))
    out_dir = opj(cfg.RESULT_DIR, stage)
    os.makedirs(out_dir, exist_ok=True)

    cache = Cache(sub)
    variants = stage_variants(stage)
    names = [v["name"] for v in variants]
    assert len(names) == len(set(names)), "duplicate variant names"
    print(f"{sub}: {len(variants)} variants, {len(cache.trials)} trials, "
          f"{cache.trials['run'].nunique()} runs", flush=True)

    tasks = []
    for v in variants:
        S = cache.samples(v["smooth"], v["win"])
        vox = cache.voxels(v["roi"], v["rel"])
        for kind in ("object", "scene"):
            sel = (cache.trials["kind"] == kind).values
            tasks.append(delayed(evaluate)(sub, v, kind, S[sel][:, vox],
                                           cache.trials[sel].reset_index(drop=True)))
    res = Parallel(n_jobs=n_jobs, verbose=5)(tasks)

    rows = [r for rr, _ in res for r in rr]
    pd.DataFrame(rows).to_csv(opj(out_dir, f"{sub}_{stage}.csv"), index=False)
    preds = [p for _, p in res if p is not None]
    if preds:
        pd.concat(preds).to_csv(opj(out_dir, f"{sub}_{stage}_predictions_current.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
