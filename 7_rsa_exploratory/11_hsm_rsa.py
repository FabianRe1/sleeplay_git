# -*- coding: utf-8 -*-
"""Hierarchical sequence model regressed onto the single-trial neural RDMs (retrieval).

Same empirical RDM and the same regression as 02_binary_rsa.py, variant "main":
correlation distance between trial patterns (mean-centred per run and trial type),
between-run pairs only, z-scored, OLS per subject / ROI / pair set / pair type.
The predictors come from 10_hsm_model_rdms.py. All predictors are distances, so a
positive beta means: the closer two trials are on that predictor, the more similar
their patterns, over and above the other predictors of that model set.

Model sets (one regression each; "binary" reproduces 02)
    binary          the binary predictors of 02
    tree            tree distance instead of context / scene / object, plus position distances
    split           binary predictors plus scene / object position distance (digit distance for os)
    parts           as split, with the position distances split into same / other context
                    and same / other scene   <- main model
    split_chain50, parts_chain50   plus the subject's encoding order (subjects with an encoding table)
Scene position WITHIN a context is nearly the same predictor as encoding order within a
context (r = .94-.97), so "scene_pos_same_ctx" stands for both.

Outputs (outputs/11_hsm_rsa/<source>/)
    subject_betas.csv               beta per subject, ROI, pair set, pair type, model set, predictor
    subject_fit.csv                 R2 and adjusted R2 of every model set
    group_beta_vs_zero.csv          one-sample t-tests (FDR across ROIs)
    group_beta_session_contrasts.csv, group_beta_session_lme.csv
    group_fit_vs_binary.csv         adjusted R2 of each model set minus binary, paired t-test
    group_position_maps.npz         group-mean 25 x 25 (scene position x object position) maps
    check_against_02.txt            largest difference between "binary" here and 02
    tmap_<pairtype>_<model set>.png, position_map.png

Usage:  python 11_hsm_rsa.py <source> [n_jobs]      source as in 02 (raw, lss, glmsingle, ...)
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "raw"
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def _import_hsm():
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        spec = importlib.util.spec_from_file_location("hsm_model_rdms_10", opj(cfg.HERE, "10_hsm_model_rdms.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


hsm = _import_hsm()
rsa02 = hsm.rsa02
rsa02.SOURCE = SOURCE                                  # load_sets() of 02 reads these module globals
rsa02.IN_DIR = opj(cfg.OUT_ROOT, "01_trial_patterns" if SOURCE == "raw" else f"patterns_{SOURCE}")
OUT_DIR = opj(cfg.OUT_ROOT, "11_hsm_rsa", SOURCE)
WINDOW = rsa02.MAIN_WINDOW if SOURCE == "raw" else (0, 0)

MODEL_SETS = {}
for _pt, _sets in hsm.MODEL_SETS.items():
    MODEL_SETS[_pt] = {
        "binary": list(rsa02.PREDICTORS[_pt]),
        "tree": _sets["tree"],
        "split": _sets["split"],
        "parts": _sets["split, parts"],
        "split_chain50": _sets["split+chain50"],
        "parts_chain50": _sets["split, parts"] + ["chain50"],
    }
KEY_ROIS = ["vis", "mtl", "hpc", "parietal_precuneus"]
SETS = ["pre", "post1", "post2", "post1xpost2", "prexpost1", "prexpost2"]


# ---------------------------------------------------------------------------

def fit(y, X):
    """Betas as in regress() of 02, plus R2 and adjusted R2 of the same regression."""
    beta, _, n = rsa02.regress(y, X)
    r2 = adj = np.nan
    ok = np.isfinite(y)
    use = np.isfinite(beta)
    if use.any():
        yz = (y[ok] - y[ok].mean()) / y[ok].std()
        A = np.column_stack([np.ones(ok.sum()), X[ok][:, use]])
        resid = yz - A @ np.linalg.lstsq(A, yz, rcond=None)[0]
        r2 = 1 - (resid ** 2).sum() / (yz ** 2).sum()
        k = int(use.sum())
        adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return beta, r2, adj, n


def analyse_subject_roi(sub: str, roi: str):
    sets = {k: v for k, v in rsa02.load_sets(sub, roi).items() if k != "enc"}
    ctx1 = hsm.first_context(sub)
    beta_rows, fit_rows, maps = [], [], {}
    for sa, sb in hsm.RETRIEVAL_PAIR_SETS:
        if sa not in sets or sb not in sets:
            continue
        (ta, pa), (tb, pb) = sets[sa], sets[sb]
        same_set = sa == sb
        set_name = sa if same_set else f"{sa}x{sb}"
        pairs = hsm.hsm_pair_table(ta, tb, same_set, ctx1)
        d = rsa02.corr_distance(rsa02.trial_vectors(ta, pa, WINDOW, True),
                                rsa02.trial_vectors(tb, pb, WINDOW, True))
        y_all = d[pairs.ia.values, pairs.ib.values]
        between = (pairs.same_run == 0).values
        for ptype, model_sets in MODEL_SETS.items():
            sel = (pairs.pairtype == ptype).values & between
            y = y_all[sel]
            if ptype == "oo" and np.isfinite(y).sum() > 10:
                yz = np.full(len(y_all), np.nan)
                yz[sel] = (y - np.nanmean(y)) / np.nanstd(y)
                maps[set_name] = hsm.position_map(yz, ta, tb, pairs)
            for mname, cols in model_sets.items():
                X = pairs.loc[sel, cols].values
                if not np.isfinite(X).all():            # e.g. chain50 without an encoding table
                    continue
                beta, r2, adj, n = fit(y, X)
                if not np.isfinite(beta).any():
                    continue
                fit_rows.append(dict(subject=sub, roi=roi, pair_set=set_name, pairtype=ptype,
                                     model_set=mname, r2=r2, adj_r2=adj, n_pairs=n,
                                     n_predictors=int(np.isfinite(beta).sum())))
                for k, name in enumerate(cols):
                    if np.isfinite(beta[k]):
                        beta_rows.append(dict(subject=sub, roi=roi, pair_set=set_name, pairtype=ptype,
                                              window="win", variant=mname, model=name,
                                              beta=beta[k], n_pairs=n))
    return pd.DataFrame(beta_rows), pd.DataFrame(fit_rows), maps


def fit_vs_binary(fits: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair_set", "pairtype", "roi"]
    wide = fits.pivot_table(index=keys + ["subject"], columns="model_set", values="adj_r2")
    rows = []
    for mname in [c for c in wide.columns if c != "binary"]:
        for k, g in wide[[mname, "binary"]].dropna().groupby(keys):
            if len(g) < 5:
                continue
            diff = (g[mname] - g["binary"]).values
            t, p = stats.ttest_1samp(diff, 0)
            rows.append(dict(zip(keys, k), model_set=mname, n=len(diff), adj_r2_binary=g["binary"].mean(),
                             adj_r2_model=g[mname].mean(), mean_diff=diff.mean(), t=t, p=p))
    res = pd.DataFrame(rows)
    res["p_fdr_rois"] = res.groupby(["pair_set", "pairtype", "model_set"])["p"].transform(rsa02.fdr_bh)
    return res


def check_against_02(betas: pd.DataFrame) -> str:
    f = opj(cfg.OUT_ROOT, "02_binary_rsa", SOURCE, "subject_betas.csv")
    if not os.path.exists(f):
        return "no 02 output for this source"
    old = pd.read_csv(f)
    old = old[(old.window == "win") & (old.variant == "main")]
    new = betas[betas.variant == "binary"]
    keys = ["subject", "roi", "pair_set", "pairtype", "model"]
    m = new.merge(old, on=keys, suffixes=("_new", "_02"))
    return (f"binary model set vs 02 ({SOURCE}): {len(m)} of {len(new)} betas matched, "
            f"largest absolute difference {np.abs(m.beta_new - m.beta_02).max():.2e}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(stat: pd.DataFrame, maps: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, div = hsm._colormaps()
    ink = hsm.INK
    plt.rcParams.update({"font.size": 7, "figure.dpi": 150})
    text = "TRs 4-6 after the cue" if SOURCE == "raw" else f"{SOURCE} betas"
    for ptype in MODEL_SETS:
        for mname in ("parts", "tree"):
            s = stat[(stat.pairtype == ptype) & (stat.variant == mname)]
            sets = [x for x in SETS if (s.pair_set == x).any()]
            models = [c for c in MODEL_SETS[ptype][mname] if (s.model == c).any()]
            if not sets:
                continue
            fig, axes = plt.subplots(1, len(sets), figsize=(0.55 * len(models) * len(sets) + 2.6,
                                                             0.3 * len(cfg.ROIS) + 2.6),
                                     sharey=True, squeeze=False)
            for ax, pset in zip(axes[0], sets):
                g = s[s.pair_set == pset]
                t = g.pivot(index="roi", columns="model", values="t").reindex(index=cfg.ROIS, columns=models)
                p = g.pivot(index="roi", columns="model", values="p").reindex(index=cfg.ROIS, columns=models)
                q = g.pivot(index="roi", columns="model", values="p_fdr_rois").reindex(index=cfg.ROIS, columns=models)
                ax.imshow(np.ma.masked_invalid(t.values), cmap=div, vmin=-5, vmax=5, aspect="auto")
                for i in range(t.shape[0]):
                    for j in range(t.shape[1]):
                        if np.isfinite(t.values[i, j]):
                            mark = "**" if q.values[i, j] < .05 else "*" if p.values[i, j] < .05 else ""
                            ax.text(j, i, f"{t.values[i, j]:.1f}{mark}", ha="center", va="center", fontsize=5.5,
                                    color="white" if abs(t.values[i, j]) > 3.6 else ink,
                                    fontweight="bold" if mark else "normal")
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([hsm.LABEL[c] for c in models], rotation=45, ha="right")
                ax.set_yticks(range(len(cfg.ROIS))); ax.set_yticklabels(cfg.ROIS)
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                n = int(g.n.max()) if len(g) else 0
                ax.set_title(f"{hsm.SET_LABEL[pset]} (n = {n})", fontsize=8, color=ink, loc="left")
            fig.suptitle(f"{hsm.PAIR_LABEL[ptype]}, model set '{mname}', {text}: t of the beta against zero. "
                         "Red = closer on the predictor, more similar.  * p < .05 uncorrected, ** FDR across ROIs",
                         fontsize=8.5, color=ink, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(opj(OUT_DIR, f"tmap_{ptype}_{mname}.png"), bbox_inches="tight")
            plt.close(fig)

    rois = [r for r in KEY_ROIS if f"{r}__pre" in maps]
    sets = ["pre", "post1", "post2", "post1xpost2"]
    if rois:
        fig, axes = plt.subplots(len(rois), len(sets), figsize=(3.1 * len(sets) + 0.8, 3.0 * len(rois) + 0.6),
                                 squeeze=False)
        for i, roi in enumerate(rois):
            for j, pset in enumerate(sets):
                ax = axes[i, j]
                m = maps.get(f"{roi}__{pset}")
                if m is None:
                    ax.axis("off"); continue
                lim = np.nanmax(np.abs(m - np.nanmean(m)))
                im = ax.imshow(np.ma.masked_invalid(m - np.nanmean(m)), cmap=div, vmin=-lim, vmax=lim)
                for e in np.arange(4.5, 24, 5):
                    ax.axhline(e, color="white", lw=1); ax.axvline(e, color="white", lw=1)
                ax.set_xticks(np.arange(2, 25, 5)); ax.set_yticks(np.arange(2, 25, 5))
                ax.set_xticklabels([f"sc {k}" for k in range(1, 6)]); ax.set_yticklabels([f"sc {k}" for k in range(1, 6)])
                ax.tick_params(length=0)
                ax.set_title(f"{roi}, {hsm.SET_LABEL[pset]}", fontsize=8, color=ink, loc="left")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
                cb.ax.tick_params(length=0, labelsize=6)
        fig.suptitle(f"Position map ({text}): group-mean z-scored dissimilarity between (scene position, object position) "
                     "cells, pairs from different scenes only.\nBlocks = scene position 1-5, within a block object "
                     "position 1-5. Blue = more similar than average, red = less similar.",
                     fontsize=8.5, color=ink, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(opj(OUT_DIR, "position_map.png"), bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(opj(rsa02.IN_DIR, "*sleep", "*_trials.csv"))})
    jobs = [(s, r) for s in subs for r in cfg.ROIS]
    print(f"{SOURCE}: {len(subs)} subjects, {len(cfg.ROIS)} ROIs, {N_JOBS} jobs", flush=True)
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(analyse_subject_roi)(s, r) for s, r in jobs)

    betas = pd.concat([r[0] for r in results], ignore_index=True)
    fits = pd.concat([r[1] for r in results], ignore_index=True)
    betas.to_csv(opj(OUT_DIR, "subject_betas.csv"), index=False)
    fits.to_csv(opj(OUT_DIR, "subject_fit.csv"), index=False)

    maps = {}
    for roi in cfg.ROIS:
        for set_name in {k for (_, r), res in zip(jobs, results) if r == roi for k in res[2]}:
            stack = [res[2][set_name] for (_, r), res in zip(jobs, results) if r == roi and set_name in res[2]]
            with np.errstate(all="ignore"):
                maps[f"{roi}__{set_name}"] = np.nanmean(np.stack(stack), axis=0)
    np.savez_compressed(opj(OUT_DIR, "group_position_maps.npz"), **maps)

    stat = rsa02.one_sample(betas, "beta")
    stat.to_csv(opj(OUT_DIR, "group_beta_vs_zero.csv"), index=False)
    rsa02.session_contrasts(betas, "beta").to_csv(opj(OUT_DIR, "group_beta_session_contrasts.csv"), index=False)
    fit_vs_binary(fits).to_csv(opj(OUT_DIR, "group_fit_vs_binary.csv"), index=False)
    msg = check_against_02(betas)
    open(opj(OUT_DIR, "check_against_02.txt"), "w").write(msg + "\n")
    print(msg, flush=True)
    figures(stat, maps)
    rsa02.session_lme(betas, "beta").to_csv(opj(OUT_DIR, "group_beta_session_lme.csv"), index=False)
    print("subjects per pair set:", betas.groupby("pair_set").subject.nunique().to_dict())
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
