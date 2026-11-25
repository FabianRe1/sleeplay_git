# -*- coding: utf-8 -*-
"""Shape of the object-position code: free 5 x 5 position structure and model comparison.

Question: objects at the same or a neighbouring position in DIFFERENT scenes have more
similar patterns (11_hsm_rsa.py). Is that an evenly spaced line (as a number code for the
digit on screen would give), or does it have another shape (e.g. start / end anchors)?

Per subject, ROI and retrieval set
    1. correlation distance between object trials as in 02 (centred, between-run pairs),
       z-scored over all object x object between-run pairs;
    2. keep pairs from different scenes AND different categories. Same category implies
       same position, so the diagonal (same position) comes only from the other category
       sequence: position without category, and never the same scene image;
    3. remove nuisance by regression (variants below) and average the residual in the
       15 cells {position i, position j}.
Variants (nuisance regressors)
    base   context, category sequence
    acc    base + whether the two trials differ in correctness
    rt     base + |RT difference|, only pairs in which both trials have an RT
           (difficulty changes with position; these two ask whether that explains the shape)
"pooled" = mean of the subject's evening, morning 1 and morning 2 cell values.

Group level
    cells            mean, SEM, t of every cell
    model fit        per subject, Pearson r (Fisher z) between the cells and a model distance:
                     linear |i-j|; log |ln i - ln j| (compressive number line);
                     startend, Euclidean distance between Henson's (1998) start and end
                     marker strengths (start 0.80^(i-1), end 0.60 * 0.48^(5-i));
                     edge, 1 if exactly one of the two positions is first or last.
                     Once on the 10 off-diagonal cells (different positions only: is there
                     structure beyond same vs different?) and once on all 15.
                     t against zero, and paired t of every model against linear.
    joint            off-diagonal cells ~ linear + edge (z-scored), betas against zero
    contrasts        ends:      d(1,5) - mean of the cells two steps apart; a line predicts > 0
                     neighbour: mean of cells >= 2 steps apart - mean of adjacent cells
                     same:      mean of adjacent cells - mean of the diagonal

Outputs (outputs/12_hsm_position_shape/<source>/): subject_cells.csv, group_cells.csv,
group_model_fit.csv, group_joint.csv, group_contrasts.csv, position_shape.png

Usage:  python 12_hsm_position_shape.py <source> [n_jobs]
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
rsa02.SOURCE = SOURCE
rsa02.IN_DIR = opj(cfg.OUT_ROOT, "01_trial_patterns" if SOURCE == "raw" else f"patterns_{SOURCE}")
OUT_DIR = opj(cfg.OUT_ROOT, "12_hsm_position_shape", SOURCE)
WINDOW = rsa02.MAIN_WINDOW if SOURCE == "raw" else (0, 0)
SETS = ["pre", "post1", "post2"]
CELLS = [(i, j) for i in range(1, 6) for j in range(i, 6)]
OFF = [c for c in CELLS if c[0] != c[1]]
KEY_ROIS = ["vis", "parietal_precuneus", "superior_frontal", "mtl"]


def model_distances() -> dict:
    pos = np.arange(1, 6)
    start, end = 0.80 ** (pos - 1), 0.60 * 0.48 ** (5 - pos)
    edge = np.isin(pos, [1, 5]).astype(float)
    full = {
        "linear": np.abs(pos[:, None] - pos[None, :]).astype(float),
        "log": np.abs(np.log(pos)[:, None] - np.log(pos)[None, :]),
        "startend": np.sqrt((start[:, None] - start[None, :]) ** 2 + (end[:, None] - end[None, :]) ** 2),
        "edge": np.abs(edge[:, None] - edge[None, :]),
    }
    return {k: {c: v[c[0] - 1, c[1] - 1] for c in CELLS} for k, v in full.items()}


MODELS = model_distances()


def analyse_subject_roi(sub: str, roi: str) -> pd.DataFrame:
    sets = rsa02.load_sets(sub, roi)
    rows = []
    for name in SETS:
        if name not in sets:
            continue
        trials, pat = sets[name]
        pairs = hsm.hsm_pair_table(trials, trials, True)
        x = rsa02.trial_vectors(trials, pat, WINDOW, True)
        y_all = rsa02.corr_distance(x, x)[pairs.ia.values, pairs.ib.values]
        oo = ((pairs.pairtype == "oo") & (pairs.same_run == 0)).values & np.isfinite(y_all)
        if oo.sum() < 50:
            continue
        z = (y_all - y_all[oo].mean()) / y_all[oo].std()
        keep = oo & (pairs.scene == 1).values & (pairs.category == 1).values
        La, Lb = trials.position.values[pairs.ia.values], trials.position.values[pairs.ib.values]
        lo, hi = np.minimum(La, Lb), np.maximum(La, Lb)
        acc = (trials.correct.values[pairs.ia.values] != trials.correct.values[pairs.ib.values]).astype(float)
        rt = np.abs(trials.rt.values[pairs.ia.values] - trials.rt.values[pairs.ib.values])
        base = [pairs.context.values, pairs.catseq.values]
        for variant, extra, sel in (("base", [], keep), ("acc", [acc], keep),
                                    ("rt", [rt], keep & np.isfinite(rt))):
            if sel.sum() < 50:
                continue
            A = np.column_stack([np.ones(sel.sum())] + [c[sel] for c in base + extra])
            resid = z[sel] - A @ np.linalg.lstsq(A, z[sel], rcond=None)[0]
            for i, j in CELLS:
                m = (lo[sel] == i) & (hi[sel] == j)
                if m.any():
                    rows.append(dict(subject=sub, roi=roi, pair_set=name, variant=variant,
                                     i=i, j=j, value=resid[m].mean(), n_pairs=int(m.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def _t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return dict(n=len(x), mean=np.nan, sem=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(x, 0)
    return dict(n=len(x), mean=x.mean(), sem=stats.sem(x), t=t, p=p)


def group_level(cells: pd.DataFrame):
    pooled = (cells.groupby(["subject", "roi", "variant", "i", "j"], as_index=False)
              .agg(value=("value", "mean"), n_pairs=("n_pairs", "sum"), n_sets=("pair_set", "nunique")))
    pooled = pooled[pooled.n_sets == pooled.groupby("subject").n_sets.transform("max")]
    pooled["pair_set"] = "pooled"
    allc = pd.concat([cells, pooled.drop(columns="n_sets")], ignore_index=True)
    keys = ["roi", "pair_set", "variant"]

    g_cells = pd.DataFrame([dict(zip(keys + ["i", "j"], k), **_t(g.value), n_pairs=g.n_pairs.mean())
                            for k, g in allc.groupby(keys + ["i", "j"])])

    fit_rows, joint_rows, con_rows = [], [], []
    for k, g in allc.groupby(keys):
        wide = g.pivot_table(index="subject", columns=["i", "j"], values="value").dropna()
        if len(wide) < 5:
            continue
        zs = {}
        for cellset, cl in (("offdiag", OFF), ("all", CELLS)):
            v = wide[cl].values
            for mname, md in MODELS.items():
                d = np.array([md[c] for c in cl])
                r = np.array([np.corrcoef(row, d)[0, 1] for row in v])
                zs[(cellset, mname)] = np.arctanh(np.clip(r, -0.999, 0.999))
                fit_rows.append(dict(zip(keys, k), cells=cellset, model=mname, mean_r=np.nanmean(r),
                                     **{a: b for a, b in _t(zs[(cellset, mname)]).items() if a != "mean"}))
            for mname in MODELS:
                if mname != "linear":
                    res = _t(zs[(cellset, mname)] - zs[(cellset, "linear")])
                    fit_rows.append(dict(zip(keys, k), cells=cellset, model=f"{mname} - linear",
                                         mean_r=np.nan, **{a: b for a, b in res.items() if a != "mean"},
                                         mean_dz=res["mean"]))
        X = np.column_stack([[MODELS[m][c] for c in OFF] for m in ("linear", "edge")])
        X = (X - X.mean(0)) / X.std(0)
        A = np.column_stack([np.ones(len(OFF)), X])
        v = wide[OFF].values
        vz = (v - v.mean(1, keepdims=True)) / v.std(1, keepdims=True)
        b = np.linalg.lstsq(A, vz.T, rcond=None)[0][1:]
        for name, bb in zip(("linear", "edge"), b):
            joint_rows.append(dict(zip(keys, k), predictor=name, **_t(bb)))
        adjacent = wide[[c for c in OFF if c[1] - c[0] == 1]].mean(axis=1)
        far = wide[[c for c in OFF if c[1] - c[0] >= 2]].mean(axis=1)
        two = wide[[c for c in OFF if c[1] - c[0] == 2]].mean(axis=1)
        diag = wide[[c for c in CELLS if c[0] == c[1]]].mean(axis=1)
        for name, x in (("ends: d(1,5) - two steps apart", wide[(1, 5)] - two),
                        ("neighbour: far - adjacent", far - adjacent),
                        ("same: adjacent - same position", adjacent - diag)):
            con_rows.append(dict(zip(keys, k), contrast=name, **_t(x)))
    return g_cells, pd.DataFrame(fit_rows), pd.DataFrame(joint_rows), pd.DataFrame(con_rows)


def figure(g_cells: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, div = hsm._colormaps()
    ink = hsm.INK
    plt.rcParams.update({"font.size": 7, "figure.dpi": 150})
    cols = ["pre", "post1", "post2", "pooled"]
    labels = dict(hsm.SET_LABEL, pooled="three sets pooled")
    fig, axes = plt.subplots(len(KEY_ROIS) + 1, 4, figsize=(11, 2.6 * (len(KEY_ROIS) + 1)), squeeze=False)

    def show(ax, m, title, lim, fmt):
        ax.imshow(np.ma.masked_invalid(m), cmap=div, vmin=-lim, vmax=lim)
        for i in range(5):
            for j in range(5):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, fmt(m[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if abs(m[i, j]) > 0.7 * lim else ink)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(range(1, 6)); ax.set_yticklabels(range(1, 6))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8, color=ink, loc="left")

    for ax, (mname, md) in zip(axes[0], MODELS.items()):
        m = np.full((5, 5), np.nan)
        for (i, j), v in md.items():
            m[i - 1, j - 1] = m[j - 1, i - 1] = v
        m = m - np.nanmean(m)
        show(ax, m, f"model: {mname} (centred)", np.nanmax(np.abs(m)), lambda v: f"{v:.1f}")
    for r, roi in enumerate(KEY_ROIS, start=1):
        for ax, pset in zip(axes[r], cols):
            g = g_cells[(g_cells.roi == roi) & (g_cells.pair_set == pset) & (g_cells.variant == "base")]
            m = np.full((5, 5), np.nan)
            for _, row in g.iterrows():
                m[int(row.i) - 1, int(row.j) - 1] = m[int(row.j) - 1, int(row.i) - 1] = row["mean"]
            n = int(g.n.max()) if len(g) else 0
            show(ax, m, f"{roi}, {labels[pset]} (n = {n})", 0.12, lambda v: f"{v:+.2f}".replace("0.", "."))
    text = "TRs 4-6 after the cue" if SOURCE == "raw" else f"{SOURCE} betas"
    fig.suptitle(f"Object position x object position ({text}): group-mean dissimilarity (z, nuisance removed) between "
                 "object trials\nfrom different scenes and different categories. Blue = more similar than average, "
                 "red = less similar. Top row: what each model predicts.",
                 fontsize=8.5, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(opj(OUT_DIR, "position_shape.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(opj(rsa02.IN_DIR, "*sleep", "*_trials.csv"))})
    print(f"{SOURCE}: {len(subs)} subjects, {len(cfg.ROIS)} ROIs, {N_JOBS} jobs", flush=True)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(analyse_subject_roi)(s, r) for s in subs for r in cfg.ROIS)
    cells = pd.concat(res, ignore_index=True)
    cells.to_csv(opj(OUT_DIR, "subject_cells.csv"), index=False)
    g_cells, fit, joint, con = group_level(cells)
    g_cells.to_csv(opj(OUT_DIR, "group_cells.csv"), index=False)
    fit.to_csv(opj(OUT_DIR, "group_model_fit.csv"), index=False)
    joint.to_csv(opj(OUT_DIR, "group_joint.csv"), index=False)
    con.to_csv(opj(OUT_DIR, "group_contrasts.csv"), index=False)
    figure(g_cells)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
