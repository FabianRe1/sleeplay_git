#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C12 - Group maps of the searchlight (B11).

For every sphere and participant the trial values of B11 are averaged per time
point (evening, first morning retrieval, repeated morning retrieval, all
trials) and tested across participants. Session comparisons are made on these
averages, never on single runs.

Measures (all retrieved minus control, see B11): item, predecessor, successor,
abs_slope, abs_slope_wo; and the accuracy of the classifier on the localizers
minus chance (10 %).

Statistics per map: one-sample t test across participants in every sphere
present in at least MIN_SUBJECTS participants; family-wise error over all
spheres by sign-flipping (N_PERM permutations, maximum |t|), two-sided.

Region check: the sphere values are averaged over the spheres whose centre
lies in a region (each participant's own FreeSurfer labels in MNI space) and
tested as in C11, to compare with the region analyses.

Input   outputs/B11_searchlight_retrieval/centres.npz, participants/*.npz, *_trials.csv
        <derivatives>/sub-XX/ses-04/func/*space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
        <derivatives>/desc-aparcaseg_dseg.tsv
Output  (all under outputs/C12_searchlight_group/<space>_r<radius>/, chosen with C12_TAG, default T1w_r20)
        outputs/C12_searchlight_group/maps/<measure>_<time point>_{t,mean,pfwe}.nii.gz  (4 mm grid, MNI)
        outputs/C12_searchlight_group/group_maps.csv       one line per map
        outputs/C12_searchlight_group/group_clusters.csv   clusters at p < .001 uncorrected
        outputs/C12_searchlight_group/region_check.csv
Plots   plots/C12_searchlight_group/<measure>.{png,pdf}    rows = time points
        plots/C12_searchlight_group/accuracy.{png,pdf}
        plots/C12_searchlight_group/change.{png,pdf}       first morning minus evening, all measures

Not a port of the original pipeline: added 2026-11-25. Exploratory.

Usage:  python C12_searchlight_group.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DERIVATIVES_DIR, step_input_dir, step_output_dir, step_plot_dir

MEASURES = {"item": "Retrieved item", "predecessor": "Item before", "successor": "Item after",
            "abs_slope": "Absolute slope, five items", "abs_slope_wo": "Absolute slope without the retrieved item"}
CONDS = {"pooled": "All trials", "pre": "Evening", "post1": "First morning retrieval", "post2": "Repeated morning retrieval",
         "post1 - pre": "First morning minus evening", "post2 - pre": "Repeated morning minus evening"}
FIGURE_ROWS = ["pooled", "pre", "post1", "post2", "post1 - pre"]
MIN_SUBJECTS = 30
N_PERM = 5000
P_UNC = 0.001
REGIONS = {"vis": (1005, 2005, 1011, 2011, 1021, 2021, 1029, 2029, 1013, 2013, 1008, 2008, 1007, 2007, 1009, 2009, 1015, 2015),
           "parietal_precuneus": (1008, 2008, 1025, 2025), "mtl": (17, 53, 1016, 2016, 1006, 2006), "hpc": (17, 53),
           "somato": (1022, 2022), "caudal_acc": (1002, 2002)}
INK = "#0b0b0b"
TAG = os.environ.get("C12_TAG", "T1w_r20")        # which searchlight: <space>_r<radius>, as in the B11 file names
ACCURACY = {"accuracy": "All trials", "accuracy_shown": "Image shown", "accuracy_recalled": "Image recalled"}


def load():
    src = str(step_input_dir("B11_searchlight_retrieval"))
    cen = np.load(opj(src, "centres.npz"))
    data, labels = {}, {}
    for f in sorted(glob.glob(opj(src, "participants", f"s*_searchlight_{TAG}.npz"))):
        sub = os.path.basename(f).split("_")[0]
        z = np.load(f)
        tr = pd.read_csv(opj(src, "participants", f"{sub}_trials.csv"))
        d = {}
        for m in MEASURES:
            x = z[m].astype(np.float32)
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                d[m] = {c: np.nanmean(x[:, (tr.cond == c).values], axis=1) if (tr.cond == c).any() else None
                        for c in ("pre", "post1", "post2")}
                d[m]["pooled"] = np.nanmean(x, axis=1)
        for a in ACCURACY:
            if a in z.files:
                d[a] = {"pooled": z[a] * 100.0 - 10.0}
        d["n_voxels"] = z["n_voxels"]
        data[sub] = d
        seg = sorted(glob.glob(opj(str(DERIVATIVES_DIR), "sub-" + sub[1:], "ses-0[246]", "func",
                                   "*space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz")))
        if seg:
            a = np.asarray(nib.load(seg[0]).dataobj)
            labels[sub] = a[tuple(cen["ijk"].T)].astype(int)
    return cen, data, labels


def matrix(data, measure, cond):
    """Participants x spheres for one map; a contrast needs both time points."""
    rows, subs = [], []
    for sub, d in data.items():
        if " - " in cond:
            a, b = (d[measure].get(c) for c in cond.split(" - "))
            v = None if a is None or b is None else a - b
        else:
            v = d[measure].get(cond)
        if v is not None:
            rows.append(v); subs.append(sub)
    return np.array(rows), subs


def tmap(X, rng):
    """t per sphere (NaN-aware) and family-wise p by sign-flipping, maximum |t|."""
    ok = np.isfinite(X)
    n = ok.sum(0)
    use = n >= MIN_SUBJECTS
    X0 = np.where(ok, X, 0.0)[:, use].astype(np.float64)
    nn = n[use].astype(float)
    ss = (X0 ** 2).sum(0)

    def t_of(S):
        var = (ss - S ** 2 / nn) / (nn - 1)
        return (S / nn) / np.sqrt(var / nn)

    t = t_of(X0.sum(0))
    mx = np.empty(N_PERM)
    for i in range(0, N_PERM, 250):
        signs = rng.choice([-1.0, 1.0], size=(min(250, N_PERM - i), X0.shape[0]))
        mx[i:i + len(signs)] = np.abs(t_of(signs @ X0)).max(1)
    pfwe = (1 + (mx[None, :] >= np.abs(t)[:, None]).sum(1)) / (N_PERM + 1)
    out = {k: np.full(X.shape[1], np.nan) for k in ("t", "mean", "pfwe", "p", "n")}
    out["t"][use], out["mean"][use], out["pfwe"][use] = t, X0.sum(0) / nn, pfwe
    out["p"][use] = 2 * stats.t.sf(np.abs(t), nn - 1)
    out["n"] = n.astype(float)
    return out


def to_img(values, cen):
    shape = tuple(int(np.ceil(s / 2)) for s in cen["shape"])
    a = np.zeros(shape, np.float32)
    a[tuple((cen["ijk"] // 2).T)] = np.nan_to_num(values)
    return nib.Nifti1Image(a, cen["affine"] @ np.diag([2, 2, 2, 1]))


def clusters(res, cen, names, mode_label, tag):
    """Connected spheres at p < P_UNC (same sign), on the 4 mm grid."""
    rows = []
    shape = tuple(int(np.ceil(s / 2)) for s in cen["shape"])
    idx = tuple((cen["ijk"] // 2).T)
    for sign in (1, -1):
        sig = np.zeros(shape, bool)
        sig[idx] = (res["p"] < P_UNC) & (np.sign(res["t"]) == sign)
        lab, k = ndimage.label(sig)
        cl = lab[idx]
        for c in range(1, k + 1):
            members = np.where(cl == c)[0]
            peak = members[np.argmax(np.abs(res["t"][members]))]
            xyz = nib.affines.apply_affine(cen["affine"], cen["ijk"][peak])
            rows.append(dict(**tag, direction="positive" if sign > 0 else "negative", spheres=len(members),
                             volume_cm3=len(members) * 0.064, peak_t=res["t"][peak], peak_p_fwe=res["pfwe"][peak],
                             spheres_fwe=int((res["pfwe"][members] < 0.05).sum()),
                             x=xyz[0], y=xyz[1], z=xyz[2], peak_region=names.get(mode_label[peak], str(mode_label[peak])),
                             regions=", ".join(f"{names.get(l, l)} ({n})" for l, n in
                                               pd.Series(mode_label[members]).value_counts().head(4).items())))
    return rows


def figure(maps, title, rows, plot_dir, name, cen, labels_of_rows, vmax=None):
    fig = plt.figure(figsize=(15, 2.05 * len(rows) + 0.85))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[7.6, 2.4], hspace=0.05, wspace=0.01, left=0.01, right=0.99,
                          top=1 - 0.75 / (2.05 * len(rows) + 0.6), bottom=0.01)
    df = MIN_SUBJECTS
    for i, key in enumerate(rows):
        res = maps[key]
        thr = stats.t.isf(P_UNC / 2, np.nanmedian(res["n"][np.isfinite(res["t"])]) - 1)
        img = to_img(res["t"], cen)
        fwe = to_img((res["pfwe"] < 0.05).astype(float), cen)
        vm = vmax or max(6.0, float(np.nanpercentile(np.abs(res["t"]), 99.9)))
        for j, (mode, cuts) in enumerate((("z", [-18, -6, 6, 18, 30, 42, 54, 66]), ("x", [-6, 40]))):
            ax = fig.add_subplot(gs[i, j])
            disp = plotting.plot_stat_map(img, display_mode=mode, cut_coords=cuts, threshold=thr, vmax=vm, axes=ax,
                                          colorbar=(j == 1), annotate=True, draw_cross=False, cmap="RdBu_r",
                                          black_bg=False, symmetric_cbar=True)
            if np.nansum(res["pfwe"] < 0.05) > 0:
                try:
                    disp.add_contours(fwe, levels=[0.5], colors=[INK], linewidths=0.9)
                except Exception:
                    pass
            if j == 0:
                n_fwe = int(np.nansum(res["pfwe"] < 0.05))
                ax.set_title(f"{labels_of_rows[key]}   (n = {int(np.nanmax(res['n']))}; peak t = {np.nanmax(res['t']):.1f}, "
                             f"lowest t = {np.nanmin(res['t']):.1f}; {n_fwe} spheres corrected)", loc="left", fontsize=9.5,
                             fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.12 / (2.05 * len(rows) + 0.6), f"{title} ({TAG.replace('_r', ', spheres of ')} mm radius). t across participants, "
             "shown at p < .001 uncorrected; black outline: p < .05 corrected over the whole brain.", fontsize=10.5, va="top")
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"{name}.{ext}"), **kw)
    plt.close(fig)


def main():
    out, plot_dir = str(step_output_dir(__file__, TAG)), str(step_plot_dir(__file__, TAG))
    os.makedirs(opj(out, "maps"), exist_ok=True)
    cen, data, labels = load()
    print(f"{len(data)} participants, {len(cen['ijk'])} spheres")
    lut = pd.read_csv(opj(str(DERIVATIVES_DIR), "desc-aparcaseg_dseg.tsv"), sep="\t")
    names = dict(zip(lut["index"], lut["name"].str.strip('"')))
    L = np.array(list(labels.values()))
    mode_label = stats.mode(L, axis=0, keepdims=False).mode

    rng = np.random.default_rng(1)
    summary, clus, results = [], [], {}
    acc_keys = [a for a in ACCURACY if all(a in d for d in data.values())]
    todo = [(m, c) for m in MEASURES for c in CONDS] + [(a, "pooled") for a in acc_keys]
    print(f"{TAG}: median voxels per sphere {np.median([np.median(d['n_voxels'][d['n_voxels'] > 0]) for d in data.values()]):.0f}")
    for measure, cond in todo:
        X, subs = matrix(data, measure, cond)
        res = tmap(X, rng)
        results[(measure, cond)] = res
        tag = dict(measure=measure, cond=cond)
        name = f"{measure}_{cond.replace(' - ', '_minus_')}"
        for k in ("t", "mean", "pfwe"):
            nib.save(to_img(res[k] if k != "pfwe" else np.where(np.isfinite(res[k]), 1 - res[k], 0), cen),
                     opj(out, "maps", f"{name}_{k if k != 'pfwe' else 'one_minus_pfwe'}.nii.gz"))
        hi, lo = np.nanargmax(res["t"]), np.nanargmin(res["t"])
        summary.append(dict(**tag, participants=len(subs), spheres=int(np.isfinite(res["t"]).sum()),
                            max_t=res["t"][hi], max_region=names.get(mode_label[hi], mode_label[hi]), p_fwe_max=res["pfwe"][hi],
                            min_t=res["t"][lo], min_region=names.get(mode_label[lo], mode_label[lo]), p_fwe_min=res["pfwe"][lo],
                            spheres_p001_pos=int(np.nansum((res["p"] < P_UNC) & (res["t"] > 0))),
                            spheres_p001_neg=int(np.nansum((res["p"] < P_UNC) & (res["t"] < 0))),
                            expected_by_chance=float(np.isfinite(res["t"]).sum() * P_UNC / 2),
                            spheres_fwe=int(np.nansum(res["pfwe"] < 0.05))))
        clus += clusters(res, cen, names, mode_label, tag)
        print(summary[-1])
    pd.DataFrame(summary).to_csv(opj(out, "group_maps.csv"), index=False)
    pd.DataFrame(clus).sort_values(["measure", "cond", "spheres"], ascending=[True, True, False]).to_csv(
        opj(out, "group_clusters.csv"), index=False)

    # --- region check -----------------------------------------------------------------------
    rows = []
    for measure in list(MEASURES) + acc_keys:
        for cond in (CONDS if measure in MEASURES else ["pooled"]):
            X, subs = matrix(data, measure, cond)
            for roi, ids in REGIONS.items():
                v = np.array([np.nanmean(x[np.isin(labels[s], ids)]) for x, s in zip(X, subs) if s in labels])
                v = v[np.isfinite(v)]
                r = stats.ttest_1samp(v, 0)
                rows.append(dict(tag=TAG, measure=measure, cond=cond, roi=roi, n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue,
                                 spheres=float(np.mean([np.isin(labels[s], ids).sum() for s in subs if s in labels]))))
    pd.DataFrame(rows).to_csv(opj(out, "region_check.csv"), index=False)

    # --- figures ------------------------------------------------------------------------------
    for measure, title in MEASURES.items():
        figure({c: results[(measure, c)] for c in FIGURE_ROWS}, f"{title}, retrieved minus control", FIGURE_ROWS,
               plot_dir, measure, cen, CONDS)
    figure({m: results[(m, "post1 - pre")] for m in MEASURES}, "Change from the evening to the first morning retrieval",
           list(MEASURES), plot_dir, "change", cen, MEASURES)
    figure({m: results[(m, "pooled")] for m in MEASURES}, "All trials", list(MEASURES), plot_dir, "all_trials", cen, MEASURES)
    figure({a: results[(a, "pooled")] for a in acc_keys}, "Classifier accuracy on the localizers against chance "
           "(trained on one localizer session, tested on the other)", acc_keys, plot_dir, "accuracy", cen, ACCURACY, vmax=12)
    # mean accuracy in percent, not thresholded by a test: how well can the objects be decoded at all
    fig = plt.figure(figsize=(15, 2.05 * len(acc_keys) + 0.6))
    for i, a in enumerate(acc_keys):
        ax = fig.add_axes([0.01, 1 - (i + 1) * 0.95 / len(acc_keys) - 0.02, 0.98, 0.9 / len(acc_keys)])
        plotting.plot_stat_map(to_img(results[(a, "pooled")]["mean"] + 10.0, cen), display_mode="z",
                               cut_coords=[-18, -6, 6, 18, 30, 42, 54, 66], threshold=10.5, vmax=22, axes=ax, cmap="inferno",
                               black_bg=False, symmetric_cbar=False, draw_cross=False, colorbar=True)
        ax.set_title(f"{ACCURACY[a]}: mean accuracy in percent (chance 10; shown above 10.5; best sphere "
                     f"{np.nanmax(results[(a, 'pooled')]['mean']) + 10:.1f})", loc="left", fontsize=9.5, fontweight="bold")
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(opj(plot_dir, f"accuracy_percent.{ext}"), **kw)
    plt.close(fig)
    print(f"-> {out}\n-> {plot_dir}")


if __name__ == "__main__":
    main()
