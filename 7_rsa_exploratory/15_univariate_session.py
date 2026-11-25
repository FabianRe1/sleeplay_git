#!/usr/bin/env python3
"""
Univariate retrieval activity, evening versus morning, for the same items.

Question: when the same item is retrieved in the evening and in the morning, does
the mean activity of a region change (for example more parietal / precuneus
engagement and less hippocampal engagement after sleep)?

Data: LSS single-trial betas of the retrieval cue (04_single_trial_betas.py),
whole brain, percent signal change relative to the run's implicit baseline.
Regions: ANATOMICAL masks from FreeSurfer labels (2_get_threshold_roi, T1w space,
no signal threshold), not the reliability masks, because voxel selection by
category reliability is not meaningful for mean activity.

Per trial and region: mean beta over the region's voxels. Then
  item-matched difference : per participant, the mean over items of
                            (beta of the item in session A - beta of the same item
                            in session B); items = objects (object trials) and
                            scenes (scene trials, two trials per session averaged)
  variants                : all trials / item correct in both sessions /
                            activity relative to the whole-brain mean of the trial
                            (removes global session differences)
  RT-adjusted             : per participant, beta ~ session + response time; the
                            session coefficients. Response times are about 0.5 s
                            shorter in the morning, and shorter retrieval alone
                            lowers the measured activity.
  region contrast         : (parietal + precuneus) - hippocampus, by session.
Sessions: pre = evening, post1 / post2 = morning pass 1 / 2, post = mean of both.
Main family for correction (p_fdr_main): parietal + precuneus, hippocampus, MTL,
visual, caudal ACC (control). Auditory and somatosensory cortex, the ventricles
and the whole brain are included as further references.

Usage:  python 15_univariate_session.py [n_jobs] [lss|lss_rt]
Output: outputs/15_univariate_session/ (lss) or outputs/15_univariate_session_lss_rt/
"""

import glob
import os
import sys
from os.path import join as opj

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import resample_to_img
from scipy import stats

import rsa_config as cfg

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SOURCE = sys.argv[2] if len(sys.argv) > 2 else "lss"        # "lss" (fixed 3 s events) or "lss_rt" (cue to response)
OUT_DIR = opj(cfg.OUT_ROOT, "15_univariate_session" + ("" if SOURCE == "lss" else f"_{SOURCE}"))
BETA_DIR = opj(cfg.OUT_ROOT, "04_betas", SOURCE)
MASK_DIR = opj(cfg.OUT_ROOT, "04_betas", "masks")
ANAT_DIR = opj(cfg.CODE_ROOT, "2_get_threshold_roi", "localizer_getThresholdedROI", "thresholded_roi_output")

ROIS = {"PARIETAL_PRECUNEUS": "Parietal + precuneus", "PARIETAL": "Parietal", "PRECUNEUS": "Precuneus",
        "HPC": "Hippocampus", "PHPC": "Parahippocampal", "ERH": "Entorhinal", "MTL": "MTL",
        "VIS": "Visual", "CAUDAL_ACC": "Caudal ACC (control)", "RSC": "Retrosplenial",
        "POSTERIOR_CINGULATE": "Posterior cingulate", "SUPERIOR_FRONTAL": "Superior frontal",
        "MEDIAL_OFC": "Medial OFC", "AUD": "Auditory", "SOMATO": "Somatosensory",
        "ventricles": "Ventricles", "wholebrain": "Whole brain"}
MAIN = ["PARIETAL_PRECUNEUS", "HPC", "MTL", "VIS", "CAUDAL_ACC"]
CONTRASTS = [("post1", "pre"), ("post", "pre"), ("post2", "post1")]


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


def subject_trials(sub):
    """Trial table with the mean beta of every region (sub = 's14')."""
    f_mask = opj(MASK_DIR, f"{sub}_brainmask.nii.gz")
    if not os.path.exists(f_mask):
        return None
    brain = nib.load(f_mask)
    bm = brain.get_fdata() > 0
    num = sub[1:]
    roi_idx, n_vox = {}, {}
    for roi in ROIS:
        hits = sorted(glob.glob(opj(ANAT_DIR, f"sub-{num}", "ses-04", f"{roi}_sub-{num}_ses-04_space-T1w_T-0_run-*.nii")))
        if not hits:
            continue
        img = nib.load(hits[0])
        if img.shape != brain.shape or not np.allclose(img.affine, brain.affine, atol=1e-3):
            img = resample_to_img(img, brain, interpolation="nearest")
        m = (img.get_fdata() > 0)[bm]                 # in the order of the beta columns
        if m.sum() >= 10:
            roi_idx[roi], n_vox[roi] = m, int(m.sum())
    parts = []
    for phase in ("presleep", "postsleep"):
        f_b, f_t = opj(BETA_DIR, f"{sub}_{phase}_betas.npy"), opj(BETA_DIR, f"{sub}_{phase}_trials.csv")
        if not (os.path.exists(f_b) and os.path.exists(f_t)):
            continue
        b = np.load(f_b).astype(np.float32)
        t = pd.read_csv(f_t)
        if b.shape[1] != bm.sum():
            raise ValueError(f"{sub}: {b.shape[1]} beta columns but {bm.sum()} mask voxels")
        out = t[["subject", "kind", "object", "scene", "position", "correct", "rt", "run_idx"]].copy()
        out["cond"] = "pre" if phase == "presleep" else np.where(t.run_idx < 3, "post1", "post2")
        out["item"] = np.where(t.kind == "object", t.object, t.scene)
        for roi, m in roi_idx.items():
            out[roi] = np.nanmean(b[:, m], axis=1)
        parts.append(out)
    if not parts:
        return None
    x = pd.concat(parts, ignore_index=True)
    x.attrs["n_vox"] = n_vox
    return x, n_vox


def item_matched(tr, rois, both_correct=False):
    """Per participant and kind: mean over items of the between-session difference, per region."""
    rows = []
    for (sub, kind), g in tr.groupby(["subject", "kind"]):
        m = g.groupby(["item", "cond"])[rois].mean().unstack("cond")
        ok = g.groupby(["item", "cond"])["correct"].min().unstack("cond")
        for a, b in CONTRASTS:
            conds_a = ["post1", "post2"] if a == "post" else [a]
            if not all(c in ok.columns for c in conds_a + [b]):
                continue
            keep = ok[conds_a + [b]].notna().all(axis=1)
            if both_correct:
                keep &= (ok[conds_a + [b]] == True).all(axis=1)          # noqa: E712
            row = dict(subject=sub, kind=kind, contrast=f"{a} - {b}", n_items=int(keep.sum()))
            if keep.sum() < 5:
                continue
            for r in rois:
                va = m[r][conds_a].mean(axis=1)
                row[r] = float((va - m[r][b])[keep].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def rt_adjusted(tr, rois):
    rows = []
    for (sub, kind), g in tr.groupby(["subject", "kind"]):
        g = g.dropna(subset=["rt"])
        if g.cond.nunique() < 3 or len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g)), (g.cond == "post1").values, (g.cond == "post2").values,
                             g.rt.values - g.rt.mean()]).astype(float)
        for r in rois:
            b = np.linalg.lstsq(X, g[r].values, rcond=None)[0]
            rows.append(dict(subject=sub, kind=kind, roi=r, post1_minus_pre=b[1], post2_minus_pre=b[2],
                             post_minus_pre=(b[1] + b[2]) / 2, per_second_rt=b[3]))
    return pd.DataFrame(rows)


def group(df, rois, extra):
    rows = []
    for keys, g in df.groupby(["kind", "contrast"]):
        for r in rois:
            rows.append(dict(kind=keys[0], contrast=keys[1], roi=r, **extra, **tt(g[r])))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(BETA_DIR, "*_presleep_betas.npy"))})
    res = [r for r in Parallel(n_jobs=N_JOBS)(delayed(subject_trials)(s) for s in subs) if r is not None]
    tr = pd.concat([r[0] for r in res], ignore_index=True)
    rois = [r for r in ROIS if r in tr.columns]
    pd.DataFrame([dict(subject=r[0].subject.iloc[0], **r[1]) for r in res]).to_csv(opj(OUT_DIR, "roi_voxel_counts.csv"), index=False)
    tr.to_csv(opj(OUT_DIR, "trial_roi_means.csv"), index=False)

    rel = tr.copy()                                            # relative to the whole-brain mean of the trial
    for r in rois:
        if r != "wholebrain":
            rel[r] = tr[r] - tr["wholebrain"]
    tr["PARPREC_minus_HPC"] = tr["PARIETAL_PRECUNEUS"] - tr["HPC"]
    rel["PARPREC_minus_HPC"] = tr["PARPREC_minus_HPC"]
    rois_x = rois + ["PARPREC_minus_HPC"]

    rows, subj = [], []
    for vname, d, bc in (("all trials", tr, False), ("both correct", tr, True),
                         ("relative to whole brain", rel, False), ("relative, both correct", rel, True)):
        im = item_matched(d, rois_x, bc)
        subj.append(im.assign(variant=vname))
        rows += group(im, rois_x, dict(variant=vname))
    g = pd.DataFrame(rows)

    ra = rt_adjusted(tr, rois_x)
    ra.to_csv(opj(OUT_DIR, "subject_rt_adjusted.csv"), index=False)
    rrows = []
    for (kind, roi), gg in ra.groupby(["kind", "roi"]):
        for col, lab in (("post1_minus_pre", "post1 - pre"), ("post_minus_pre", "post - pre"), ("per_second_rt", "per second RT")):
            rrows.append(dict(kind=kind, contrast=lab, roi=roi, variant="RT-adjusted", **tt(gg[col])))
    g = pd.concat([g, pd.DataFrame(rrows)], ignore_index=True)
    g["p_fdr_main"] = np.nan
    m = g.roi.isin(MAIN)
    g.loc[m, "p_fdr_main"] = g[m].groupby(["kind", "contrast", "variant"])["p"].transform(fdr)
    g["region"] = g.roi.map(ROIS).fillna(g.roi)
    g.to_csv(opj(OUT_DIR, "group_session_differences.csv"), index=False)
    pd.concat(subj, ignore_index=True).to_csv(opj(OUT_DIR, "subject_item_matched.csv"), index=False)

    # level per session (activity against the implicit baseline)
    lev = tr.groupby(["subject", "kind", "cond"])[rois_x].mean().reset_index()
    lrows = []
    for (kind, cond), gg in lev.groupby(["kind", "cond"]):
        for r in rois_x:
            lrows.append(dict(kind=kind, cond=cond, roi=r, region=ROIS.get(r, r), **tt(gg[r])))
    pd.DataFrame(lrows).to_csv(opj(OUT_DIR, "group_level_per_session.csv"), index=False)
    print("written to", OUT_DIR, "| subjects:", tr.subject.nunique())


if __name__ == "__main__":
    main()
