#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B11 - Searchlight version of the retrieval decoding (whole brain).

The region analyses (B4, B7, C9, C11) are repeated in spheres all over the
brain. For every sphere the object classifier is trained on the two localizers
exactly as in B4 (shown and recalled trials, 4 mm smoothing, average of four
volumes per trial, top 30 percent of voxels by F, logistic regression) and
applied to the ten volumes after each retrieval cue as in B7. From the class
probabilities three measures are taken per trial, each as retrieved minus
control (the control is the other category sequence, as in C9 and C11):

    item          probability of the retrieved item minus the item at the same
                  position of the other sequence, mean over 4.5-12 s
    abs_slope     mean absolute slope over 4.5-12 s, all five items,
                  retrieved minus control sequence
    abs_slope_wo  the same without the retrieved item (and without the item
                  at its position in the control sequence)
    predecessor,  as item, for the item before and after the retrieved one
    successor     (missing when the first or last item is retrieved)

In addition the accuracy of the classifier on the localizers is stored
(trained on one localizer session, tested on the other, both directions;
all, shown and recalled trials).

Space. By default the data are taken in each participant's own anatomy (T1w
space, as in the region analyses). The sphere centres are defined once in
standard space (every second voxel, 4 mm grid, of the group grey matter) and
carried into each participant's anatomy with the fMRIPrep transform, so that
sphere k is the same place in everyone and the results can be tested on the
standard grid without resampling any map. B11_SPACE=MNI uses the data in
standard space instead (MNI152NLin2009cAsym).

Spheres: radii B11_RADII (default 10, 15 and 20 mm: about 520, 1770 and 4190
voxels of 2 mm). Spheres with less than 30 percent of their volume inside a
participant's brain mask are skipped for that participant.

Differences from the region analyses:
  - no reliability mask: all brain voxels of a sphere enter, the F-statistic
    selection is done within each sphere
  - the sinusoid amplitude is not computed: its fit takes about 50 ms per
    trial and sequence, which is not feasible for some 20,000 spheres

Input   outputs/B4_train_classifiers/behavior_with_training_columns/sXX/
        outputs/A1_add_tr_columns_main, outputs/A2_add_tr_columns_localizer
        outputs/B7_apply_classifiers_retrieval/<session>/decoding_results_tables_for_SODA/
            sXX_ROI_vis_*.csv   (trial list, sequences, button press, search time)
        <derivatives>/sub-XX/ses-0{2,3,4,6}/func/  BOLD and brain masks
        <derivatives>/sub-XX/anat/                 grey-matter probability in MNI space, T1w-to-MNI transform
Output  outputs/B11_searchlight_retrieval/centres.npz, grey_matter_mask.nii.gz
        outputs/B11_searchlight_retrieval/participants/sXX_searchlight_<space>_r<radius>.npz
            (per sphere and trial: item, abs_slope, abs_slope_wo, predecessor, successor;
             per sphere: accuracy, n_voxels)
        outputs/B11_searchlight_retrieval/participants/sXX_trials.csv

Not a port of the original pipeline: added 2026-11-25.

The cross-validated accuracy on the localizers (leave one run out, as in B4/B5;
all, shown and recalled trials) is computed in a separate pass with
B11_ONLY_ACCURACY=1 and written to participants/sXX_accuracy_<space>_r<radius>.npz.

Usage:  python B11_searchlight_retrieval.py centres           (once, before the participants)
        python B11_searchlight_retrieval.py <subject>         e.g. sub-44
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking, signal
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "C_statistics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import B4_train_classifiers as B4
import B7_apply_classifiers_retrieval as B7
from C9_item_time_courses import POST1_MAX_TRIAL, category_sequences
from config import DERIVATIVES_DIR, is_excluded, step_input_dir, step_output_dir
from decoding_utils import CONFOUND_VARS, assert_func_conf_match, parse_subject_arg

warnings.filterwarnings("ignore", category=FutureWarning)        # sklearn multi_class, as in B4
warnings.filterwarnings("ignore", category=DeprecationWarning)

MNI = "MNI152NLin2009cAsym"
SPACE_KEY = os.environ.get("B11_SPACE", "T1w")              # T1w (each participant's own anatomy) or MNI
SPACE = {"T1w": "T1w", "MNI": MNI}[SPACE_KEY]
RADII = [float(r) for r in os.environ.get("B11_RADII", "10,15,20").split(",")]
STRIDE = 2                  # centres on every second voxel in each direction
GM_THRESHOLD = 0.25         # mean grey-matter probability across participants
SMOOTH, F_PERCENTILE = 4, 30
TEMPORAL_OFFSETS = (-1, 0, 1, 2)     # B4: temporal smoothing level 3
TR_WINDOW = np.arange(0, 10)
WINDOW = slice(3, 9)                 # volumes 3-8 = 4.5-12 s after the cue
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
MAX_CENTRES = int(os.environ.get("B11_MAX_CENTRES") or 0)   # for a timing test
ONLY_ACCURACY = os.environ.get("B11_ONLY_ACCURACY") == "1"  # cross-validated accuracy on the localizers only


# =====================================================================
# Centres (once)
# =====================================================================

def _anat(sub_mri, name):
    hits = sorted(glob.glob(opj(str(DERIVATIVES_DIR), sub_mri, "anat", f"{sub_mri}_{name}"))
                  + glob.glob(opj(str(DERIVATIVES_DIR), sub_mri, "ses-*", "anat", f"{sub_mri}_*{name}")))
    return hits[0] if hits else None


def make_centres():
    out = str(step_output_dir(__file__))
    tables = glob.glob(opj(str(step_input_dir("B7_apply_classifiers_retrieval", opj("presleep", "decoding_results_tables_for_SODA"))),
                           "s*_ROI_vis_*.csv"))
    subs = sorted({"sub-" + re.match(r"s(\d+)_", os.path.basename(t)).group(1) for t in tables})
    ref = nib.load(sorted(glob.glob(opj(str(DERIVATIVES_DIR), subs[0], "ses-04", "func", f"*space-{MNI}_boldref.nii.gz")))[0])
    gm, n = np.zeros(ref.shape[:3]), 0
    for s in subs:
        f = _anat(s, f"space-{MNI}_label-GM_probseg.nii.gz")
        if f is None:
            print(f"no grey-matter image for {s}")
            continue
        gm += image.resample_to_img(f, ref, interpolation="continuous", force_resample=True, copy_header=True).get_fdata()
        n += 1
    gm /= n
    mask = gm > GM_THRESHOLD
    grid = np.zeros_like(mask)
    grid[::STRIDE, ::STRIDE, ::STRIDE] = True
    ijk = np.argwhere(mask & grid)
    np.savez(opj(out, "centres.npz"), ijk=ijk, affine=ref.affine, shape=np.array(ref.shape[:3]))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine), opj(out, "grey_matter_mask.nii.gz"))
    nib.save(nib.Nifti1Image(gm.astype(np.float32), ref.affine), opj(out, "grey_matter_mean.nii.gz"))
    print(f"{n} participants, {int(mask.sum())} grey-matter voxels, {len(ijk)} centres -> {out}")


# =====================================================================
# Data of one participant
# =====================================================================

def _runs(sub_mri, session, pattern):
    d = opj(str(DERIVATIVES_DIR), sub_mri, session, "func")
    func = sorted(glob.glob(opj(d, f"{sub_mri}_{session}_{pattern}_space-{SPACE}_desc-preproc_bold.nii.gz")), key=os.path.basename)
    conf = sorted(glob.glob(opj(d, f"{sub_mri}_{session}_{pattern}_desc-confounds_timeseries.tsv")), key=os.path.basename)
    assert_func_conf_match(func, conf)
    return func, conf


def _clean_rows(func, conf, mask_img, rows):
    """Mask, smooth and clean one run as in B4/B7; return the requested volumes (float32). rows: list of index arrays."""
    x = masking.apply_mask(func, mask_img, smoothing_fwhm=SMOOTH)
    x = signal.clean(signals=x, t_r=B4.TR, detrend=True, standardize="zscore_sample",
                     confounds=pd.read_csv(conf, sep="\t")[CONFOUND_VARS].fillna(0))
    return [x[r].astype(np.float32) for r in rows]


def localizer_plan(sub):
    """Runs, labels and volumes of the object training trials (as B4.prepare_localizer_behavior)."""
    sub_num, sub_mri = sub[1:], "sub-" + sub[1:]
    folder = opj(str(step_input_dir("B4_train_classifiers", "behavior_with_training_columns")), sub)
    plan = []          # (func, conf, onsets (volumes), labels, session index)
    for k, (loc, ses) in enumerate((("loc1", "ses-02"), ("loc2", "ses-03"))):
        df = pd.read_csv(opj(folder, f"s{sub_num}_df_{loc}_w_HRF_with_training_columns.csv"), sep=";")
        df = df.dropna(subset=["training_category"]).reset_index(drop=True)
        func, conf = _runs(sub_mri, ses, "task-*_run-*")
        groups = [g for _, g in sorted(df.groupby("run_nr"), key=lambda x: x[0])]
        if len(groups) != len(func):
            raise ValueError(f"{sub} {loc}: {len(groups)} behavioural runs, {len(func)} fMRI runs")
        for g, f, c in zip(groups, func, conf):
            g = g[g.training_category.isin(B4.OBJECT_CATEGORIES)]
            plan.append((f, c, g.TR_time_onset_w_HRF.values, g.training_category.values, k,
                         (g.trial_phase == "retrieval").values, len(plan)))
    return plan


def retrieval_plan(sub, session):
    """Trials of one retrieval session: fMRI run, onset volume, sequences, position, behaviour."""
    cfg = B7.SESSION_CONFIG[session]
    sub_mri = "sub-" + sub[1:]
    table = glob.glob(opj(str(step_input_dir("B7_apply_classifiers_retrieval", opj(session, "decoding_results_tables_for_SODA"))),
                          f"{sub}_ROI_vis_decoding_results_retrieval*.csv"))
    if not table or is_excluded(sub, session):
        return None
    d = pd.read_csv(table[0])
    seqs = category_sequences(d)
    info = d[d.object_scene_retrieval == "object"].drop_duplicates("trial").set_index("trial")

    _, obj_enc_to_cat = B7.load_localizer_behavior(str(step_input_dir("A2_add_tr_columns_localizer")), sub)
    _, df_ret = B7.load_main_task_behavior(str(step_input_dir("A1_add_tr_columns_main")), sub, cfg)
    df_ret = B7.prepare_retrieval_df(df_ret, obj_enc_to_cat)
    df_obj = df_ret[df_ret.label_type == "sc_no4obj"]
    df_sc = df_ret[df_ret.label_type == "d_no4sc"]
    run_map = B7.build_run_nr_mapping(df_obj, df_sc, cfg["n_retrieval_runs"])

    # the fMRI runs B7.select_retrieval_runs would take, decided on the run lengths
    func, conf = _runs(sub_mri, cfg["mri_session"], "task-mem*_run*")
    long = [(f, c) for f, c in zip(func, conf) if nib.load(f).shape[3] > 120]
    if cfg["expected_exact_runs"]:
        if len(long) != 6:
            raise ValueError(f"{sub} {session}: expected 6 runs longer than 120 volumes, got {len(long)}")
        runs = long[-cfg["n_retrieval_runs"]:]
    else:
        if len(long) < 6:
            raise ValueError(f"{sub} {session}: fewer than 6 retrieval runs")
        keep = sorted(sorted(range(len(long)), key=lambda i: nib.load(long[i][0]).shape[3], reverse=True)[:6])
        runs = [long[i] for i in keep]

    rows = []
    for _, r in df_obj.iterrows():
        t = r.trl_nr
        if t not in info.index or pd.isna(r.run_nr) or int(r.run_nr) not in run_map or pd.isna(r.TR_onset_anchor):
            continue
        b = info.loc[t]
        if b.retrieval_target != r.obj_cat:
            raise ValueError(f"{sub} {session} trial {t}: target {r.obj_cat} differs from the B7 table ({b.retrieval_target})")
        seq = seqs[0] if b.retrieval_target in seqs[0] else seqs[1]
        other = seqs[1] if seq is seqs[0] else seqs[0]
        rows.append(dict(subject=sub, session=session, trial=int(t), run=run_map[int(r.run_nr)],
                         onset=int(np.round(r.TR_onset_anchor)), position=int(b.retrieval_cue_position),
                         cond="pre" if session == "presleep" else ("post1" if t <= POST1_MAX_TRIAL else "post2"),
                         press=bool(b.behav_correct), rt=b.ret_rt, retrieved="|".join(seq), control="|".join(other)))
    return pd.DataFrame(rows), runs


# =====================================================================
# One sphere
# =====================================================================

G = {}      # data shared with the worker processes (fork)





def _weights():
    pos = np.arange(1, 6, dtype=float)
    w5 = (pos - pos.mean()) / ((pos - pos.mean()) ** 2).sum()
    wo = np.zeros((5, 5))
    for p in range(5):
        k = [i for i in range(5) if i != p]
        xc = pos[k] - pos[k].mean()
        wo[p, k] = xc / (xc ** 2).sum()
    return w5, wo


def sphere(c):
    vox = G["neigh"][c]
    if len(vox) < G["min_vox"]:
        return None
    Xtr, y = G["Xtr"][:, vox], G["y"]
    clf = B4.build_clf(F_PERCENTILE).fit(*G["resample"](Xtr, y))
    P = clf.predict_proba(G["Xret"][:, vox]) * 100.0
    P[~G["valid"]] = np.nan
    P = P.reshape(-1, len(TR_WINDOW), P.shape[1])                       # trials x volumes x classes
    R = np.take_along_axis(P, G["Ridx"][:, None, :], axis=2)            # trials x volumes x 5 positions
    C = np.take_along_axis(P, G["Cidx"][:, None, :], axis=2)
    t = np.arange(len(R))
    item = np.nanmean(R[t, WINDOW, G["pos"]] - C[t, WINDOW, G["pos"]], axis=1)
    w5, wo = G["w5"], G["wo"][G["pos"]]                                 # wo: trials x 5
    a5 = np.nanmean(np.abs(R @ w5)[:, WINDOW] - np.abs(C @ w5)[:, WINDOW], axis=1)
    a4 = np.nanmean(np.abs((R * wo[:, None, :]).sum(2))[:, WINDOW] - np.abs((C * wo[:, None, :]).sum(2))[:, WINDOW], axis=1)
    # the items next to the retrieved one (missing at the ends of the sequence)
    near = []
    for lag in (-1, 1):
        q = G["pos"] + lag
        ok = (q >= 0) & (q <= 4)
        qq = np.clip(q, 0, 4)
        v = np.nanmean(R[t, WINDOW, qq] - C[t, WINDOW, qq], axis=1)
        v[~ok] = np.nan
        near.append(v.astype(np.float16))
    hit = np.zeros(len(y), bool)
    for k in (0, 1):                                                    # train on one localizer session, test on the other
        tr, te = G["loc"] == k, G["loc"] != k
        hit[te] = B4.build_clf(F_PERCENTILE).fit(*G["resample"](Xtr[tr], y[tr])).predict(Xtr[te]) == y[te]
    rec = G["recalled"]
    return (item.astype(np.float16), a5.astype(np.float16), a4.astype(np.float16), near[0], near[1],
            (hit.mean(), hit[~rec].mean(), hit[rec].mean()), len(vox))


def _chunk(cs):
    return [(c, sphere(c)) for c in cs]


def sphere_accuracy(c):
    """Leave-one-run-out accuracy on the localizers (as B4/B5): all, shown and recalled trials."""
    vox = G["neigh"][c]
    if len(vox) < G["min_vox"]:
        return None
    X, y, runs = G["Xtr"][:, vox], G["y"], G["runs"]
    hit = np.zeros(len(y), bool)
    for r in np.unique(runs):
        te = runs == r
        hit[te] = B4.build_clf(F_PERCENTILE).fit(*G["resample"](X[~te], y[~te])).predict(X[te]) == y[te]
    rec = G["recalled"]
    return hit.mean(), hit[~rec].mean(), hit[rec].mean()


def _chunk_accuracy(cs):
    return [(c, sphere_accuracy(c)) for c in cs]


def accuracy_maps(sub, name, radii, aff, centres, centres_mm, m, Xtr, y, recalled, runs, t0):
    good = (Xtr.std(0) > 0) & np.isfinite(Xtr).all(0)
    Xtr = Xtr[:, good]
    tree = cKDTree(nib.affines.apply_affine(aff, np.argwhere(m)[good]))
    counts = pd.Series(y).value_counts()
    if counts.nunique() > 1:
        from imblearn.over_sampling import SMOTE
        resample = lambda X, yy: SMOTE(random_state=B4.SMOTE_SEED).fit_resample(X, yy)
    else:
        resample = lambda X, yy: (X, yy)
    G.update(Xtr=Xtr, y=y, runs=runs, recalled=recalled, resample=resample)
    print(f"{len(y)} training trials ({int((~recalled).sum())} shown, {int(recalled.sum())} recalled), {len(np.unique(runs))} runs")
    chunks = [np.arange(i, min(i + 20, len(centres))) for i in range(0, len(centres), 20)]
    for radius in radii:
        G["neigh"] = tree.query_ball_point(centres_mm, radius)
        G["min_vox"] = 0.3 * 4 / 3 * np.pi * radius ** 3 / 8.0        # 30 percent of a full sphere of 2 mm voxels
        acc = np.full((len(centres), 3), np.nan, np.float32)
        t1 = time.time()
        with mp.get_context("fork").Pool(N_JOBS) as pool:
            for k, part in enumerate(pool.imap_unordered(_chunk_accuracy, chunks)):
                for c, r in part:
                    if r is not None:
                        acc[c] = r
                if k % 100 == 0:
                    print(f"  r = {radius:.0f} mm: {k * 20}/{len(centres)} spheres, {(time.time() - t1) / 60:.1f} min", flush=True)
        f = name(radius).replace(".npz", f"_test{MAX_CENTRES}.npz") if MAX_CENTRES else name(radius)
        np.savez_compressed(f, all=acc[:, 0], shown=acc[:, 1], recalled=acc[:, 2], centres=centres)
        print(f"r = {radius:.0f} mm accuracy: all {np.nanmean(acc[:, 0]) * 100:.1f} %, shown {np.nanmean(acc[:, 1]) * 100:.1f} %, "
              f"recalled {np.nanmean(acc[:, 2]) * 100:.1f} % (mean over spheres); best sphere {np.nanmax(acc[:, 0]) * 100:.1f} %"
              f"  ({(time.time() - t0) / 60:.1f} min in total)", flush=True)


def run_job(job):
    """Read one run and return the volumes needed from it."""
    mask_img = G["mask_img"]
    if job[0] == "loc":
        _, f, c, onsets, labels, k, recalled, run = job
        n = nib.load(f).shape[3]
        keep = onsets < n                                    # B4.check_tr_alignment
        on = onsets[keep].astype(int)
        ok = [((on + o) >= 0) & ((on + o) < n) for o in TEMPORAL_OFFSETS]
        parts = _clean_rows(f, c, mask_img, [np.clip(on + o, 0, n - 1) for o in TEMPORAL_OFFSETS])
        avg = sum(p * o[:, None] for p, o in zip(parts, ok)) / sum(o[:, None] for o in ok)   # B4.apply_temporal_smoothing
        return avg.astype(np.float32), labels[keep], np.full(int(keep.sum()), k), recalled[keep], np.full(int(keep.sum()), run)
    _, f, c, s, i, _ = job
    tr = G["ret"][s][0]
    tr = tr[tr.run == i]
    n = nib.load(f).shape[3]
    idx = tr.onset.values[:, None] + TR_WINDOW[None, :]
    valid = (idx >= 0) & (idx < n)
    parts = _clean_rows(f, c, mask_img, [np.clip(idx, 0, n - 1).ravel()])
    return parts[0], valid.ravel(), tr


# =====================================================================
# Participant
# =====================================================================

def participant(sub):
    out = str(step_output_dir(__file__, "participants"))
    name = lambda r: opj(out, f"{sub}_{'accuracy' if ONLY_ACCURACY else 'searchlight'}_{SPACE_KEY}_r{r:.0f}.npz")
    radii = [r for r in RADII if MAX_CENTRES or not os.path.exists(name(r))]
    if not radii:
        print(f"{sub}: all outputs exist")
        return
    cen = np.load(opj(str(step_output_dir(__file__)), "centres.npz"))
    t0 = time.time()

    loc = localizer_plan(sub)
    ret = {s: retrieval_plan(sub, s) for s in ("presleep", "postsleep")}
    ret = {s: v for s, v in ret.items() if v is not None}
    all_funcs = [p[0] for p in loc] + [f for v in ret.values() for f, _ in v[1]]      # same mask in both modes

    # brain mask: voxels inside the brain mask of every run used
    first = nib.load(all_funcs[0].replace("desc-preproc_bold", "desc-brain_mask"))
    aff = first.affine
    m = np.ones(first.shape, bool)
    for f in all_funcs:
        img = nib.load(f.replace("desc-preproc_bold", "desc-brain_mask"))
        if img.shape != first.shape or not np.allclose(img.affine, aff, atol=1e-3):
            raise ValueError(f"{sub}: the runs are not on one grid ({f})")
        m &= img.get_fdata() > 0
    mask_img = nib.Nifti1Image(m.astype(np.uint8), aff)

    # sphere centres: the same points in standard space for everyone; for T1w they are carried
    # into the participant's own anatomy (the T1w-to-MNI image transform maps MNI points to T1w points)
    centres = cen["ijk"][:MAX_CENTRES] if MAX_CENTRES else cen["ijk"]
    centres_mm = nib.affines.apply_affine(cen["affine"], centres)
    if SPACE_KEY == "T1w":
        from nitransforms.manip import load as ntload
        sub_mri = "sub-" + sub[1:]
        centres_mm = ntload(_anat(sub_mri, f"from-T1w_to-{MNI}_mode-image_xfm.h5"), fmt="itk").map(centres_mm)
    print(f"{sub}: {len(loc)} localizer runs, {sum(len(v[1]) for v in ret.values())} retrieval runs, {int(m.sum())} voxels")

    # --- jobs: one per run, each returns the volumes needed -------------------------------
    jobs = []
    for f, c, onsets, labels, k, recalled, run in loc:
        jobs.append(("loc", f, c, onsets, labels, k, recalled, run))
    for s, (trials, runs) in ret.items():
        for i, (f, c) in enumerate(runs):
            if not ONLY_ACCURACY:
                jobs.append(("ret", f, c, s, i, None))

    G.update(ret=ret, mask_img=mask_img)
    with mp.get_context("fork").Pool(min(N_JOBS, 6)) as pool:
        res = pool.map(run_job, jobs)
    print(f"data read and cleaned in {(time.time() - t0) / 60:.1f} min")

    nloc = len(loc)
    Xtr = np.vstack([r[0] for r in res[:nloc]])
    y = np.concatenate([r[1] for r in res[:nloc]])
    locs = np.concatenate([r[2] for r in res[:nloc]])
    recalled = np.concatenate([r[3] for r in res[:nloc]])
    runs = np.concatenate([r[4] for r in res[:nloc]])
    if ONLY_ACCURACY:
        return accuracy_maps(sub, name, radii, aff, centres, centres_mm, m, Xtr, y, recalled, runs, t0)
    Xret = np.vstack([r[0] for r in res[nloc:]])
    valid = np.concatenate([r[1] for r in res[nloc:]])
    trials = pd.concat([r[2] for r in res[nloc:]], ignore_index=True)
    del res

    good = (Xtr.std(0) > 0) & np.isfinite(Xtr).all(0) & np.isfinite(Xret).all(0)
    Xtr, Xret = Xtr[:, good], Xret[:, good]
    vox_ijk = np.argwhere(m)[good]
    tree = cKDTree(nib.affines.apply_affine(aff, vox_ijk))

    classes = np.unique(y)
    cidx = {c: i for i, c in enumerate(classes)}
    counts = pd.Series(y).value_counts()
    print(f"{len(y)} training trials ({counts.min()}-{counts.max()} per class), {len(trials)} retrieval trials, "
          f"{Xtr.shape[1]} voxels, {len(centres)} centres")
    if counts.nunique() > 1:
        from imblearn.over_sampling import SMOTE
        resample = lambda X, yy: SMOTE(random_state=B4.SMOTE_SEED).fit_resample(X, yy)     # as B4
    else:
        resample = lambda X, yy: (X, yy)
    w5, wo = _weights()
    G.update(Xtr=Xtr, y=y, loc=locs, recalled=recalled, Xret=Xret, valid=valid, resample=resample, w5=w5, wo=wo,
             Ridx=np.array([[cidx[c] for c in s.split("|")] for s in trials.retrieved]),
             Cidx=np.array([[cidx[c] for c in s.split("|")] for s in trials.control]),
             pos=trials.position.values - 1)

    trials.to_csv(opj(out, f"{sub}_trials.csv"), index=False)
    chunks = [np.arange(i, min(i + 20, len(centres))) for i in range(0, len(centres), 20)]
    n_tr = len(trials)
    for radius in radii:
        t1 = time.time()
        G["neigh"] = tree.query_ball_point(centres_mm, radius)
        G["min_vox"] = 0.3 * 4 / 3 * np.pi * radius ** 3 / 8.0        # 30 percent of a full sphere of 2 mm voxels
        item, a5, a4, pred, succ = (np.full((len(centres), n_tr), np.nan, np.float16) for _ in range(5))
        acc, nvox = np.full((len(centres), 3), np.nan, np.float32), np.zeros(len(centres), np.int32)
        with mp.get_context("fork").Pool(N_JOBS) as pool:
            for k, part in enumerate(pool.imap_unordered(_chunk, chunks)):
                for c, r in part:
                    if r is not None:
                        item[c], a5[c], a4[c], pred[c], succ[c], acc[c], nvox[c] = r
                if k % 100 == 0:
                    print(f"  r = {radius:.0f} mm: {k * 20}/{len(centres)} spheres, {(time.time() - t1) / 60:.1f} min", flush=True)
        print(f"r = {radius:.0f} mm done in {(time.time() - t1) / 60:.1f} min; {np.median(nvox[nvox > 0]):.0f} voxels per sphere; "
              f"accuracy mean {np.nanmean(acc[:, 0]) * 100:.1f} %, best sphere {np.nanmax(acc[:, 0]) * 100:.1f} %", flush=True)
        f = name(radius).replace(".npz", f"_test{MAX_CENTRES}.npz") if MAX_CENTRES else name(radius)
        np.savez_compressed(f, item=item, abs_slope=a5, abs_slope_wo=a4, predecessor=pred, successor=succ,
                            accuracy=acc[:, 0], accuracy_shown=acc[:, 1], accuracy_recalled=acc[:, 2], n_voxels=nvox,
                            centres=centres, centres_mm=centres_mm)
    print(f"-> {out}  ({(time.time() - t0) / 60:.1f} min in total)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "centres":
        make_centres()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
