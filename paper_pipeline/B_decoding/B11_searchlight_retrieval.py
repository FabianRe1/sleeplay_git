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
(trained on one localizer session, tested on the other, both directions).

Differences from the region analyses, all forced by the whole-brain setting:
  - the data are taken in MNI space (fMRIPrep output MNI152NLin2009cAsym,
    2 mm) so that all participants share one grid; the regions used T1w space
  - no reliability mask: all brain voxels of a sphere enter, the F-statistic
    selection is done within each sphere
  - the sinusoid amplitude is not computed: its fit takes about 50 ms per
    trial and sequence, which is not feasible for some 20,000 spheres

Spheres: radius RADIUS_MM (10 mm, about 515 voxels) around every second voxel
(4 mm grid) of the group grey matter. Spheres with fewer than MIN_VOXELS
voxels inside a participant's brain mask are skipped for that participant.

Input   outputs/B4_train_classifiers/behavior_with_training_columns/sXX/
        outputs/A1_add_tr_columns_main, outputs/A2_add_tr_columns_localizer
        outputs/B7_apply_classifiers_retrieval/<session>/decoding_results_tables_for_SODA/
            sXX_ROI_vis_*.csv   (trial list, sequences, button press, search time)
        <derivatives>/sub-XX/ses-0{2,3,4,6}/func/  BOLD and brain masks in MNI space
        <derivatives>/sub-XX/anat/                 grey-matter probability in MNI space
Output  outputs/B11_searchlight_retrieval/centres.npz, grey_matter_mask.nii.gz
        outputs/B11_searchlight_retrieval/participants/sXX_searchlight.npz
            (per sphere and trial: item, abs_slope, abs_slope_wo, predecessor, successor;
             per sphere: accuracy, n_voxels)
        outputs/B11_searchlight_retrieval/participants/sXX_trials.csv

Not a port of the original pipeline: added 2026-11-25.

The cross-validated accuracy on the localizers (leave one run out, as in B4/B5;
all, shown and recalled trials) is computed in a separate pass with
B11_ONLY_ACCURACY=1 and written to participants/sXX_accuracy.npz.

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

SPACE = "MNI152NLin2009cAsym"
RADIUS_MM = 10.0
STRIDE = 2                  # centres on every second voxel in each direction
MIN_VOXELS = 150
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
    ref = nib.load(sorted(glob.glob(opj(str(DERIVATIVES_DIR), subs[0], "ses-04", "func", f"*space-{SPACE}_boldref.nii.gz")))[0])
    gm, n = np.zeros(ref.shape[:3]), 0
    for s in subs:
        f = _anat(s, f"space-{SPACE}_label-GM_probseg.nii.gz")
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
    if len(vox) < MIN_VOXELS:
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
    acc = []
    for k in (0, 1):                                                    # train on one localizer session, test on the other
        tr, te = G["loc"] == k, G["loc"] != k
        acc.append((B4.build_clf(F_PERCENTILE).fit(*G["resample"](Xtr[tr], y[tr])).predict(Xtr[te]) == y[te]).mean())
    return item.astype(np.float16), a5.astype(np.float16), a4.astype(np.float16), near[0], near[1], float(np.mean(acc)), len(vox)


def _chunk(cs):
    return [(c, sphere(c)) for c in cs]


def sphere_accuracy(c):
    """Leave-one-run-out accuracy on the localizers (as B4/B5): all, shown and recalled trials."""
    vox = G["neigh"][c]
    if len(vox) < MIN_VOXELS:
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


def accuracy_maps(sub, out, cen, m, Xtr, y, recalled, runs, t0):
    good = (Xtr.std(0) > 0) & np.isfinite(Xtr).all(0)
    Xtr = Xtr[:, good]
    mm = nib.affines.apply_affine(cen["affine"], np.argwhere(m)[good])
    centres = cen["ijk"][:MAX_CENTRES] if MAX_CENTRES else cen["ijk"]
    neigh = cKDTree(mm).query_ball_point(nib.affines.apply_affine(cen["affine"], centres), RADIUS_MM)
    counts = pd.Series(y).value_counts()
    if counts.nunique() > 1:
        from imblearn.over_sampling import SMOTE
        resample = lambda X, yy: SMOTE(random_state=B4.SMOTE_SEED).fit_resample(X, yy)
    else:
        resample = lambda X, yy: (X, yy)
    G.update(Xtr=Xtr, y=y, runs=runs, recalled=recalled, neigh=neigh, resample=resample)
    print(f"{len(y)} training trials ({int((~recalled).sum())} shown, {int(recalled.sum())} recalled), {len(np.unique(runs))} runs")
    acc = np.full((len(centres), 3), np.nan, np.float32)
    chunks = [np.arange(i, min(i + 20, len(centres))) for i in range(0, len(centres), 20)]
    t1 = time.time()
    with mp.get_context("fork").Pool(N_JOBS) as pool:
        for k, part in enumerate(pool.imap_unordered(_chunk_accuracy, chunks)):
            for c, r in part:
                if r is not None:
                    acc[c] = r
            if k % 100 == 0:
                print(f"  {k * 20}/{len(centres)} spheres, {(time.time() - t1) / 60:.1f} min", flush=True)
    suffix = f"_test{MAX_CENTRES}" if MAX_CENTRES else ""
    np.savez_compressed(opj(out, f"{sub}_accuracy{suffix}.npz"), all=acc[:, 0], shown=acc[:, 1], recalled=acc[:, 2], centres=centres)
    print(f"accuracy: all {np.nanmean(acc[:, 0]) * 100:.1f} %, shown {np.nanmean(acc[:, 1]) * 100:.1f} %, "
          f"recalled {np.nanmean(acc[:, 2]) * 100:.1f} % (mean over spheres); best sphere {np.nanmax(acc[:, 0]) * 100:.1f} %"
          f"  ({(time.time() - t0) / 60:.1f} min in total)")


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
    target = opj(out, f"{sub}_searchlight.npz")
    if ONLY_ACCURACY:
        target = opj(out, f"{sub}_accuracy.npz")
    if os.path.exists(target) and not MAX_CENTRES:
        print(f"{target} exists")
        return
    cen = np.load(opj(str(step_output_dir(__file__)), "centres.npz"))
    t0 = time.time()

    loc = localizer_plan(sub)
    ret = {s: retrieval_plan(sub, s) for s in ("presleep", "postsleep")}
    ret = {s: v for s, v in ret.items() if v is not None}
    all_funcs = [p[0] for p in loc] + [f for v in ret.values() for f, _ in v[1]]      # same mask in both modes

    # brain mask: voxels inside the brain mask of every run used
    m = np.ones(tuple(cen["shape"]), bool)
    for f in all_funcs:
        m &= nib.load(f.replace("desc-preproc_bold", "desc-brain_mask")).get_fdata() > 0
    mask_img = nib.Nifti1Image(m.astype(np.uint8), cen["affine"])
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
        return accuracy_maps(sub, out, cen, m, Xtr, y, recalled, runs, t0)
    Xret = np.vstack([r[0] for r in res[nloc:]])
    valid = np.concatenate([r[1] for r in res[nloc:]])
    trials = pd.concat([r[2] for r in res[nloc:]], ignore_index=True)
    del res

    good = (Xtr.std(0) > 0) & np.isfinite(Xtr).all(0) & np.isfinite(Xret).all(0)
    Xtr, Xret = Xtr[:, good], Xret[:, good]
    vox_ijk = np.argwhere(m)[good]
    mm = nib.affines.apply_affine(cen["affine"], vox_ijk)
    centres = cen["ijk"][:MAX_CENTRES] if MAX_CENTRES else cen["ijk"]
    neigh = cKDTree(mm).query_ball_point(nib.affines.apply_affine(cen["affine"], centres), RADIUS_MM)

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
    G.update(Xtr=Xtr, y=y, loc=locs, Xret=Xret, valid=valid, neigh=neigh, resample=resample, w5=w5, wo=wo,
             Ridx=np.array([[cidx[c] for c in s.split("|")] for s in trials.retrieved]),
             Cidx=np.array([[cidx[c] for c in s.split("|")] for s in trials.control]),
             pos=trials.position.values - 1)

    t1 = time.time()
    order = np.arange(len(centres))
    chunks = [order[i:i + 20] for i in range(0, len(order), 20)]
    n_tr = len(trials)
    item, a5, a4, pred, succ = (np.full((len(centres), n_tr), np.nan, np.float16) for _ in range(5))
    acc, nvox = np.full(len(centres), np.nan, np.float32), np.zeros(len(centres), np.int32)
    with mp.get_context("fork").Pool(N_JOBS) as pool:
        for k, part in enumerate(pool.imap_unordered(_chunk, chunks)):
            for c, r in part:
                if r is not None:
                    item[c], a5[c], a4[c], pred[c], succ[c], acc[c], nvox[c] = r
            if k % 50 == 0:
                print(f"  {k * 20}/{len(centres)} spheres, {(time.time() - t1) / 60:.1f} min", flush=True)
    print(f"spheres done in {(time.time() - t1) / 60:.1f} min; accuracy mean {np.nanmean(acc) * 100:.1f} %, max {np.nanmax(acc) * 100:.1f} %")

    suffix = f"_test{MAX_CENTRES}" if MAX_CENTRES else ""
    np.savez_compressed(opj(out, f"{sub}_searchlight{suffix}.npz"), item=item, abs_slope=a5, abs_slope_wo=a4, predecessor=pred, successor=succ,
                        accuracy=acc, n_voxels=nvox, centres=centres)
    trials.to_csv(opj(out, f"{sub}_trials{suffix}.csv"), index=False)
    print(f"-> {out}  ({(time.time() - t0) / 60:.1f} min in total)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "centres":
        make_centres()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
