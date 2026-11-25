# -*- coding: utf-8 -*-
"""Whole-brain single-trial betas for encoding and retrieval, two estimators.

Events that get a beta
    encoding (evening, the 3 learning runs)
        scene      scene shown during scene learning (run 1)
        scenecue   scene shown alone before each object in the object-learning blocks
        object     object shown on its scene (3 presentations per object)
        dncue      day / night cue
    presleep / postsleep retrieval
        object, scene   one event per retrieval trial, from cue onset, 3 s

Estimators
    lss        Least squares separate (Mumford et al. 2012). One GLM per event:
               that event, one regressor per event kind for all other events,
               plus nuisance (cosine drift 1/128 Hz, 20 confounds, and for
               retrieval the response screen). Canonical SPM HRF. OLS.
    glmsingle  GLMsingle (Prince et al. 2022): HRF chosen per voxel from a
               library, GLMdenoise noise regressors, fractional ridge.
               Conditions are the stimuli (objects, scenes), so the repeats it
               needs for cross-validation come from: encoding - the repeated
               presentations; retrieval - evening and morning fitted TOGETHER
               (each object 3x: evening, morning pass 1, morning pass 2).
               Two outputs are kept: "glmsingle" (full model, type D) and
               "glmsingle_hrf" (fitted HRF only, no denoising, no ridge, type B).

    glmsingle_sep  The same, but evening and morning retrieval fitted SEPARATELY
               (outputs "glmsingle_sep", "glmsingle_sep_hrf"), so that session can
               enter a mixed model without any shared fitting. Note the asymmetry
               this creates: in the evening only the 10 scenes repeat (2x), in the
               morning objects repeat 2x and scenes 4x, so the number of noise
               regressors and the ridge strength are tuned on different amounts
               of data per session (both are printed in the log). Encoding is not
               refitted; 02 takes it from the "glmsingle" outputs.

Data: T1w space, 4 mm smoothing (as in the decoding pipeline), percent signal
change per run, all voxels inside the subject's brain mask (intersection of
the fMRIPrep masks of all used runs of both sessions).

Output
    outputs/04_betas/<estimator>/<sub>_<phase>_betas.npy    events x brain voxels, float16
    outputs/04_betas/<estimator>/<sub>_<phase>_trials.csv   one row per event
    outputs/04_betas/masks/<sub>_brainmask.nii.gz
    outputs/patterns_<estimator>/<phase>/<sub>_<phase>_<roi>_patterns.npy  events x 1 x ROI voxels
        (+ trials.csv) - same layout as 01, read by 02_binary_rsa.py

    lss_rt         LSS for retrieval only, with every event lasting from the cue to the
                   response (output "lss_rt"): activity per unit time during retrieval.

    lsa            Retrieval only: ONE model per run with one regressor per trial (per
                   object / scene), all estimated together (outputs "lsa" with the fixed
                   event duration and "lsa_rt" with events from cue to response).

Usage:  python 04_single_trial_betas.py sub-14 lss|lss_rt|lsa|glmsingle|glmsingle_sep
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import sys
from os.path import join as opj

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking
from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix

import rsa_config as cfg

_spec = importlib.util.spec_from_file_location("extract01", opj(cfg.HERE, "01_extract_trial_patterns.py"))
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
dec = ex.dec

from utils import (  # noqa: E402
    CONFOUND_VARS, assert_func_conf_match, build_sequence_dict_from_df_behavior_enc,
    load_mask, parse_subject_arg,
)

OUT_BETAS = opj(cfg.OUT_ROOT, "04_betas")
DUR = {"encoding": 2.5, "presleep": 3.0, "postsleep": 3.0}      # event duration in s
HIGH_PASS = 1.0 / 128
CUE_TO_PROMPT = 1.015       # s from the retrieval cue to the response prompt (fixed in the task)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def session_files(sub_mri: str, mri_session: str):
    func_dir = opj(cfg.PATH_FMRIPREP, sub_mri, mri_session, "func")
    funcs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_space-T1w_desc-preproc_bold.nii.gz")),
        key=os.path.basename)
    confs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv")),
        key=os.path.basename)
    assert_func_conf_match(funcs, confs)
    return funcs, confs


def encoding_files(sub_mri: str):
    """The 3 learning runs: first three of the six long (>120 TR) runs of ses-04."""
    funcs, confs = session_files(sub_mri, "ses-04")
    long = [i for i, f in enumerate(funcs) if nib.load(f).shape[-1] > 120]
    if len(long) != 6:
        raise ValueError(f"expected 6 long runs in ses-04, found {len(long)}")
    return [funcs[i] for i in long[:3]], [confs[i] for i in long[:3]]


def brain_mask(sub: str, func_paths: list[str]):
    path = opj(OUT_BETAS, "masks", f"{sub}_brainmask.nii.gz")
    if os.path.exists(path):
        return nib.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    masks = [f.replace("desc-preproc_bold", "desc-brain_mask") for f in func_paths]
    mask = masking.intersect_masks(masks, threshold=1, connected=False)
    tmp = path.replace(".nii.gz", f"_tmp{os.getpid()}.nii.gz")
    mask.to_filename(tmp)
    os.replace(tmp, path)
    return mask


def load_runs(func_paths, conf_paths, mask):
    """Smoothed, masked, percent-signal-change data and confounds per run."""
    data, confs = [], []
    for f, c in zip(func_paths, conf_paths):
        print(f"  loading {os.path.basename(f)}", flush=True)
        y = masking.apply_mask(image.smooth_img(f, cfg.SMOOTHING_FWHM), mask).astype(np.float32)
        m = y.mean(axis=0)
        m[m <= 0] = np.inf
        data.append((y / m - 1.0) * 100.0)
        confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
    return data, confs


# ---------------------------------------------------------------------------
# event tables
# ---------------------------------------------------------------------------

def encoding_events(df_enc: pd.DataFrame, hierarchy: dict, obj_enc_to_cat: dict):
    obj_info, scene_info = ex.hierarchy_lookup(hierarchy, obj_enc_to_cat)
    df = df_enc.copy()
    df["run_nr"] = df["run_nr"].ffill()
    df = df[df["t_onset"].notna() & df["label_type"].notna()]
    enc_runs = sorted(df.loc[df.label_type.isin(["dn+sc_cue", "dn+sc+obj_cue"]), "run_nr"].unique())
    if len(enc_runs) != 3:
        raise ValueError(f"expected 3 encoding runs in behaviour, found {enc_runs}")
    run_idx = {r: i for i, r in enumerate(enc_runs)}
    rows, nuis = [], []
    for _, r in df[df.run_nr.isin(enc_runs)].iterrows():
        lt, ri = r["label_type"], run_idx[r["run_nr"]]
        base = dict(run_idx=ri, onset_s=float(r["t_onset"]), object=None, category=None,
                    scene=None, context=None, position=np.nan, catseq=None, scene_position=np.nan)
        if lt == "dn_cue":
            base.update(kind="dncue", context="night" if "night" in str(r["dn_cue"]).lower() else "day")
        elif lt == "dn+sc_cue":
            sc = str(r["sc_enc_stim"])
            si = scene_info[sc]
            base.update(kind="scene" if ri == 0 else "scenecue", scene=sc, context=si["context"],
                        position=si["scene_enc_order"], catseq=si["catseq"],
                        scene_position=si["scene_enc_order"])
        elif lt == "dn+sc+obj_cue":
            ob = str(r["obj_enc_stim"])
            oi = obj_info[ob]
            si = scene_info[oi["scene"]]
            base.update(kind="object", object=ob, category=obj_enc_to_cat.get(ob), scene=oi["scene"],
                        context=oi["context"], position=oi["position"], catseq=si["catseq"],
                        scene_position=si["scene_enc_order"])
        elif lt in ("slider", "instr_txt"):
            nuis.append(dict(run_idx=ri, onset_s=float(r["t_onset"]), duration=3.0, name=lt))
            continue
        else:
            continue
        rows.append(base)
    ev = pd.DataFrame(rows)
    ev["condition"] = np.where(ev.kind == "object", ev.object,
                               np.where(ev.kind == "dncue", "dn_" + ev.context.astype(str),
                                        ev.kind + "_" + ev.scene.astype(str)))
    ev["rep"] = ev.groupby("condition").cumcount() + 1
    return ev, pd.DataFrame(nuis, columns=["run_idx", "onset_s", "duration", "name"])


def retrieval_events(sub: str, session: str, hierarchy: dict, obj_enc_to_cat: dict):
    scfg = dec.SESSION_CONFIG[session]
    _, df_ret = dec.load_main_task_behavior(cfg.ROOT_FOLDER, sub, scfg)
    df_ret = dec.prepare_retrieval_df(df_ret, obj_enc_to_cat)
    mapping = dec.build_run_nr_mapping(df_ret[df_ret.label_type == "sc_no4obj"],
                                       df_ret[df_ret.label_type == "d_no4sc"], scfg["n_retrieval_runs"])
    ev = ex.build_trial_table(df_ret, hierarchy, obj_enc_to_cat, mapping)
    if ev["cue_matches_hierarchy"].mean() < 0.9:
        raise ValueError(f"{sub} {session}: retrieval cues do not match the encoded hierarchy")
    ev["condition"] = np.where(ev.kind == "object", ev.object, "scene_" + ev.scene.astype(str))
    catch = df_ret[(df_ret.label_type == "catch_trl") & df_ret.t_onset.notna()]
    nuis = pd.DataFrame(dict(run_idx=catch.run_nr.astype(int).map(mapping).values,
                             onset_s=catch.t_onset.astype(float).values, duration=2.5, name="catch_trl"))
    return ev, nuis


def finalise_events(ev: pd.DataFrame, n_trs: list[int], phase: str, sub: str) -> pd.DataFrame:
    """Sort chronologically, add the onset TR, and check that events fit into their runs."""
    ev = ev.sort_values(["run_idx", "onset_s"]).reset_index(drop=True)
    ev["onset_tr"] = np.round(ev.onset_s / cfg.TR).astype(int)
    for ri, n in enumerate(n_trs):
        last = ev.loc[ev.run_idx == ri, "onset_s"].max()
        print(f"  run {ri}: {n} TRs = {n * cfg.TR:.0f} s, last event at {last:.0f} s")
        if last > n * cfg.TR:
            raise ValueError(f"{sub} {phase} run {ri}: events run past the end of the scan "
                             f"({last:.0f} s > {n * cfg.TR:.0f} s) - behaviour and fMRI runs do not match")
    ev.insert(0, "phase", phase)
    ev.insert(0, "subject", sub)
    return ev


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def hrf_regressor(onsets, durations, n_trs):
    frame_times = np.arange(n_trs) * cfg.TR
    cond = np.vstack([onsets, durations, np.ones(len(onsets))])
    return compute_regressor(cond, "spm", frame_times)[0][:, 0]


def nuisance_matrix(n_trs, conf, nuis_run):
    frame_times = np.arange(n_trs) * cfg.TR
    drift = make_first_level_design_matrix(frame_times, drift_model="cosine", high_pass=HIGH_PASS).values
    cols = [drift, conf]
    for _, g in nuis_run.groupby("name"):
        cols.append(hrf_regressor(g.onset_s.values, g.duration.values, n_trs)[:, None])
    return np.column_stack(cols)


def lss(data, confs, ev, nuis, dur, name="lss"):
    """Least squares separate. `dur` is one duration in seconds for all events, or one per event."""
    durs = np.broadcast_to(np.asarray(dur, float), (len(ev),))
    durs = pd.Series(durs, index=ev.index)
    betas = np.full((len(ev), data[0].shape[1]), np.nan, dtype=np.float32)
    for ri, (y, conf) in enumerate(zip(data, confs)):
        e = ev[ev.run_idx == ri]
        if e.empty:
            continue
        N = nuisance_matrix(y.shape[0], conf, nuis[nuis.run_idx == ri])
        X = np.column_stack([hrf_regressor([o], [d], y.shape[0])
                             for o, d in zip(e.onset_s.values, durs.loc[e.index].values)])
        P = np.linalg.pinv(N)
        yr = y - N @ (P @ y)                        # nuisance projected out of data ...
        xr = X - N @ (P @ X)                        # ... and out of every event regressor
        kinds = e.kind.values
        kind_list = sorted(set(kinds))
        ksum = np.column_stack([xr[:, kinds == k].sum(axis=1) for k in kind_list])
        for j, row in enumerate(e.index):
            others = ksum.copy()
            others[:, kind_list.index(kinds[j])] -= xr[:, j]
            D = np.column_stack([xr[:, j], others])
            D = D[:, np.abs(D).sum(axis=0) > 1e-8]   # a kind with no other event drops out
            betas[row] = np.linalg.lstsq(D, yr, rcond=None)[0][0]
        print(f"  LSS run {ri}: {len(e)} events", flush=True)
    return {name: betas}


def lsa(data, confs, ev, nuis, dur, name="lsa"):
    """Least squares all: ONE model per run with one regressor per event (per object / scene
    trial) plus the nuisance terms; all events are estimated together.
    `dur` is one duration in seconds for all events, or one per event."""
    durs = pd.Series(np.broadcast_to(np.asarray(dur, float), (len(ev),)), index=ev.index)
    betas = np.full((len(ev), data[0].shape[1]), np.nan, dtype=np.float32)
    for ri, (y, conf) in enumerate(zip(data, confs)):
        e = ev[ev.run_idx == ri]
        if e.empty:
            continue
        N = nuisance_matrix(y.shape[0], conf, nuis[nuis.run_idx == ri])
        X = np.column_stack([hrf_regressor([o], [d], y.shape[0])
                             for o, d in zip(e.onset_s.values, durs.loc[e.index].values)])
        D = np.column_stack([X, N])
        b = np.linalg.lstsq(D, y, rcond=None)[0]
        betas[e.index.values] = b[:X.shape[1]]
        print(f"  LSA run {ri}: {len(e)} event regressors + {N.shape[1]} nuisance, {y.shape[0]} TRs, "
              f"design rank {np.linalg.matrix_rank(D)} of {D.shape[1]}", flush=True)
    return {name: betas}


def glmsingle_fit(data, confs, ev, nuis, dur, session_ids, workdir, prefix="glmsingle"):
    sys.path.insert(0, opj(cfg.OUT_ROOT, "_pylibs"))
    from glmsingle.glmsingle import GLM_single

    conds = sorted(ev.condition.unique())
    cidx = {c: i for i, c in enumerate(conds)}
    design, extra = [], []
    for ri, y in enumerate(data):
        d = np.zeros((y.shape[0], len(conds)), dtype=int)
        e = ev[ev.run_idx == ri]
        if e.onset_tr.duplicated().any():
            raise ValueError("two events fall on the same TR")
        d[e.onset_tr.values, e.condition.map(cidx).values] = 1
        design.append(d)
        nr = nuis[nuis.run_idx == ri]
        extra.append(np.column_stack([hrf_regressor(g.onset_s.values, g.duration.values, y.shape[0])
                                      for _, g in nr.groupby("name")]) if len(nr) else None)
    opt = dict(wantlibrary=1, wantglmdenoise=1, wantfracridge=1,
               wantfileoutputs=[0, 0, 0, 0], wantmemoryoutputs=[0, 1, 0, 1],
               sessionindicator=np.asarray(session_ids)[None, :])
    if all(x is not None for x in extra):
        opt["extra_regressors"] = extra
    model = GLM_single(opt)
    # GLMsingle converts its betas to percent signal change by dividing by the voxel mean,
    # so it needs data with the mean still in: percent-signal-change data shifted to mean 100.
    res = model.fit(design, [y.T + 100.0 for y in data], dur, cfg.TR,
                    outputdir=opj(workdir, "out"), figuredir=opj(workdir, "fig"))
    n_vox = data[0].shape[1]
    out = {}
    for name, key in ((prefix, "typed"), (f"{prefix}_hrf", "typeb")):
        if key not in res:
            # GLMsingle drops denoising and ridge (type D) when no condition repeats across
            # runs, which happens in an evening-only fit when both scene trials share a run.
            print(f"  {name}: GLMsingle returned no {key} output (no repeats across runs) - not saved",
                  flush=True)
            continue
        b = np.asarray(res[key]["betasmd"]).reshape(n_vox, -1).T.astype(np.float32)
        assert b.shape[0] == len(ev), f"{b.shape[0]} betas for {len(ev)} events"
        out[name] = b
    if "typed" not in res:
        return out
    d = res["typed"]
    info = dict(pcnum=int(np.asarray(d["pcnum"]).squeeze()),
                frac_median=float(np.nanmedian(d["FRACvalue"])),
                r2_median=float(np.nanmedian(d["R2"])),
                hrf_index_mode=int(np.bincount(np.asarray(d["HRFindex"]).astype(int).ravel()).argmax()))
    print("  GLMsingle:", info, flush=True)
    return out


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def save(sub, phase, ev, betas_by_name, mask, roi_masks):
    for name, betas in betas_by_name.items():
        out = opj(OUT_BETAS, name)
        os.makedirs(out, exist_ok=True)
        np.save(opj(out, f"{sub}_{phase}_betas.npy"), betas.astype(np.float16))
        ev.to_csv(opj(out, f"{sub}_{phase}_trials.csv"), index=False)
        pdir = opj(cfg.OUT_ROOT, f"patterns_{name}", phase)
        os.makedirs(pdir, exist_ok=True)
        for roi, idx in roi_masks.items():
            np.save(opj(pdir, f"{sub}_{phase}_{roi}_patterns.npy"), betas[:, None, idx].astype(np.float16))
        ev.to_csv(opj(pdir, f"{sub}_{phase}_trials.csv"), index=False)
        print(f"  saved {name} {phase}: {betas.shape}", flush=True)


def main():
    sub = parse_subject_arg(sys.argv[1])
    method = sys.argv[2]
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} {method} ===", flush=True)

    _, obj_enc_to_cat = dec.load_localizer_behavior(cfg.ROOT_FOLDER, sub)
    df_enc, _ = dec.load_main_task_behavior(cfg.ROOT_FOLDER, sub, dec.SESSION_CONFIG["presleep"])
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)

    # ---- which phases are usable for this subject ----
    phases = {}
    try:
        f, c = encoding_files(sub_mri)
        ev, nuis = encoding_events(df_enc, hierarchy, obj_enc_to_cat)
        phases["encoding"] = dict(funcs=f, confs=c, ev=ev, nuis=nuis)
    except Exception as err:
        print(f"encoding not usable: {err}")
    for session in ("presleep", "postsleep"):
        try:
            scfg = dec.SESSION_CONFIG[session]
            ev, nuis = retrieval_events(sub, session, hierarchy, obj_enc_to_cat)
            f, c = ex.select_run_files(sub_mri, scfg["mri_session"], scfg)
            phases[session] = dict(funcs=f, confs=c, ev=ev, nuis=nuis)
        except Exception as err:
            print(f"{session} not usable: {err}")
    if not phases:
        raise RuntimeError("no usable phase")

    # brain mask over all long runs of both sessions (identical whichever phases are usable)
    all_funcs = []
    for ses in ("ses-04", "ses-06"):
        try:
            all_funcs += [f for f in session_files(sub_mri, ses)[0] if nib.load(f).shape[-1] > 120]
        except Exception:
            pass
    mask = brain_mask(sub, all_funcs)
    mask_flat = np.asarray(mask.dataobj).astype(bool)
    print(f"brain mask: {mask_flat.sum()} voxels")
    lin = np.full(mask_flat.shape, -1)
    lin[mask_flat] = np.arange(mask_flat.sum())
    roi_masks = {}
    for roi in cfg.ROIS:
        try:
            r = np.asarray(load_mask(cfg.PATH_MASKS, sub, roi, cfg.RELIABILITY_THRESHOLD).dataobj) > 0
        except FileNotFoundError:
            continue
        idx = lin[r]
        roi_masks[roi] = idx[idx >= 0]
        if (idx < 0).any():
            print(f"  {roi}: {(idx < 0).sum()} of {r.sum()} ROI voxels lie outside the brain mask")

    if method in ("glmsingle_sep", "lss_rt", "lsa"):
        phases.pop("encoding", None)
    for p in phases.values():
        p["data"], p["conf_arr"] = load_runs(p["funcs"], p["confs"], mask)
    for name in list(phases):
        p = phases[name]
        try:
            p["ev"] = finalise_events(p["ev"], [y.shape[0] for y in p["data"]], name, sub)
        except ValueError as err:               # drop this phase only, keep the others
            print(f"{name} not usable: {err}")
            del phases[name]
    if not phases:
        raise RuntimeError("no usable phase")

    if method == "lss":
        for name, p in phases.items():
            save(sub, name, p["ev"], lss(p["data"], p["conf_arr"], p["ev"], p["nuis"], DUR[name]),
                 mask, roi_masks)
    elif method == "lss_rt":
        # Retrieval only. Each event lasts from the cue to the response (cue -> prompt is
        # CUE_TO_PROMPT s, then the response time), so the beta is activity per unit time
        # during retrieval instead of the total response to a fixed 3 s event. Trials
        # without a response get the participant's median duration of that session.
        for name, p in phases.items():
            ev = p["ev"]
            d = CUE_TO_PROMPT + ev["rt"].astype(float)
            n_missing = int(d.isna().sum())
            ev["duration_s"] = d.fillna(d.median()).clip(lower=0.5)
            ev["duration_imputed"] = d.isna().values
            print(f"  {name}: duration median {ev.duration_s.median():.2f} s, "
                  f"range {ev.duration_s.min():.2f}-{ev.duration_s.max():.2f}, {n_missing} without response",
                  flush=True)
            save(sub, name, ev, lss(p["data"], p["conf_arr"], ev, p["nuis"], ev["duration_s"].values,
                                    name="lss_rt"), mask, roi_masks)
    elif method == "lsa":
        # Retrieval only, one model per run with a regressor per trial. Two versions:
        # "lsa" with the fixed event duration, "lsa_rt" with events lasting from cue to response.
        for name, p in phases.items():
            ev = p["ev"]
            d = CUE_TO_PROMPT + ev["rt"].astype(float)
            ev["duration_s"] = d.fillna(d.median()).clip(lower=0.5)
            ev["duration_imputed"] = d.isna().values
            save(sub, name, ev, lsa(p["data"], p["conf_arr"], ev, p["nuis"], DUR[name], name="lsa"),
                 mask, roi_masks)
            save(sub, name, ev, lsa(p["data"], p["conf_arr"], ev, p["nuis"], ev["duration_s"].values,
                                    name="lsa_rt"), mask, roi_masks)
    elif method == "glmsingle":
        workdir = opj(OUT_BETAS, "_glmsingle_work", sub)
        if "encoding" in phases:
            p = phases["encoding"]
            res = glmsingle_fit(p["data"], p["conf_arr"], p["ev"], p["nuis"], DUR["encoding"],
                                [1] * len(p["data"]), workdir)
            save(sub, "encoding", p["ev"], res, mask, roi_masks)
        ret = [s for s in ("presleep", "postsleep") if s in phases]
        if ret:
            # evening and morning retrieval are fitted together so that every stimulus has repeats
            data, confs, evs, nuiss, sess, offset = [], [], [], [], [], 0
            for si, s in enumerate(ret):
                p = phases[s]
                e, n = p["ev"].copy(), p["nuis"].copy()
                e["run_idx"] += offset
                n["run_idx"] += offset
                data += p["data"]; confs += p["conf_arr"]; evs.append(e); nuiss.append(n)
                sess += [si + 1] * len(p["data"])
                offset += len(p["data"])
            ev_all = pd.concat(evs, ignore_index=True)
            res = glmsingle_fit(data, confs, ev_all, pd.concat(nuiss, ignore_index=True),
                                DUR["presleep"], sess, workdir)
            start = 0
            for s in ret:
                n = len(phases[s]["ev"])
                save(sub, s, phases[s]["ev"], {k: v[start:start + n] for k, v in res.items()},
                     mask, roi_masks)
                start += n
        shutil.rmtree(workdir, ignore_errors=True)
    elif method == "glmsingle_sep":
        workdir = opj(OUT_BETAS, "_glmsingle_work", f"{sub}_sep")
        for s in ("presleep", "postsleep"):
            if s not in phases:
                continue
            p = phases[s]
            print(f"  separate GLMsingle fit: {s}", flush=True)
            res = glmsingle_fit(p["data"], p["conf_arr"], p["ev"], p["nuis"], DUR[s],
                                [1] * len(p["data"]), workdir, prefix="glmsingle_sep")
            save(sub, s, p["ev"], res, mask, roi_masks)
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        raise ValueError("method must be lss, lss_rt, lsa, glmsingle or glmsingle_sep")
    print("done", flush=True)


if __name__ == "__main__":
    main()
