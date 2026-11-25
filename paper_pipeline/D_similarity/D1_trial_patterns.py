#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 - One activity pattern per retrieval trial.

For one participant and both retrieval sessions (evening = presleep, morning =
postsleep): estimates the response to every retrieval trial with ONE model per
run that has one regressor per trial ("least squares all"), plus nuisance
regressors (cosine drifts slower than 128 s, the 20 fMRIPrep confounds of the
decoding steps, and the catch-trial response screen). Data: T1w space, smoothed
with 4 mm, percent signal change per run. Two event models:

    fixed            every trial is a 3 s event from the retrieval cue
    cue_to_response  every trial lasts from the cue to the response
                     (CUE_TO_PROMPT + reaction time; trials without a response
                     get the participant's median duration of that session)

Saved per event model
    patterns/<events>/<session>/<sub>_<session>_<roi>_patterns.npy   trials x voxels, float16,
                                                                     reliability-mask ROIs (B3)
    patterns/<events>/<session>/<sub>_<session>_trials.csv           one row per trial (same order)
    region_means/<events>/<sub>_<session>_region_means.csv           trial table + mean beta in the
                                                                     anatomical regions (B1, T-0)
and masks/<sub>_brainmask.nii.gz (intersection of the fMRIPrep brain masks of
all long runs of both sessions). Whole-brain betas are not saved.

The trial table carries: kind (object / scene), object, category, scene, context,
position (object position, or scene position for scene trials), scene_position,
catseq (category sequence A / B), run_idx, onset_s, correct (reported retrieval),
rt, duration_s, duration_imputed.

Input   outputs/A2_add_tr_columns_localizer/sXX/sXX_df_loc1_w_HRF.csv
        outputs/A1_add_tr_columns_main/sXX/*_nses_{0,2}_step*/output_df_subXX_w_HRF.csv
        outputs/B3_combine_localizer_masks/                       reliability masks (threshold 70)
        outputs/B1_threshold_roi_masks/thresholded_roi_output/    anatomical masks (T-0, ses-04, T1w)
        <derivatives>/sub-XX/ses-0{4,6}/func/                     BOLD (T1w), brain masks, confounds
Output  outputs/D1_trial_patterns/...

Usage:  python D1_trial_patterns.py <subject>          e.g. sub-14

Not a port of the original pipeline: added 2026-11-19, ported from
7_rsa_exploratory/04_single_trial_betas.py (method `lsa`) and
01_extract_trial_patterns.py (trial table). Behaviour loading, run selection and
run-number mapping are the functions of B7.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking
from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix
from nilearn.image import resample_to_img

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
from config import DERIVATIVES_DIR, TR, step_input_dir, step_output_dir
from decoding_utils import (
    CONFOUND_VARS, assert_func_conf_match, build_sequence_dict_from_df_behavior_enc,
    load_mask, parse_subject_arg,
)

SMOOTHING_FWHM = 4
RELIABILITY_THRESHOLD = 70
EVENT_DURATION = 3.0        # s, fixed event model
CUE_TO_PROMPT = 1.015       # s from the retrieval cue to the response prompt (fixed in the task)
HIGH_PASS = 1.0 / 128
SESSIONS = ("presleep", "postsleep")
EVENT_MODELS = ("fixed", "cue_to_response")

# reliability-mask ROIs (pattern analyses)
ROIS = ["vis", "mtl", "parietal_precuneus", "hpc", "parietal", "precuneus", "cortical_memory",
        "superior_frontal", "posterior_cingulate", "rsc", "medial_ofc", "rostral_acc", "caudal_acc"]
# anatomical regions (mean activity)
REGIONS = ["PARIETAL_PRECUNEUS", "PARIETAL", "PRECUNEUS", "HPC", "PHPC", "ERH", "MTL", "VIS", "CAUDAL_ACC",
           "RSC", "POSTERIOR_CINGULATE", "SUPERIOR_FRONTAL", "MEDIAL_OFC", "AUD", "SOMATO", "ventricles",
           "wholebrain"]


def _load_b7():
    """B7 as a module: validated behaviour loading, run selection, run-number mapping."""
    path = Path(__file__).resolve().parents[1] / "B_decoding" / "B7_apply_classifiers_retrieval.py"
    spec = importlib.util.spec_from_file_location("b7_retrieval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dec = _load_b7()


# ---------------------------------------------------------------------------
# trial table (as 7_rsa_exploratory/01_extract_trial_patterns.py)
# ---------------------------------------------------------------------------

def hierarchy_lookup(hierarchy: dict, obj_enc_to_cat: dict):
    """Per-object and per-scene labels from the encoded hierarchy."""
    obj_info, scene_info = {}, {}
    for ctx, scenes in hierarchy.items():
        for sc_i, (scene, objs) in enumerate(scenes.items()):
            cats = tuple(obj_enc_to_cat.get(o, "UNMAPPED") for o in objs)
            scene_info[scene] = dict(context=ctx, scene_enc_order=sc_i + 1, cats=cats)
            for pos, obj in enumerate(objs):
                obj_info[obj] = dict(context=ctx, scene=scene, position=pos + 1)
    # the two category sequences: label by their sorted category set
    catsets = sorted({tuple(sorted(v["cats"])) for v in scene_info.values()})
    if len(catsets) != 2:
        print(f"WARNING: expected 2 category sequences, found {len(catsets)}")
    for v in scene_info.values():
        v["catseq"] = "AB"[catsets.index(tuple(sorted(v["cats"])))] if len(catsets) <= 2 else "?"
    return obj_info, scene_info


def build_trial_table(df_ret: pd.DataFrame, hierarchy: dict, obj_enc_to_cat: dict,
                      run_nr_to_idx: dict) -> pd.DataFrame:
    """One row per retrieval trial with every label the models need."""
    obj_info, scene_info = hierarchy_lookup(hierarchy, obj_enc_to_cat)
    rows = []
    cue_rows = df_ret[df_ret["label_type"].isin(dec.RETRIEVAL_CUE_LABELS)]
    for _, r in cue_rows.iterrows():
        kind = "object" if r["label_type"] == "sc_no4obj" else "scene"
        cue_pos = int(str(r["seq_pos"])[1]) if pd.notna(r["seq_pos"]) else np.nan
        if kind == "object":
            obj = str(r["obj_ret_targ"])
            info = obj_info.get(obj, {})
            scene = info.get("scene")
            context = info.get("context")
            position = info.get("position", np.nan)
            cue_ok = (scene == str(r["ret_cue"])) and (position == cue_pos)
            category = obj_enc_to_cat.get(obj)
        else:
            obj, category = None, None
            scene = str(r["sc_ret_targ"])
            context = scene_info.get(scene, {}).get("context")
            position = cue_pos
            cue_ok = context is not None and context in str(r["ret_cue"]).lower()
        sinfo = scene_info.get(scene, {})
        run_nr = int(r["run_nr"])
        trial_rows = df_ret[df_ret["trl_nr"] == r["trl_nr"]]
        correct = trial_rows["catch_trial"].dropna()
        rows.append(dict(
            trial=int(r["trl_nr"]), run_nr=run_nr, run_idx=run_nr_to_idx.get(run_nr, -1),
            kind=kind, object=obj, category=category, scene=scene, context=context,
            position=position, catseq=sinfo.get("catseq"),
            scene_enc_order=sinfo.get("scene_enc_order", np.nan),
            cue_matches_hierarchy=bool(cue_ok),
            correct=correct.iloc[0] if len(correct) else np.nan,
            rt=r.get("ret_rt", np.nan),
            onset_s=float(r["t_onset"]), onset_tr=int(np.round(r["TR_onset_anchor"])),
        ))
    trials = pd.DataFrame(rows)
    # position of each scene within its context, taken from the scene-retrieval cues
    sc_pos = (trials[trials.kind == "scene"].groupby("scene")["position"]
              .agg(lambda x: x.mode().iloc[0]).to_dict())
    trials["scene_position"] = trials["scene"].map(sc_pos)
    return trials


# ---------------------------------------------------------------------------
# files and data
# ---------------------------------------------------------------------------

def session_files(sub_mri: str, mri_session: str):
    func_dir = opj(str(DERIVATIVES_DIR), sub_mri, mri_session, "func")
    funcs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_space-T1w_desc-preproc_bold.nii.gz")),
        key=os.path.basename)
    confs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv")),
        key=os.path.basename)
    if not funcs:
        raise FileNotFoundError(f"No T1w BOLD runs for {sub_mri} {mri_session}")
    assert_func_conf_match(funcs, confs)
    return funcs, confs


def select_run_files(sub_mri: str, mri_session: str, session_cfg: dict):
    """Pick the retrieval runs with B7's rule, from the file headers only."""
    funcs, confs = session_files(sub_mri, mri_session)
    # dummy arrays carry the file index through B7's selection logic unchanged
    dummies = [np.full((nib.load(f).shape[-1], 1), i) for i, f in enumerate(funcs)]
    chosen = [int(d[0, 0]) for d in dec.select_retrieval_runs(dummies, session_cfg)]
    return [funcs[i] for i in chosen], [confs[i] for i in chosen]


def brain_mask(sub: str, func_paths: list[str]):
    path = opj(str(step_output_dir(__file__, "masks")), f"{sub}_brainmask.nii.gz")
    if os.path.exists(path):
        return nib.load(path)
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
        y = masking.apply_mask(image.smooth_img(f, SMOOTHING_FWHM), mask).astype(np.float32)
        m = y.mean(axis=0)
        m[m <= 0] = np.inf
        data.append((y / m - 1.0) * 100.0)
        confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
    return data, confs


def retrieval_events(sub: str, session: str, hierarchy: dict, obj_enc_to_cat: dict, path_main: str):
    scfg = dec.SESSION_CONFIG[session]
    _, df_ret = dec.load_main_task_behavior(path_main, sub, scfg)
    df_ret = dec.prepare_retrieval_df(df_ret, obj_enc_to_cat)
    mapping = dec.build_run_nr_mapping(df_ret[df_ret.label_type == "sc_no4obj"],
                                       df_ret[df_ret.label_type == "d_no4sc"], scfg["n_retrieval_runs"])
    ev = build_trial_table(df_ret, hierarchy, obj_enc_to_cat, mapping)
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
    ev["onset_tr"] = np.round(ev.onset_s / TR).astype(int)
    for ri, n in enumerate(n_trs):
        last = ev.loc[ev.run_idx == ri, "onset_s"].max()
        print(f"  run {ri}: {n} TRs = {n * TR:.0f} s, last event at {last:.0f} s")
        if last > n * TR:
            raise ValueError(f"{sub} {phase} run {ri}: events run past the end of the scan "
                             f"({last:.0f} s > {n * TR:.0f} s) - behaviour and fMRI runs do not match")
    ev.insert(0, "phase", phase)
    ev.insert(0, "subject", sub)
    return ev


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def hrf_regressor(onsets, durations, n_trs):
    frame_times = np.arange(n_trs) * TR
    cond = np.vstack([onsets, durations, np.ones(len(onsets))])
    return compute_regressor(cond, "spm", frame_times)[0][:, 0]


def nuisance_matrix(n_trs, conf, nuis_run):
    frame_times = np.arange(n_trs) * TR
    drift = make_first_level_design_matrix(frame_times, drift_model="cosine", high_pass=HIGH_PASS).values
    cols = [drift, conf]
    for _, g in nuis_run.groupby("name"):
        cols.append(hrf_regressor(g.onset_s.values, g.duration.values, n_trs)[:, None])
    return np.column_stack(cols)


def lsa(data, confs, ev, nuis, dur):
    """One model per run with one regressor per trial plus the nuisance terms.
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
        print(f"  run {ri}: {len(e)} trial regressors + {N.shape[1]} nuisance, {y.shape[0]} TRs, "
              f"design rank {np.linalg.matrix_rank(D)} of {D.shape[1]}", flush=True)
    return betas


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def reliability_rois(sub, mask_flat, lin):
    path_masks = str(step_input_dir("B3_combine_localizer_masks"))
    out = {}
    for roi in ROIS:
        try:
            r = np.asarray(load_mask(path_masks, sub, roi, RELIABILITY_THRESHOLD).dataobj) > 0
        except FileNotFoundError:
            continue
        idx = lin[r]
        out[roi] = idx[idx >= 0]
        if (idx < 0).any():
            print(f"  {roi}: {(idx < 0).sum()} of {r.sum()} ROI voxels lie outside the brain mask")
    return out


def anatomical_regions(sub_mri, brain_img, bm):
    base = opj(str(step_input_dir("B1_threshold_roi_masks")), "thresholded_roi_output", sub_mri, "ses-04")
    out = {}
    for roi in REGIONS:
        hits = sorted(glob.glob(opj(base, f"{roi}_{sub_mri}_ses-04_space-T1w_T-0_run-*.nii")))
        if not hits:
            continue
        img = nib.load(hits[0])
        if img.shape != brain_img.shape or not np.allclose(img.affine, brain_img.affine, atol=1e-3):
            img = resample_to_img(img, brain_img, interpolation="nearest")
        m = (img.get_fdata() > 0)[bm]                 # in the order of the beta columns
        if m.sum() >= 10:
            out[roi] = m
    return out


# ---------------------------------------------------------------------------

def main():
    sub = parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None)
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    path_loc = str(step_input_dir("A2_add_tr_columns_localizer"))
    path_main = str(step_input_dir("A1_add_tr_columns_main"))

    _, obj_enc_to_cat = dec.load_localizer_behavior(path_loc, sub)
    df_enc, _ = dec.load_main_task_behavior(path_main, sub, dec.SESSION_CONFIG["presleep"])
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)

    phases = {}
    for session in SESSIONS:
        try:
            scfg = dec.SESSION_CONFIG[session]
            ev, nuis = retrieval_events(sub, session, hierarchy, obj_enc_to_cat, path_main)
            f, c = select_run_files(sub_mri, scfg["mri_session"], scfg)
            phases[session] = dict(funcs=f, confs=c, ev=ev, nuis=nuis)
        except Exception as err:
            print(f"{session} not usable: {err}")
    if not phases:
        raise RuntimeError("no usable session")

    # brain mask over all long runs of both sessions (identical whichever sessions are usable)
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
    rois = reliability_rois(sub, mask_flat, lin)
    regions = anatomical_regions(sub_mri, mask, mask_flat)
    print(f"{len(rois)} reliability-mask ROIs, {len(regions)} anatomical regions")

    for name in list(phases):
        p = phases[name]
        p["data"], p["conf_arr"] = load_runs(p["funcs"], p["confs"], mask)
        try:
            p["ev"] = finalise_events(p["ev"], [y.shape[0] for y in p["data"]], name, sub)
        except ValueError as err:               # drop this session only, keep the other
            print(f"{name} not usable: {err}")
            del phases[name]
    if not phases:
        raise RuntimeError("no usable session")

    for name, p in phases.items():
        ev = p["ev"]
        d = CUE_TO_PROMPT + ev["rt"].astype(float)
        ev["duration_s"] = d.fillna(d.median()).clip(lower=0.5)
        ev["duration_imputed"] = d.isna().values
        print(f"  {name}: duration median {ev.duration_s.median():.2f} s, {int(d.isna().sum())} without response")
        for events, dur in (("fixed", EVENT_DURATION), ("cue_to_response", ev["duration_s"].values)):
            betas = lsa(p["data"], p["conf_arr"], ev, p["nuis"], dur)
            pdir = str(step_output_dir(__file__, opj("patterns", events, name)))
            for roi, idx in rois.items():
                np.save(opj(pdir, f"{sub}_{name}_{roi}_patterns.npy"), betas[:, idx].astype(np.float16))
            ev.to_csv(opj(pdir, f"{sub}_{name}_trials.csv"), index=False)
            rm = ev.copy()
            for roi, m in regions.items():
                rm[roi] = np.nanmean(betas[:, m], axis=1)
            rdir = str(step_output_dir(__file__, opj("region_means", events)))
            rm.to_csv(opj(rdir, f"{sub}_{name}_region_means.csv"), index=False)
            print(f"  saved {events} {name}: {betas.shape[0]} trials", flush=True)
    pd.DataFrame([dict(subject=sub, **{r: int(m.sum()) for r, m in regions.items()})]).to_csv(
        opj(str(step_output_dir(__file__, "region_voxel_counts")), f"{sub}_region_voxel_counts.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
