#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 - Cache the cleaned MTL time series of both localizers, once per smoothing level.

For one participant: every localizer run is read ONCE. For each smoothing level
(none, 2 mm, 4 mm) the whole image is smoothed, the voxels of the anatomical MTL
mask (union with all reliability masks) are extracted and cleaned exactly as in
B4_train_classifiers.py (detrend, z-score, confound regression per run).
Smoothing is done on the whole image and cleaning is per voxel, so any subset of
the cached columns equals what B4 would extract with the corresponding mask.

Output (outputs/14_mtl_decoder/cache/):
    sXX_ts_smooth{0,2,4}.npy   float32, all TRs of all 26 runs stacked x voxels
    sXX_meta.npz               run_start, run_len, voxel flags (anat_mtl, anat_hpc,
                               anat_phc, anat_erh, rel10 ... rel90), voxel ijk
    sXX_trials.csv             one row per training trial (shown or recalled)

Usage:  python 01_cache_roi_timeseries.py sub-13
"""

import glob
import os
import sys
from os.path import join as opj

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking, signal

import mtl_config as cfg

cfg.add_pipeline_to_path()
from decoding_utils import CONFOUND_VARS, assert_func_conf_match, parse_subject_arg  # noqa: E402
import B4_train_classifiers as b4  # noqa: E402  (pure helpers only; nothing is written)


def load_behavior(sub, loc):
    """Same logic as B4.load_behavioral_data, without writing a file."""
    df = pd.read_csv(opj(cfg.PATH_BEHAVIOR, sub, "behav", "loc", f"{sub}_df_{loc}_w_HRF.csv"), sep=";")
    df["encoding_stimulus"] = b4._exclusive_merge(df, "obj_enc_stim", "sc_enc_stim")
    df["encoding_category"] = b4._exclusive_merge(df, "obj_cat", "sc_enc_stim")
    only_cat = df["encoding_stimulus"].isna() & df["encoding_category"].notna()
    df.loc[only_cat, "encoding_category"] = np.nan

    df["retrieval_category"] = np.nan
    df["retrieval_category"] = df["retrieval_category"].astype(object)
    m = df["label_type"] == "sc_ret_cue"
    df.loc[m, "retrieval_category"] = df.loc[m, "sc_ret_targ"]
    m = df["label_type"] == "obj_ret_cue"
    df.loc[m, "retrieval_category"] = df.loc[m, "obj_cat"]

    df["training_category"] = df["encoding_category"].fillna(df["retrieval_category"])
    df["trial_phase"] = np.nan
    df["trial_phase"] = df["trial_phase"].astype(object)
    df.loc[df["encoding_category"].notna(), "trial_phase"] = "encoding"
    df.loc[df["retrieval_category"].notna(), "trial_phase"] = "retrieval"
    return df.dropna(subset=["training_category"]).reset_index(drop=True)


def union_mask(paths):
    out, ref = None, None
    for p in paths:
        img = nib.load(p)
        d = np.asarray(img.dataobj) > 0
        out = d if out is None else (out | d)
        ref = img
    return out, ref


def main():
    sub = parse_subject_arg(sys.argv[1])
    num = sub[1:]
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    done = opj(cfg.CACHE_DIR, f"{sub}_meta.npz")
    if os.path.isfile(done):
        print("cache exists, nothing to do")
        return

    # ---------------- masks ----------------
    flags3d = {}
    ref_img = None
    for key, name in cfg.ANAT_ROIS.items():
        paths = []
        for _, ses in cfg.LOC_SESSIONS:
            paths += sorted(glob.glob(opj(
                cfg.PATH_ANAT_MASKS, f"sub-{num}", ses,
                f"{name}_sub-{num}_{ses}_space-T1w_T-{cfg.ANAT_SIGNAL_THRESHOLD}_run-*.nii")))
        if not paths:
            raise FileNotFoundError(f"no anatomical {name} masks for {sub}")
        flags3d[f"anat_{key}"], ref_img = union_mask(paths)
        print(f"anat_{key}: {len(paths)} run masks, {int(flags3d[f'anat_{key}'].sum())} voxels (union)")
    for thr in cfg.REL_THRESHOLDS:
        img = nib.load(opj(cfg.PATH_REL_MASKS, f"sub-{num}_mask_mtl_threshold_{thr}_combined.nii"))
        if not np.allclose(img.affine, ref_img.affine) or img.shape != ref_img.shape:
            raise ValueError("reliability and anatomical masks are on different grids")
        flags3d[f"rel{thr}"] = np.asarray(img.dataobj) > 0
        print(f"rel{thr}: {int(flags3d[f'rel{thr}'].sum())} voxels, "
              f"{int((flags3d[f'rel{thr}'] & ~flags3d['anat_mtl']).sum())} outside anat_mtl")

    union = np.zeros(ref_img.shape, bool)
    for v in flags3d.values():
        union |= v
    union_img = nib.Nifti1Image(union.astype(np.uint8), ref_img.affine)
    flags = {k: v[union] for k, v in flags3d.items()}      # same voxel order as apply_mask
    ijk = np.argwhere(union)
    print(f"union: {int(union.sum())} voxels")

    # ---------------- behaviour ----------------
    trials = []
    run_offset = 0
    n_runs_expected = []
    for loc, ses in cfg.LOC_SESSIONS:
        df = load_behavior(sub, loc)
        df["loc"] = loc
        df["run_in_ses"] = df["run_nr"].astype(int)
        df["run"] = df["run_in_ses"] + run_offset          # B4: loc2 runs are +13
        trials.append(df)
        n_runs_expected.append(df["run_in_ses"].nunique())
        run_offset += 13
    trials = pd.concat(trials, ignore_index=True)
    trials["kind"] = np.where(trials["training_category"].isin(b4.SCENE_CATEGORIES), "scene",
                              np.where(trials["training_category"].isin(b4.OBJECT_CATEGORIES),
                                       "object", "other"))

    # ---------------- functional data ----------------
    ts = {s: [] for s in cfg.SMOOTHING_LEVELS}
    run_ids, run_len = [], []
    for (loc, ses), n_expected in zip(cfg.LOC_SESSIONS, n_runs_expected):
        func_dir = opj(cfg.PATH_FMRIPREP, f"sub-{num}", ses, "func")
        func_paths = sorted(glob.glob(opj(
            func_dir, f"sub-{num}_{ses}_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz")),
            key=os.path.basename)
        conf_paths = sorted(glob.glob(opj(
            func_dir, f"sub-{num}_{ses}_task-*_run*_desc-confounds_timeseries.tsv")),
            key=os.path.basename)
        assert_func_conf_match(func_paths, conf_paths)
        print(f"{ses}: {len(func_paths)} runs on disk, {n_expected} runs in behaviour")
        # B4 pairs the i-th behavioural run (sorted) with the i-th image (sorted)
        beh_runs = sorted(trials.loc[trials["loc"] == loc, "run_in_ses"].unique())
        for i, (fp, cp) in enumerate(zip(func_paths, conf_paths)):
            if i >= len(beh_runs):
                print(f"  extra image without behaviour, skipped: {os.path.basename(fp)}")
                continue
            src = nib.load(fp)
            if not np.allclose(src.affine, ref_img.affine) or src.shape[:3] != ref_img.shape:
                raise ValueError(f"grid mismatch: {fp}")
            img = nib.Nifti1Image(np.asarray(src.dataobj, dtype=np.float32), src.affine)
            conf = pd.read_csv(cp, sep="\t")[CONFOUND_VARS].fillna(0)
            for s in cfg.SMOOTHING_LEVELS:
                masked = masking.apply_mask(img, union_img, smoothing_fwhm=(s if s else None))
                cleaned = signal.clean(signals=masked, t_r=cfg.TR, detrend=True,
                                       standardize="zscore_sample", confounds=conf)
                ts[s].append(cleaned.astype(np.float32))
            glob_run = beh_runs[i] + (0 if loc == "loc1" else 13)
            run_ids.append(glob_run)
            run_len.append(img.shape[3])
            print(f"  {os.path.basename(fp)} -> run {glob_run}, {img.shape[3]} TRs", flush=True)
            del img

    run_len = np.array(run_len)
    run_start = np.concatenate([[0], np.cumsum(run_len)[:-1]])
    for s in cfg.SMOOTHING_LEVELS:
        np.save(opj(cfg.CACHE_DIR, f"{sub}_ts_smooth{s}.npy"), np.vstack(ts[s]))

    # trial table: TR index inside the run, position of the run in the cache
    pos = {r: k for k, r in enumerate(run_ids)}
    trials = trials[trials["run"].isin(run_ids)].copy()
    trials["run_pos"] = trials["run"].map(pos)
    trials["tr"] = trials["TR_time_onset_w_HRF"].astype(int)
    trials["n_trs_run"] = run_len[trials["run_pos"].values]
    trials["valid"] = trials["tr"] < trials["n_trs_run"]      # B4.check_tr_alignment
    keep = ["loc", "run", "run_pos", "run_in_ses", "trl_nr", "label_type", "trial_phase", "kind",
            "training_category", "time_onset", "TR_time_onset", "tr", "n_trs_run", "valid"]
    trials[keep].to_csv(opj(cfg.CACHE_DIR, f"{sub}_trials.csv"), index=False)
    np.savez(done, run_ids=np.array(run_ids), run_start=run_start, run_len=run_len, ijk=ijk, **flags)
    print("done", sub, {s: np.vstack(ts[s]).shape for s in cfg.SMOOTHING_LEVELS})


if __name__ == "__main__":
    main()
