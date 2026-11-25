# -*- coding: utf-8 -*-
"""Single-trial voxel patterns for every retrieval trial, per ROI.

For one subject and one session this script
  1. loads behaviour and the encoded hierarchy with the functions of 4.4
     (same folder choice, same run selection, same run-number mapping),
  2. loads the selected retrieval runs in T1w space (the ROI masks are in
     T1w space), smooths them once (4 mm), and per ROI masks and cleans them
     exactly as the decoding pipeline does (detrend, z-score, 20 confounds),
  3. cuts out TRs 0..9 after the cue of every retrieval trial.

Output (outputs/01_trial_patterns/<session>/):
  <sub>_<session>_trials.csv              one row per retrieval trial
  <sub>_<session>_<roi>_patterns.npy      float16, trials x TRs x voxels
The row order of the csv is the first axis of the pattern arrays.

Usage:  python 01_extract_trial_patterns.py sub-14 presleep [roi ...]
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking, signal

import rsa_config as cfg

dec = cfg.import_decoding_module()
from utils import (  # noqa: E402  (4_decoding_claude/utils.py, put on sys.path by 4.4)
    CONFOUND_VARS,
    assert_func_conf_match,
    build_sequence_dict_from_df_behavior_enc,
    load_mask,
    parse_subject_arg,
)


def hierarchy_lookup(hierarchy: dict, obj_enc_to_cat: dict):
    """Per-object and per-scene labels from the encoded hierarchy."""
    obj_info, scene_info = {}, {}
    for ctx, scenes in hierarchy.items():
        for sc_i, (scene, objs) in enumerate(scenes.items()):
            cats = tuple(obj_enc_to_cat.get(o, "UNMAPPED") for o in objs)
            scene_info[scene] = dict(context=ctx, scene_enc_order=sc_i + 1, cats=cats)
            for pos, obj in enumerate(objs):
                obj_info[obj] = dict(context=ctx, scene=scene, position=pos + 1)

    # the two category sequences (A-E / F-J): label by their sorted category set
    catsets = sorted({tuple(sorted(v["cats"])) for v in scene_info.values()})
    if len(catsets) != 2:
        print(f"WARNING: expected 2 category sequences, found {len(catsets)}")
    for v in scene_info.values():
        v["catseq"] = "AB"[catsets.index(tuple(sorted(v["cats"])))] if len(catsets) <= 2 else "?"
    return obj_info, scene_info


def build_trial_table(df_ret: pd.DataFrame, hierarchy: dict, obj_enc_to_cat: dict,
                      run_nr_to_idx: dict) -> pd.DataFrame:
    """One row per retrieval trial with every label the model RDMs need."""
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


def select_run_files(sub_mri: str, mri_session: str, session_cfg: dict):
    """Pick the retrieval runs with 4.4's rule, from the file headers only."""
    func_dir = opj(cfg.PATH_FMRIPREP, sub_mri, mri_session, "func")
    funcs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_space-T1w_desc-preproc_bold.nii.gz")),
        key=os.path.basename)
    confs = sorted(glob.glob(opj(
        func_dir, f"{sub_mri}_{mri_session}_task-mem*_run*_desc-confounds_timeseries.tsv")),
        key=os.path.basename)
    if not funcs:
        raise FileNotFoundError(f"No T1w BOLD runs for {sub_mri} {mri_session}")
    assert_func_conf_match(funcs, confs)
    # dummy arrays carry the file index through 4.4's selection logic unchanged
    dummies = [np.full((nib.load(f).shape[-1], 1), i) for i, f in enumerate(funcs)]
    chosen = [int(d[0, 0]) for d in dec.select_retrieval_runs(dummies, session_cfg)]
    return [funcs[i] for i in chosen], [confs[i] for i in chosen]


def main():
    sub = parse_subject_arg(sys.argv[1])
    session = sys.argv[2]
    rois = sys.argv[3:] or cfg.ROIS
    session_cfg = dec.SESSION_CONFIG[session]
    sub_mri = f"sub-{sub[1:]}"
    out_dir = opj(cfg.OUT_ROOT, "01_trial_patterns", session)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {sub} {session}: {len(rois)} ROIs ===")

    # ---- behaviour and hierarchy (4.4's loaders) ----
    _, obj_enc_to_cat = dec.load_localizer_behavior(cfg.ROOT_FOLDER, sub)
    df_enc, df_ret = dec.load_main_task_behavior(cfg.ROOT_FOLDER, sub, session_cfg)
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)
    df_ret = dec.prepare_retrieval_df(df_ret, obj_enc_to_cat)
    df_obj = df_ret[df_ret["label_type"] == "sc_no4obj"]
    df_sc = df_ret[df_ret["label_type"] == "d_no4sc"]
    run_nr_to_idx = dec.build_run_nr_mapping(df_obj, df_sc, session_cfg["n_retrieval_runs"])

    trials = build_trial_table(df_ret, hierarchy, obj_enc_to_cat, run_nr_to_idx)
    trials.insert(0, "session", session)
    trials.insert(0, "subject", sub)
    match = trials["cue_matches_hierarchy"].mean()
    print(f"{len(trials)} retrieval trials; cue matches the encoded hierarchy in {match:.1%}")
    if match < 0.9:
        raise ValueError(
            f"{sub} {session}: retrieval cues do not match the encoded hierarchy "
            f"({match:.1%}). This session was probably recorded under another "
            f"subject's stimulus set (known for s41 postsleep). Not extracting.")

    # ---- fMRI: selected retrieval runs, smoothed once ----
    func_paths, conf_paths = select_run_files(sub_mri, session_cfg["mri_session"], session_cfg)
    masks = {}
    for roi in rois:
        try:
            masks[roi] = load_mask(cfg.PATH_MASKS, sub, roi, cfg.RELIABILITY_THRESHOLD)
        except FileNotFoundError:
            print(f"  no mask for {roi}, skipped")
    cleaned = {roi: [] for roi in masks}
    for f, c in zip(func_paths, conf_paths):
        print(f"  loading {os.path.basename(f)}")
        img = image.smooth_img(f, cfg.SMOOTHING_FWHM)
        conf = pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0)
        for roi, mask in masks.items():
            x = masking.apply_mask(img, mask)
            cleaned[roi].append(signal.clean(
                x, t_r=cfg.TR, detrend=True, standardize="zscore_sample", confounds=conf))
        del img

    # ---- cut out the trial windows ----
    win = np.asarray(cfg.TR_WINDOW)
    n_trs_run = [r.shape[0] for r in next(iter(cleaned.values()))]
    trials["n_valid_trs"] = [
        int(((t.onset_tr + win >= 0) & (t.onset_tr + win < n_trs_run[t.run_idx])).sum())
        for t in trials.itertuples()]
    for roi, runs in cleaned.items():
        pat = np.full((len(trials), len(win), runs[0].shape[1]), np.nan, dtype=np.float32)
        for i, t in enumerate(trials.itertuples()):
            idx = t.onset_tr + win
            ok = (idx >= 0) & (idx < runs[t.run_idx].shape[0])
            pat[i, ok] = runs[t.run_idx][idx[ok]]
        np.save(opj(out_dir, f"{sub}_{session}_{roi}_patterns.npy"), pat.astype(np.float16))
        print(f"  {roi}: {pat.shape}")
    trials.to_csv(opj(out_dir, f"{sub}_{session}_trials.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
