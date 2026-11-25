#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 - Whole-brain retrieval activity per participant, in standard space.

E1 compares mean activity in regions chosen in advance. This step gives the
picture without that choice: for one participant it fits one model per
retrieval run on the fMRIPrep data in MNI152NLin2009cAsym space (2 mm, smoothed
with 4 mm as in D1, percent signal change per run) with ONE regressor for all object
trials and ONE for all scene trials of the run, plus the nuisance regressors of
D1 (cosine drifts slower than 128 s, the 20 fMRIPrep confounds, the catch-trial
response screen). Run estimates are averaged per time point:

    pre     evening retrieval (ses-04, the three retrieval runs)
    post1   first morning retrieval  (ses-06, retrieval runs 1-3)
    post2   repeated morning retrieval (ses-06, retrieval runs 4-6)

Three models, as in D1 / E1:
    fixed            every trial is a 3 s event from the retrieval cue
    cue_to_response  every trial lasts from the cue to the response
    fixed_rt         as `fixed`, plus one regressor per trial type whose height is
                     the trial's reaction time (centred within the run; trials
                     without a response get 0). Its estimate is the change in
                     activity per second of reaction time.

    fixed_fixbase    as `fixed`, but with the fixation cross before each trial as the
                     ONLY unmodelled time, so estimates are relative to fixation.
                     A retrieval trial runs: fixation (3-7 s, jittered) -> trial-type
                     cue (about 2 s) -> retrieval cue -> 1.015 s -> question mark;
                     button press if retrieved; 11-13 s after the question mark the
                     screen for the spoken answer (about 2.6 s; only if the button was
                     pressed) -> next fixation. Three extra regressors cover everything
                     but the fixation: `ordercue` (trial-type cue until the retrieval
                     cue), `wait` (end of the 3 s retrieval event until the speaking
                     screen, or until the next fixation when there is none) and `speak`
                     (speaking screen until the next fixation). The trial-type cue
                     always precedes the retrieval cue by about 2 s, so `ordercue` and
                     the retrieval regressors are correlated; the log prints by how much.
                     The estimates of the three phases are saved as well
                     (<sub>_<time point>_phase_ordercue / _phase_wait / _phase_speak).

    response_fixbase The same idea with the retrieval event
                     ending at the button press, and the preparatory trial-type cue split
                     by trial type (`ordercue_object`, `ordercue_scene`):
                     object / scene regressors hold the trials WITH a button press, each
                     lasting from the retrieval cue to the press (retrieval until the item
                     is found); `wait` runs from the press to the speaking screen (holding
                     the answer); trials WITHOUT a press get their own regressor
                     `noretrieval`, from the cue to the timeout (next fixation). Fixation
                     is again the only unmodelled time. Phases are saved as above.

    trial_fixbase    as `response_fixbase`, but the trial-type cue has NO regressor of its
                     own: the retrieval event starts at the trial-type cue and ends at the
                     button press (`noretrieval`: trial-type cue until the timeout). The
                     trial-type cue precedes the retrieval cue by a fixed 2 s, so a model
                     with both regressors divides the response between them by the assumed
                     shape of the haemodynamic response only (in `response_fixbase` the
                     response to the scene picture of the retrieval cue largely lands on
                     `ordercue`). Here the estimate is the whole trial up to the press
                     against fixation.

    cue_search_fixbase
                     Main model from 2026-11-22 until 2026-11-23; replaced by `ret_press`.
                     As `response_fixbase`, with the retrieval event divided in two per
                     trial type: `<type>_cue`, the 1.015 s in which the retrieval cue is on
                     screen (all trials), and `<type>_search`, from the question mark to
                     the button press (trials with a press; its length is the reaction
                     time). `noretrieval` then runs from the question mark to the timeout.
                     The cue is always followed by the question mark after 1.015 s, but the
                     search phase varies in length from trial to trial, which is what
                     separates the two.
    fixed_fixbase_rt, response_fixbase_rt
                     the two fixation-baseline models with one more regressor per trial
                     type whose height is the trial's reaction time (centred within run and
                     trial type), as in `fixed_rt`. With the 3 s event it asks whether
                     activity grows with the time until the press; with the event that ends
                     at the press it asks whether anything remains once that duration is in
                     the model. Only the reaction-time estimates are saved.

    ret_press, ret_rtmod, ret_rtdur
                     The models decided on 2026-11-23. The trial-type cue (the word that says
                     whether an object or a scene will be asked) has NO regressor; it is part of
                     the reference together with the fixation cross. The retrieval cue and the
                     search are ONE event, for trials with a button press. `wait`, `speak` and
                     `noretrieval` (trials without a press: cue until the timeout) are modelled
                     as before.
                       ret_press   MAIN MODEL (decided 2026-11-23). The event lasts from the onset of the retrieval cue to the
                                   button press (1.015 s + reaction time). "RTDur" in Mumford
                                   et al. 2024 (Nat Hum Behav, The response time paradox).
                       ret_rtmod   the event has the same length in every trial (RT_CONSTANT_DUR,
                                   the 1.015 s the cue is on screen), plus ONE regressor `rtslope`
                                   over object and scene trials with the same onsets and length
                                   whose height is the reaction time minus RT_REFERENCE (1.5 s,
                                   the SAME constant for every participant and run; centring on a
                                   participant's or a run's mean is not acceptable according to
                                   the paper). "ConsDurRTMod": the object / scene estimates are
                                   the activity of a trial with a reaction time of 1.5 s, at
                                   every time point; `rtslope` is the change per second. The
                                   paper finds this form inadequate for reaction times above
                                   about 2 s (26 % of the trials here).
                       ret_rtdur   the form the paper recommends ("ConsDurRTDur"): the same
                                   constant event per trial type plus ONE regressor `rtdur` over
                                   object and scene trials that lasts from the cue to the press.
                                   The object / scene estimates then refer to a trial with a
                                   reaction time of 0 (limited interpretation against baseline,
                                   as the paper states); `rtdur` is the activity per unit of
                                   time until the press.
                     For these models the estimates of `object`, `rtslope` and `rtdur` are also
                     saved for every run: <sub>_<time point>R<1-3>_<name>.nii.gz.

Trials, run selection, onsets and durations are those of D1 (same functions).

Input   outputs/A1_add_tr_columns_main, outputs/A2_add_tr_columns_localizer
        <derivatives>/sub-XX/ses-0{4,6}/func/   BOLD in MNI space, brain masks, confounds
Output  outputs/E3_wholebrain_maps/maps/<model>/<sub>_<time point>_<object|scene>[_per_second_rt].nii.gz
        outputs/E3_wholebrain_maps/masks/<sub>_brainmask_mni.nii.gz
        outputs/E3_wholebrain_maps/runs/<sub>_runs.csv        runs and trial counts that went in

All retrieval trials enter, whether or not retrieval was indicated by button press.
A version restricted to trials with a button press gave the same group maps (t maps
correlate at .99 across voxels, 95 % of significant voxels shared; checked 2026-11-22)
and was removed again to keep the number of variants down.

Usage:  python E3_wholebrain_maps.py <subject>          e.g. sub-14

Environment (optional): E3_MODELS=a,b restricts the models fitted; E3_ONLY_NEW=1 leaves
existing maps untouched; E3_SPACE=T1w fits in the participant's own anatomy instead of
standard space and writes to maps_T1w/ and masks_T1w/ (used by E8 to check that fitting
first and transforming afterwards gives the same result).

Not a port of the original pipeline: added 2026-11-22.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "D_similarity"))
from config import step_input_dir, step_output_dir
from decoding_utils import CONFOUND_VARS, build_sequence_dict_from_df_behavior_enc, parse_subject_arg

SMOOTHING_FWHM = 4
SPACE = "MNI152NLin2009cAsym"
KINDS = ("object", "scene")
MODELS = ("fixed", "cue_to_response", "fixed_rt", "fixed_fixbase", "response_fixbase", "trial_fixbase",
          "cue_search_fixbase",
          "fixed_fixbase_rt", "response_fixbase_rt", "ret_press", "ret_rtmod", "ret_rtdur")
NEW_MODELS = ("ret_press", "ret_rtmod", "ret_rtdur")   # no regressor for the trial-type cue; also saved per run
RT_CONSTANT_DUR = 1.015                       # s, constant event of `ret_rtmod` / `ret_rtdur` (cue on screen)
RT_REFERENCE = 1.5                            # s, reaction time the estimates of `ret_rtmod` refer to


def _load_d1():
    spec = importlib.util.spec_from_file_location(
        "d1", Path(__file__).resolve().parents[1] / "D_similarity" / "D1_trial_patterns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d1 = _load_d1()
dec = d1.dec


NATIVE = os.environ.get("E3_SPACE", "MNI") == "T1w"
MAP_DIR, MASK_DIR = ("maps_T1w", "masks_T1w") if NATIVE else ("maps", "masks")


def mni_path(t1w_path: str) -> str:
    if NATIVE:
        return t1w_path
    p = t1w_path.replace("space-T1w", f"space-{SPACE}")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def amplitude_regressor(onsets, durations, amplitudes, n_trs):
    from nilearn.glm.first_level import compute_regressor
    cond = np.vstack([onsets, durations, amplitudes])
    return compute_regressor(cond, "spm", np.arange(n_trs) * d1.TR)[0][:, 0]


def phase_table(sub, session, path_main, obj_enc_to_cat):
    """Onsets (s, within run) of every phase of every retrieval trial, for model `fixed_fixbase`."""
    scfg = dec.SESSION_CONFIG[session]
    _, df_ret = dec.load_main_task_behavior(path_main, sub, scfg)
    df_ret = dec.prepare_retrieval_df(df_ret, obj_enc_to_cat)
    mapping = dec.build_run_nr_mapping(df_ret[df_ret.label_type == "sc_no4obj"],
                                       df_ret[df_ret.label_type == "d_no4sc"], scfg["n_retrieval_runs"])
    d = df_ret[df_ret.t_onset.notna()]
    w = d.pivot_table(index="trl_nr", columns="label_type", values="t_onset", aggfunc="first")
    col = lambda a, b: w.get(a, pd.Series(np.nan, index=w.index)).fillna(w.get(b, pd.Series(np.nan, index=w.index)))
    ph = pd.DataFrame(dict(fix=w.get("ret_fix"), ordercue=col("obj_order_cue", "sc_order_cue"),
                           cue=col("sc_no4obj", "d_no4sc"), speak=w.get("catch_trl")))
    ph["object_trial"] = w.get("obj_order_cue", pd.Series(np.nan, index=w.index)).notna()
    ph["run_idx"] = d.groupby("trl_nr").run_nr.first().astype(int).map(mapping)
    ph = ph[ph.cue.notna() & ph.run_idx.notna()].sort_values(["run_idx", "cue"])
    ph["next_fix"] = ph.groupby("run_idx").fix.shift(-1)
    return ph


def phase_regressors(ph_run, n, press=None, merge_ordercue=False, split_ordercue=False, noret_from_prompt=False,
                     no_ordercue=False):
    """ordercue, wait and speak regressors of one run (columns), and their names.
    `press`: time of the button press per trial (NaN = none); then `wait` starts at the press,
    and trials without a press form the regressor `noretrieval` (cue until the timeout)."""
    cue, end = ph_run.cue.values, ph_run.next_fix.values
    speak = ph_run.speak.values
    wait_on = cue + d1.EVENT_DURATION if press is None else press
    # last trial of a run has no following fixation: assume the usual lengths
    wait_end = np.where(np.isfinite(speak), speak, np.where(np.isfinite(end), end, cue + d1.CUE_TO_PROMPT + 12.0))
    speak_end = np.where(np.isfinite(end), end, speak + 2.65)
    cols, names = [], []
    oc = ph_run.ordercue.values
    ok = np.isfinite(oc) & (cue > oc)
    start = np.where(ok, oc, cue) if merge_ordercue else cue      # where a trial without a press starts
    if noret_from_prompt:
        start = cue + d1.CUE_TO_PROMPT
    if no_ordercue:                                   # the trial-type cue stays in the reference
        pass
    elif split_ordercue:                              # one preparatory regressor per trial type
        obj = ph_run.object_trial.values.astype(bool)
        for sel, nm in ((ok & obj, "ordercue_object"), (ok & ~obj, "ordercue_scene")):
            if sel.any():
                cols.append(d1.hrf_regressor(oc[sel], (cue - oc)[sel], n)); names.append(nm)
    elif ok.any() and not merge_ordercue:
        cols.append(d1.hrf_regressor(oc[ok], (cue - oc)[ok], n)); names.append("ordercue")
    ok = np.isfinite(wait_on) & (wait_end > wait_on)
    cols.append(d1.hrf_regressor(wait_on[ok], (wait_end - wait_on)[ok], n)); names.append("wait")
    if press is not None:
        ok = ~np.isfinite(press) & (wait_end > start)
        if ok.any():
            cols.append(d1.hrf_regressor(start[ok], (wait_end - start)[ok], n)); names.append("noretrieval")
    ok = np.isfinite(speak) & (speak_end > speak)
    if ok.any():
        cols.append(d1.hrf_regressor(speak[ok], (speak_end - speak)[ok], n)); names.append("speak")
    return cols, names


def fit_run(y, conf, e, nuis_run, model, ph_run=None, rt_ref=None):
    """Betas (rows) for the regressors of interest of one run; names in the same order."""
    n = y.shape[0]
    rt_mod = model.endswith("_rt")                    # add reaction time as a regressor
    model = model[:-3] if rt_mod else model
    dur_col = "duration_s" if model == "cue_to_response" else None
    cols, names = [], []
    for kind in KINDS:
        k = e[e.kind == kind]
        if model == "cue_search_fixbase":             # cue on screen (all trials) and search until the press
            if k.empty:
                continue
            cols.append(d1.hrf_regressor(k.onset_s.values, np.full(len(k), d1.CUE_TO_PROMPT), n))
            names.append(f"{kind}_cue")
            kp = k[k.rt.notna()]
            if len(kp):
                cols.append(d1.hrf_regressor(kp.onset_s.values + d1.CUE_TO_PROMPT,
                                             kp.rt.astype(float).clip(lower=0.2).values, n))
                names.append(f"{kind}_search")
            continue
        if model in ("response_fixbase", "trial_fixbase") + NEW_MODELS:
            k = k[k.rt.notna()]                       # trials with a button press; cue until the press
        if k.empty:
            continue
        dur = k[dur_col].values if dur_col else np.full(len(k), d1.EVENT_DURATION)
        onsets = k.onset_s.values
        if model in ("response_fixbase", "trial_fixbase", "ret_press"):
            dur = (d1.CUE_TO_PROMPT + k.rt.astype(float)).clip(lower=0.5).values
        if model in ("ret_rtmod", "ret_rtdur"):
            dur = np.full(len(k), RT_CONSTANT_DUR)
        if model == "trial_fixbase":                  # start at the trial-type cue
            oc = pd.Series(ph_run.ordercue.values, index=np.round(ph_run.cue.values, 2))
            oc = np.round(k.onset_s, 2).map(oc[~oc.index.duplicated()]).values
            lead = np.where(np.isfinite(oc) & (oc < onsets), onsets - oc, 0.0)
            onsets, dur = onsets - lead, dur + lead
        cols.append(d1.hrf_regressor(onsets, dur, n))
        names.append(kind)
        if rt_mod:
            rt = k["rt"].astype(float)
            has = rt.notna().values
            if has.sum() >= 5 and rt[has].std() > 0:
                amp = np.where(has, rt.fillna(0).values - rt[has].mean(), 0.0)
                cols.append(amplitude_regressor(onsets, dur, amp, n))
                names.append(f"{kind}_per_second_rt")
    if model in ("ret_rtmod", "ret_rtdur"):           # one reaction-time regressor over both trial types
        k = e[e.rt.notna()]
        rt = k.rt.astype(float).values
        if len(k) >= 5 and rt.std() > 0:
            if model == "ret_rtmod":
                cols.append(amplitude_regressor(k.onset_s.values, np.full(len(k), RT_CONSTANT_DUR),
                                                rt - RT_REFERENCE, n))
                names.append("rtslope")
            else:
                cols.append(d1.hrf_regressor(k.onset_s.values, d1.CUE_TO_PROMPT + rt, n))
                names.append("rtdur")
    if model in ("fixed_fixbase", "response_fixbase", "trial_fixbase", "cue_search_fixbase") + NEW_MODELS:
        press = None
        if model in ("response_fixbase", "trial_fixbase", "cue_search_fixbase") + NEW_MODELS:
            rt = pd.Series(e.rt.astype(float).values, index=np.round(e.onset_s.values, 2))
            rt = rt[~rt.index.duplicated()]
            press = (ph_run.cue + d1.CUE_TO_PROMPT + np.round(ph_run.cue, 2).map(rt)).values
        extra, enames = phase_regressors(ph_run, n, press, merge_ordercue=(model == "trial_fixbase"),
                                         split_ordercue=(model in ("response_fixbase", "cue_search_fixbase")),
                                         noret_from_prompt=(model == "cue_search_fixbase"),
                                         no_ordercue=(model in NEW_MODELS))
        N = d1.nuisance_matrix(n, conf, nuis_run.iloc[0:0])      # the speaking screen is in `extra`
        for c, nm in zip(cols, names):                           # how far the phases overlap with retrieval
            X = np.column_stack(extra + [x for x in cols if x is not c] + [np.ones(n)])
            r2 = 1 - np.sum((c - X @ np.linalg.lstsq(X, c, rcond=None)[0]) ** 2) / np.sum((c - c.mean()) ** 2)
            print(f"    {nm}: r with {enames[0]} {np.corrcoef(c, extra[0])[0, 1]:.2f}, "
                  f"variance inflation {1 / (1 - r2):.1f}", flush=True)
        cols, names = cols + extra, names + [f"phase_{x}" for x in enames]   # the phases are saved too
    else:
        N = d1.nuisance_matrix(n, conf, nuis_run)
    D = np.column_stack(cols + [N])
    if np.linalg.matrix_rank(D) < D.shape[1]:
        print(f"    warning: design not of full rank ({np.linalg.matrix_rank(D)} of {D.shape[1]})")
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    return b[:len(names)], names


def participant_rt_reference(phases):
    """Mean reaction time of the participant over all trials with a press, all time points."""
    rt = pd.concat([p["ev"]["rt"].astype(float) for p in phases.values()]).dropna()
    return float(rt.mean())


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
    for session in d1.SESSIONS:
        try:
            scfg = dec.SESSION_CONFIG[session]
            ev, nuis = d1.retrieval_events(sub, session, hierarchy, obj_enc_to_cat, path_main)
            f, c = d1.select_run_files(sub_mri, scfg["mri_session"], scfg)
            phases[session] = dict(funcs=[mni_path(x) for x in f], confs=c, ev=ev, nuis=nuis,
                                   ph=phase_table(sub, session, path_main, obj_enc_to_cat))
        except Exception as err:
            print(f"{session} not usable: {err}")
    if not phases:
        raise RuntimeError("no usable session")

    # brain mask: voxels inside the fMRIPrep brain mask of every run used
    all_funcs = [f for p in phases.values() for f in p["funcs"]]
    mask = masking.intersect_masks([f.replace("desc-preproc_bold", "desc-brain_mask") for f in all_funcs],
                                   threshold=1, connected=False)
    mask.to_filename(opj(str(step_output_dir(__file__, MASK_DIR)), f"{sub}_brainmask_{'t1w' if NATIVE else 'mni'}.nii.gz"))
    print(f"brain mask: {int(np.asarray(mask.dataobj).sum())} voxels", flush=True)

    rt_ref = participant_rt_reference(phases)
    print(f"mean reaction time over all time points: {rt_ref:.2f} s", flush=True)
    sums, counts, run_rows = {}, {}, []
    for session, p in phases.items():
        data, confs = [], []
        for f, c in zip(p["funcs"], p["confs"]):
            print(f"  loading {os.path.basename(f)}", flush=True)
            y = masking.apply_mask(image.smooth_img(f, SMOOTHING_FWHM), mask).astype(np.float32)
            m = y.mean(axis=0)
            m[m <= 0] = np.inf
            data.append((y / m - 1.0) * 100.0)
            confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
        try:
            ev = d1.finalise_events(p["ev"], [y.shape[0] for y in data], session, sub)
        except ValueError as err:
            print(f"{session} not usable: {err}")
            continue
        d = d1.CUE_TO_PROMPT + ev["rt"].astype(float)
        ev["duration_s"] = d.fillna(d.median()).clip(lower=0.5)
        for ri, (y, conf) in enumerate(zip(data, confs)):
            e = ev[ev.run_idx == ri]
            if e.empty:
                continue
            cond = "pre" if session == "presleep" else ("post1" if ri < 3 else "post2")
            run_rows.append(dict(subject=sub, session=session, run_idx=ri, cond=cond,
                                 file=os.path.basename(p["funcs"][ri]), n_trs=y.shape[0],
                                 n_object=int((e.kind == "object").sum()), n_scene=int((e.kind == "scene").sum()),
                                 n_with_rt=int(e.rt.notna().sum()),
                                 mean_rt_object=float(e[e.kind == "object"].rt.astype(float).mean()),
                                 n_object_retrieved=int(((e.kind == "object") & e.correct.astype(bool)).sum()),
                                 n_scene_retrieved=int(((e.kind == "scene") & e.correct.astype(bool)).sum())))
            for model in [m for m in MODELS if m in os.environ.get("E3_MODELS", ",".join(MODELS)).split(",")]:
                b, names = fit_run(y, conf, e, p["nuis"][p["nuis"].run_idx == ri], model,
                                   p["ph"][p["ph"].run_idx == ri], rt_ref=rt_ref)
                for row, name in zip(b, names):
                    if model in NEW_MODELS and name in ("object", "rtslope", "rtdur"):      # every run on its own
                        masking.unmask(row.astype(np.float32), mask).to_filename(opj(
                            str(step_output_dir(__file__, opj(MAP_DIR, model))),
                            f"{sub}_{cond}R{ri % 3 + 1}_{name}.nii.gz"))
                    if model.endswith("_rt") and not name.endswith("per_second_rt"):
                        continue                      # main effects are taken from `fixed`
                    key = (model, cond, name)
                    sums[key] = sums.get(key, 0) + row
                    counts[key] = counts.get(key, 0) + 1
            print(f"  {session} run {ri} ({cond}): {len(e)} trials fitted", flush=True)
        del data

    for (model, cond, name), s in sums.items():
        out = opj(str(step_output_dir(__file__, opj(MAP_DIR, model))), f"{sub}_{cond}_{name}.nii.gz")
        if os.environ.get("E3_ONLY_NEW") == "1" and os.path.exists(out):
            continue                                  # leave maps alone that other jobs may be reading
        masking.unmask((s / counts[(model, cond, name)]).astype(np.float32), mask).to_filename(out)
    pd.DataFrame(run_rows).to_csv(opj(str(step_output_dir(__file__, "runs")), f"{sub}_runs.csv"), index=False)
    print(f"saved {len(sums)} maps")
    print("done")


if __name__ == "__main__":
    main()
