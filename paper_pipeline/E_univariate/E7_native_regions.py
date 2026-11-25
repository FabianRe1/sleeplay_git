#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E7 - The whole-brain models of E3, fitted in each participant's own anatomy.

Verification for small structures (hippocampus, parahippocampal and entorhinal
cortex, MTL), which are aligned less precisely across participants in standard
space. For one participant: the fMRIPrep data in T1w space, NOT smoothed, in
percent signal change; the signal is averaged over the voxels of each anatomical
region (B1 masks, as in D1 / E1) and the models of E3 are fitted to these
regional time courses with the same regressors (same functions). Nothing is
transformed to standard space.

Models fitted: cue_search_fixbase (main model), response_fixbase, fixed_fixbase.
Time points as in E3 (pre, post1, post2).

Input   as E3, with BOLD in T1w space; outputs/B1_threshold_roi_masks (anatomical masks)
Output  outputs/E7_native_regions/participants/<sub>_region_estimates.csv
        (columns: subject, model, cond, regressor, region, n_voxels, beta, n_runs)

Usage:  python E7_native_regions.py <subject>          e.g. sub-14
        python E7_native_regions.py group              group tables from all participants:
            outputs/E7_native_regions/group_levels.csv        each regressor against fixation
            outputs/E7_native_regions/group_differences.csv   differences between time points

Not a port of the original pipeline: added 2026-11-22.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import E3_wholebrain_maps as e3
from config import step_input_dir, step_output_dir
from decoding_utils import CONFOUND_VARS, build_sequence_dict_from_df_behavior_enc, parse_subject_arg

d1, dec = e3.d1, e3.dec
MODELS = tuple(os.environ.get("E7_MODELS", "cue_search_fixbase,response_fixbase,fixed_fixbase").split(","))
CONDS = ("pre", "post1", "post2")
CONTRASTS = {"post1_minus_pre": ("post1", "pre"), "post2_minus_post1": ("post2", "post1"),
             "post2_minus_pre": ("post2", "pre")}


def participant(sub):
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    path_loc = str(step_input_dir("A2_add_tr_columns_localizer"))
    path_main = str(step_input_dir("A1_add_tr_columns_main"))
    _, o2c = dec.load_localizer_behavior(path_loc, sub)
    df_enc, _ = dec.load_main_task_behavior(path_main, sub, dec.SESSION_CONFIG["presleep"])
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)
    phases = {}
    for session in d1.SESSIONS:
        try:
            scfg = dec.SESSION_CONFIG[session]
            ev, nuis = d1.retrieval_events(sub, session, hierarchy, o2c, path_main)
            f, c = d1.select_run_files(sub_mri, scfg["mri_session"], scfg)
            phases[session] = dict(funcs=f, confs=c, ev=ev, nuis=nuis, ph=e3.phase_table(sub, session, path_main, o2c))
        except Exception as err:
            print(f"{session} not usable: {err}")
    if not phases:
        raise RuntimeError("no usable session")
    all_funcs = [f for p in phases.values() for f in p["funcs"]]
    mask = masking.intersect_masks([f.replace("desc-preproc_bold", "desc-brain_mask") for f in all_funcs],
                                   threshold=1, connected=False)
    bm = np.asarray(mask.dataobj).astype(bool)
    regions = d1.anatomical_regions(sub_mri, mask, bm)          # name -> boolean over mask voxels
    names = list(regions)
    print(f"{len(names)} regions; voxels: " + ", ".join(f"{k} {int(v.sum())}" for k, v in regions.items()), flush=True)

    rt_ref = e3.participant_rt_reference(phases)
    sums, counts = {}, {}
    for session, p in phases.items():
        ts, confs = [], []
        for f, c in zip(p["funcs"], p["confs"]):
            print(f"  loading {os.path.basename(f)}", flush=True)
            y = masking.apply_mask(f, mask).astype(np.float32)   # not smoothed
            m = y.mean(axis=0)
            m[m <= 0] = np.inf
            y = (y / m - 1.0) * 100.0
            ts.append(np.column_stack([y[:, regions[r]].mean(axis=1) for r in names]))
            confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
            del y
        try:
            ev = d1.finalise_events(p["ev"], [y.shape[0] for y in ts], session, sub)
        except ValueError as err:
            print(f"{session} not usable: {err}")
            continue
        d = d1.CUE_TO_PROMPT + ev["rt"].astype(float)
        ev["duration_s"] = d.fillna(d.median()).clip(lower=0.5)
        for ri, (y, conf) in enumerate(zip(ts, confs)):
            e = ev[ev.run_idx == ri]
            if e.empty:
                continue
            cond = "pre" if session == "presleep" else ("post1" if ri < 3 else "post2")
            for model in MODELS:
                b, rn = e3.fit_run(y, conf, e, p["nuis"][p["nuis"].run_idx == ri], model, p["ph"][p["ph"].run_idx == ri],
                                   rt_ref=rt_ref)
                for row, name in zip(b, rn):
                    for key in [(model, cond, name)] + ([(model, f"{cond}R{ri % 3 + 1}", name)] if model in e3.NEW_MODELS else []):
                        sums[key] = sums.get(key, 0) + row
                        counts[key] = counts.get(key, 0) + 1
    rows = [dict(subject=sub, model=m, cond=c, regressor=r, region=reg, n_voxels=int(regions[reg].sum()),
                 beta=float(v[i] / counts[(m, c, r)]), n_runs=counts[(m, c, r)])
            for (m, c, r), v in sums.items() for i, reg in enumerate(names)]
    tag = "" if "E7_MODELS" not in os.environ else "_" + "+".join(MODELS)
    pd.DataFrame(rows).to_csv(opj(str(step_output_dir(__file__, "participants" + tag)), f"{sub}_region_estimates.csv"), index=False)
    print("done")


def group():
    out = str(step_output_dir(__file__))
    tag = "" if "E7_MODELS" not in os.environ else "_" + "+".join(MODELS)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants" + tag, "*_region_estimates.csv")))])
    w = d.pivot_table(index=["model", "regressor", "region", "subject"], columns="cond", values="beta").reset_index()

    def tt(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v) < 10:
            return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
        r = stats.ttest_1samp(v, 0)
        return dict(n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue)

    lev, dif = [], []
    for (m, r, reg), g in w.groupby(["model", "regressor", "region"]):
        full = g.dropna(subset=list(CONDS))
        lev.append(dict(model=m, regressor=r, region=reg, cond="pooled", **tt(full[list(CONDS)].mean(axis=1))))
        for c in [x for x in g.columns if x in CONDS or "R" in str(x)[-2:]]:
            lev.append(dict(model=m, regressor=r, region=reg, cond=c, **tt(g[c])))
        for name, (a, b) in CONTRASTS.items():
            dif.append(dict(model=m, regressor=r, region=reg, contrast=name, **tt(g[a] - g[b])))
        dif.append(dict(model=m, regressor=r, region=reg, contrast="post_minus_pre",
                        **tt((full.post1 + full.post2) / 2 - full.pre)))
    pd.DataFrame(lev).to_csv(opj(out, f"group_levels{tag}.csv"), index=False)
    pd.DataFrame(dif).to_csv(opj(out, f"group_differences{tag}.csv"), index=False)
    print(f"{d.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
