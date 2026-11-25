#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9 - Mean activity while the object sequences are learned (evening, ses-04),
     in each participant's own anatomy.

Same approach as E7 (T1w space, NOT smoothed, percent signal change, signal
averaged over the voxels of each anatomical region of B1, cosine drifts slower
than 128 s and the 20 fMRIPrep confounds), for the two runs in which the object
sequences are shown (behaviour runs 2 and 3; BOLD runs with index 3 and 4 of
ses-04, as in B6a). The second of these BOLD runs continues with a rest phase;
it is cut 15 s after the last event.

One learning trial (a sequence of five objects in one scene):
    obj_enc_fix    fixation, 3-5 s jittered           -> the ONLY unmodelled time
    dn_cue         context (day / night), 3.05 s
    dn+sc_cue      the scene appears, 2.5-4.6 s jittered
    dn+sc+obj_cue  an object on the scene, 2.54 s     } five times; between objects
    dn+sc_cue      the scene alone, 2.5-4.6 s jittered }  the scene stays on screen
    slider         rating until the response
Every sequence is shown three times (repetition 1 and 2 in the first run,
repetition 3 in the second).

Three models, all relative to fixation:
    events      context, scene (first appearance), object_rep1..3 (the objects, by
                repetition), gap (scene alone between objects), rating
    block       context, sequence_rep1..3 (scene onset until the rating), rating
    position    as `events`, with the objects split by their position in the
                sequence (object_pos1..5) and not by repetition

Output  outputs/E9_encoding_regions/participants/<sub>_region_estimates.csv
        (columns: subject, model, regressor, region, n_voxels, beta, n_runs)
Usage:  python E9_encoding_regions.py <subject>      e.g. sub-14
        python E9_encoding_regions.py group          -> group_levels.csv, group_differences.csv

Not a port of the original pipeline: added 2026-11-23.
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
from decoding_utils import CONFOUND_VARS, find_encoding_folder, parse_subject_arg

d1 = e3.d1
RUN_MAPPING = {2: 3, 3: 4}          # behaviour run -> index of the BOLD run in ses-04 (B6a)
TAIL_S = 15.0
MODELS = ("events", "block", "position")


def encoding_events(sub):
    folder = find_encoding_folder(str(step_input_dir("A1_add_tr_columns_main")), sub)
    d = pd.read_csv(opj(folder, f"output_df_sub{sub[1:]}_w_HRF.csv"))
    d["run_nr"] = d.run_nr.ffill()
    d = d[d.run_nr.isin(RUN_MAPPING) & d.t_onset.notna() & d.label_type.notna()].copy()
    out = {}
    for run, g in d.groupby("run_nr"):
        g = g.reset_index(drop=True)
        g = g.iloc[:g.index[g.label_type.isin(["slider", "slider_resp"])].max() + 1].copy()   # up to the last rating
        g["end"] = g.t_onset.shift(-1)
        g.loc[g.index[-1], "end"] = g.t_onset.iloc[-1] + 0.5
        g["seq"] = (g.label_type == "obj_enc_fix").cumsum()
        g["rep"] = g.rep_nr.where(g.label_type == "obj_enc_fix").ffill() + 1
        g["pos"] = g.groupby("seq").label_type.transform(lambda x: (x == "dn+sc+obj_cue").cumsum())
        out[int(run)] = g
    return out


def design(g, n, model):
    cols, names = [], []

    def add(sel, name, on=None, end=None):
        on = g.t_onset[sel].values if on is None else on
        end = g.end[sel].values if end is None else end
        ok = end > on
        if ok.any():
            cols.append(d1.hrf_regressor(on[ok], (end - on)[ok], n)); names.append(name)

    is_obj = g.label_type == "dn+sc+obj_cue"
    is_sc = g.label_type == "dn+sc_cue"
    first_sc = is_sc & (g.pos == 0)
    add(g.label_type == "dn_cue", "context")
    if model == "block":
        for r in sorted(g.rep.dropna().unique()):
            on = g[first_sc & (g.rep == r)].groupby("seq").t_onset.first()
            end = g[(g.label_type == "slider") & (g.rep == r)].groupby("seq").t_onset.first()
            k = on.index.intersection(end.index)
            add(None, f"sequence_rep{int(r)}", on[k].values, end[k].values)
    else:
        add(first_sc, "scene")
        if model == "events":
            for r in sorted(g.rep.dropna().unique()):
                add(is_obj & (g.rep == r), f"object_rep{int(r)}")
        else:
            for p in range(1, 6):
                add(is_obj & (g.pos == p), f"object_pos{p}")
        add(is_sc & (g.pos > 0), "gap")
    add(g.label_type.isin(["slider", "slider_resp"]), "rating")
    add(g.label_type.isin(["instr_txt", "break_text", "rest_phase"]), "instruction")
    return cols, names


def participant(sub):
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    ev = encoding_events(sub)
    funcs, confs = d1.session_files(sub_mri, "ses-04")
    for i in (0, 2):
        if nib.load(funcs[i]).shape[-1] != 120:
            raise ValueError(f"BOLD run index {i} is not a 120-volume run; run mapping may be wrong for {sub}")
    use = {run: RUN_MAPPING[run] for run in ev}
    mask = masking.intersect_masks([funcs[i].replace("desc-preproc_bold", "desc-brain_mask") for i in use.values()],
                                   threshold=1, connected=False)
    bm = np.asarray(mask.dataobj).astype(bool)
    regions = d1.anatomical_regions(sub_mri, mask, bm)
    names_r = list(regions)
    sums, counts = {}, {}
    for run, fi in use.items():
        g = ev[run]
        print(f"  loading {os.path.basename(funcs[fi])}", flush=True)
        y = masking.apply_mask(funcs[fi], mask).astype(np.float32)
        conf = pd.read_csv(confs[fi], sep="\t")[CONFOUND_VARS].fillna(0).values
        n = min(y.shape[0], int(np.ceil((g.end.max() + TAIL_S) / d1.TR)))
        if g.end.max() > y.shape[0] * d1.TR:
            raise ValueError(f"events of run {run} run past the end of the BOLD run")
        y, conf = y[:n], conf[:n]
        m = y.mean(axis=0); m[m <= 0] = np.inf
        y = (y / m - 1.0) * 100.0
        ts = np.column_stack([y[:, regions[r]].mean(axis=1) for r in names_r])
        fix = g[g.label_type == "obj_enc_fix"]
        print(f"  run {run}: {n} volumes, {int((g.label_type == 'dn+sc+obj_cue').sum())} objects, "
              f"{len(fix)} fixations ({(fix.end - fix.t_onset).sum():.0f} s)", flush=True)
        N = d1.nuisance_matrix(n, conf, pd.DataFrame(columns=["name", "onset_s", "duration"]))
        for model in MODELS:
            cols, nm = design(g, n, model)
            D = np.column_stack(cols + [N])
            for c, x in zip(cols, nm):
                X = np.column_stack([o for o in cols if o is not c] + [np.ones(n)])
                r2 = 1 - np.sum((c - X @ np.linalg.lstsq(X, c, rcond=None)[0]) ** 2) / np.sum((c - c.mean()) ** 2)
                print(f"    {model} {x}: variance inflation {1 / (1 - r2):.1f}", flush=True)
            b = np.linalg.lstsq(D, ts, rcond=None)[0]
            for row, x in zip(b, nm):
                if x == "instruction":
                    continue
                sums[(model, x)] = sums.get((model, x), 0) + row
                counts[(model, x)] = counts.get((model, x), 0) + 1
    rows = [dict(subject=sub, model=m, regressor=r, region=reg, n_voxels=int(regions[reg].sum()),
                 beta=float(v[i] / counts[(m, r)]), n_runs=counts[(m, r)])
            for (m, r), v in sums.items() for i, reg in enumerate(names_r)]
    pd.DataFrame(rows).to_csv(opj(str(step_output_dir(__file__, "participants")), f"{sub}_region_estimates.csv"), index=False)
    print("done")


def group():
    out = str(step_output_dir(__file__))
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_region_estimates.csv")))])
    w = d.pivot_table(index=["model", "region", "subject"], columns="regressor", values="beta").reset_index()
    for m, a, b in (("events", "object", "object_rep"), ("block", "sequence", "sequence_rep"), ("position", "object", "object_pos")):
        c = [x for x in w.columns if x.startswith(b)]
        w.loc[w.model == m, f"{a}_all"] = w.loc[w.model == m, c].mean(axis=1)
    w.loc[w.model == "events", "object_minus_gap"] = w.object_all - w.gap
    for a, b in (("object_rep3", "object_rep1"), ("object_rep2", "object_rep1"), ("sequence_rep3", "sequence_rep1"),
                 ("object_pos5", "object_pos1")):
        w[f"{a}_minus_{b.split('_')[1]}"] = w[a] - w[b]
    rows = []
    for (m, reg), g in w.groupby(["model", "region"]):
        for c in [x for x in w.columns if x not in ("model", "region", "subject")]:
            v = g[c].dropna().values
            if len(v) >= 10:
                r = stats.ttest_1samp(v, 0)
                rows.append(dict(model=m, region=reg, effect=c, n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue))
    pd.DataFrame(rows).to_csv(opj(out, "group_levels.csv"), index=False)
    print(f"{d.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
