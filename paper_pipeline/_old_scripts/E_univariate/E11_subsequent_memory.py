#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E11 - Subsequent memory: activity while an object is learned (evening, ses-04),
      split by what happens to that object at later retrieval.

Same data, regions and nuisance terms as E9 (T1w space, NOT smoothed, percent
signal change, FreeSurfer regions of B1). The comparison is between objects
within the learning runs, so it does not depend on the short fixation baseline.

The logs hold no scored answers. "Retrieved" = the participant pressed the button
at the question mark of the object's retrieval trial; "not retrieved" = the trial
timed out. Search time = time from the question mark to the press.

Outcome taken from   eve   the evening retrieval (nses_0, same session as learning)
                     morn  the first morning retrieval (first trial of each object in nses_2)

Models (context, scene, gap, rating as in the `events` model of E9):
    cat_<outcome>   object_ret, object_notret             -> ret minus notret
                    (participants with at least MIN_EVENTS objects of each kind)
    rt_<outcome>    object (retrieved objects), object_notret, and a modulator
                    `rtmod` on the retrieved objects: log search time, z-scored
                    within participant. Negative = more activity while learning
                    objects that are later found faster.
    rtpos_<outcome> the same, with the mean log search time of each position in
                    the sequence (1-5) removed within participant first, because
                    search time depends on the position.

Output  outputs/E11_subsequent_memory/participants/<sub>_region_estimates.csv
        outputs/E11_subsequent_memory/participants/<sub>_counts.csv
Usage:  python E11_subsequent_memory.py <subject>
        python E11_subsequent_memory.py group     -> group_effects.csv, counts.csv

Not a port of the original pipeline: added 2026-11-23.
"""

from __future__ import annotations

import glob
import os
import re
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
import E9_encoding_regions as e9
from config import step_input_dir, step_output_dir
from decoding_utils import CONFOUND_VARS, parse_subject_arg

d1 = e3.d1
MIN_EVENTS = 5                      # objects per outcome needed for the categorical contrast
OUTCOMES = {"eve": 0, "morn": 2}    # outcome name -> behaviour session (nses)


def retrieval_outcomes(sub, nses):
    """object file name -> (pressed, search time) at its first retrieval trial of the session."""
    base = opj(str(step_input_dir("A1_add_tr_columns_main")), sub)
    fs = sorted(glob.glob(opj(base, f"*_nses_{nses}_step*", "output_df_*_w_HRF.csv")),
                key=lambda x: int(re.search(r"step(\d+)", x).group(1)))
    out = {}
    for f in fs[::-1]:                                   # highest step first
        d = pd.read_csv(f, low_memory=False)
        d["tr"] = (d.label_type == "ret_fix").cumsum()
        for _, g in d[d.tr > 0].groupby("tr"):
            c = g[g.label_type == "sc_no4obj"]
            if c.empty or c.obj_ret_targ.iloc[0] in out:
                continue
            p = g[g.label_type == "q_mark_press"]
            out[c.obj_ret_targ.iloc[0]] = (len(p) > 0, float(p.ret_resp.iloc[0]) if len(p) else np.nan)
        if out:
            break
    return out


def design(g, n, kind, outc):
    cols, names = [], []

    def add(sel, name, mod=None):
        on, end = g.t_onset[sel].values, g.end[sel].values
        ok = end > on
        if ok.any():
            cols.append(d1.hrf_regressor(on[ok], (end - on)[ok], n) if mod is None else
                        hrf_mod(on[ok], (end - on)[ok], mod[ok], n)); names.append(name)

    is_obj = g.label_type == "dn+sc+obj_cue"
    is_sc = g.label_type == "dn+sc_cue"
    pressed = g.obj_enc_stim.map(lambda o: outc.get(o, (None, np.nan))[0])
    add(g.label_type == "dn_cue", "context")
    add(is_sc & (g.pos == 0), "scene")
    add(is_obj & (pressed == True), "object_ret" if kind == "cat" else "object")      # noqa: E712
    add(is_obj & (pressed == False), "object_notret")                                  # noqa: E712
    if kind != "cat":
        sel = is_obj & (pressed == True)                                               # noqa: E712
        add(sel, "rtmod", mod=g.z[sel].values)
    add(is_sc & (g.pos > 0), "gap")
    add(g.label_type.isin(["slider", "slider_resp"]), "rating")
    add(g.label_type.isin(["instr_txt", "break_text", "rest_phase"]), "instruction")
    return cols, names


def hrf_mod(on, dur, height, n):
    """Sum of single-event regressors scaled by `height`."""
    return sum(h * d1.hrf_regressor(np.array([o]), np.array([d]), n) for o, d, h in zip(on, dur, height))


def participant(sub):
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    ev = e9.encoding_events(sub)
    funcs, confs = d1.session_files(sub_mri, "ses-04")
    use = {run: e9.RUN_MAPPING[run] for run in ev}
    mask = masking.intersect_masks([funcs[i].replace("desc-preproc_bold", "desc-brain_mask") for i in use.values()],
                                   threshold=1, connected=False)
    bm = np.asarray(mask.dataobj).astype(bool)
    regions = d1.anatomical_regions(sub_mri, mask, bm)
    names_r = list(regions)
    outcomes, counts_rows = {}, []
    pos_of = pd.concat(ev.values()).query("label_type == 'dn+sc+obj_cue'").groupby("obj_enc_stim").pos.first().to_dict()
    for oname, nses in OUTCOMES.items():
        o = retrieval_outcomes(sub, nses)
        rt = pd.Series({k: v[1] for k, v in o.items() if v[0]}, dtype=float)
        lz = np.log(rt)
        z = ((lz - lz.mean()) / lz.std()).to_dict() if len(rt) > 2 and lz.std() > 0 else {}
        lp = lz - lz.groupby(lz.index.map(pos_of)).transform("mean")       # position in the sequence removed
        zp = (lp / lp.std()).to_dict() if len(rt) > 2 and lp.std() > 0 else {}
        outcomes[oname] = (o, z, zp)
        counts_rows.append(dict(subject=sub, outcome=oname, n_objects=len(o), n_ret=int(sum(v[0] for v in o.values())),
                                n_notret=int(sum(not v[0] for v in o.values())), mean_rt=float(rt.mean()) if len(rt) else np.nan))
        print(f"  {oname}: {counts_rows[-1]}", flush=True)
    sums, cnt = {}, {}
    for run, fi in use.items():
        g = ev[run]
        y = masking.apply_mask(funcs[fi], mask).astype(np.float32)
        conf = pd.read_csv(confs[fi], sep="\t")[CONFOUND_VARS].fillna(0).values
        n = min(y.shape[0], int(np.ceil((g.end.max() + e9.TAIL_S) / d1.TR)))
        if g.end.max() > y.shape[0] * d1.TR:
            raise ValueError(f"events of run {run} run past the end of the BOLD run")
        y, conf = y[:n], conf[:n]
        m = y.mean(axis=0); m[m <= 0] = np.inf
        y = (y / m - 1.0) * 100.0
        ts = np.column_stack([y[:, regions[r]].mean(axis=1) for r in names_r])
        N = d1.nuisance_matrix(n, conf, pd.DataFrame(columns=["name", "onset_s", "duration"]))
        for oname, (o, z, zp) in outcomes.items():
            for kind in ("cat", "rt", "rtpos"):
                g = g.copy(); g["z"] = g.obj_enc_stim.map(zp if kind == "rtpos" else z)
                if kind != "cat" and not (zp if kind == "rtpos" else z):
                    continue
                cols, nm = design(g, n, kind, o)
                b = np.linalg.lstsq(np.column_stack(cols + [N]), ts, rcond=None)[0]
                for row, x in zip(b, nm):
                    if x != "instruction":
                        key = (f"{kind}_{oname}", x)
                        sums[key] = sums.get(key, 0) + row; cnt[key] = cnt.get(key, 0) + 1
    rows = [dict(subject=sub, model=mo, regressor=r, region=reg, beta=float(v[i] / cnt[(mo, r)]), n_runs=cnt[(mo, r)])
            for (mo, r), v in sums.items() for i, reg in enumerate(names_r)]
    out = str(step_output_dir(__file__, "participants"))
    pd.DataFrame(rows).to_csv(opj(out, f"{sub}_region_estimates.csv"), index=False)
    pd.DataFrame(counts_rows).to_csv(opj(out, f"{sub}_counts.csv"), index=False)
    print("done")


def group():
    out = str(step_output_dir(__file__))
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_region_estimates.csv")))])
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_counts.csv")))])
    c.to_csv(opj(out, "counts.csv"), index=False)
    d = d[d.n_runs == 2]                                 # regressor present in both learning runs
    w = d.pivot_table(index=["model", "region", "subject"], columns="regressor", values="beta").reset_index()
    w["ret_minus_notret"] = w.object_ret - w.object_notret
    rows = []
    for (mo, reg), g in w.groupby(["model", "region"]):
        oname = mo.split("_")[1]
        ok = c[(c.outcome == oname) & (c.n_ret >= MIN_EVENTS) & (c.n_notret >= MIN_EVENTS)].subject
        for eff in ("object_ret", "object_notret", "ret_minus_notret", "rtmod"):
            gg = g[g.subject.isin(ok)] if eff != "rtmod" else g
            v = gg[eff].dropna().values
            if len(v) >= 10:
                r = stats.ttest_1samp(v, 0)
                rows.append(dict(model=mo, region=reg, effect=eff, n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue,
                                 n_positive=int((v > 0).sum())))
    pd.DataFrame(rows).to_csv(opj(out, "group_effects.csv"), index=False)
    w.to_csv(opj(out, "participant_effects.csv"), index=False)
    print(f"{d.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
