#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E10 - Evening against morning on trials matched for search time.

The models of E3 take the reaction time into account through the length of the
event or through an extra regressor, and their evening-to-morning contrasts
disagree. This step avoids the assumption: it estimates every object trial with
a button press on its own and compares the time points only on trials with
(nearly) the same time from the question mark to the press.

Participant mode. As E7 (T1w space, not smoothed, percent signal change, mean
signal per anatomical region), one model per run with ONE regressor per object
trial with a press, in two forms:
    press   each trial lasts from the retrieval cue to its button press
    const   each trial is the 1.015 s the cue is on screen
plus, as in the E3 models `ret_press` / `ret_rtdur`: one regressor for the scene
trials with a press (same form), `wait`, `speak`, `noretrieval`, cosine drifts
and confounds. The trial-type cue has no regressor.

Group mode. Within each participant, evening trials are paired with morning
trials whose reaction time differs by at most CALIPER seconds (each trial used
once; pairs chosen to minimise the summed difference). The mean difference
morning minus evening over the pairs is tested across participants. For
comparison the same difference over ALL trials, and the time-point effect from
a regression of the trial estimates on time point and reaction time (linear and
squared) within participant.

Output  outputs/E10_matched_search_time/participants/<sub>_trials.csv
        outputs/E10_matched_search_time/group_matched.csv, participant_matched.csv
Usage:  python E10_matched_search_time.py <subject>
        python E10_matched_search_time.py group

Environment (optional): E10_KIND=scene estimates every SCENE trial with a press
on its own (the object trials then get the one shared regressor) and writes to
participants_scene/, group_matched_scene.csv and participant_matched_scene.csv.
Added 2026-11-24; the default (object trials) is unchanged.

Not a port of the original pipeline: added 2026-11-23.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking
from scipy import stats
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import E3_wholebrain_maps as e3
from config import step_input_dir, step_output_dir
from decoding_utils import CONFOUND_VARS, build_sequence_dict_from_df_behavior_enc, parse_subject_arg

d1, dec = e3.d1, e3.dec
CALIPER = 0.25          # s, largest difference in reaction time within a pair
MIN_PAIRS = 8
FORMS = ("press", "const")
KIND = os.environ.get("E10_KIND", "object")              # trial type estimated trial by trial
OTHER = "scene" if KIND == "object" else "object"
TAG = "" if KIND == "object" else "_" + KIND
COMPARISONS = {"post1_minus_pre": ("post1", "pre"), "post2_minus_pre": ("post2", "pre"),
               "post_minus_pre": ("post", "pre"), "post2_minus_post1": ("post2", "post1")}


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
    regions = d1.anatomical_regions(sub_mri, mask, bm)
    names = list(regions)
    rows = []
    for session, p in phases.items():
        ts, confs = [], []
        for f, c in zip(p["funcs"], p["confs"]):
            print(f"  loading {os.path.basename(f)}", flush=True)
            y = masking.apply_mask(f, mask).astype(np.float32)
            m = y.mean(axis=0); m[m <= 0] = np.inf
            y = (y / m - 1.0) * 100.0
            ts.append(np.column_stack([y[:, regions[r]].mean(axis=1) for r in names]))
            confs.append(pd.read_csv(c, sep="\t")[CONFOUND_VARS].fillna(0).values)
            del y
        try:
            ev = d1.finalise_events(p["ev"], [y.shape[0] for y in ts], session, sub)
        except ValueError as err:
            print(f"{session} not usable: {err}")
            continue
        for ri, (y, conf) in enumerate(zip(ts, confs)):
            e = ev[ev.run_idx == ri]
            ph = p["ph"][p["ph"].run_idx == ri]
            if e.empty:
                continue
            n = y.shape[0]
            cond = "pre" if session == "presleep" else ("post1" if ri < 3 else "post2")
            obj = e[(e.kind == KIND) & e.rt.notna()]
            sc = e[(e.kind == OTHER) & e.rt.notna()]
            if obj.empty:
                continue
            rt = pd.Series(e.rt.astype(float).values, index=np.round(e.onset_s.values, 2))
            rt = rt[~rt.index.duplicated()]
            press = (ph.cue + d1.CUE_TO_PROMPT + np.round(ph.cue, 2).map(rt)).values
            extra, _ = e3.phase_regressors(ph, n, press, no_ordercue=True)
            N = d1.nuisance_matrix(n, conf, p["nuis"].iloc[0:0])
            est = {}
            for form in FORMS:
                dur = (lambda k: (d1.CUE_TO_PROMPT + k.rt.astype(float)).clip(lower=0.5).values) if form == "press" \
                    else (lambda k: np.full(len(k), d1.CUE_TO_PROMPT))
                X = [d1.hrf_regressor([o], [d], n) for o, d in zip(obj.onset_s.values, dur(obj))]
                other = [d1.hrf_regressor(sc.onset_s.values, dur(sc), n)] if len(sc) else []
                D = np.column_stack(X + other + extra + [N])
                est[form] = np.linalg.lstsq(D, y, rcond=None)[0][:len(obj)]
            for i, (_, t) in enumerate(obj.iterrows()):
                row = dict(subject=sub, cond=cond, run=ri % 3 + 1, object=t.get("object"), scene=t.get("scene"), position=t.get("position"),
                           rt=float(t.rt), onset_s=float(t.onset_s))
                for form in FORMS:
                    row.update({f"{form}_{r}": float(est[form][i, j]) for j, r in enumerate(names)})
                rows.append(row)
            print(f"  {session} run {ri} ({cond}): {len(obj)} {KIND} trials", flush=True)
    pd.DataFrame(rows).to_csv(opj(str(step_output_dir(__file__, "participants" + TAG)), f"{sub}_trials.csv"), index=False)
    print("done")


def match(a_rt, b_rt, caliper=CALIPER):
    """Pairs (index in a, index in b) with |difference| <= caliper, each trial once, as many as possible."""
    cost = np.abs(a_rt[:, None] - b_rt[None, :])
    big = 1e3
    r, c = linear_sum_assignment(np.where(cost <= caliper, cost, big))
    ok = cost[r, c] <= caliper
    return r[ok], c[ok]


def group():
    out = str(step_output_dir(__file__))
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants" + TAG, "*_trials.csv")))])
    regs = sorted({c.split("_", 1)[1] for c in d.columns if c.startswith("press_")})
    rows = []
    for sub, g in d.groupby("subject"):
        for name, (a, b) in COMPARISONS.items():
            A = g[g.cond.isin(["post1", "post2"])] if a == "post" else g[g.cond == a]
            B = g[g.cond == b]
            if len(A) < MIN_PAIRS or len(B) < MIN_PAIRS:
                continue
            ia, ib = match(A.rt.values, B.rt.values)
            base = dict(subject=sub, comparison=name, n_a=len(A), n_b=len(B), n_pairs=len(ia),
                        rt_a_all=A.rt.mean(), rt_b_all=B.rt.mean(),
                        rt_a_matched=A.rt.values[ia].mean() if len(ia) else np.nan,
                        rt_b_matched=B.rt.values[ib].mean() if len(ib) else np.nan)
            both = pd.concat([A.assign(tp=1.0), B.assign(tp=0.0)])
            rtc = both.rt.values - 1.5
            Xr = np.column_stack([np.ones(len(both)), both.tp.values, rtc, rtc ** 2])
            for form in FORMS:
                for r in regs:
                    col = f"{form}_{r}"
                    rows.append(dict(base, form=form, region=r,
                                     all_trials=A[col].mean() - B[col].mean(),
                                     matched=(A[col].values[ia].mean() - B[col].values[ib].mean())
                                     if len(ia) >= MIN_PAIRS else np.nan,
                                     regression=np.linalg.lstsq(Xr, both[col].values, rcond=None)[0][1]))
    p = pd.DataFrame(rows)
    p.to_csv(opj(out, f"participant_matched{TAG}.csv"), index=False)

    def tt(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        r = stats.ttest_1samp(v, 0)
        return len(v), v.mean(), r.statistic, r.pvalue

    res = []
    for (form, comp, r), g in p.groupby(["form", "comparison", "region"]):
        row = dict(form=form, comparison=comp, region=r, pairs_mean=g.n_pairs.mean(), pairs_min=g.n_pairs.min(),
                   rt_diff_all=(g.rt_a_all - g.rt_b_all).mean(), rt_diff_matched=(g.rt_a_matched - g.rt_b_matched).mean())
        for k in ("all_trials", "matched", "regression"):
            n, m, t, pv = tt(g[k])
            row.update({f"{k}_n": n, f"{k}_mean": m, f"{k}_t": t, f"{k}_p": pv})
        res.append(row)
    pd.DataFrame(res).to_csv(opj(out, f"group_matched{TAG}.csv"), index=False)
    print(f"{p.subject.nunique()} participants -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
