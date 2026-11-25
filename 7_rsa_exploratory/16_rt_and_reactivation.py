#!/usr/bin/env python3
"""
Does the drop in response time from evening to morning have a neural counterpart?

Response times fall by about half a second from the evening to the morning.
This script asks, for object trials and the four main ROIs (vis,
parietal_precuneus, mtl, control caudal_acc):

A  Behaviour: response time by retrieved position and session; the slope over
   positions 1-4 (seconds per position; stepping through the sequence predicts a
   positive slope, direct access a flat one) and its change between sessions.
B  Within participants: is a reactivation measure larger on slow trials?
   Per participant and session: slope of the measure on response time (per
   second); tested across participants.
C  Session difference with response time held constant: per participant,
   measure ~ session + response time; the session coefficients.
D  Timing: time course (TR 0-9 after the cue) of the evidence for the retrieved
   item and for its predecessor per session; centre of mass over TRs 2-9 of the
   positive part, and early (TR 3-4) minus late (TR 7-8) evidence. An earlier
   reinstatement in the morning predicts a smaller centre of mass.
E  Between participants: change in response time (and in its slope over
   positions) against change in every measure.

Measures per trial: retrieved item, predecessor, successor, predecessor - successor,
sequence evidence (all TRs 3-8), |slope| five items, sinusoid five items, and the
control sequence's |slope| five items (the "quieting" of the control sequence).

Inputs: the 4.6 / 4.7 tables of the decoding pipeline, and the trial tables of
04_single_trial_betas.py for response time.
Usage:  python 16_rt_and_reactivation.py [n_jobs]
Output: outputs/16_rt_and_reactivation/
"""

import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import rsa_config as cfg

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_DIR = opj(cfg.OUT_ROOT, "16_rt_and_reactivation")
SODA = opj(cfg.PATH_DECODING, "4.6_SODA_based_on_singlesubj_4.4_{s}", "SODA_results_csv", "{r}_soda_results_per_combination.csv")
SINUS = opj(cfg.PATH_DECODING, "4.7_sinus_1cycle", "{r}", "{r}_windowed_fit_results_{s}.csv")
TRIALS = opj(cfg.OUT_ROOT, "patterns_lss", "{s}", "*_{s}_trials.csv")
ROIS = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]
MEASURES = ["retrieved item", "predecessor", "successor", "predecessor - successor", "sequence evidence",
            "|slope| five items", "sinusoid five items", "control |slope| five items"]
SAFE = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
CONDS = ["pre", "post1", "post2"]


def tt(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return dict(n=int(ok.sum()), r=np.nan, p=np.nan, rho=np.nan)
    r, p = stats.pearsonr(x[ok], y[ok])
    return dict(n=int(ok.sum()), r=r, p=p, rho=stats.spearmanr(x[ok], y[ok])[0])


def behaviour():
    import glob
    parts = []
    for s in ("presleep", "postsleep"):
        for f in glob.glob(TRIALS.format(s=s)):
            t = pd.read_csv(f)
            t["session"] = s
            t["cond"] = "pre" if s == "presleep" else np.where(t.run_idx < 3, "post1", "post2")
            parts.append(t[["subject", "session", "cond", "trial", "kind", "position", "rt", "correct"]])
    return pd.concat(parts, ignore_index=True)


def load_roi(roi, beh):
    trial_rows, tc_rows = [], []
    for s in ("presleep", "postsleep"):
        d = pd.read_csv(SODA.format(s=s, r=roi), usecols=["subject", "trial", "tr", "object_scene_retrieval",
                        "retrieved_position", "retrieved_probs", "control_probs", "retrieved_soda_abs",
                        "control_soda_abs"], low_memory=False)
        d = d[d.object_scene_retrieval == "object"].reset_index(drop=True)
        E = 100 * (np.array([eval(x, SAFE) for x in d.retrieved_probs], float)
                   - np.array([eval(x, SAFE) for x in d.control_probs], float))
        p = d.retrieved_position.astype(int).values
        k = np.arange(len(d))
        mid = np.isin(p, [2, 3, 4])
        row = pd.DataFrame(dict(subject=d.subject.values, trial=d.trial.values, tr=d.tr.values))
        row["target_tr"] = E[k, p - 1]
        row["pred_tr"] = np.where(p >= 2, E[k, np.clip(p - 2, 0, 4)], np.nan)
        win = (row.tr >= 3) & (row.tr <= 8)
        row["retrieved item"] = np.where(win, row.target_tr, np.nan)
        row["predecessor"] = np.where(win & mid, E[k, np.clip(p - 2, 0, 4)], np.nan)
        row["successor"] = np.where(win & mid, E[k, np.clip(p, 0, 4)], np.nan)
        row["predecessor - successor"] = row["predecessor"] - row["successor"]
        row["sequence evidence"] = np.where(win, E.mean(1), np.nan)
        row["|slope| five items"] = 100 * (d.retrieved_soda_abs.values - d.control_soda_abs.values)
        row["control |slope| five items"] = 100 * d.control_soda_abs.values
        tri = row.groupby(["subject", "trial"], as_index=False)[[m for m in MEASURES if m in row]].mean()
        f = pd.read_csv(SINUS.format(s=s, r=roi), usecols=["subject", "trial", "retrieved_w_amplitude", "control_w_amplitude"])
        f["sinusoid five items"] = f.retrieved_w_amplitude.abs() - f.control_w_amplitude.abs()
        tri = tri.merge(f[["subject", "trial", "sinusoid five items"]], on=["subject", "trial"], how="left")
        tri["session"] = s
        trial_rows.append(tri)
        tc = row[["subject", "trial", "tr", "target_tr", "pred_tr"]].copy()
        tc["session"] = s
        tc_rows.append(tc)
    tri = pd.concat(trial_rows, ignore_index=True).merge(
        beh[beh.kind == "object"], on=["subject", "session", "trial"], how="inner")
    tc = pd.concat(tc_rows, ignore_index=True).merge(
        beh.loc[beh.kind == "object", ["subject", "session", "trial", "cond"]], on=["subject", "session", "trial"])
    tri["roi"] = roi
    tc = tc.groupby(["subject", "cond", "tr"], as_index=False)[["target_tr", "pred_tr"]].mean()
    tc["roi"] = roi
    return tri, tc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    beh = behaviour()
    obj = beh[(beh.kind == "object") & beh.rt.notna()]

    # A behaviour
    rt_pos = obj.groupby(["subject", "cond", "position"]).rt.median().reset_index()
    rt_pos.groupby(["cond", "position"]).rt.agg(["mean", "sem", "count"]).reset_index().to_csv(
        opj(OUT_DIR, "A_rt_by_position.csv"), index=False)
    sl = obj[obj.position <= 4].groupby(["subject", "cond"]).apply(
        lambda g: np.polyfit(g.position, g.rt, 1)[0] if g.position.nunique() >= 3 else np.nan).unstack()
    med = obj.groupby(["subject", "cond"]).rt.median().unstack()
    accs = beh[beh.kind == "object"].groupby(["subject", "cond"]).correct.mean().unstack()
    a_rows = []
    for name, tab in (("median RT (s)", med), ("RT slope over positions 1-4 (s per position)", sl)):
        for c in CONDS:
            a_rows.append(dict(measure=name, what=c, **tt(tab[c])))
        for a, b in (("post1", "pre"), ("post2", "post1")):
            a_rows.append(dict(measure=name, what=f"{a} - {b}", **tt(tab[a] - tab[b])))
    pd.DataFrame(a_rows).to_csv(opj(OUT_DIR, "A_behaviour_tests.csv"), index=False)

    res = Parallel(n_jobs=min(N_JOBS, len(ROIS)))(delayed(load_roi)(r, beh) for r in ROIS)
    tri = pd.concat([r[0] for r in res], ignore_index=True)
    tc = pd.concat([r[1] for r in res], ignore_index=True)
    tri.to_csv(opj(OUT_DIR, "trial_table.csv"), index=False)

    # B within participants: measure ~ RT ; C session with RT held constant
    b_rows, c_rows = [], []
    per_b, per_c = [], []
    for (roi, sub), g in tri.dropna(subset=["rt"]).groupby(["roi", "subject"]):
        for m in MEASURES:
            gg = g.dropna(subset=[m])
            for c in CONDS + ["pooled"]:
                x = gg if c == "pooled" else gg[gg.cond == c]
                if len(x) >= 15 and x.rt.std() > 0:
                    if c == "pooled":          # remove session means so that only within-session variation counts
                        xr = x.rt - x.groupby("cond").rt.transform("mean")
                        yr = x[m] - x.groupby("cond")[m].transform("mean")
                    else:
                        xr, yr = x.rt - x.rt.mean(), x[m] - x[m].mean()
                    per_b.append(dict(roi=roi, subject=sub, measure=m, cond=c, slope=np.polyfit(xr, yr, 1)[0]))
            if gg.cond.nunique() == 3 and len(gg) >= 30:
                X = np.column_stack([np.ones(len(gg)), gg.cond == "post1", gg.cond == "post2", gg.rt - gg.rt.mean()]).astype(float)
                b = np.linalg.lstsq(X, gg[m].values, rcond=None)[0]
                X0 = X[:, :3]
                b0 = np.linalg.lstsq(X0, gg[m].values, rcond=None)[0]
                per_c.append(dict(roi=roi, subject=sub, measure=m, post1_adj=b[1], post_adj=(b[1] + b[2]) / 2,
                                  post1_raw=b0[1], post_raw=(b0[1] + b0[2]) / 2))
    per_b, per_c = pd.DataFrame(per_b), pd.DataFrame(per_c)
    for (roi, m, c), g in per_b.groupby(["roi", "measure", "cond"]):
        b_rows.append(dict(roi=roi, measure=m, cond=c, **tt(g.slope)))
    for (roi, m), g in per_c.groupby(["roi", "measure"]):
        for col in ("post1_raw", "post1_adj", "post_raw", "post_adj"):
            c_rows.append(dict(roi=roi, measure=m, contrast=col, **tt(g[col])))
    pd.DataFrame(b_rows).to_csv(opj(OUT_DIR, "B_measure_per_second_rt.csv"), index=False)
    pd.DataFrame(c_rows).to_csv(opj(OUT_DIR, "C_session_effect_rt_adjusted.csv"), index=False)

    # D timing
    tc.groupby(["roi", "cond", "tr"])[["target_tr", "pred_tr"]].agg(["mean", "sem"]).to_csv(opj(OUT_DIR, "D_timecourses.csv"))
    d_rows, lat = [], []
    for (roi, sub, c), g in tc.groupby(["roi", "subject", "cond"]):
        g = g.sort_values("tr")
        for col, name in (("target_tr", "retrieved item"), ("pred_tr", "predecessor")):
            w = g[(g.tr >= 2)]
            pos = np.clip(w[col].values, 0, None)
            com = (pos * w.tr.values).sum() / pos.sum() if pos.sum() > 0 else np.nan
            early = g[g.tr.isin([3, 4])][col].mean()
            late = g[g.tr.isin([7, 8])][col].mean()
            lat.append(dict(roi=roi, subject=sub, cond=c, item=name, centre_of_mass_tr=com, early_minus_late=early - late,
                            early=early, late=late))
    lat = pd.DataFrame(lat)
    lat.to_csv(opj(OUT_DIR, "D_latency_subject.csv"), index=False)
    for (roi, item), g in lat.groupby(["roi", "item"]):
        for col in ("centre_of_mass_tr", "early_minus_late", "early", "late"):
            w = g.pivot(index="subject", columns="cond", values=col)
            for c in CONDS:
                d_rows.append(dict(roi=roi, item=item, measure=col, what=c, n=w[c].notna().sum(), mean=w[c].mean(), t=np.nan, p=np.nan))
            for a, b in (("post1", "pre"), ("post2", "post1")):
                d_rows.append(dict(roi=roi, item=item, measure=col, what=f"{a} - {b}", **tt(w[a] - w[b])))
    pd.DataFrame(d_rows).to_csv(opj(OUT_DIR, "D_latency_group.csv"), index=False)

    # E between participants: change in RT against change in measure
    subj = tri.groupby(["roi", "subject", "cond"])[MEASURES].mean()
    e_rows = []
    for a, b in (("post1", "pre"), ("post2", "post1")):
        d_rt, d_sl, d_acc = med[a] - med[b], sl[a] - sl[b], accs[a] - accs[b]
        for roi in ROIS:
            dm = subj.loc[roi].unstack("cond")
            for m in MEASURES:
                delta = dm[m][a] - dm[m][b]
                for bname, bv in (("change in median RT", d_rt), ("change in RT slope over positions", d_sl),
                                  ("change in accuracy", d_acc)):
                    e_rows.append(dict(contrast=f"{a} - {b}", roi=roi, measure=m, behaviour=bname,
                                       **corr(delta.values, bv.reindex(delta.index).values)))
    pd.DataFrame(e_rows).to_csv(opj(OUT_DIR, "E_change_vs_change.csv"), index=False)
    print("written to", OUT_DIR, "| participants:", tri.subject.nunique())


if __name__ == "__main__":
    main()
