#!/usr/bin/env python3
"""
Supplementary checks that were first run as one-off calculations.

A  Positions 3 and 4 only (classifier): evidence for the retrieved item, its
   predecessor, the item two before, the successor, predecessor - successor, and
   the slope measures (five items / up to the retrieved item; absolute / signed),
   per session and for the change between sessions.          -> A_positions_3_4.csv
B  What changes in the control sequence from evening to morning ("quieting"):
   target, other retrieved items, the control item at the same position, the other
   control items, peak and spread of the control items.       -> B_control_sequence.csv
C  Does the control sequence relate to performance? Level (between participants),
   correct versus incorrect trials (within), and change against change.
                                                              -> C_control_vs_behaviour.csv
D  Relation of univariate activity to response time, separately per session, for
   events of a fixed duration and events from cue to response (needs the trial
   tables of 15_univariate_session.py).                       -> D_rt_relation_per_session.csv

E  Slope without the retrieved item: all five items, without the retrieved item,
   also without its predecessor, also without both neighbours; per retrieved
   position; change split into retrieved and control sequence. -> E_slope_without_target.csv
F  Successor (the item after the retrieved one), positions 1-4: per session, change
   from evening to morning pass 1, and that change against the other ROIs.
                                                              -> F_successor_change.csv
G  Sinusoid amplitude with and without the retrieved item (needs
   outputs/19_without_target_sinusoid/, made by 19_without_target_sinusoid.sh):
   per session, change, split into retrieved and control sequence, comparison
   between ROIs, per retrieved position.                      -> G_sinusoid_without_target.csv

Object trials throughout; classifier evidence = retrieved sequence minus control
sequence in percentage points; t-tests across participants, uncorrected.
ROIs: vis, parietal_precuneus, mtl, caudal_acc (control); A also precuneus,
parietal and cortical_memory.

Usage:  python 18_supplementary_checks.py [n_jobs]
Output: outputs/18_supplementary_checks/
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
OUT_DIR = opj(cfg.OUT_ROOT, "18_supplementary_checks")
SODA = opj(cfg.PATH_DECODING, "4.6_SODA_based_on_singlesubj_4.4_{s}", "SODA_results_csv", "{r}_soda_results_per_combination.csv")
SINUS = opj(cfg.PATH_DECODING, "4.7_sinus_1cycle", "{r}", "{r}_windowed_fit_results_{s}.csv")
MAIN = ["parietal_precuneus", "vis", "mtl", "caudal_acc"]      # the first is compared with the others
ROIS_A = ["parietal_precuneus", "precuneus", "parietal", "vis", "cortical_memory", "mtl", "caudal_acc"]
SAFE = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
SETS = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post1": ["post1"], "post2": ["post2"],
        "post": ["post1", "post2"]}


def T(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.ttest_1samp(v, 0)[0] if len(v) > 9 else np.nan


def load(roi, extra=()):
    """Object trials x TR of one ROI: probabilities (pp) and requested extra columns, both sessions."""
    parts = []
    for s in ("presleep", "postsleep"):
        d = pd.read_csv(SODA.format(s=s, r=roi), usecols=["subject", "trial", "tr", "object_scene_retrieval",
                        "retrieved_position", "retrieved_probs", "control_probs", "behav_correct", *extra],
                        low_memory=False)
        d = d[d.object_scene_retrieval == "object"].reset_index(drop=True)
        d["session"] = s
        d["cond"] = np.where(s == "presleep", "pre", np.where(d.trial <= 70, "post1", "post2"))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    R = 100 * np.array([eval(x, SAFE) for x in d.retrieved_probs], float)
    C = 100 * np.array([eval(x, SAFE) for x in d.control_probs], float)
    return d.drop(columns=["retrieved_probs", "control_probs"]), R, C


def cond_table(d, col):
    m = d.dropna(subset=[col]).groupby(["subject", "cond"])[col].mean().unstack()
    m["post"] = m[["post1", "post2"]].mean(axis=1)
    m["pooled"] = m[["pre", "post1", "post2"]].mean(axis=1)
    return m


# ---------------------------------------------------------------------------
def part_a(roi):
    cols = [f"{a}_{v}{s}" for v in ("soda", "soda_trunc") for a in ("retrieved", "control") for s in ("", "_abs")]
    d, R, C = load(roi, cols)
    keep = (d.retrieved_position.isin([3, 4]) & (d.tr >= 3) & (d.tr <= 8)).values
    d, E = d[keep].reset_index(drop=True), (R - C)[keep]
    p, k = d.retrieved_position.astype(int).values, np.arange(keep.sum())
    d["retrieved item"] = E[k, p - 1]
    d["predecessor"] = E[k, p - 2]
    d["two before"] = E[k, p - 3]
    d["successor"] = E[k, p]
    d["predecessor - successor"] = d["predecessor"] - d["successor"]
    for lab, v in (("five items", "soda"), ("up to retrieved", "soda_trunc")):
        d[f"|slope| {lab}"] = d[f"retrieved_{v}_abs"] - d[f"control_{v}_abs"]
        d[f"slope {lab}"] = d[f"retrieved_{v}"] - d[f"control_{v}"]
    measures = ["retrieved item", "predecessor", "two before", "successor", "predecessor - successor",
                "|slope| five items", "|slope| up to retrieved", "slope five items", "slope up to retrieved"]
    rows = []
    for sel, dd in (("positions 3-4", d), ("position 3", d[d.retrieved_position == 3]),
                    ("position 4", d[d.retrieved_position == 4]),
                    ("positions 3-4, correct", d[d.behav_correct.astype(bool)])):
        for m_ in measures:
            m = cond_table(dd, m_)
            rows.append(dict(roi=roi, positions=sel, measure=m_, n=int(m.pooled.notna().sum()),
                             **{f"t_{c}": T(m[c]) for c in SETS},
                             **{"t_post1-pre": T(m.post1 - m.pre), "t_post-pre": T(m.post - m.pre)}))
    return pd.DataFrame(rows)


def control_trials(roi):
    """Per object trial: measures of the control sequence (and of the retrieved one for reference)."""
    d, R, C = load(roi, ("retrieved_soda_abs", "control_soda_abs"))
    p, k = d.retrieved_position.astype(int).values, np.arange(len(d))
    oth = np.arange(1, 6)[None, :] != p[:, None]
    d["target (rel. chance)"] = R[k, p - 1] - 10
    d["other retrieved items (rel. chance)"] = np.where(oth, R, 0).sum(1) / 4 - 10
    d["control same position (rel. chance)"] = C[k, p - 1] - 10
    d["control other items (rel. chance)"] = np.where(oth, C, 0).sum(1) / 4 - 10
    d["control items mean (rel. chance)"] = C.mean(1) - 10
    d["control peak probability"] = C.max(1)
    d["control spread over items"] = C.std(1)
    d["control |slope| five items"] = 100 * d.control_soda_abs
    d["retrieved |slope| five items"] = 100 * d.retrieved_soda_abs
    return d


B_MEASURES = ["target (rel. chance)", "other retrieved items (rel. chance)", "control same position (rel. chance)",
              "control other items (rel. chance)", "control peak probability", "control spread over items"]
C_MEASURES = ["control |slope| five items", "retrieved |slope| five items", "control sinusoid five items",
              "retrieved sinusoid five items", "control peak probability", "control spread over items",
              "control items mean (rel. chance)"]


def part_bc(roi):
    d = control_trials(roi)
    b_rows = []
    for win, dd in (("TR 3-8", d[(d.tr >= 3) & (d.tr <= 8)]), ("all TRs", d)):
        for trials, de in (("all", dd), ("correct", dd[dd.behav_correct.astype(bool)])):
            for m_ in B_MEASURES:
                m = cond_table(de, m_)
                b_rows.append(dict(roi=roi, window=win, trials=trials, measure=m_, mean_pre=m.pre.mean(),
                                   mean_post1=m.post1.mean(), mean_post=m.post.mean(), t_pre=T(m.pre),
                                   t_post1=T(m.post1), **{"t_post1-pre": T(m.post1 - m.pre), "t_post-pre": T(m.post - m.pre)}))
    # part C: one value per trial (all TRs), plus the sinusoid amplitudes
    tri = d.groupby(["subject", "session", "trial", "cond"], as_index=False).mean(numeric_only=True)
    tri["correct"] = tri.behav_correct > 0.5
    sin = []
    for s in ("presleep", "postsleep"):
        f = pd.read_csv(SINUS.format(s=s, r=roi), usecols=["subject", "trial", "retrieved_w_amplitude", "control_w_amplitude"])
        f["session"] = s
        f["control sinusoid five items"] = f.control_w_amplitude.abs()
        f["retrieved sinusoid five items"] = f.retrieved_w_amplitude.abs()
        sin.append(f)
    tri = tri.merge(pd.concat(sin)[["subject", "session", "trial", "control sinusoid five items",
                                    "retrieved sinusoid five items"]], on=["subject", "session", "trial"], how="left")
    acc = {c: tri[tri.cond.isin(cs)].groupby("subject").correct.mean() for c, cs in SETS.items()}
    mean = {c: tri[tri.cond.isin(cs)].groupby("subject")[C_MEASURES].mean() for c, cs in SETS.items()}
    low = acc["pooled"].idxmin()
    n = tri.groupby(["subject", "correct"]).size().unstack(fill_value=0)
    ok = n[(n[True] >= 5) & (n[False] >= 5)].index
    g = tri[tri.subject.isin(ok)].groupby(["subject", "correct"])[C_MEASURES].mean()
    dif = g.xs(True, level="correct") - g.xs(False, level="correct")
    c_rows = []
    for m_ in C_MEASURES:
        row = dict(roi=roi, measure=m_)
        for c in ("pooled", "pre", "post"):
            a, v = acc[c].reindex(mean[c].index), mean[c][m_]
            kk = v.notna() & a.notna()
            row[f"r_{c}"], row[f"p_{c}"] = stats.pearsonr(v[kk], a[kk])
            if c == "pooled":
                row["rho_pooled"] = stats.spearmanr(v[kk], a[kk])[0]
                k2 = kk & (v.index != low)
                row["r_pooled_without_lowest"] = stats.pearsonr(v[k2], a[k2])[0]
        row["t_correct_minus_incorrect"] = T(dif[m_])
        for a_, b_ in (("post1", "pre"), ("post", "pre")):
            dv = mean[a_][m_] - mean[b_][m_]
            da = (acc[a_] - acc[b_]).reindex(dv.index)
            kk = dv.notna() & da.notna()
            row[f"change_r_{a_}-{b_}"], row[f"change_p_{a_}-{b_}"] = stats.pearsonr(dv[kk], da[kk])
        c_rows.append(row)
    return pd.DataFrame(b_rows), pd.DataFrame(c_rows)


def part_d():
    regions = {"PRECUNEUS": "Precuneus", "PARIETAL_PRECUNEUS": "Parietal + precuneus", "RSC": "Retrosplenial",
               "CAUDAL_ACC": "Caudal ACC", "PHPC": "Parahippocampal", "HPC": "Hippocampus", "VIS": "Visual"}
    out = []
    for folder, lab in (("15_univariate_session", "fixed 3 s"), ("15_univariate_session_lss_rt", "cue to response")):
        f = opj(cfg.OUT_ROOT, folder, "trial_roi_means.csv")
        if not os.path.exists(f):
            print("missing", f)
            continue
        t = pd.read_csv(f).dropna(subset=["rt"])
        for kind in ("object", "scene"):
            rows = []
            for (sub, cond), g in t[t.kind == kind].groupby(["subject", "cond"]):
                if len(g) >= 10 and g.rt.std() > 0:
                    for r in regions:
                        rows.append(dict(subject=sub, cond=cond, roi=r, slope=np.polyfit(g.rt - g.rt.mean(), g[r], 1)[0],
                                         r=np.corrcoef(g.rt, g[r])[0, 1]))
            d = pd.DataFrame(rows)
            for r, rl in regions.items():
                w = d[d.roi == r].pivot(index="subject", columns="cond", values="slope")
                wr = d[d.roi == r].pivot(index="subject", columns="cond", values="r")
                row = dict(events=lab, kind=kind, region=rl)
                for c in ("pre", "post1", "post2"):
                    row[f"t_{c}"], row[f"slope_{c}"], row[f"mean_r_{c}"] = T(w[c]), w[c].mean(), wr[c].mean()
                row["t_post1-pre"], row["t_post2-post1"] = T(w.post1 - w.pre), T(w.post2 - w.post1)
                out.append(row)
    return pd.DataFrame(out)


XPOS = np.arange(1, 6)[None, :].astype(float)
POS_GROUPS = (("all positions", [1, 2, 3, 4, 5]), ("positions 1-2", [1, 2]), ("positions 3-4", [3, 4]), ("position 5", [5]))


def _slope(Y, keep):
    n = keep.sum(1, keepdims=True)
    xm, ym = (XPOS * keep).sum(1, keepdims=True) / n, (Y * keep).sum(1, keepdims=True) / n
    den = (((XPOS - xm) ** 2) * keep).sum(1)
    return np.where(den > 0, ((XPOS - xm) * (Y - ym) * keep).sum(1) / np.where(den > 0, den, 1), np.nan)


def _tp(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    return stats.ttest_1samp(v, 0)


def part_ef(roi):
    d, R, C = load(roi)
    R, C = R / 100, C / 100
    p = d.retrieved_position.astype(int).values
    k, pos = np.arange(len(d)), np.arange(1, 6)[None, :]
    keeps = {"five items": np.ones_like(R, bool), "without retrieved item": pos != p[:, None]}
    keeps["without retrieved item and predecessor"] = keeps["without retrieved item"] & (pos != p[:, None] - 1)
    keeps["without retrieved item and both neighbours"] = keeps["without retrieved item and predecessor"] & (pos != p[:, None] + 1)
    for name, kp in keeps.items():
        d[f"{name}|ret"], d[f"{name}|ctl"] = 100 * np.abs(_slope(R, kp)), 100 * np.abs(_slope(C, kp))
    e_rows = []
    for sel, plist in POS_GROUPS:
        dd = d[d.retrieved_position.isin(plist)]
        for name in keeps:
            g = dd.groupby(["subject", "cond"])[[f"{name}|ret", f"{name}|ctl"]].mean()
            r, c = g[f"{name}|ret"].unstack(), g[f"{name}|ctl"].unstack()
            df = r - c
            e_rows.append(dict(roi=roi, positions=sel, items=name, t_pre=T(df.pre), t_post1=T(df.post1),
                               t_post2=T(df.post2), t_change=T(df.post1 - df.pre),
                               t_change_retrieved=T(r.post1 - r.pre), t_change_control=T(c.post1 - c.pre)))
    w = ((d.tr >= 3) & (d.tr <= 8) & (p <= 4)).values                    # successor, TRs 3-8
    E = 100 * (R - C)
    s = pd.DataFrame(dict(subject=d.subject.values[w], cond=d.cond.values[w],
                          correct=d.behav_correct.astype(bool).values[w], succ=E[k, np.clip(p, 0, 4)][w]))
    out = {trials: x.groupby(["subject", "cond"]).succ.mean().unstack()
           for trials, x in (("all", s), ("correct", s[s.correct]))}
    return pd.DataFrame(e_rows), out


def part_g():
    base = opj(cfg.OUT_ROOT, "19_without_target_sinusoid", "B9_thesis")
    if not os.path.isdir(base):
        print("missing", base, "- run 19_without_target_sinusoid.sh")
        return pd.DataFrame()
    res = {}
    for roi in MAIN:
        parts = []
        for s in ("presleep", "postsleep"):
            f = pd.read_csv(opj(base, roi, f"{roi}_windowed_fit_results_{s}.csv"))
            f = f[f.object_scene_retrieval == "object"]
            parts.append(pd.DataFrame({
                "subject": f.subject.values, "position": f.retrieved_position.astype(int).values,
                "cond": np.where(s == "presleep", "pre", np.where(f.trial.values <= 70, "post1", "post2")),
                "five items|ret": f.retrieved_w_amplitude.abs().values, "five items|ctl": f.control_w_amplitude.abs().values,
                "without retrieved item|ret": f.wo_retrieved_w_amplitude.abs().values,
                "without retrieved item|ctl": f.wo_control_w_amplitude.abs().values}))
        res[roi] = pd.concat(parts, ignore_index=True)
    rows = []
    for sel, plist in POS_GROUPS:
        for name in ("five items", "without retrieved item"):
            ch, chp = {}, {}
            for roi in MAIN:
                x = res[roi][res[roi].position.isin(plist)]
                g = x.groupby(["subject", "cond"])[[f"{name}|ret", f"{name}|ctl"]].mean()
                r, c = g[f"{name}|ret"].unstack(), g[f"{name}|ctl"].unstack()
                df = r - c
                df["post"] = df[["post1", "post2"]].mean(axis=1)
                ch[roi], chp[roi] = df.post1 - df.pre, df.post - df.pre
                (t1, p1), (t2, p2) = _tp(ch[roi]), _tp(chp[roi])
                rows.append(dict(positions=sel, items=name, roi=roi, compared_with="", t_pre=T(df.pre),
                                 t_post1=T(df.post1), t_post2=T(df.post2), t_change_post1=t1, p_change_post1=p1,
                                 t_change_post=t2, p_change_post=p2, t_change_retrieved=T(r.post1 - r.pre),
                                 t_change_control=T(c.post1 - c.pre)))
            for other in MAIN[1:]:
                (t1, p1), (t2, p2) = _tp(ch[MAIN[0]] - ch[other]), _tp(chp[MAIN[0]] - chp[other])
                rows.append(dict(positions=sel, items=name, roi=MAIN[0], compared_with=other,
                                 t_change_post1=t1, p_change_post1=p1, t_change_post=t2, p_change_post=p2))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ef = Parallel(n_jobs=min(N_JOBS, len(MAIN)))(delayed(part_ef)(r) for r in MAIN)
    pd.concat([x[0] for x in ef], ignore_index=True).to_csv(opj(OUT_DIR, "E_slope_without_target.csv"), index=False)
    f_rows = []
    for trials in ("all", "correct"):
        ch = {}
        for roi, x in zip(MAIN, ef):
            m = x[1][trials]
            ch[roi] = m.post1 - m.pre
            t, p = _tp(ch[roi])
            f_rows.append(dict(trials=trials, roi=roi, compared_with="", t_pre=T(m.pre), t_post1=T(m.post1),
                               t_post2=T(m.post2), t_change=t, p_change=p))
        for other in MAIN[1:]:
            t, p = _tp(ch[MAIN[0]] - ch[other])
            f_rows.append(dict(trials=trials, roi=MAIN[0], compared_with=other, t_change=t, p_change=p))
    pd.DataFrame(f_rows).to_csv(opj(OUT_DIR, "F_successor_change.csv"), index=False)
    part_g().to_csv(opj(OUT_DIR, "G_sinusoid_without_target.csv"), index=False)
    a = Parallel(n_jobs=min(N_JOBS, len(ROIS_A)))(delayed(part_a)(r) for r in ROIS_A)
    pd.concat(a, ignore_index=True).to_csv(opj(OUT_DIR, "A_positions_3_4.csv"), index=False)
    bc = Parallel(n_jobs=min(N_JOBS, len(MAIN)))(delayed(part_bc)(r) for r in MAIN)
    pd.concat([x[0] for x in bc], ignore_index=True).to_csv(opj(OUT_DIR, "B_control_sequence.csv"), index=False)
    pd.concat([x[1] for x in bc], ignore_index=True).to_csv(opj(OUT_DIR, "C_control_vs_behaviour.csv"), index=False)
    part_d().to_csv(opj(OUT_DIR, "D_rt_relation_per_session.csv"), index=False)
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
