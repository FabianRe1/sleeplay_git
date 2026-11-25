#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C6 - Retrieved sequence versus control sequence, per session.

The original analysis compares the classifier evidence for the five elements
of the RETRIEVED sequence with the five elements of the CONTROL sequence (the
category sequence that is not being retrieved). This step reports that
comparison in the same form as C5: the four main ROIs (vis, parietal_precuneus,
mtl and the control region caudal_acc), per session, with direct tests of the
change between sessions and correction across the four main ROIs.

Measures (each for the retrieved sequence, the control sequence and their difference)
    sequence evidence        mean classifier probability of the five elements (pp)
    sequence without target  the same without the retrieved item (the four other
                             elements against the same four positions of the control)
    |slope| five items       absolute SODA slope over all five elements (the thesis metric)
    |slope| without target   absolute slope over the four elements other than the
                             retrieved item, at their original positions; the control
                             sequence drops the same position (B8 `*_wo_target`)
    |slope| up to retrieved  absolute slope over elements 1..p (retrieved position p >= 3)
    slope five items         signed slope (positive = later elements higher)
    slope up to retrieved    signed slope over elements 1..p
    sinusoid five items      absolute amplitude of the windowed sinusoid fitted to the
                             five-item slope time course of the trial (the thesis
                             "SinAmp"; B9 preset `thesis`)
    sinusoid without target  the same model fitted to the slope without the retrieved
                             item (B9 preset `thesis`, `wo_*_w_amplitude`). Control for
                             the raised probability of the retrieved item, which is part
                             of the retrieved sequence only (added 2026-11-19)
    sinusoid up to retrieved the same model fitted to the slope over elements 1..p,
                             p >= 3 (B9 preset `trunc_tapered`; amplitudes are in units
                             of the null standard deviation, see B9)
The slope measures are the slope per TR averaged over the TRs of the window; the
sinusoid measures are one fitted amplitude per trial (window "trial fit").

Variants
    window     all 10 TRs after the cue (as in the original statistics) / TRs 3-8 (as in C2, C5)
    positions  all retrieved positions / positions 3 and 4 only
    trials     all / correct only
    kind       object / scene trials

Tests, per ROI: difference retrieved - control against zero in every condition
(pooled, pre, post1, post2, post), and paired session contrasts (post1 - pre,
post - pre, post2 - post1) of the difference, of the retrieved sequence alone
and of the control sequence alone. `p_fdr_main` corrects across the four main
ROIs, `p_fdr_rois` across all twelve.

Conditions: pre = evening; post1 = morning trials 1-70; post2 = trials 71-140;
post = mean of post1 and post2.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
        outputs/B9_sinusoid_summary/{thesis,trunc_tapered}/<roi>/<roi>_windowed_fit_results_<session>.csv
Output  outputs/C6_sequence_vs_control/*.csv
Plots   plots/C6_sequence_vs_control/*.pdf and *.png

Usage:  python C6_sequence_vs_control.py

Not a port: added on 2026-11-19. The slope measures themselves are computed in
B8 and are also analysed in C1 (mixed models, trial matching) and C3; this step
adds the plain sequence evidence, the session contrasts in one common format
and the control region.
"""

import sys
import warnings
from os.path import join as opj
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t as t_dist, ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

OUT_DIR = str(step_output_dir(__file__))
PLOT_DIR = str(step_plot_dir(__file__))
SODA_DIR = opj(str(step_input_dir("B8_sequentiality_soda")), "{session}", "SODA_results_csv")
SINUS_DIR = str(step_input_dir("B9_sinusoid_summary"))
# measure -> (B9 preset, retrieved column, control column)
SINUS_PRESETS = {
    "sinusoid five items": ("thesis", "retrieved_w_amplitude", "control_w_amplitude"),
    "sinusoid without target": ("thesis", "wo_retrieved_w_amplitude", "wo_control_w_amplitude"),
    "sinusoid up to retrieved": ("trunc_tapered", "retrieved_w_amplitude", "control_w_amplitude"),
}

ALL_ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl",
            "superior_frontal", "rsc", "posterior_cingulate", "medial_ofc",
            "rostral_acc", "caudal_acc"]
MAIN_ROIS = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]      # caudal_acc = control region
ROI_LABELS = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL",
              "caudal_acc": "Caudal ACC (control)"}

POST1_MAX_TRIAL = 70
WINDOWS = {"all TRs": (0, 9), "TR 3-8": (3, 8)}
POSITIONS = {"all positions": [1, 2, 3, 4, 5], "positions 3-4": [3, 4]}
COND_LIST = ["pre", "post1", "post2"]
COND_LABELS = {"pre": "Evening", "post1": "Morning\npass 1", "post2": "Morning\npass 2"}
SESSION_CONTRASTS = [("post1", "pre"), ("post", "pre"), ("post2", "post1")]
MEASURES = ["sequence evidence", "sequence without target", "|slope| five items",
            "|slope| without target", "|slope| up to retrieved", "slope five items",
            "slope up to retrieved"]
SLOPE_COLS = {"|slope| five items": "soda_abs", "|slope| without target": "soda_wo_target_abs",
              "|slope| up to retrieved": "soda_trunc_abs",
              "slope five items": "soda", "slope up to retrieved": "soda_trunc"}

SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def _test(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def _fdr(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def _add_fdr(df, by):
    df["p_fdr_rois"] = df.groupby(by)["p"].transform(_fdr)
    df["p_fdr_main"] = np.nan
    m = df["roi"].isin(MAIN_ROIS)
    df.loc[m, "p_fdr_main"] = df[m].groupby(by)["p"].transform(_fdr)
    return df


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), sem(v) * t_dist.ppf(0.975, len(v) - 1)


def load_roi(roi: str) -> pd.DataFrame:
    """One row per subject x trial x TR with every measure for the retrieved and the control sequence."""
    parts = []
    slope_cols = [f"{a}_{c}" for c in SLOPE_COLS.values() for a in ("retrieved", "control")]
    for session in ["presleep", "postsleep"]:
        path = opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv")
        wanted = ["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                  "retrieved_probs", "control_probs", "behav_correct"] + slope_cols
        d = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        for col in slope_cols:
            if col not in d.columns:           # B8 table written before the column existed
                if roi == ALL_ROIS[0] and session == "presleep":
                    print(f"  NOTE: {col} is not in the B8 tables; that measure is left empty. Rerun B8.")
                d[col] = np.nan
        R = 100.0 * np.array([eval(x, SAFE_EVAL) for x in d["retrieved_probs"]], dtype=float)
        C = 100.0 * np.array([eval(x, SAFE_EVAL) for x in d["control_probs"]], dtype=float)
        p = d["retrieved_position"].astype(int).values
        other = np.arange(1, 6)[None, :] != p[:, None]
        out = pd.DataFrame(dict(
            subject=d["subject"].values, tr=d["tr"].values, position=p,
            kind=d["object_scene_retrieval"].values, correct=d["behav_correct"].astype(bool).values,
            cond=np.where(session == "presleep", "pre",
                          np.where(d["trial"].values <= POST1_MAX_TRIAL, "post1", "post2"))))
        out["sequence evidence|retrieved"] = R.mean(1)
        out["sequence evidence|control"] = C.mean(1)
        out["sequence without target|retrieved"] = np.where(other, R, 0).sum(1) / 4.0
        out["sequence without target|control"] = np.where(other, C, 0).sum(1) / 4.0
        for name, col in SLOPE_COLS.items():
            out[f"{name}|retrieved"] = d[f"retrieved_{col}"].values
            out[f"{name}|control"] = d[f"control_{col}"].values
        parts.append(out)
    x = pd.concat(parts, ignore_index=True)
    x["roi"] = roi
    return x


def load_sinus(roi: str) -> pd.DataFrame:
    """One row per subject x trial with the fitted sinusoid amplitudes (absolute values)."""
    out = None
    for measure, (preset, ret_col, ctrl_col) in SINUS_PRESETS.items():
        parts = []
        for session in ["presleep", "postsleep"]:
            wanted = ["subject", "trial", "object_scene_retrieval", "retrieved_position",
                      "behav_correct", ret_col, ctrl_col]
            d = pd.read_csv(opj(SINUS_DIR, preset, roi, f"{roi}_windowed_fit_results_{session}.csv"),
                            usecols=lambda c: c in wanted)
            for col in (ret_col, ctrl_col):
                if col not in d.columns:       # B9 table written before the column existed
                    if roi == ALL_ROIS[0] and session == "presleep":
                        print(f"  NOTE: {col} is not in the B9 tables; that measure is left empty. Rerun B8 and B9.")
                    d[col] = np.nan
            parts.append(pd.DataFrame({
                "subject": d["subject"].values, "trial": d["trial"].values, "session": session,
                "position": d["retrieved_position"].astype(int).values,
                "kind": d["object_scene_retrieval"].values, "correct": d["behav_correct"].astype(bool).values,
                "cond": np.where(session == "presleep", "pre",
                                 np.where(d["trial"].values <= POST1_MAX_TRIAL, "post1", "post2")),
                f"{measure}|retrieved": d[ret_col].abs().values,
                f"{measure}|control": d[ctrl_col].abs().values}))
        x = pd.concat(parts, ignore_index=True)
        keys = ["subject", "trial", "session", "position", "kind", "correct", "cond"]
        out = x if out is None else out.merge(x, on=keys, how="outer")
    out["roi"] = roi
    return out


def analyses(data: pd.DataFrame, sinus: pd.DataFrame):
    det, ses, subj = [], [], []
    frames = [(wname, data[(data["tr"] >= t0) & (data["tr"] <= t1)], MEASURES)
              for wname, (t0, t1) in WINDOWS.items()]
    frames.append(("trial fit", sinus, list(SINUS_PRESETS)))
    for wname, dw, measures in frames:
        cols = [f"{m}|{s}" for m in measures for s in ("retrieved", "control")]
        for pname, plist in POSITIONS.items():
            dp = dw[dw["position"].isin(plist)]
            for trials in ("all", "correct"):
                d = dp if trials == "all" else dp[dp["correct"]]
                s = d.groupby(["roi", "kind", "subject", "cond"])[cols].mean()       # NaN rows are skipped per column
                s = s.unstack("cond")
                for m in measures:
                    r, c = s[f"{m}|retrieved"].copy(), s[f"{m}|control"].copy()
                    for tab in (r, c):
                        tab["post"] = tab[["post1", "post2"]].mean(axis=1)
                        tab["pooled"] = tab[COND_LIST].mean(axis=1)
                    diff = r - c
                    tag = dict(window=wname, positions=pname, trials=trials, measure=m)
                    for cond in ["pooled", "pre", "post1", "post2", "post"]:
                        for (roi, kind), g in diff.groupby(level=["roi", "kind"]):
                            det.append(dict(**tag, cond=cond, roi=roi, kind=kind,
                                            retrieved=r.loc[g.index, cond].mean(),
                                            control=c.loc[g.index, cond].mean(), **_test(g[cond])))
                    for a, b in SESSION_CONTRASTS:
                        for what, tab in (("retrieved - control", diff), ("retrieved", r), ("control", c)):
                            dd = tab[a] - tab[b]
                            for (roi, kind), g in dd.groupby(level=["roi", "kind"]):
                                ses.append(dict(**tag, contrast=f"{a} - {b}", what=what, roi=roi, kind=kind,
                                                **_test(g)))
                    if trials == "all":
                        long = diff[COND_LIST].stack().rename("diff").reset_index()
                        long["retrieved"] = r[COND_LIST].stack().values
                        long["control"] = c[COND_LIST].stack().values
                        subj.append(long.assign(window=wname, positions=pname, measure=m))
    det = _add_fdr(pd.DataFrame(det), ["window", "positions", "trials", "measure", "cond", "kind"])
    ses = _add_fdr(pd.DataFrame(ses), ["window", "positions", "trials", "measure", "contrast", "what", "kind"])
    return det, ses, pd.concat(subj, ignore_index=True)


def fig_by_session(subj, window, positions, name,
                   meas=("sequence evidence", "sequence without target", "|slope| five items",
                         "|slope| without target", "|slope| up to retrieved")):
    x = subj[(subj["window"] == window) & (subj["positions"] == positions) & (subj["kind"] == "object")]
    meas = list(meas)
    fig, axes = plt.subplots(len(meas), len(MAIN_ROIS), figsize=(2.05 * len(MAIN_ROIS) + 0.8, 1.75 * len(meas)),
                             sharex=True, sharey="row")
    for i, m in enumerate(meas):
        for j, roi in enumerate(MAIN_ROIS):
            ax = axes[i, j]
            g = x[(x["measure"] == m) & (x["roi"] == roi)]
            for off, col, c, lab in ((-0.13, "retrieved", BLUE, "Retrieved sequence"),
                                     (0.13, "control", ORANGE, "Control sequence")):
                for k, cond in enumerate(COND_LIST):
                    mu, h = _ci(g[g["cond"] == cond][col])
                    ax.errorbar(k + off, mu, yerr=h, fmt="o", ms=4.2, color=c, mec="white", mew=0.7, lw=1.2,
                                label=lab if k == 0 else None)
            if i == 0:
                ax.set_title(ROI_LABELS[roi])
            if j == 0:
                unit = "(pp)" if m.startswith("sequence") else ("(amplitude)" if m.startswith("sinusoid")
                                                                 else "(pp per element)")
                ax.set_ylabel(m + "\n" + unit, fontsize=7)
            ax.set_xticks(range(3), list(COND_LABELS.values()), fontsize=6.5)
            ax.set_xlim(-0.5, 2.5)
            ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(opj(PLOT_DIR, f"{name}.pdf")); fig.savefig(opj(PLOT_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def main():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
    print("Loading...")
    data = pd.concat([load_roi(r) for r in ALL_ROIS], ignore_index=True)
    # the slopes are stored as probability per element; show them in percentage points like the evidence
    for m in SLOPE_COLS:
        for s in ("retrieved", "control"):
            data[f"{m}|{s}"] = 100.0 * data[f"{m}|{s}"]
    sinus = pd.concat([load_sinus(r) for r in ALL_ROIS], ignore_index=True)
    det, ses, subj = analyses(data, sinus)
    for name, t in {"1_retrieved_vs_control": det, "2_session_change": ses, "subject_values": subj}.items():
        t.to_csv(opj(OUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}.csv")
    print("Figures")
    fig_by_session(subj, "all TRs", "all positions", "1_by_session_objects_all_positions")
    fig_by_session(subj, "all TRs", "positions 3-4", "2_by_session_objects_positions_3_4")
    fig_by_session(subj, "TR 3-8", "all positions", "3_by_session_objects_all_positions_TR3-8")
    fig_by_session(subj, "trial fit", "all positions", "4_sinusoid_by_session_objects_all_positions",
                   meas=list(SINUS_PRESETS))
    fig_by_session(subj, "trial fit", "positions 3-4", "5_sinusoid_by_session_objects_positions_3_4",
                   meas=list(SINUS_PRESETS))
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
