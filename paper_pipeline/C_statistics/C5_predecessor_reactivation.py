#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C5 - Is the neighbour before the retrieved item reactivated (forward stepping)?

Question: if item k of a sequence is reached by stepping through the items
before it, the retrieval of item k should carry its predecessor (item k-1) but
not its successor (item k+1). C2 (analysis 6) averages all preceding items on
trials at positions 3-5 and finds nothing; this step separates the items by
their lag to the retrieved item.

Evidence for the item at lag l = (item position - retrieved position):
    P(item of the retrieved sequence) - P(item at the same position of the
    non-retrieved sequence), in percentage points of classifier probability,
    averaged over TRs SIGNAL_TRS after the cue. The classifiers are the ones
    trained on the localizers (B4), so the item templates are independent of
    the retrieval sessions.

Analyses (per subject, then across subjects; object and scene trials apart;
all trials and correct trials only)
    1  Lag profile: evidence at every lag -4..+4 (lag 0 = retrieved item).
    2  Retrieved item (lag 0), predecessor (lag -1), successor (lag +1) and predecessor - successor on
       trials at positions 2-4, where both neighbours exist. Also all preceding
       minus all following items on the same trials.
       Correction: `p_fdr_main` across the four main ROIs (vis,
       parietal_precuneus, mtl and the control region caudal_acc);
       `p_fdr_rois` across all twelve ROIs.
    3  Session change of the measures in 2: paired tests between evening (pre),
       morning pass 1 (post1), morning pass 2 (post2) and morning (mean of the
       two passes). FDR across ROIs.
    4  Evidence per TR for lag -1, 0 and +1 (positions 2-4, sessions pooled).
    5  Upcoming versus preceding items with a horizon: discounted mean of the
       items after the retrieved one (weight gamma^(k-1) for the item k steps
       ahead), of the items before it, and their difference, on trials at
       positions 2-4; gamma = 0.01 (immediate neighbour only), 0.5 and 1.0
       (all items equal). A successor-like (predictive) code predicts forward
       evidence, stepping through the sequence predicts backward evidence.
    6  Region difference: forward-minus-backward of each ROI against every
       other ROI (paired across subjects), sessions pooled and per session.

Conditions: pre = evening; post1 = morning trials 1-70; post2 = trials 71-140.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
Output  outputs/C5_predecessor_reactivation/*.csv
Plots   plots/C5_predecessor_reactivation/*.pdf and *.png

Usage:  python C5_predecessor_reactivation.py

Not a port: added on 2026-11-19. First run as an exploratory analysis in
7_rsa_exploratory/09_directional.py (part B); the numbers of analyses 1, 2 and 4
are the same. Analyses 5 and 6 were added the same day and first run in
7_rsa_exploratory/13_successor_test.py (part A); the numbers of 5 are the same.
"""

import os
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

ALL_ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl",
            "superior_frontal", "rsc", "posterior_cingulate", "medial_ofc",
            "rostral_acc", "caudal_acc"]
# Main ROIs (decided 2026-11-19): three regions of interest and one control region.
# The control, caudal ACC, has a classifier at chance in cross-validation (about 10.5 %),
# so it should show none of the effects; it guards against effects that the analysis
# itself could produce. Column `p_fdr_main` corrects across these four only;
# `p_fdr_rois` across all twelve.
MAIN_ROIS = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]
FIG_ROIS = MAIN_ROIS
ROI_LABELS = {"vis": "Visual", "cortical_memory": "Cortical memory",
              "parietal_precuneus": "Parietal + precuneus", "parietal": "Parietal",
              "precuneus": "Precuneus", "mtl": "MTL", "superior_frontal": "Sup. frontal",
              "rsc": "RSC", "posterior_cingulate": "Post. cingulate", "medial_ofc": "Medial OFC",
              "rostral_acc": "Rostral ACC", "caudal_acc": "Caudal ACC"}

SIGNAL_TRS = (3, 8)          # inclusive, as in C2
POST1_MAX_TRIAL = 70
MID_POSITIONS = [2, 3, 4]    # retrieved positions that have a predecessor and a successor

CONDS = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post": ["post1", "post2"],
         "post1": ["post1"], "post2": ["post2"]}
COND_LABELS = {"pre": "Evening", "post1": "Morning\npass 1", "post2": "Morning\npass 2"}
# The retrieved item is always reported next to its neighbours, on the same trials.
MEASURES = ["retrieved item", "predecessor", "successor", "predecessor - successor",
            "all preceding - all following"]
SESSION_CONTRASTS = [("post1", "pre"), ("post", "pre"), ("post2", "post1")]
GAMMAS = [0.01, 0.5, 1.0]    # horizon of analysis 5

SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


# =====================================================================
# HELPERS
# =====================================================================

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
    """FDR across all ROIs (p_fdr_rois) and across the four main ROIs (p_fdr_main)."""
    df["p_fdr_rois"] = df.groupby(by)["p"].transform(_fdr)
    df["p_fdr_main"] = np.nan
    m = df["roi"].isin(MAIN_ROIS)
    df.loc[m, "p_fdr_main"] = df[m].groupby(by)["p"].transform(_fdr)
    return df


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), sem(v) * t_dist.ppf(0.975, len(v) - 1)


def _save(fig, name):
    fig.savefig(opj(PLOT_DIR, f"{name}.pdf"))
    fig.savefig(opj(PLOT_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def setup_style():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


# =====================================================================
# DATA
# =====================================================================

def load_lag_evidence(roi: str) -> pd.DataFrame:
    """One row per subject x trial x TR x item of the sequence, with the item's lag to the target."""
    parts, horizon = [], []
    for session in ["presleep", "postsleep"]:
        path = opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv")
        d = pd.read_csv(path, usecols=["subject", "trial", "tr", "object_scene_retrieval",
                                       "retrieved_position", "retrieved_probs", "control_probs",
                                       "behav_correct"], low_memory=False)
        R = 100.0 * np.array([eval(x, SAFE_EVAL) for x in d["retrieved_probs"]], dtype=float)   # (n, 5)
        C = 100.0 * np.array([eval(x, SAFE_EVAL) for x in d["control_probs"]], dtype=float)     # (n, 5)
        p = d["retrieved_position"].astype(int).values
        cond = np.where(session == "presleep", "pre",
                        np.where(d["trial"].values <= POST1_MAX_TRIAL, "post1", "post2"))
        # analysis 5: per trial x TR, discounted evidence ahead of and behind the retrieved item
        lagm = np.arange(1, 6)[None, :] - p[:, None]
        E = R - C
        hz = pd.DataFrame(dict(subject=d["subject"].values, cond=cond, tr=d["tr"].values, position=p,
                               kind=d["object_scene_retrieval"].values,
                               correct=d["behav_correct"].astype(bool).values))
        for g in GAMMAS:
            for name, m in (("forward", lagm > 0), ("backward", lagm < 0)):
                w = np.where(m, g ** (np.abs(lagm) - 1.0), 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    hz[f"{name}_{g}"] = np.where(w.sum(1) > 0, (w * E).sum(1) / w.sum(1), np.nan)
            hz[f"forward - backward_{g}"] = hz[f"forward_{g}"] - hz[f"backward_{g}"]
        horizon.append(hz)
        for j in range(1, 6):
            parts.append(pd.DataFrame(dict(
                subject=d["subject"].values, cond=cond, kind=d["object_scene_retrieval"].values,
                tr=d["tr"].values, position=p, correct=d["behav_correct"].astype(bool).values,
                lag=j - p, ev=R[:, j - 1] - C[:, j - 1], ret=R[:, j - 1], ctl=C[:, j - 1])))
    out = pd.concat(parts, ignore_index=True)
    out["roi"] = roi
    hz = pd.concat(horizon, ignore_index=True)
    hz["roi"] = roi
    return out, hz


def subject_measures(w: pd.DataFrame) -> pd.DataFrame:
    """Per roi / kind / subject: the four measures of analysis 2 (rows = subjects)."""
    mid = w[w["position"].isin(MID_POSITIONS)]
    nb = mid[mid["lag"].isin([-1, 0, 1])].groupby(["roi", "kind", "subject", "lag"])["ev"].mean().unstack("lag")
    side = mid[mid["lag"] != 0].assign(side=lambda x: np.where(x["lag"] < 0, "prec", "foll"))
    sd = side.groupby(["roi", "kind", "subject", "side"])["ev"].mean().unstack("side")
    out = pd.DataFrame({"retrieved item": nb[0], "predecessor": nb[-1], "successor": nb[1],
                        "predecessor - successor": nb[-1] - nb[1],
                        "all preceding - all following": sd["prec"] - sd["foll"]})
    return out


# =====================================================================
# ANALYSES
# =====================================================================

def analyses(data: pd.DataFrame):
    win = data[(data["tr"] >= SIGNAL_TRS[0]) & (data["tr"] <= SIGNAL_TRS[1])]
    lag_rows, dir_rows, ses_rows, subj_tables = [], [], [], []
    for trials in ("all", "correct"):
        w0 = win if trials == "all" else win[win["correct"]]
        per_cond = {}
        for cname, cs in CONDS.items():
            w = w0[w0["cond"].isin(cs)]
            # 1 lag profile
            s = w.groupby(["roi", "kind", "subject", "lag"], as_index=False)[["ev", "ret", "ctl"]].mean()
            for (roi, kind, lag), g in s.groupby(["roi", "kind", "lag"]):
                lag_rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, lag=lag,
                                     **_test(g["ev"]), retrieved_seq=g["ret"].mean(),
                                     control_seq=g["ctl"].mean()))
            # 2 neighbours
            m = subject_measures(w)
            per_cond[cname] = m
            subj_tables.append(m.reset_index().assign(trials=trials, cond=cname))
            for (roi, kind), g in m.groupby(level=["roi", "kind"]):
                for meas in MEASURES:
                    dir_rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, measure=meas,
                                         **_test(g[meas])))
        # 3 session change (paired: subjects present in both conditions)
        for a, b in SESSION_CONTRASTS:
            diff = (per_cond[a] - per_cond[b]).dropna(how="all")
            for (roi, kind), g in diff.groupby(level=["roi", "kind"]):
                for meas in MEASURES:
                    ses_rows.append(dict(trials=trials, contrast=f"{a} - {b}", roi=roi, kind=kind,
                                         measure=meas, **_test(g[meas])))
    lag = pd.DataFrame(lag_rows)
    dr = pd.DataFrame(dir_rows)
    lag = _add_fdr(lag, ["trials", "cond", "kind", "lag"])
    dr = _add_fdr(dr, ["trials", "cond", "kind", "measure"])
    ses = pd.DataFrame(ses_rows)
    ses = _add_fdr(ses, ["trials", "contrast", "kind", "measure"])

    # 4 per TR
    mid = data[data["position"].isin(MID_POSITIONS) & data["lag"].isin([-1, 0, 1])]
    s = mid.groupby(["roi", "kind", "subject", "tr", "lag"], as_index=False)["ev"].mean()
    tr_rows = [dict(roi=roi, kind=kind, tr=tr, lag=lag, **_test(g["ev"]), sem=sem(g["ev"].dropna()))
               for (roi, kind, tr, lag), g in s.groupby(["roi", "kind", "tr", "lag"])]
    return lag, dr, ses, pd.DataFrame(tr_rows), pd.concat(subj_tables, ignore_index=True)


def horizon_analyses(hz: pd.DataFrame):
    """Analyses 5 and 6 on the per-trial horizon scores."""
    cols = [f"{n}_{g}" for g in GAMMAS for n in ("forward", "backward", "forward - backward")]
    w0 = hz[(hz["tr"] >= SIGNAL_TRS[0]) & (hz["tr"] <= SIGNAL_TRS[1]) & hz["position"].isin(MID_POSITIONS)]
    rows, reg = [], []
    for trials in ("all", "correct"):
        w1 = w0 if trials == "all" else w0[w0["correct"]]
        for cname, cs in CONDS.items():
            s = w1[w1["cond"].isin(cs)].groupby(["roi", "kind", "subject"])[cols].mean()
            for (roi, kind), g in s.groupby(level=["roi", "kind"]):
                for c in cols:
                    name, gamma = c.rsplit("_", 1)
                    rows.append(dict(trials=trials, cond=cname, roi=roi, kind=kind, measure=name,
                                     gamma=float(gamma), **_test(g[c])))
            for g_ in GAMMAS:
                a = s[f"forward - backward_{g_}"].unstack("roi")
                for kind, ak in a.groupby(level="kind"):
                    for i, r1 in enumerate(ALL_ROIS):
                        for r2 in ALL_ROIS[i + 1:]:
                            reg.append(dict(trials=trials, cond=cname, kind=kind, gamma=g_, roi_a=r1, roi_b=r2,
                                            **_test(ak[r1] - ak[r2])))
    hor = pd.DataFrame(rows)
    hor = _add_fdr(hor, ["trials", "cond", "kind", "measure", "gamma"])
    return hor, pd.DataFrame(reg)


# =====================================================================
# FIGURES
# =====================================================================

def fig_lag_profile(data):
    win = data[(data["tr"] >= SIGNAL_TRS[0]) & (data["tr"] <= SIGNAL_TRS[1]) & (data["kind"] == "object")]
    s = win.groupby(["roi", "subject", "lag"], as_index=False)["ev"].mean()
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.0 * len(FIG_ROIS) + 0.6, 2.4), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        g = s[s["roi"] == roi]
        lags = list(range(-4, 5))
        st = [_ci(g[g["lag"] == k]["ev"]) for k in lags]
        cols = [ORANGE if k == 0 else BLUE for k in lags]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        for k, (mu, h), c in zip(lags, st, cols):
            ax.errorbar(k, mu, yerr=h, fmt="o", ms=4, color=c, mec="white", mew=0.7, lw=1.2)
        ax.set_title(ROI_LABELS[roi]); ax.set_xticks([-4, -2, 0, 2, 4])
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Evidence vs control sequence (pp)")
    fig.supxlabel("Item position relative to the retrieved item", fontsize=8, y=-0.04)
    _save(fig, "1_lag_profile_objects")


def fig_neighbours_by_session(subj):
    x = subj[(subj["trials"] == "all") & (subj["kind"] == "object") & subj["cond"].isin(COND_LABELS)]
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.0 * len(FIG_ROIS) + 0.6, 2.5), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        g = x[x["roi"] == roi]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        for off, meas, c in ((-0.22, "predecessor", BLUE), (0.0, "retrieved item", MUTED),
                             (0.22, "successor", ORANGE)):
            for i, cond in enumerate(COND_LABELS):
                mu, h = _ci(g[g["cond"] == cond][meas])
                ax.errorbar(i + off, mu, yerr=h, fmt="o", ms=4.5, color=c, mec="white", mew=0.7, lw=1.2,
                            label=meas.capitalize() if i == 0 else None)
        ax.set_title(ROI_LABELS[roi])
        ax.set_xticks(range(3), list(COND_LABELS.values()), fontsize=7)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Evidence vs control sequence (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    _save(fig, "2_predecessor_successor_by_session")


def fig_by_tr(trtab):
    rois = FIG_ROIS
    fig, axes = plt.subplots(1, len(rois), figsize=(2.3 * len(rois) + 0.5, 2.3), sharey=True)
    for ax, roi in zip(axes, rois):
        g = trtab[(trtab["roi"] == roi) & (trtab["kind"] == "object")]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        ax.axvspan(SIGNAL_TRS[0] - 0.5, SIGNAL_TRS[1] + 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        for lag, c, lab in ((0, MUTED, "Retrieved item"), (-1, BLUE, "Predecessor"), (1, ORANGE, "Successor")):
            y = g[g["lag"] == lag].sort_values("tr")
            ax.errorbar(y["tr"], y["mean"], yerr=y["sem"], fmt="-o", ms=3.2, lw=1.5, color=c,
                        mec="white", mew=0.6, elinewidth=0.8, label=lab)
        ax.set_title(ROI_LABELS[roi]); ax.set_xlabel("TR after cue (1.5 s each)")
    axes[0].set_ylabel("Evidence vs control sequence (pp)")
    axes[-1].legend(loc="upper left")
    _save(fig, "3_evidence_by_tr_objects")


# =====================================================================
# MAIN
# =====================================================================

def main():
    setup_style()
    print("Loading item probabilities...")
    loaded = [load_lag_evidence(roi) for roi in ALL_ROIS]
    data = pd.concat([x[0] for x in loaded], ignore_index=True)
    hz = pd.concat([x[1] for x in loaded], ignore_index=True)
    n_pre = data.query("cond == 'pre'")["subject"].nunique()
    n_post = data.query("cond != 'pre'")["subject"].nunique()
    print(f"  {len(ALL_ROIS)} ROIs, {n_pre} pre-sleep / {n_post} post-sleep subjects")

    lag, dr, ses, trtab, subj = analyses(data)
    hor, reg = horizon_analyses(hz)
    for name, t in {"1_lag_profile": lag, "2_predecessor_successor": dr, "3_session_change": ses,
                    "4_by_tr": trtab, "5_forward_backward": hor, "6_region_difference": reg,
                    "subject_measures": subj}.items():
        t.to_csv(opj(OUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}.csv")

    print("Figures")
    fig_lag_profile(data)
    fig_neighbours_by_session(subj)
    fig_by_tr(trtab)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
