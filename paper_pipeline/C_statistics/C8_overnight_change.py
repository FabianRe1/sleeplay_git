#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C8 - What changes from the evening to the morning, and where?

Brings together the tests behind the overnight change in parietal_precuneus
(C6: sequentiality appears at the first morning retrieval). Object trials, the
four main ROIs (parietal_precuneus, vis, mtl and the control region caudal_acc).
Three time points: pre = evening, post1 = morning trials 1-70, post2 = trials
71-140. Evidence = P(item of the retrieved sequence) - P(item at the same
position of the control sequence), percentage points.

  1  Items. Evidence for the retrieved item, its predecessor and its successor
     (TRs 3-8, every retrieved position that has the item) per time point; change
     post1 - pre, post2 - post1, post - pre; and the change in the first ROI
     against the change in every other ROI (is it specific to that region?).
  2  Which items carry the slope. Absolute slope over all five items, without the
     retrieved item, also without its predecessor, also without both neighbours
     (the control sequence drops the same positions); per time point, change, and
     the change split into the retrieved and the control sequence; also per
     retrieved position (1-2, 3-4, 5).
  3  Order test. The absolute slope is positive for any uneven set of
     probabilities. Here the absolute slope of the retrieved sequence is compared
     with the mean absolute slope over ALL orderings of the same probabilities
     (120 for five items; 24 without the retrieved item). A value above zero means
     the items are arranged along the sequence, beyond their being unequal.
  4  Sinusoid amplitude with and without the retrieved item (B9 preset `thesis`;
     the version without the item needs the `wo_*` columns): per time point,
     change, split into retrieved and control sequence, comparison between ROIs,
     per retrieved position.

t-tests across participants; p-values uncorrected. `p_fdr_main` in table 1 and 4
corrects the per-ROI change across the four ROIs.

Input   outputs/B8_sequentiality_soda/<session>/SODA_results_csv/<roi>_soda_results_per_combination.csv
        outputs/B9_sinusoid_summary/thesis/<roi>/<roi>_windowed_fit_results_<session>.csv
Output  outputs/C8_overnight_change/*.csv
Plots   plots/C8_overnight_change/*.pdf and *.png

Usage:  python C8_overnight_change.py

Not a port: added on 2026-11-19. First run as
7_rsa_exploratory/18_supplementary_checks.py (parts E to G); analyses 1, 2 and 4
give the same numbers. Analysis 3 is new.
"""

import itertools
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
SINUS_DIR = opj(str(step_input_dir("B9_sinusoid_summary")), "thesis")

ROIS = ["parietal_precuneus", "vis", "mtl", "caudal_acc"]        # the first is compared with the others
ROI_LABELS = {"parietal_precuneus": "Parietal + precuneus", "vis": "Visual", "mtl": "MTL",
              "caudal_acc": "Caudal ACC (control)"}
SIGNAL_TRS = (3, 8)
POST1_MAX_TRIAL = 70
CONDS = ["pre", "post1", "post2"]
COND_LABELS = {"pre": "Evening", "post1": "Morning\npass 1", "post2": "Morning\npass 2"}
CONTRASTS = [("post1", "pre"), ("post2", "post1"), ("post", "pre")]
POS_GROUPS = (("all positions", [1, 2, 3, 4, 5]), ("positions 1-2", [1, 2]), ("positions 3-4", [3, 4]),
              ("position 5", [5]))
SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
XPOS = np.arange(1, 6)[None, :].astype(float)
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def _tp(v):
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


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), sem(v) * t_dist.ppf(0.975, len(v) - 1)


def _slope(Y, keep):
    """Regression slope of Y on position over the kept positions (rows = trial x TR)."""
    n = keep.sum(1, keepdims=True)
    xm, ym = (XPOS * keep).sum(1, keepdims=True) / n, (Y * keep).sum(1, keepdims=True) / n
    den = (((XPOS - xm) ** 2) * keep).sum(1)
    return np.where(den > 0, ((XPOS - xm) * (Y - ym) * keep).sum(1) / np.where(den > 0, den, 1), np.nan)


def _permutation_mean_abs_slope(Y, keep):
    """Mean |slope| over all orderings of the kept values on the kept positions."""
    out = np.full(len(Y), np.nan)
    patterns = np.unique(keep, axis=0)
    for pat in patterns:
        rows = np.where((keep == pat).all(1))[0]
        idx = np.where(pat)[0]
        x = XPOS[0, idx]
        xc = x - x.mean()
        den = (xc ** 2).sum()
        if den == 0:
            continue
        y = Y[rows][:, idx]
        acc = np.zeros(len(rows))
        perms = list(itertools.permutations(range(len(idx))))
        for perm in perms:
            acc += np.abs((y[:, list(perm)] * xc).sum(1) / den)
        out[rows] = acc / len(perms)
    return out


def cond_table(d, col):
    m = d.dropna(subset=[col]).groupby(["subject", "cond"])[col].mean().unstack()
    m["post"] = m[["post1", "post2"]].mean(axis=1)
    return m


def load_roi(roi):
    parts = []
    for session in ("presleep", "postsleep"):
        d = pd.read_csv(opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv"),
                        usecols=["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position",
                                 "retrieved_probs", "control_probs", "behav_correct"], low_memory=False)
        d = d[d["object_scene_retrieval"] == "object"].reset_index(drop=True)
        d["cond"] = np.where(session == "presleep", "pre", np.where(d["trial"] <= POST1_MAX_TRIAL, "post1", "post2"))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    R = np.array([eval(x, SAFE_EVAL) for x in d["retrieved_probs"]], dtype=float)
    C = np.array([eval(x, SAFE_EVAL) for x in d["control_probs"]], dtype=float)
    p = d["retrieved_position"].astype(int).values
    k, pos = np.arange(len(d)), np.arange(1, 6)[None, :]
    win = ((d["tr"] >= SIGNAL_TRS[0]) & (d["tr"] <= SIGNAL_TRS[1])).values
    E = 100 * (R - C)
    d["retrieved item"] = np.where(win, E[k, p - 1], np.nan)
    d["predecessor"] = np.where(win & (p >= 2), E[k, np.clip(p - 2, 0, 4)], np.nan)
    d["successor"] = np.where(win & (p <= 4), E[k, np.clip(p, 0, 4)], np.nan)
    keeps = {"five items": np.ones_like(R, bool), "without retrieved item": pos != p[:, None]}
    keeps["without retrieved item and predecessor"] = keeps["without retrieved item"] & (pos != p[:, None] - 1)
    keeps["without retrieved item and both neighbours"] = (keeps["without retrieved item and predecessor"]
                                                           & (pos != p[:, None] + 1))
    for name, kp in keeps.items():
        d[f"{name}|ret"], d[f"{name}|ctl"] = 100 * np.abs(_slope(R, kp)), 100 * np.abs(_slope(C, kp))
    for name in ("five items", "without retrieved item"):
        kp = keeps[name]
        d[f"order {name}|ret"] = d[f"{name}|ret"] - 100 * _permutation_mean_abs_slope(R, kp)
        d[f"order {name}|ctl"] = d[f"{name}|ctl"] - 100 * _permutation_mean_abs_slope(C, kp)
    d["roi"] = roi
    return d.drop(columns=["retrieved_probs", "control_probs"]), list(keeps)


def load_sinus(roi):
    parts = []
    for session in ("presleep", "postsleep"):
        f = pd.read_csv(opj(SINUS_DIR, roi, f"{roi}_windowed_fit_results_{session}.csv"))
        f = f[f["object_scene_retrieval"] == "object"]
        o = pd.DataFrame({"subject": f["subject"].values, "position": f["retrieved_position"].astype(int).values,
                          "cond": np.where(session == "presleep", "pre",
                                           np.where(f["trial"].values <= POST1_MAX_TRIAL, "post1", "post2")),
                          "five items|ret": f["retrieved_w_amplitude"].abs().values,
                          "five items|ctl": f["control_w_amplitude"].abs().values})
        if "wo_retrieved_w_amplitude" in f.columns:
            o["without retrieved item|ret"] = f["wo_retrieved_w_amplitude"].abs().values
            o["without retrieved item|ctl"] = f["wo_control_w_amplitude"].abs().values
        parts.append(o)
    return pd.concat(parts, ignore_index=True)


def region_rows(changes, tag):
    """Change in the first ROI against the change in each other ROI."""
    rows = []
    for other in ROIS[1:]:
        for con, ch in changes.items():
            rows.append(dict(**tag, roi=ROIS[0], compared_with=other, test=f"change {con}",
                             **_tp(ch[ROIS[0]] - ch[other])))
    return rows


def arm_analysis(frames, names, by_position_col):
    """Per-ROI levels, changes, arm split and region comparison for retrieved-minus-control measures."""
    rows = []
    for sel, plist in POS_GROUPS:
        for name in names:
            changes = {f"{a} - {b}": {} for a, b in CONTRASTS}
            for roi in ROIS:
                d = frames[roi]
                if f"{name}|ret" not in d.columns:
                    continue
                g = d[d[by_position_col].isin(plist)].groupby(["subject", "cond"])[[f"{name}|ret", f"{name}|ctl"]].mean()
                r, c = g[f"{name}|ret"].unstack(), g[f"{name}|ctl"].unstack()
                for tab in (r, c):
                    tab["post"] = tab[["post1", "post2"]].mean(axis=1)
                df = r - c
                tag = dict(positions=sel, items=name)
                for cond in CONDS + ["post"]:
                    rows.append(dict(**tag, roi=roi, compared_with="", test=cond, **_tp(df[cond])))
                for a, b in CONTRASTS:
                    changes[f"{a} - {b}"][roi] = df[a] - df[b]
                    rows.append(dict(**tag, roi=roi, compared_with="", test=f"change {a} - {b}", **_tp(df[a] - df[b]),
                                     t_retrieved=_tp(r[a] - r[b])["t"], t_control=_tp(c[a] - c[b])["t"]))
            if all(len(v) == len(ROIS) for v in changes.values()):
                rows += region_rows(changes, dict(positions=sel, items=name))
    out = pd.DataFrame(rows)
    out["p_fdr_main"] = np.nan
    m = out["compared_with"] == ""
    out.loc[m, "p_fdr_main"] = out[m].groupby(["positions", "items", "test"])["p"].transform(_fdr)
    return out


def main():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300})
    print("Loading...")
    frames, keep_names = {}, None
    for roi in ROIS:
        frames[roi], keep_names = load_roi(roi)

    # 1 items
    rows, changes_all = [], {}
    for item in ("retrieved item", "predecessor", "successor"):
        for trials in ("all", "correct"):
            changes = {f"{a} - {b}": {} for a, b in CONTRASTS}
            for roi in ROIS:
                d = frames[roi] if trials == "all" else frames[roi][frames[roi]["behav_correct"].astype(bool)]
                m = cond_table(d, item)
                tag = dict(item=item, trials=trials)
                for cond in CONDS + ["post"]:
                    rows.append(dict(**tag, roi=roi, compared_with="", test=cond, **_tp(m[cond])))
                for a, b in CONTRASTS:
                    changes[f"{a} - {b}"][roi] = m[a] - m[b]
                    rows.append(dict(**tag, roi=roi, compared_with="", test=f"change {a} - {b}", **_tp(m[a] - m[b])))
            rows += region_rows(changes, dict(item=item, trials=trials))
            changes_all[(item, trials)] = changes
    # successor change against predecessor change, first ROI
    for trials in ("all", "correct"):
        s, pr = changes_all[("successor", trials)]["post1 - pre"][ROIS[0]], changes_all[("predecessor", trials)]["post1 - pre"][ROIS[0]]
        rows.append(dict(item="successor minus predecessor", trials=trials, roi=ROIS[0], compared_with="",
                         test="change post1 - pre", **_tp(s - pr)))
    items = pd.DataFrame(rows)
    items["p_fdr_main"] = np.nan
    m = (items["compared_with"] == "") & items["item"].isin(["retrieved item", "predecessor", "successor"])
    items.loc[m, "p_fdr_main"] = items[m].groupby(["item", "trials", "test"])["p"].transform(_fdr)

    # 2 slopes by which items are included; 3 order test
    slopes = arm_analysis(frames, keep_names, "retrieved_position")
    order = arm_analysis(frames, ["order five items", "order without retrieved item"], "retrieved_position")
    # order index of the retrieved sequence alone, against zero
    o_rows = []
    for name in ("order five items", "order without retrieved item"):
        for roi in ROIS:
            g = frames[roi].groupby(["subject", "cond"])[[f"{name}|ret", f"{name}|ctl"]].mean()
            for arm in ("ret", "ctl"):
                w = g[f"{name}|{arm}"].unstack()
                for cond in CONDS:
                    o_rows.append(dict(items=name, roi=roi, sequence="retrieved" if arm == "ret" else "control",
                                       test=cond, **_tp(w[cond])))
    order_arms = pd.DataFrame(o_rows)

    # 4 sinusoid
    sin_frames = {roi: load_sinus(roi) for roi in ROIS}
    sinus = arm_analysis(sin_frames, ["five items", "without retrieved item"], "position")

    for name, tab in {"1_items": items, "2_slope_by_items_included": slopes, "3_order_test": order,
                      "3b_order_test_per_sequence": order_arms, "4_sinusoid_with_and_without_retrieved_item": sinus}.items():
        tab.to_csv(opj(OUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}.csv")

    # figure: items per time point and ROI
    fig, axes = plt.subplots(1, len(ROIS), figsize=(2.05 * len(ROIS) + 0.7, 2.5), sharey=True)
    for ax, roi in zip(axes, ROIS):
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        for off, item, col in ((-0.22, "predecessor", BLUE), (0.0, "retrieved item", MUTED), (0.22, "successor", ORANGE)):
            mtab = cond_table(frames[roi], item)
            for i, cond in enumerate(CONDS):
                mu, h = _ci(mtab[cond])
                ax.errorbar(i + off, mu, yerr=h, fmt="o", ms=4.5, color=col, mec="white", mew=0.7, lw=1.2,
                            label=item.capitalize() if i == 0 else None)
        ax.set_title(ROI_LABELS[roi]); ax.set_xticks(range(3), list(COND_LABELS.values())); ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Evidence vs control sequence (pp)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(opj(PLOT_DIR, "1_items_by_time_point.pdf")); fig.savefig(opj(PLOT_DIR, "1_items_by_time_point.png"))
    plt.close(fig)

    # figure: sinusoid with and without the retrieved item
    names = [n for n in ("five items", "without retrieved item") if f"{n}|ret" in sin_frames[ROIS[0]].columns]
    fig, axes = plt.subplots(1, len(ROIS), figsize=(2.05 * len(ROIS) + 0.7, 2.5), sharey=True)
    for ax, roi in zip(axes, ROIS):
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        for off, name, col in zip((-0.14, 0.14), names, (BLUE, ORANGE)):
            g = sin_frames[roi].groupby(["subject", "cond"])[[f"{name}|ret", f"{name}|ctl"]].mean()
            df = g[f"{name}|ret"].unstack() - g[f"{name}|ctl"].unstack()
            for i, cond in enumerate(CONDS):
                mu, h = _ci(df[cond])
                ax.errorbar(i + off, mu, yerr=h, fmt="o", ms=4.5, color=col, mec="white", mew=0.7, lw=1.2,
                            label=name.capitalize() if i == 0 else None)
        ax.set_title(ROI_LABELS[roi]); ax.set_xticks(range(3), list(COND_LABELS.values())); ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Sinusoid amplitude,\nretrieved - control")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(opj(PLOT_DIR, "2_sinusoid_by_time_point.pdf")); fig.savefig(opj(PLOT_DIR, "2_sinusoid_by_time_point.png"))
    plt.close(fig)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
