# -*- coding: utf-8 -*-
"""Item-level reactivation during retrieval — no slopes, no sinusoids.

Works directly on the classifier probabilities of the sequence items, so that
the replay results (4.6 / 4.7 / 5.2) can be decomposed into what drives them.

Inputs
------
4.6_SODA_based_on_singlesubj_4.4_{presleep,postsleep}/SODA_results_csv/
    <roi>_soda_results_per_combination.csv
    `retrieved_probs` = classifier probabilities of the 5 items of the retrieved
    sequence, in sequence order; `control_probs` = the 5 items of the
    non-retrieved sequence. Together they are all 10 classes of the classifier,
    so P(other classes) can be computed without the 4.4 tables.

Analyses (all per subject, then across subjects)
------------------------------------------------
1  Target timecourse: P(target) − mean P(other 9 classes), per TR.
2  Retrieved item vs the item at the SAME position of the non-retrieved
   sequence, TR window SIGNAL_TRS, per session / post-sleep pass.
3  Sleep: raw P(retrieved item) and P(control item), pre vs post pass 1
   (trial-matched) and pass 1 vs pass 2.
4  Retrieved position 1-5: RM-ANOVA, linear and quadratic trends.
5  Objects vs scenes, and whether the sleep change differs between them.
6  Preceding items 1..p-1 (p >= 3) vs the same positions of the control
   sequence; target vs control; preceding vs following items of the retrieved
   sequence; per-TR profile of preceding items vs target.

Conditions: pre = pre-sleep; post1 = post-sleep trials 1-70 (first pass through
the 70 questions); post2 = trials 71-140 (repeat); post = post1 + post2.

Outputs: 5.3_item_reactivation/  (CSV tables + PDF figures)

Usage
-----
    python 5.3_item_reactivation.py
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = opj(SCRIPT_DIR, "5.3_item_reactivation")
SODA_DIR = opj(SCRIPT_DIR, "4.6_SODA_based_on_singlesubj_4.4_{session}", "SODA_results_csv")

ALL_ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl",
            "superior_frontal", "rsc", "posterior_cingulate", "medial_ofc",
            "rostral_acc", "caudal_acc"]
# ROIs shown in the detailed figures (the ones with decoding evidence)
FIG_ROIS = ["vis", "cortical_memory", "parietal_precuneus", "parietal", "precuneus", "mtl"]
ROI_LABELS = {"vis": "Visual", "cortical_memory": "Cortical memory",
              "parietal_precuneus": "Parietal + precuneus", "parietal": "Parietal",
              "precuneus": "Precuneus", "mtl": "MTL", "superior_frontal": "Sup. frontal",
              "rsc": "RSC", "posterior_cingulate": "Post. cingulate", "medial_ofc": "Medial OFC",
              "rostral_acc": "Rostral ACC", "caudal_acc": "Caudal ACC"}

SIGNAL_TRS = (3, 8)          # inclusive; retrieval response rises at TR 3, decays after ~TR 8
POST1_MAX_TRIAL = 70         # post-sleep trials 1-70 = first pass, 71-140 = repeat

CONDS = {"pooled": ["pre", "post1", "post2"], "pre": ["pre"], "post": ["post1", "post2"],
         "post1": ["post1"], "post2": ["post2"]}
COND_LABELS = {"pre": "pre", "post1": "post\npass 1", "post2": "post\npass 2"}
COND_COLORS = {"pre": "#7BAFD4", "post1": "#D4534E", "post2": "#E8A09C"}

SAFE_EVAL = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}


# =====================================================================
# STYLE / HELPERS
# =====================================================================

def setup_style():
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def _save(fig, name):
    fig.savefig(opj(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.pdf")


def _sig(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def _paired(a, b):
    """Paired t over common subjects; returns (mean diff, t, p, n)."""
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) < 3:
        return np.nan, np.nan, np.nan, len(idx)
    t, p = ttest_rel(a[idx], b[idx])
    return float((a[idx] - b[idx]).mean()), float(t), float(p), len(idx)


def _fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


# =====================================================================
# DATA
# =====================================================================

def load_item_probabilities(roi: str) -> pd.DataFrame:
    """One row per subject x trial x TR with target / control / preceding / following probabilities."""
    parts = []
    for session in ["presleep", "postsleep"]:
        path = opj(SODA_DIR.format(session=session), f"{roi}_soda_results_per_combination.csv")
        d = pd.read_csv(path, usecols=["subject", "trial", "tr", "object_scene_retrieval",
                                       "retrieved_position", "retrieved_probs", "control_probs"],
                        low_memory=False)
        R = np.array([eval(x, SAFE_EVAL) for x in d["retrieved_probs"]], dtype=float)   # (n, 5)
        C = np.array([eval(x, SAFE_EVAL) for x in d["control_probs"]], dtype=float)     # (n, 5)
        p = d["retrieved_position"].astype(int).values
        k = np.arange(len(d))
        idx = np.arange(5)[None, :]
        prec = idx < (p[:, None] - 1)
        foll = idx > (p[:, None] - 1)

        def masked_mean(A, M):
            n = M.sum(1)
            out = np.where(M, A, 0.0).sum(1) / np.maximum(n, 1)
            out[n == 0] = np.nan
            return out

        out = d[["subject", "trial", "tr", "object_scene_retrieval", "retrieved_position"]].copy()
        out["target"] = R[k, p - 1]
        out["control_item"] = C[k, p - 1]                              # same position, other sequence
        out["others"] = (np.nansum(R, 1) + np.nansum(C, 1) - out["target"]) / 9.0   # other 9 classes
        out["prec_ret"] = masked_mean(R, prec)                         # items 1..p-1, retrieved seq
        out["prec_ctl"] = masked_mean(C, prec)                         # items 1..p-1, control seq
        out["foll_ret"] = masked_mean(R, foll)                         # items p+1..5, retrieved seq
        out["session"] = session
        parts.append(out)
    df = pd.concat(parts, ignore_index=True)
    df["cond"] = np.where(df["session"] == "presleep", "pre",
                          np.where(df["trial"] <= POST1_MAX_TRIAL, "post1", "post2"))
    for c in ["target", "control_item", "others", "prec_ret", "prec_ctl", "foll_ret"]:
        df[c] = 100.0 * df[c]                                          # percent (chance = 10)
    df["roi"] = roi
    return df


def window(df):
    return df[(df["tr"] >= SIGNAL_TRS[0]) & (df["tr"] <= SIGNAL_TRS[1])]


def subj_mean(df, col, conds=None, **filters):
    x = df if conds is None else df[df["cond"].isin(conds)]
    for k, v in filters.items():
        x = x[x[k] == v]
    return x.groupby("subject")[col].mean()


# =====================================================================
# ANALYSES
# =====================================================================

def a1_timecourse(data):
    rows = []
    for roi, df in data.items():
        for session, conds in [("presleep", ["pre"]), ("postsleep", ["post1", "post2"])]:
            x = df[df["cond"].isin(conds)]
            for kind in ["object", "scene"]:
                xx = x[x["object_scene_retrieval"] == kind]
                s = xx.groupby(["subject", "tr"]).apply(lambda g: (g["target"] - g["others"]).mean()).unstack()
                for tr in s.columns:
                    v = s[tr].dropna()
                    t, p = ttest_1samp(v, 0)
                    rows.append(dict(roi=roi, session=session, kind=kind, tr=int(tr),
                                     mean=v.mean(), sem=v.std(ddof=1) / np.sqrt(len(v)),
                                     t=t, p=p, n=len(v)))
    return pd.DataFrame(rows)


def a2_target_vs_control(data):
    rows = []
    for roi, df in data.items():
        w = window(df)
        for cond, cs in CONDS.items():
            a, b = subj_mean(w, "target", cs), subj_mean(w, "control_item", cs)
            d, t, p, n = _paired(a, b)
            rows.append(dict(roi=roi, cond=cond, target=a.mean(), control_item=b.mean(),
                             diff=d, t=t, p=p, n=n))
    out = pd.DataFrame(rows)
    out["q_fdr_across_rois"] = np.nan
    for cond in CONDS:
        m = out["cond"] == cond
        out.loc[m, "q_fdr_across_rois"] = _fdr(out.loc[m, "p"])
    return out


def a3_sleep(data):
    rows = []
    for roi, df in data.items():
        w = window(df)
        for col in ["target", "control_item"]:
            pre, p1, p2 = (subj_mean(w, col, [c]) for c in ["pre", "post1", "post2"])
            for lab, (a, b) in {"post1_vs_pre": (p1, pre), "post2_vs_post1": (p2, p1),
                                "post_vs_pre": (subj_mean(w, col, ["post1", "post2"]), pre)}.items():
                d, t, p, n = _paired(a, b)
                rows.append(dict(roi=roi, measure=col, contrast=lab, mean_a=a.mean(), mean_b=b.mean(),
                                 diff=d, t=t, p=p, n=n))
    out = pd.DataFrame(rows)
    out["q_fdr_across_rois"] = np.nan
    for (m, c), g in out.groupby(["measure", "contrast"]):
        out.loc[g.index, "q_fdr_across_rois"] = _fdr(g["p"])
    return out


def a4_position(data):
    rows = []
    lin, quad = np.array([-2, -1, 0, 1, 2]), np.array([2, -1, -2, -1, 2])
    for roi, df in data.items():
        w = window(df).assign(d=lambda z: z["target"] - z["control_item"])
        for kind in ["all", "object", "scene"]:
            x = w if kind == "all" else w[w["object_scene_retrieval"] == kind]
            s = x.groupby(["subject", "retrieved_position"])["d"].mean().unstack().dropna()
            aov = AnovaRM(s.stack().rename("d").reset_index(), "d", "subject",
                          within=["retrieved_position"]).fit().anova_table
            tl, pl = ttest_1samp(s.values @ lin, 0)
            tq, pq = ttest_1samp(s.values @ quad, 0)
            r = dict(roi=roi, kind=kind, n=len(s), F=aov["F Value"].iloc[0], p_F=aov["Pr > F"].iloc[0],
                     t_linear=tl, p_linear=pl, t_quadratic=tq, p_quadratic=pq)
            for pos in range(1, 6):
                r[f"diff_p{pos}"] = s[pos].mean()
                r[f"sem_p{pos}"] = s[pos].std(ddof=1) / np.sqrt(len(s))
            rows.append(r)
    return pd.DataFrame(rows)


def a5_trial_type(data):
    rows = []
    for roi, df in data.items():
        w = window(df).assign(d=lambda z: z["target"] - z["control_item"])
        for cond, cs in CONDS.items():
            o = subj_mean(w, "d", cs, object_scene_retrieval="object")
            s = subj_mean(w, "d", cs, object_scene_retrieval="scene")
            d, t, p, n = _paired(o, s)
            rows.append(dict(roi=roi, test="object_vs_scene", cond=cond, object=o.mean(),
                             scene=s.mean(), diff=d, t=t, p=p, n=n))
        # does the sleep change (post1 - pre) differ between objects and scenes?
        ch = {}
        for kind in ["object", "scene"]:
            pre = subj_mean(w, "d", ["pre"], object_scene_retrieval=kind)
            p1 = subj_mean(w, "d", ["post1"], object_scene_retrieval=kind)
            idx = pre.index.intersection(p1.index)
            ch[kind] = (p1[idx] - pre[idx])
        d, t, p, n = _paired(ch["object"], ch["scene"])
        rows.append(dict(roi=roi, test="sleep_change_object_vs_scene", cond="post1-pre",
                         object=ch["object"].mean(), scene=ch["scene"].mean(), diff=d, t=t, p=p, n=n))
    return pd.DataFrame(rows)


def a6_preceding(data):
    rows, prof = [], []
    for roi, df in data.items():
        x = df[df["retrieved_position"] >= 3]
        w = window(x)
        for cond, cs in CONDS.items():
            for lab, (a, b, sub) in {
                "preceding_vs_control": ("prec_ret", "prec_ctl", w),
                "target_vs_control": ("target", "control_item", w),
                "preceding_vs_following": ("prec_ret", "foll_ret", w[w["retrieved_position"] < 5]),
            }.items():
                d, t, p, n = _paired(subj_mean(sub, a, cs), subj_mean(sub, b, cs))
                rows.append(dict(roi=roi, test=lab, cond=cond, position="3-5", diff=d, t=t, p=p, n=n))
            for pos in [3, 4, 5]:
                d, t, p, n = _paired(subj_mean(w, "prec_ret", cs, retrieved_position=pos),
                                     subj_mean(w, "prec_ctl", cs, retrieved_position=pos))
                rows.append(dict(roi=roi, test="preceding_vs_control", cond=cond, position=str(pos),
                                 diff=d, t=t, p=p, n=n))
        for tr in range(int(x["tr"].min()), int(x["tr"].max()) + 1):
            xt = x[x["tr"] == tr]
            for lab, (a, b) in {"preceding": ("prec_ret", "prec_ctl"), "target": ("target", "control_item")}.items():
                d, t, p, n = _paired(subj_mean(xt, a), subj_mean(xt, b))
                prof.append(dict(roi=roi, tr=tr, series=lab, diff=d, t=t, p=p, n=n))
    return pd.DataFrame(rows), pd.DataFrame(prof)


# =====================================================================
# FIGURES
# =====================================================================

def fig_timecourse(tc):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
    for ax, roi in zip(axes.flat, FIG_ROIS):
        for session, color in [("presleep", COND_COLORS["pre"]), ("postsleep", COND_COLORS["post1"])]:
            x = tc[(tc["roi"] == roi) & (tc["session"] == session) & (tc["kind"] == "object")]
            ax.plot(x["tr"], x["mean"], color=color, lw=2, label=session)
            ax.fill_between(x["tr"], x["mean"] - x["sem"], x["mean"] + x["sem"], color=color, alpha=.25, lw=0)
            for _, r in x[x["p"] < .05].iterrows():
                ax.text(r["tr"], r["mean"] + r["sem"] + 0.3, "*", ha="center", color=color, fontsize=9)
        ax.axhline(0, color="k", lw=.6, ls=":")
        ax.axvspan(*SIGNAL_TRS, color="0.9", zorder=0)
        ax.set_title(ROI_LABELS[roi])
    for ax in axes[-1]:
        ax.set_xlabel("TR after cue")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(target) − P(others)  [%]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Reinstatement of the retrieved object (grey = analysis window)", y=1.01)
    _save(fig, "1_target_timecourse_objects")


def _bars(ax, data_by_cond, conds, ylabel=None, title=None):
    for i, c in enumerate(conds):
        v = data_by_cond[c].dropna()
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color=COND_COLORS.get(c, "0.6"),
               edgecolor="0.3", capsize=3, width=.7)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, .07, len(v)), v,
                   s=6, color="0.4", alpha=.6, zorder=3)
        t, p = ttest_1samp(v, 0)
        ax.text(i, 0.99, _sig(p), ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds], fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def fig_target_vs_control(data, a2, a3):
    conds = ["pre", "post1", "post2"]
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.1 * len(FIG_ROIS), 3.2), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        w = window(data[roi]).assign(d=lambda z: z["target"] - z["control_item"])
        vals = {c: subj_mean(w, "d", [c]) for c in conds}
        _bars(ax, vals, conds, ylabel="retrieved − control item [pp]" if roi == FIG_ROIS[0] else None,
              title=ROI_LABELS[roi])
        s = a3[(a3["roi"] == roi) & (a3["measure"] == "target")].set_index("contrast")
        ax.set_xlabel(f"pass1 vs pre p={s.loc['post1_vs_pre','p']:.2f}\npass2 vs pass1 p={s.loc['post2_vs_post1','p']:.2f}",
                      fontsize=6.5)
    fig.suptitle("Retrieved item vs the same position of the non-retrieved sequence (TR 3-8)", y=1.03)
    _save(fig, "2_target_vs_control_by_session")

    # all 12 ROIs, pooled over sessions, FDR marked
    x = a2[a2["cond"] == "pooled"].set_index("roi").loc[ALL_ROIS]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(x)), x["diff"], color=["#2CA02C" if q < .05 else "0.7" for q in x["q_fdr_across_rois"]],
           edgecolor="0.3")
    for i, (d, q) in enumerate(zip(x["diff"], x["q_fdr_across_rois"])):
        ax.text(i, d + .05, _sig(q), ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([ROI_LABELS[r] for r in x.index], rotation=45, ha="right")
    ax.set_ylabel("retrieved − control item [pp]")
    ax.set_title("Item reinstatement per ROI, sessions pooled (green: FDR q < .05 across 12 ROIs)")
    _save(fig, "2b_target_vs_control_all_rois")


def fig_raw_sleep(data):
    conds = ["pre", "post1", "post2"]
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.1 * len(FIG_ROIS), 3), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        w = window(data[roi])
        for col, color, lab in [("target", "#D4534E", "retrieved item"), ("control_item", "0.5", "control item")]:
            m = [subj_mean(w, col, [c]) for c in conds]
            ax.errorbar(range(3), [v.mean() for v in m],
                        yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in m], color=color, marker="o", lw=1.5,
                        capsize=3, label=lab)
        ax.axhline(10, color="k", lw=.6, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=7)
        ax.set_title(ROI_LABELS[roi])
    axes[0].set_ylabel("classifier probability [%] (chance = 10)")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "3_raw_probability_by_session")


def fig_position(a4):
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.1 * len(FIG_ROIS), 3), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        for kind, color in [("object", "#8C564B"), ("scene", "#9467BD")]:
            r = a4[(a4["roi"] == roi) & (a4["kind"] == kind)].iloc[0]
            m = [r[f"diff_p{p}"] for p in range(1, 6)]
            e = [r[f"sem_p{p}"] for p in range(1, 6)]
            ax.errorbar(range(1, 6), m, yerr=e, color=color, marker="o", capsize=2, lw=1.2, label=kind)
        r = a4[(a4["roi"] == roi) & (a4["kind"] == "all")].iloc[0]
        ax.set_title(f"{ROI_LABELS[roi]}\nF={r['F']:.2f} p={r['p_F']:.2f}", fontsize=8)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("retrieved position")
        ax.set_xticks(range(1, 6))
    axes[0].set_ylabel("retrieved − control item [pp]")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "4_by_position")


def fig_trial_type(a5):
    x = a5[(a5["test"] == "object_vs_scene") & (a5["cond"] == "pooled")].set_index("roi").loc[FIG_ROIS]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    i = np.arange(len(x))
    ax.bar(i - .18, x["object"], .36, color="#8C564B", label="objects")
    ax.bar(i + .18, x["scene"], .36, color="#9467BD", label="scenes")
    for k, p in enumerate(x["p"]):
        ax.text(k, max(x["object"].iloc[k], x["scene"].iloc[k]) + .1, _sig(p), ha="center")
    ax.set_xticks(i)
    ax.set_xticklabels([ROI_LABELS[r] for r in x.index], rotation=30, ha="right")
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("retrieved − control item [pp]")
    ax.legend(frameon=False)
    ax.set_title("Objects vs scenes, sessions pooled (* = difference)")
    _save(fig, "5_objects_vs_scenes")


def fig_preceding(a6, prof):
    tests = [("target_vs_control", "target\nvs ctrl", "#D4534E"),
             ("preceding_vs_control", "preced.\nvs ctrl", "#4C72B0"),
             ("preceding_vs_following", "preced.\nvs foll.", "0.6")]
    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.1 * len(FIG_ROIS), 3), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        for i, (tst, lab, color) in enumerate(tests):
            r = a6[(a6["roi"] == roi) & (a6["test"] == tst) & (a6["cond"] == "pooled") & (a6["position"] == "3-5")].iloc[0]
            ax.bar(i, r["diff"], color=color, edgecolor="0.3")
            ax.text(i, r["diff"] + .05 if r["diff"] >= 0 else r["diff"] - .3, _sig(r["p"]), ha="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels([t[1] for t in tests], fontsize=6)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(ROI_LABELS[roi])
    axes[0].set_ylabel("difference [pp]  (p ≥ 3 trials, TR 3-8)")
    fig.suptitle("Only the retrieved item is reactivated — not the items before it", y=1.03)
    _save(fig, "6_preceding_items")

    fig, axes = plt.subplots(1, len(FIG_ROIS), figsize=(2.1 * len(FIG_ROIS), 2.8), sharey=True)
    for ax, roi in zip(axes, FIG_ROIS):
        for ser, color in [("target", "#D4534E"), ("preceding", "#4C72B0")]:
            x = prof[(prof["roi"] == roi) & (prof["series"] == ser)]
            ax.plot(x["tr"], x["diff"], color=color, marker="o", ms=3, label=ser)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(ROI_LABELS[roi])
        ax.set_xlabel("TR after cue")
    axes[0].set_ylabel("retrieved − control [pp]")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "6b_preceding_vs_target_by_TR")


# =====================================================================
# MAIN
# =====================================================================

def main():
    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading item probabilities...")
    data = {roi: load_item_probabilities(roi) for roi in ALL_ROIS}
    n_pre = data["vis"].query("cond == 'pre'")["subject"].nunique()
    n_post = data["vis"].query("cond != 'pre'")["subject"].nunique()
    print(f"  {len(data)} ROIs, {n_pre} pre-sleep / {n_post} post-sleep subjects")

    tables = {}
    print("1 timecourse");            tables["1_target_timecourse"] = a1_timecourse({r: data[r] for r in FIG_ROIS})
    print("2 target vs control");     tables["2_target_vs_control"] = a2_target_vs_control(data)
    print("3 sleep");                 tables["3_sleep"] = a3_sleep(data)
    print("4 position");              tables["4_position"] = a4_position({r: data[r] for r in FIG_ROIS})
    print("5 trial type");            tables["5_trial_type"] = a5_trial_type({r: data[r] for r in FIG_ROIS})
    print("6 preceding items")
    tables["6_preceding_items"], tables["6b_preceding_tr_profile"] = a6_preceding({r: data[r] for r in FIG_ROIS})

    for name, t in tables.items():
        t.to_csv(opj(OUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}.csv")

    print("Figures")
    fig_timecourse(tables["1_target_timecourse"])
    fig_target_vs_control(data, tables["2_target_vs_control"], tables["3_sleep"])
    fig_raw_sleep(data)
    fig_position(tables["4_position"])
    fig_trial_type(tables["5_trial_type"])
    fig_preceding(tables["6_preceding_items"], tables["6b_preceding_tr_profile"])
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
