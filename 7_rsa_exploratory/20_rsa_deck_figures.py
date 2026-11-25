# -*- coding: utf-8 -*-
"""Figures for the RSA slide deck (2026-11-23), drawn from existing result tables.

No new estimation except one test (figure 7): does the position structure of the
pattern similarity follow each participant's mean response time per position?

Input   outputs/D_E_on_existing_results/D2_binary_rsa, D3_position_and_neighbours   (one model per run)
        outputs/12_hsm_position_shape/lss/subject_cells.csv                         (5 x 5 position cells)
        outputs/patterns_lsa/*/*_trials.csv                                         (response times)
Output  outputs/20_rsa_deck_figures/*.png and rt_profile_test.csv

Usage:  python 20_rsa_deck_figures.py
"""
import glob
import os
from os.path import join as opj

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = opj(HERE, "outputs")
D2 = opj(OUT, "D_E_on_existing_results", "D2_binary_rsa")
D3 = opj(OUT, "D_E_on_existing_results", "D3_position_and_neighbours")
FIG = opj(OUT, "20_rsa_deck_figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, INK, SOFT, GRID, BG = "#1F63B8", "#B8481C", "#17212B", "#46525E", "#DAD5C9", "#FDFDFB"
PALE = "#B9CDE6"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F3F1EA", "#1B4F96"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#EDEBE6", ORANGE])
ROIS = ["vis", "parietal_precuneus", "superior_frontal", "rsc", "mtl", "caudal_acc"]
LAB = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "superior_frontal": "Superior frontal",
       "rsc": "Retrosplenial", "mtl": "MTL", "caudal_acc": "Caudal ACC (control)", "parietal": "Parietal",
       "precuneus": "Precuneus", "cortical_memory": "Cortical memory"}
SETS = ["pre", "post1", "post2"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def save(fig, name):
    fig.savefig(opj(FIG, name), dpi=200)
    plt.close(fig)
    print("saved", name)


def tval(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.ttest_1samp(v, 0)[0]


# ------------------------------------------------------------------ model matrices (50 object trials)
ctx = np.repeat([0, 1], 25)
scene = np.repeat(np.arange(10), 5)
scene_pos = np.tile(np.repeat(np.arange(5), 5), 2)
pos = np.tile(np.arange(5), 10)
catseq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])[scene]          # schematic: both sequences in both contexts
MEAN_RT = None                                                   # filled below


def frame(ax, title):
    ax.set_title(title, pad=8)
    ax.set_xticks([12, 37]); ax.set_xticklabels(["Day", "Night"])
    ax.set_yticks([12, 37]); ax.set_yticklabels(["Day", "Night"], rotation=90, va="center")
    ax.tick_params(length=0)
    for v in (24.5,):
        ax.axhline(v, color=BG, lw=2); ax.axvline(v, color=BG, lw=2)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_models_binary():
    mats = {"Context": ctx[:, None] != ctx[None, :], "Scene": scene[:, None] != scene[None, :],
            "Position": pos[:, None] != pos[None, :],
            "Category": (pos[:, None] != pos[None, :]) | (catseq[:, None] != catseq[None, :])}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, (name, m) in zip(axes, mats.items()):
        ax.imshow(m.astype(float), cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        frame(ax, name)
    fig.text(0.012, 0.03, "50 object trials, ordered by context, scene and position.  Light = same (predicted similar), "
             "dark = different.", fontsize=12, color=SOFT)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.9, bottom=0.14, wspace=0.12)
    save(fig, "1_models_binary.png")


def fig_models_graded(rt):
    mats = {"Object position distance": np.abs(pos[:, None] - pos[None, :]).astype(float),
            "Scene order distance": np.where(ctx[:, None] == ctx[None, :],
                                             np.abs(scene_pos[:, None] - scene_pos[None, :]), np.nan).astype(float),
            "Response-time profile (control)": np.abs(rt[pos][:, None] - rt[pos][None, :])}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    cm = SEQ.copy(); cm.set_bad("#E4E0D6")
    for ax, (name, m) in zip(axes, mats.items()):
        im = ax.imshow(np.ma.masked_invalid(m), cmap=cm, interpolation="nearest")
        frame(ax, name)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.set_label("seconds" if "Response" in name else "steps", color=SOFT)
    fig.text(0.012, 0.03, "Light = close (predicted similar), dark = far.  Grey = pairs from different contexts "
             "(scene order is tested within a context).", fontsize=12, color=SOFT)
    fig.subplots_adjust(left=0.035, right=0.955, top=0.9, bottom=0.12, wspace=0.22)
    save(fig, "2_models_graded.png")


# ------------------------------------------------------------------ empirical matrices
def cells_table(variant="base"):
    c = pd.read_csv(opj(OUT, "12_hsm_position_shape", "lss", "subject_cells.csv"))
    return c[c.variant == variant].groupby(["subject", "roi", "i", "j"], as_index=False).value.mean()


def fig_empirical(cells):
    z = np.load(opj(D2, "group_mean_rdms.npz"))
    m = np.nanmean(np.stack([z[f"parietal_precuneus__{s}"][:50, :50] for s in SETS]), axis=0)
    g = cells[cells.roi == "parietal_precuneus"].groupby(["i", "j"]).value.mean()
    c5 = np.full((5, 5), np.nan)
    for (i, j), v in g.items():
        c5[i - 1, j - 1] = c5[j - 1, i - 1] = v
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    lim = np.nanpercentile(np.abs(m), 98)
    im = ax.imshow(np.ma.masked_invalid(m), cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    frame(ax, "Group mean, 50 object trials")
    ax.axhline(24.5, color=INK, lw=1); ax.axvline(24.5, color=INK, lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    cb.set_label("Dissimilarity (z)", color=SOFT)
    ax = axes[1]
    lim = 0.11
    ax.imshow(c5, cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{c5[i, j]:+.2f}", ha="center", va="center", fontsize=13,
                    color="white" if abs(c5[i, j]) > 0.07 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(range(1, 6)); ax.set_yticklabels(range(1, 6)); ax.tick_params(length=0)
    ax.set_xlabel("Object position"); ax.set_ylabel("Object position")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Averaged by position pair", pad=8)
    fig.text(0.012, 0.03, "Blue = more similar than the average pair, orange = less similar.  Right: pairs from "
             "different scenes and categories only.", fontsize=12, color=SOFT)
    fig.subplots_adjust(left=0.0, right=0.98, top=0.92, bottom=0.14, wspace=0.12)
    save(fig, "3_empirical.png")


# ------------------------------------------------------------------ results
def hbars(ax, vals, title, labels=True, color=BLUE):
    y = np.arange(len(ROIS))[::-1]
    cols = [color if abs(v) >= 1.96 else PALE for v in vals]
    ax.barh(y, vals, height=0.62, color=cols)
    ax.axvline(0, color=SOFT, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LAB[r] for r in ROIS] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, pad=8)
    for yy, v in zip(y, vals):
        ax.text(v + (0.15 if v >= 0 else -0.15), yy, f"{v:.1f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=12, color=INK)
    ax.spines["left"].set_visible(False)


def fig_four_models():
    l = pd.read_csv(opj(D2, "group_beta_session_lme.csv"))
    l = l[(l.variant == "four") & (l.pairtype == "oo")].set_index(["model", "roi"]).z_overall
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.1), sharex=True)
    for k, (ax, m) in enumerate(zip(axes, ["context", "scene", "position", "category"])):
        hbars(ax, [l[(m, r)] for r in ROIS], m.capitalize(), labels=k == 0)
        ax.set_xlim(-2.2, 5.6); ax.set_xlabel("z, three time points pooled")
    fig.text(0.012, 0.02, "Object x object pairs, all four models in one regression.  Dark bars: |z| > 1.96.",
             fontsize=12, color=SOFT)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.9, bottom=0.24, wspace=0.12)
    save(fig, "4_four_models.png")


def fig_scene_cue():
    l = pd.read_csv(opj(D2, "group_beta_session_lme.csv"))
    l = l[(l.variant == "four") & (l.model == "scene")].set_index(["pairtype", "roi"]).z_overall
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y = np.arange(len(ROIS))[::-1]
    a, b = [l[("oo", r)] for r in ROIS], [l[("os", r)] for r in ROIS]
    ax.barh(y + 0.2, a, height=0.36, color=BLUE, label="Two object trials (scene picture on screen in both)")
    ax.barh(y - 0.2, b, height=0.36, color=ORANGE, label="Object trial and scene trial (picture in one only)")
    for yy, v, w in zip(y, a, b):
        ax.text(v + 0.1, yy + 0.2, f"{v:.1f}", va="center", fontsize=12)
        ax.text(w + (0.1 if w >= 0 else -0.1), yy - 0.2, f"{w:.1f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=12)
    ax.axvline(0, color=SOFT, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LAB[r] for r in ROIS]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.6, 5.4); ax.set_xlabel("Same-scene effect (z, three time points pooled)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.2), fontsize=12)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.14)
    save(fig, "5_scene_cue.png")


def fig_distance_curve():
    d = pd.read_csv(opj(D3, "A_subject_distance_curve.csv"))
    d = d[d.pair_set.isin(SETS) & d["where"].isin(["within_scene", "across_scenes"])]
    d = d.groupby(["subject", "roi", "where", "distance"], as_index=False).z_dissim.mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), sharey=True)
    for ax, roi in zip(axes, ["parietal_precuneus", "vis"]):
        for where, col, lab in (("across_scenes", BLUE, "different scenes"), ("within_scene", ORANGE, "same scene")):
            g = d[(d.roi == roi) & (d["where"] == where)].groupby("distance").z_dissim.agg(["mean", "sem"])
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=col, lw=2, marker="o", ms=8, mec=BG, mew=2,
                        capsize=0, label=lab)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(LAB[roi], pad=8); ax.set_xlabel("Distance between positions")
        ax.set_xticks(range(5)); ax.set_xlim(-0.3, 4.3)
    axes[0].set_ylabel("Dissimilarity (z)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=12, title="Objects from", title_fontsize=12)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.14, wspace=0.08)
    save(fig, "6_distance_curve.png")


def rt_profile_test(cells_by_variant):
    T = pd.concat([pd.read_csv(f) for f in glob.glob(opj(OUT, "patterns_lsa", "*", "*_trials.csv"))])
    T = T[T.kind == "object"]
    rtm = T.groupby(["subject", "position"]).rt.mean().unstack()
    off = [(i, j) for i in range(1, 6) for j in range(i + 1, 6)]
    lin = np.array([j - i for i, j in off], float)
    z = lambda v: (v - v.mean()) / v.std()
    rows = []
    for variant, cc in cells_by_variant.items():
        for roi in ROIS + ["parietal", "precuneus", "cortical_memory"]:
            acc = []
            for sub, g in cc[cc.roi == roi].groupby("subject"):
                if sub not in rtm.index or rtm.loc[sub].isna().any():
                    continue
                w = g.set_index(["i", "j"]).value
                if not all(k in w.index for k in off):
                    continue
                v = np.array([w[k] for k in off])
                d = np.array([abs(rtm.loc[sub, i] - rtm.loc[sub, j]) for i, j in off])
                b = np.linalg.lstsq(np.column_stack([np.ones(10), z(lin), z(d)]), z(v), rcond=None)[0]
                acc.append((np.arctanh(np.corrcoef(v, lin)[0, 1]), np.arctanh(np.corrcoef(v, d)[0, 1]), b[1], b[2]))
            a = np.array(acc)
            rows.append(dict(variant=variant, roi=roi, n=len(a), t_linear_alone=tval(a[:, 0]),
                             t_rt_profile_alone=tval(a[:, 1]), t_linear_joint=tval(a[:, 2]),
                             t_rt_profile_joint=tval(a[:, 3])))
    res = pd.DataFrame(rows)
    res.to_csv(opj(FIG, "rt_profile_test.csv"), index=False)
    return res, T.groupby("position").rt.mean()


def fig_rt_profile(res, rt, cells):
    res = res[res.variant == "base"].set_index("roi")
    fig = plt.figure(figsize=(14, 4.6))
    axes = [fig.add_axes(r) for r in ([0.055, 0.17, 0.17, 0.7], [0.3, 0.17, 0.24, 0.7], [0.705, 0.3, 0.28, 0.57])]
    ax = axes[0]
    ax.bar(rt.index, rt.values, width=0.62, color=BLUE)
    for x, v in zip(rt.index, rt.values):
        ax.text(x, v + 0.04, f"{v:.1f}", ha="center", fontsize=12)
    ax.set_xlabel("Object position"); ax.set_ylabel("Mean response time (s)"); ax.set_ylim(0, 2.6)
    ax.set_title("Response time", pad=8)
    ax = axes[1]
    g = cells[cells.roi == "parietal_precuneus"].groupby(["i", "j"]).value.mean()
    for (i, j), v in g.items():
        if i == j:
            continue
        x = abs(rt[i] - rt[j])
        ax.scatter(x, v, s=70, color=BLUE, edgecolor=BG, lw=1.5, zorder=3)
        ax.annotate(f"{i}-{j}", (x, v), xytext=(7, 5), textcoords="offset points", fontsize=12, color=SOFT)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("Difference in mean response time (s)"); ax.set_ylabel("Dissimilarity (z)")
    ax.set_title("Position pairs, parietal + precuneus", pad=8)
    ax = axes[2]
    y = np.arange(len(ROIS))[::-1]
    for yy, r in zip(y, ROIS):
        a, b, c = res.loc[r, "t_linear_alone"], res.loc[r, "t_linear_joint"], res.loc[r, "t_rt_profile_joint"]
        ax.plot([b, a], [yy + 0.15, yy + 0.15], color=PALE, lw=3, zorder=1)
        ax.scatter(a, yy + 0.15, s=90, facecolor=BG, edgecolor=BLUE, lw=2, zorder=3)
        ax.scatter(b, yy + 0.15, s=90, color=BLUE, edgecolor=BG, lw=1.5, zorder=3)
        ax.scatter(c, yy - 0.2, s=90, color=ORANGE, marker="D", edgecolor=BG, lw=1.5, zorder=3)
    ax.axvline(0, color=SOFT, lw=1); ax.axvline(2.03, color=GRID, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels([LAB[r] for r in ROIS]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("t across participants"); ax.set_xlim(-1.5, 6.6)
    ax.set_title("Tests across participants", pad=8)
    h = [plt.Line2D([], [], marker="o", ls="", mfc=BG, mec=BLUE, mew=2, ms=9, label="linear distance alone"),
         plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="linear, both in the model"),
         plt.Line2D([], [], marker="D", ls="", color=ORANGE, ms=8, label="response-time profile, both in the model")]
    ax.legend(handles=h, frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.3, -0.22), ncol=1)
    save(fig, "7_rt_profile.png")


def fig_neighbours():
    c = pd.read_csv(opj(D3, "C_group_neighbour_scene.csv"))
    c = c[c.measure == "diff"].set_index(["group", "roi"]).t
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y = np.arange(len(ROIS))[::-1]
    a, b = [c[("evening", r)] for r in ROIS], [c[("morning", r)] for r in ROIS]
    ax.barh(y + 0.2, a, height=0.36, color=BLUE, label="Evening")
    ax.barh(y - 0.2, b, height=0.36, color=ORANGE, label="Morning (both passes)")
    for yy, v, w in zip(y, a, b):
        ax.text(v + (0.08 if v >= 0 else -0.08), yy + 0.2, f"{v:.1f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=12)
        ax.text(w + (0.08 if w >= 0 else -0.08), yy - 0.2, f"{w:.1f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=12)
    ax.axvline(0, color=SOFT, lw=1); ax.axvline(2.03, color=GRID, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels([LAB[r] for r in ROIS]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.0, 3.6); ax.set_xlabel("Neighbour advantage, same scene minus other scenes (t)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, 1.14), ncol=2, fontsize=12)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.14)
    save(fig, "8_neighbours_scene.png")


def fig_sessions():
    b = pd.read_csv(opj(D3, "A_subject_betas.csv"))
    b = b[(b.model == "distance") & b.pair_set.isin(SETS)]
    fig, axes = plt.subplots(1, 6, figsize=(14, 3.9), sharey=True)
    for k, (ax, roi) in enumerate(zip(axes, ROIS)):
        g = b[b.roi == roi].groupby("pair_set").beta.agg(["mean", "sem"]).reindex(SETS)
        ax.errorbar(range(3), g["mean"], yerr=g["sem"], color=BLUE, lw=2, marker="o", ms=8, mec=BG, mew=2)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Eve", "Morn 1", "Morn 2"], fontsize=11)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(LAB[roi].replace(" (control)", ""), pad=8, fontsize=12)
    axes[0].set_ylabel("Position distance effect (beta)")
    fig.text(0.012, 0.02, "Mean and standard error across participants.  Eve = evening, Morn 1 and 2 = first and "
             "repeated morning retrieval.", fontsize=12, color=SOFT)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.24, wspace=0.1)
    save(fig, "9_sessions.png")


def fig_across():
    """Model matrices of D3 part F and the data by distance."""
    cm = SEQ.copy(); cm.set_bad("#E4E0D6")
    mo = np.abs(pos[:, None] - pos[None, :]).astype(float)
    mo[(scene[:, None] == scene[None, :]) | (mo == 0)] = np.nan                 # same sequence or same digit: left out
    sctx, spos = np.repeat([0, 1], 10), np.tile(np.repeat(np.arange(5), 2), 2)   # 20 scene trials
    ms = np.abs(spos[:, None] - spos[None, :]).astype(float)
    ms[(sctx[:, None] == sctx[None, :]) | (ms == 0)] = np.nan
    cur = pd.read_csv(opj(D3, "F_group_across_curve.csv"))
    grp = pd.read_csv(opj(D3, "F_group_across.csv"))
    grp = grp[(grp.pair_set == "all_sessions") & (grp.model == "distance")].set_index(["pairs", "roi"]).t
    fig = plt.figure(figsize=(14, 5.0))
    ax = fig.add_axes([0.03, 0.25, 0.19, 0.62])
    ax.imshow(np.ma.masked_invalid(mo), cmap=cm, vmin=1, vmax=4, interpolation="nearest"); frame(ax, "Model: objects")
    ax = fig.add_axes([0.25, 0.25, 0.19, 0.62])
    im = ax.imshow(np.ma.masked_invalid(ms), cmap=cm, vmin=1, vmax=4, interpolation="nearest")
    ax.set_title("Model: scenes", pad=8)
    ax.set_xticks([4.5, 14.5]); ax.set_xticklabels(["Day", "Night"])
    ax.set_yticks([4.5, 14.5]); ax.set_yticklabels(["Day", "Night"], rotation=90, va="center"); ax.tick_params(length=0)
    ax.axhline(9.5, color=BG, lw=2); ax.axvline(9.5, color=BG, lw=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([0.447, 0.3, 0.007, 0.52]); cb = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4])
    cb.outline.set_visible(False); cb.set_label("steps apart", color=SOFT)
    for k, (label, title) in enumerate((("objects_other_scene", "Objects, other scenes"),
                                        ("scenes_other_context", "Scenes, other context"))):
        ax = fig.add_axes([0.575 + 0.225 * k, 0.33, 0.185, 0.54])
        for roi, col in (("parietal_precuneus", BLUE), ("vis", ORANGE)):
            g = cur[(cur.pairs == label) & (cur.roi == roi)].sort_values("distance")
            ax.errorbar(g.distance + (-0.05 if col == BLUE else 0.05), g.z_dissim, yerr=g["sem"], color=col, lw=2,
                        marker="o", ms=8, mec=BG, mew=2,
                        label=f"{LAB[roi]} (t = {grp[(label, roi)]:.1f})")
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4); ax.set_xlabel("Steps apart")
        ax.set_title(title, pad=8, fontsize=13)
        ax.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        if k == 0:
            ax.set_ylabel("Dissimilarity (z)")
    fig.text(0.03, 0.1, "Grey = left out: same sequence (shared picture or\ncontext cue) or same position (shared digit).",
             fontsize=12, color=SOFT, va="top")
    save(fig, "10_across_sequences.png")


def fig_gradient_step():
    """Gradient and step model (D3 part F): matrices, predictions against the data, and fit."""
    cm = SEQ.copy(); cm.set_bad("#E4E0D6")
    grad = np.abs(pos[:, None] - pos[None, :]).astype(float)
    grad[(scene[:, None] == scene[None, :]) | (grad == 0)] = np.nan
    step = np.where(np.isnan(grad), np.nan, (grad >= 2).astype(float))
    cur = pd.read_csv(opj(D3, "F_group_across_curve.csv"))
    sub = pd.read_csv(opj(D3, "F_subject_across_curve.csv"))
    comp = pd.read_csv(opj(D3, "F_group_model_comparison.csv"))
    comp = comp[comp.pairs == "objects_other_scene"].set_index("roi")
    fig = plt.figure(figsize=(14, 4.9))
    for k, (m, title) in enumerate(((grad, "Gradient model"), (step, "Step model"))):
        ax = fig.add_axes([0.03 + 0.2 * k, 0.24, 0.175, 0.62])
        ax.imshow(np.ma.masked_invalid(m), cmap=cm, interpolation="nearest")
        ax.set_title(title, pad=8)
        ax.set_xticks([12, 37]); ax.set_xticklabels(["Day", "Night"])
        ax.set_yticks([12, 37]); ax.set_yticklabels(["Day", "Night"], rotation=90, va="center")
        ax.tick_params(length=0)
        for v in np.arange(4.5, 49, 5):
            ax.axhline(v, color=BG, lw=0.6); ax.axvline(v, color=BG, lw=0.6)
        ax.axhline(24.5, color=BG, lw=2.5); ax.axvline(24.5, color=BG, lw=2.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.03, 0.12, "All 50 objects of the 10 scenes, within and across contexts. Light = predicted\n"
             "similar, dark = dissimilar. Grey = left out (same scene or same position).", fontsize=11.5, color=SOFT, va="top")
    # data and the two fitted predictions, parietal + precuneus
    ax = fig.add_axes([0.5, 0.24, 0.19, 0.62])
    g = cur[(cur.pairs == "objects_other_scene") & (cur.roi == "parietal_precuneus")].sort_values("distance")
    w = (sub[(sub.pairs == "objects_other_scene") & (sub.roi == "parietal_precuneus")]
         .groupby("distance").n_pairs.mean().reindex(g.distance).values)
    x, y = g.distance.values.astype(float), g.z_dissim.values
    def wfit(pred):
        A = np.column_stack([np.ones(4), pred]) * np.sqrt(w)[:, None]
        return np.column_stack([np.ones(4), pred]) @ np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
    pg, ps = wfit(x), wfit((x >= 2).astype(float))
    ax.plot(x, pg, color=ORANGE, lw=2.5, label="gradient")
    ax.plot([0.6, 1.5, 1.5, 4.4], [ps[0], ps[0], ps[1], ps[1]], color=BLUE, lw=2.5, label="step")
    ax.errorbar(x, y, yerr=g["sem"], color=INK, ls="", marker="o", ms=9, mec=BG, mew=2, zorder=5, label="data")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4); ax.set_xlabel("Steps apart"); ax.set_ylabel("Dissimilarity (z)")
    ax.set_title("Predictions and data", pad=8)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    # fit per region
    ax = fig.add_axes([0.825, 0.3, 0.16, 0.56])
    yy = np.arange(len(ROIS))[::-1]
    ax.barh(yy + 0.2, [comp.loc[r, "mean_r_step"] for r in ROIS], height=0.36, color=BLUE, label="step")
    ax.barh(yy - 0.2, [comp.loc[r, "mean_r_gradient"] for r in ROIS], height=0.36, color=ORANGE, label="gradient")
    for v, r in zip(yy, ROIS):
        t = comp.loc[r, "t_step_minus_gradient"]
        ax.text(max(comp.loc[r, "mean_r_step"], comp.loc[r, "mean_r_gradient"], 0) + 0.002, v, f"t = {t:.1f}",
                va="center", fontsize=10.5, color=INK if abs(t) >= 2.03 else SOFT)
    ax.axvline(0, color=SOFT, lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([LAB[r].replace(" (control)", "") for r in ROIS], fontsize=11)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.005, 0.05); ax.set_xlabel("Mean correlation")
    ax.set_title("Model fit", pad=8)
    ax.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.3, -0.22), ncol=2)
    fig.text(0.725, 0.03, "t: step minus gradient, paired test", fontsize=11, color=SOFT)
    save(fig, "11_gradient_step.png")


def fig_hierarchy():
    """Hierarchy-distance model (D3 part G): matrix for 50 object and 20 scene trials, and the data per level."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#F3F1EA", "#8FA9CF", "#1B4F96"]); cmap.set_bad("#E4E0D6")
    s_ctx, s_scene = np.repeat([0, 1], 10), np.repeat(np.arange(10), 2)       # 20 scene trials, two per scene
    s_digit = np.tile(np.repeat(np.arange(5), 2), 2)                           # digit = position of the scene
    C = np.concatenate([ctx, s_ctx]); S = np.concatenate([scene, s_scene]); Dg = np.concatenate([pos, s_digit])
    is_obj = np.arange(70) < 50
    M = np.where(C[:, None] != C[None, :], 3.0, np.where(S[:, None] != S[None, :], 2.0, 1.0))
    both_obj, both_sc = is_obj[:, None] & is_obj[None, :], ~is_obj[:, None] & ~is_obj[None, :]
    M[(Dg[:, None] == Dg[None, :]) | both_sc | (both_obj & (S[:, None] == S[None, :]))] = np.nan
    lv = pd.read_csv(opj(D3, "G_group_hierarchy_levels.csv"))
    gh = pd.read_csv(opj(D3, "G_group_hierarchy.csv"))
    gh = gh[gh.pair_set == "all_sessions"].set_index(["model", "roi"]).t
    fig = plt.figure(figsize=(14, 5.0))
    ax = fig.add_axes([0.03, 0.2, 0.26, 0.68])
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=3.5, interpolation="nearest")
    ax.set_title("Hierarchy model", pad=26)
    ax.set_xticks([12, 37, 54.5, 64.5]); ax.set_xticklabels(["Day", "Night", "Day", "Night"], fontsize=11)
    ax.set_yticks([12, 37, 54.5, 64.5]); ax.set_yticklabels(["Day", "Night", "Day", "Night"], rotation=90, va="center", fontsize=11)
    ax.tick_params(length=0)
    for v, w in ((24.5, 1.5), (49.5, 3), (59.5, 1.5)):
        ax.axhline(v, color=BG, lw=w); ax.axvline(v, color=BG, lw=w)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(24.5, -3.5, "object trials", ha="center", fontsize=11, color=SOFT)
    ax.text(59.5, -3.5, "scene trials", ha="center", fontsize=11, color=SOFT)
    cax = fig.add_axes([0.3, 0.3, 0.008, 0.45]); cb = fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    cb.ax.set_yticklabels(["1  own scene", "2  same context", "3  other context"], fontsize=11); cb.outline.set_visible(False)
    fig.text(0.03, 0.1, "Grey = left out: same scene picture, same digit,\nand all scene x scene pairs.",
             fontsize=11.5, color=SOFT, va="top")
    panels = (("object_x_scene", "Object trial x scene trial", [1, 2, 3], "os_scene_step", "os_context_step"),
              ("object_x_object", "Two object trials", [2, 3], None, "oo_context_step"))
    for k, (label, title, levels, m1, m2) in enumerate(panels):
        ax = fig.add_axes([0.5 + 0.26 * k, 0.42, 0.215, 0.46])
        for roi, col, dx in (("parietal_precuneus", BLUE, -0.04), ("vis", ORANGE, 0.04)):
            g = lv[(lv.pairs == label) & (lv.roi == roi)].set_index("level").reindex(levels)
            lab = LAB[roi] + "\n" + (f"1 vs 2: t = {gh[(m1, roi)]:.1f};  " if m1 else "") + f"2 vs 3: t = {gh[(m2, roi)]:.1f}"
            ax.errorbar(np.array(levels) + dx, g.z_dissim, yerr=g["sem"], color=col, lw=2, marker="o", ms=8,
                        mec=BG, mew=2, label=lab)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["own\nscene", "same\ncontext", "other\ncontext"], fontsize=11)
        ax.set_xlim(0.6, 3.4); ax.set_ylim(-0.012, 0.028); ax.set_title(title, pad=8, fontsize=13)
        ax.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.3))
        if k == 0:
            ax.set_ylabel("Dissimilarity (z)")
    save(fig, "12_hierarchy.png")


if __name__ == "__main__":
    cells = {v: cells_table(v) for v in ("base", "rt")}
    res, rt = rt_profile_test(cells)
    print(res.round(2).to_string())
    fig_models_binary()
    fig_models_graded(rt.values)
    fig_empirical(cells["base"])
    fig_four_models()
    fig_scene_cue()
    fig_distance_curve()
    fig_rt_profile(res, rt, cells["base"])
    fig_neighbours()
    fig_sessions()
    if os.path.exists(opj(D3, "F_group_across_curve.csv")):
        fig_across()
    if os.path.exists(opj(D3, "F_group_model_comparison.csv")):
        fig_gradient_step()
    if os.path.exists(opj(D3, "G_group_hierarchy_levels.csv")):
        fig_hierarchy()
