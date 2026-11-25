#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C14 - Do the changes of the decoding measures between time points hold on trials
      matched for search time?

Search time (ret_rt of the B7 tables, on trials with a button press) falls from
the evening to the first and again to the repeated morning retrieval. The
decoding measures are taken from a fixed window after the cue (4.5-12 s) and
from the ten volumes of a trial, so the length of the search does not enter
them by construction; but a shorter search moves the button press, the waiting
period and the spoken answer forward within that window. This step repeats the
comparison of the three time points on trials with the same search times, as
E10 / E13 do for mean activity.

Measures per object trial and region (retrieved minus control unless stated):
    absolute slope and sinusoid amplitude over all five items (C11), also for the
        retrieved and the control sequence alone
    the same two without the retrieved item (C11)
    evidence for the retrieved item, its predecessor and its successor (C9 window)

Trial sets
    all          all object trials (the analysis of C9 / C11)
    press        trials with a button press (only these have a search time)
    matched      press trials matched over the three time points: search times are
                 sorted into bins of BIN s and from each bin the same number of
                 trials is taken at every time point (as E13); repeated N_DRAWS
                 times with random choices, participant means averaged over draws;
                 participants with at least MIN_MATCHED matched trials per time point
    matched_pair as `matched`, but matched only over the two time points of each
                 contrast (more trials; values per time point are not reported)
    adjusted     press trials; per participant the measure is regressed on time
                 point and search time, and the time-point coefficients are tested

Input   outputs/C11_slope_amplitude_by_position/participants/<sub>_amplitudes.csv, abs_slopes_v2.csv
        outputs/B7_apply_classifiers_retrieval/<session>/decoding_results_tables_for_SODA/*.csv (via C9)
Output  outputs/C14_matched_search_time/trial_table.csv        one row per trial and region
        outputs/C14_matched_search_time/matching.csv           trials and search time per participant
        outputs/C14_matched_search_time/group_contrasts.csv    means per time point, tests against zero, contrasts
Plots   plots/C14_matched_search_time/matched_search_time.{png,pdf}
Usage:  python C14_matched_search_time.py

Not a port of the original pipeline: added 2026-11-25. Exploratory.
"""

from __future__ import annotations

import glob
import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import C9_item_time_courses as c9
from config import step_input_dir, step_output_dir, step_plot_dir
from figure_style import (BLUE, COND_LABEL, CONDS, GREY, INK, MUTED, ORANGE, REGION_COLOUR, REGION_LABEL, REGIONS, WIDTH,
                          apply_style, ci95, panel_letter, save, stars, title)

BIN, N_DRAWS, MIN_MATCHED = 0.25, 200, 8
CONTRASTS = {"post1 - pre": ("post1", "pre"), "post2 - post1": ("post2", "post1"), "post2 - pre": ("post2", "pre")}
SETS = {"all": "All trials", "press": "Trials with a press", "matched": "Matched for search time"}
KEYS = ["subject", "roi", "session", "trial"]


def trial_table(out):
    """One row per object trial and region with every measure, the button press and the search time."""
    c11 = str(step_input_dir("C11_slope_amplitude_by_position"))
    ab = pd.read_csv(opj(c11, "abs_slopes_v2.csv"))
    am = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(c11, "participants", "*_amplitudes.csv")))], ignore_index=True)
    t = ab[ab.variant == "all"][KEYS + ["cond", "position", "press", "rt"]].copy()
    for name, d in (("abs slope", ab), ("amplitude", am)):
        five = d[d.variant == "all"].set_index(KEYS)
        wo = d[(d.variant == "without") & (d.q == d.position)].set_index(KEYS)
        add = pd.DataFrame({f"{name}, five items": five.retrieved - five.control,
                            f"{name}, five items, retrieved sequence": five.retrieved,
                            f"{name}, five items, control sequence": five.control,
                            f"{name}, without retrieved item": (wo.retrieved - wo.control).reindex(five.index)})
        t = t.merge(add.reset_index(), on=KEYS, how="left")
    items, _ = c9.load()
    items = items[items.item.isin(["retrieved", "predecessor", "successor"]) & items.tr.between(*c9.WINDOW)]
    items = items.assign(difference=items.item_prob - items.control_prob,
                         session=np.where(items.cond == "pre", "presleep", "postsleep"))
    ev = items.groupby(["subject", "roi", "session", "trial", "item"]).difference.mean().unstack("item")
    ev.columns = [f"evidence, {c}" + (" item" if c == "retrieved" else "") for c in ev.columns]
    t = t.merge(ev.reset_index(), on=KEYS, how="left")
    t.to_csv(opj(out, "trial_table.csv"), index=False)
    return t


def matched_means(g, cols, conds, rng):
    """Mean of `cols` per time point over trials matched for search time across `conds`, averaged over the draws."""
    g = g[g.press & g.rt.notna() & g.cond.isin(conds)]
    bins, cond, rt, v = np.floor(g.rt.values / BIN).astype(int), g.cond.values, g.rt.values, g[cols].values
    idx = {(b, c): np.flatnonzero((bins == b) & (cond == c)) for b in np.unique(bins) for c in conds}
    k = {b: min(len(idx[(b, c)]) for c in conds) for b in np.unique(bins)}
    n = sum(k.values())
    if n < MIN_MATCHED:
        return None, None
    acc, rts = np.zeros((len(conds), len(cols))), np.zeros(len(conds))
    for _ in range(N_DRAWS):
        for ci, c in enumerate(conds):
            sel = np.concatenate([rng.choice(idx[(b, c)], kb, replace=False) for b, kb in k.items() if kb])
            with np.errstate(all="ignore"):
                acc[ci] += np.nanmean(v[sel], axis=0)
            rts[ci] += rt[sel].mean()
    info = dict(n_matched=n, **{f"rt_matched_{c}": rts[i] / N_DRAWS for i, c in enumerate(conds)},
                **{f"rt_press_{c}": rt[cond == c].mean() for c in conds}, **{f"n_press_{c}": int((cond == c).sum()) for c in conds})
    return pd.DataFrame(acc / N_DRAWS, index=pd.Index(conds, name="cond"), columns=cols), info


def adjusted(g, cols):
    """Time-point coefficients of measure ~ time point + search time, on a participant's press trials."""
    g = g[g.press & g.rt.notna()]
    X = np.column_stack([np.ones(len(g)), g.cond.values == "post1", g.cond.values == "post2", g.rt.values - g.rt.mean()]).astype(float)
    rows = {}
    for c in cols:
        ok = np.isfinite(g[c].values)
        b = np.linalg.lstsq(X[ok], g[c].values[ok], rcond=None)[0]
        rows[c] = {"post1 - pre": b[1], "post2 - pre": b[2], "post2 - post1": b[2] - b[1], "per second": b[3]}
    return pd.DataFrame(rows).T


def _t(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    r = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), ci95=ci95(v), t=r.statistic, p=r.pvalue)


def analyse(t, out):
    cols = [c for c in t.columns if c not in KEYS + ["cond", "position", "press", "rt"]]
    rng = np.random.default_rng(0)
    per, pair, adj, match = [], [], [], []
    for (roi, sub), g in t.groupby(["roi", "subject"]):
        if set(g.cond) != set(CONDS):
            continue
        tag = dict(roi=roi, subject=sub)
        per.append(g.groupby("cond")[cols].mean().assign(**tag, trials="all"))
        per.append(g[g.press].groupby("cond")[cols].mean().assign(**tag, trials="press"))
        m, info = matched_means(g, cols, CONDS, rng)
        if m is not None:
            per.append(m.assign(**tag, trials="matched"))
            if roi == REGIONS[0]:
                match.append(dict(subject=sub, matching="three time points", **info))
        for name, (a, b) in CONTRASTS.items():
            m, info = matched_means(g, cols, [a, b], rng)
            if m is not None:
                pair.append((m.loc[a] - m.loc[b]).to_frame().T.assign(**tag, contrast=name))
                if roi == REGIONS[0]:
                    match.append(dict(subject=sub, matching=name, **info))
        if g.press.sum() >= 3 * MIN_MATCHED:
            adj.append(adjusted(g, cols).T.rename_axis("contrast").reset_index().assign(**tag))
    per, pair, adj = pd.concat(per).reset_index(), pd.concat(pair), pd.concat(adj)
    pd.DataFrame(match).to_csv(opj(out, "matching.csv"), index=False)
    per.to_csv(opj(out, "participant_means.csv"), index=False)

    rows = []
    for (trials, roi), g in per.groupby(["trials", "roi"]):
        for m in cols:
            w = g.pivot(index="subject", columns="cond", values=m)[CONDS].dropna()
            for c in CONDS:
                rows.append(dict(trials=trials, roi=roi, measure=m, test=c, **_t(w[c])))
            for name, (a, b) in CONTRASTS.items():
                rows.append(dict(trials=trials, roi=roi, measure=m, test=name, **_t(w[a] - w[b])))
    for trials, d in (("matched_pair", pair), ("adjusted", adj)):
        for (roi, name), g in d.groupby(["roi", "contrast"]):
            for m in cols:
                rows.append(dict(trials=trials, roi=roi, measure=m, test=name, **_t(g[m])))
    res = pd.DataFrame(rows)
    res.to_csv(opj(out, "group_contrasts.csv"), index=False)
    return per, res, pd.DataFrame(match)


def figure(per, res, match, plot_dir):
    apply_style()
    roi, colour = "parietal_precuneus", REGION_COLOUR["parietal_precuneus"]
    set_colour = {"all": MUTED, "press": BLUE, "matched": ORANGE}
    fig = plt.figure(figsize=(WIDTH, 5.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.5, wspace=1.5)

    # A: search time before and after matching
    ax = fig.add_subplot(gs[0, 0:2])
    m3 = match[match.matching == "three time points"]
    x = np.arange(3)
    for key, c, lab, dx in (("rt_press", BLUE, "Trials with a press", -0.08), ("rt_matched", ORANGE, "Matched", 0.08)):
        V = m3[[f"{key}_{k}" for k in CONDS]].values
        ax.plot(x + dx, V.T, color=c, lw=0.35, alpha=0.25)
        ax.errorbar(x + dx, V.mean(0), yerr=ci95(V), color=c, marker="o", ms=4, lw=1.6, capsize=2.5, mec=INK, mew=0.5, ecolor=INK, label=lab, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in CONDS]); ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("Search time (s)"); ax.set_ylim(0.5, 4.5)
    ax.legend(loc="upper right", handlelength=1.2)
    title(ax, "Search time"); panel_letter(ax, "A", dx=-0.16)

    # B: the two measures in parietal + precuneus at the three time points, per trial set
    for j, (m, lab) in enumerate((("abs slope, five items", "Absolute slope"), ("amplitude, five items", "Sinusoidal amplitude"))):
        ax = fig.add_subplot(gs[0, 2 + 2 * j:4 + 2 * j])
        ax.axhline(0, color=INK, lw=0.6)
        top = 0
        for (key, name), dx in zip(SETS.items(), (-0.2, 0, 0.2)):
            g = per[(per.roi == roi) & (per.trials == key)]
            W = g.pivot(index="subject", columns="cond", values=m)[CONDS].dropna()
            mean, c = W.mean().values, ci95(W.values)
            ax.errorbar(x + dx, mean, yerr=c, color=set_colour[key], marker="o", ms=4, lw=1.4, capsize=2, mec=INK, mew=0.5,
                        label=f"{name} (N = {len(W)})", zorder=4)
            top = max(top, (mean + c).max())
            r = res[(res.roi == roi) & (res.trials == key) & (res.measure == m)].set_index("test")
            for i, cnd in enumerate(CONDS):
                ax.text(i + dx, mean[i] + c[i] + 0.02, stars(r.loc[cnd, "p"]), ha="center", va="bottom", fontsize=7, color=set_colour[key])
        ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c in CONDS]); ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(top=top * 1.75)
        ax.set_ylabel("Retrieved - control\n(points per position)" if j == 0 else "")
        title(ax, lab, colour)
        if j == 0:
            panel_letter(ax, "B", dx=-0.2)
            ax.legend(loc="upper left", handlelength=1.2, fontsize=6, borderaxespad=0.1)

    # C: change from the evening to the first morning retrieval per region, per version
    versions = {"all": ("All trials", MUTED, "o"), "press": ("Trials with a press", BLUE, "o"),
                "matched": ("Matched, three time points", ORANGE, "o"), "matched_pair": ("Matched, the two time points", ORANGE, "s"),
                "adjusted": ("Search time as covariate", "#1baf7a", "D")}
    for j, (m, lab) in enumerate((("abs slope, five items", "Absolute slope"), ("amplitude, five items", "Sinusoidal amplitude"))):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        ax.axvline(0, color=INK, lw=0.6)
        for i, r in enumerate(REGIONS):
            for (key, (name, c, mk)), dy in zip(versions.items(), np.linspace(-0.3, 0.3, len(versions))):
                row = res[(res.roi == r) & (res.trials == key) & (res.measure == m) & (res.test == "post1 - pre")].iloc[0]
                ax.errorbar(row["mean"], i + dy, xerr=row["ci95"], color=c, marker=mk, ms=3.6, capsize=1.6, lw=1, mec=INK, mew=0.4,
                            ecolor=c, label=name if i == 0 else None)
                ax.text(row["mean"] + row["ci95"] + 0.02, i + dy, stars(row["p"]), va="center", fontsize=6.5, color=c)
        ax.set_yticks(range(len(REGIONS))); ax.set_yticklabels([REGION_LABEL[r] for r in REGIONS] if j == 0 else [])
        for lab_, r in zip(ax.get_yticklabels(), REGIONS):
            lab_.set_color(REGION_COLOUR[r])
        ax.invert_yaxis()
        ax.set_xlabel("Change from the evening to the first morning\n(retrieved - control, points per position)")
        title(ax, lab)
        if j == 0:
            panel_letter(ax, "C", dx=-0.42)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=6.5, handlelength=1.2)
    save(fig, opj(plot_dir, "matched_search_time"))


def main():
    out, plot_dir = str(step_output_dir(__file__)), str(step_plot_dir(__file__))
    t = trial_table(out)
    per, res, match = analyse(t, out)
    figure(per, res, match, plot_dir)
    m3 = match[match.matching == "three time points"]
    print(m3.drop(columns=["subject", "matching"]).describe().loc[["count", "mean", "min"]].round(2).T.to_string())
    show = res[(res.roi == "parietal_precuneus") & res.test.isin(list(CONTRASTS) + ["post1"])]
    print(show.pivot_table(index=["measure", "test"], columns="trials", values="t").round(2).to_string())
    print(f"-> {out}\n-> {plot_dir}")


if __name__ == "__main__":
    main()
