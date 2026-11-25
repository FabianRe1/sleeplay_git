#!/usr/bin/env python3
"""
Do the session differences in the classifier measures hold on trials matched
for search time?

Search time (question mark to button press) falls from the evening to the
morning. 16_rt_and_reactivation.py already relates the measures to search time
within participants (part B) and adds search time as a covariate (part C).
This script adds the matched comparison used for the univariate analyses
(E13): within each participant the object trials with a button press are sorted
into bins of BIN seconds of search time, and from every bin the same number of
trials is taken at each of the three time points (evening, morning trials 1-70,
morning trials 71-140). This is repeated N_DRAWS times with different random
choices and the participant's mean per time point is averaged over the draws.

Measures (object trials, four main ROIs):
    from 16_rt_and_reactivation: retrieved item, predecessor, successor,
        |slope| five items, control |slope| five items
    from the sinusoid tables with the without-target columns (19_...):
        sinusoid five items / without target, each for the retrieved sequence,
        the control sequence and their difference (absolute amplitudes, as C6)

For every measure and ROI: mean per time point and the three paired contrasts,
for (a) all trials, (b) trials with a button press, (c) matched trials.

Input   outputs/16_rt_and_reactivation/trial_table.csv
        outputs/19_without_target_sinusoid/B9_thesis/<roi>/<roi>_windowed_fit_results_<session>.csv
Output  outputs/21_decoder_search_time_matched/session_contrasts.csv, matching.csv
Usage:  python 21_decoder_search_time_matched.py
"""

import os
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = opj(HERE, "outputs", "21_decoder_search_time_matched")
ROIS = ["vis", "parietal_precuneus", "mtl", "caudal_acc"]
CONDS = ["pre", "post1", "post2"]
CONTRASTS = {"post1 - pre": ("post1", "pre"), "post2 - pre": ("post2", "pre"), "post2 - post1": ("post2", "post1")}
BIN, N_DRAWS, MIN_MATCHED = 0.25, 200, 8
SINUS = {"sinusoid five items": ("retrieved_w_amplitude", "control_w_amplitude"),
         "sinusoid without target": ("wo_retrieved_w_amplitude", "wo_control_w_amplitude")}


def load():
    t = pd.read_csv(opj(HERE, "outputs", "16_rt_and_reactivation", "trial_table.csv"))
    t["trial"] = t.trial.astype(int)
    t["slope retrieved - control"] = t["|slope| five items"] - t["control |slope| five items"]
    parts = []
    for roi in ROIS:
        for session in ("presleep", "postsleep"):
            d = pd.read_csv(opj(HERE, "outputs", "19_without_target_sinusoid", "B9_thesis", roi,
                                f"{roi}_windowed_fit_results_{session}.csv"))
            x = pd.DataFrame(dict(subject=d.subject, trial=d.trial.astype(int), session=session, roi=roi))
            for name, (r, c) in SINUS.items():
                x[f"{name} | retrieved"], x[f"{name} | control"] = d[r].abs(), d[c].abs()
                x[f"{name} | retrieved - control"] = d[r].abs() - d[c].abs()
            parts.append(x)
    t = t.drop(columns=["sinusoid five items"]).merge(pd.concat(parts), on=["subject", "trial", "session", "roi"], how="left")
    return t


def matched_means(g, cols, rng):
    """Mean of `cols` per time point over the matched trials, averaged over N_DRAWS draws; and the matching summary."""
    g = g[g.rt.notna()]
    bins = np.floor(g.rt.values / BIN).astype(int)
    cond = g.cond.values
    idx = {(b, c): np.flatnonzero((bins == b) & (cond == c)) for b in np.unique(bins) for c in CONDS}
    k = {b: min(len(idx[(b, c)]) for c in CONDS) for b in np.unique(bins)}
    n = sum(k.values())
    if n < MIN_MATCHED:
        return None, None
    v = g[cols].values
    acc = np.zeros((len(CONDS), len(cols)))
    rts = np.zeros(len(CONDS))
    for _ in range(N_DRAWS):
        for ci, c in enumerate(CONDS):
            sel = np.concatenate([rng.choice(idx[(b, c)], kb, replace=False) for b, kb in k.items() if kb])
            acc[ci] += np.nanmean(v[sel], axis=0)
            rts[ci] += g.rt.values[sel].mean()
    return (pd.DataFrame(acc / N_DRAWS, index=CONDS, columns=cols),
            dict(n_matched=n, **{f"rt_{c}": rts[i] / N_DRAWS for i, c in enumerate(CONDS)},
                 **{f"n_{c}": int((cond == c).sum()) for c in CONDS}))


def main():
    os.makedirs(OUT, exist_ok=True)
    t = load()
    cols = [c for c in t.columns if c not in ("subject", "trial", "session", "cond", "kind", "position", "rt", "correct", "roi",
                                              "predecessor - successor", "sequence evidence")]
    rng = np.random.default_rng(0)
    per, match = [], []
    for (roi, sub), g in t.groupby(["roi", "subject"]):
        if set(g.cond) != set(CONDS):
            continue
        per.append(g.groupby("cond")[cols].mean().assign(roi=roi, subject=sub, trials="all"))
        per.append(g[g.rt.notna()].groupby("cond")[cols].mean().assign(roi=roi, subject=sub, trials="with press"))
        m, info = matched_means(g, cols, rng)
        if m is not None:
            m.index.name = "cond"
            per.append(m.assign(roi=roi, subject=sub, trials="matched"))
            if roi == ROIS[0]:
                match.append(dict(subject=sub, **info))
    per = pd.concat(per).reset_index()
    pd.DataFrame(match).to_csv(opj(OUT, "matching.csv"), index=False)
    rows = []
    for (trials, roi), g in per.groupby(["trials", "roi"]):
        for m in cols:
            w = g.pivot(index="subject", columns="cond", values=m)[CONDS].dropna()
            row = dict(trials=trials, roi=roi, measure=m, n=len(w), **{c: w[c].mean() for c in CONDS})
            for name, (a, b) in CONTRASTS.items():
                r = stats.ttest_1samp(w[a] - w[b], 0)
                row[f"{name} t"], row[f"{name} p"] = r.statistic, r.pvalue
            for c in CONDS:
                row[f"{c} vs 0 t"] = stats.ttest_1samp(w[c], 0).statistic
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(opj(OUT, "session_contrasts.csv"), index=False)
    print(pd.DataFrame(match).describe().loc[["mean", "min"]].round(2).to_string())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
