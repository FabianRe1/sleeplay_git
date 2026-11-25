#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A6 - Shape of the reaction-time profile over sequence positions.

A5 reports the slope of reaction time over all five positions. This step asks
two further questions that matter for how a sequence item is reached:

  1  Last-position dip. If items are reached by stepping forward through the
     sequence, reaction time should keep rising up to position 5. A faster
     position 5 suggests another access route for the last item (for example
     from the end). Tested as RT(5) - RT(4), and as RT(5) minus the value a
     straight line through positions 1-4 predicts for position 5.
  2  Does the position cost disappear with practice? Slope over positions 1-4
     (seconds per position) per session, and the same slope as a share of the
     participant's mean reaction time in that session. Direct access ("cached"
     answers) predicts a slope near zero; general speeding predicts a constant share.

Per participant, session (evening, morning 1, morning 2) and trial type; reaction
times of retrieved trials only (as in A5). One-sample t-tests against zero per
session, paired t-tests between sessions.

Input   outputs/A4_aggregate_behavior/all_subjects_behav_data.csv
Output  outputs/A6_position_cost/position_cost_per_participant.csv
        outputs/A6_position_cost/position_cost_tests.csv
        outputs/A6_position_cost/position_cost_summary.txt
Plots   plots/A6_position_cost/rt_profile_by_session.{pdf,png}

Usage:  python A6_position_cost.py

Not a port: added on 2026-11-19. Uses the trial extraction of A5 unchanged.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

OUT_DIR = step_output_dir(__file__)
PLOT_DIR = step_plot_dir(__file__)
BLUE, ORANGE, AQUA, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
MEASURES = {"slope_1_4": "slope over positions 1-4 (s per position)",
            "slope_share": "slope over positions 1-4 as a share of mean RT",
            "dip_5_minus_4": "RT(5) - RT(4) (s)",
            "dip_5_vs_line": "RT(5) - value predicted from positions 1-4 (s)"}


def _load_a5():
    spec = importlib.util.spec_from_file_location("a5", Path(__file__).with_name("A5_analyse_behavior.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_participant(trials):
    ret = trials[trials["retrieved"].astype(bool)].dropna(subset=["rt", "seq_pos"])
    cell = ret.groupby(["sub_id", "session_label", "trial_type", "seq_pos"])["rt"].mean().unstack("seq_pos")
    rows = []
    for (sid, ses, tt), r in cell.iterrows():
        first = r.reindex([1, 2, 3, 4]).astype(float)
        if first.notna().sum() < 3:
            continue
        x = first.dropna().index.values.astype(float)
        slope, icpt = np.polyfit(x, first.dropna().values, 1)
        mean_rt = r.mean()
        rt5 = r.get(5, np.nan)
        rows.append(dict(sub_id=sid, session_label=ses, trial_type=tt, mean_rt=mean_rt, slope_1_4=slope,
                         slope_share=slope / mean_rt, dip_5_minus_4=rt5 - r.get(4, np.nan),
                         dip_5_vs_line=rt5 - (icpt + slope * 5), **{f"rt_pos{k}": r.get(k, np.nan) for k in range(1, 6)}))
    return pd.DataFrame(rows)


def _t(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def main():
    a5 = _load_a5()
    df = pd.read_csv(step_input_dir("A4_aggregate_behavior") / "all_subjects_behav_data.csv", low_memory=False)
    trials = a5.extract_retrieval_trials(df)
    sessions = [s for s in a5.SESSION_LABELS.values() if s in set(trials["session_label"])]
    pp = per_participant(trials)
    pp.to_csv(OUT_DIR / "position_cost_per_participant.csv", index=False)

    rows = []
    for tt, g in pp.groupby("trial_type"):
        for m in MEASURES:
            w = g.pivot(index="sub_id", columns="session_label", values=m)
            for s in sessions:
                rows.append(dict(trial_type=tt, measure=m, test=s, **_t(w[s])))
            for a, b in ((1, 0), (2, 1), (2, 0)):
                if len(sessions) > max(a, b):
                    rows.append(dict(trial_type=tt, measure=m, test=f"{sessions[a]} minus {sessions[b]}",
                                     **_t(w[sessions[a]] - w[sessions[b]])))
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT_DIR / "position_cost_tests.csv", index=False)

    lines = ["Shape of the reaction-time profile over sequence positions (A6).", ""]
    for tt in sorted(tests.trial_type.unique()):
        for m, lab in MEASURES.items():
            lines.append(f"{tt} trials, {lab}")
            for _, r in tests[(tests.trial_type == tt) & (tests.measure == m)].iterrows():
                lines.append(f"  {r.test:42s} N = {int(r.n):2d}  mean = {r['mean']:+.3f}  t = {r.t:+.2f}  p = {r.p:.4g}")
            lines.append("")
    (OUT_DIR / "position_cost_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    types = sorted(pp.trial_type.unique())
    fig, axes = plt.subplots(1, len(types), figsize=(3.3 * len(types), 2.7), sharey=True, squeeze=False)
    for ax, tt in zip(axes[0], types):
        for s, col in zip(sessions, (BLUE, ORANGE, AQUA)):
            g = pp[(pp.trial_type == tt) & (pp.session_label == s)]
            y = [g[f"rt_pos{k}"].mean() for k in range(1, 6)]
            e = [stats.sem(g[f"rt_pos{k}"].dropna()) * stats.t.ppf(0.975, g[f"rt_pos{k}"].notna().sum() - 1)
                 for k in range(1, 6)]
            ax.errorbar(range(1, 6), y, yerr=e, fmt="-o", ms=4, lw=1.6, color=col, mec="white", mew=0.7,
                        elinewidth=0.9, label=s)
        ax.set_title(f"{tt} trials"); ax.set_xlabel("Sequence position"); ax.set_xticks(range(1, 6))
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Reaction time (s)")
    axes[0, -1].legend(loc="upper left")
    fig.savefig(PLOT_DIR / "rt_profile_by_session.pdf"); fig.savefig(PLOT_DIR / "rt_profile_by_session.png")
    plt.close(fig)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
