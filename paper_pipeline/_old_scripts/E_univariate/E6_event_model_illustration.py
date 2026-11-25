#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E6 - Illustration: what the two event models estimate (simulation, no data).

Stage E reports every comparison under two event models: every trial as a 3 s
event, or every trial lasting from the cue to the response. This step draws why.
Three kinds of region are simulated without noise:

    brief        a short response to the cue, the same on every trial
    sustained    active from the cue until the response (same strength, longer on slow trials)
    both         longer AND stronger per second on slow trials

For each, the figure shows the neural activity and the measured signal on a
fast and a slow trial, and the estimate of both models as a function of the
trial's reaction time (least-squares fit of the model's predicted response to
the simulated signal; canonical SPM response function, as in D1 / E3).

Reading: with fixed events the estimate rises with reaction time whenever a
region is active for longer; with events from cue to response it is flat for
such a region and falls for a region that only responds briefly to the cue.
Limit: a region active for a fixed time with a strength exactly proportional to
the reaction time gives nearly the same signal as a sustained region; the two
models cannot tell these apart.

Input   none
Plots   plots/E6_event_model_illustration/event_models.{png,pdf}

Usage:  python E6_event_model_illustration.py

Not a port of the original pipeline: added 2026-11-22.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn.glm.first_level.hemodynamic_models import spm_hrf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_plot_dir

PLOT_DIR = step_plot_dir(__file__)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1d", "#52514e", "#e4e3df"
DT = 0.1
T = np.arange(0, 30, DT)
HRF = spm_hrf(DT, oversampling=1, time_length=32.0)
HRF = HRF / HRF.max()
CUE = 1.015                 # s from the retrieval cue to the response prompt (as in D1)
FIXED = 3.0                 # s, fixed event model (as in D1)
FAST, SLOW = 1.0, 4.0       # reaction times of the two example trials


def box(dur, amp=1.0):
    x = np.zeros_like(T)
    x[(T >= 0) & (T < dur)] = amp
    return x


def bold(neural):
    return np.convolve(neural, HRF)[:len(T)] * DT


REGIONS = [("Brief response to the cue", lambda rt: box(0.5, 3.0), "same response on fast and slow trials"),
           ("Active until the answer is found", lambda rt: box(CUE + rt), "same strength, lasts longer on slow trials"),
           ("Longer and stronger on slow trials", lambda rt: box(CUE + rt, 0.4 * (CUE + rt)),
            "lasts longer and is stronger per second")]
MODELS = [("Fixed 3 s", lambda rt: box(FIXED), BLUE), ("cue to response", lambda rt: box(CUE + rt), ORANGE)]
VERDICT = ["Estimate: flat / falls with reaction time", "Estimate: rises / flat", "Estimate: rises / still rises"]


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(3, 3, figsize=(10.5, 7.6),
                           gridspec_kw=dict(height_ratios=[0.8, 1.2, 1.3], hspace=0.62, wspace=0.28))
    rts = np.linspace(0.5, 5, 46)
    for j, (name, neural, note) in enumerate(REGIONS):
        a = ax[0, j]
        for rt, ls, lab in ((FAST, "--", "fast trial (answer after 1 s)"), (SLOW, "-", "slow trial (answer after 4 s)")):
            a.step(T, neural(rt), where="post", color=INK, lw=1.4, ls=ls, label=lab)
        a.set_xlim(-0.5, 8); a.set_ylim(0, 3.4); a.set_title(name); a.set_xlabel("Time from cue (s)")
        a.text(0.98, 0.9, note, transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
        if j == 0:
            a.set_ylabel("Neural activity")
            a.legend(loc="center right", fontsize=7.5, handlelength=2.2)
        a = ax[1, j]
        for rt, ls in ((FAST, "--"), (SLOW, "-")):
            a.plot(T, bold(neural(rt)), color=INK, lw=1.6, ls=ls)
        a.set_xlim(-0.5, 20); a.set_ylim(0, [2.9, 5.2, 10.5][j]); a.set_xlabel("Time from cue (s)")
        a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
        if j == 0:
            a.set_ylabel("Measured signal")
        a = ax[2, j]
        for mname, reg, col in MODELS:
            b = np.array([bold(reg(rt)) @ bold(neural(rt)) / (bold(reg(rt)) @ bold(reg(rt))) for rt in rts])
            a.plot(rts, b, color=col, lw=2)
            a.plot([rts[-1]], [b[-1]], "o", color=col, ms=4)
            a.text(rts[-1] + 0.08, b[-1], mname, color=INK, fontsize=7.5, va="center")
        a.set_xlim(0.3, 6.9); a.set_ylim(0, None); a.set_xticks([1, 2, 3, 4, 5])
        a.set_xlabel("Reaction time of the trial (s)"); a.set_title(VERDICT[j], fontsize=8.5)
        a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
        if j == 0:
            a.set_ylabel("Estimated activity (beta)")
    fig.text(0.125, 0.0, "Blue: every trial modelled as a 3 s event.  Orange: every trial modelled from the cue to the "
             "response.  Simulated signals, no noise.", fontsize=7.8, color=MUTED)
    fig.savefig(PLOT_DIR / "event_models.png", dpi=180, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "event_models.pdf", bbox_inches="tight")
    print(f"Done -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
