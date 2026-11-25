#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5 - Figures of the whole-brain group maps (E4).

For the main model `ret_press` (other models: E5_MODELS=a,b) and every trial type or trial phase
that E4 has results for:
  1_engagement_<model>_<kind>     rows: all time points pooled, evening, first morning
                                  retrieval, repeated morning retrieval
  2_changes_<model>_<kind>        rows: first morning minus evening, repeated minus first morning
  3_reaction_time_<kind>          change in activity per second of reaction time
Each map is shown twice, as axial slices and as sagittal slices (x = 0 is the midline),
with t values where the TFCE-corrected p is below .05
(two-sided; warm = above baseline / increase, cool = below baseline / decrease),
on the MNI template. Rows whose map is missing are left out.

Input   outputs/E4_wholebrain_group/<model>/<kind>/<contrast>_t.nii.gz, _t_thresholded.nii.gz
Plots   plots/E5_wholebrain_figures/*.png, *.pdf

Usage:  python E5_wholebrain_figures.py

Not a port of the original pipeline: added 2026-11-22.
"""

from __future__ import annotations

import os
import sys
from os.path import join as opj
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_plot_dir

PLOT_DIR = step_plot_dir(__file__)
CUTS = (-18, -8, 2, 12, 22, 32, 42, 52, 62)            # axial, z in mm
CUTS_X = (-50, -36, -22, -8, 0, 8, 22, 36, 50)          # sagittal, x in mm (negative = left)
TITLES = {"pooled": "All three time points", "pre": "Evening", "post1": "First morning retrieval",
          "post2": "Repeated morning retrieval", "post1_minus_pre": "First morning retrieval minus evening",
          "post2_minus_post1": "Repeated minus first morning retrieval",
          "post2_minus_pre": "Repeated morning retrieval minus evening",
          "post_minus_pre": "Both morning retrievals minus evening",
          "per_second_rt": "Per second of reaction time"}
FIGURES = [("1_engagement", ("pooled", "pre", "post1", "post2"), 12),
           ("2_changes", ("post1_minus_pre", "post2_minus_post1", "post2_minus_pre", "post_minus_pre"), 8),
           ("3_reaction_time", ("per_second_rt",), 8)]


def draw(rows, vmax, path, src):
    rows = [(c, opj(src, f"{c}_t_thresholded.nii.gz")) for c in rows]
    rows = [(c, f) for c, f in rows if os.path.exists(f)]
    if not rows:
        return False
    fig, axes = plt.subplots(2 * len(rows), 1, figsize=(11, 3.5 * len(rows) + 0.3), squeeze=False)
    for i, (c, f) in enumerate(rows):
        img = nib.load(f)
        n = len(pd.read_csv(f.replace("_t_thresholded.nii.gz", "_participants.csv")))
        arr = np.asarray(img.dataobj)
        for ax, mode, cuts in ((axes[2 * i, 0], "z", CUTS), (axes[2 * i + 1, 0], "x", CUTS_X)):
            if not (arr != 0).any():
                ax.axis("off")
                if mode == "z":
                    ax.text(0.5, 0.5, "no voxel with corrected p < .05", ha="center", va="center", fontsize=9)
            else:
                plotting.plot_stat_map(img, display_mode=mode, cut_coords=cuts, axes=ax, vmax=vmax, threshold=1e-6,
                                       cmap="RdBu_r", colorbar=True, annotate=True, draw_cross=False,
                                       black_bg=False)
        axes[2 * i, 0].set_title(f"{TITLES[c]}  (n = {n})", fontsize=9, fontweight="bold", loc="left")
    fig.text(0.99, 0.005, "colour: t value where TFCE-corrected p < .05 (two-sided); coordinates in mm (MNI), x < 0 = left",
             ha="right", va="bottom", fontsize=7, color="#52514e")
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    base = str(step_input_dir("E4_wholebrain_group"))
    for root, suffix in ((base, ""),):
        for model in os.environ.get("E5_MODELS", "ret_press").split(","):     # main model; earlier ones via E5_MODELS=a,b
            for kind in ("object", "scene", "phase_wait", "phase_speak", "object_cue", "object_search", "scene_cue", "scene_search"):
                src = opj(root, model, kind)
                if not os.path.isdir(src):
                    continue
                for name, rows, vmax in FIGURES:
                    if name == "3_reaction_time" and model not in ("fixed", "fixed_fixbase", "response_fixbase"):
                        continue
                    tag = f"{name}_{model}_{kind}" + suffix
                    if draw(rows, vmax, opj(str(PLOT_DIR), tag), src):
                        print("saved", tag)


if __name__ == "__main__":
    main()
