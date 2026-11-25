"""
Shared look of the paper figures.

Colours, region names and time-point names follow the figures of the univariate
stage (E14, E15), so that decoding and univariate figures can stand side by side.

    from figure_style import *        # after sys.path contains paper_pipeline/
    apply_style()

Conventions
    - one colour per region where regions are compared (REGION_COLOUR)
    - blue / orange for a pair of conditions (retrieved / control sequence;
      all five items / without the retrieved item; shown / recalled)
    - time points always in the order evening, first morning, repeated morning
    - participant data as thin grey lines or small dots, group mean on top with
      the 95 % confidence interval across participants
    - stars: * p < .05, ** p < .01, *** p < .001; the caption says whether corrected
    - panel letters top left, bold; titles left-aligned; no top and right axis lines
    - sizes are for a figure placed at full text width (WIDTH inches), text 7-9 pt
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
GREY, GRID, LIGHT = "#b9b8b3", "#e4e3df", "#f3f2ee"
WIDTH = 7.2                                    # inches, full text width

# region names as in the univariate figures (keys: decoding names in lower case)
REGION_LABEL = {"vis": "Visual", "parietal_precuneus": "Parietal + precuneus", "mtl": "MTL", "hpc": "Hippocampus",
                "somato": "Somatosensory (control)", "caudal_acc": "Caudal ACC (control)"}
REGION_SHORT = {"vis": "Visual", "parietal_precuneus": "Parietal +\nprecuneus", "mtl": "MTL", "hpc": "Hippo-\ncampus",
                "somato": "Somato-\nsensory", "caudal_acc": "Caudal\nACC"}
# visual, MTL and hippocampus as in E14/E15; parietal + precuneus takes the precuneus colour; controls in grey
REGION_COLOUR = {"vis": "#e87ba4", "parietal_precuneus": "#eda100", "mtl": "#1baf7a", "hpc": "#eb6834",
                 "somato": "#8a8984", "caudal_acc": "#52514e"}
REGIONS = list(REGION_LABEL)

CONDS = ["pre", "post1", "post2"]
COND_LABEL = {"pre": "Evening", "post1": "First\nmorning", "post2": "Repeated\nmorning", "pooled": "All"}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 400, "figure.dpi": 150,
    })


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def p_text(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def ci95(x, axis=0):
    """Half-width of the 95 % confidence interval of the mean across participants."""
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    return stats.t.ppf(0.975, n - 1) * np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def panel_letter(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def title(ax, text, colour=INK):
    ax.set_title(text, loc="left", color=colour, fontweight="bold", pad=4)


def save(fig, path_without_extension):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_without_extension}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
