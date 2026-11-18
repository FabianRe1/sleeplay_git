# -*- coding: utf-8 -*-
"""Phase-constrained windowed sinusoidal fitting for Project 5 (LCI / generalization).

Fits a single-cycle sinusoid (onset fixed at the start of the fit window;
amplitude, duration, and offset free) to per-trial × per-condition SODA slope
timecourses. Runs the fit twice per trial-condition: once on the full window
(sensory + non-sensory) and once restricted to the non-sensory window.
Then runs an LME per (window × response × ROI) comparing the four conditions
(sensory, semantic_gen, non_semantic_gen, no_gen) with no_gen as reference.

See PROJECT5_REPLAY_HANDOFF.txt for the rationale and design.

Usage
-----
    1. Edit CONFIG below to match your column names and fit windows.
    2. Load your long-format SODA timecourse data:
           df = pd.read_csv("your_soda_per_tr.csv")
           df = rename_to_internal(df)
    3. Fit sinusoids:
           metrics_df = compute_trial_level_fits(df)
    4. Run statistics + plots:
           run_pipeline(metrics_df, "results/")
"""

import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, ttest_1samp
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION — edit to match your project
# ============================================================

CONFIG = {
    # -- COLUMN MAPPING --
    # Internal name -> your column name
    "col_subject":   "subject",
    "col_trial":     "trial",
    "col_condition": "condition",     # values listed below
    "col_tr":        "tr",            # TR index within the ISI window
    "col_slope":     "slope",         # SODA regression slope at each TR
    "col_roi":       "roi",           # set to None if single ROI
    "col_block":     "block",         # optional; set to None if absent
    "col_day":       "day",           # optional; set to None if absent
    "col_is_reversal": "is_reversal", # optional; set to None if absent

    # -- CONDITION LABELS --
    # The four strings that should appear in your condition column.
    # Edit only if your data uses different label strings.
    "cond_sensory":           "sensory",
    "cond_semantic_gen":      "semantic_gen",
    "cond_non_semantic_gen":  "non_semantic_gen",
    "cond_no_gen":            "no_gen",

    # -- FIT WINDOWS --
    # Define which TRs belong to each window. Two options:
    #
    # Option A — explicit lists of TR indices:
    #   "full_window_trs":  list(range(-15, 12)),  # all TRs
    #   "non_sensory_trs":  list(range(2, 12)),    # post-sensory only
    #
    # Option B — boundary-based (set window_mode="bounds"):
    #   "full_window_bounds":        (-np.inf, np.inf),  # use all available TRs
    #   "non_sensory_window_bounds": (2, np.inf),         # TRs >= 2
    #
    # The script accepts either; if both are provided, explicit lists win.
    "window_mode":               "bounds",  # "explicit" or "bounds"
    "full_window_bounds":        (-np.inf, np.inf),  # tuple (min_tr, max_tr) inclusive
    "non_sensory_window_bounds": (2, np.inf),        # tuple (min_tr, max_tr) inclusive
    "full_window_trs":           None,
    "non_sensory_trs":           None,

    # -- SINUSOID FIT PARAMETERS --
    "duration_min":   3.0,          # minimum cycle duration (TRs)
    "duration_max":   None,         # None = let it grow to ~1.2x window length
    "duration_max_factor": 1.2,     # used when duration_max is None
    "taper_frac":     0.15,         # cosine taper fraction
    "n_restarts":     10,           # random restarts for curve_fit

    # -- ANALYSIS OPTIONS --
    "reference_condition": "no_gen",  # baseline condition in LME
    "min_trs_for_fit":     4,         # min valid TRs needed to attempt fit
}

# ROIs to iterate over in statistics. If `col_roi` is None, set to [None] and
# the code treats the data as a single ROI.
ROIS = ["visual", "mtl"]   # edit to match your ROI labels


# ============================================================
# COLUMN RENAMING HELPER
# ============================================================

def rename_to_internal(df, cfg=CONFIG):
    """Return a copy of df with columns renamed from user names to internal names.

    Internal names used throughout the rest of this module:
        subject, trial, condition, tr, slope, roi, block, day, is_reversal
    """
    rename_map = {}
    for internal_key, user_col in cfg.items():
        if not internal_key.startswith("col_") or user_col is None:
            continue
        internal_col = internal_key[4:]
        if user_col in df.columns and user_col != internal_col:
            rename_map[user_col] = internal_col
    return df.rename(columns=rename_map)


def wide_to_long(df_wide, condition_cols, id_cols=("subject", "trial", "tr", "roi"),
                 slope_name="slope", condition_name="condition"):
    """Reshape wide-format (one slope column per condition) to long.

    Parameters
    ----------
    df_wide : DataFrame with columns id_cols + condition_cols
    condition_cols : dict mapping condition label -> column name in df_wide,
                     e.g. {"sensory": "sensory_slope", "no_gen": "no_gen_slope", ...}
    """
    id_cols = [c for c in id_cols if c in df_wide.columns]
    pieces = []
    for cond_label, col_name in condition_cols.items():
        if col_name not in df_wide.columns:
            continue
        piece = df_wide[id_cols + [col_name]].copy()
        piece = piece.rename(columns={col_name: slope_name})
        piece[condition_name] = cond_label
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# ============================================================
# WINDOWED SINUSOIDAL MODEL
# ============================================================

def windowed_sinusoid(t, amplitude, duration, onset, offset,
                      taper_frac=CONFIG["taper_frac"]):
    """Single-cycle windowed sinusoid with cosine on/off ramps.

    y(t) = A * sin(2π/λ * (t - d)) * W(t; d, λ, taper) + c
    """
    t = np.asarray(t, dtype=float)
    phase = 2 * np.pi / duration * (t - onset)
    sine_part = amplitude * np.sin(phase)

    active = (t >= onset) & (t <= onset + duration)
    taper_len = taper_frac * duration
    window = np.zeros_like(t)
    window[active] = 1.0

    if taper_len > 0:
        rise = active & (t < onset + taper_len)
        if rise.any():
            window[rise] = 0.5 * (1 - np.cos(np.pi * (t[rise] - onset) / taper_len))
        fall = active & (t > onset + duration - taper_len)
        if fall.any():
            window[fall] = 0.5 * (1 - np.cos(np.pi * (onset + duration - t[fall]) / taper_len))

    return sine_part * window + offset


# ============================================================
# PHASE-CONSTRAINED FIT (onset fixed at window start)
# ============================================================

def _empty_fit_result(n_trs, onset=np.nan):
    return {
        "amplitude": np.nan, "duration": np.nan,
        "onset": onset, "offset": np.nan,
        "frequency": np.nan, "r_squared": np.nan,
        "fit_success": False, "y_hat": np.full(n_trs, np.nan),
    }


def fit_phase_constrained_sinusoid(trs, slope_values,
                                    duration_min=CONFIG["duration_min"],
                                    duration_max=CONFIG["duration_max"],
                                    duration_max_factor=CONFIG["duration_max_factor"],
                                    n_restarts=CONFIG["n_restarts"],
                                    taper_frac=CONFIG["taper_frac"],
                                    onset_override=None):
    """Fit windowed sinusoid with onset FIXED to the first valid TR.

    Free parameters: amplitude (A), duration (λ), offset (c).
    Onset is fixed at t_min by default (or onset_override if provided).

    Returns dict with: amplitude, duration, onset, offset, frequency,
                       r_squared, fit_success, y_hat
    The amplitude SIGN is meaningful here:
        A > 0  →  cycle starts going up    (forward-replay-like)
        A < 0  →  cycle starts going down  (backward-replay-like)
    """
    trs = np.asarray(trs, dtype=float)
    y_full = np.asarray(slope_values, dtype=float)
    valid = ~np.isnan(y_full)

    if valid.sum() < CONFIG["min_trs_for_fit"]:
        return _empty_fit_result(len(trs))

    t, y = trs[valid], y_full[valid]
    t_min, t_max = t.min(), t.max()
    window_length = t_max - t_min

    onset = float(onset_override) if onset_override is not None else float(t_min)

    if duration_max is None:
        duration_max = window_length * duration_max_factor
    duration_max = max(duration_max, duration_min + 0.1)

    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    # Free params: [amplitude, duration, offset]
    lower = [-np.inf, duration_min, -np.inf]
    upper = [np.inf,  duration_max, np.inf]

    def _model(t_, amplitude, duration, offset):
        return windowed_sinusoid(t_, amplitude, duration, onset, offset, taper_frac)

    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, window_length, y_mean]
        elif restart == 1:
            p0 = [-a0, window_length, y_mean]
        else:
            p0 = [
                rng.uniform(-a0 * 2, a0 * 2),
                rng.uniform(duration_min, duration_max),
                y_mean + rng.normal(0, a0 * 0.1),
            ]
        p0 = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, lower, upper)]
        try:
            popt, _ = curve_fit(_model, t, y, p0=p0,
                                bounds=(lower, upper), maxfev=5000)
            residual = np.sum((y - _model(t, *popt)) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_fit_result(len(trs), onset=onset)

    amplitude, duration, offset = best_result
    y_hat_valid = _model(t, *best_result)
    ss_res = np.sum((y - y_hat_valid) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    y_hat_full = np.full(len(trs), np.nan)
    y_hat_full[valid] = y_hat_valid

    return {
        "amplitude": amplitude, "duration": duration,
        "onset": onset, "offset": offset,
        "frequency": 1.0 / duration, "r_squared": r_squared,
        "fit_success": True, "y_hat": y_hat_full,
    }


# ============================================================
# TR-WINDOW MASKING
# ============================================================

def _get_window_mask(trs, window_name, cfg=CONFIG):
    """Boolean mask over `trs` selecting TRs that belong to a given fit window."""
    trs = np.asarray(trs)
    mode = cfg.get("window_mode", "bounds")

    if window_name == "full":
        explicit = cfg.get("full_window_trs")
        bounds   = cfg.get("full_window_bounds")
    elif window_name == "non_sensory":
        explicit = cfg.get("non_sensory_trs")
        bounds   = cfg.get("non_sensory_window_bounds")
    else:
        raise ValueError(f"Unknown window_name: {window_name}")

    if mode == "explicit" and explicit is not None:
        return np.isin(trs, list(explicit))
    if bounds is None:
        return np.ones_like(trs, dtype=bool)
    lo, hi = bounds
    return (trs >= lo) & (trs <= hi)


# ============================================================
# TRIAL-LEVEL FIT (per subject × trial × condition × ROI)
# ============================================================

def compute_trial_level_fits(df, cfg=CONFIG, verbose=True):
    """Fit a phase-constrained sinusoid to each trial × condition timecourse.

    Each row of df is one (subject, trial, condition, tr) with a slope value.
    The fit is run twice per trial × condition: full window and non-sensory.

    Returns
    -------
    DataFrame with one row per subject × trial × condition × ROI, with
    columns documented in PROJECT5_REPLAY_HANDOFF.txt.
    """
    required = ["subject", "trial", "condition", "tr", "slope"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Did you call rename_to_internal()?")

    group_cols = ["subject", "trial", "condition"]
    if "roi" in df.columns:
        group_cols.append("roi")

    extra_meta_cols = [c for c in ["block", "day", "is_reversal"] if c in df.columns]

    results = []
    groups = df.groupby(group_cols, sort=False)
    n_total = len(groups)
    if verbose:
        print(f"Fitting {n_total} (subject × trial × condition × ROI) combinations...")

    for i, (keys, trial_df) in enumerate(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        info = dict(zip(group_cols, keys))
        trial_df = trial_df.sort_values("tr")
        trs_all = trial_df["tr"].values
        y_all = trial_df["slope"].values

        # FULL WINDOW fit
        mask_full = _get_window_mask(trs_all, "full", cfg)
        full_fit = fit_phase_constrained_sinusoid(trs_all[mask_full], y_all[mask_full])
        info["full_amplitude"]     = full_fit["amplitude"]
        info["full_amplitude_abs"] = abs(full_fit["amplitude"]) if full_fit["fit_success"] else np.nan
        info["full_duration"]      = full_fit["duration"]
        info["full_onset"]         = full_fit["onset"]
        info["full_offset"]        = full_fit["offset"]
        info["full_r_squared"]     = full_fit["r_squared"]
        info["full_fit_success"]   = full_fit["fit_success"]

        # NON-SENSORY WINDOW fit
        mask_ns = _get_window_mask(trs_all, "non_sensory", cfg)
        ns_fit = fit_phase_constrained_sinusoid(trs_all[mask_ns], y_all[mask_ns])
        info["ns_amplitude"]     = ns_fit["amplitude"]
        info["ns_amplitude_abs"] = abs(ns_fit["amplitude"]) if ns_fit["fit_success"] else np.nan
        info["ns_duration"]      = ns_fit["duration"]
        info["ns_onset"]         = ns_fit["onset"]
        info["ns_offset"]        = ns_fit["offset"]
        info["ns_r_squared"]     = ns_fit["r_squared"]
        info["ns_fit_success"]   = ns_fit["fit_success"]

        # Carry over per-trial metadata
        for c in extra_meta_cols:
            vals = trial_df[c].dropna()
            info[c] = vals.iloc[0] if len(vals) > 0 else np.nan

        results.append(info)

        if verbose and (i + 1) % max(1, n_total // 20) == 0:
            print(f"  {i+1}/{n_total} done")

    return pd.DataFrame(results)


# ============================================================
# LME PIPELINE
# ============================================================

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


def fit_lme(df, formula, group_var="subject"):
    df_clean = df.dropna(subset=[group_var]).copy()
    dep_var = formula.split("~")[0].strip()
    if dep_var in df_clean.columns:
        df_clean = df_clean.dropna(subset=[dep_var])
    model = smf.mixedlm(formula, df_clean, groups=df_clean[group_var])
    return model.fit(reml=True, method="powell")


# Conditions and response variables to test
RESPONSES = [
    ("full_amplitude_abs", "|amplitude| (full window)"),
    ("full_amplitude",     "signed amplitude (full window)"),
    ("full_r_squared",     "R² (full window)"),
    ("ns_amplitude_abs",   "|amplitude| (non-sensory)"),
    ("ns_amplitude",       "signed amplitude (non-sensory)"),
    ("ns_r_squared",       "R² (non-sensory)"),
]


def analyze_condition_effects(metrics_df, log, rois=ROIS,
                               responses=RESPONSES,
                               reference=CONFIG["reference_condition"]):
    """LME for each (response × ROI): response ~ C(condition, Treatment(ref)) + (1|subject)."""
    log.append("=" * 70)
    log.append("CONDITION CONTRASTS — LME with subject random intercept")
    log.append(f"   Reference condition: {reference}")
    log.append("=" * 70)

    if "roi" not in metrics_df.columns:
        rois = [None]

    rows = []
    for response_col, response_label in responses:
        if response_col not in metrics_df.columns:
            continue
        log.append(f"\n--- {response_label}  [{response_col}] ---")
        for roi in rois:
            if roi is None:
                sub = metrics_df.copy()
                roi_label = "all"
            else:
                sub = metrics_df[metrics_df["roi"] == roi].copy()
                roi_label = roi

            sub = sub.dropna(subset=[response_col, "condition", "subject"])
            if sub.empty:
                log.append(f"  {roi_label:12s}: no data")
                continue

            sub["condition"] = sub["condition"].astype(str)
            n_subj = sub["subject"].nunique()
            n_trials = len(sub)

            try:
                formula = f'{response_col} ~ C(condition, Treatment("{reference}"))'
                res = fit_lme(sub, formula)
                log.append(f"\n  {roi_label}  (N={n_subj} subjects, {n_trials} trials):")
                for param in res.params.index:
                    if param == "Intercept" or param.startswith("Group Var"):
                        continue
                    coef = res.params[param]
                    p = res.pvalues[param]
                    z = res.tvalues[param]
                    se = res.bse[param]
                    nice = (param.replace(f'C(condition, Treatment("{reference}"))[T.', "")
                                 .replace("]", ""))
                    log.append(
                        f"    {nice:25s} vs {reference}: coef={coef:+.6f} "
                        f"(SE={se:.6f}), z={z:+.3f}, p={p:.4f}{sig_stars(p)}"
                    )
                    rows.append(dict(
                        response=response_label, response_col=response_col,
                        roi=roi_label, contrast=f"{nice}_vs_{reference}",
                        coef=coef, se=se, z=z, p_value=p,
                        n_subjects=n_subj, n_trials=n_trials,
                    ))
                intercept = res.params["Intercept"]
                log.append(f"    (intercept = {reference} mean = {intercept:+.6f})")
            except Exception as e:
                log.append(f"  {roi_label:12s}: LME FAILED – {e}")

    return pd.DataFrame(rows)


def analyze_pairwise_gen_contrast(metrics_df, log, rois=ROIS, responses=RESPONSES,
                                   cond_a="semantic_gen", cond_b="non_semantic_gen"):
    """Direct contrast between two non-reference conditions, e.g. semantic vs non-semantic."""
    log.append("\n" + "=" * 70)
    log.append(f"DIRECT PAIRWISE CONTRAST: {cond_a} vs {cond_b}")
    log.append(f"   LME: response ~ is_{cond_a} + (1|subject), on trials of these two conditions")
    log.append("=" * 70)

    if "roi" not in metrics_df.columns:
        rois = [None]

    rows = []
    for response_col, response_label in responses:
        if response_col not in metrics_df.columns:
            continue
        log.append(f"\n--- {response_label} ---")
        for roi in rois:
            if roi is None:
                sub = metrics_df.copy()
                roi_label = "all"
            else:
                sub = metrics_df[metrics_df["roi"] == roi].copy()
                roi_label = roi

            sub = sub[sub["condition"].isin([cond_a, cond_b])].copy()
            sub = sub.dropna(subset=[response_col])
            if sub.empty:
                continue
            sub["is_a"] = (sub["condition"] == cond_a).astype(int)
            try:
                res = fit_lme(sub, f"{response_col} ~ is_a")
                coef = res.params["is_a"]
                p = res.pvalues["is_a"]
                z = res.tvalues["is_a"]
                log.append(
                    f"  {roi_label:12s}: {cond_a} vs {cond_b}: "
                    f"coef={coef:+.6f}, z={z:+.3f}, p={p:.4f}{sig_stars(p)}"
                )
                rows.append(dict(
                    response=response_label, response_col=response_col,
                    roi=roi_label, contrast=f"{cond_a}_vs_{cond_b}",
                    coef=coef, z=z, p_value=p, n=sub["subject"].nunique(),
                ))
            except Exception as e:
                log.append(f"  {roi_label:12s}: FAILED – {e}")

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

COND_COLORS = {
    "sensory":          "#777777",
    "semantic_gen":     "#E1812C",
    "non_semantic_gen": "#3274A1",
    "no_gen":           "#A0A0A0",
}


def plot_mean_timecourse(df_long, metrics_df, save_path=None, cfg=CONFIG):
    """Plot mean slope timecourse per condition, with average fitted sinusoid overlaid.

    Recreates the style of thesis Fig 5.5A/B but with the fit curves on top.
    """
    if "condition" not in df_long.columns:
        return
    conditions = [cfg["cond_sensory"], cfg["cond_semantic_gen"],
                  cfg["cond_non_semantic_gen"], cfg["cond_no_gen"]]

    rois = sorted(df_long["roi"].unique()) if "roi" in df_long.columns else [None]
    fig, axes = plt.subplots(len(rois), 2, figsize=(13, 4.5 * len(rois)), squeeze=False)

    for r_idx, roi in enumerate(rois):
        if roi is None:
            sub_long = df_long
            sub_metrics = metrics_df
        else:
            sub_long = df_long[df_long["roi"] == roi]
            sub_metrics = metrics_df[metrics_df["roi"] == roi] if "roi" in metrics_df.columns else metrics_df

        # Mean timecourse per condition
        mean_tc = sub_long.groupby(["condition", "tr"])["slope"].mean().reset_index()
        sem_tc  = sub_long.groupby(["condition", "tr"])["slope"].sem().reset_index().rename(columns={"slope": "sem"})
        mean_tc = mean_tc.merge(sem_tc, on=["condition", "tr"])

        for col_idx, (window_name, mask_fn, t_label) in enumerate([
            ("Full window", lambda x: _get_window_mask(x, "full", cfg), "TR"),
            ("Non-sensory window", lambda x: _get_window_mask(x, "non_sensory", cfg), "TR"),
        ]):
            ax = axes[r_idx, col_idx]
            for cond in conditions:
                cdat = mean_tc[mean_tc["condition"] == cond].sort_values("tr")
                if cdat.empty:
                    continue
                trs = cdat["tr"].values
                mask = mask_fn(trs)
                color = COND_COLORS.get(cond, "k")
                # Plot mean ± SEM
                ax.plot(trs, cdat["slope"], color=color, alpha=0.3, lw=1)
                ax.plot(trs[mask], cdat["slope"].values[mask], color=color, lw=2, label=cond)
                ax.fill_between(trs[mask], (cdat["slope"]-cdat["sem"]).values[mask],
                                (cdat["slope"]+cdat["sem"]).values[mask],
                                color=color, alpha=0.2)

            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(t_label)
            ax.set_ylabel("SODA slope")
            title = f"{roi or 'all'} — {window_name}"
            ax.set_title(title)
            ax.legend(fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_condition_summary(metrics_df, save_path=None, cfg=CONFIG):
    """Boxplots of |amplitude| and R² across conditions, with subject dots."""
    conditions = [cfg["cond_no_gen"], cfg["cond_sensory"],
                  cfg["cond_semantic_gen"], cfg["cond_non_semantic_gen"]]

    panels = [
        ("full_amplitude_abs", "|Amplitude|  (full window)"),
        ("full_r_squared",     "R²  (full window)"),
        ("ns_amplitude_abs",   "|Amplitude|  (non-sensory)"),
        ("ns_r_squared",       "R²  (non-sensory)"),
    ]

    rois = sorted(metrics_df["roi"].unique()) if "roi" in metrics_df.columns else [None]
    fig, axes = plt.subplots(len(rois), 4, figsize=(18, 4.5 * len(rois)), squeeze=False)

    for r_idx, roi in enumerate(rois):
        sub = metrics_df if roi is None else metrics_df[metrics_df["roi"] == roi]

        for c_idx, (col, label) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            if col not in sub.columns:
                ax.set_visible(False)
                continue

            subj_means = sub.groupby(["subject", "condition"])[col].mean().reset_index()
            data, colors, labels = [], [], []
            for cond in conditions:
                vals = subj_means[subj_means["condition"] == cond][col].dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals)
                colors.append(COND_COLORS.get(cond, "k"))
                labels.append(cond)

            if not data:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(data, positions=range(len(data)),
                            widths=0.5, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            rng = np.random.default_rng(42)
            for i, (vals, color) in enumerate(zip(data, colors)):
                jitter = rng.normal(0, 0.06, size=len(vals))
                ax.scatter(np.full_like(vals, i) + jitter, vals,
                           color=color, edgecolor="k", linewidth=0.5,
                           s=22, zorder=5, alpha=0.8)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"{roi or 'all'}")
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_example_fits(df_long, metrics_df, n_per_cell=4, save_path=None, cfg=CONFIG):
    """Plot best and worst fits for each condition × window."""
    conditions = [cfg["cond_sensory"], cfg["cond_semantic_gen"],
                  cfg["cond_non_semantic_gen"], cfg["cond_no_gen"]]
    fig, axes = plt.subplots(len(conditions), 2, figsize=(11, 3 * len(conditions)),
                             squeeze=False)
    for r_idx, cond in enumerate(conditions):
        cdat = metrics_df[metrics_df["condition"] == cond]
        if cdat.empty:
            continue
        for c_idx, (window_name, r2_col) in enumerate([
            ("full", "full_r_squared"),
            ("non_sensory", "ns_r_squared"),
        ]):
            ax = axes[r_idx, c_idx]
            valid = cdat.dropna(subset=[r2_col])
            if valid.empty:
                ax.set_visible(False)
                continue
            best  = valid.nlargest(1, r2_col)
            worst = valid.nsmallest(1, r2_col)
            for i, row in pd.concat([best, worst]).iterrows():
                grp = df_long[
                    (df_long["subject"] == row["subject"])
                    & (df_long["trial"] == row["trial"])
                    & (df_long["condition"] == cond)
                ].sort_values("tr")
                if grp.empty:
                    continue
                trs = grp["tr"].values
                slopes = grp["slope"].values
                mask = _get_window_mask(trs, window_name, cfg)
                ax.plot(trs, slopes, "o-", color="grey", alpha=0.3, ms=3)
                ax.plot(trs[mask], slopes[mask], "ko", ms=5,
                        label=f"R²={row[r2_col]:.2f}")
                # Refit and overlay
                fit = fit_phase_constrained_sinusoid(trs[mask], slopes[mask])
                if fit["fit_success"]:
                    t_fine = np.linspace(trs[mask].min(), trs[mask].max(), 200)
                    y_fit = windowed_sinusoid(t_fine, fit["amplitude"], fit["duration"],
                                              fit["onset"], fit["offset"])
                    ax.plot(t_fine, y_fit, "-", lw=2, alpha=0.7)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.set_title(f"{cond} — {window_name}", fontsize=9)
            ax.set_xlabel("TR", fontsize=8)
            ax.set_ylabel("slope", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(metrics_df, results_dir, df_long=None, label=""):
    """Run LME + plotting on the trial-level fits DataFrame.

    Parameters
    ----------
    metrics_df : DataFrame from compute_trial_level_fits()
    results_dir : output directory
    df_long : optional, the original long-format slope data (for plots)
    label : header label for the results log
    """
    os.makedirs(results_dir, exist_ok=True)
    log = []
    csvs = {}

    if label:
        log.append("=" * 70)
        log.append(f"  {label}")
        log.append("=" * 70)

    # Data description
    log.append("\nData summary:")
    log.append(f"  N subjects: {metrics_df['subject'].nunique()}")
    if "roi" in metrics_df.columns:
        log.append(f"  ROIs:       {sorted(metrics_df['roi'].unique().tolist())}")
    log.append(f"  Conditions: {sorted(metrics_df['condition'].unique().tolist())}")
    log.append(f"  Total fits: {len(metrics_df)}")
    log.append("\n  Trials per condition:")
    for cond, n in metrics_df["condition"].value_counts().items():
        log.append(f"    {cond:25s}: {n}")
    log.append("\n  Fit success rates:")
    for col in ["full_fit_success", "ns_fit_success"]:
        if col in metrics_df.columns:
            rate = metrics_df[col].mean()
            log.append(f"    {col:25s}: {100*rate:.1f}%")

    # Main LME
    print("  Running condition contrasts...")
    csvs["lme_condition_contrasts"] = analyze_condition_effects(metrics_df, log)

    # Pairwise: semantic vs non-semantic
    print("  Running pairwise semantic vs non-semantic...")
    csvs["lme_semantic_vs_nonsemantic"] = analyze_pairwise_gen_contrast(
        metrics_df, log,
        cond_a=CONFIG["cond_semantic_gen"], cond_b=CONFIG["cond_non_semantic_gen"]
    )

    # Plots
    print("  Generating plots...")
    plot_condition_summary(metrics_df,
                           save_path=opj(results_dir, "condition_summary.png"))
    if df_long is not None:
        plot_mean_timecourse(df_long, metrics_df,
                             save_path=opj(results_dir, "mean_timecourse.png"))
        plot_example_fits(df_long, metrics_df,
                          save_path=opj(results_dir, "example_fits.png"))

    # Write outputs
    log_path = opj(results_dir, "results.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log))
    print(f"  Results log: {log_path}")

    for name, tbl in csvs.items():
        if tbl is not None and len(tbl) > 0:
            path = opj(results_dir, f"{name}.csv")
            tbl.to_csv(path, index=False)
            print(f"    {path}")

    metrics_path = opj(results_dir, "trial_level_fits.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"    {metrics_path}")

    return log


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # ---- 1. Load long-format data ----
    # df = pd.read_csv("your_soda_per_tr.csv")
    # df = rename_to_internal(df)
    #
    # Expected columns after renaming: subject, trial, condition, tr, slope
    # (+ optionally roi, block, day, is_reversal)

    # ---- 2. (If wide-format) reshape to long ----
    # df = wide_to_long(df_wide,
    #     condition_cols={
    #         "sensory":          "sensory_slope",
    #         "semantic_gen":     "semantic_gen_slope",
    #         "non_semantic_gen": "non_semantic_gen_slope",
    #         "no_gen":           "no_gen_slope",
    #     })

    # ---- 3. Fit sinusoids ----
    # metrics_df = compute_trial_level_fits(df)

    # ---- 4. Run statistics + plots ----
    # run_pipeline(metrics_df, "results/", df_long=df, label="Project 5 — Visual ROI")

    print("project5_sinusoid_analysis.py loaded. Edit CONFIG and call the functions.")
