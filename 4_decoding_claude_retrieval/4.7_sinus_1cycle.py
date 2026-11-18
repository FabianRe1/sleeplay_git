# -*- coding: utf-8 -*-
"""Windowed single-cycle sinusoidal fitting of SODA timecourses.

Following Wittkuhn & Schuck (2021, Nature Communications).

Usage
-----
    python 4.7_sinus_1cycle.py [ROI] [presleep|postsleep]
"""

import os
import sys

# Import utils from the original pipeline directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "4_decoding_claude"))
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt

from utils import script_output_dir


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

# Which SODA timecourses to fit.
#   "trunc" : position-truncated coefficients (elements 1..p, p >= 3)
#   "full"  : the original full 5-element coefficients
FIT_ON = "trunc"

# How each trial's coefficient timecourse is put on a common scale before
# fitting. A slope fitted over p elements carries a mechanical scale factor
# (SD amplification 1/sqrt(sum (i-mean)^2) = 0.707 / 0.447 / 0.316 for
# p = 3 / 4 / 5), so amplitudes are not comparable across retrieved positions
# unless the timecourses are normalised.
#   "null_sd" : divide by the permutation-null SD of that trial's arm, averaged
#               over its TRs. The null SD is computed ACROSS SEQUENCE ELEMENTS
#               within each TR, so it rescales the noise floor without touching
#               the across-TR shape the sinusoid is fitted to. Amplitude is then
#               in units of "multiples of the chance-level coefficient spread".
#   "l1"      : divide by sum |y(t)| over the trial. Safe (always positive) but
#               computed ACROSS TRs, so it partly divides out the very biphasic
#               swing being measured.
#   "sum"     : divide by sum y(t). NOT recommended: these are signed
#               coefficients, so the divisor is negative in ~45% of trials (which
#               flips the lobe order) and near zero in ~12% (which explodes the
#               scale). Guarded below.
#   "none"    : no normalisation.
NORMALISATION = "null_sd"
SUM_NORM_MIN_RATIO = 0.10   # for "sum": require |sum y| >= ratio * sum|y|

# Which scale each trial is divided by (only used with NORMALISATION = "null_sd").
#   "control_pooled" : the control (non-retrieved) arm's null SD, averaged over
#                      all of that subject's trials with the same trial type
#                      (object/scene) and the same window length p, applied to
#                      EVERY arm of the trial. This is the default because:
#                        - one scale for both arms: the retrieved arm's own null
#                          SD is 2-5% larger in ROIs with retrieval signal (the
#                          elevated target widens the spread), so "own" scaling
#                          shaves the effect under test off the retrieved arm;
#                        - pooled, not per trial: a single trial's null SD is a
#                          noisy denominator (down to 0.006; retrieved/control
#                          ratio up to 26x). Dividing the retrieved amplitude by
#                          it, while the control amplitude co-varies with its own
#                          scale, manufactured a spurious retrieved > control
#                          effect (vis presleep: +17.6%, t = 5.3, vs +3.9% raw).
#                      Pooling within p still removes the window-length scaling.
#   "own"            : each arm divided by its own per-trial null SD (biased
#                      against retrieved > control; kept for reproducibility).
SCALE_REFERENCE = "control_pooled"

# The contrast of interest is retrieved vs the non-retrieved sequence.
# Shuffle arms are only fitted if this is True.
FIT_SHUFFLE_ARMS = False

# Single-cycle sinusoid search grid (in TRs).
PERIOD_MIN = 4.0            # shortest cycle; 4 TRs still spans 5 samples
PERIOD_MAX = None           # None -> the data span
GRID_STEP_PERIOD = 0.25
GRID_STEP_ONSET = 0.25

# Where the cycle has to sit relative to the analysis window.
#   "peaks_inside" : both the positive and the negative peak must fall inside
#                    the window (peaks are at d + P/4 and d + 3P/4)
#   "cycle_inside" : the entire cycle [d, d+P] must fall inside the window
CYCLE_CONSTRAINT = "peaks_inside"

# Which information criterion picks `best_model`. Both are always reported.
#   "aicc" : small-sample corrected (recommended: n is only ~10 TRs)
#   "aic"  : plain AIC
AIC_CRITERION = "aicc"

# Which sinusoid model is fitted.
#   "grid"    : 4-parameter hard-truncated sinusoid, exhaustive grid (section 3b)
#   "tapered" : the original thesis SinAmp model (fit_windowed_sinusoid, section 1):
#               15% cosine taper, duration 3-12 TRs, 10 seeded curve_fit restarts,
#               >= 90% of the cycle inside the window. Deterministic.
MODEL = "grid"

# Output folder (under the script directory) and parallel workers.
OUT_DIR = "4.7_sinus_1cycle_trunc"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# Named configurations, selected with the 3rd command-line argument.
#   thesis : exactly the analysis reported in the thesis — slope over all 5
#            sequence elements for every trial (all retrieved positions), the
#            tapered SinAmp fit on the raw signed SODA timecourse, shuffle arms
#            included (5.2 section 9 uses them). Writes to 4.7_sinus_1cycle/,
#            which is what 5.1 reads.
#   trunc  : position-truncated slopes (elements 1..p, p >= 3), grid fit,
#            pooled control-scale normalisation, retrieved vs control only.
#   trunc_tapered : as trunc, but with the thesis tapered SinAmp model.
PRESETS = {
    "thesis": dict(FIT_ON="full", NORMALISATION="none", MODEL="tapered",
                   FIT_SHUFFLE_ARMS=True, OUT_DIR="4.7_sinus_1cycle"),
    "trunc":  dict(FIT_ON="trunc", NORMALISATION="null_sd", MODEL="grid",
                   SCALE_REFERENCE="control_pooled", FIT_SHUFFLE_ARMS=False,
                   OUT_DIR="4.7_sinus_1cycle_trunc"),
    # truncated window with the THESIS SinAmp model, so that full vs truncated
    # windows can be compared with the same amplitude model (used by 5.4)
    "trunc_tapered": dict(FIT_ON="trunc", NORMALISATION="null_sd", MODEL="tapered",
                          SCALE_REFERENCE="control_pooled", FIT_SHUFFLE_ARMS=False,
                          OUT_DIR="4.7_sinus_1cycle_trunc_tapered"),
}
DEFAULT_PRESET = "thesis"


# =====================================================================
# 1. WINDOWED SINGLE-CYCLE SINUSOIDAL MODEL
# =====================================================================

def windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
    """Single-cycle sinusoidal model with smooth on/off ramps.

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


def _windowed_for_curvefit(t, amplitude, duration, onset, offset):
    return windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15)


def _full_cycle_visible(onset, duration, t_min, t_max, min_cycle_frac=0.90):
    """Check that at least *min_cycle_frac* of the cycle falls within the data range.

    This ensures both the positive and negative lobes of the sinusoid are
    captured by the data, preventing half-cycle fits that chase a single
    strong deflection.
    """
    cycle_start = onset
    cycle_end = onset + duration
    overlap_start = max(cycle_start, t_min)
    overlap_end = min(cycle_end, t_max)
    overlap = max(0.0, overlap_end - overlap_start)
    return (overlap / duration) >= min_cycle_frac


def fit_windowed_sinusoid(
    trs: np.ndarray,
    soda_values: np.ndarray,
    duration_min: float = 3.0,
    duration_max: float = 12.0,
    n_restarts: int = 10,
    min_cycle_frac: float = 0.90,
) -> dict:
    """Fit the windowed single-cycle sinusoidal model to a SODA timecourse.

    Parameters
    ----------
    min_cycle_frac : float
        Minimum fraction of the full cycle that must fall within the data
        range.  Set to 0.90 by default so that both the positive and
        negative phases of the sinusoid are required to be visible.
    """
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(soda_values)

    if valid.sum() < 4:
        return _empty_windowed_result(len(trs))

    t, y = trs[valid], soda_values[valid]
    t_min, t_max = t.min(), t.max()
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = y_range / 2 if y_range > 0 else 0.01

    # Tighten onset: cycle must start no earlier than data start and must
    # leave room for at least duration_min within the data range.
    onset_lower = t_min
    onset_upper = max(t_min, t_max - duration_min)

    lower = [-np.inf, duration_min, onset_lower, -np.inf]
    upper = [np.inf, duration_max, onset_upper, np.inf]

    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, (t_max - t_min) * 0.8, t_min, y_mean]
        elif restart == 1:
            p0 = [-a0, (t_max - t_min) * 0.8, t_min, y_mean]
        else:
            p0 = [
                rng.uniform(-a0 * 2, a0 * 2),
                rng.uniform(duration_min, min(duration_max, t_max - t_min + 2)),
                rng.uniform(onset_lower, onset_lower + (onset_upper - onset_lower) * 0.5),
                y_mean + rng.normal(0, a0 * 0.1),
            ]
        # Clip p0 to bounds
        p0 = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, lower, upper)]
        try:
            popt, _ = curve_fit(_windowed_for_curvefit, t, y, p0=p0, bounds=(lower, upper), maxfev=5000)
            # Reject fits where full cycle is not visible in the data
            if not _full_cycle_visible(popt[2], popt[1], t_min, t_max, min_cycle_frac):
                continue
            residual = np.sum((y - _windowed_for_curvefit(t, *popt)) ** 2)
            if residual < best_residual:
                best_residual = residual
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return _empty_windowed_result(len(trs))

    amplitude, duration, onset, offset = best_result
    y_hat_valid = _windowed_for_curvefit(t, *best_result)
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


def _empty_windowed_result(n_trs: int) -> dict:
    return {
        "amplitude": np.nan, "duration": np.nan,
        "onset": np.nan, "offset": np.nan,
        "frequency": np.nan, "r_squared": np.nan,
        "fit_success": False, "y_hat": np.full(n_trs, np.nan),
    }


# =====================================================================
# 3b. TRUNCATED SINGLE-CYCLE SINUSOID — 4 parameters, exhaustive grid
# =====================================================================
#
#   y(t) = c + A * sin(2*pi*f*(t - d))   for  d <= t <= d + 1/f
#   y(t) = c                             otherwise
#
# Free parameters: signed amplitude A, frequency f (period P = 1/f), onset
# delay d, offset c. The sign of A carries the direction: A > 0 means the
# positive lobe comes first, A < 0 the negative lobe.
#
# FITTING. For a FIXED (P, d) the model is LINEAR in A and c, so the exact
# least-squares solution is available in closed form. The fit therefore grids
# (P, d) and solves (A, c) analytically at every grid point, returning the
# global optimum on the grid. This replaces random restarts: the hard
# truncation makes the objective non-smooth in P and d (the support boundary
# moves discontinuously across samples), which is precisely the situation in
# which gradient-based fitting lands in local minima. It is also faster —
# every trial's whole grid is two matrix-vector products, because all trials
# share the same TR sampling.
#
# Two nested alternatives are fitted on the same grid and scored by AIC:
#   - offset-free sinusoid (c = 0, 3 parameters)
#   - linear baseline      (slope + intercept, 2 parameters)


def truncated_sinusoid(t, amplitude, frequency, onset, offset):
    """Single cycle of a sinusoid, hard-truncated outside [onset, onset+1/f]."""
    t = np.asarray(t, dtype=float)
    period = 1.0 / frequency
    y = np.full(t.shape, float(offset))
    active = (t >= onset) & (t <= onset + period)
    y[active] = offset + amplitude * np.sin(2 * np.pi * frequency * (t[active] - onset))
    return y


def _cycle_grid(t_min, t_max, period_min, period_max, step_period, step_onset, constraint):
    """Enumerate (period, onset) pairs satisfying the placement constraint."""
    out = []
    p = period_min
    while p <= period_max + 1e-9:
        if constraint == "peaks_inside":
            # positive peak at d + P/4, negative peak at d + 3P/4
            d_lo, d_hi = t_min - p / 4.0, t_max - 3.0 * p / 4.0
        elif constraint == "cycle_inside":
            d_lo, d_hi = t_min, t_max - p
        else:
            raise ValueError(f"Unknown CYCLE_CONSTRAINT: {constraint}")
        if d_hi >= d_lo - 1e-9:
            d = d_lo
            while d <= d_hi + 1e-9:
                out.append((p, d))
                d += step_onset
        p += step_period
    return out


_BASIS_CACHE: dict = {}


def _sinusoid_basis(t, period_min, period_max, step_period, step_onset, constraint):
    """(grid, S) with S[k] the unit-amplitude cycle for grid point k on *t*."""
    key = (tuple(np.round(t, 6)), period_min, period_max, step_period, step_onset, constraint)
    if key not in _BASIS_CACHE:
        grid = _cycle_grid(t.min(), t.max(), period_min, period_max,
                           step_period, step_onset, constraint)
        S = np.zeros((len(grid), len(t)))
        for k, (p, d) in enumerate(grid):
            active = (t >= d) & (t <= d + p)
            S[k, active] = np.sin(2 * np.pi * (t[active] - d) / p)
        keep = S.any(axis=1)
        _BASIS_CACHE[key] = (np.asarray(grid)[keep], S[keep])
    return _BASIS_CACHE[key]


def _aic(sse: float, n: int, n_params: int, corrected: bool = False) -> float:
    """AIC (or AICc) for a least-squares fit; n_params excludes the noise variance.

    With only ~10 TRs per trial, n/k is about 2 and plain AIC badly
    under-penalises the 4-parameter sinusoid: on pure noise it selects a
    sinusoid over the linear baseline about 77% of the time. The small-sample
    correction (standard whenever n/k < 40) is therefore reported alongside and
    is what AIC_CRITERION selects on by default.
    """
    if not np.isfinite(sse) or n <= 0:
        return np.nan
    sse = max(float(sse), 1e-300)
    k = n_params + 1                      # + noise variance
    aic = n * np.log(sse / n) + 2 * k
    if not corrected:
        return aic
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_truncated_sinusoid(
    trs, values,
    period_min: float = PERIOD_MIN,
    period_max: float | None = PERIOD_MAX,
    step_period: float = GRID_STEP_PERIOD,
    step_onset: float = GRID_STEP_ONSET,
    constraint: str = CYCLE_CONSTRAINT,
    min_points: int = 5,
) -> dict:
    """Fit the 4-parameter truncated sinusoid plus its two nested alternatives."""
    t = np.asarray(trs, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = ~np.isnan(y)
    t, y = t[ok], y[ok]
    n = len(y)
    if n < min_points:
        return _empty_truncated_result()

    p_max = (t.max() - t.min()) if period_max is None else period_max
    if p_max < period_min:
        return _empty_truncated_result()
    grid, S = _sinusoid_basis(t, period_min, p_max, step_period, step_onset, constraint)
    if len(grid) == 0:
        return _empty_truncated_result()

    y_mean = y.mean()
    yc = y - y_mean
    syy = float(yc @ yc)

    # --- model 1: amplitude + offset (4 parameters) ---
    Sc = S - S.mean(axis=1, keepdims=True)
    sss = np.einsum("ij,ij->i", Sc, Sc)
    ssy = Sc @ yc
    good = sss > 1e-12
    amp = np.where(good, ssy / np.where(good, sss, 1.0), 0.0)
    sse = np.where(good, syy - amp * ssy, np.inf)
    k1 = int(np.argmin(sse))
    period, onset = grid[k1]
    amplitude = float(amp[k1])
    offset = float(y_mean - amplitude * S[k1].mean())
    sse1 = float(max(sse[k1], 0.0))

    # --- model 2: amplitude only, no offset (3 parameters) ---
    ss2 = np.einsum("ij,ij->i", S, S)
    sy2 = S @ y
    good2 = ss2 > 1e-12
    amp2 = np.where(good2, sy2 / np.where(good2, ss2, 1.0), 0.0)
    sse_2 = np.where(good2, float(y @ y) - amp2 * sy2, np.inf)
    k2 = int(np.argmin(sse_2))
    sse2 = float(max(sse_2[k2], 0.0))

    # --- model 3: linear baseline (2 parameters) ---
    slope, intercept = np.polyfit(t, y, 1)
    sse3 = float(np.sum((y - (slope * t + intercept)) ** 2))

    aics = {"sinusoid": _aic(sse1, n, 4), "sinusoid_no_offset": _aic(sse2, n, 3),
            "linear": _aic(sse3, n, 2)}
    aiccs = {"sinusoid": _aic(sse1, n, 4, True), "sinusoid_no_offset": _aic(sse2, n, 3, True),
             "linear": _aic(sse3, n, 2, True)}
    crit = aiccs if AIC_CRITERION == "aicc" else aics
    best = min(crit, key=lambda m: crit[m])
    r2 = (lambda sse_: 1 - sse_ / syy if syy > 0 else np.nan)

    return {
        # 4-parameter model (primary)
        "amplitude": amplitude,
        "amplitude_abs": abs(amplitude),
        "direction": float(np.sign(amplitude)),
        "period": float(period),
        "frequency": 1.0 / float(period),
        "onset": float(onset),
        "offset": offset,
        "r_squared": r2(sse1),
        "sse": sse1,
        "aic": aics["sinusoid"],
        "aicc": aiccs["sinusoid"],
        # offset-free variant
        "no_offset_amplitude": float(amp2[k2]),
        "no_offset_period": float(grid[k2][0]),
        "no_offset_onset": float(grid[k2][1]),
        "no_offset_r_squared": r2(sse2),
        "no_offset_aic": aics["sinusoid_no_offset"],
        "no_offset_aicc": aiccs["sinusoid_no_offset"],
        # linear baseline
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_r_squared": r2(sse3),
        "linear_aic": aics["linear"],
        "linear_aicc": aiccs["linear"],
        # comparison
        "best_model": best,
        "delta_aic_vs_linear": float(crit["linear"] - crit["sinusoid"]),
        "delta_aic_vs_no_offset": float(crit["sinusoid_no_offset"] - crit["sinusoid"]),
        "n_points": n,
        "fit_success": True,
    }


def _empty_truncated_result() -> dict:
    keys = ["amplitude", "amplitude_abs", "direction", "period", "frequency", "onset",
            "offset", "r_squared", "sse", "aic", "aicc", "no_offset_amplitude",
            "no_offset_period", "no_offset_onset", "no_offset_r_squared", "no_offset_aic",
            "no_offset_aicc", "linear_slope", "linear_intercept", "linear_r_squared",
            "linear_aic", "linear_aicc", "delta_aic_vs_linear",
            "delta_aic_vs_no_offset", "n_points"]
    out = {k: np.nan for k in keys}
    out["best_model"] = None
    out["fit_success"] = False
    return out


# =====================================================================
# 3c. NORMALISATION OF THE COEFFICIENT TIMECOURSE
# =====================================================================

def normalisation_scale(values: np.ndarray, null_sd: np.ndarray | None, method: str) -> float:
    """Return the scalar a trial's coefficient timecourse is divided by.

    One scalar per trial × arm, so the SHAPE of the timecourse is untouched and
    only its units change — the sinusoid is fitted to that shape.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan
    if method == "none":
        return 1.0
    if method == "null_sd":
        if null_sd is None:
            return np.nan
        sd = np.asarray(null_sd, dtype=float)
        sd = sd[~np.isnan(sd)]
        return float(sd.mean()) if len(sd) and sd.mean() > 0 else np.nan
    if method == "l1":
        s = float(np.abs(v).sum())
        return s if s > 0 else np.nan
    if method == "sum":
        s = float(v.sum())
        l1 = float(np.abs(v).sum())
        # guard: a near-zero or negative divisor would explode the scale or
        # invert the lobe order
        if l1 <= 0 or abs(s) < SUM_NORM_MIN_RATIO * l1:
            return np.nan
        return s
    raise ValueError(f"Unknown NORMALISATION: {method}")


# =====================================================================
# 4. TRIAL-LEVEL COMPUTATION
# =====================================================================

def _soda_column_map(fit_on: str) -> dict:
    """Arm label -> (coefficient column, permutation-null SD column)."""
    if fit_on == "trunc":
        return {
            "retrieved":          ("retrieved_soda_trunc",                    "retrieved_soda_trunc_null_sd"),
            "control":            ("control_soda_trunc",                      "control_soda_trunc_null_sd"),
            "shuf_retrieved":     ("retrieved_soda_trunc_shuffled",           "retrieved_soda_trunc_null_sd"),
            "shuf_control":       ("control_soda_trunc_shuffled",             "control_soda_trunc_null_sd"),
            "shuf_sel_retrieved": ("retrieved_soda_trunc_shuffled_selected",  "retrieved_soda_trunc_null_sd"),
            "shuf_sel_control":   ("control_soda_trunc_shuffled_selected",    "control_soda_trunc_null_sd"),
        }
    return {
        "retrieved":          ("retrieved_soda", None),
        "control":            ("control_soda", None),
        "shuf_retrieved":     ("retrieved_soda_shuffled", None),
        "shuf_control":       ("control_soda_shuffled", None),
        "shuf_sel_retrieved": ("retrieved_soda_shuffled_selected", None),
        "shuf_sel_control":   ("control_soda_shuffled_selected", None),
    }


def _fit_trial_arms(trial_info: dict, trs: np.ndarray, arms: dict, model: str) -> dict:
    """Fit every arm of one trial; returns trial_info extended with the fit columns."""
    out = dict(trial_info)
    for cond_label, (vals, scale) in arms.items():
        out[f"{cond_label}_scale"] = scale
        if not np.isfinite(scale):
            fit = _empty_truncated_result() if model == "grid" else _empty_windowed_result(len(trs))
        elif model == "grid":
            fit = fit_truncated_sinusoid(trs, vals / scale)
        elif model == "tapered":
            fit = fit_windowed_sinusoid(trs, vals / scale)
        else:
            raise ValueError(f"Unknown MODEL: {model}")
        fit = {k: v for k, v in fit.items() if k != "y_hat"}          # array, not a column
        amp = fit.get("amplitude", np.nan)
        fit["amplitude_abs"] = abs(amp) if amp is not None and np.isfinite(amp) else np.nan
        for k, v in fit.items():
            out[f"{cond_label}_w_{k}"] = v
    return out


def compute_trial_level_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the truncated single-cycle sinusoid to each trial's SODA timecourse.

    With SCALE_REFERENCE = "control", every arm of a trial is divided by the
    same scalar (the control arm's scale), so amplitudes of different arms
    within a trial are directly comparable. With "own", a shuffled arm uses the
    scale of the arm it permutes.
    """
    soda_cols = _soda_column_map(FIT_ON)
    if not FIT_SHUFFLE_ARMS:
        soda_cols = {k: v for k, v in soda_cols.items() if not k.startswith("shuf")}
    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"] if c in df.columns]
    results = []
    n_skipped_scale = 0

    # --- pooled control scale per subject x trial type x window length ---
    def _pool_key(tdf):
        ne = tdf["n_elements_trunc"].dropna() if "n_elements_trunc" in tdf.columns else pd.Series(dtype=float)
        return (tdf["subject"].iloc[0], tdf["object_scene_retrieval"].iloc[0],
                ne.iloc[0] if len(ne) else np.nan)

    pooled_scale = {}
    ctrl_sd_col = soda_cols.get("control", (None, None))[1]
    if SCALE_REFERENCE == "control_pooled" and NORMALISATION == "null_sd" and ctrl_sd_col in df.columns:
        per_trial = (df.groupby(group_cols + ["n_elements_trunc"], dropna=False)[ctrl_sd_col]
                       .mean().reset_index())
        per_trial = per_trial[per_trial[ctrl_sd_col] > 0]
        pooled = per_trial.groupby(["subject", "object_scene_retrieval", "n_elements_trunc"])[ctrl_sd_col].mean()
        pooled_scale = {k: float(v) for k, v in pooled.items()}

    # --- collect one payload per trial (scales need the whole dataframe, so
    #     they are computed here; the fits themselves run in parallel below) ---
    payloads = []
    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))
        trial_df = trial_df.sort_values("tr")
        trs = trial_df["tr"].values
        if FIT_ON == "trunc" and ("n_elements_trunc" not in trial_df.columns
                                  or trial_df["n_elements_trunc"].isna().all()):
            continue   # retrieved position < 3: no truncated timecourse by design

        # common scale for all arms, pooled over the subject's trials with the
        # same trial type and window length (see SCALE_REFERENCE)
        common_scale = None
        if SCALE_REFERENCE == "control_pooled" and NORMALISATION == "null_sd":
            common_scale = pooled_scale.get(_pool_key(trial_df), np.nan)
        elif SCALE_REFERENCE not in ("own", "control_pooled"):
            raise ValueError(f"Unknown SCALE_REFERENCE: {SCALE_REFERENCE}")

        arms = {}
        for cond_label, (col_name, sd_col) in soda_cols.items():
            if col_name not in trial_df.columns:
                continue
            vals = trial_df[col_name].values

            if common_scale is not None:
                scale = common_scale
            else:
                null_sd = trial_df[sd_col].values if (sd_col and sd_col in trial_df.columns) else None
                scale = normalisation_scale(vals, null_sd, NORMALISATION)
            if np.isnan(scale):
                n_skipped_scale += 1
            arms[cond_label] = (np.asarray(vals, dtype=float), scale)

        if "n_elements_trunc" in trial_df.columns:
            ne = trial_df["n_elements_trunc"].dropna()
            trial_info["n_elements"] = ne.iloc[0] if len(ne) > 0 else np.nan

        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

        payloads.append((trial_info, np.asarray(trs, dtype=float), arms))

    if N_JOBS > 1 and len(payloads) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=N_JOBS, batch_size=32)(
            delayed(_fit_trial_arms)(info, trs, arms, MODEL) for info, trs, arms in payloads)
    else:
        results = [_fit_trial_arms(info, trs, arms, MODEL) for info, trs, arms in payloads]

    out = pd.DataFrame(results)
    if FIT_ON == "trunc" and "n_elements" in out.columns:
        # trials with the target at position 1 or 2 have no truncated timecourse
        out = out[out["n_elements"].notna()].reset_index(drop=True)
    print(f"[fits] {len(out)} trials fitted "
          f"(MODEL={MODEL}, FIT_ON={FIT_ON}, NORMALISATION={NORMALISATION}, SCALE_REFERENCE={SCALE_REFERENCE}, N_JOBS={N_JOBS}, "
          f"period {PERIOD_MIN}-{'span' if PERIOD_MAX is None else PERIOD_MAX} TRs, "
          f"constraint={CYCLE_CONSTRAINT})")
    if n_skipped_scale:
        print(f"[fits] {n_skipped_scale} arm-timecourses skipped: normalisation scale undefined")
    if "retrieved_w_best_model" in out.columns:
        print("[fits] AIC winner for the retrieved arm: "
              + str(out["retrieved_w_best_model"].value_counts(dropna=False).to_dict()))
    return out


# =====================================================================
# 5. PLOTTING
# =====================================================================

def plot_example_fits(df, metrics_df, soda_col="retrieved_soda", n_examples=8, save_path=None):
    """Plot example windowed sinusoidal fits sorted by R²."""
    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"] if c in metrics_df.columns]

    mask = metrics_df["retrieved_w_fit_success"].fillna(False)
    valid = metrics_df[mask].copy()
    if valid.empty:
        print("No trials with successful windowed fits.")
        return

    # Show best and worst fits
    best = valid.nlargest(n_examples // 2, "retrieved_w_r_squared")
    worst = valid.nsmallest(n_examples // 2, "retrieved_w_r_squared")
    examples = pd.concat([best, worst])

    n_plots = len(examples)
    if n_plots == 0:
        return

    ncols = min(4, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for i, (_, row) in enumerate(examples.iterrows()):
        ax = axes[i // ncols, i % ncols]

        mask_trial = pd.Series(True, index=df.index)
        for col in group_cols:
            if col in df.columns:
                mask_trial &= df[col] == row[col]
        trial_data = df[mask_trial].sort_values("tr")

        if trial_data.empty:
            ax.set_visible(False)
            continue

        trs = trial_data["tr"].values
        # plot the series that was actually fitted (column and normalisation
        # depend on the preset), with the fitted parameters of the model used
        fitted_col = _soda_column_map(FIT_ON)["retrieved"][0]
        scale = row.get("retrieved_scale", 1.0)
        scale = scale if (scale is not None and np.isfinite(scale) and scale != 0) else 1.0
        soda_vals = trial_data[fitted_col].values / scale
        t_fine = np.linspace(trs.min(), trs.max(), 200)

        ax.plot(trs, soda_vals, "ko", ms=5, zorder=5, label="Data")

        if row["retrieved_w_fit_success"]:
            if MODEL == "tapered":
                y_w = windowed_sinusoid(t_fine, row["retrieved_w_amplitude"], row["retrieved_w_duration"],
                                        row["retrieved_w_onset"], row["retrieved_w_offset"])
            else:
                y_w = truncated_sinusoid(t_fine, row["retrieved_w_amplitude"], row["retrieved_w_frequency"],
                                         row["retrieved_w_onset"], row["retrieved_w_offset"])
            ax.plot(t_fine, y_w, "r-", lw=2, alpha=0.8, label=f"{MODEL} fit")

        ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.set_title(f"R²={row['retrieved_w_r_squared']:.2f}", fontsize=8)
        ax.set_xlabel("TR", fontsize=8)
        ax.set_ylabel("SODA", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)

    for j in range(n_plots, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Windowed single-cycle sinusoidal fits (best & worst R²)", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_amplitude_summary(metrics_df, save_path=None):
    """Boxplot of windowed sinusoidal |amplitude| across all 6 conditions."""
    conditions = [
        "retrieved", "control",
        "shuf_retrieved", "shuf_control",
        "shuf_sel_retrieved", "shuf_sel_control",
    ]
    cond_labels = [
        "Retrieved", "Control",
        "Shuf Ret", "Shuf Ctrl",
        "Shuf Sel Ret", "Shuf Sel Ctrl",
    ]
    colors = ["#3274A1", "#E1812C", "#8CB8D0", "#F0C08A", "#A0D0A0", "#D0A0D0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: R² distribution for retrieved
    ax = axes[0]
    r2_col = "retrieved_w_r_squared"
    if r2_col in metrics_df.columns:
        vals = metrics_df[r2_col].dropna()
        ax.hist(vals, bins=40, color="#3274A1", alpha=0.7, edgecolor="white")
        ax.set_xlabel("R² (windowed sinusoid)")
        ax.set_ylabel("Count")
        ax.set_title(f"Windowed fit quality (retrieved)\nmedian R²={vals.median():.3f}")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Amplitude boxplot across conditions
    ax = axes[1]
    amp_cols = [f"{c}_w_amplitude" for c in conditions]
    available = [c for c in amp_cols if c in metrics_df.columns]
    if len(available) >= 2 and "subject" in metrics_df.columns:
        subj = metrics_df.groupby("subject")[available].apply(lambda x: x.abs().mean()).dropna()
        if len(subj) >= 3:
            plot_data = [subj[c].values for c in available]
            plot_labels = [cond_labels[amp_cols.index(c)] for c in available]
            plot_colors = [colors[amp_cols.index(c)] for c in available]

            bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            rng = np.random.default_rng(42)
            for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
                jitter = rng.normal(0, 0.06, size=len(vals))
                ax.scatter(np.full_like(vals, i) + jitter, vals,
                           color=color, edgecolor="k", linewidth=0.5, s=25, zorder=5, alpha=0.8)

            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("|Amplitude| (windowed model)")
            ax.set_title("Windowed amplitude across conditions")

            ret_col, ctrl_col = "retrieved_w_amplitude", "control_w_amplitude"
            if ret_col in available and ctrl_col in available:
                t_stat, p_val = ttest_rel(subj[ret_col].values, subj[ctrl_col].values)
                sig_str = f"p={p_val:.1e}" if p_val < 0.001 else f"p={p_val:.3f}"
                print(f"  Ret vs Ctrl |amplitude|: t({len(subj[ret_col])-1})={t_stat:.3f}, {sig_str}")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_windowed_parameters(metrics_df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax_idx, (param, xlabel, title) in enumerate([
        ("duration", "Duration λ (TRs)", "Cycle duration"),
        ("onset", "Onset d (TRs)", "Retrieval onset"),
    ]):
        ax = axes[ax_idx]
        for cond, color, label in [("retrieved", "#3274A1", "Retrieved"), ("control", "#E1812C", "Control")]:
            col = f"{cond}_w_{param}"
            if col in metrics_df.columns:
                ax.hist(metrics_df[col].dropna(), bins=30, color=color, alpha=0.5, edgecolor="white", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    d_col, o_col = "retrieved_w_duration", "retrieved_w_onset"
    if d_col in metrics_df.columns and o_col in metrics_df.columns:
        v = metrics_df.dropna(subset=[d_col, o_col])
        scatter = ax.scatter(v[o_col], v[d_col],
                             c=v.get("behav_correct", pd.Series(np.nan)),
                             cmap="RdBu", s=10, alpha=0.4, edgecolors="none")
        ax.set_xlabel("Onset d (TRs)")
        ax.set_ylabel("Duration λ (TRs)")
        ax.set_title("Onset vs duration (retrieved)")
        if "behav_correct" in v.columns:
            plt.colorbar(scatter, ax=ax, label="Accuracy")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 6. ENTRY POINT
# =====================================================================

def main():
    roi_filter = sys.argv[1] if len(sys.argv) > 1 else None
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"
    preset = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PRESET
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")
    globals().update(PRESETS[preset])
    print(f"[preset] {preset}: " + ", ".join(f"{k}={v}" for k, v in PRESETS[preset].items()))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{pre_post}", "SODA_results_csv")

    csv_files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
    if roi_filter:
        # read only the requested ROI — loading all twelve ~60 MB files first
        # and filtering afterwards is what made single-ROI runs slow
        csv_files = [f for f in csv_files
                     if os.path.basename(f) == f"{roi_filter}_soda_results_per_combination.csv"]
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    all_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if "ROI" not in df.columns:
            roi_name = os.path.basename(fp).replace("_soda_results_per_combination.csv", "")
            df["ROI"] = roi_name
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    numeric_cols = [
        "tr", "retrieved_soda", "control_soda", "behav_correct",
        "retrieved_soda_shuffled", "control_soda_shuffled",
        "retrieved_soda_shuffled_selected", "control_soda_shuffled_selected",
    ]
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    if roi_filter:
        master_df = master_df[master_df["ROI"] == roi_filter]
        if master_df.empty:
            raise ValueError(f"ROI '{roi_filter}' not found.")

    rois = sorted(master_df["ROI"].unique())
    print(f"Found {len(rois)} ROIs: {rois}")

    all_roi_summaries = []

    for roi in rois:
        print(f"\n{'='*50}\nROI: {roi}\n{'='*50}")

        df = master_df[master_df["ROI"] == roi].copy()
        out_dir = opj(script_dir, OUT_DIR, roi)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Fitting windowed single-cycle sinusoid per trial...")

        metrics_df = compute_trial_level_fits(df)
        print(f"Computed fits for {len(metrics_df)} trials")

        metrics_df.to_csv(opj(out_dir, f"{roi}_windowed_fit_results_{pre_post}.csv"), index=False)

        for cond in ["retrieved", "control"]:
            r2_col = f"{cond}_w_r_squared"
            if r2_col in metrics_df.columns:
                vals = metrics_df[r2_col].dropna()
                print(f"  {cond}: median R²={vals.median():.3f}, mean R²={vals.mean():.3f}")

        print("Generating plots...")

        plot_example_fits(df, metrics_df, save_path=opj(out_dir, f"{roi}_example_fits_{pre_post}.png"))
        plot_amplitude_summary(metrics_df, save_path=opj(out_dir, f"{roi}_amplitude_summary_{pre_post}.png"))
        plot_windowed_parameters(metrics_df, save_path=opj(out_dir, f"{roi}_windowed_parameters_{pre_post}.png"))

        roi_summary = {"ROI": roi, "n_subjects": df["subject"].nunique(), "n_trials": len(metrics_df),
                       "preset": preset, "model": MODEL, "fit_on": FIT_ON,
                       "normalisation": NORMALISATION}

        # Primary contrast: retrieved vs the non-retrieved (control) sequence,
        # on |amplitude| (absolute, so trials with opposite lobe order cannot
        # cancel) and on fit quality. Subject-level means, paired t-test.
        contrasts = [("amplitude_abs", "ret_vs_ctrl"), ("r_squared", "ret_vs_ctrl_r2")]
        if FIT_SHUFFLE_ARMS:
            contrasts.append(("amplitude_abs", "ret_vs_shuf_sel"))
        for metric, label in contrasts:
            a = "retrieved_w_" + metric
            b = ("shuf_sel_retrieved_w_" if label.endswith("shuf_sel") else "control_w_") + metric
            if a in metrics_df.columns and b in metrics_df.columns:
                subj = metrics_df.groupby("subject")[[a, b]].mean().dropna()
                if len(subj) >= 3:
                    diff = subj[a] - subj[b]
                    t, p = ttest_rel(subj[a], subj[b])
                    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
                    print(f"  {label}: {subj[a].mean():.4f} vs {subj[b].mean():.4f}  "
                          f"t({len(subj)-1})={t:.3f}, p={p:.4f}, dz={dz:+.3f}")
                    roi_summary[f"mean_{label}_ret"] = subj[a].mean()
                    roi_summary[f"mean_{label}_ctrl"] = subj[b].mean()
                    roi_summary[f"t_{label}"] = t
                    roi_summary[f"p_{label}"] = p
                    roi_summary[f"dz_{label}"] = dz

        all_roi_summaries.append(roi_summary)
        print(f"\nDone with ROI {roi}.")

    summary_dir = opj(script_dir, OUT_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = opj(summary_dir, f"windowed_fit_summary_{pre_post}.csv")
    summary_df = pd.DataFrame(all_roi_summaries)
    # Merge with existing summary to avoid overwriting other ROIs
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["ROI"].isin(summary_df["ROI"])]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    # FDR (Benjamini-Hochberg) across the ROIs present in the summary. Only a
    # correct family once all ROIs of this session have been run.
    from statsmodels.stats.multitest import multipletests
    for col in [c for c in summary_df.columns if c.startswith("p_")]:
        ok = summary_df[col].notna()
        summary_df[col.replace("p_", "q_", 1)] = np.nan
        if ok.sum() > 0:
            summary_df.loc[ok, col.replace("p_", "q_", 1)] = multipletests(summary_df.loc[ok, col], method="fdr_bh")[1]
    summary_df["n_rois_in_fdr_family"] = int(summary_df["ROI"].nunique())
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to {summary_path}.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
