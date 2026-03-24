#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.8 Enhanced SODA Analysis: Template Matching & Bayesian Model Comparison

Extends the existing SODA pipeline (4.6/4.7) with the best-performing
metrics identified in the primer simulations:

  1. Template matching with optimal lag (highest d' in simulations)
  2. Bayesian model comparison: replay vs null (BIC-based Bayes factor)
  3. Windowed 1-cycle sinusoidal fit (from 4.7, retained for comparison)
  4. Simple metrics: peak-to-trough, variance, abs_mean (benchmarks)

For each trial × condition (retrieved, control, shuffled), all metrics
are computed and compared at the group level.

Usage:
    python 4_8_enhanced_soda.py [ROI] [presleep|postsleep]

Reads from:
    4.6_SODA_based_on_singlesubj_4.4_{presleep|postsleep}/SODA_results_csv/

Outputs:
    - CSV with trial-level metrics for all conditions
    - Summary plots comparing metrics across conditions
    - Group-level statistics (paired t-tests, effect sizes)
"""

import os
import sys
import glob
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, ttest_1samp, pearsonr, wilcoxon
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


def create_script_named_folder(folder_name=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    script_name_without_ext = os.path.splitext(script_name)[0]
    if folder_name is None:
        new_dir_path = os.path.join(script_dir, script_name_without_ext)
    else:
        new_dir_path = os.path.join(script_dir, script_name_without_ext, folder_name)
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    return new_dir_path


# =====================================================================
# 1. SIMPLE METRICS
# =====================================================================

def metric_mean_slope(slopes):
    v = slopes[~np.isnan(slopes)]
    return np.mean(v) if len(v) >= 1 else np.nan

def metric_abs_mean(slopes):
    v = slopes[~np.isnan(slopes)]
    return np.mean(np.abs(v)) if len(v) >= 1 else np.nan

def metric_variance(slopes):
    v = slopes[~np.isnan(slopes)]
    return np.var(v) if len(v) >= 2 else np.nan

def metric_peak_to_trough(slopes):
    v = slopes[~np.isnan(slopes)]
    return np.ptp(v) if len(v) >= 2 else np.nan


# =====================================================================
# 2. WINDOWED 1-CYCLE SINUSOIDAL FIT
# =====================================================================

def _windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15):
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
    return _windowed_sinusoid(t, amplitude, duration, onset, offset, taper_frac=0.15)


def fit_windowed_sinusoid(trs, soda_values, n_restarts=8):
    trs = np.asarray(trs, dtype=float)
    soda_values = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(soda_values)
    if valid.sum() < 4:
        return {'amplitude': np.nan, 'duration': np.nan, 'onset': np.nan,
                'offset': np.nan, 'r_squared': np.nan, 'fit_success': False}
    
    t, y = trs[valid], soda_values[valid]
    t_min, t_max = t.min(), t.max()
    y_range = np.ptp(y)
    y_mean = np.mean(y)
    a0 = max(y_range / 2, 0.001)
    
    lower = [-np.inf, 3.0, t_min - 1.0, -np.inf]
    upper = [np.inf, 12.0, t_max, np.inf]
    
    best_result, best_residual = None, np.inf
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if restart == 0:
            p0 = [a0, (t_max - t_min) * 0.8, t_min, y_mean]
        elif restart == 1:
            p0 = [-a0, (t_max - t_min) * 0.8, t_min, y_mean]
        else:
            p0 = [rng.uniform(-a0*2, a0*2),
                  rng.uniform(3.0, min(12.0, t_max - t_min + 2)),
                  rng.uniform(t_min - 0.5, t_min + (t_max - t_min) * 0.3),
                  y_mean + rng.normal(0, a0 * 0.1)]
        try:
            popt, _ = curve_fit(_windowed_for_curvefit, t, y, p0=p0,
                               bounds=(lower, upper), maxfev=5000)
            resid = np.sum((y - _windowed_for_curvefit(t, *popt))**2)
            if resid < best_residual:
                best_residual = resid
                best_result = popt
        except (RuntimeError, ValueError):
            continue
    
    if best_result is None:
        return {'amplitude': np.nan, 'duration': np.nan, 'onset': np.nan,
                'offset': np.nan, 'r_squared': np.nan, 'fit_success': False}
    
    amplitude, duration, onset, offset = best_result
    y_hat = _windowed_for_curvefit(t, *best_result)
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {'amplitude': amplitude, 'duration': duration, 'onset': onset,
            'offset': offset, 'r_squared': r_squared, 'fit_success': True}


# =====================================================================
# 3. TEMPLATE MATCHING
# =====================================================================

def _generate_template(n_trs, n_items=5, isi_seconds=0.05, onset_tr=0.0):
    """Generate expected SODA slope timecourse for a replay event."""
    from scipy.stats import linregress
    
    # Simplified HRF: flattened sine wave (Wittkuhn & Schuck parameters)
    wavelength = 5.26
    delay = 0.56
    amplitude = 1.0  # unit amplitude for template
    baseline = 0.0
    
    t = np.arange(1, n_trs + 1, dtype=float)
    probs = np.zeros((n_items, n_trs))
    
    for i in range(n_items):
        item_onset = onset_tr + i * (isi_seconds + 0.0) / 1.25  # convert to TRs
        t_shifted = t - item_onset
        
        f = 1.0 / wavelength
        h = (amplitude / 2) * np.sin(2 * np.pi * f * t_shifted
                                      - 2 * np.pi * f * delay - np.pi / 2) + baseline + amplitude / 2
        response = np.where((t_shifted >= delay) & (t_shifted <= delay + wavelength), h, baseline)
        probs[i] = response
    
    # Compute slopes
    positions = np.arange(1, n_items + 1, dtype=float)
    slopes = np.zeros(n_trs)
    for t_idx in range(n_trs):
        slope, _, _, _, _ = linregress(positions, probs[:, t_idx])
        slopes[t_idx] = -slope
    
    # Normalize to unit norm
    norm = np.linalg.norm(slopes)
    return slopes / norm if norm > 0 else slopes


def template_matching_optimal_lag(
    soda_values,
    n_trs=None,
    n_items=5,
    isi_seconds=0.05,
    lag_min=-2.0,
    lag_max=6.0,
    n_lags=25,
):
    """
    Slide template across onset lags, return best correlation and lag.
    """
    soda_values = np.asarray(soda_values, dtype=float)
    if n_trs is None:
        n_trs = len(soda_values)
    
    lags = np.linspace(lag_min, lag_max, n_lags)
    best_r, best_lag = -np.inf, 0.0
    
    for lag in lags:
        template = _generate_template(n_trs, n_items, isi_seconds, onset_tr=lag)
        valid = ~(np.isnan(soda_values) | np.isnan(template))
        if valid.sum() < 4:
            continue
        r, _ = pearsonr(soda_values[valid], template[valid])
        if not np.isnan(r) and r > best_r:
            best_r = r
            best_lag = lag
    
    # Also compute absolute best (for unsigned detection: max |r|)
    best_abs_r = abs(best_r) if not np.isinf(best_r) else np.nan
    
    return {
        'template_r': best_r if not np.isinf(best_r) else np.nan,
        'template_abs_r': best_abs_r,
        'template_lag': best_lag,
    }


# =====================================================================
# 4. BAYESIAN MODEL COMPARISON
# =====================================================================

def bayesian_model_comparison(soda_values, trs=None):
    """
    Compare null (Gaussian noise) vs replay (windowed sinusoid) via BIC.
    
    Returns log10 Bayes factor (positive = evidence for replay).
    """
    soda_values = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(soda_values)
    y = soda_values[valid]
    n = len(y)
    
    if n < 5:
        return {'log10_bf': np.nan, 'delta_bic': np.nan, 'preferred': 'insufficient_data'}
    
    if trs is None:
        trs = np.arange(1, len(soda_values) + 1, dtype=float)
    t = trs[valid]
    
    # M0: Null — i.i.d. Gaussian (2 params: mean, variance)
    sigma2_null = np.var(y)
    if sigma2_null <= 0:
        sigma2_null = 1e-10
    ll_null = -0.5 * n * np.log(2 * np.pi * sigma2_null) - 0.5 * n
    bic_null = -2 * ll_null + 2 * np.log(n)
    
    # M3: Any replay — windowed sinusoid + noise (5 params)
    w_fit = fit_windowed_sinusoid(t, y, n_restarts=5)
    
    if w_fit['fit_success']:
        y_hat = _windowed_for_curvefit(t, w_fit['amplitude'], w_fit['duration'],
                                        w_fit['onset'], w_fit['offset'])
        residuals = y - y_hat
        sigma2_replay = np.var(residuals)
        if sigma2_replay <= 0:
            sigma2_replay = 1e-10
        ll_replay = (-0.5 * n * np.log(2 * np.pi * sigma2_replay)
                     - 0.5 * np.sum(residuals**2) / sigma2_replay)
        bic_replay = -2 * ll_replay + 5 * np.log(n)
    else:
        bic_replay = np.inf
        ll_replay = -np.inf
    
    delta_bic = bic_replay - bic_null  # negative = replay preferred
    bf = np.exp(-0.5 * delta_bic) if np.isfinite(delta_bic) else np.nan
    log10_bf = np.log10(bf) if (bf is not None and bf > 0 and np.isfinite(bf)) else np.nan
    
    return {
        'log10_bf': log10_bf,
        'delta_bic': delta_bic,
        'preferred': 'replay' if delta_bic < 0 else 'null',
    }


# =====================================================================
# 5. TRIAL-LEVEL COMPUTATION: ALL METRICS
# =====================================================================

def compute_all_trial_metrics(
    df: pd.DataFrame,
    soda_cols: dict = None,
) -> pd.DataFrame:
    """
    For each trial × condition, compute all metrics:
    - Simple: mean_slope, abs_mean, variance, peak_to_trough
    - Windowed sinusoid: amplitude, duration, onset, R²
    - Template matching: optimal-lag correlation
    - Bayesian: log10 Bayes factor
    """
    if soda_cols is None:
        soda_cols = {
            "retrieved": "retrieved_soda",
            "control": "control_soda",
            "shuf_retrieved": "retrieved_soda_shuffled",
            "shuf_control": "control_soda_shuffled",
        }
    
    group_cols = ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
    group_cols = [c for c in group_cols if c in df.columns]
    
    results = []
    n_trials_total = df.groupby(group_cols, sort=False).ngroups
    
    for i, (keys, trial_df) in enumerate(df.groupby(group_cols, sort=False)):
        if i % 50 == 0:
            print(f"  Processing trial {i+1}/{n_trials_total}...")
        
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))
        
        trial_df = trial_df.sort_values("tr")
        trs = trial_df["tr"].values
        
        for cond_label, col_name in soda_cols.items():
            if col_name not in trial_df.columns:
                continue
            
            vals = trial_df[col_name].values.astype(float)
            prefix = cond_label
            
            # --- Simple metrics ---
            trial_info[f"{prefix}_mean_slope"] = metric_mean_slope(vals)
            trial_info[f"{prefix}_abs_mean"] = metric_abs_mean(vals)
            trial_info[f"{prefix}_variance"] = metric_variance(vals)
            trial_info[f"{prefix}_peak_to_trough"] = metric_peak_to_trough(vals)
            
            # --- Windowed sinusoidal fit ---
            w_fit = fit_windowed_sinusoid(trs, vals)
            trial_info[f"{prefix}_w_amplitude"] = w_fit['amplitude']
            trial_info[f"{prefix}_w_abs_amplitude"] = abs(w_fit['amplitude']) if not np.isnan(w_fit['amplitude']) else np.nan
            trial_info[f"{prefix}_w_duration"] = w_fit['duration']
            trial_info[f"{prefix}_w_onset"] = w_fit['onset']
            trial_info[f"{prefix}_w_r_squared"] = w_fit['r_squared']
            trial_info[f"{prefix}_w_fit_success"] = w_fit['fit_success']
            
            # --- Template matching ---
            tm = template_matching_optimal_lag(vals, n_trs=len(trs))
            trial_info[f"{prefix}_template_r"] = tm['template_r']
            trial_info[f"{prefix}_template_abs_r"] = tm['template_abs_r']
            trial_info[f"{prefix}_template_lag"] = tm['template_lag']
            
            # --- Bayesian model comparison ---
            bmc = bayesian_model_comparison(vals, trs)
            trial_info[f"{prefix}_log10_bf"] = bmc['log10_bf']
            trial_info[f"{prefix}_delta_bic"] = bmc['delta_bic']
            trial_info[f"{prefix}_preferred"] = bmc['preferred']
        
        # --- Metadata ---
        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan
        
        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan
        
        results.append(trial_info)
    
    return pd.DataFrame(results)


# =====================================================================
# 6. GROUP-LEVEL STATISTICS
# =====================================================================

def compute_group_stats(metrics_df, metric_col, ctrl_col, label=""):
    """Paired t-test at subject level: metric_col vs ctrl_col."""
    if metric_col not in metrics_df.columns or ctrl_col not in metrics_df.columns:
        return None
    
    subj = metrics_df.groupby("subject")[[metric_col, ctrl_col]].mean().dropna()
    if len(subj) < 3:
        return None
    
    t_stat, p_val = ttest_rel(subj[metric_col], subj[ctrl_col])
    mean_diff = (subj[metric_col] - subj[ctrl_col]).mean()
    cohens_d = mean_diff / (subj[metric_col] - subj[ctrl_col]).std()
    
    return {
        'comparison': label,
        'metric': metric_col,
        'control': ctrl_col,
        'n_subjects': len(subj),
        't_stat': t_stat,
        'p_value': p_val,
        'mean_diff': mean_diff,
        'cohens_d': cohens_d,
        'mean_metric': subj[metric_col].mean(),
        'mean_control': subj[ctrl_col].mean(),
    }


# =====================================================================
# 7. PLOTTING
# =====================================================================

def plot_metric_comparison_overview(metrics_df, roi_name, save_path=None):
    """
    Main summary figure: compare all metrics across conditions.
    4 panels showing retrieved vs control for each metric type.
    """
    # Define comparisons
    metric_pairs = [
        # (retrieved_col, control_col, shuf_retrieved_col, shuf_control_col, label)
        ('retrieved_abs_mean', 'control_abs_mean', 'shuf_retrieved_abs_mean', 'shuf_control_abs_mean', '|Slope| mean'),
        ('retrieved_peak_to_trough', 'control_peak_to_trough', 'shuf_retrieved_peak_to_trough', 'shuf_control_peak_to_trough', 'Peak-to-trough'),
        ('retrieved_variance', 'control_variance', 'shuf_retrieved_variance', 'shuf_control_variance', 'Slope variance'),
        ('retrieved_w_abs_amplitude', 'control_w_abs_amplitude', 'shuf_retrieved_w_abs_amplitude', 'shuf_control_w_abs_amplitude', 'Windowed |A|'),
        ('retrieved_template_abs_r', 'control_template_abs_r', 'shuf_retrieved_template_abs_r', 'shuf_control_template_abs_r', 'Template |r|'),
        ('retrieved_log10_bf', 'control_log10_bf', 'shuf_retrieved_log10_bf', 'shuf_control_log10_bf', 'log₁₀ BF'),
    ]

    # Filter to available columns
    metric_pairs = [(r, c, sr, sc, l) for r, c, sr, sc, l in metric_pairs
                    if r in metrics_df.columns and c in metrics_df.columns]
    
    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = {'retrieved': '#3274A1', 'control': '#E1812C',
              'shuf_retrieved': '#8CB8D0', 'shuf_control': '#F5C28A'}

    for idx, (ret_col, ctrl_col, shuf_ret_col, shuf_ctrl_col, label) in enumerate(metric_pairs):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Subject-level means
        cols_to_use = [ret_col, ctrl_col]
        col_labels = ['Retrieved', 'Control']
        col_colors = [colors['retrieved'], colors['control']]

        if shuf_ret_col in metrics_df.columns:
            cols_to_use.append(shuf_ret_col)
            col_labels.append('Shuf Ret')
            col_colors.append(colors['shuf_retrieved'])

        if shuf_ctrl_col in metrics_df.columns:
            cols_to_use.append(shuf_ctrl_col)
            col_labels.append('Shuf Ctrl')
            col_colors.append(colors['shuf_control'])
        
        subj = metrics_df.groupby("subject")[cols_to_use].mean().dropna()
        if len(subj) < 3:
            ax.set_title(f'{label}\n(insufficient data)')
            continue
        
        # Boxplot + individual points
        plot_data = [subj[c].values for c in cols_to_use]
        bp = ax.boxplot(plot_data, positions=range(len(plot_data)),
                       widths=0.5, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], col_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        rng = np.random.default_rng(42)
        for i, (vals, color) in enumerate(zip(plot_data, col_colors)):
            jitter = rng.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                      color=color, edgecolor='k', linewidth=0.5,
                      s=30, zorder=5, alpha=0.8)
        
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_ylabel(label)
        
        # Paired t-test: retrieved vs control
        t_stat, p_val = ttest_rel(subj[ret_col], subj[ctrl_col])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.set_title(f'{label}\nRet vs Ctrl: t={t_stat:.2f}, p={p_val:.3f} {sig}',
                    fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    
    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Enhanced SODA Metrics: {roi_name}\nSubject-level means (n={metrics_df["subject"].nunique()})',
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metric_correlation_matrix(metrics_df, roi_name, save_path=None):
    """Show correlations between metrics at the trial level."""
    
    metric_cols = [
        'retrieved_abs_mean', 'retrieved_variance', 'retrieved_peak_to_trough',
        'retrieved_w_abs_amplitude', 'retrieved_template_abs_r', 'retrieved_log10_bf',
    ]
    labels = ['|Slope| mean', 'Variance', 'Peak-to-trough',
              'Windowed |A|', 'Template |r|', 'log₁₀ BF']
    
    available = [(c, l) for c, l in zip(metric_cols, labels) if c in metrics_df.columns]
    if len(available) < 2:
        return
    
    cols, labs = zip(*available)
    data = metrics_df[list(cols)].dropna()
    
    corr = data.corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white' if abs(corr.values[i,j]) > 0.6 else 'black')
    
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=9)
    plt.colorbar(im, label='Pearson r')
    ax.set_title(f'Trial-level metric correlations: {roi_name}')
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_stats_summary_table(all_stats, roi_name, save_path=None):
    """Create a summary bar chart of t-statistics across metrics."""
    
    if not all_stats:
        return
    
    stats_df = pd.DataFrame(all_stats)
    
    # Focus on retrieved vs control
    ret_vs_ctrl = stats_df[stats_df['comparison'].str.contains('vs_ctrl')].copy()
    
    if ret_vs_ctrl.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = ret_vs_ctrl['comparison'].values
    t_vals = ret_vs_ctrl['t_stat'].values
    p_vals = ret_vs_ctrl['p_value'].values
    
    colors = ['#2ca02c' if p < 0.05 else '#d62728' if p < 0.1 else '#999999'
              for p in p_vals]
    
    bars = ax.barh(range(len(labels)), t_vals, color=colors, alpha=0.7, edgecolor='white')
    
    for i, (t, p) in enumerate(zip(t_vals, p_vals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(t + 0.1 * np.sign(t), i, f't={t:.2f}, p={p:.3f} {sig}',
               va='center', fontsize=9)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('t-statistic (paired)')
    ax.axvline(0, color='gray', ls='--')
    ax.set_title(f'Group-level statistics: Retrieved vs Control\n{roi_name}')
    ax.spines[['top', 'right']].set_visible(False)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# 8. ENTRY POINT
# =====================================================================

def main():
    roi_filter = sys.argv[1] if len(sys.argv) > 1 else None
    Pre_post = sys.argv[2] if len(sys.argv) > 2 else 'presleep'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir,
                  f'4.6_SODA_based_on_singlesubj_4.4_{Pre_post}', 'SODA_results_csv')
    
    csv_files = sorted(glob.glob(opj(csv_dir, '*_soda_results_per_combination.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")
    
    all_dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if 'ROI' not in df.columns:
            roi_name = os.path.basename(fp).replace('_soda_results_per_combination.csv', '')
            df['ROI'] = roi_name
        all_dfs.append(df)
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    for col in ["tr", "retrieved_soda", "control_soda", "behav_correct",
                "retrieved_soda_shuffled", "control_soda_shuffled"]:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
    
    if roi_filter:
        master_df = master_df[master_df['ROI'] == roi_filter]
        if master_df.empty:
            available = sorted(pd.concat(all_dfs, ignore_index=True)['ROI'].unique())
            raise ValueError(f"ROI '{roi_filter}' not found. Available: {available}")
    
    rois = sorted(master_df['ROI'].unique())
    print(f"Found {len(rois)} ROIs: {rois}")
    
    all_roi_summaries = []
    
    for roi in rois:
        print(f"\n{'='*60}")
        print(f"ROI: {roi}")
        print(f"{'='*60}")
        
        df = master_df[master_df['ROI'] == roi].copy()
        out_dir = create_script_named_folder(roi)
        
        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Computing all metrics per trial...")
        
        metrics_df = compute_all_trial_metrics(df)
        print(f"Computed metrics for {len(metrics_df)} trials")
        
        # Save
        out_csv = opj(out_dir, f"{roi}_enhanced_metrics.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"Saved to: {out_csv}")
        
        # --- Group-level statistics ---
        print("\n--- Group-level statistics ---")
        all_stats = []
        
        comparisons = [
            # (retrieved_col, control_col, label)
            ('retrieved_mean_slope', 'control_mean_slope', 'mean_slope_vs_ctrl'),
            ('retrieved_abs_mean', 'control_abs_mean', 'abs_mean_vs_ctrl'),
            ('retrieved_variance', 'control_variance', 'variance_vs_ctrl'),
            ('retrieved_peak_to_trough', 'control_peak_to_trough', 'ptt_vs_ctrl'),
            ('retrieved_w_abs_amplitude', 'control_w_abs_amplitude', 'windowed_amp_vs_ctrl'),
            ('retrieved_template_abs_r', 'control_template_abs_r', 'template_r_vs_ctrl'),
            ('retrieved_log10_bf', 'control_log10_bf', 'log10_bf_vs_ctrl'),
        ]
        
        # Also add retrieved vs shuffled
        comparisons += [
            ('retrieved_abs_mean', 'shuf_retrieved_abs_mean', 'abs_mean_vs_shuf'),
            ('retrieved_w_abs_amplitude', 'shuf_retrieved_w_abs_amplitude', 'windowed_amp_vs_shuf'),
            ('retrieved_template_abs_r', 'shuf_retrieved_template_abs_r', 'template_r_vs_shuf'),
            ('retrieved_log10_bf', 'shuf_retrieved_log10_bf', 'log10_bf_vs_shuf'),
        ]
        
        for ret_col, ctrl_col, label in comparisons:
            stats = compute_group_stats(metrics_df, ret_col, ctrl_col, label)
            if stats is not None:
                sig = '***' if stats['p_value'] < 0.001 else '**' if stats['p_value'] < 0.01 else '*' if stats['p_value'] < 0.05 else 'n.s.'
                print(f"  {label}: t({stats['n_subjects']-1})={stats['t_stat']:.3f}, "
                      f"p={stats['p_value']:.4f} {sig}, d={stats['cohens_d']:.3f}")
                all_stats.append(stats)
        
        # --- Plots ---
        print("\nGenerating plots...")
        
        plot_metric_comparison_overview(
            metrics_df, roi,
            save_path=opj(out_dir, f"{roi}_metric_comparison_overview.png"),
        )
        
        plot_metric_correlation_matrix(
            metrics_df, roi,
            save_path=opj(out_dir, f"{roi}_metric_correlations.png"),
        )
        
        plot_stats_summary_table(
            all_stats, roi,
            save_path=opj(out_dir, f"{roi}_stats_summary.png"),
        )
        
        # --- ROI summary ---
        roi_summary = {
            'ROI': roi,
            'n_subjects': df['subject'].nunique(),
            'n_trials': len(metrics_df),
        }
        for s in all_stats:
            roi_summary[f"t_{s['comparison']}"] = s['t_stat']
            roi_summary[f"p_{s['comparison']}"] = s['p_value']
            roi_summary[f"d_{s['comparison']}"] = s['cohens_d']
        
        all_roi_summaries.append(roi_summary)
        print(f"\nDone with ROI {roi}.")
    
    # Cross-ROI summary
    summary_dir = create_script_named_folder('summary')
    summary_df = pd.DataFrame(all_roi_summaries)
    summary_path = opj(summary_dir, 'enhanced_metrics_summary_all_rois.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()