#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A5 - Behavioural analysis: retrieval rate and reaction time.

Extracts the retrieval trials from the combined behaviour table and reports
  - retrieval rate and reaction time by trial type (object / scene)
  - both by sequence position
  - both by session (evening, morning 1, morning 2)
  - mixed models with participant as random intercept:
        pct_retrieved ~ session            mean_rt ~ session
        pct_retrieved ~ session * type     mean_rt ~ session * type

  - reaction time by sequence position, separately for each session, with
    the per-participant slope (seconds per position) and a mixed model
        mean_rt ~ seq_pos * session      (per trial type)

Input   outputs/A4_aggregate_behavior/all_subjects_behav_data.csv
Output  outputs/A5_analyse_behavior/lme_session_models.txt
        outputs/A5_analyse_behavior/rt_by_position_by_session.txt   (slope tests, models)
        outputs/A5_analyse_behavior/rt_by_position_by_session_means.csv
        outputs/A5_analyse_behavior/rt_position_slopes_per_participant.csv
Plots   plots/A5_analyse_behavior/fig1_retrieval_overview.{pdf,png}
                                  fig2_session_comparison.{pdf,png}
                                  fig3_overall_retrieval_by_session.{pdf,png}
                                  fig4_rt_by_position_per_session.{pdf,png}

Ported from 1_prepare_behavior/4_analyse_behav_data.py. The analysis and
plotting functions are copied unchanged; only the input/output paths were
replaced, figures are no longer shown on screen, and the model summaries are
also written to a text file. The reaction time by position per session
(figure 4 and its statistics) was added on 2026-11-18 and is not in the
original script.

Usage:  python A5_analyse_behavior.py
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_input_dir, step_output_dir, step_plot_dir

warnings.filterwarnings('ignore')

A4_DIR = step_input_dir("A4_aggregate_behavior")
STATS_LOG = []

# ── Publication-quality style (Nature Neuroscience) ──────────────────────────

def set_nature_style():
    """Configure matplotlib for Nature Neuroscience quality figures."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


COLORS = {
    'scene':    '#2166AC',
    'object':   '#B2182B',
    'evening':  '#4A4A4A',
    'morning1': '#E08214',
    'morning2': '#8073AC',
    'bar_edge': '#333333',
    'chance':   '#999999',
}

SESSION_LABELS = {0: 'Evening (S0)', 1: 'Morning 1 (S1)', 2: 'Morning 2 (S2)'}
SESSION_COLORS = {v: c for v, c in zip(
    SESSION_LABELS.values(),
    [COLORS['evening'], COLORS['morning1'], COLORS['morning2']],
)}



# ── Trial extraction ─────────────────────────────────────────────────────────

def extract_retrieval_trials(df):
    """
    Parse the raw event-log DataFrame into one row per retrieval trial.

    Groups events by (sub_id, ses_nr, trl_nr) — where trl_nr is back-filled
    from ret_fix rows that mark trial boundaries. Within each trial, checks
    for sc_no4obj / d_no4sc (retrieval cue) and q_mark_press (response).

    Returns a tidy DataFrame with columns:
        sub_id, ses_nr, session_label, trl_nr, trial_type, ret_cue,
        obj_ret_targ, sc_ret_targ, seq_pos, retrieved, rt
    """
    # Back-fill trl_nr so every event inherits the trl_nr of its closing ret_fix
    df = df.copy()
    df['trl_nr'] = df.groupby(['sub_id', 'ses_nr'])['trl_nr'].bfill()

    # Keep only rows with a valid trl_nr
    df = df.dropna(subset=['trl_nr'])

    trials = []
    for (sub_id, ses_nr, trl_nr), trl_df in df.groupby(['sub_id', 'ses_nr', 'trl_nr']):
        labels = set(trl_df['label_type'])

        # Find the retrieval cue row (sc_no4obj or d_no4sc)
        cue_rows = trl_df[trl_df['label_type'].isin(['sc_no4obj', 'd_no4sc'])]
        if cue_rows.empty:
            continue

        cue = cue_rows.iloc[0]
        trial_type = 'Object' if cue['label_type'] == 'sc_no4obj' else 'Scene'

        # Check for q_mark_press within this trial
        pressed = 'q_mark_press' in labels
        rt = np.nan
        if pressed:
            press_row = trl_df[trl_df['label_type'] == 'q_mark_press'].iloc[0]
            rt = press_row['ret_resp']

        # Parse sequence position (e.g. "03.png" -> 3)
        try:
            seq_pos = int(str(cue['seq_pos']).replace('.png', '').replace('.0', ''))
        except (ValueError, TypeError):
            seq_pos = np.nan

        trials.append({
            'sub_id':        sub_id,
            'ses_nr':        ses_nr,
            'session_label': SESSION_LABELS.get(ses_nr, f'Session {ses_nr}'),
            'trl_nr':        int(trl_nr),
            'trial_type':    trial_type,
            'ret_cue':       cue['ret_cue'],
            'obj_ret_targ':  cue['obj_ret_targ'],
            'sc_ret_targ':   cue['sc_ret_targ'],
            'seq_pos':       seq_pos,
            'retrieved':     pressed,
            'rt':            rt if pressed else np.nan,
        })

    return pd.DataFrame(trials)


# ── Statistics helpers ────────────────────────────────────────────────────────

def _agg_mean_sem(series):
    """Return mean and SEM for a boolean or numeric series."""
    return series.mean(), series.sem(), series.count()


def _subject_then_group(tdf, groupby_cols, value_col, agg_func='mean'):
    """
    Two-level aggregation: first collapse within subject, then across subjects.

    1. Compute each subject's mean for `value_col` within each combination of `groupby_cols`.
    2. Then compute the grand mean and SEM (across subjects) for each group.

    Returns a DataFrame with columns: *groupby_cols, 'mean', 'sem', 'count'
    where 'count' = number of subjects and 'sem' = SEM across subject means.
    """
    # Step 1: subject-level means
    sub_means = (
        tdf.groupby(['sub_id'] + groupby_cols)[value_col]
        .mean()
        .reset_index(name='sub_mean')
    )
    # Step 2: aggregate across subjects
    result = (
        sub_means.groupby(groupby_cols)['sub_mean']
        .agg(mean='mean', sem='sem', count='count')
        .reset_index()
    )
    return result


def compute_retrieval_rates(tdf):
    """Compute retrieval success rates, aggregated at the subject level."""
    r = {}
    # Overall: mean of subject means
    sub_overall = tdf.groupby('sub_id')['retrieved'].mean()
    r['overall'] = sub_overall.mean()

    r['by_type'] = _subject_then_group(tdf, ['trial_type'], 'retrieved')
    r['by_seqpos'] = _subject_then_group(tdf, ['seq_pos'], 'retrieved')
    r['by_type_seqpos'] = _subject_then_group(tdf, ['trial_type', 'seq_pos'], 'retrieved')

    if 'session_label' in tdf.columns and tdf['session_label'].nunique() > 1:
        r['by_session'] = _subject_then_group(tdf, ['session_label'], 'retrieved')
        r['by_session_type'] = _subject_then_group(tdf, ['session_label', 'trial_type'], 'retrieved')
        r['by_session_type_seqpos'] = _subject_then_group(
            tdf, ['session_label', 'trial_type', 'seq_pos'], 'retrieved')
    return r


def compute_rt_stats(tdf):
    """Compute RT statistics (retrieved trials only), aggregated at the subject level."""
    ret = tdf[tdf['retrieved']].copy()
    r = {}

    # by_type: also include median (median of subject means)
    sub_rt = ret.groupby(['sub_id', 'trial_type'])['rt'].mean().reset_index(name='sub_mean')
    by_type = sub_rt.groupby('trial_type')['sub_mean'].agg(
        mean='mean', sem='sem', count='count', median='median').reset_index()
    r['by_type'] = by_type

    r['by_seqpos'] = _subject_then_group(ret, ['seq_pos'], 'rt')
    r['by_type_seqpos'] = _subject_then_group(ret, ['trial_type', 'seq_pos'], 'rt')

    if 'session_label' in tdf.columns and tdf['session_label'].nunique() > 1:
        r['by_session_type'] = _subject_then_group(ret, ['session_label', 'trial_type'], 'rt')
        r['by_session_seqpos'] = _subject_then_group(ret, ['session_label', 'seq_pos'], 'rt')
    return r


# ── Plotting functions ────────────────────────────────────────────────────────

def _type_color(trial_type):
    return COLORS['scene'] if trial_type == 'Scene' else COLORS['object']


def plot_retrieval_by_type(rates, ax=None, title='Retrieval by memory type', trials=None):
    data = rates['by_type']
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4.5))

    x = np.arange(len(data))
    colors = [_type_color(t) for t in data['trial_type']]

    ax.bar(x, data['mean'], yerr=data['sem'], width=0.55,
           color=colors, edgecolor=COLORS['bar_edge'], linewidth=1,
           capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2})

    if trials is not None:
        sub_means = trials.groupby(['sub_id', 'trial_type'])['retrieved'].mean().reset_index()
        types_list = list(data['trial_type'])
        for sid in sub_means['sub_id'].unique():
            sdata = sub_means[sub_means['sub_id'] == sid]
            jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.12, 0.12)
            xs_s, ys_s = [], []
            for i, tt in enumerate(types_list):
                row_s = sdata[sdata['trial_type'] == tt]
                if not row_s.empty:
                    xs_s.append(x[i] + jit)
                    ys_s.append(row_s['retrieved'].values[0])
            ax.plot(xs_s, ys_s, color='black', lw=0.5, alpha=0.25, zorder=2)
            ax.scatter(xs_s, ys_s, color='black', s=18, alpha=0.45, zorder=3, linewidths=0)

    ax.axhline(0.5, color=COLORS['chance'], ls='--', lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(data['trial_type'], fontweight='medium')
    ax.set_ylabel('Retrieval rate')
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight='bold', pad=12)

    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(i, row['mean'] + row['sem'] + 0.04,
                f"N={int(row['count'])}", ha='center', fontsize=9, color='#555')
    return ax


def plot_retrieval_by_seqpos(rates, ax=None, title='Retrieval by sequence position', trials=None):
    data = rates['by_type_seqpos']
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))

    if trials is not None:
        sub_means = trials.groupby(['sub_id', 'trial_type', 'seq_pos'])['retrieved'].mean().reset_index()

    for tt, col in [('Scene', COLORS['scene']), ('Object', COLORS['object'])]:
        sub = data[data['trial_type'] == tt].sort_values('seq_pos')
        if sub.empty:
            continue
        if trials is not None:
            pts = sub_means[sub_means['trial_type'] == tt]
            for sid in pts['sub_id'].unique():
                sdata = pts[pts['sub_id'] == sid].sort_values('seq_pos')
                jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.08, 0.08)
                ax.plot(sdata['seq_pos'] + jit, sdata['retrieved'],
                        color=col, lw=0.5, alpha=0.2, zorder=1)
                ax.scatter(sdata['seq_pos'] + jit, sdata['retrieved'],
                           color=col, s=14, alpha=0.25, zorder=1, linewidths=0)
        ax.errorbar(sub['seq_pos'], sub['mean'], yerr=sub['sem'],
                    marker='o', color=col, markerfacecolor=col,
                    markeredgecolor='white', markeredgewidth=1.2,
                    label=tt, capsize=3, capthick=1, zorder=4)

    ax.axhline(0.5, color=COLORS['chance'], ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Retrieval rate')
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(sorted(data['seq_pos'].dropna().unique()))
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(frameon=False, loc='lower left')
    return ax


def plot_rt_by_type(rt_stats, ax=None, title='RT by memory type', trials=None):
    data = rt_stats['by_type']
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4.5))

    x = np.arange(len(data))
    colors = [_type_color(t) for t in data['trial_type']]

    ax.bar(x, data['mean'], yerr=data['sem'], width=0.55,
           color=colors, edgecolor=COLORS['bar_edge'], linewidth=1,
           capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2}, alpha=0.85)

    if trials is not None:
        ret = trials[trials['retrieved']]
        sub_means = ret.groupby(['sub_id', 'trial_type'])['rt'].mean().reset_index()
        types_list = list(data['trial_type'])
        for sid in sub_means['sub_id'].unique():
            sdata = sub_means[sub_means['sub_id'] == sid]
            jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.12, 0.12)
            xs_s, ys_s = [], []
            for i, tt in enumerate(types_list):
                row_s = sdata[sdata['trial_type'] == tt]
                if not row_s.empty:
                    xs_s.append(x[i] + jit)
                    ys_s.append(row_s['rt'].values[0])
            ax.plot(xs_s, ys_s, color='black', lw=0.5, alpha=0.25, zorder=2)
            ax.scatter(xs_s, ys_s, color='black', s=18, alpha=0.45, zorder=3, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(data['trial_type'], fontweight='medium')
    ax.set_ylabel('Reaction time (s)')
    ax.set_ylim(0, None)
    ax.set_title(title, fontweight='bold', pad=12)

    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(i, row['mean'] + row['sem'] + 0.1,
                f"N={int(row['count'])}", ha='center', fontsize=9, color='#555')
    return ax


def plot_rt_by_seqpos(rt_stats, ax=None, title='RT by sequence position', trials=None):
    data = rt_stats['by_type_seqpos']
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))

    if trials is not None:
        ret = trials[trials['retrieved']]
        sub_means = ret.groupby(['sub_id', 'trial_type', 'seq_pos'])['rt'].mean().reset_index()

    for tt, col in [('Scene', COLORS['scene']), ('Object', COLORS['object'])]:
        sub = data[data['trial_type'] == tt].sort_values('seq_pos')
        if sub.empty:
            continue
        if trials is not None:
            pts = sub_means[sub_means['trial_type'] == tt]
            for sid in pts['sub_id'].unique():
                sdata = pts[pts['sub_id'] == sid].sort_values('seq_pos')
                jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.08, 0.08)
                ax.plot(sdata['seq_pos'] + jit, sdata['rt'],
                        color=col, lw=0.5, alpha=0.2, zorder=1)
                ax.scatter(sdata['seq_pos'] + jit, sdata['rt'],
                           color=col, s=14, alpha=0.25, zorder=1, linewidths=0)
        ax.errorbar(sub['seq_pos'], sub['mean'], yerr=sub['sem'],
                    marker='o', color=col, markerfacecolor=col,
                    markeredgecolor='white', markeredgewidth=1.2,
                    label=tt, capsize=3, capthick=1, zorder=4)

    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Reaction time (s)')
    ax.set_xticks(sorted(data['seq_pos'].dropna().unique()))
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(frameon=False, loc='upper left')
    return ax


def plot_session_comparison(rates, ax=None, title='Retrieval across sessions', trials=None):
    data = rates.get('by_session_type')
    if data is None or data.empty:
        return ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    sessions = sorted(data['session_label'].unique())
    types = ['Scene', 'Object']
    x = np.arange(len(sessions))
    width = 0.32

    if trials is not None:
        sub_means = trials.groupby(['sub_id', 'session_label', 'trial_type'])['retrieved'].mean().reset_index()

    for i, tt in enumerate(types):
        sub = data[data['trial_type'] == tt]
        vals, errs = [], []
        for s in sessions:
            row = sub[sub['session_label'] == s]
            vals.append(row['mean'].values[0] if len(row) else 0)
            errs.append(row['sem'].values[0] if len(row) else 0)

        bar_x = x + (i - 0.5) * width
        ax.bar(bar_x, vals, width=width, yerr=errs,
               label=tt, color=_type_color(tt),
               edgecolor=COLORS['bar_edge'], linewidth=0.8,
               capsize=3, error_kw={'linewidth': 1, 'capthick': 1})

        if trials is not None:
            tt_data = sub_means[sub_means['trial_type'] == tt]
            for sid in tt_data['sub_id'].unique():
                sdata = tt_data[tt_data['sub_id'] == sid]
                jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.06, 0.06)
                xs_s, ys_s = [], []
                for j, s in enumerate(sessions):
                    row_s = sdata[sdata['session_label'] == s]
                    if not row_s.empty:
                        xs_s.append(bar_x[j] + jit)
                        ys_s.append(row_s['retrieved'].values[0])
                ax.plot(xs_s, ys_s, color='black', lw=0.5, alpha=0.2, zorder=2)
                ax.scatter(xs_s, ys_s, color='black', s=14, alpha=0.4, zorder=3, linewidths=0)

    ax.axhline(0.5, color=COLORS['chance'], ls='--', lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontweight='medium')
    ax.set_ylabel('Retrieval rate')
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(frameon=False)
    return ax


def plot_session_rt(rt_stats, ax=None, title='RT across sessions', trials=None):
    data = rt_stats.get('by_session_type')
    if data is None or data.empty:
        return ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    sessions = sorted(data['session_label'].unique())
    types = ['Scene', 'Object']
    x = np.arange(len(sessions))
    width = 0.32

    if trials is not None:
        ret = trials[trials['retrieved']]
        sub_means = ret.groupby(['sub_id', 'session_label', 'trial_type'])['rt'].mean().reset_index()

    for i, tt in enumerate(types):
        sub = data[data['trial_type'] == tt]
        vals, errs = [], []
        for s in sessions:
            row = sub[sub['session_label'] == s]
            vals.append(row['mean'].values[0] if len(row) else 0)
            errs.append(row['sem'].values[0] if len(row) else 0)

        bar_x = x + (i - 0.5) * width
        ax.bar(bar_x, vals, width=width, yerr=errs,
               label=tt, color=_type_color(tt),
               edgecolor=COLORS['bar_edge'], linewidth=0.8,
               capsize=3, error_kw={'linewidth': 1, 'capthick': 1}, alpha=0.85)

        if trials is not None:
            tt_data = sub_means[sub_means['trial_type'] == tt]
            for sid in tt_data['sub_id'].unique():
                sdata = tt_data[tt_data['sub_id'] == sid]
                jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.06, 0.06)
                xs_s, ys_s = [], []
                for j, s in enumerate(sessions):
                    row_s = sdata[sdata['session_label'] == s]
                    if not row_s.empty:
                        xs_s.append(bar_x[j] + jit)
                        ys_s.append(row_s['rt'].values[0])
                ax.plot(xs_s, ys_s, color='black', lw=0.5, alpha=0.2, zorder=2)
                ax.scatter(xs_s, ys_s, color='black', s=14, alpha=0.4, zorder=3, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontweight='medium')
    ax.set_ylabel('Reaction time (s)')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(frameon=False)
    return ax


def plot_overall_retrieval_by_session(trials, ax=None, title='Overall retrieval by session'):
    """Bar plot of retrieval rate per session, collapsed across Scene/Object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Subject-level means per session
    sub_means = trials.groupby(['sub_id', 'session_label'])['retrieved'].mean().reset_index()

    # Group-level stats
    sessions = ['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)']
    sessions = [s for s in sessions if s in sub_means['session_label'].values]
    x = np.arange(len(sessions))
    session_colors = [SESSION_COLORS.get(s, '#888888') for s in sessions]

    means, sems, counts = [], [], []
    for s in sessions:
        vals = sub_means[sub_means['session_label'] == s]['retrieved']
        means.append(vals.mean())
        sems.append(vals.sem())
        counts.append(len(vals))

    ax.bar(x, means, yerr=sems, width=0.55,
           color=session_colors, edgecolor=COLORS['bar_edge'], linewidth=1,
           capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Subject dots with connecting lines
    for sid in sub_means['sub_id'].unique():
        sdata = sub_means[sub_means['sub_id'] == sid]
        jit = np.random.default_rng(hash(sid) % 2**31).uniform(-0.12, 0.12)
        xs_s, ys_s = [], []
        for i, s in enumerate(sessions):
            row_s = sdata[sdata['session_label'] == s]
            if not row_s.empty:
                xs_s.append(x[i] + jit)
                ys_s.append(row_s['retrieved'].values[0])
        ax.plot(xs_s, ys_s, color='black', lw=0.5, alpha=0.25, zorder=2)
        ax.scatter(xs_s, ys_s, color='black', s=18, alpha=0.45, zorder=3, linewidths=0)

    ax.axhline(0.5, color=COLORS['chance'], ls='--', lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontweight='medium')
    ax.set_ylabel('Retrieval rate')
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight='bold', pad=12)

    for i in range(len(sessions)):
        ax.text(i, means[i] + sems[i] + 0.04,
                f"N={counts[i]}", ha='center', fontsize=9, color='#555')
    return ax


def rt_position_slopes(trials, min_positions=3):
    """Per participant, session and trial type: slope of RT over sequence position.

    RT is first averaged per position (retrieved trials only), then a straight
    line is fitted over the positions. The slope is in seconds per position.
    Cells with fewer than `min_positions` positions are left out.
    """
    ret = trials[trials['retrieved'].astype(bool)].dropna(subset=['rt', 'seq_pos'])
    cell = (ret.groupby(['sub_id', 'session_label', 'trial_type', 'seq_pos'])['rt']
            .mean().reset_index())
    rows = []
    for (sid, ses, tt), g in cell.groupby(['sub_id', 'session_label', 'trial_type']):
        if g['seq_pos'].nunique() < min_positions:
            continue
        slope = np.polyfit(g['seq_pos'].astype(float), g['rt'].astype(float), 1)[0]
        rows.append({'sub_id': sid, 'session_label': ses, 'trial_type': tt,
                     'n_positions': int(g['seq_pos'].nunique()), 'rt_slope_s_per_position': slope})
    return pd.DataFrame(rows)


def plot_rt_slope_by_session(slopes, session_order, ax=None,
                             title='RT increase per position'):
    """Mean (± SEM across participants) RT slope per session, by trial type."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    x = np.arange(len(session_order))
    for tt, off in [('Scene', -0.08), ('Object', 0.08)]:
        col = _type_color(tt)
        d = slopes[slopes['trial_type'] == tt]
        wide = d.pivot(index='sub_id', columns='session_label',
                       values='rt_slope_s_per_position').reindex(columns=session_order)
        for _, row in wide.iterrows():
            ax.plot(x + off, row.values, color=col, lw=0.5, alpha=0.2, zorder=1)
        ax.errorbar(x + off, wide.mean(), yerr=wide.sem(), marker='o', color=col,
                    markerfacecolor=col, markeredgecolor='white', markeredgewidth=1.2,
                    label=tt, capsize=3, capthick=1, zorder=4)
    ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.split(' (')[0] for lab in session_order])
    ax.set_xlim(-0.5, len(session_order) - 0.5)
    ax.set_ylabel('RT slope (s per position)')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(frameon=False, loc='best')
    return ax


def add_panel_labels(axes, labels=None):
    """Add a, b, c, d panel labels to axes array."""
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes.flat))]
    for ax, lbl in zip(axes.flat, labels):
        ax.text(-0.15, 1.08, lbl, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')


# ── Main analysis ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    df = pd.read_csv(A4_DIR / 'all_subjects_behav_data.csv')

    # df and log come from load_all_data() above
    if df.empty:
        raise SystemExit("No data loaded. Run A4_aggregate_behavior.py first.")

    set_nature_style()

    # ── 1. Extract retrieval trials ──────────────────────────────────────────
    print("\nExtracting retrieval trials …")
    trials = extract_retrieval_trials(df)
    print(f"  Total retrieval trials: {len(trials)}")
    print(f"  Subjects: {sorted(trials['sub_id'].unique())}")
    print(f"  Sessions: {sorted(trials['ses_nr'].unique())}")
    print(f"  Scene trials:  {(trials['trial_type']=='Scene').sum()}")
    print(f"  Object trials: {(trials['trial_type']=='Object').sum()}")

    # ── 2. Overall retrieval rates ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("OVERALL RETRIEVAL RATES")
    print(f"{'─'*60}")
    print(f"  Overall: {trials['retrieved'].mean():.1%}")
    for tt in ['Scene', 'Object']:
        sub = trials[trials['trial_type'] == tt]
        print(f"  {tt:8s}: {sub['retrieved'].mean():.1%}  (n={len(sub)})")

    # ── 3. Compute statistics ────────────────────────────────────────────────
    rates = compute_retrieval_rates(trials)
    rt_stats = compute_rt_stats(trials)

    print(f"\n{'─'*60}")
    print("RETRIEVAL RATES BY TYPE × SEQUENCE POSITION")
    print(f"{'─'*60}")
    pivot = rates['by_type_seqpos'].pivot(
        index='seq_pos', columns='trial_type', values='mean')
    print(pivot.to_string(float_format='{:.0%}'.format))

    print(f"\n{'─'*60}")
    print("REACTION TIMES (retrieved trials only)")
    print(f"{'─'*60}")
    print(rt_stats['by_type'][['trial_type', 'mean', 'median', 'sem', 'count']]
          .to_string(float_format='{:.3f}'.format, index=False))

    if 'by_session_type' in rates:
        print(f"\n{'─'*60}")
        print("RETRIEVAL RATES BY SESSION × TYPE")
        print(f"{'─'*60}")
        pivot_ses = rates['by_session_type'].pivot(
            index='session_label', columns='trial_type', values='mean')
        print(pivot_ses.to_string(float_format='{:.0%}'.format))

    # ── 4. Figure 1: 4-panel overview (collapsed across sessions) ────────────
    OUT_DIR = str(step_plot_dir(__file__))

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle('Hierarchical Memory Retrieval — All Subjects',
                 fontsize=16, fontweight='bold', y=0.98)

    plot_retrieval_by_type(rates, ax=axes[0, 0], trials=trials)
    plot_retrieval_by_seqpos(rates, ax=axes[0, 1], trials=trials)
    plot_rt_by_type(rt_stats, ax=axes[1, 0], trials=trials)
    plot_rt_by_seqpos(rt_stats, ax=axes[1, 1], trials=trials)
    add_panel_labels(axes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, 'fig1_retrieval_overview.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_retrieval_overview.png'))
    print(f"\n✓ Saved fig1_retrieval_overview  →  {OUT_DIR}")

    # ── 5. Figure 2: Session comparison (evening vs morning) ─────────────────
    if 'by_session_type' in rates:
        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
        fig2.suptitle('Evening vs. Morning Retrieval',
                      fontsize=16, fontweight='bold', y=1.01)

        plot_session_comparison(rates, ax=axes2[0], trials=trials)
        plot_session_rt(rt_stats, ax=axes2[1], trials=trials)
        add_panel_labels(axes2.reshape(1, -1), ['a', 'b'])

        plt.tight_layout()
        fig2.savefig(os.path.join(OUT_DIR, 'fig2_session_comparison.pdf'))
        fig2.savefig(os.path.join(OUT_DIR, 'fig2_session_comparison.png'))
        print(f"✓ Saved fig2_session_comparison  →  {OUT_DIR}")

    # ── 6. Figure 3: Overall retrieval by session (collapsed across types) ───
    if trials['session_label'].nunique() > 1:
        fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
        plot_overall_retrieval_by_session(trials, ax=ax3)
        plt.tight_layout()
        fig3.savefig(os.path.join(OUT_DIR, 'fig3_overall_retrieval_by_session.pdf'))
        fig3.savefig(os.path.join(OUT_DIR, 'fig3_overall_retrieval_by_session.png'))
        print(f"✓ Saved fig3_overall_retrieval_by_session  →  {OUT_DIR}")

    # ── 7. LME: retrieval ~ session with random intercept per subject ────────
    import statsmodels.formula.api as smf

    # Convert pandas StringDtype columns to plain object dtype (patsy compat)
    for col in trials.columns:
        if pd.api.types.is_string_dtype(trials[col]):
            trials[col] = trials[col].astype(object)

    print(f"\n{'─'*60}")
    print("LME: retrieved ~ session_label  |  (1 | sub_id)")
    print(f"{'─'*60}")

    # Aggregate: percentage retrieved per subject per session
    trials['retrieved'] = trials['retrieved'].astype(int)
    sub_ses = (trials
               .groupby(['sub_id', 'session_label'])['retrieved']
               .mean()
               .reset_index()
               .rename(columns={'retrieved': 'pct_retrieved'}))
    sub_ses['pct_retrieved'] *= 100  # convert to percentage

    sub_ses['session_label'] = pd.Categorical(
        sub_ses['session_label'],
        categories=['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)'],
        ordered=True,
    )

    lme = smf.mixedlm(
        "pct_retrieved ~ C(session_label)",
        data=sub_ses,
        groups=sub_ses["sub_id"],
    )
    lme_result = lme.fit()
    print(lme_result.summary())
    STATS_LOG.append(str(lme_result.summary()))

    # ── 8. LME: reaction time ~ session with random intercept per subject ─────
    print(f"\n{'─'*60}")
    print("LME: mean_rt ~ session_label  |  (1 | sub_id)")
    print(f"{'─'*60}")

    # Aggregate: mean RT per subject per session (retrieved trials only)
    rt_trials = trials[trials['retrieved'] == 1].copy()
    sub_ses_rt = (rt_trials
                  .groupby(['sub_id', 'session_label'])['rt']
                  .mean()
                  .reset_index()
                  .rename(columns={'rt': 'mean_rt'}))

    sub_ses_rt['session_label'] = pd.Categorical(
        sub_ses_rt['session_label'],
        categories=['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)'],
        ordered=True,
    )

    lme_rt = smf.mixedlm(
        "mean_rt ~ C(session_label)",
        data=sub_ses_rt,
        groups=sub_ses_rt["sub_id"],
    )
    lme_rt_result = lme_rt.fit()
    print(lme_rt_result.summary())
    STATS_LOG.append(str(lme_rt_result.summary()))

    # ── 9. LME: retrieval ~ session * trial_type (Object vs Scene) ────────────
    print(f"\n{'─'*60}")
    print("LME: pct_retrieved ~ session_label * trial_type  |  (1 | sub_id)")
    print(f"{'─'*60}")

    # Aggregate: percentage retrieved per subject per session per trial_type
    sub_ses_type = (trials
                    .groupby(['sub_id', 'session_label', 'trial_type'])['retrieved']
                    .mean()
                    .reset_index()
                    .rename(columns={'retrieved': 'pct_retrieved'}))
    sub_ses_type['pct_retrieved'] *= 100

    sub_ses_type['session_label'] = pd.Categorical(
        sub_ses_type['session_label'],
        categories=['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)'],
        ordered=True,
    )

    lme_type = smf.mixedlm(
        "pct_retrieved ~ C(session_label) * C(trial_type)",
        data=sub_ses_type,
        groups=sub_ses_type["sub_id"],
    )
    lme_type_result = lme_type.fit()
    print(lme_type_result.summary())
    STATS_LOG.append(str(lme_type_result.summary()))

    # ── 10. LME: RT ~ session * trial_type (Object vs Scene) ─────────────────
    print(f"\n{'─'*60}")
    print("LME: mean_rt ~ session_label * trial_type  |  (1 | sub_id)")
    print(f"{'─'*60}")

    # Aggregate: mean RT per subject per session per trial_type
    sub_ses_type_rt = (rt_trials
                       .groupby(['sub_id', 'session_label', 'trial_type'])['rt']
                       .mean()
                       .reset_index()
                       .rename(columns={'rt': 'mean_rt'}))

    sub_ses_type_rt['session_label'] = pd.Categorical(
        sub_ses_type_rt['session_label'],
        categories=['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)'],
        ordered=True,
    )

    lme_type_rt = smf.mixedlm(
        "mean_rt ~ C(session_label) * C(trial_type)",
        data=sub_ses_type_rt,
        groups=sub_ses_type_rt["sub_id"],
    )
    lme_type_rt_result = lme_type_rt.fit()
    print(lme_type_rt_result.summary())
    STATS_LOG.append(str(lme_type_rt_result.summary()))

    # ── 11. RT by sequence position, separately per session ──────────────────
    # (added 2026-11-18; not part of the original script)
    from scipy import stats as sstats

    session_order = [lab for lab in ['Evening (S0)', 'Morning 1 (S1)', 'Morning 2 (S2)']
                     if lab in set(rt_trials['session_label'])]
    bool_trials = trials.copy()
    bool_trials['retrieved'] = bool_trials['retrieved'].astype(bool)

    fig4, axes4 = plt.subplots(1, len(session_order) + 1,
                               figsize=(4.2 * (len(session_order) + 1), 4.6))
    fig4.suptitle('Reaction time by sequence position, per session',
                  fontsize=16, fontweight='bold', y=1.02)
    for ax, lab in zip(axes4, session_order):
        ses_trials = bool_trials[bool_trials['session_label'] == lab]
        n_sub = ses_trials.loc[ses_trials['retrieved'], 'sub_id'].nunique()
        plot_rt_by_seqpos(compute_rt_stats(ses_trials), ax=ax,
                          title=f"{lab.split(' (')[0]} (N = {n_sub})", trials=ses_trials)
    # same y-axis in the three session panels so they can be compared directly
    ymax = max(ax.get_ylim()[1] for ax in axes4[:len(session_order)])
    ymin = min(ax.get_ylim()[0] for ax in axes4[:len(session_order)])
    for ax in axes4[:len(session_order)]:
        ax.set_ylim(ymin, ymax)

    slopes = rt_position_slopes(bool_trials)
    plot_rt_slope_by_session(slopes, session_order, ax=axes4[-1])
    add_panel_labels(axes4.reshape(1, -1), list('abcd')[:len(axes4)])
    plt.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, 'fig4_rt_by_position_per_session.pdf'))
    fig4.savefig(os.path.join(OUT_DIR, 'fig4_rt_by_position_per_session.png'))
    print(f"✓ Saved fig4_rt_by_position_per_session  →  {OUT_DIR}")

    # Tables: group means per session x type x position, and the slopes
    means_tbl = _subject_then_group(
        bool_trials[bool_trials['retrieved']], ['session_label', 'trial_type', 'seq_pos'], 'rt')
    means_tbl.to_csv(step_output_dir(__file__) / 'rt_by_position_by_session_means.csv', index=False)
    slopes.to_csv(step_output_dir(__file__) / 'rt_position_slopes_per_participant.csv', index=False)

    pos_log = []
    pos_log.append("RT slope over sequence position (seconds per position), per participant.\n"
                   "One-sample t-test of the slopes against 0, per session and trial type.\n")
    for tt in ['Scene', 'Object']:
        for lab in session_order:
            v = slopes[(slopes['trial_type'] == tt) & (slopes['session_label'] == lab)][
                'rt_slope_s_per_position'].dropna()
            if len(v) < 3:
                continue
            t, p_val = sstats.ttest_1samp(v, 0)
            pos_log.append(f"  {tt:6s} {lab:15s} N = {len(v):2d}  mean slope = {v.mean():+.3f} s "
                           f"(SEM {v.sem():.3f})  t = {t:+.2f}  p = {p_val:.4g}")
    pos_log.append("\nPaired t-tests of the slopes between sessions (participants with both sessions).\n")
    for tt in ['Scene', 'Object']:
        wide = slopes[slopes['trial_type'] == tt].pivot(
            index='sub_id', columns='session_label', values='rt_slope_s_per_position')
        for i in range(len(session_order)):
            for j in range(i + 1, len(session_order)):
                a, b = session_order[i], session_order[j]
                if a not in wide or b not in wide:
                    continue
                both = wide[[a, b]].dropna()
                if len(both) < 3:
                    continue
                t, p_val = sstats.ttest_rel(both[b], both[a])
                pos_log.append(f"  {tt:6s} {b} minus {a}: N = {len(both):2d}  "
                               f"mean difference = {(both[b] - both[a]).mean():+.3f} s  "
                               f"t = {t:+.2f}  p = {p_val:.4g}")

    # Mixed model on participant cell means: does the position effect differ by session?
    cell = (rt_trials.dropna(subset=['rt', 'seq_pos'])
            .groupby(['sub_id', 'session_label', 'trial_type', 'seq_pos'])['rt']
            .mean().reset_index().rename(columns={'rt': 'mean_rt'}))
    cell['seq_pos'] = cell['seq_pos'].astype(float)
    cell['session_label'] = pd.Categorical(cell['session_label'], categories=session_order)
    for tt in ['Scene', 'Object']:
        d = cell[cell['trial_type'] == tt]
        print(f"\n{'─'*60}")
        print(f"LME ({tt} trials): mean_rt ~ seq_pos * session_label  |  (1 | sub_id)")
        print(f"{'─'*60}")
        res = smf.mixedlm("mean_rt ~ seq_pos * C(session_label)", data=d, groups=d["sub_id"]).fit()
        print(res.summary())
        pos_log.append(f"\nLME ({tt} trials): mean_rt ~ seq_pos * session_label  |  (1 | sub_id)\n"
                       f"Data: participant mean RT per session and position. `seq_pos` is the\n"
                       f"slope in the evening session; `seq_pos:C(session_label)[...]` is how much\n"
                       f"the slope differs in that morning session.\n{res.summary()}")
    print("\n".join(pos_log[:2 + 2 * len(session_order) + 8]))
    with open(step_output_dir(__file__) / 'rt_by_position_by_session.txt', 'w') as fh:
        fh.write("\n".join(pos_log) + "\n")

    print("\n✓ Analysis complete.")

    stats_file = step_output_dir(__file__) / "lme_session_models.txt"
    titles = ["pct_retrieved ~ session", "mean_rt ~ session",
              "pct_retrieved ~ session * trial_type", "mean_rt ~ session * trial_type"]
    with open(stats_file, "w") as fh:
        for title, summary in zip(titles, STATS_LOG):
            fh.write(f"LME: {title}  |  (1 | sub_id)\n{summary}\n\n")
    print(f"\nA5 done: figures in {OUT_DIR}, model summaries in {stats_file}")
