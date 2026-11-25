#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E15 - Learning, one estimate per sequence and showing: does a sequence whose
      medial temporal activity falls more from the first to the second showing
      get retrieved better and faster than the participant's other sequences?

E9 gives one estimate per showing over all sequences, and E14 relates its change
to behaviour ACROSS participants. That relation can reflect a trait of the
participant. This step tests it WITHIN participants, across the ten sequences
(scenes) each participant learns.

Participant mode. Data, regions and nuisance terms as E9 (evening learning runs,
T1w space, not smoothed, percent signal change, mean signal per FreeSurfer region
of B1; fixation is the only unmodelled time). Two models, each with ONE regressor
per sequence trial (a sequence at one showing):
    block    scene onset until the rating (the whole sequence); plus context, rating
    objects  the five object events of the sequence; plus context, the first
             appearance of the scene (all sequences), the scene alone between
             objects (all sequences), rating
The first and second showing of every sequence are in the same run, the third in
the next. Behaviour per sequence, from the object retrieval trials of the evening
and of the first and the repeated morning retrieval: retrieval success (share of
the sequence's five objects with a button press) and mean reaction time (response
prompt to press, trials with a press).

Group mode. Change = second minus first showing. For every region and model:
    slope      per participant, the slope of the behaviour on the change across
               sequences (participants in whose data the behaviour varies across
               sequences); tested against zero across participants
    mixed      mixed model over all sequences with the change centred within
               participant and a random intercept per participant
    complete   mean change of the sequences retrieved completely minus that of the
               others, per participant (participants with both kinds); tested
               against zero
    scene      the mixed model with the identity of the scene as a further factor
               (a scene that is easy to remember could produce both a large change
               and good retrieval)
    showings   the mixed model with the activity at the first and at the second
               showing as two predictors in place of their difference

Output  outputs/E15_learning_by_sequence/participants/<sub>_sequence_estimates.csv
        (subject, model, scene, showing, region, beta)
        outputs/E15_learning_by_sequence/participants/<sub>_sequence_behaviour.csv
        outputs/E15_learning_by_sequence/group_within_participant.csv, sequence_table.csv
Plots   plots/E15_learning_by_sequence/learning_vs_behaviour_within_participants.{png,pdf}
        (one line per participant: that participant's fit across its sequences)
Usage:  python E15_learning_by_sequence.py <subject>
        python E15_learning_by_sequence.py group

The morning session of s41 was started under another participant's number; since the afternoon of
2026-11-25 it is used with the retrieved items taken from the cue (config.RELABELLED_SESSIONS), and the
outputs include s41's morning behaviour.

Main version (decided 2026-11-25): the whole sequence (model `block`); the figure shows it. The
objects-only model is kept in the tables for comparison.

Not a port of the original pipeline: added 2026-11-25. Exploratory.
"""

from __future__ import annotations

import glob
import os
import sys
from os.path import join as opj
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import E9_encoding_regions as e9
from config import step_input_dir, step_output_dir, step_plot_dir
from decoding_utils import CONFOUND_VARS, build_sequence_dict_from_df_behavior_enc, parse_subject_arg

d1 = e9.d1
dec = e9.e3.dec
MODELS = ("block", "objects")
REGIONS = ["HPC", "PHPC", "ERH", "MTL", "PRECUNEUS", "POSTERIOR_CINGULATE", "RSC", "VIS", "CAUDAL_ACC"]
N_PER_SEQUENCE = 5
SCATTER_REGIONS = {"PHPC": "#2a78d6", "HPC": "#eb6834", "MTL": "#1baf7a", "PRECUNEUS": "#eda100", "VIS": "#e87ba4"}   # as E14
LABELS = {"PHPC": "Parahippocampal", "HPC": "Hippocampus", "MTL": "MTL", "PRECUNEUS": "Precuneus", "VIS": "Visual"}


def figure_within(t, res, model="block"):          # main version: the whole sequence
    """One line per participant (fit across its sequences, both axes centred within participant) and the mean slope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(2, len(SCATTER_REGIONS), figsize=(3.3 * len(SCATTER_REGIONS), 6.6), sharey="row")
    for i, (beh, ylab) in enumerate((("success_pre", "Retrieval success (evening),\nrelative to the participant's mean"),
                                     ("rt_pre", "Reaction time (s, evening),\nrelative to the participant's mean"))):
        for ax, (reg, colour) in zip(axes[i], SCATTER_REGIONS.items()):
            d = t[(t.model == model) & (t.region == reg)].dropna(subset=["change", beh])
            slopes, xmax = [], 0.0
            for _, s_ in d.groupby("subject"):
                if len(s_) < 6 or s_[beh].std() == 0 or s_.change.std() == 0:
                    continue
                x, y = s_.change - s_.change.mean(), s_[beh] - s_[beh].mean()
                k = np.polyfit(x, y, 1)[0]
                slopes.append(k)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, k * xs, color=colour, lw=1.0, alpha=0.45, zorder=2)
                xmax = max(xmax, np.abs(x).quantile(0.9))
            xs = np.array([-xmax, xmax])
            ax.plot(xs, np.mean(slopes) * xs, color=ink, lw=2.6, zorder=4)
            r = res[(res.model == model) & (res.region == reg) & (res.behaviour == beh)].iloc[0]
            ax.text(0.03, 0.04, f"t = {r.slope_t:.1f}, p = " + f"{r.slope_p:.3f}".replace("0.", "."), transform=ax.transAxes,
                    fontsize=9, color=ink, zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
            ax.axhline(0, color=muted, lw=0.6, ls=":", zorder=1); ax.axvline(0, color=muted, lw=0.6, ls=":", zorder=1)
            ax.set_xlim(-1.6 * xmax, 1.6 * xmax)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"{LABELS[reg]}  (n = {len(slopes)})", fontsize=11, loc="left", color=ink)
            else:
                ax.set_title(f"n = {len(slopes)}", fontsize=9, loc="left", color=muted)
                ax.set_xlabel("Change, 2nd minus 1st showing (%),\nrelative to the participant's mean", fontsize=9)
        axes[i, 0].set_ylabel(ylab, fontsize=9)
        lim = 0.25 if i == 0 else 1.2
        axes[i, 0].set_ylim(-lim, lim)
    fig.suptitle("Within participants: one coloured line per participant (fit across that participant's ten sequences); "
                 "black: mean slope", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(opj(str(step_plot_dir(__file__)), f"learning_vs_behaviour_within_participants.{ext}"), dpi=300)
    plt.close(fig)


def design(g, n, model):
    """Columns, and for the per-sequence columns their (scene, showing)."""
    cols, keys = [], []

    def add(on, end, key=None):
        on, end = np.atleast_1d(on).astype(float), np.atleast_1d(end).astype(float)
        ok = end > on
        if ok.any():
            cols.append(d1.hrf_regressor(on[ok], (end - on)[ok], n)); keys.append(key)

    is_obj = g.label_type == "dn+sc+obj_cue"
    is_sc = g.label_type == "dn+sc_cue"
    first_sc = is_sc & (g.pos == 0)
    sel = g.label_type == "dn_cue"
    add(g.t_onset[sel].values, g.end[sel].values)
    if model == "objects":
        add(g.t_onset[first_sc].values, g.end[first_sc].values)
        sel = is_sc & (g.pos > 0)
        add(g.t_onset[sel].values, g.end[sel].values)
    for seq, s in g.groupby("seq"):
        scene = s.sc_enc_stim.dropna()
        rating = s[s.label_type == "slider"]
        if scene.empty or rating.empty or pd.isna(s.rep.iloc[0]) or not (s.label_type == "dn+sc+obj_cue").any():
            continue
        key = (str(scene.iloc[0]), int(s.rep.iloc[0]))
        if model == "block":
            add(s.t_onset[first_sc.loc[s.index]].iloc[0], rating.t_onset.iloc[0], key)
        else:
            o = s[is_obj.loc[s.index]]
            add(o.t_onset.values, o.end.values, key)
    sel = g.label_type.isin(["slider", "slider_resp"])
    add(g.t_onset[sel].values, g.end[sel].values)
    sel = g.label_type.isin(["instr_txt", "break_text", "rest_phase"])
    add(g.t_onset[sel].values, g.end[sel].values)
    return cols, keys


def sequence_behaviour(sub):
    path_loc = str(step_input_dir("A2_add_tr_columns_localizer"))
    path_main = str(step_input_dir("A1_add_tr_columns_main"))
    _, o2c = dec.load_localizer_behavior(path_loc, sub)
    df_enc, _ = dec.load_main_task_behavior(path_main, sub, dec.SESSION_CONFIG["presleep"])
    hierarchy = build_sequence_dict_from_df_behavior_enc(df_enc)
    rows = []
    for session in d1.SESSIONS:
        try:
            ev, _ = d1.retrieval_events(sub, session, hierarchy, o2c, path_main)
        except Exception as err:
            print(f"{session} not usable: {err}")
            continue
        ev = ev[(ev.kind == "object") & (ev.run_idx >= 0)].copy()
        ev["cond"] = "pre" if session == "presleep" else np.where(ev.run_idx < 3, "post1", "post2")
        for (cond, scene), t in ev.groupby(["cond", "scene"]):
            rows.append(dict(subject=sub, cond=cond, scene=str(scene), n_trials=len(t), success=t.rt.notna().mean(),
                             rt=t.rt.astype(float).mean()))
    return pd.DataFrame(rows)


def participant(sub):
    sub_mri = f"sub-{sub[1:]}"
    print(f"=== {sub} ===", flush=True)
    out = str(step_output_dir(__file__, "participants"))
    sequence_behaviour(sub).to_csv(opj(out, f"{sub}_sequence_behaviour.csv"), index=False)
    ev = e9.encoding_events(sub)
    funcs, confs = d1.session_files(sub_mri, "ses-04")
    for i in (0, 2):
        if nib.load(funcs[i]).shape[-1] != 120:
            raise ValueError(f"BOLD run index {i} is not a 120-volume run; run mapping may be wrong for {sub}")
    use = {run: e9.RUN_MAPPING[run] for run in ev}
    mask = masking.intersect_masks([funcs[i].replace("desc-preproc_bold", "desc-brain_mask") for i in use.values()],
                                   threshold=1, connected=False)
    bm = np.asarray(mask.dataobj).astype(bool)
    regions = d1.anatomical_regions(sub_mri, mask, bm)
    names_r = [r for r in REGIONS if r in regions]
    rows = []
    for run, fi in use.items():
        g = ev[run]
        print(f"  loading {os.path.basename(funcs[fi])}", flush=True)
        y = masking.apply_mask(funcs[fi], mask).astype(np.float32)
        conf = pd.read_csv(confs[fi], sep="\t")[CONFOUND_VARS].fillna(0).values
        n = min(y.shape[0], int(np.ceil((g.end.max() + e9.TAIL_S) / d1.TR)))
        y, conf = y[:n], conf[:n]
        m = y.mean(axis=0); m[m <= 0] = np.inf
        y = (y / m - 1.0) * 100.0
        ts = np.column_stack([y[:, regions[r]].mean(axis=1) for r in names_r])
        N = d1.nuisance_matrix(n, conf, pd.DataFrame(columns=["name", "onset_s", "duration"]))
        for model in MODELS:
            cols, keys = design(g, n, model)
            b = np.linalg.lstsq(np.column_stack(cols + [N]), ts, rcond=None)[0]
            k = [x for x in keys if x is not None]
            print(f"  run {run} {model}: {len(k)} sequence regressors, {n} volumes, {len(cols) + N.shape[1]} columns", flush=True)
            for row, key in zip(b, keys):
                if key is not None:
                    rows += [dict(subject=sub, model=model, scene=key[0], showing=key[1], region=reg, beta=float(row[i]))
                             for i, reg in enumerate(names_r)]
    pd.DataFrame(rows).to_csv(opj(out, f"{sub}_sequence_estimates.csv"), index=False)
    print("done")


def _tt(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 8:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    r = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=r.statistic, p=r.pvalue)


def group():
    import statsmodels.formula.api as smf
    out = str(step_output_dir(__file__))
    est = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_sequence_estimates.csv")))])
    beh = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(opj(out, "participants", "*_sequence_behaviour.csv")))])
    w = est.pivot_table(index=["subject", "model", "region", "scene"], columns="showing", values="beta").reset_index()
    w["change"] = w[2] - w[1]
    w = w.rename(columns={1: "showing1", 2: "showing2", 3: "showing3"})
    b = beh.pivot_table(index=["subject", "scene"], columns="cond", values=["success", "rt"])
    b.columns = [f"{a}_{c}" for a, c in b.columns]
    t = w.merge(b.reset_index(), on=["subject", "scene"], how="inner")
    t.to_csv(opj(out, "sequence_table.csv"), index=False)
    print(f"{t.subject.nunique()} participants, {t.groupby('subject').scene.nunique().median():.0f} sequences per participant (median)")
    rows = []
    for (model, reg), g in t.groupby(["model", "region"]):
        for beh_col in ("success_pre", "rt_pre", "success_post1", "rt_post1"):
            if beh_col not in g:
                continue
            d = g.dropna(subset=["change", beh_col]).copy()
            slopes = []
            for _, s in d.groupby("subject"):
                if len(s) >= 6 and s[beh_col].std() > 0 and s.change.std() > 0:
                    slopes.append(np.polyfit(s.change, s[beh_col], 1)[0])
            row = dict(model=model, region=reg, behaviour=beh_col, n_sequences=len(d),
                       **{f"slope_{k}": v for k, v in _tt(slopes).items()})
            d["change_c"] = d.change - d.groupby("subject").change.transform("mean")
            try:
                fit = smf.mixedlm(f"{beh_col} ~ change_c", d, groups=d["subject"]).fit(reml=True)
                row.update(mixed_beta=fit.params["change_c"], mixed_z=fit.tvalues["change_c"], mixed_p=fit.pvalues["change_c"])
                for col in ("showing1", "showing2"):
                    d[col + "_c"] = d[col] - d.groupby("subject")[col].transform("mean")
                d["scene_id"] = d.scene.astype(str).str.replace(".", "_", regex=False)
                f1 = smf.mixedlm(f"{beh_col} ~ change_c + C(scene_id)", d, groups=d["subject"]).fit(reml=True)
                f2 = smf.mixedlm(f"{beh_col} ~ showing1_c + showing2_c", d, groups=d["subject"]).fit(reml=True)
                row.update(scene_z=f1.tvalues["change_c"], scene_p=f1.pvalues["change_c"],
                           showing1_z=f2.tvalues["showing1_c"], showing1_p=f2.pvalues["showing1_c"],
                           showing2_z=f2.tvalues["showing2_c"], showing2_p=f2.pvalues["showing2_c"])
            except Exception as err:
                print(f"mixed model failed for {model} {reg} {beh_col}: {err}")
            if beh_col.startswith("success"):
                full = d[beh_col] >= 1.0
                diff = d[full].groupby("subject").change.mean() - d[~full].groupby("subject").change.mean()
                row.update({f"complete_{k}": v for k, v in _tt(diff.dropna()).items()})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(opj(out, "group_within_participant.csv"), index=False)
    figure_within(t, res)
    print(f"-> {out}")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group()
    else:
        participant(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None))
