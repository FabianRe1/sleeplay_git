#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D3 - Position code, neighbours within a scene, scene order, and a probe-order check.

All analyses use object x object trial pairs from different runs, with the
dissimilarities of D2 (patterns centred per run and trial type, 1 - Pearson r,
z-scored within the pair set).

A  Graded position code. On object trials the cue shows the scene picture and
   the position digit, so "same position" pairs share the digit. Pairs at
   different positions never share it, so a graded effect among them cannot come
   from an identical digit.
       z(dissimilarity) ~ context + scene + category + same_position + distance
     same_position : 0 same digit, 1 different
     distance      : |pos_i - pos_j| centred among different-position pairs, 0 for
                     same-position pairs. Positive beta = further apart, less similar.
   Also: mean z-dissimilarity per ordinal distance (0-4) for pairs within a scene
   and pairs from different scenes, and the neighbour advantage (distance >= 2
   minus distance 1) for pairs in the same scene, in other scenes of the same
   context, and in the other context.
B  Scene order. The scene's position within its context is not shown on object
   trials. Added to the model of A:
     so_within : |scene position_i - scene position_j| among different scenes of
                 the same context, centred there, 0 elsewhere
     so_across : the same distance among pairs from different contexts
C  Neighbours within a scene. Neighbour advantage for pairs of the same scene
   minus the same measure for pairs of other scenes ("diff"); the same for trials
   with reported retrieval in both ("diff_correct"); and the regression
       z(dissim) ~ context + scene + distance + distance x same_scene  ("slope_x_scene").
   Tested over all six pair sets, the three within-session sets, the three
   between-session sets, evening, morning, and morning minus evening.
D  Probe order. For object trials of the same scene in different runs: how many
   trials apart were direct neighbours and more distant objects probed? Uses the
   trial tables only.
E  Controls for the position code of A (added 2026-11-23). Response time differs
   by position (about 1.3 s at position 1, 2.2 s at positions 3 and 4), and 11 to
   23 % of trials have no response. The model of A is refitted
     answered        only pairs in which both trials have a response
     rt              answered pairs, plus |response time_i - response time_j| (`rt_diff`)
     profile         all pairs, plus the difference between the participant's MEAN
                     response times at the two positions (`rt_profile`; 0 for
                     same-position pairs). Stricter than `rt`: the trial-level
                     difference is a noisy measure of how positions differ.
     rt_profile      answered pairs, with both covariates
     object_x_scene  object x scene pairs: context + scene + same_digit + distance
                     between the object's position and the scene's position. Both
                     trial types show a digit with different meanings, so an effect
                     here is a digit or number effect, not an object-position code.
   Covariates are z-scored; a positive beta means similar response times go with
   similar patterns. A control is skipped for a participant and pair set with fewer
   than MIN_PAIRS_SUBSET pairs.
F  Position distance ACROSS sequences (added 2026-11-23). One model matrix per
   trial type: the distance between two positions (1 to 4 steps), defined only for
   pairs that (a) are at DIFFERENT positions, so they never show the same digit,
   and (b) do not belong to the same sequence, so they share no scene picture
   (objects) or no context cue (scenes). All other pairs are left out.
     objects_other_scene           object x object, different scenes        (+ context, category sequence)
     objects_other_context         object x object, different contexts      (+ category sequence)
     scenes_other_context          scene x scene, different contexts
     scenes_same_context           scene x scene, same context (shares the context cue; for comparison)
     object_x_scene_other_context  object x scene, different contexts: object position
                                   against scene position, i.e. the digit at two levels
   Each also with the response-time difference of the two trials as covariate
   (`*_rt`, answered pairs). Model `distance`: positive beta = further apart, less
   similar. Also the mean z-dissimilarity per distance. What this cannot exclude:
   neighbouring positions show neighbouring digits.
   Shape of the effect, on the same pairs: two model matrices with one parameter each,
     gradient   1, 2, 3, 4 steps apart (the further apart, the less similar)
     step       0 for adjacent positions, 1 for all others
   `gradient_r` and `step_r` are each model's partial correlation with the
   dissimilarity (nuisance removed from both); F_group_model_comparison.csv tests
   them against zero and against each other (Fisher z, paired). `joint_step` and
   `joint_beyond` are the betas of one regression with the step and, next to it,
   the distance among non-adjacent pairs (2, 3, 4, centred): does distance matter
   beyond the step?
G  Hierarchy distance without a shared picture or digit (added 2026-11-23). Distance
   in the tree context > scene > object between two trials:
     1  an object trial and the scene trial that retrieves the object's scene
     2  two trials of the same context, different scenes
     3  two trials of different contexts
   Left out: two object trials of the same scene (same picture), every pair with the
   same digit, and all scene x scene pairs (same scene = identical cue; the others
   differ only in the context cue). Positive beta = closer in the tree, more similar.
     hierarchy_all       object x object and object x scene pairs, + pair type; levels 1-3
     os_scene_step       object x scene, level 1 against 2. Both levels share the context
                         cue, the scene picture is on screen in one trial only: the test
                         that the cue does not explain.
     os_context_step     object x scene, level 2 against 3
     oo_context_step     object x object, level 2 against 3
   The context is shown on every trial, so the step from 2 to 3 is confounded with a
   shared context cue; no pair of this task removes that.

Input   outputs/D1_trial_patterns/patterns/fixed/<session>/
Output  outputs/D3_position_and_neighbours/
          A_subject_betas.csv, A_group_beta_vs_zero.csv            (models distance, same_position, so_within, so_across, ...)
          A_subject_distance_curve.csv, A_group_distance_curve.csv
          A_subject_neighbour.csv, A_group_neighbour.csv
          C_subject_neighbour_scene.csv, C_group_neighbour_scene.csv
          D_probe_order_by_subject.csv, D_probe_order_summary.csv
          E_subject_controls.csv, E_group_controls.csv             (control x model; pooled = all_sessions)
          F_subject_across.csv, F_group_across.csv, F_subject_across_curve.csv, F_group_across_curve.csv,
          F_group_model_comparison.csv
          G_subject_hierarchy.csv, G_group_hierarchy.csv, G_group_hierarchy_levels.csv

Usage:  python D3_position_and_neighbours.py [n_jobs]

Not a port of the original pipeline: added 2026-11-19, ported from
7_rsa_exploratory/05_position_distance.py, 06_followups.py (part A) and
07_probe_order_check.py; logic unchanged. Part E was added on 2026-11-23, together
with the minimum number of pairs for the analyses restricted to trials with
reported retrieval in C (`*_correct`); tables A and D and the other columns of C
are unchanged. Part F was added the same day.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from os.path import join as opj
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import step_output_dir

_spec = importlib.util.spec_from_file_location("d2_binary_rsa", Path(__file__).with_name("D2_binary_rsa.py"))
rsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsa)

SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
        ("pre", "post1"), ("pre", "post2"), ("post1", "post2")]
COLS = ["context", "scene", "category", "same_position", "distance"]
GROUPS = {"all_sets": ["pre", "post1", "post2", "prexpost1", "prexpost2", "post1xpost2"],
          "within_session": ["pre", "post1", "post2"],
          "between_session": ["prexpost1", "prexpost2", "post1xpost2"],
          "evening": ["pre"], "morning": ["post1", "post2"]}
MIN_PAIRS_SUBSET = rsa.MIN_PAIRS_SUBSET      # fewest pairs for an analysis restricted to a subset of trials
MIN_PAIRS_ACROSS = 30                        # part F: scene x scene sets have only 40 to 55 usable pairs per time point


def _z(v):
    return (v - v.mean()) / v.std()


def controls(sub, roi, name, ta, tb, pairs, y_all, same_set):
    """Part E: the position model of A with response-time controls, and for object x scene pairs."""
    rows = []
    ia, ib = pairs.ia.values, pairs.ib.values
    dist = np.abs(ta.position.values[ia] - tb.position.values[ib]).astype(float)
    between = (pairs.same_run == 0).values & np.isfinite(y_all) & np.isfinite(dist)
    oo = between & (pairs.pairtype == "oo").values
    drt = pairs.rt_diff.values
    answered = oo & np.isfinite(drt)
    # the participant's mean response time per object position, over the object trials of this pair set
    obj = ta[ta.kind == "object"] if same_set else pd.concat([ta[ta.kind == "object"], tb[tb.kind == "object"]])
    prof = obj.groupby("position").rt.mean()
    dprof = np.abs(ta.position.map(prof).values[ia] - tb.position.map(prof).values[ib]).astype(float)
    have_prof = len(prof) == 5 and np.isfinite(prof.values).all()

    def fit(control, sel, extra, digit_name="same_position"):
        if sel.sum() < MIN_PAIRS_SUBSET:
            return
        p, d = pairs[sel], dist[sel]
        diff = d > 0
        lin = np.where(diff, d - d[diff].mean(), 0.0)
        cols, names = [p.context, p.scene], ["context", "scene"]
        if control != "object_x_scene":
            cols.append(p.category); names.append("category")
        cols += [diff.astype(float), lin]; names += [digit_name, "distance"]
        for cname, c in extra:
            cols.append(_z(c[sel])); names.append(cname)
        b, _, n = rsa.regress(y_all[sel], np.column_stack(cols))
        rows.extend(dict(subject=sub, roi=roi, pair_set=name, control=control, model=m, beta=v, n_pairs=n)
                    for m, v in zip(names, b) if m in (digit_name, "distance", "rt_diff", "rt_profile"))

    fit("answered", answered, [])
    fit("rt", answered, [("rt_diff", drt)])
    if have_prof:
        fit("profile", oo, [("rt_profile", dprof)])
        fit("rt_profile", answered, [("rt_diff", drt), ("rt_profile", dprof)])
    fit("object_x_scene", between & (pairs.pairtype == "os").values, [], digit_name="same_digit")
    return rows


def across(sub, roi, name, ta, tb, pairs, y_all):
    """Part F: position distance among pairs at different positions that do not share a sequence."""
    rows, curve = [], []
    ia, ib = pairs.ia.values, pairs.ib.values
    dist = np.abs(ta.position.values[ia] - tb.position.values[ib]).astype(float)
    ok = (pairs.same_run == 0).values & np.isfinite(y_all) & (dist >= 1)
    pt, ctx, scn = pairs.pairtype.values, pairs.context.values, pairs.scene.values
    drt = pairs.rt_diff.values
    sets = {"objects_other_scene": ((pt == "oo") & (scn == 1), ["context", "catseq"]),
            "objects_other_context": ((pt == "oo") & (ctx == 1), ["catseq"]),
            "scenes_other_context": ((pt == "ss") & (ctx == 1), []),
            "scenes_same_context": ((pt == "ss") & (ctx == 0) & (scn == 1), []),
            "object_x_scene_other_context": ((pt == "os") & (ctx == 1), [])}
    for label, (m, nuisance) in sets.items():
        for tag, sel in (("", m & ok), ("_rt", m & ok & np.isfinite(drt))):
            if sel.sum() < MIN_PAIRS_ACROSS:
                continue
            cols = [pairs[c].values[sel] for c in nuisance] + [dist[sel] - dist[sel].mean()]
            if tag:
                cols.append(_z(drt[sel]))
            b, _, n = rsa.regress(y_all[sel], np.column_stack(cols))
            rows.append(dict(subject=sub, roi=roi, pair_set=name, pairs=label + tag, model="distance",
                             beta=b[len(nuisance)], n_pairs=n))
            # gradient against step: partial correlations and the joint regression
            dd = dist[sel]
            far = dd >= 2
            beyond = np.where(far, dd - dd[far].mean(), 0.0)
            nuis = [pairs[c].values[sel] for c in nuisance] + ([_z(drt[sel])] if tag else [])
            N = np.column_stack([np.ones(sel.sum())] + nuis)
            resid = lambda v: v - N @ np.linalg.lstsq(N, v, rcond=None)[0]
            yr = resid(_z(y_all[sel]))
            out = {"gradient_r": np.corrcoef(yr, resid(dd))[0, 1], "step_r": np.corrcoef(yr, resid(far.astype(float)))[0, 1]}
            bj, _, _ = rsa.regress(y_all[sel], np.column_stack(nuis + [far.astype(float), beyond]))
            out["joint_step"], out["joint_beyond"] = bj[-2], bj[-1]
            rows.extend(dict(subject=sub, roi=roi, pair_set=name, pairs=label + tag, model=m, beta=v, n_pairs=n)
                        for m, v in out.items())
            if not tag:
                z = _z(y_all[sel])
                for k in range(1, 5):
                    mm = dist[sel] == k
                    if mm.sum() >= 5:
                        curve.append(dict(subject=sub, roi=roi, pair_set=name, pairs=label, distance=k,
                                          z_dissim=z[mm].mean(), n_pairs=int(mm.sum())))
    return rows, curve


def hierarchy(sub, roi, name, ta, tb, pairs, y_all):
    """Part G: tree distance among pairs that share neither a scene picture nor a digit."""
    rows, levels = [], []
    ia, ib = pairs.ia.values, pairs.ib.values
    dist = np.abs(ta.position.values[ia] - tb.position.values[ib]).astype(float)
    ok = (pairs.same_run == 0).values & np.isfinite(y_all) & (dist >= 1)          # different digit
    pt, ctx, scn = pairs.pairtype.values, pairs.context.values, pairs.scene.values
    h = np.where(ctx == 1, 3.0, np.where(scn == 1, 2.0, 1.0))
    oo, os_ = ok & (pt == "oo") & (scn == 1), ok & (pt == "os")
    is_os = (pt == "os").astype(float)
    fits = {"hierarchy_all": (oo | os_, [is_os, h]),
            "os_scene_step": (os_ & (h <= 2), [(h == 2).astype(float)]),
            "os_context_step": (os_ & (h >= 2), [(h == 3).astype(float)]),
            "oo_context_step": (oo, [(h == 3).astype(float)])}
    for model, (sel, cols) in fits.items():
        if sel.sum() < MIN_PAIRS_ACROSS:
            continue
        b, _, n = rsa.regress(y_all[sel], np.column_stack([c[sel] for c in cols]))
        rows.append(dict(subject=sub, roi=roi, pair_set=name, model=model, beta=b[-1], n_pairs=n))
    for label, sel in (("object_x_scene", os_), ("object_x_object", oo)):
        if sel.sum() < MIN_PAIRS_ACROSS:
            continue
        z = _z(y_all[sel])
        for k in (1, 2, 3):
            mm = h[sel] == k
            if mm.sum() >= 5:
                levels.append(dict(subject=sub, roi=roi, pair_set=name, pairs=label, level=k,
                                   z_dissim=z[mm].mean(), n_pairs=int(mm.sum())))
    return rows, levels


def one(sub, roi):
    sets = rsa.load_sets(sub, roi)
    betas, curve, neigh, scene_rows, control_rows, across_rows, across_curve = [], [], [], [], [], [], []
    hier_rows, hier_levels = [], []
    for sa, sb in SETS:
        if sa not in sets or sb not in sets:
            continue
        (ta, pa), (tb, pb) = sets[sa], sets[sb]
        pairs = rsa.pair_table(ta, tb, sa == sb)
        d = rsa.corr_distance(rsa.trial_vectors(ta, pa), rsa.trial_vectors(tb, pb))
        y = d[pairs.ia.values, pairs.ib.values]
        control_rows += controls(sub, roi, sa if sa == sb else f"{sa}x{sb}", ta, tb, pairs, y, sa == sb)
        ar, ac = across(sub, roi, sa if sa == sb else f"{sa}x{sb}", ta, tb, pairs, y)
        across_rows += ar
        across_curve += ac
        hr, hl = hierarchy(sub, roi, sa if sa == sb else f"{sa}x{sb}", ta, tb, pairs, y)
        hier_rows += hr
        hier_levels += hl
        sel = ((pairs.pairtype == "oo") & (pairs.same_run == 0) & np.isfinite(y)).values
        p, y = pairs[sel].reset_index(drop=True), y[sel]
        if len(y) < 50:
            continue
        y = (y - y.mean()) / y.std()
        dist = np.abs(ta.position.values[p.ia.values] - tb.position.values[p.ib.values]).astype(float)
        diff = dist > 0
        lin = np.where(diff, dist - dist[diff].mean(), 0.0)
        X = np.column_stack([p.context, p.scene, p.category, diff.astype(float), lin])
        b, _, n = rsa.regress(y, X)
        name = sa if sa == sb else f"{sa}x{sb}"
        betas += [dict(subject=sub, roi=roi, pair_set=name, model=c, beta=v) for c, v in zip(COLS, b)]
        # B scene order
        sd = np.abs(ta.scene_position.values[p.ia.values] - tb.scene_position.values[p.ib.values]).astype(float)
        w = (p.context.values == 0) & (p.scene.values == 1)
        a = p.context.values == 1
        so_w = np.where(w, sd - sd[w].mean(), 0.0)
        so_a = np.where(a, sd - sd[a].mean(), 0.0)
        b2, _, _ = rsa.regress(y, np.column_stack([X, so_w, so_a]))
        betas += [dict(subject=sub, roi=roi, pair_set=name, model=c, beta=v)
                  for c, v in zip(["so_within", "so_across"], b2[-2:])]
        for where, m in (("same_context", w), ("other_context", a)):
            for k in range(5):
                mm = m & (sd == k)
                if mm.sum() >= 5:
                    curve.append(dict(subject=sub, roi=roi, pair_set=name, where=f"sceneorder_{where}",
                                      distance=k, z_dissim=y[mm].mean(), n_pairs=int(mm.sum())))
        # neighbour advantage of object position, by how the two objects are related
        for where, m in (("same_scene", p.scene.values == 0), ("same_context_other_scene", w),
                         ("other_context", a)):
            near, far = m & (dist == 1), m & (dist >= 2)
            if near.sum() >= 5 and far.sum() >= 5:
                neigh.append(dict(subject=sub, roi=roi, pair_set=name, where=where,
                                  advantage=y[far].mean() - y[near].mean()))
        for where, m in (("within_scene", p.scene.values == 0), ("across_scenes", p.scene.values == 1)):
            for k in range(5):
                mm = m & (dist == k)
                if mm.sum() >= 5:
                    curve.append(dict(subject=sub, roi=roi, pair_set=name, where=where, distance=k,
                                      z_dissim=y[mm].mean(), n_pairs=int(mm.sum())))
        # C neighbours within a scene against other scenes
        ok = (ta.correct.values[p.ia.values] == True) & (tb.correct.values[p.ib.values] == True)  # noqa: E712
        same_scene = p.scene.values == 0
        out = dict(subject=sub, roi=roi, pair_set=name)
        for tag, keep in (("", np.ones(len(y), bool)), ("_correct", ok)):
            adv = {}
            enough = keep.sum() >= MIN_PAIRS_SUBSET          # participants with few reported retrievals drop out
            for where, m in (("same_scene", same_scene), ("other_scene", ~same_scene)):
                near, far = keep & m & (dist == 1), keep & m & (dist >= 2)
                adv[where] = (y[far].mean() - y[near].mean()
                              if enough and near.sum() >= 5 and far.sum() >= 5 else np.nan)
            out[f"adv_same_scene{tag}"] = adv["same_scene"]
            out[f"adv_other_scene{tag}"] = adv["other_scene"]
            out[f"diff{tag}"] = adv["same_scene"] - adv["other_scene"]
        m = dist >= 1
        dc = dist[m] - dist[m].mean()
        Xs = np.column_stack([p.context.values[m], p.scene.values[m], dc, dc * same_scene[m]])
        bs, _, _ = rsa.regress(y[m], Xs)
        out["slope"], out["slope_x_scene"] = bs[2], bs[3]
        scene_rows.append(out)
    return (pd.DataFrame(betas), pd.DataFrame(curve), pd.DataFrame(neigh), pd.DataFrame(scene_rows),
            pd.DataFrame(control_rows), pd.DataFrame(across_rows), pd.DataFrame(across_curve),
            pd.DataFrame(hier_rows), pd.DataFrame(hier_levels))


def ttests(df, value, keys):
    rows = []
    for k, g in df.groupby(keys):
        v = g[value].dropna().values
        if len(v) >= 10:
            t, p = stats.ttest_1samp(v, 0)
            rows.append(dict(zip(keys, k), n=len(v), mean=v.mean(), t=t, p=p))
    return pd.DataFrame(rows)


def tt(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(n=len(v), mean=np.nan, t=np.nan, p=np.nan)
    t, p = stats.ttest_1samp(v, 0)
    return dict(n=len(v), mean=v.mean(), t=t, p=p)


def probe_order(out_dir):
    rows = []
    for session in rsa.SESSIONS:
        for f in sorted(glob.glob(opj(rsa.in_dir(), session, "*_trials.csv"))):
            t = pd.read_csv(f)
            t = t[t.kind == "object"].sort_values(["run_idx", "onset_s"]).reset_index(drop=True)
            parts = [("evening", t)] if session == "presleep" else \
                    [("morning pass 1", t[t.run_idx < 3]), ("morning pass 2", t[t.run_idx >= 3])]
            for name, g in parts:
                g = g.reset_index(drop=True)
                i, j = np.triu_indices(len(g), 1)
                dpos = np.abs(g.position.values[i] - g.position.values[j])
                dord = np.abs(i - j)                                   # trials apart in probe order
                between = g.run_idx.values[i] != g.run_idx.values[j]
                m = (g.scene.values[i] == g.scene.values[j]) & between
                use = between & (dpos > 0)
                rows.append(dict(subject=os.path.basename(f).split("_")[0], set=name,
                                 n_pairs=int(m.sum()),
                                 trials_apart_neighbours=dord[m & (dpos == 1)].mean(),
                                 trials_apart_distant=dord[m & (dpos >= 2)].mean(),
                                 r_position_vs_probe_distance=np.corrcoef(dpos[use], dord[use])[0, 1]))
    d = pd.DataFrame(rows)
    d.to_csv(opj(out_dir, "D_probe_order_by_subject.csv"), index=False)
    out = []
    for name, g in d.groupby("set", sort=False):
        t, p = stats.ttest_rel(g.trials_apart_neighbours, g.trials_apart_distant, nan_policy="omit")
        out.append(dict(set=name, n=len(g), neighbours=g.trials_apart_neighbours.mean(),
                        distant=g.trials_apart_distant.mean(), t=t, p=p,
                        mean_r=g.r_position_vs_probe_distance.mean()))
    pd.DataFrame(out).to_csv(opj(out_dir, "D_probe_order_summary.csv"), index=False)


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_dir = str(step_output_dir(__file__))
    subs = rsa.subjects()
    rois = [r for r in rsa.ROIS if r != "hpc"]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(s, r) for s in subs for r in rois)
    betas = pd.concat([r[0] for r in res], ignore_index=True)
    curve = pd.concat([r[1] for r in res], ignore_index=True)
    neigh = pd.concat([r[2] for r in res], ignore_index=True)
    scene = pd.concat([r[3] for r in res], ignore_index=True)
    ctrl = pd.concat([r[4] for r in res], ignore_index=True)
    acr = pd.concat([r[5] for r in res], ignore_index=True)
    acr_curve = pd.concat([r[6] for r in res], ignore_index=True)
    hier = pd.concat([r[7] for r in res], ignore_index=True)
    hier_lv = pd.concat([r[8] for r in res], ignore_index=True)

    # A / B
    neigh.to_csv(opj(out_dir, "A_subject_neighbour.csv"), index=False)
    wide = neigh.pivot_table(index=["subject", "roi", "pair_set"], columns="where", values="advantage").reset_index()
    wide["scene_minus_other"] = wide["same_scene"] - wide[["same_context_other_scene", "other_context"]].mean(axis=1)
    wide["context_minus_other"] = wide["same_context_other_scene"] - wide["other_context"]
    long = wide.melt(id_vars=["subject", "roi", "pair_set"], var_name="where", value_name="advantage")
    gn = ttests(long, "advantage", ["pair_set", "where", "roi"])
    pn = long[long.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "where", "roi"], as_index=False).advantage.mean()
    gpn = ttests(pn, "advantage", ["where", "roi"])
    gpn.insert(0, "pair_set", "all_sessions")
    pd.concat([gn, gpn], ignore_index=True).to_csv(opj(out_dir, "A_group_neighbour.csv"), index=False)
    betas.to_csv(opj(out_dir, "A_subject_betas.csv"), index=False)
    curve.to_csv(opj(out_dir, "A_subject_distance_curve.csv"), index=False)
    g = ttests(betas, "beta", ["pair_set", "model", "roi"])
    within = betas[betas.pair_set.isin(["pre", "post1", "post2"])]       # pooled: one value per participant
    pooled = within.groupby(["subject", "model", "roi"], as_index=False).beta.mean()
    gp = ttests(pooled, "beta", ["model", "roi"])
    gp.insert(0, "pair_set", "all_sessions")
    pd.concat([g, gp], ignore_index=True).to_csv(opj(out_dir, "A_group_beta_vs_zero.csv"), index=False)
    curve.groupby(["pair_set", "roi", "where", "distance"], as_index=False).agg(
        z_dissim=("z_dissim", "mean"), sem=("z_dissim", "sem"), n=("subject", "nunique")
    ).to_csv(opj(out_dir, "A_group_distance_curve.csv"), index=False)

    # C
    scene.to_csv(opj(out_dir, "C_subject_neighbour_scene.csv"), index=False)
    measures = [c for c in scene.columns if c not in ("subject", "roi", "pair_set")]
    rows, per_group = [], {}
    for gname, members in GROUPS.items():
        avg = scene[scene.pair_set.isin(members)].groupby(["subject", "roi"], as_index=False)[measures].mean()
        per_group[gname] = avg.set_index(["subject", "roi"])
        for roi, gg in avg.groupby("roi"):
            for m in measures:
                rows.append(dict(group=gname, roi=roi, measure=m, **tt(gg[m])))
    delta = (per_group["morning"] - per_group["evening"]).reset_index()
    for roi, gg in delta.groupby("roi"):
        for m in measures:
            rows.append(dict(group="morning_minus_evening", roi=roi, measure=m, **tt(gg[m])))
    grp = pd.DataFrame(rows)
    grp["p_fdr_rois"] = grp.groupby(["group", "measure"])["p"].transform(rsa.fdr_bh)
    grp.to_csv(opj(out_dir, "C_group_neighbour_scene.csv"), index=False)

    # E controls for the position code
    ctrl.to_csv(opj(out_dir, "E_subject_controls.csv"), index=False)
    ge = ttests(ctrl, "beta", ["pair_set", "control", "model", "roi"])
    pooled_e = ctrl[ctrl.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "control", "model", "roi"], as_index=False).beta.mean()
    gpe = ttests(pooled_e, "beta", ["control", "model", "roi"])
    gpe.insert(0, "pair_set", "all_sessions")
    ge = pd.concat([ge, gpe], ignore_index=True)
    ge["p_fdr_rois"] = ge.groupby(["pair_set", "control", "model"])["p"].transform(rsa.fdr_bh)
    ge.to_csv(opj(out_dir, "E_group_controls.csv"), index=False)

    # F position distance across sequences
    acr.to_csv(opj(out_dir, "F_subject_across.csv"), index=False)
    gf = ttests(acr, "beta", ["pair_set", "pairs", "model", "roi"])
    pooled_f = acr[acr.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "pairs", "model", "roi"], as_index=False).beta.mean()
    gpf = ttests(pooled_f, "beta", ["pairs", "model", "roi"])
    gpf.insert(0, "pair_set", "all_sessions")
    gf = pd.concat([gf, gpf], ignore_index=True)
    gf["p_fdr_rois"] = gf.groupby(["pair_set", "pairs", "model"])["p"].transform(rsa.fdr_bh)
    gf.to_csv(opj(out_dir, "F_group_across.csv"), index=False)
    # step against gradient: paired comparison of the two partial correlations (Fisher z), sessions pooled
    wide = pooled_f.pivot_table(index=["subject", "pairs", "roi"], columns="model", values="beta").reset_index()
    wide["step_minus_gradient"] = np.arctanh(wide["step_r"]) - np.arctanh(wide["gradient_r"])
    comp = []
    for (pr, roi), g in wide.groupby(["pairs", "roi"]):
        comp.append(dict(pairs=pr, roi=roi, mean_r_gradient=g.gradient_r.mean(), mean_r_step=g.step_r.mean(),
                         t_gradient=tt(np.arctanh(g.gradient_r))["t"], t_step=tt(np.arctanh(g.step_r))["t"],
                         t_joint_step=tt(g.joint_step)["t"], t_joint_beyond=tt(g.joint_beyond)["t"],
                         p_joint_beyond=tt(g.joint_beyond)["p"],
                         **{f"{k}_step_minus_gradient": v for k, v in tt(g.step_minus_gradient).items()}))
    pd.DataFrame(comp).to_csv(opj(out_dir, "F_group_model_comparison.csv"), index=False)
    acr_curve.to_csv(opj(out_dir, "F_subject_across_curve.csv"), index=False)
    pc = acr_curve[acr_curve.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "roi", "pairs", "distance"], as_index=False).z_dissim.mean()
    pc.groupby(["roi", "pairs", "distance"], as_index=False).agg(
        z_dissim=("z_dissim", "mean"), sem=("z_dissim", "sem"), n=("subject", "nunique")
    ).to_csv(opj(out_dir, "F_group_across_curve.csv"), index=False)

    # G hierarchy distance without shared picture or digit
    hier.to_csv(opj(out_dir, "G_subject_hierarchy.csv"), index=False)
    gh = ttests(hier, "beta", ["pair_set", "model", "roi"])
    pooled_h = hier[hier.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "model", "roi"], as_index=False).beta.mean()
    gph = ttests(pooled_h, "beta", ["model", "roi"])
    gph.insert(0, "pair_set", "all_sessions")
    gh = pd.concat([gh, gph], ignore_index=True)
    gh["p_fdr_rois"] = gh.groupby(["pair_set", "model"])["p"].transform(rsa.fdr_bh)
    gh.to_csv(opj(out_dir, "G_group_hierarchy.csv"), index=False)
    pl = hier_lv[hier_lv.pair_set.isin(["pre", "post1", "post2"])].groupby(
        ["subject", "roi", "pairs", "level"], as_index=False).agg(z_dissim=("z_dissim", "mean"), n_pairs=("n_pairs", "mean"))
    pl.groupby(["roi", "pairs", "level"], as_index=False).agg(
        z_dissim=("z_dissim", "mean"), sem=("z_dissim", "sem"), n=("subject", "nunique"), n_pairs=("n_pairs", "mean")
    ).to_csv(opj(out_dir, "G_group_hierarchy_levels.csv"), index=False)

    probe_order(out_dir)
    print("written to", out_dir)


if __name__ == "__main__":
    main()
