# -*- coding: utf-8 -*-
"""Hierarchical sequence model RDMs for the Sleeplay retrieval trials (step 2: no neural data).

Inspired by Fan, Wang, Fang, Ding & Luo (2025, Nat Hum Behav 9:360-375), adapted to
context > scene > object. The model is a set of predictors that are meant to be
regressed onto the neural RDM side by side, next to the binary predictors of
02_binary_rsa.py (0 = same, 1 = different). All are DISTANCES (larger = less similar).

Every trial has coordinates
    context         day / night
    scene position  G = 1-5 within its context   (object trials: position of the object's scene)
    object position L = 1-5 within its scene     (object trials only)
    digit           the digit on screen: L on object trials, G on scene trials

New predictors (columns added to the pair table of 02)
    tree              finest level two trials share: 0 same node (same object / same scene),
                      1 same scene, 2 same context, 3 different context.
                      Equals context + scene (+ object) of the binary predictors, so it
                      REPLACES them in a regression, it cannot stand next to them.
    scene_pos_dist    |G_a - G_b|, scene position aligned across contexts (0-4)
    obj_pos_dist      |L_a - L_b|, object position aligned across scenes (0-4); object x object.
                      On screen this is also the distance between the two digits
                      ("position or digit").
    digit_dist        |digit_a - digit_b|. For object x scene pairs this compares an object
                      position with a scene position: a level-general position code, or the digit.
    chain50           distance along the subject's own encoding order: first context
                      (25 steps) then second context. A scene trial sits at the centre of
                      its block of five objects. Needs the subject's encoding trial table.
    chain25           the same within a context only; pairs from different contexts get a
                      constant (see "parts" below).
  parts, to ask where a graded effect comes from
    scene_pos_same_ctx / scene_pos_other_ctx      scene_pos_dist for pairs within / across contexts
    obj_pos_same_scene / obj_pos_other_scene      obj_pos_dist for pairs within / across scenes
    A part holds its distance on its own pairs and a constant (the mean of its own
    pairs) on all others. The constant is arbitrary: it is exactly collinear with the
    binary context (or scene) predictor, so with that predictor in the model the beta
    of the part does not depend on it.

Also defined, for the fitting stage: position_map(), which averages any pairwise
dissimilarity into the 25 x 25 cells (scene position x object position) using only
pairs from different scenes.

Outputs (outputs/10_hsm_model_rdms/)
    model_rdms_example.png             every predictor as a 70 x 70 matrix, canonical order
    predictor_correlations_<pt>.png    mean correlation among predictors on the pairs the
                                       regression uses (between-run pairs), per pair type
    predictor_correlations_subject.csv / _mean.csv
    vif.csv                            variance inflation of each predictor in candidate regressions
    pair_counts.csv                    number of usable pairs per distance level

Usage:  python 10_hsm_model_rdms.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd

import rsa_config as cfg


def _import_binary_rsa():
    """Import 02_binary_rsa.py as a module (it reads sys.argv[1] as its pattern source)."""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        spec = importlib.util.spec_from_file_location("binary_rsa_02", opj(cfg.HERE, "02_binary_rsa.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


rsa02 = _import_binary_rsa()

OUT_DIR = opj(cfg.OUT_ROOT, "10_hsm_model_rdms")
TRIAL_DIR = opj(cfg.OUT_ROOT, "01_trial_patterns")          # trial tables written by 01
RETRIEVAL_PAIR_SETS = [("pre", "pre"), ("post1", "post1"), ("post2", "post2"),
                       ("post1", "post2"), ("pre", "post1"), ("pre", "post2")]

NEW_PREDICTORS = {
    "oo": ["tree", "scene_pos_dist", "obj_pos_dist", "chain50", "chain25",
           "scene_pos_same_ctx", "scene_pos_other_ctx", "obj_pos_same_scene", "obj_pos_other_scene"],
    # scene x scene: chain25 is 5 x scene_pos_same_ctx there, so it is left out
    "ss": ["tree", "scene_pos_dist", "chain50", "scene_pos_same_ctx", "scene_pos_other_ctx"],
    "os": ["tree", "scene_pos_dist", "digit_dist", "chain50", "chain25",
           "scene_pos_same_ctx", "scene_pos_other_ctx"],
}
# candidate regressions, only used here to report variance inflation
MODEL_SETS = {
    "oo": {
        "tree": ["tree", "catseq", "category", "position", "scene_pos_dist", "obj_pos_dist"],
        "split": ["context", "scene", "object", "catseq", "category", "position",
                  "scene_pos_dist", "obj_pos_dist"],
        "split+chain50": ["context", "scene", "object", "catseq", "category", "position",
                          "scene_pos_dist", "obj_pos_dist", "chain50"],
        "split, parts": ["context", "scene", "object", "catseq", "category", "position",
                         "scene_pos_same_ctx", "scene_pos_other_ctx",
                         "obj_pos_same_scene", "obj_pos_other_scene"],
    },
    "ss": {
        "tree": ["tree", "position", "scene_pos_dist"],
        "split": ["context", "scene", "position", "scene_pos_dist"],
        "split+chain50": ["context", "scene", "position", "scene_pos_dist", "chain50"],
        "split, parts": ["context", "scene", "position", "scene_pos_same_ctx", "scene_pos_other_ctx"],
    },
    "os": {
        "tree": ["tree", "position", "scene_pos_dist", "digit_dist"],
        "split": ["context", "scene", "position", "scene_pos_dist", "digit_dist"],
        "split+chain50": ["context", "scene", "position", "scene_pos_dist", "digit_dist", "chain50"],
        "split, parts": ["context", "scene", "position", "scene_pos_same_ctx", "scene_pos_other_ctx",
                         "digit_dist"],
    },
}
LABEL = {
    "context": "context (binary)", "scene": "scene (binary)", "catseq": "category sequence (binary)",
    "category": "category (binary)", "position": "position / digit (binary)", "object": "object (binary)",
    "tree": "tree distance", "scene_pos_dist": "scene position distance",
    "obj_pos_dist": "object position distance", "digit_dist": "digit distance",
    "chain50": "encoding order, 50 steps", "chain25": "encoding order, within context",
    "scene_pos_same_ctx": "scene position, same context", "scene_pos_other_ctx": "scene position, other context",
    "obj_pos_same_scene": "object position, same scene", "obj_pos_other_scene": "object position, other scene",
}
PAIR_LABEL = {"oo": "object trials x object trials", "ss": "scene trials x scene trials",
              "os": "object trials x scene trials"}
SET_LABEL = {"pre": "evening", "post1": "morning pass 1", "post2": "morning pass 2",
             "post1xpost2": "morning 1 x morning 2", "prexpost1": "evening x morning 1",
             "prexpost2": "evening x morning 2"}


# ---------------------------------------------------------------------------
# trial tables
# ---------------------------------------------------------------------------

def subjects() -> list:
    return sorted({os.path.basename(f).split("_")[0]
                   for f in glob.glob(opj(TRIAL_DIR, "*", "*_trials.csv"))})


def load_trial_sets(sub: str) -> dict:
    """{pre, post1, post2: trial table}, split as in load_sets() of 02 (no patterns loaded)."""
    sets = {}
    f = opj(TRIAL_DIR, "presleep", f"{sub}_presleep_trials.csv")
    if os.path.exists(f):
        sets["pre"] = pd.read_csv(f).reset_index(drop=True)
    f = opj(TRIAL_DIR, "postsleep", f"{sub}_postsleep_trials.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        sets["post1"] = t[t.run_idx < 3].reset_index(drop=True)
        sets["post2"] = t[t.run_idx >= 3].reset_index(drop=True)
    return sets


def first_context(sub: str):
    """Context that was learned first ('day' / 'night'), from the encoding trial table; None if missing."""
    files = sorted(glob.glob(opj(cfg.OUT_ROOT, "patterns_*", "encoding", f"{sub}_encoding_trials.csv")))
    if not files:
        return None
    t = pd.read_csv(files[0])
    obj = t[t.kind == "object"].sort_values(["run_idx", "onset_s"])
    return obj.context.iloc[0] if len(obj) else None


def trial_coordinates(trials: pd.DataFrame, first_ctx=None) -> pd.DataFrame:
    """Coordinates of every trial in the hierarchy (see module docstring)."""
    obj = (trials.kind == "object").values
    G = trials.scene_position.values.astype(float)
    L = np.where(obj, trials.position.values.astype(float), np.nan)
    within = (G - 1) * 5 + np.where(obj, L, 3.0)             # 1-25; scene trial = centre of its block
    if first_ctx is None:
        ctx_rank = np.full(len(trials), np.nan)
    else:
        ctx_rank = (trials.context.values != first_ctx).astype(float)
    return pd.DataFrame({"is_object": obj, "G": G, "L": L,
                         "digit": trials.position.values.astype(float),
                         "chain25": within, "chain50": ctx_rank * 25 + within})


# ---------------------------------------------------------------------------
# pair table
# ---------------------------------------------------------------------------

def _part(dist, own, pairtype):
    """dist on the pairs in `own`, the mean of those pairs elsewhere (per pair type)."""
    out = np.full(len(dist), np.nan)
    for pt in np.unique(pairtype):
        sel = pairtype == pt
        o = sel & own & np.isfinite(dist)
        if o.any():
            out[sel] = dist[o].mean()
            out[o] = dist[o]
        out[sel & ~np.isfinite(dist)] = np.nan
    return out


def hsm_pair_table(ta: pd.DataFrame, tb: pd.DataFrame, same_set: bool, first_ctx=None) -> pd.DataFrame:
    """pair_table() of 02 plus the hierarchical sequence model predictors."""
    p = rsa02.pair_table(ta, tb, same_set)
    a = trial_coordinates(ta, first_ctx).iloc[p.ia.values].reset_index(drop=True)
    b = trial_coordinates(tb, first_ctx).iloc[p.ib.values].reset_index(drop=True)
    pt = p.pairtype.values
    same_ctx, same_scene = (p.context.values == 0), (p.scene.values == 0)

    tree = np.where(~same_ctx, 3.0, np.where(~same_scene, 2.0, 1.0))
    same_node = ((pt == "oo") & (p.object.values == 0)) | ((pt == "ss") & same_scene)
    p["tree"] = np.where(same_node, 0.0, tree)

    p["scene_pos_dist"] = np.abs(a.G.values - b.G.values)
    p["obj_pos_dist"] = np.abs(a.L.values - b.L.values)              # NaN unless object x object
    p["digit_dist"] = np.abs(a.digit.values - b.digit.values)
    p["chain50"] = np.abs(a.chain50.values - b.chain50.values)       # NaN without encoding table
    p["chain25"] = _part(np.abs(a.chain25.values - b.chain25.values), same_ctx, pt)
    p["scene_pos_same_ctx"] = _part(p.scene_pos_dist.values, same_ctx, pt)
    p["scene_pos_other_ctx"] = _part(p.scene_pos_dist.values, ~same_ctx, pt)
    p["obj_pos_same_scene"] = _part(p.obj_pos_dist.values, same_scene, pt)
    p["obj_pos_other_scene"] = _part(p.obj_pos_dist.values, ~same_scene, pt)
    return p


def position_map(dissimilarity, ta, tb, pairs) -> np.ndarray:
    """25 x 25 map of a pairwise dissimilarity between (scene position, object position) cells.

    dissimilarity : one value per row of `pairs` (e.g. d[pairs.ia, pairs.ib])
    Only object x object pairs from different scenes and different runs are used, so
    that neither item identity nor a shared scene image enters. Cell index =
    (scene position - 1) * 5 + (object position - 1). Symmetric; NaN where no pair exists.
    """
    ca, cb = trial_coordinates(ta), trial_coordinates(tb)
    ia, ib = pairs.ia.values, pairs.ib.values
    cell_a = ((ca.G.values - 1) * 5 + (ca.L.values - 1))[ia]
    cell_b = ((cb.G.values - 1) * 5 + (cb.L.values - 1))[ib]
    y = np.asarray(dissimilarity, float)
    ok = ((pairs.pairtype == "oo") & (pairs.scene == 1) & (pairs.same_run == 0)).values & np.isfinite(y)
    total, count = np.zeros((25, 25)), np.zeros((25, 25))
    for i, j, v in zip(cell_a[ok].astype(int), cell_b[ok].astype(int), y[ok]):
        total[i, j] += v; count[i, j] += 1
        if i != j:
            total[j, i] += v; count[j, i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


# ---------------------------------------------------------------------------
# collinearity on the pairs the regression uses
# ---------------------------------------------------------------------------

def vif(X: np.ndarray) -> np.ndarray:
    out = np.full(X.shape[1], np.nan)
    for k in range(X.shape[1]):
        others = np.column_stack([np.ones(len(X)), np.delete(X, k, axis=1)])
        resid = X[:, k] - others @ np.linalg.lstsq(others, X[:, k], rcond=None)[0]
        ss_tot = ((X[:, k] - X[:, k].mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot
        out[k] = np.inf if r2 > 1 - 1e-10 else 1 / (1 - r2)
    return out


def analyse_subject(sub: str):
    sets, ctx1 = load_trial_sets(sub), first_context(sub)
    corr_rows, vif_rows, count_rows = [], [], []
    for sa, sb in RETRIEVAL_PAIR_SETS:
        if sa not in sets or sb not in sets:
            continue
        same_set = sa == sb
        set_name = sa if same_set else f"{sa}x{sb}"
        pairs = hsm_pair_table(sets[sa], sets[sb], same_set, ctx1)
        pairs = pairs[pairs.same_run == 0]                    # the regression uses between-run pairs
        for ptype in ("oo", "ss", "os"):
            g = pairs[pairs.pairtype == ptype]
            cols = [c for c in rsa02.PREDICTORS[ptype] + NEW_PREDICTORS[ptype]
                    if g[c].notna().all() and g[c].std() > 0]
            r = np.corrcoef(g[cols].values.T)
            for i, ci in enumerate(cols):
                for j, cj in enumerate(cols):
                    if i < j:
                        corr_rows.append(dict(subject=sub, pair_set=set_name, pairtype=ptype,
                                              a=ci, b=cj, r=r[i, j], n_pairs=len(g)))
            for mname, mcols in MODEL_SETS[ptype].items():
                use = [c for c in mcols if c in cols]
                if len(use) < len([c for c in mcols if c != "object"]):
                    continue                                  # e.g. chain50 without an encoding table
                for c, v in zip(use, vif(g[use].values)):
                    vif_rows.append(dict(subject=sub, pair_set=set_name, pairtype=ptype,
                                         model_set=mname, predictor=c, vif=v))
            for c in ("tree", "scene_pos_same_ctx", "scene_pos_other_ctx",
                      "obj_pos_same_scene", "obj_pos_other_scene", "digit_dist"):
                if c not in NEW_PREDICTORS[ptype]:
                    continue
                own = np.ones(len(g), bool)
                if c.endswith("same_ctx"): own = (g.context == 0).values
                if c.endswith("other_ctx"): own = (g.context == 1).values
                if c.endswith("same_scene"): own = (g.scene == 0).values
                if c.endswith("other_scene"): own = (g.scene == 1).values
                for value, n in g.loc[own, c].value_counts().items():
                    count_rows.append(dict(subject=sub, pair_set=set_name, pairtype=ptype,
                                           predictor=c, distance=value, n_pairs=n))
    return pd.DataFrame(corr_rows), pd.DataFrame(vif_rows), pd.DataFrame(count_rows), ctx1


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

INK, MUTED = "#222222", "#8a8a8a"


def _colormaps():
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fb", "#9cc3ee", "#2a78d6", "#0f2f5c"])
    seq.set_bad("#d9d9d9")
    div = LinearSegmentedColormap.from_list("div", ["#1f5fb0", "#8fb8e8", "#eeeeee", "#f0a98a", "#c8451f"])
    div.set_bad("#ffffff")
    return seq, div


def figure_model_rdms(sub: str):
    """Every predictor as a 70 x 70 matrix (evening trials of one subject, all pairs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seq, _ = _colormaps()
    trials, ctx1 = load_trial_sets(sub)["pre"], first_context(sub)
    pairs = hsm_pair_table(trials, trials, True, ctx1)
    canon = rsa02.canonical_index(trials)
    ia, ib = canon[pairs.ia.values], canon[pairs.ib.values]
    panels = ["context", "scene", "catseq", "category", "position",
              "tree", "scene_pos_dist", "digit_dist", "chain50", "chain25",
              "scene_pos_same_ctx", "scene_pos_other_ctx", "obj_pos_same_scene", "obj_pos_other_scene"]
    titles = dict(LABEL, digit_dist="position / digit distance\n(object block = object position)")
    plt.rcParams.update({"font.size": 7, "figure.dpi": 150, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.6))
    for ax, name in zip(axes.ravel(), panels):
        m = np.full((rsa02.N_CANON, rsa02.N_CANON), np.nan)
        m[ia, ib] = pairs[name].values
        m[ib, ia] = pairs[name].values
        im = ax.imshow(np.ma.masked_invalid(m), cmap=seq, interpolation="nearest")
        for edge in (24.5, 49.5, 59.5):
            ax.axhline(edge, color="white", lw=1.2); ax.axvline(edge, color="white", lw=1.2)
        ax.set_xticks([12, 37, 54.5, 64.5]); ax.set_yticks([12, 37, 54.5, 64.5])
        ax.set_xticklabels(["day\nobjects", "night\nobjects", "day\nsc.", "night\nsc."])
        ax.set_yticklabels(["day obj.", "night obj.", "day", "night"], rotation=90, va="center")
        ax.tick_params(length=0)
        ax.set_title(titles[name], fontsize=8, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=6)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.95, "\n".join([
        "All values are distances: light = similar, dark = far.",
        "Grey = not defined for that pair.",
        "", "Order within a block:",
        "objects: scene position 1-5, within it",
        "              object position 1-5",
        "scenes (sc.): scene position 1-5, two trials each",
        "", f"Example: {sub}, evening retrieval;",
        f"first context at encoding: {ctx1}.",
        "Top row: binary predictors of 02.",
        "Other rows: new predictors."]),
        va="top", fontsize=7.5, color=INK, transform=axes.ravel()[-1].transAxes)
    fig.suptitle("Model RDMs of the hierarchical sequence model (70 retrieval trials, structure order)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(opj(OUT_DIR, "model_rdms_example.png"), bbox_inches="tight")
    plt.close(fig)


def figure_correlations(mean_corr: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, div = _colormaps()
    plt.rcParams.update({"font.size": 7, "figure.dpi": 150})
    for ptype in ("oo", "ss", "os"):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
        for ax, pset in zip(axes, ("pre", "post1xpost2")):
            g = mean_corr[(mean_corr.pairtype == ptype) & (mean_corr.pair_set == pset)]
            cols = [c for c in rsa02.PREDICTORS[ptype] + NEW_PREDICTORS[ptype]
                    if c in set(g.a) | set(g.b)]
            m = np.full((len(cols), len(cols)), np.nan)
            for _, row in g.iterrows():
                i, j = cols.index(row.a), cols.index(row.b)
                m[max(i, j), min(i, j)] = row.r
            ax.imshow(np.ma.masked_invalid(m), cmap=div, vmin=-1, vmax=1, interpolation="nearest")
            for i in range(len(cols)):
                for j in range(i):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.2f}".replace("0.", ".", 1), ha="center", va="center",
                                fontsize=6.5, color="white" if abs(m[i, j]) > 0.7 else INK)
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
            ax.set_xticklabels([LABEL[c] for c in cols], rotation=45, ha="right")
            ax.set_yticklabels([LABEL[c] for c in cols])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            n_old = sum(c in cols for c in rsa02.PREDICTORS[ptype])
            ax.axhline(n_old - 0.5, color=MUTED, lw=0.8); ax.axvline(n_old - 0.5, color=MUTED, lw=0.8)
            ax.set_title(f"{SET_LABEL[pset]} (mean r over {int(g.n_subjects.max())} subjects)",
                         fontsize=8.5, color=INK, loc="left")
        fig.suptitle(f"Correlation among predictors, {PAIR_LABEL[ptype]}, between-run pairs. "
                     "Above / left of the grey line: binary predictors of 02. Blue = negative, red = positive.",
                     fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(opj(OUT_DIR, f"predictor_correlations_{ptype}.png"), bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = subjects()
    results = [analyse_subject(s) for s in subs]
    corr = pd.concat([r[0] for r in results], ignore_index=True)
    vifs = pd.concat([r[1] for r in results], ignore_index=True)
    counts = pd.concat([r[2] for r in results], ignore_index=True)
    ctx = pd.DataFrame({"subject": subs, "first_context": [r[3] for r in results]})

    corr.to_csv(opj(OUT_DIR, "predictor_correlations_subject.csv"), index=False)
    mean_corr = (corr.groupby(["pair_set", "pairtype", "a", "b"], sort=False)
                 .agg(r=("r", "mean"), r_min=("r", "min"), r_max=("r", "max"),
                      n_subjects=("subject", "nunique"), n_pairs=("n_pairs", "mean")).reset_index())
    mean_corr.to_csv(opj(OUT_DIR, "predictor_correlations_mean.csv"), index=False)
    (vifs.groupby(["pair_set", "pairtype", "model_set", "predictor"], sort=False)
     .agg(vif=("vif", "mean"), vif_max=("vif", "max"), n_subjects=("subject", "nunique"))
     .reset_index().to_csv(opj(OUT_DIR, "vif.csv"), index=False))
    (counts.groupby(["pair_set", "pairtype", "predictor", "distance"])
     .agg(n_pairs=("n_pairs", "mean"), n_min=("n_pairs", "min"), n_subjects=("subject", "nunique"))
     .reset_index().to_csv(opj(OUT_DIR, "pair_counts.csv"), index=False))
    ctx.to_csv(opj(OUT_DIR, "first_context.csv"), index=False)

    example = next((s for s in subs if first_context(s) is not None and "pre" in load_trial_sets(s)), subs[0])
    figure_model_rdms(example)
    figure_correlations(mean_corr)
    print(f"{len(subs)} subjects; encoding order known for {ctx.first_context.notna().sum()}")
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
