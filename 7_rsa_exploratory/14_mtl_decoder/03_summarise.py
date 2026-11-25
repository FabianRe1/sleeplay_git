#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 - Group summary of the decoder variants (localizer cross-validation only).

For every variant, classifier (object / scene) and test set (all / shown / recalled):
mean balanced accuracy over participants, one-sample t against chance (10 %), and a
paired t against the current setting (`mtl_current`; for hippocampus variants also
against `hpc_current`). p values are uncorrected; q is Benjamini-Hochberg over all
variants within one classifier x test set.

Also an honest estimate of what "picking the best variant" buys: for each participant
the variant is chosen on the OTHER participants and scored on the left-out one.

Usage:  python 03_summarise.py stage1 [stage2 ...]     (stages are pooled)
Output: outputs/14_mtl_decoder/summary/<stages>_long.csv      per participant
        outputs/14_mtl_decoder/summary/<stages>_table.csv     variants x measures
        outputs/14_mtl_decoder/summary/<stages>_loso_selection.csv
        outputs/14_mtl_decoder/summary/validation_against_existing.csv
"""

import glob
import os
import sys
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import mtl_config as cfg

CHANCE = 10.0


def load(stages):
    frames = []
    for st in stages:
        for f in sorted(glob.glob(opj(cfg.RESULT_DIR, st, f"s*_{st}.csv"))):
            d = pd.read_csv(f)
            d["stage"] = st
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["sub", "variant", "kind", "test"], keep="first")


def validation():
    """Existing 4.1 tables against this harness, current setting, all participants."""
    rows = []
    for f in sorted(glob.glob(opj(cfg.RESULT_DIR, "*", "s*_predictions_current.csv"))):
        sub = os.path.basename(f).split("_")[0]
        p = pd.read_csv(f)
        for kind, k in (("scene", 1), ("object", 2)):
            e = pd.read_csv(opj(
                cfg.PATH_EXISTING_CV,
                f"only_scenes1_objects2_both3_{k}_{sub}_Mask_mtl_Smooth_4_Threshold_30"
                f"_reliability_threshold_70_tmpsmoothing_3.csv"))
            pk = p[p["kind"] == kind].reset_index(drop=True)
            row = dict(sub=sub, kind=kind, n_existing=len(e), n_harness=len(pk))
            for name, ph in (("all", None), ("shown", "encoding"), ("recalled", "retrieval")):
                ee = e if ph is None else e[e["trial_phase"] == ph]
                pp = pk if ph is None else pk[pk["trial_phase"] == ph]
                row[f"existing_{name}"] = ee.groupby("training_category")["label_match"].mean().mean() * 100
                row[f"harness_{name}"] = (pp["predicted_labels"] == pp["training_category"]).groupby(
                    pp["training_category"]).mean().mean() * 100
            if len(e) == len(pk):
                row["same_trials"] = bool((e["training_category"].values == pk["training_category"].values).all())
                row["prediction_agreement"] = float((e["predicted_labels"].values == pk["predicted_labels"].values).mean())
            row["balanced_classes"] = bool(pk["training_category"].value_counts().nunique() == 1)
            rows.append(row)
    v = pd.DataFrame(rows).drop_duplicates(subset=["sub", "kind"])
    v.to_csv(opj(cfg.SUMMARY_DIR, "validation_against_existing.csv"), index=False)
    return v


def paired(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or np.allclose(a[ok], b[ok]):
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_rel(a[ok], b[ok])
    return float((a[ok] - b[ok]).mean()), float(t), float(p)


def table(d):
    rows = []
    wide = d.pivot_table(index="sub", columns=["variant", "kind", "test"], values="acc")
    info = d.groupby("variant").agg(n_vox=("n_vox", "mean")).round(0)
    variants = list(dict.fromkeys(d["variant"]))
    for var in variants:
        for kind in ("object", "scene"):
            for test in ("all", "shown", "recalled"):
                x = wide[(var, kind, test)]
                t0, p0 = stats.ttest_1samp(x.dropna(), CHANCE)
                row = dict(variant=var, kind=kind, test=test, n=int(x.notna().sum()),
                           n_vox=info.loc[var, "n_vox"], mean=x.mean(), sd=x.std(),
                           t_chance=t0, p_chance=p0)
                row["diff_mtl"], row["t_mtl"], row["p_mtl"] = paired(x, wide[("mtl_current", kind, test)])
                if var.startswith("hpc_") and ("hpc_current", kind, test) in wide:
                    row["diff_hpc"], row["t_hpc"], row["p_hpc"] = paired(x, wide[("hpc_current", kind, test)])
                rows.append(row)
    t = pd.DataFrame(rows)
    t["q_mtl"] = np.nan
    for _, idx in t.groupby(["kind", "test"]).groups.items():
        p = t.loc[idx, "p_mtl"]
        ok = p.notna()
        if ok.any():
            t.loc[p.index[ok], "q_mtl"] = multipletests(p[ok], method="fdr_bh")[1]
    return t, wide


def loso_selection(wide, variants, label):
    """Choose the variant on N-1 participants, score it on the left-out one."""
    rows = []
    for kind in ("object", "scene"):
        for crit in ("all", "recalled"):
            W = pd.DataFrame({v: wide[(v, kind, crit)] for v in variants}).dropna()
            picked, gains = [], {t: [] for t in ("all", "shown", "recalled")}
            for s in W.index:
                best = W.drop(index=s).mean().idxmax()
                picked.append(best)
                for t in gains:
                    gains[t].append(wide.loc[s, (best, kind, t)] - wide.loc[s, ("mtl_current", kind, t)])
            row = dict(pool=label, kind=kind, chosen_by=crit, n=len(W),
                       most_often_picked=pd.Series(picked).value_counts().index[0],
                       n_distinct_picked=len(set(picked)),
                       in_sample_best=W.mean().idxmax(),
                       in_sample_gain=W.mean().max() - W["mtl_current"].mean())
            for t, g in gains.items():
                g = np.array(g)
                row[f"heldout_gain_{t}"] = g.mean()
                row[f"t_{t}"], row[f"p_{t}"] = (stats.ttest_1samp(g, 0) if g.std() > 0 else (np.nan, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    stages = sys.argv[1:] or ["stage1"]
    tag = "_".join(stages)
    os.makedirs(cfg.SUMMARY_DIR, exist_ok=True)
    d = load(stages)
    d.to_csv(opj(cfg.SUMMARY_DIR, f"{tag}_long.csv"), index=False)
    print(f"{d['sub'].nunique()} participants: {' '.join(sorted(d['sub'].unique()))}")

    v = validation()
    print("\nValidation against the existing tables (current setting)")
    print(v[[c for c in v.columns if c.startswith(("existing", "harness"))]].mean().round(2).to_string())
    print("participants x classifiers with identical predictions:",
          int((v["prediction_agreement"] == 1).sum()), "of", len(v))
    print("max |difference| in accuracy (all trials):",
          round((v["existing_all"] - v["harness_all"]).abs().max(), 2))

    t, wide = table(d)
    t.to_csv(opj(cfg.SUMMARY_DIR, f"{tag}_table.csv"), index=False)

    # compact print: variants x (object, scene) x (all, shown, recalled), difference to mtl_current
    m = t.pivot(index="variant", columns=["kind", "test"], values="mean").reindex(dict.fromkeys(t["variant"]))
    dm = t.pivot(index="variant", columns=["kind", "test"], values="t_mtl").reindex(m.index)
    out = pd.DataFrame(index=m.index)
    out["n_vox"] = t.drop_duplicates("variant").set_index("variant")["n_vox"]
    for kind in ("object", "scene"):
        for test in ("all", "shown", "recalled"):
            out[f"{kind[:3]}_{test[:3]}"] = m[(kind, test)].round(2)
            out[f"t_{kind[:3]}_{test[:3]}"] = dm[(kind, test)].round(2)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    print("\nMean balanced accuracy (%), and paired t against mtl_current")
    print(out.to_string())
    out.to_csv(opj(cfg.SUMMARY_DIR, f"{tag}_table_compact.csv"))

    variants = list(dict.fromkeys(d["variant"]))
    pools = {"all_variants": variants,
             "mtl_variants": [x for x in variants if x.startswith("mtl_")],
             "hpc_variants": [x for x in variants if x.startswith("hpc_")] + ["mtl_current"]}
    sel = pd.concat([loso_selection(wide, vs, lab) for lab, vs in pools.items() if len(vs) > 1])
    sel.to_csv(opj(cfg.SUMMARY_DIR, f"{tag}_loso_selection.csv"), index=False)
    print("\nVariant chosen on the other participants, scored on the left-out one (gain over mtl_current)")
    print(sel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
