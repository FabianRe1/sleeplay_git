#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SODA Diagnostics: probability heterogeneity, position effects, trial-type splits.

Produces 4 diagnostic figure sets:
  1. Classifier probability distributions & heterogeneity
  2. Position effects (does later cue position → more sequentiality?)
  3. Scene vs Object retrieval split
  4. Shuffle comparison (random vs selected)

Usage:
    python 4.6b_SODA_diagnostics.py [ROI] [presleep|postsleep]
"""

import os
import sys
import ast
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, pearsonr, friedmanchisquare
from matplotlib import pyplot as plt


# =====================================================================
# DATA LOADING
# =====================================================================

def parse_prob_list(x):
    """Parse string representation of probability list."""
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, str):
        # Handle np.float64(...) format from CSV
        import re
        if 'np.float' in x:
            floats = re.findall(r'np\.float\d*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)', x)
            if floats:
                return [float(v) for v in floats]
        try:
            result = ast.literal_eval(x)
            if isinstance(result, (list, tuple)):
                return [float(v) for v in result]
        except (ValueError, SyntaxError):
            pass
    return [np.nan] * 5


def load_soda_results(csv_dir, roi):
    """Load SODA results CSV for a given ROI."""
    pattern = opj(csv_dir, f"{roi}_soda_results_per_combination.csv")
    files = glob.glob(pattern)
    if not files:
        files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
        files = [f for f in files if roi in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No SODA results for ROI '{roi}' in {csv_dir}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        "tr", "retrieved_soda", "control_soda",
        "retrieved_soda_abs", "control_soda_abs",
        "retrieved_soda_shuffled", "control_soda_shuffled",
        "retrieved_soda_shuffled_selected", "control_soda_shuffled_selected",
        "retrieved_soda_shuffled_abs", "control_soda_shuffled_abs",
        "retrieved_soda_shuffled_selected_abs", "control_soda_shuffled_selected_abs",
        "retrieved_position", "behav_correct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["retrieved_probs", "control_probs",
                "retrieved_probs_shuffle", "control_probs_shuffle",
                "retrieved_probs_shuffle_selected", "control_probs_shuffle_selected"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_prob_list)

    return df


# =====================================================================
# SECTION 1: PROBABILITY DIAGNOSTICS
# =====================================================================

def _sig_str(p):
    """Return significance string for p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_probability_diagnostics(df, roi, out_dir):
    """Analyze classifier probability distributions, split by condition."""
    # Prepare per-item columns
    ret_probs_matrix = np.array(df["retrieved_probs"].tolist())
    ctrl_probs_matrix = np.array(df["control_probs"].tolist())
    df_tmp = df.copy()
    for i in range(5):
        df_tmp[f"ret_p{i + 1}"] = ret_probs_matrix[:, i]
        df_tmp[f"ctrl_p{i + 1}"] = ctrl_probs_matrix[:, i]
    ret_p_cols = [f"ret_p{i}" for i in range(1, 6)]
    ctrl_p_cols = [f"ctrl_p{i}" for i in range(1, 6)]

    # Determine conditions
    if "object_scene_retrieval" in df.columns:
        conditions = sorted(df["object_scene_retrieval"].dropna().unique())
    else:
        conditions = ["all"]
    n_cond = len(conditions)
    trs = sorted(df["tr"].dropna().unique())

    # Layout: rows = conditions, cols = 5 panels
    fig, axes = plt.subplots(n_cond, 5, figsize=(28, 5.5 * n_cond), squeeze=False)
    x = np.arange(1, 6)
    w = 0.35

    print("\n--- Probability Diagnostics (by condition) ---")

    for row_idx, cond in enumerate(conditions):
        if cond == "all":
            dfc = df_tmp
        else:
            dfc = df_tmp[df_tmp["object_scene_retrieval"] == cond]
        n_sub = dfc["subject"].nunique()

        # === Panel A: Per-item probability with significance tests ===
        ax = axes[row_idx, 0]
        subj_ret = dfc.groupby("subject")[ret_p_cols].mean()
        subj_ctrl = dfc.groupby("subject")[ctrl_p_cols].mean()
        ret_means = subj_ret.mean().values
        ret_sems = subj_ret.sem().values
        ctrl_means = subj_ctrl.mean().values
        ctrl_sems = subj_ctrl.sem().values

        ax.bar(x - w / 2, ret_means, w, yerr=ret_sems, label="Retrieved",
               color="#3274A1", alpha=0.7, capsize=3)
        ax.bar(x + w / 2, ctrl_means, w, yerr=ctrl_sems, label="Control",
               color="#E1812C", alpha=0.7, capsize=3)

        # Paired t-tests per position
        ymax = max(np.max(ret_means + ret_sems), np.max(ctrl_means + ctrl_sems))
        for i in range(5):
            t, p = ttest_rel(subj_ret.iloc[:, i], subj_ctrl.iloc[:, i])
            sig = _sig_str(p)
            if sig != "n.s.":
                ax.text(x[i], ymax * 1.02, sig, ha="center", fontsize=9,
                        fontweight="bold", color="red")
            print(f"  {cond} — Pos {i + 1}: Ret={ret_means[i]:.5f} Ctrl={ctrl_means[i]:.5f} "
                  f"t={t:.2f} p={p:.4f} {sig}")

        # Overall test
        subj_ret_m = subj_ret.mean(axis=1)
        subj_ctrl_m = subj_ctrl.mean(axis=1)
        t_all, p_all = ttest_rel(subj_ret_m, subj_ctrl_m)
        sig_all = _sig_str(p_all)
        print(f"  {cond} — Overall: Ret={subj_ret_m.mean():.5f} Ctrl={subj_ctrl_m.mean():.5f} "
              f"t={t_all:.2f} p={p_all:.4f} {sig_all}")

        ax.set_xlabel("Item position in sequence")
        ax.set_ylabel("Mean P(item)")
        ax.set_title(f"{cond.upper()}: Per-item P (n={n_sub})\n"
                     f"Overall Ret vs Ctrl: t={t_all:.2f}, p={p_all:.3f} {sig_all}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(x)
        ax.spines[["top", "right"]].set_visible(False)

        # === Panel B: Per-item P across TRs (items peak at different TRs) ===
        ax = axes[row_idx, 1]
        for i in range(5):
            tr_means = []
            tr_sems = []
            for tr in trs:
                sv = dfc[dfc["tr"] == tr].groupby("subject")[f"ret_p{i + 1}"].mean()
                tr_means.append(sv.mean())
                tr_sems.append(sv.std() / np.sqrt(len(sv)) if len(sv) > 1 else 0)
            ax.plot(trs, tr_means, "o-", ms=3, lw=1.5, label=f"Item {i + 1}")
            ax.fill_between(trs, np.array(tr_means) - np.array(tr_sems),
                            np.array(tr_means) + np.array(tr_sems), alpha=0.15)
        ax.axhline(0.2, color="grey", ls=":", lw=0.7)
        ax.set_xlabel("TR")
        ax.set_ylabel("Mean P(item)")
        ax.set_title(f"{cond.upper()}: Retrieved per-item P across TRs\n"
                     "(items peak at different TRs → heterogeneity expected)")
        ax.legend(fontsize=6, frameon=False, ncol=5)
        ax.spines[["top", "right"]].set_visible(False)

        # === Panel C: Within-TR heterogeneity (expected — confirms item discrimination) ===
        ax = axes[row_idx, 2]
        ret_pm = np.array(dfc[ret_p_cols].values)
        ctrl_pm = np.array(dfc[ctrl_p_cols].values)
        ret_sd = np.nanstd(ret_pm, axis=1)
        ctrl_sd = np.nanstd(ctrl_pm, axis=1)

        # Subject-level test
        dfc2 = dfc.copy()
        dfc2["ret_sd"] = ret_sd
        dfc2["ctrl_sd"] = ctrl_sd
        subj_ret_sd = dfc2.groupby("subject")["ret_sd"].mean()
        subj_ctrl_sd = dfc2.groupby("subject")["ctrl_sd"].mean()
        t_het, p_het = ttest_rel(subj_ret_sd, subj_ctrl_sd)
        sig_het = _sig_str(p_het)

        ax.hist(ret_sd[~np.isnan(ret_sd)], bins=50, alpha=0.6, color="#3274A1",
                label=f"Retrieved (M={np.nanmean(ret_sd):.4f})")
        ax.hist(ctrl_sd[~np.isnan(ctrl_sd)], bins=50, alpha=0.6, color="#E1812C",
                label=f"Control (M={np.nanmean(ctrl_sd):.4f})")
        ax.set_xlabel("SD of 5 item probabilities (per TR)")
        ax.set_ylabel("Count")
        ax.set_title(f"{cond.upper()}: Within-TR heterogeneity\n"
                     f"(expected: classifier picks items)\n"
                     f"Ret vs Ctrl: t={t_het:.2f}, p={p_het:.3f} {sig_het}")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        print(f"  {cond} — Heterogeneity: Ret={subj_ret_sd.mean():.5f} "
              f"Ctrl={subj_ctrl_sd.mean():.5f} t={t_het:.2f} p={p_het:.4f} {sig_het}")

        # === Panel D: Across-TR balance (items equally represented when aggregated) ===
        ax = axes[row_idx, 3]
        # Average each item's P across TRs within trial, then across trials per subject
        trial_agg = dfc.groupby(["subject", "trial"])[ret_p_cols].mean()
        subj_balance = trial_agg.groupby("subject").mean()
        bal_means = subj_balance.mean().values
        bal_sems = subj_balance.sem().values

        colors_items = plt.cm.Set2(np.linspace(0, 1, 5))
        ax.bar(x, bal_means, yerr=bal_sems, color=colors_items, capsize=4, edgecolor="white")
        for sub in subj_balance.index:
            ax.plot(x, subj_balance.loc[sub].values, "o-", color="grey", alpha=0.2, ms=3)

        # Friedman test: are items balanced?
        fr_arrays = [subj_balance.iloc[:, i].values for i in range(5)]
        chi2_bal, p_bal = friedmanchisquare(*fr_arrays)
        sig_bal = _sig_str(p_bal)

        ax.axhline(np.mean(bal_means), color="grey", ls=":", lw=1)
        ax.set_xlabel("Item position in sequence")
        ax.set_ylabel("Mean P(item) aggregated across TRs")
        ax.set_title(f"{cond.upper()}: Across-TR item balance (Ret)\n"
                     f"Friedman chi2={chi2_bal:.2f}, p={p_bal:.3f} {sig_bal}\n"
                     "(should be n.s.: heterogeneity within TRs, not across)")
        ax.set_xticks(x)
        ax.spines[["top", "right"]].set_visible(False)

        print(f"  {cond} — Across-TR balance (Friedman): chi2={chi2_bal:.2f} p={p_bal:.4f} {sig_bal}")
        for i in range(5):
            print(f"    Item {i + 1}: M={bal_means[i]:.6f} ± {bal_sems[i]:.6f}")

        # === Panel E: Heterogeneity–|SODA| correlation (validates SODA mechanism) ===
        ax = axes[row_idx, 4]
        if "retrieved_soda_abs" in dfc.columns:
            soda_abs = dfc["retrieved_soda_abs"].values
            valid = ~np.isnan(ret_sd) & ~np.isnan(soda_abs)
            if valid.sum() > 10:
                ax.scatter(ret_sd[valid], soda_abs[valid], s=3, alpha=0.15, color="#3274A1")
                r, p = pearsonr(ret_sd[valid], soda_abs[valid])
                ax.set_xlabel("Within-TR P(item) SD")
                ax.set_ylabel("|SODA slope|")
                ax.set_title(f"{cond.upper()}: Heterogeneity ↔ |SODA|\n"
                             f"r={r:.3f}, p={p:.1e}\n"
                             "(expected: item discrimination enables sequentiality)")
                print(f"  {cond} — Heterogeneity–|SODA|: r={r:.3f} p={p:.1e}")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Probability Diagnostics — {roi}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_01_probability_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# SECTION 1b: STIMULUS-LEVEL BIAS
# =====================================================================

def plot_stimulus_bias(df, roi, out_dir):
    """Check whether individual stimuli / categories have systematic P biases."""
    if "full_structured_sequence_object" not in df.columns:
        print("  No sequence info — skipping stimulus bias.")
        return
    if "full_structured_sequence_category" not in df.columns:
        print("  No category info — skipping stimulus bias.")
        return

    ret_probs_matrix = np.array(df["retrieved_probs"].tolist())
    for i in range(5):
        df[f"_ret_p{i + 1}"] = ret_probs_matrix[:, i]

    def _get_seq_items(row):
        try:
            seq = ast.literal_eval(row["full_structured_sequence_object"])
            cue = row["retrieval_cue"]
            for session in seq.values():
                if cue in session:
                    return session[cue]
        except Exception:
            pass
        return None

    def _get_seq_cats(row):
        try:
            seq = ast.literal_eval(row["full_structured_sequence_category"])
            cue = row["retrieval_cue"]
            for session in seq.values():
                if cue in session:
                    return session[cue]
        except Exception:
            pass
        return None

    df["_seq_items"] = df.apply(_get_seq_items, axis=1)
    df["_seq_cats"] = df.apply(_get_seq_cats, axis=1)

    # Build long format: one row per item per TR
    rows_long = []
    for _, row in df.iterrows():
        items = row["_seq_items"]
        cats = row["_seq_cats"]
        if items is None or len(items) != 5:
            continue
        for pos_i in range(5):
            stim = items[pos_i]
            if stim is None:
                continue
            rows_long.append({
                "subject": row["subject"],
                "trial": row["trial"],
                "tr": row["tr"],
                "stimulus": stim.replace(".png", ""),
                "category": cats[pos_i] if cats and len(cats) == 5 else None,
                "prob": row[f"_ret_p{pos_i + 1}"],
            })

    dflong = pd.DataFrame(rows_long)
    if len(dflong) == 0:
        print("  Could not map stimuli — skipping.")
        return

    n_stim = dflong["stimulus"].nunique()
    print(f"\n--- Stimulus-Level Bias ({n_stim} stimuli) ---")

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))

    # === Panel A: Per-stimulus mean P (sorted) ===
    ax = axes[0, 0]
    subj_stim = dflong.groupby(["subject", "stimulus"])["prob"].mean().reset_index()
    stim_order = subj_stim.groupby("stimulus")["prob"].mean().sort_values(ascending=False).index.tolist()
    stim_means = [subj_stim[subj_stim["stimulus"] == s]["prob"].mean() for s in stim_order]
    stim_sems = [subj_stim[subj_stim["stimulus"] == s]["prob"].sem() for s in stim_order]

    grand = np.mean(stim_means)
    colors_bar = ["#E1812C" if abs(m - grand) > 2 * np.std(stim_means) else "#3274A1"
                  for m in stim_means]
    ax.bar(range(len(stim_order)), stim_means, yerr=stim_sems, color=colors_bar,
           alpha=0.7, capsize=1, edgecolor="white")
    ax.axhline(grand, color="red", ls="--", lw=1.5, label=f"Grand mean={grand:.4f}")
    ax.set_xticks(range(len(stim_order)))
    ax.set_xticklabels(stim_order, rotation=90, fontsize=5.5)
    ax.set_ylabel("Mean P(stimulus)")

    pivot_stim = subj_stim.pivot(index="subject", columns="stimulus", values="prob").dropna(axis=1)
    arrays_s = [pivot_stim.iloc[:, i].values for i in range(pivot_stim.shape[1])]
    chi2_s, p_s = friedmanchisquare(*arrays_s)
    sig_s = _sig_str(p_s)
    ax.set_title(f"Per-stimulus classifier P (sorted)\n"
                 f"Friedman chi2={chi2_s:.1f}, p={p_s:.1e} {sig_s}\n"
                 f"Range: {min(stim_means):.4f}–{max(stim_means):.4f} "
                 f"(ratio {max(stim_means)/min(stim_means):.1f}x)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"  Stimulus Friedman: chi2={chi2_s:.1f}, p={p_s:.1e} {sig_s}")
    print(f"  Range: {min(stim_means):.4f}–{max(stim_means):.4f} "
          f"(ratio {max(stim_means)/min(stim_means):.1f}x)")

    # === Panel B: Per-category mean P ===
    ax = axes[0, 1]
    dfcat = dflong.dropna(subset=["category"])
    if len(dfcat) > 0:
        subj_cat = dfcat.groupby(["subject", "category"])["prob"].mean().reset_index()
        cat_order = subj_cat.groupby("category")["prob"].mean().sort_values(ascending=False).index.tolist()
        cat_means = [subj_cat[subj_cat["category"] == c]["prob"].mean() for c in cat_order]
        cat_sems = [subj_cat[subj_cat["category"] == c]["prob"].sem() for c in cat_order]

        ax.bar(range(len(cat_order)), cat_means, yerr=cat_sems, color="#3274A1",
               alpha=0.7, capsize=3, edgecolor="white")
        ax.axhline(np.mean(cat_means), color="red", ls="--", lw=1.5)
        for sub in subj_cat["subject"].unique():
            sd = subj_cat[subj_cat["subject"] == sub].set_index("category").reindex(cat_order)
            ax.plot(range(len(cat_order)), sd["prob"].values, "o-", color="grey", alpha=0.15, ms=2)

        pivot_cat = subj_cat.pivot(index="subject", columns="category", values="prob").dropna(axis=1)
        arrays_c = [pivot_cat.iloc[:, i].values for i in range(pivot_cat.shape[1])]
        chi2_c, p_c = friedmanchisquare(*arrays_c)
        sig_c = _sig_str(p_c)
        ax.set_title(f"Per-category classifier P\nFriedman chi2={chi2_c:.1f}, p={p_c:.4f} {sig_c}")
        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(cat_order, rotation=45, fontsize=8, ha="right")
        ax.set_ylabel("Mean P(category)")
        print(f"  Category Friedman: chi2={chi2_c:.1f}, p={p_c:.4f} {sig_c}")
        for c in cat_order:
            m = subj_cat[subj_cat["category"] == c]["prob"].mean()
            print(f"    {c}: {m:.5f}")
    ax.spines[["top", "right"]].set_visible(False)

    # === Panel C: Stimulus bias SD within sequence vs |SODA| ===
    ax = axes[1, 0]
    stim_grand = dflong.groupby("stimulus")["prob"].mean().to_dict()
    trial_bias = []
    for (subj, trial), grp in df.groupby(["subject", "trial"]):
        row0 = grp.iloc[0]
        items = row0["_seq_items"]
        if items is None or len(items) != 5:
            continue
        stim_names = [it.replace(".png", "") for it in items if it is not None]
        if len(stim_names) != 5:
            continue
        grand_probs = [stim_grand.get(s, np.nan) for s in stim_names]
        if any(np.isnan(grand_probs)):
            continue
        trial_bias.append({
            "subject": subj, "trial": trial,
            "bias_sd": np.std(grand_probs),
            "soda_abs": grp["retrieved_soda_abs"].mean(),
        })

    dfbias = pd.DataFrame(trial_bias).dropna()
    if len(dfbias) > 10:
        ax.scatter(dfbias["bias_sd"], dfbias["soda_abs"], s=5, alpha=0.3, color="#3274A1")
        r, p_corr = spearmanr(dfbias["bias_sd"], dfbias["soda_abs"])
        ax.set_xlabel("SD of stimulus grand-mean biases within sequence")
        ax.set_ylabel("Mean |SODA| for that trial")
        ax.set_title(f"Does stimulus bias composition affect |SODA|?\n"
                     f"Spearman rho={r:.3f}, p={p_corr:.3f}")
        print(f"  Stimulus bias SD vs |SODA|: rho={r:.3f}, p={p_corr:.3f}")
    ax.spines[["top", "right"]].set_visible(False)

    # === Panel D: Median split — |SODA| in low vs high stimulus-bias trials ===
    ax = axes[1, 1]
    if len(dfbias) > 10:
        med = dfbias["bias_sd"].median()
        low = dfbias[dfbias["bias_sd"] <= med]
        high = dfbias[dfbias["bias_sd"] > med]

        subj_low = low.groupby("subject")["soda_abs"].mean()
        subj_high = high.groupby("subject")["soda_abs"].mean()
        common = subj_low.index.intersection(subj_high.index)

        if len(common) >= 3:
            t_ms, p_ms = ttest_rel(subj_high.loc[common], subj_low.loc[common])
            sig_ms = _sig_str(p_ms)
            means = [subj_low.loc[common].mean(), subj_high.loc[common].mean()]
            sems = [subj_low.loc[common].sem(), subj_high.loc[common].sem()]

            ax.bar([0, 1], means, yerr=sems, color=["#8CB8D0", "#E1812C"], alpha=0.7,
                   capsize=5, width=0.5, edgecolor="white")
            for sub in common:
                ax.plot([0, 1], [subj_low.loc[sub], subj_high.loc[sub]],
                        "o-", color="grey", alpha=0.3, ms=4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Low bias\nsequences", "High bias\nsequences"])
            ax.set_ylabel("Mean |SODA|")
            ax.set_title(f"|SODA| in low vs high stimulus-bias trials\n"
                         f"(median split on within-sequence bias SD)\n"
                         f"t={t_ms:.2f}, p={p_ms:.3f} {sig_ms}")
            print(f"  Low vs high bias: t={t_ms:.2f}, p={p_ms:.3f} {sig_ms}")
            print(f"    Low={means[0]:.5f}, High={means[1]:.5f}")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Stimulus-Level Classifier Bias — {roi}\n"
                 "Do individual stimuli have systematically different P?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_01b_stimulus_bias.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Clean up temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")


# =====================================================================
# SECTION 2: POSITION EFFECTS
# =====================================================================

def plot_position_effects(df, roi, out_dir):
    """Test whether later cue positions show higher sequentiality."""
    if "retrieved_position" not in df.columns:
        print("  No retrieved_position column — skipping.")
        return

    positions = sorted(df["retrieved_position"].dropna().unique())
    if len(positions) < 2:
        print("  Only one position value — skipping.")
        return

    metrics = {}
    for col, label in [
        ("retrieved_soda_abs", "|SODA| retrieved"),
        ("control_soda_abs", "|SODA| control"),
        ("retrieved_soda", "SODA signed"),
        ("retrieved_soda_shuffled_abs", "|SODA| shuffled"),
    ]:
        if col in df.columns:
            metrics[col] = label

    n_met = len(metrics)
    if n_met == 0:
        return

    fig, axes = plt.subplots(1, n_met, figsize=(5.5 * n_met, 5), squeeze=False)
    axes = axes[0]

    print("\n--- Position Effects (all trial types combined) ---")

    for ax, (col, label) in zip(axes, metrics.items()):
        subj_pos = df.groupby(["subject", "retrieved_position"])[col].mean().reset_index()

        pos_stats = subj_pos.groupby("retrieved_position")[col].agg(["mean", "std", "count"])
        pos_stats["sem"] = pos_stats["std"] / np.sqrt(pos_stats["count"])

        ax.bar(pos_stats.index, pos_stats["mean"], yerr=pos_stats["sem"],
               color="#3274A1", alpha=0.7, capsize=4, edgecolor="white")

        for sub in subj_pos["subject"].unique():
            sd = subj_pos[subj_pos["subject"] == sub]
            ax.plot(sd["retrieved_position"], sd[col], "o-", alpha=0.2, ms=3, color="grey")

        ax.set_xlabel("Cue position in sequence")
        ax.set_ylabel(label)
        ax.set_xticks([int(p) for p in positions])
        ax.spines[["top", "right"]].set_visible(False)

        valid = subj_pos.dropna(subset=[col])
        if len(valid) > 5:
            rho, p = spearmanr(valid["retrieved_position"], valid[col])
            ax.set_title(f"{label}\nSpearman rho={rho:.3f}, p={p:.3f}")
            print(f"  {label}: rho={rho:.3f}, p={p:.3f}")
        else:
            ax.set_title(label)

    fig.suptitle(
        f"Position Effects — {roi}\n"
        "Hypothesis: later cue position → more items to replay → higher sequentiality",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_02_position_effects.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# SECTION 3: SCENE vs OBJECT RETRIEVAL
# =====================================================================

def plot_trial_type_split(df, roi, out_dir):
    """Separate SODA analyses for scene and object retrieval."""
    if "object_scene_retrieval" not in df.columns:
        print("  No object_scene_retrieval column — skipping.")
        return

    trial_types = sorted(df["object_scene_retrieval"].dropna().unique())
    if len(trial_types) < 1:
        return

    # --- 3a: SODA timecourse per trial type ---
    fig, axes = plt.subplots(1, len(trial_types), figsize=(7 * len(trial_types), 5), squeeze=False)
    axes = axes[0]

    print("\n--- Trial-Type Split ---")

    for ax, ttype in zip(axes, trial_types):
        df_sub = df[df["object_scene_retrieval"] == ttype]
        n_trials = df_sub.groupby(["subject", "trial"]).ngroups

        for condition, color, label in [
            ("retrieved_soda", "#3274A1", "Retrieved"),
            ("control_soda", "#E1812C", "Control"),
            ("retrieved_soda_shuffled", "#8CB8D0", "Shuffled"),
        ]:
            if condition not in df_sub.columns:
                continue
            subj_tr = df_sub.groupby(["subject", "tr"])[condition].mean().reset_index()
            agg = subj_tr.groupby("tr")[condition].agg(["mean", "std", "count"])
            agg["sem"] = agg["std"] / np.sqrt(agg["count"])

            ax.plot(agg.index, agg["mean"], color=color, lw=2, label=label)
            ax.fill_between(agg.index, agg["mean"] - agg["sem"],
                            agg["mean"] + agg["sem"], color=color, alpha=0.2, linewidth=0)

        ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.set_xlabel("TR")
        ax.set_ylabel("Mean SODA slope")
        n_sub = df_sub["subject"].nunique()
        ax.set_title(f"{ttype.upper()} retrieval  (n_sub={n_sub}, n_trials={n_trials})")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"SODA Timecourse by Trial Type — {roi}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_03a_soda_timecourse_by_trialtype.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3b: Subject-level metric comparison per trial type ---
    metric_pairs = [
        ("retrieved_soda_abs", "control_soda_abs", "|SODA|"),
        ("retrieved_soda", "control_soda", "SODA signed"),
        ("retrieved_soda_abs", "retrieved_soda_shuffled_abs", "|SODA| ret vs shuf"),
    ]
    metric_pairs = [(r, c, l) for r, c, l in metric_pairs if r in df.columns and c in df.columns]

    n_met = len(metric_pairs)
    if n_met == 0:
        return

    fig, axes = plt.subplots(len(trial_types), n_met,
                             figsize=(5 * n_met, 5 * len(trial_types)), squeeze=False)

    for row_idx, ttype in enumerate(trial_types):
        df_sub = df[df["object_scene_retrieval"] == ttype]

        for col_idx, (ret_col, ctrl_col, label) in enumerate(metric_pairs):
            ax = axes[row_idx, col_idx]

            subj = df_sub.groupby("subject")[[ret_col, ctrl_col]].mean().dropna()
            if len(subj) < 3:
                ax.set_title(f"{ttype}: {label}\n(insufficient data)")
                continue

            # Paired dot plot
            for _, srow in subj.iterrows():
                ax.plot([0, 1], [srow[ret_col], srow[ctrl_col]], "o-", color="grey", alpha=0.4, ms=5)
            ax.bar([0, 1], [subj[ret_col].mean(), subj[ctrl_col].mean()],
                   color=["#3274A1", "#E1812C"], alpha=0.35, width=0.5)
            ax.errorbar([0, 1],
                        [subj[ret_col].mean(), subj[ctrl_col].mean()],
                        yerr=[subj[ret_col].sem(), subj[ctrl_col].sem()],
                        fmt="none", color="black", capsize=5, lw=2)

            t, p = ttest_rel(subj[ret_col], subj[ctrl_col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_xticks([0, 1])
            x_labels = [ret_col.split("_soda")[0].replace("retrieved", "Ret").replace("control", "Ctrl"),
                         ctrl_col.split("_soda")[0].replace("retrieved", "Ret").replace("control", "Ctrl")]
            # Cleaner labels
            if "shuf" in ctrl_col:
                x_labels = ["Retrieved", "Shuffled"]
            else:
                x_labels = ["Retrieved", "Control"]
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(label)
            ax.set_title(f"{ttype}: {label}\nt={t:.2f}, p={p:.3f} {sig} (n={len(subj)})")
            ax.spines[["top", "right"]].set_visible(False)

            print(f"  {ttype} — {label}: t={t:.2f}, p={p:.3f} {sig}")

    fig.suptitle(f"Metrics by Trial Type — {roi}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_03b_metrics_by_trialtype.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3c: Position effects split by trial type ---
    if "retrieved_position" not in df.columns or "retrieved_soda_abs" not in df.columns:
        return

    fig, axes = plt.subplots(1, len(trial_types), figsize=(7 * len(trial_types), 5), squeeze=False)
    axes = axes[0]

    print("\n--- Position Effects by Trial Type ---")

    for ax, ttype in zip(axes, trial_types):
        df_sub = df[df["object_scene_retrieval"] == ttype]

        subj_pos = df_sub.groupby(["subject", "retrieved_position"])["retrieved_soda_abs"].mean().reset_index()
        pos_stats = subj_pos.groupby("retrieved_position")["retrieved_soda_abs"].agg(["mean", "std", "count"])
        pos_stats["sem"] = pos_stats["std"] / np.sqrt(pos_stats["count"])

        ax.bar(pos_stats.index, pos_stats["mean"], yerr=pos_stats["sem"],
               color="#3274A1", alpha=0.7, capsize=4, edgecolor="white")
        for sub in subj_pos["subject"].unique():
            sd = subj_pos[subj_pos["subject"] == sub]
            ax.plot(sd["retrieved_position"], sd["retrieved_soda_abs"],
                    "o-", alpha=0.2, ms=3, color="grey")

        ax.set_xlabel("Cue position in sequence")
        ax.set_ylabel("|SODA|")
        ax.spines[["top", "right"]].set_visible(False)

        valid = subj_pos.dropna(subset=["retrieved_soda_abs"])
        if len(valid) > 5:
            rho, p = spearmanr(valid["retrieved_position"], valid["retrieved_soda_abs"])
            ax.set_title(f"{ttype}: |SODA| by position\nrho={rho:.3f}, p={p:.3f}")
            print(f"  {ttype} position effect: rho={rho:.3f}, p={p:.3f}")
        else:
            ax.set_title(f"{ttype}: |SODA| by position")

    fig.suptitle(f"Position Effects by Trial Type — {roi}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_03c_position_by_trialtype.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# SECTION 4: SHUFFLE COMPARISON
# =====================================================================

def plot_shuffle_comparison(df, roi, out_dir):
    """Compare random vs selected shuffle distributions."""
    random_col = "retrieved_soda_shuffled_abs"
    selected_col = "retrieved_soda_shuffled_selected_abs"
    ret_col = "retrieved_soda_abs"

    has_random = random_col in df.columns
    has_selected = selected_col in df.columns
    has_ret = ret_col in df.columns

    if not (has_random or has_selected):
        print("  Missing shuffle columns — skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    print("\n--- Shuffle Comparison ---")

    # 4a: Distribution
    ax = axes[0]
    if has_random:
        vals = df[random_col].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color="#8CB8D0",
                label=f"Random (M={vals.mean():.5f})")
    if has_selected:
        vals = df[selected_col].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color="#A0D0A0",
                label=f"Selected (M={vals.mean():.5f})")
    if has_ret:
        vals = df[ret_col].dropna()
        ax.axvline(vals.mean(), color="#3274A1", ls="--", lw=2, label=f"Retrieved mean ({vals.mean():.5f})")
    ax.set_xlabel("|SODA| shuffled")
    ax.set_ylabel("Count")
    ax.set_title("Shuffle type distributions")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # 4b: Subject-level comparison
    ax = axes[1]
    cols_avail = []
    labels_avail = []
    colors_avail = []
    if has_ret:
        cols_avail.append(ret_col)
        labels_avail.append("Retrieved")
        colors_avail.append("#3274A1")
    if has_random:
        cols_avail.append(random_col)
        labels_avail.append("Shuf Random")
        colors_avail.append("#8CB8D0")
    if has_selected:
        cols_avail.append(selected_col)
        labels_avail.append("Shuf Selected")
        colors_avail.append("#A0D0A0")

    subj = df.groupby("subject")[cols_avail].mean().dropna()
    if len(subj) >= 3:
        bp = ax.boxplot([subj[c].values for c in cols_avail],
                        positions=range(len(cols_avail)), widths=0.5,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors_avail):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

        rng = np.random.default_rng(42)
        for i, (c, color) in enumerate(zip(cols_avail, colors_avail)):
            vals = subj[c].values
            jitter = rng.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color=color, edgecolor="k", linewidth=0.5, s=30, zorder=5, alpha=0.8)

        ax.set_xticks(range(len(labels_avail)))
        ax.set_xticklabels(labels_avail, fontsize=9)
        ax.set_ylabel("Mean |SODA|")
        ax.set_title("Subject-level means")

        # T-tests
        if has_ret and has_random:
            t, p = ttest_rel(subj[ret_col], subj[random_col])
            print(f"  Retrieved vs random shuffle:   t={t:.2f}, p={p:.3f}")
        if has_ret and has_selected:
            t, p = ttest_rel(subj[ret_col], subj[selected_col])
            print(f"  Retrieved vs selected shuffle: t={t:.2f}, p={p:.3f}")
        if has_random and has_selected:
            t, p = ttest_rel(subj[random_col], subj[selected_col])
            print(f"  Random vs selected shuffle:    t={t:.2f}, p={p:.3f}")

    ax.spines[["top", "right"]].set_visible(False)

    # 4c: Effect of shuffle type on SODA difference (ret − shuf)
    ax = axes[2]
    if has_ret and has_random and has_selected:
        diff_random = subj[ret_col] - subj[random_col]
        diff_selected = subj[ret_col] - subj[selected_col]

        for sub_idx in range(len(diff_random)):
            ax.plot([0, 1], [diff_random.iloc[sub_idx], diff_selected.iloc[sub_idx]],
                    "o-", color="grey", alpha=0.4, ms=5)
        ax.bar([0, 1], [diff_random.mean(), diff_selected.mean()],
               color=["#8CB8D0", "#A0D0A0"], alpha=0.5, width=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ret − Random", "Ret − Selected"], fontsize=9)
        ax.set_ylabel("Mean |SODA| difference")
        ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.set_title("Effect of shuffle type on\nretrieval advantage")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Shuffle Comparison — {roi}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"{roi}_04_shuffle_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    roi = sys.argv[1] if len(sys.argv) > 1 else None
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{pre_post}", "SODA_results_csv")

    csv_files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No SODA result CSVs found in: {csv_dir}")

    available_rois = [
        os.path.basename(f).replace("_soda_results_per_combination.csv", "")
        for f in csv_files
    ]

    if roi:
        if roi not in available_rois:
            raise ValueError(f"ROI '{roi}' not found. Available: {available_rois}")
        rois_to_process = [roi]
    else:
        rois_to_process = available_rois

    print(f"Processing {len(rois_to_process)} ROIs: {rois_to_process}")
    print(f"Session: {pre_post}")

    for roi_name in rois_to_process:
        print(f"\n{'=' * 60}")
        print(f"ROI: {roi_name}")
        print(f"{'=' * 60}")

        out_dir = opj(script_dir, "4.6b_SODA_diagnostics", roi_name)
        os.makedirs(out_dir, exist_ok=True)

        df = load_soda_results(csv_dir, roi_name)
        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects, "
              f"{df.groupby(['subject', 'trial']).ngroups} trials")

        plot_probability_diagnostics(df, roi_name, out_dir)
        plot_stimulus_bias(df, roi_name, out_dir)
        plot_position_effects(df, roi_name, out_dir)
        plot_trial_type_split(df, roi_name, out_dir)
        plot_shuffle_comparison(df, roi_name, out_dir)

        print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
