# -*- coding: utf-8 -*-
"""Bayesian model comparison: replay (windowed sinusoid) vs null (noise).

Compares two models per trial via BIC-based approximate Bayes factor:
  M0 (null):   SODA ~ i.i.d. Gaussian noise (2 params: mean, variance)
  M1 (replay): SODA ~ windowed single-cycle sinusoid + noise (5 params)

Positive log10 BF indicates evidence for replay.
Negative delta_BIC indicates replay is preferred.

Usage
-----
    python 4.9_bayesian_replay.py [ROI] [presleep|postsleep]
"""

import os
import sys
import glob
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt

# Re-use the windowed sinusoid from 4.7 (numeric prefix requires importlib.util)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "sinus_1cycle",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "4.7_sinus_1cycle.py"),
)
_sinus = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sinus)
fit_windowed_sinusoid = _sinus.fit_windowed_sinusoid
_windowed_for_curvefit = _sinus._windowed_for_curvefit


# =====================================================================
# 1. BAYESIAN MODEL COMPARISON
# =====================================================================

def bayesian_model_comparison(soda_values, trs=None):
    """Compare null (Gaussian noise) vs replay (windowed sinusoid) via BIC.

    Parameters
    ----------
    soda_values : array (n_trs,)
        SODA slope values at each TR for a single trial.
    trs : array, optional
        TR indices. Defaults to 1..N.

    Returns
    -------
    dict with log10_bf, delta_bic, preferred
        log10_bf: positive = evidence for replay
        delta_bic: negative = replay preferred (BIC_replay - BIC_null)
    """
    soda_values = np.asarray(soda_values, dtype=float)
    valid = ~np.isnan(soda_values)
    y = soda_values[valid]
    n = len(y)

    if n < 5:
        return {"log10_bf": np.nan, "delta_bic": np.nan, "preferred": "insufficient_data"}

    if trs is None:
        trs = np.arange(1, len(soda_values) + 1, dtype=float)
    t = trs[valid]

    # M0: Null — i.i.d. Gaussian (2 params: mean, variance)
    sigma2_null = np.var(y)
    if sigma2_null <= 0:
        sigma2_null = 1e-10
    ll_null = -0.5 * n * np.log(2 * np.pi * sigma2_null) - 0.5 * n
    bic_null = -2 * ll_null + 2 * np.log(n)

    # M1: Replay — windowed sinusoid + noise (5 params: A, λ, d, c, σ²)
    w_fit = fit_windowed_sinusoid(t, y)

    if w_fit["fit_success"]:
        y_hat = _windowed_for_curvefit(t, w_fit["amplitude"], w_fit["duration"],
                                       w_fit["onset"], w_fit["offset"])
        residuals = y - y_hat
        sigma2_replay = np.var(residuals)
        if sigma2_replay <= 0:
            sigma2_replay = 1e-10
        ll_replay = (-0.5 * n * np.log(2 * np.pi * sigma2_replay)
                     - 0.5 * np.sum(residuals**2) / sigma2_replay)
        bic_replay = -2 * ll_replay + 5 * np.log(n)
    else:
        bic_replay = np.inf

    delta_bic = bic_replay - bic_null  # negative = replay preferred
    if np.isfinite(delta_bic):
        bf = np.exp(-0.5 * delta_bic)
        log10_bf = np.log10(bf) if bf > 0 else np.nan
    else:
        log10_bf = np.nan

    return {
        "log10_bf": log10_bf,
        "delta_bic": delta_bic,
        "preferred": "replay" if delta_bic < 0 else "null",
    }


# =====================================================================
# 2. TRIAL-LEVEL COMPUTATION
# =====================================================================

SODA_COLS = {
    "retrieved": "retrieved_soda",
    "control": "control_soda",
    "shuf_retrieved": "retrieved_soda_shuffled",
    "shuf_control": "control_soda_shuffled",
    "shuf_sel_retrieved": "retrieved_soda_shuffled_selected",
    "shuf_sel_control": "control_soda_shuffled_selected",
}


def compute_trial_level_bayes(
    df: pd.DataFrame,
    soda_cols: dict | None = None,
) -> pd.DataFrame:
    """Compute Bayesian replay vs null comparison per trial for all conditions."""
    if soda_cols is None:
        soda_cols = SODA_COLS

    group_cols = [c for c in ["subject", "trial", "object_scene_retrieval", "retrieval_target"]
                  if c in df.columns]
    results = []

    for keys, trial_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trial_info = dict(zip(group_cols, keys))
        trial_df = trial_df.sort_values("tr")
        trs = trial_df["tr"].values

        for cond_label, col_name in soda_cols.items():
            if col_name not in trial_df.columns:
                continue

            vals = trial_df[col_name].values
            bmc = bayesian_model_comparison(vals, trs)

            trial_info[f"{cond_label}_log10_bf"] = bmc["log10_bf"]
            trial_info[f"{cond_label}_delta_bic"] = bmc["delta_bic"]
            trial_info[f"{cond_label}_preferred"] = bmc["preferred"]

        if "behav_correct" in trial_df.columns:
            bc = trial_df["behav_correct"].dropna()
            trial_info["behav_correct"] = bc.iloc[0] if len(bc) > 0 else np.nan

        if "retrieved_position" in trial_df.columns:
            rp = trial_df["retrieved_position"].dropna()
            trial_info["retrieved_position"] = rp.iloc[0] if len(rp) > 0 else np.nan

        results.append(trial_info)

    return pd.DataFrame(results)


# =====================================================================
# 3. PLOTTING
# =====================================================================

COND_STYLES = {
    "retrieved":          {"color": "#3274A1", "label": "Retrieved"},
    "control":            {"color": "#E1812C", "label": "Control"},
    "shuf_retrieved":     {"color": "#8CB8D0", "label": "Shuf Ret"},
    "shuf_control":       {"color": "#F0C08A", "label": "Shuf Ctrl"},
    "shuf_sel_retrieved": {"color": "#A0D0A0", "label": "Shuf Sel Ret"},
    "shuf_sel_control":   {"color": "#D0A0D0", "label": "Shuf Sel Ctrl"},
}


def plot_bayes_summary(metrics_df, save_path=None):
    """Two-panel figure: log10 BF distribution + boxplot across conditions."""
    conditions = list(SODA_COLS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: log10 BF distribution for retrieved
    ax = axes[0]
    bf_col = "retrieved_log10_bf"
    if bf_col in metrics_df.columns:
        vals = metrics_df[bf_col].dropna()
        ax.hist(vals, bins=40, color="#3274A1", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", ls="--", lw=1.5)
        n_replay = (vals > 0).sum()
        n_null = (vals <= 0).sum()
        ax.set_xlabel("log$_{10}$ Bayes factor (replay vs null)")
        ax.set_ylabel("Count")
        ax.set_title(f"Model evidence (retrieved)\nReplay: {n_replay} | Null: {n_null}")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: log10 BF boxplot across conditions
    ax = axes[1]
    bf_cols = [f"{c}_log10_bf" for c in conditions]
    available = [c for c in bf_cols if c in metrics_df.columns]

    if len(available) >= 2 and "subject" in metrics_df.columns:
        subj = metrics_df.groupby("subject")[available].mean().dropna()
        if len(subj) >= 3:
            plot_data = [subj[c].values for c in available]
            plot_labels = [COND_STYLES.get(conditions[bf_cols.index(c)], {}).get("label", c)
                           for c in available]
            plot_colors = [COND_STYLES.get(conditions[bf_cols.index(c)], {}).get("color", "grey")
                           for c in available]

            bp = ax.boxplot(plot_data, positions=range(len(plot_data)), widths=0.5,
                            patch_artist=True, showfliers=False)
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
                           color=color, edgecolor="k", linewidth=0.5,
                           s=25, zorder=5, alpha=0.8)

            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Mean log$_{10}$ BF")
            ax.set_title("Bayesian evidence across conditions")

            ret_col = "retrieved_log10_bf"
            ctrl_col = "control_log10_bf"
            if ret_col in available and ctrl_col in available:
                t_stat, p_val = ttest_rel(subj[ret_col], subj[ctrl_col])
                sig = f"p={p_val:.1e}" if p_val < 0.001 else f"p={p_val:.3f}"
                print(f"  Ret vs Ctrl log10 BF: t({len(subj[ret_col])-1})={t_stat:.3f}, {sig}")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# 4. ENTRY POINT
# =====================================================================

def main():
    roi_filter = sys.argv[1] if len(sys.argv) > 1 else None
    pre_post = sys.argv[2] if len(sys.argv) > 2 else "presleep"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = opj(script_dir, f"4.6_SODA_based_on_singlesubj_4.4_{pre_post}", "SODA_results_csv")

    csv_files = sorted(glob.glob(opj(csv_dir, "*_soda_results_per_combination.csv")))
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

    numeric_cols = list(SODA_COLS.values()) + ["tr", "behav_correct"]
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
        out_dir = opj(script_dir, "4.9_bayesian_replay", roi)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        print("Computing Bayesian model comparison per trial...")

        metrics_df = compute_trial_level_bayes(df)
        print(f"Computed for {len(metrics_df)} trials")

        # Report replay preference rates
        for cond in ["retrieved", "control"]:
            pref_col = f"{cond}_preferred"
            if pref_col in metrics_df.columns:
                n_replay = (metrics_df[pref_col] == "replay").sum()
                n_total = metrics_df[pref_col].notna().sum()
                if n_total > 0:
                    print(f"  {cond}: replay preferred in {n_replay}/{n_total} trials ({100*n_replay/n_total:.1f}%)")

        metrics_df.to_csv(opj(out_dir, f"{roi}_bayesian_results_{pre_post}.csv"), index=False)

        print("Generating plots...")
        plot_bayes_summary(metrics_df, save_path=opj(out_dir, f"{roi}_bayesian_summary_{pre_post}.png"))

        roi_summary = {"ROI": roi, "n_subjects": df["subject"].nunique(), "n_trials": len(metrics_df)}

        for cond_a, cond_b, label in [
            ("retrieved_log10_bf", "control_log10_bf", "ret_vs_ctrl"),
            ("retrieved_log10_bf", "shuf_retrieved_log10_bf", "ret_vs_shuf"),
            ("retrieved_log10_bf", "shuf_sel_retrieved_log10_bf", "ret_vs_shuf_sel"),
        ]:
            if cond_a in metrics_df.columns and cond_b in metrics_df.columns:
                subj = metrics_df.groupby("subject")[[cond_a, cond_b]].mean().dropna()
                if len(subj) >= 3:
                    t, p = ttest_rel(subj[cond_a], subj[cond_b])
                    print(f"  {label}: t={t:.3f}, p={p:.4f} (n={len(subj)})")
                    roi_summary[f"t_{label}"] = t
                    roi_summary[f"p_{label}"] = p

        all_roi_summaries.append(roi_summary)
        print(f"Done with ROI {roi}.")

    summary_dir = opj(script_dir, "4.9_bayesian_replay", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = opj(summary_dir, f"bayesian_summary_{pre_post}.csv")
    summary_df = pd.DataFrame(all_roi_summaries)
    # Merge with existing summary to avoid overwriting other ROIs
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[~existing["ROI"].isin(summary_df["ROI"])]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-ROI summary saved to {summary_path}.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
