# -*- coding: utf-8 -*-
"""Group-level RSA: compare presleep vs postsleep RDMs.

Model RDMs tested:
  - Context model (scenes): scenes in same day/night context are similar
  - Context model (objects): object categories in same day/night context are similar

Analyses:
  A. Representational stability (presleep-postsleep RDM correlation)
  B. Element-wise distance changes by pair type (within/between context)
  C. Model RDM comparisons across sessions and ROIs

Usage
-----
    python 03_compare_sessions.py --roi vis
    python 03_compare_sessions.py            # runs all ROIs
"""

import argparse
import os
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from matplotlib import pyplot as plt

from config import SCRIPT_DIR, RSA_ROIS
from utils import (
    parse_subject_id,
    load_localizer_category_mapping,
    load_encoding_behavior,
    build_sequence_structure,
)


# =====================================================================
# Helper functions
# =====================================================================

def fisher_z(r):
    """Fisher z-transform a correlation coefficient."""
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def upper_triangle(matrix):
    """Extract upper triangle of a symmetric matrix as a vector."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def load_rdm_csv(path):
    """Load an RDM CSV (with condition labels as index/columns)."""
    df = pd.read_csv(path, index_col=0)
    return df.values, list(df.columns)


def discover_subjects(session, roi, modality):
    """Find all subjects with RDMs for a given session/roi/modality."""
    rdm_dir = opj(SCRIPT_DIR, "02_compute_rdms", session, "rdms")
    if not os.path.exists(rdm_dir):
        return []
    subjects = []
    for sub_dir in sorted(os.listdir(rdm_dir)):
        if not sub_dir.startswith("sub-"):
            continue
        rdm_file = opj(
            rdm_dir, sub_dir,
            f"{sub_dir}_{session}_{roi}_{modality}s_rdm.csv",
        )
        if os.path.exists(rdm_file):
            subjects.append(sub_dir)
    return subjects


# =====================================================================
# Model RDM builders
# =====================================================================

def build_context_model_rdm(conditions, condition_to_context):
    """Context (day/night) model: same context = 0, different context = 1.

    For scenes: scenes within the same day/night context are similar.
    For objects: object categories sharing the same context are similar.
    """
    n = len(conditions)
    rdm = np.full((n, n), np.nan)
    for i in range(n):
        rdm[i, i] = 0
        for j in range(i + 1, n):
            ctx_i = condition_to_context.get(conditions[i])
            ctx_j = condition_to_context.get(conditions[j])
            if ctx_i is not None and ctx_j is not None:
                val = 0.0 if ctx_i == ctx_j else 1.0
                rdm[i, j] = val
                rdm[j, i] = val
    return rdm


def build_position_model_rdm(conditions, condition_to_context, context_to_ordered):
    """Sequence position distance within context.

    Within same context: |position_i - position_j|
    Across contexts: max_distance + 1
    """
    n = len(conditions)
    rdm = np.full((n, n), np.nan)

    # Build condition -> position mapping
    cond_to_pos = {}
    for ctx, items in context_to_ordered.items():
        for pos, item in enumerate(items):
            cond_to_pos[item] = (ctx, pos)

    max_dist = max(len(v) for v in context_to_ordered.values()) if context_to_ordered else 5

    for i in range(n):
        rdm[i, i] = 0
        for j in range(i + 1, n):
            info_i = cond_to_pos.get(conditions[i])
            info_j = cond_to_pos.get(conditions[j])
            if info_i is None or info_j is None:
                continue
            ctx_i, pos_i = info_i
            ctx_j, pos_j = info_j
            if ctx_i == ctx_j:
                val = abs(pos_i - pos_j)
            else:
                val = max_dist + 1
            rdm[i, j] = val
            rdm[j, i] = val

    return rdm


# =====================================================================
# Analysis A: Representational stability
# =====================================================================

def analysis_stability(pre_rdms, post_rdms, subjects, conditions, modality, out_dir):
    """Correlate presleep and postsleep RDMs per subject."""
    print(f"\n  --- A. Representational stability ({modality}) ---")

    correlations = []
    for sub, pre, post in zip(subjects, pre_rdms, post_rdms):
        pre_vec = upper_triangle(pre)
        post_vec = upper_triangle(post)
        valid = ~(np.isnan(pre_vec) | np.isnan(post_vec))
        if valid.sum() < 3:
            continue
        r, p = spearmanr(pre_vec[valid], post_vec[valid])
        correlations.append({"subject": sub, "spearman_r": r, "fisher_z": fisher_z(r)})

    df = pd.DataFrame(correlations)
    if len(df) < 3:
        print("    Too few subjects for statistics.")
        return df

    z_vals = df["fisher_z"].values
    t, p = ttest_1samp(z_vals, 0)
    print(f"    Mean Fisher-z: {z_vals.mean():.3f} (SD={z_vals.std():.3f})")
    print(f"    One-sample t-test: t({len(z_vals)-1})={t:.3f}, p={p:.4f}")

    df.to_csv(opj(out_dir, f"stability_{modality}.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(df)), df["spearman_r"], color="#3274A1", alpha=0.7)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Spearman r (pre vs post)")
    ax.set_title(f"Representational stability ({modality})\n"
                 f"mean z={z_vals.mean():.3f}, t={t:.2f}, p={p:.3f}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"stability_{modality}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


# =====================================================================
# Analysis B: Element-wise distance changes by context
# =====================================================================

def analysis_distance_changes(
    pre_rdms, post_rdms, subjects, conditions, modality,
    condition_to_context, out_dir,
):
    """Compare pairwise distances between sessions, grouped by within/between context."""
    print(f"\n  --- B. Distance changes ({modality}) ---")

    n_conds = len(conditions)
    idx = np.triu_indices(n_conds, k=1)

    # Classify pairs
    pair_types = []
    for i, j in zip(idx[0], idx[1]):
        ci, cj = conditions[i], conditions[j]
        ctx_i = condition_to_context.get(ci)
        ctx_j = condition_to_context.get(cj)
        if ctx_i is not None and ctx_j is not None:
            if ctx_i == ctx_j:
                pair_types.append("within_context")
            else:
                pair_types.append("between_context")
        else:
            pair_types.append("unknown")
    pair_types = np.array(pair_types)

    # Compute per-subject delta distances
    results = []
    for sub, pre, post in zip(subjects, pre_rdms, post_rdms):
        pre_vec = upper_triangle(pre)
        post_vec = upper_triangle(post)
        delta = post_vec - pre_vec

        for ptype in ["within_context", "between_context"]:
            mask = pair_types == ptype
            if mask.sum() == 0:
                continue
            valid = ~(np.isnan(delta[mask]))
            if valid.sum() == 0:
                continue
            results.append({
                "subject": sub,
                "pair_type": ptype,
                "mean_pre": float(np.nanmean(pre_vec[mask])),
                "mean_post": float(np.nanmean(post_vec[mask])),
                "mean_delta": float(np.nanmean(delta[mask])),
                "n_pairs": int(mask.sum()),
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("    No pair-type data available.")
        return df

    df.to_csv(opj(out_dir, f"distance_changes_{modality}.csv"), index=False)

    # Statistics
    for ptype in ["within_context", "between_context"]:
        vals = df.loc[df["pair_type"] == ptype, "mean_delta"].values
        if len(vals) >= 3:
            t, p = ttest_1samp(vals, 0)
            print(f"    {ptype}: mean delta={vals.mean():.4f}, t={t:.3f}, p={p:.4f}")

    # Compare within vs between
    within = df.loc[df["pair_type"] == "within_context"].set_index("subject")["mean_delta"]
    between = df.loc[df["pair_type"] == "between_context"].set_index("subject")["mean_delta"]
    common = within.index.intersection(between.index)
    if len(common) >= 3:
        t, p = ttest_rel(within[common], between[common])
        print(f"    Within vs Between delta: t={t:.3f}, p={p:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Distances by session and pair type
    ax = axes[0]
    for ptype, color in [("within_context", "#3274A1"), ("between_context", "#E1812C")]:
        sub_df = df[df["pair_type"] == ptype]
        for sess_col, x_offset, label in [
            ("mean_pre", -0.15, "presleep"), ("mean_post", 0.15, "postsleep")
        ]:
            vals = sub_df[sess_col].values
            x = 0 if ptype == "within_context" else 1
            ax.bar(x + x_offset, np.mean(vals), width=0.25,
                   yerr=np.std(vals) / np.sqrt(len(vals)),
                   color=color if "pre" in sess_col else color,
                   alpha=0.5 if "pre" in sess_col else 0.8,
                   capsize=4, label=f"{label}" if x == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within\ncontext", "Between\ncontext"])
    ax.set_ylabel("Crossnobis distance")
    ax.set_title(f"Distances by context ({modality})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Delta distances
    ax = axes[1]
    for ptype, color in [("within_context", "#3274A1"), ("between_context", "#E1812C")]:
        vals = df.loc[df["pair_type"] == ptype, "mean_delta"].values
        x = 0 if ptype == "within_context" else 1
        ax.bar(x, np.mean(vals), width=0.5,
               yerr=np.std(vals) / np.sqrt(len(vals)),
               color=color, alpha=0.7, capsize=5)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within\ncontext", "Between\ncontext"])
    ax.set_ylabel("Delta distance\n(postsleep - presleep)")
    ax.set_title(f"Distance changes ({modality})")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(opj(out_dir, f"distance_changes_{modality}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


# =====================================================================
# Analysis C: Model RDM comparisons
# =====================================================================

def analysis_model_rdms(
    pre_rdms, post_rdms, subjects, conditions, modality,
    model_rdms, out_dir,
):
    """Correlate empirical RDMs with model RDMs, compare across sessions."""
    print(f"\n  --- C. Model RDM comparisons ({modality}) ---")

    if not model_rdms:
        print("    No model RDMs available. Skipping.")
        return pd.DataFrame()

    results = []
    for sub, pre, post in zip(subjects, pre_rdms, post_rdms):
        for model_name, model_rdm in model_rdms.items():
            model_vec = upper_triangle(model_rdm)

            for sess_name, emp_rdm in [("presleep", pre), ("postsleep", post)]:
                emp_vec = upper_triangle(emp_rdm)
                valid = ~(np.isnan(emp_vec) | np.isnan(model_vec))
                if valid.sum() < 3:
                    continue
                r, _ = spearmanr(emp_vec[valid], model_vec[valid])
                results.append({
                    "subject": sub,
                    "session": sess_name,
                    "model": model_name,
                    "spearman_r": r,
                    "fisher_z": fisher_z(r),
                })

    df = pd.DataFrame(results)
    if df.empty:
        print("    No model correlations computed.")
        return df

    df.to_csv(opj(out_dir, f"model_correlations_{modality}.csv"), index=False)

    # Statistics per model
    for model_name in model_rdms:
        print(f"\n    Model: {model_name}")
        for sess in ["presleep", "postsleep"]:
            vals = df.loc[
                (df["model"] == model_name) & (df["session"] == sess), "fisher_z"
            ].values
            if len(vals) >= 3:
                t, p = ttest_1samp(vals, 0)
                print(f"      {sess}: mean z={vals.mean():.3f}, t={t:.3f}, p={p:.4f}")

        # Paired comparison across sessions
        pre_z = df.loc[
            (df["model"] == model_name) & (df["session"] == "presleep")
        ].set_index("subject")["fisher_z"]
        post_z = df.loc[
            (df["model"] == model_name) & (df["session"] == "postsleep")
        ].set_index("subject")["fisher_z"]
        common = pre_z.index.intersection(post_z.index)
        if len(common) >= 3:
            t, p = ttest_rel(post_z[common], pre_z[common])
            print(f"      post vs pre: t={t:.3f}, p={p:.4f}")

    # Plot: model correlations by session
    models = list(model_rdms.keys())
    fig, ax = plt.subplots(figsize=(3 + 2 * len(models), 4))

    x_positions = np.arange(len(models))
    width = 0.3
    for sess, color, offset in [
        ("presleep", "#3274A1", -width / 2),
        ("postsleep", "#E1812C", width / 2),
    ]:
        means, sems = [], []
        for model_name in models:
            vals = df.loc[
                (df["model"] == model_name) & (df["session"] == sess), "fisher_z"
            ].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 0 else 0)
        ax.bar(x_positions + offset, means, width=width, yerr=sems,
               color=color, alpha=0.7, capsize=4, label=sess)

    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Fisher-z(Spearman r)")
    ax.set_title(f"Model RDM correlations ({modality})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"model_correlations_{modality}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


# =====================================================================
# Visualization: Group-average RDMs
# =====================================================================

def plot_group_rdms(pre_rdms, post_rdms, conditions, modality, out_dir):
    """Plot group-average presleep, postsleep, and difference RDMs."""
    pre_mean = np.nanmean(pre_rdms, axis=0)
    post_mean = np.nanmean(post_rdms, axis=0)
    diff = post_mean - pre_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, is_diff in zip(
        axes,
        [pre_mean, post_mean, diff],
        ["Presleep (group mean)", "Postsleep (group mean)", "Postsleep - Presleep"],
        [False, False, True],
    ):
        if is_diff:
            vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
            im = ax.imshow(data, cmap="coolwarm", aspect="equal", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(conditions, fontsize=7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Group RDMs ({modality})", fontsize=12)
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"group_rdms_{modality}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_rdms(model_rdms, conditions, modality, out_dir):
    """Visualize the model RDMs."""
    n_models = len(model_rdms)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    for ax, (name, rdm) in zip(axes[0], model_rdms.items()):
        im = ax.imshow(rdm, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(conditions, fontsize=7)
        ax.set_title(f"Model: {name}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Model RDMs ({modality})", fontsize=12)
    fig.tight_layout()
    fig.savefig(opj(out_dir, f"model_rdms_{modality}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Cross-ROI summary
# =====================================================================

def cross_roi_summary(all_roi_results, out_dir):
    """Create a summary table and figure across all ROIs."""
    if not all_roi_results:
        return

    df = pd.concat(all_roi_results, ignore_index=True)
    df.to_csv(opj(out_dir, "cross_roi_model_correlations.csv"), index=False)

    # Plot: for each model × modality, bar plot across ROIs
    models = df["model"].unique()
    modalities = df["modality"].unique()

    for modality in modalities:
        df_mod = df[df["modality"] == modality]
        models_present = df_mod["model"].unique()

        fig, axes = plt.subplots(
            1, len(models_present),
            figsize=(4 * len(models_present), 4),
            squeeze=False,
        )
        for ax, model in zip(axes[0], models_present):
            df_m = df_mod[df_mod["model"] == model]
            rois = sorted(df_m["roi"].unique())
            x = np.arange(len(rois))
            width = 0.3

            for sess, color, offset in [
                ("presleep", "#3274A1", -width / 2),
                ("postsleep", "#E1812C", width / 2),
            ]:
                means, sems = [], []
                for roi in rois:
                    vals = df_m.loc[
                        (df_m["roi"] == roi) & (df_m["session"] == sess), "fisher_z"
                    ].values
                    means.append(np.mean(vals) if len(vals) > 0 else 0)
                    sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 0 else 0)
                ax.bar(x + offset, means, width=width, yerr=sems,
                       color=color, alpha=0.7, capsize=4, label=sess)

            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(rois, fontsize=9)
            ax.set_ylabel("Fisher-z(Spearman r)")
            ax.set_title(f"{model}")
            ax.legend(frameon=False, fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Model correlations across ROIs ({modality}s)", fontsize=12)
        fig.tight_layout()
        fig.savefig(opj(out_dir, f"cross_roi_{modality}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare presleep vs postsleep RDMs")
    parser.add_argument("--roi", choices=RSA_ROIS, default=None,
                        help="ROI to analyze (default: all ROIs)")
    args = parser.parse_args()

    rois_to_run = [args.roi] if args.roi else RSA_ROIS

    print(f"\n{'='*60}")
    print(f"Session comparison: ROIs={rois_to_run}")
    print(f"{'='*60}\n")

    summary_dir = opj(SCRIPT_DIR, "03_compare_sessions")
    os.makedirs(summary_dir, exist_ok=True)

    all_roi_results = []

    for roi in rois_to_run:
        print(f"\n{'='*50}")
        print(f"ROI: {roi}")
        print(f"{'='*50}")

        out_dir = opj(summary_dir, roi)
        os.makedirs(out_dir, exist_ok=True)

        for modality in ["object", "scene"]:
            print(f"\n{'='*40}")
            print(f"Modality: {modality}")
            print(f"{'='*40}")

            # Find subjects with both presleep and postsleep RDMs
            pre_subs = set(discover_subjects("presleep", roi, modality))
            post_subs = set(discover_subjects("postsleep", roi, modality))
            common_subs = sorted(pre_subs & post_subs)

            if len(common_subs) < 3:
                print(f"  Only {len(common_subs)} subjects with both sessions. Skipping.")
                continue

            print(f"  {len(common_subs)} subjects with both sessions")

            # Load RDMs
            pre_rdms = []
            post_rdms = []
            conditions = None
            valid_subs = []

            for sub in common_subs:
                pre_path = opj(
                    SCRIPT_DIR, "02_compute_rdms", "presleep", "rdms", sub,
                    f"{sub}_presleep_{roi}_{modality}s_rdm.csv",
                )
                post_path = opj(
                    SCRIPT_DIR, "02_compute_rdms", "postsleep", "rdms", sub,
                    f"{sub}_postsleep_{roi}_{modality}s_rdm.csv",
                )
                pre_mat, pre_conds = load_rdm_csv(pre_path)
                post_mat, post_conds = load_rdm_csv(post_path)

                if conditions is None:
                    conditions = pre_conds
                if pre_conds != post_conds:
                    print(f"  WARNING: {sub} condition mismatch! Skipping.")
                    continue

                pre_rdms.append(pre_mat)
                post_rdms.append(post_mat)
                valid_subs.append(sub)

            pre_rdms = np.array(pre_rdms)
            post_rdms = np.array(post_rdms)

            if conditions is None or len(valid_subs) < 3:
                print("  Not enough valid subjects.")
                continue

            # -------------------------------------------------------
            # Build model RDMs using sequence structure
            # -------------------------------------------------------
            # Use the first subject to build model RDMs
            # (sequence structure is counterbalanced but the model
            # is built per subject in the correlation step anyway)
            sub_behav, sub_num, _ = parse_subject_id(valid_subs[0])
            obj_enc_to_cat = load_localizer_category_mapping(sub_behav, sub_num)
            df_enc = load_encoding_behavior(sub_behav, sub_num)
            seq_struct = build_sequence_structure(df_enc, obj_enc_to_cat)

            # Choose the right mapping for this modality
            if modality == "scene":
                cond_to_ctx = seq_struct["scene_to_context"]
                ctx_to_ordered = seq_struct["context_to_scenes"]
            else:  # object
                cond_to_ctx = seq_struct["category_to_context"]
                ctx_to_ordered = seq_struct["context_to_categories"]

            # Build model RDMs
            model_rdms = {}

            # 1. Context model: within day/night = similar
            context_rdm = build_context_model_rdm(conditions, cond_to_ctx)
            if not np.all(np.isnan(upper_triangle(context_rdm))):
                model_rdms["context"] = context_rdm

            # 2. Position model: position distance within sequence
            position_rdm = build_position_model_rdm(
                conditions, cond_to_ctx, ctx_to_ordered,
            )
            if not np.all(np.isnan(upper_triangle(position_rdm))):
                model_rdms["position"] = position_rdm

            print(f"  Built {len(model_rdms)} model RDMs: {list(model_rdms.keys())}")

            # Visualize model RDMs
            plot_model_rdms(model_rdms, conditions, modality, out_dir)

            # -------------------------------------------------------
            # Run analyses
            # -------------------------------------------------------
            analysis_stability(
                pre_rdms, post_rdms, valid_subs, conditions, modality, out_dir,
            )
            analysis_distance_changes(
                pre_rdms, post_rdms, valid_subs, conditions, modality,
                cond_to_ctx, out_dir,
            )
            model_df = analysis_model_rdms(
                pre_rdms, post_rdms, valid_subs, conditions, modality,
                model_rdms, out_dir,
            )

            # Add ROI and modality info for cross-ROI summary
            if not model_df.empty:
                model_df["roi"] = roi
                model_df["modality"] = modality
                all_roi_results.append(model_df)

            plot_group_rdms(
                pre_rdms, post_rdms, conditions, modality, out_dir,
            )

    # Cross-ROI summary
    if all_roi_results:
        cross_roi_summary(all_roi_results, summary_dir)

    print(f"\nDone. Results in: {summary_dir}")


if __name__ == "__main__":
    main()
