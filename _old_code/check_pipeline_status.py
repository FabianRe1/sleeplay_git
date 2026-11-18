#!/usr/bin/env python3
"""
Pipeline Status Tracker for Sleeplay_Hummel fMRI Analysis
==========================================================
Checks each analysis step for each subject and reports which subjects
are present/missing at each stage. Run from anywhere — paths are relative
to the project root detected automatically.
"""

import os
import glob
import pandas as pd
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # code/../ = Sleeplay_Hummel/
DERIVATIVES   = Path("/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives")
BEHAVIOR      = PROJECT_ROOT / "behavior"
CODE          = PROJECT_ROOT / "sleeplay_git"

# ── Subjects to exclude (failed experiments / missing data) ───────────────────
SKIP_SUBJECTS = {
    '01', '31', '33', '35', '45', '39', '49', '56',   # no behavior folder
    '54', '55', '57',                            # no localizer behavior
}

# ── All subjects are based on the behavior folder (sXX subfolders) ────────────
def _detect_all_subjects():
    """Collect every subject ID from the behavior folder, excluding known failures."""
    subs = set()
    for p in BEHAVIOR.iterdir():
        if p.is_dir() and p.name.startswith("s"):
            sid = p.name.replace("s", "")
            if sid.isdigit() and sid not in SKIP_SUBJECTS:
                subs.add(sid)
    return sorted(subs, key=int)


# ── Check functions ───────────────────────────────────────────────────────────
# Each returns True if the subject has outputs for that step.

def check_derivatives(sid):
    """fMRIPrep derivatives exist."""
    return (DERIVATIVES / f"sub-{sid}").is_dir()


def check_behavior_hrf(sid):
    """Behavior CSV with HRF correction exists (localizer + main task)."""
    loc_hrf = list((BEHAVIOR / f"s{sid}" / "behav" / "loc").glob(f"s{sid}_df_loc*_w_HRF.csv")) if (BEHAVIOR / f"s{sid}" / "behav" / "loc").exists() else []
    main_hrf = list((BEHAVIOR / f"s{sid}" / "behav" / "main").rglob(f"*_w_HRF.csv")) if (BEHAVIOR / f"s{sid}" / "behav" / "main").exists() else []
    return len(loc_hrf) > 0 and len(main_hrf) > 0


def check_thresholded_roi(sid):
    """Thresholded ROI output folder exists for this subject."""
    roi_dir = CODE / "2_get_threshold_roi" / "localizer_getThresholdedROI" / "thresholded_roi_output" / f"sub-{sid}"
    return roi_dir.is_dir() and any(roi_dir.iterdir())


def check_reliability_masks(sid):
    """Combined reliability masks exist."""
    mask_dir = CODE / "3_reliability_voxel_selection" / "combine_loc1_and_loc3_masks"
    masks = list(mask_dir.glob(f"s{sid}_mask_*_combined.nii")) + \
            list(mask_dir.glob(f"sub-{sid}_mask_*_combined.nii"))
    return len(masks) > 0


def check_classifiers(sid):
    """Trained visual classifiers exist."""
    clf_dir = CODE / "4_decoding" / "4.1_train_visual_classifier" / "classifiers"
    clfs = list(clf_dir.glob(f"s{sid}_*.joblib"))
    return len(clfs) > 0


def _check_decoding_timecourses(sid, step_dir):
    """Generic check for decoding timecourse .npy files."""
    data_dir = CODE / "4_decoding" / step_dir / "decoding_timecourses_data"
    if not data_dir.exists():
        return False
    files = list(data_dir.glob(f"s{sid}_*.npy"))
    return len(files) > 0


def check_encoding_transfer(sid):
    return _check_decoding_timecourses(sid, "4.3_transfer_visual_classifier_encoding")


def check_presleep_retrieval(sid):
    return _check_decoding_timecourses(sid, "4.4_transfer_visual_classifier_retrieval_presleep")


def check_postsleep_retrieval(sid):
    return _check_decoding_timecourses(sid, "4.5_transfer_visual_classifier_retrieval_postsleep")


def _check_soda_csv(sid, step_dir):
    """Check if subject appears in the aggregated SODA CSV."""
    csv_path = CODE / "4_decoding" / step_dir / "SODA_results_csv" / "soda_results_per_combination.csv"
    if not csv_path.exists():
        # Try any ROI-specific CSV
        csv_dir = CODE / "4_decoding" / step_dir / "SODA_results_csv"
        if csv_dir.exists():
            csvs = list(csv_dir.glob("*_soda_results_per_combination.csv"))
            if csvs:
                csv_path = csvs[0]
            else:
                return False
        else:
            return False
    try:
        df = pd.read_csv(csv_path, usecols=["subject"], nrows=10000)
        return f"s{sid}" in df["subject"].unique()
    except Exception:
        return False


def check_soda_presleep(sid):
    return _check_soda_csv(sid, "4.6_SODA_based_on_singlesubj_4.4_presleep")


def check_soda_postsleep(sid):
    return _check_soda_csv(sid, "4.6_SODA_based_on_singlesubj_4.4_postsleep")


def check_glm_per_run(sid):
    """GLM per-run beta maps exist."""
    glm_dir = CODE / "6_univariate_engagement" / "GLM_for_beta_maps" / "glm_maps" / f"sub-{sid}"
    if not glm_dir.exists():
        return False
    expected = ["run-04_stimOn_eff.nii.gz", "run-05_stimOn_eff.nii.gz", "run-06_stimOn_eff.nii.gz"]
    return all((glm_dir / f).exists() for f in expected)


def check_glm_1glm_presleep(sid):
    """Single-GLM remembered vs. forgotten contrast exists (presleep)."""
    glm_dir = CODE / "6_univariate_engagement" / "GLM_for_beta_maps_1GLM" / "glm_maps" / f"sub-{sid}"
    return glm_dir.exists() and (glm_dir / "Big_GLM_remem_vs_nonremem.nii.gz").exists()


def check_glm_1glm_postsleep(sid):
    """Single-GLM postsleep contrast exists."""
    glm_dir = CODE / "6_univariate_engagement" / "GLM_for_beta_maps_1GLM_postsleep" / "glm_maps" / f"sub-{sid}"
    return glm_dir.exists() and (glm_dir / "Big_GLM_remem_vs_nonremem.nii.gz").exists()


# ── Pipeline steps registry ───────────────────────────────────────────────────
STEPS = [
    ("0. Derivatives (fMRIPrep)",           check_derivatives),
    ("1. Behavior HRF",                     check_behavior_hrf),
    ("2. Thresholded ROIs",                 check_thresholded_roi),
    ("3. Reliability Masks",                check_reliability_masks),
    ("4.1 Train Classifiers",               check_classifiers),
    ("4.3 Encoding Transfer",               check_encoding_transfer),
    ("4.4 Presleep Retrieval",              check_presleep_retrieval),
    ("4.5 Postsleep Retrieval",             check_postsleep_retrieval),
    ("4.6 SODA Presleep",                   check_soda_presleep),
    ("4.6 SODA Postsleep",                  check_soda_postsleep),
    ("6. GLM Per-Run",                      check_glm_per_run),
    ("6. GLM 1GLM Presleep",               check_glm_1glm_presleep),
    ("6. GLM 1GLM Postsleep",              check_glm_1glm_postsleep),
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    subjects = _detect_all_subjects()
    print(f"\n{'='*80}")
    print(f"  SLEEPLAY HUMMEL — Pipeline Status Report")
    print(f"  Detected {len(subjects)} subjects: {', '.join('s'+s for s in subjects)}")
    print(f"{'='*80}\n")

    # Build results table
    results = {}
    for sid in subjects:
        results[f"s{sid}"] = {}
        for step_name, check_fn in STEPS:
            results[f"s{sid}"][step_name] = check_fn(sid)

    df = pd.DataFrame(results).T
    df.index.name = "Subject"

    # ── Per-step summary ──────────────────────────────────────────────────────
    for step_name, _ in STEPS:
        present = [s for s in df.index if df.loc[s, step_name]]
        missing = [s for s in df.index if not df.loc[s, step_name]]

        status = "OK" if not missing else "MISSING"
        symbol = "  " if not missing else "!!"
        print(f"{symbol} {step_name}")
        print(f"   Present ({len(present):2d}): {', '.join(present)}")
        if missing:
            print(f"   MISSING ({len(missing):2d}): {', '.join(missing)}")
        print()

    # ── Summary table (compact) ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Compact Overview  (✓ = present, ✗ = MISSING)")
    print(f"{'='*80}\n")

    # Abbreviate step names for the compact table
    short_names = {
        "0. Derivatives (fMRIPrep)": "Deriv",
        "1. Behavior HRF":          "Behav",
        "2. Thresholded ROIs":      "ROIs",
        "3. Reliability Masks":     "Rel",
        "4.1 Train Classifiers":    "Clf",
        "4.3 Encoding Transfer":    "Enc",
        "4.4 Presleep Retrieval":   "Pre",
        "4.5 Postsleep Retrieval":  "Post",
        "4.6 SODA Presleep":        "SODAp",
        "4.6 SODA Postsleep":       "SODAq",
        "6. GLM Per-Run":           "GLMr",
        "6. GLM 1GLM Presleep":     "GLM1p",
        "6. GLM 1GLM Postsleep":    "GLM1q",
    }

    header = f"{'Subject':<8}" + "".join(f"{short_names[s]:>7}" for s, _ in STEPS)
    print(header)
    print("-" * len(header))

    for sid in df.index:
        row = f"{sid:<8}"
        for step_name, _ in STEPS:
            val = df.loc[sid, step_name]
            mark = "✓" if val else "✗"
            row += f"{mark:>7}"
        print(row)

    # ── Totals row ────────────────────────────────────────────────────────────
    print("-" * len(header))
    totals = f"{'Total':<8}"
    for step_name, _ in STEPS:
        n = df[step_name].sum()
        totals += f"{n:>7}"
    print(totals)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = CODE / "pipeline_status.csv"
    df_out = df.copy()
    for col in df_out.columns:
        df_out[col] = df_out[col].map({True: "OK", False: "MISSING"})
    df_out.to_csv(out_csv)
    print(f"\nDetailed CSV saved to: {out_csv}\n")


if __name__ == "__main__":
    main()
