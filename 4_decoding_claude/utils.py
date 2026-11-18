# -*- coding: utf-8 -*-
"""Shared utilities for the Sleeplay decoding pipeline.

Centralises helper functions that were previously duplicated across scripts:
  - Output directory creation
  - Subject / ROI argument parsing
  - Confound variable definitions
  - fMRI data loading and preprocessing
  - Sequence hierarchy helpers (lookup_5, control_5)
  - seq_fitter (SODA slope computation)
  - FDR correction
"""

import os
import re
import glob
import json
from os.path import join as opj
from typing import Any, Literal

import numpy as np
import pandas as pd
from nilearn import image, masking, signal
from joblib import load as joblib_load
from statsmodels.stats.multitest import multipletests


# =====================================================================
# 1. OUTPUT DIRECTORY HELPERS
# =====================================================================

def ensure_dir(path: str) -> str:
    """Create *path* if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def script_output_dir(script_file: str, subfolder: str | None = None) -> str:
    """Return (and create) an output directory named after *script_file*.

    Parameters
    ----------
    script_file : str
        Typically ``__file__`` from the calling script.
    subfolder : str, optional
        Optional nested subfolder inside the script-named directory.
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    stem = os.path.splitext(os.path.basename(script_file))[0]
    if subfolder is None:
        out = opj(script_dir, stem)
    else:
        out = opj(script_dir, stem, subfolder)
    return ensure_dir(out)


# =====================================================================
# 2. SUBJECT / ROI ARGUMENT PARSING
# =====================================================================

def parse_subject_arg(arg: str | None, default: str = "s13") -> str:
    """Normalise a subject identifier to behaviour-style ``sXX`` format.

    Accepted forms: ``"sub-13"``, ``"13"``, ``"s13"``.
    Returns *default* when *arg* is ``None`` or unrecognised.
    """
    if arg is None:
        print(f"WARNING: No subject specified. Using default subject {default}.")
        return default

    m = re.match(r"^sub-(\d+)$", arg)
    if m:
        return f"s{m.group(1)}"
    if re.match(r"^\d+$", arg):
        return f"s{arg}"
    if re.match(r"^s\d+$", arg):
        return arg

    print(f"WARNING: Unexpected subject format '{arg}'. Using default {default}.")
    return default


def parse_roi_arg(arg: str | None, default: str = "vis") -> str:
    """Return *arg* if it looks like a valid ROI label, else *default*."""
    if arg is not None and re.match(r"^[a-z_]+$", arg):
        return arg
    if arg is not None:
        print(f"WARNING: Unexpected ROI format '{arg}'. Using default ROI {default}.")
    else:
        print(f"WARNING: No ROI specified. Using default ROI {default}.")
    return default


# =====================================================================
# 3. CONFOUNDS
# =====================================================================

CONFOUND_VARS = [
    "global_signal", "framewise_displacement",
    "trans_x", "trans_y", "trans_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x", "rot_y", "rot_z",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05",
]


# =====================================================================
# 4. fMRI DATA LOADING AND PREPROCESSING
# =====================================================================

def assert_func_conf_match(func_paths: list[str], conf_paths: list[str]) -> None:
    """Fail loudly if functional images and confound files do not correspond.

    Both lists are built from independent globs and are afterwards paired by
    POSITION (``confounds[i]`` cleans ``func[i]``). A missing or extra confound
    file would therefore shift the pairing and clean every subsequent run with
    another run's motion parameters — silently, and without any downstream
    symptom. The run token in the filenames is the only thing tying the two
    together, so it is checked explicitly.
    """
    def _run_tokens(paths: list[str]) -> list[str]:
        tokens = []
        for p in paths:
            m = re.search(r"run-(\d+)", os.path.basename(p))
            tokens.append(m.group(1) if m else os.path.basename(p))
        return tokens

    if len(func_paths) != len(conf_paths):
        raise ValueError(
            f"Mismatch between functional images ({len(func_paths)}) and confound "
            f"files ({len(conf_paths)}). They are paired by position, so this would "
            f"clean runs with the wrong confounds.\n"
            f"  func:      {[os.path.basename(p) for p in func_paths]}\n"
            f"  confounds: {[os.path.basename(p) for p in conf_paths]}"
        )

    runs_func, runs_conf = _run_tokens(func_paths), _run_tokens(conf_paths)
    if runs_func != runs_conf:
        raise ValueError(
            f"Functional images and confound files are not in the same run order: "
            f"func runs {runs_func} vs confound runs {runs_conf}."
        )


def load_func_and_confounds(
    fmriprep_dir: str,
    sub_mri: str,
    session: str,
    task_glob: str = "task-mem*",
    TR: float = 1.5,
) -> tuple[list, list]:
    """Load functional images and matching confound TSV files.

    Returns
    -------
    func_imgs : list[nibabel.Nifti1Image]
    confounds  : list[pd.DataFrame]
    """
    func_pattern = opj(
        fmriprep_dir, sub_mri, session, "func",
        f"{sub_mri}_{session}_{task_glob}_space-T1w_desc-preproc_bold.nii.gz",
    )
    func_paths = sorted(glob.glob(func_pattern), key=os.path.basename)
    if not func_paths:
        raise FileNotFoundError(f"No functional files matching: {func_pattern}")
    func_imgs = [image.load_img(p) for p in func_paths]

    conf_pattern = opj(
        fmriprep_dir, sub_mri, session, "func",
        f"{sub_mri}_{session}_{task_glob}_run*_desc-confounds_timeseries.tsv",
    )
    conf_paths = sorted(glob.glob(conf_pattern), key=os.path.basename)
    assert_func_conf_match(func_paths, conf_paths)
    confounds = [pd.read_csv(p, sep="\t") for p in conf_paths]

    return func_imgs, confounds


def mask_and_clean(
    func_imgs: list,
    confounds: list[pd.DataFrame],
    mask_img,
    smoothing_fwhm: float = 4.0,
    TR: float = 1.5,
    conf_vars: list[str] | None = None,
) -> list[np.ndarray]:
    """Apply mask, smooth, detrend, z-score and regress confounds.

    Returns list of 2-D arrays (n_timepoints × n_voxels).
    """
    if conf_vars is None:
        conf_vars = CONFOUND_VARS
    if len(func_imgs) != len(confounds):
        raise ValueError(
            f"{len(func_imgs)} functional runs but {len(confounds)} confound tables — "
            "they are paired by position, so cleaning would use the wrong run's confounds."
        )
    masked = [masking.apply_mask(img, mask_img, smoothing_fwhm=smoothing_fwhm) for img in func_imgs]
    cleaned = [
        signal.clean(
            signals=masked[i],
            t_r=TR,
            detrend=True,
            standardize="zscore_sample",
            confounds=confounds[i][conf_vars].fillna(0),
        )
        for i in range(len(masked))
    ]
    return cleaned


def load_mask(path_masks: str, sub: str, roi: str, reliability_threshold: int) -> "nibabel.Nifti1Image":
    """Try sub-XX then sXX naming convention for mask files."""
    sub_num = sub.lstrip("s")
    candidates = [
        f"sub-{sub_num}_mask_{roi}_threshold_{reliability_threshold}_combined.nii",
        f"{sub}_mask_{roi}_threshold_{reliability_threshold}_combined.nii",
    ]
    for fname in candidates:
        fpath = opj(path_masks, fname)
        if os.path.exists(fpath):
            return image.load_img(fpath)
    raise FileNotFoundError(
        f"Mask not found. Tried: {[opj(path_masks, c) for c in candidates]}"
    )


# =====================================================================
# 5. CLASSIFIER LOADING
# =====================================================================

def load_classifiers(
    path_clf_dir: str,
    sub: str,
    roi: str,
    smoothing: int,
    f_threshold: int,
    reliability_threshold: int,
    temporal_smoothing: int,
    use_overall: bool = False,
) -> tuple:
    """Load scene and object classifiers (or a single generic one).

    Returns
    -------
    clf_scenes, clf_objects
    """
    base = (
        f"{sub}_Mask_{roi}_Smooth_{smoothing}_Threshold_{f_threshold}"
        f"_reliability_threshold_{reliability_threshold}"
        f"_tmpsmoothing_{temporal_smoothing}"
    )

    if use_overall:
        path = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_3.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Generic classifier not found: {path}")
        clf = joblib_load(path)
        return clf, clf

    clf_scenes_path = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_1.joblib")
    clf_objects_path = opj(path_clf_dir, f"{base}_only_scenes1_objects2_both3_2.joblib")
    if not os.path.exists(clf_scenes_path):
        raise FileNotFoundError(f"Scene classifier not found: {clf_scenes_path}")
    if not os.path.exists(clf_objects_path):
        raise FileNotFoundError(f"Object classifier not found: {clf_objects_path}")
    return joblib_load(clf_scenes_path), joblib_load(clf_objects_path)


# =====================================================================
# 6. SEQUENCE HIERARCHY HELPERS
# =====================================================================

HierarchyType = dict[str, dict[str, list[str]]]


def build_sequence_dict_from_df_behavior_enc(
    df: pd.DataFrame,
    cue_col: str = "dn_cue",
    scene_col: str = "sc_enc_stim",
    obj_col: str = "obj_enc_stim",
    expected_n_obj: int = 5,
) -> HierarchyType:
    """Build ``{day: {scene: [obj×5]}, night: {scene: [obj×5]}}`` from encoding behaviour."""
    missing = [c for c in (cue_col, scene_col, obj_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    out: HierarchyType = {"day": {}, "night": {}}
    for cue, scene, obj in df[[cue_col, scene_col, obj_col]].itertuples(index=False, name=None):
        if pd.isna(cue) or pd.isna(scene):
            continue
        dn = "day" if "day" in str(cue).lower() else ("night" if "night" in str(cue).lower() else None)
        if dn is None:
            continue
        scene_key = str(scene)
        if scene_key not in out[dn]:
            out[dn][scene_key] = []
        if pd.notna(obj):
            obj_s = str(obj)
            if len(out[dn][scene_key]) < expected_n_obj and obj_s not in out[dn][scene_key]:
                out[dn][scene_key].append(obj_s)

    for dn in ("day", "night"):
        for sc, objs in out[dn].items():
            if len(objs) != expected_n_obj:
                print(f"WARNING: {dn}/{sc} has {len(objs)} objects (expected {expected_n_obj}).")
    return out


def map_sequence_objects_to_categories(
    seq_dict_obj: HierarchyType,
    obj_enc_to_cat: dict[str, str],
    normalize_basename: bool = True,
) -> HierarchyType:
    """Replace object filenames with category labels, preserving order."""
    out: HierarchyType = {"day": {}, "night": {}}
    unmapped = []
    for dn, scenes in seq_dict_obj.items():
        for scene, obj_list in scenes.items():
            mapped = []
            for obj_stim in obj_list:
                cat = obj_enc_to_cat.get(obj_stim)
                if cat is None and normalize_basename:
                    cat = obj_enc_to_cat.get(os.path.basename(str(obj_stim)))
                if cat is None:
                    unmapped.append(obj_stim)
                    cat = "UNMAPPED"
                mapped.append(cat)
            out[dn][scene] = mapped
    if unmapped:
        print(f"WARNING: {len(unmapped)} objects could not be mapped (first 10): {unmapped[:10]}")
    return out


def lookup_5(
    hier: HierarchyType,
    value: str,
    kind: Literal["object", "scene"],
) -> list[str]:
    """Return the 5-item sequence that *value* belongs to."""
    if kind == "scene":
        for scenes in hier.values():
            if value in scenes:
                return list(scenes.keys())
        raise KeyError(f"Scene '{value}' not found.")

    if kind == "object":
        for scenes in hier.values():
            for obj_list in scenes.values():
                if value in obj_list:
                    return obj_list
        raise KeyError(f"Object '{value}' not found.")

    raise ValueError("kind must be 'object' or 'scene'.")


def control_5(
    hier: HierarchyType,
    value: str,
    kind: Literal["object", "scene"],
) -> list[str]:
    """Return the complementary 5-item sequence (other block)."""
    if kind == "scene":
        for top_key, scenes in hier.items():
            if value in scenes:
                other = [k for k in hier if k != top_key]
                if len(other) != 1:
                    raise ValueError(f"Expected 2 top-level groups, got: {list(hier.keys())}")
                return list(hier[other[0]].keys())
        raise KeyError(f"Scene '{value}' not found.")

    if kind == "object":
        retrieved = lookup_5(hier, value, kind="object")
        uniq_sets: list[list[str]] = []
        for scenes in hier.values():
            for obj_list in scenes.values():
                if obj_list not in uniq_sets:
                    uniq_sets.append(obj_list)
        other = [s for s in uniq_sets if s != retrieved]
        if len(other) != 1:
            raise ValueError(f"Expected exactly 1 complementary set, got {len(other)}.")
        return other[0]

    raise ValueError("kind must be 'object' or 'scene'.")


# =====================================================================
# 7. SHUFFLE UTILITIES
# =====================================================================

def _valid_selected_shuffles(n: int) -> list[tuple[int, ...]]:
    """Pre-compute all permutations of range(n) that:

    1. Are not the identity [0,1,...,n-1]
    2. Are not the reverse  [n-1,...,1,0]
    3. Have no consecutive elements that were adjacent in the original order,
       i.e. |perm[i] - perm[i+1]| > 1 for every consecutive pair.

    This ensures that neither the forward nor backward sequential
    relationship is partially preserved in the shuffled version.
    """
    from itertools import permutations

    identity = tuple(range(n))
    reverse = tuple(range(n - 1, -1, -1))
    valid = []
    for perm in permutations(range(n)):
        if perm == identity or perm == reverse:
            continue
        if all(abs(perm[i] - perm[i + 1]) > 1 for i in range(n - 1)):
            valid.append(perm)
    return valid


# Cache per sequence length (5 in practice)
_SELECTED_SHUFFLE_CACHE: dict[int, list[tuple[int, ...]]] = {}


def get_selected_shuffles(n: int) -> list[tuple[int, ...]]:
    """Return (cached) list of valid 'selected' permutations for length *n*."""
    if n not in _SELECTED_SHUFFLE_CACHE:
        _SELECTED_SHUFFLE_CACHE[n] = _valid_selected_shuffles(n)
    return _SELECTED_SHUFFLE_CACHE[n]


def apply_selected_shuffle(
    probs: list | np.ndarray,
    rng: np.random.Generator,
) -> list:
    """Randomly pick a valid 'selected' permutation and apply it to *probs*.

    A selected shuffle guarantees:
    - Not the identity or reverse order
    - No originally-adjacent items end up at consecutive positions
    """
    n = len(probs)
    valid = get_selected_shuffles(n)
    if not valid:
        raise ValueError(f"No valid selected shuffles exist for n={n}")
    perm = valid[rng.integers(len(valid))]
    return [probs[i] for i in perm]


# =====================================================================
# 8. SODA – SEQUENTIAL SLOPE ANALYSIS
# =====================================================================

def seq_fitter(x: np.ndarray, method: str = "lm") -> float:
    """Compute a sequentiality slope over ordered classifier probabilities.

    Uses ascending predictor [1, 2, …, N]:
      - Negative slope → item 1 highest (forward replay from first item)
      - Positive slope → item N highest (forward replay from last item)

    NaN handling: NaNs are dropped but original position indices are preserved.
    """
    x = np.asarray(x, dtype=float)
    positions = np.arange(1, len(x) + 1, dtype=float)
    valid = ~np.isnan(x)
    x_valid = x[valid]
    pos_valid = positions[valid]

    if len(x_valid) < 2:
        return np.nan

    if method == "lm":
        return float(np.polyfit(pos_valid, x_valid, 1)[0])
    elif method == "spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(pos_valid, x_valid)[0])
    else:
        raise ValueError(f"Unknown method: {method}")


# =====================================================================
# 9. STATISTICS
# =====================================================================

def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction.

    Returns
    -------
    rejected : bool array
    q_values : float array
    """
    p = np.asarray(p_values, dtype=float).ravel()
    valid = ~np.isnan(p)
    rejected = np.zeros_like(valid, dtype=bool)
    q = np.full_like(p, np.nan, dtype=float)
    if valid.any():
        rej_v, q_v, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
        rejected[valid] = rej_v
        q[valid] = q_v
    return rejected, q
