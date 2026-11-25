#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4 - Whole-brain group maps of retrieval activity with TFCE.

Takes the participant maps of E3 (standard space) and tests, voxel by voxel,
one contrast across participants: a one-sample t-test with sign-flip
permutations and threshold-free cluster enhancement (TFCE), corrected for
multiple comparisons across the whole brain (family-wise error from the
permutation distribution of the maximal TFCE value; two-sided).

Contrasts (activity against the run baseline unless a difference is named):
    pooled         mean of the three time points (participants with all three)
    pre            evening retrieval
    post1          first morning retrieval
    post2          repeated morning retrieval
    post1_minus_pre, post2_minus_post1     differences between neighbouring time points
    post2_minus_pre                        repeated morning retrieval minus evening
    post_minus_pre                         mean of the two morning retrievals minus evening
    per_second_rt  change in activity per second of reaction time, mean of the
                   time points available (E3 model <model>_rt, e.g. fixed_rt)

Group mask: voxels inside the brain mask of every participant in the test.

Input   outputs/E3_wholebrain_maps/maps/<model>/, .../masks/
        <derivatives>/sub-XX/anat/*space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz  (labels for the tables)
Output  outputs/E4_wholebrain_group/<model>/<kind>/<contrast>_t.nii.gz             t values, unthresholded
                                                   <contrast>_tfce_logp.nii.gz    -log10 corrected p (TFCE)
                                                   <contrast>_t_thresholded.nii.gz t where corrected p < .05
                                                   <contrast>_peaks.csv           peaks of the thresholded map
                                                   <contrast>_regions.csv         per anatomical region: share of
                                                                                  voxels significantly above / below, mean t
                                                   <contrast>_participants.csv

Environment (optional): E4_MAPS=native_first takes the participant maps from E8 (fitted in each
participant's own anatomy and transformed to standard space afterwards) instead of E3; results then
go to outputs/E4_wholebrain_group/native_first/<model>/<kind>/, and the group mask is further limited
to voxels that carry data in every transformed map.

Usage:  python E4_wholebrain_group.py <model> <kind> <contrast> [n_permutations] [n_jobs]
        e.g. python E4_wholebrain_group.py fixed object pooled 5000 8

Not a port of the original pipeline: added 2026-11-22.
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
from nilearn import image, masking
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import resample_to_img
from nilearn.reporting import get_clusters_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DERIVATIVES_DIR, step_input_dir, step_output_dir

ALPHA = 0.05
CONDS = ("pre", "post1", "post2")
CONTRASTS = ("pooled", "pre", "post1", "post2", "post1_minus_pre", "post2_minus_post1", "post2_minus_pre",
             "post_minus_pre", "per_second_rt") + tuple(f"{c}R{k}" for c in ("pre", "post1", "post2") for k in (1, 2, 3))
# <time point>R<k>: one run on its own (E3 models ret_press / ret_rtmod)

# Desikan-Killiany names (FreeSurfer aparc: 1000 + index left, 2000 + index right) and a few aseg labels
APARC = {1: "bankssts", 2: "caudal anterior cingulate", 3: "caudal middle frontal", 5: "cuneus", 6: "entorhinal",
         7: "fusiform", 8: "inferior parietal", 9: "inferior temporal", 10: "isthmus cingulate (retrosplenial)",
         11: "lateral occipital", 12: "lateral orbitofrontal", 13: "lingual", 14: "medial orbitofrontal",
         15: "middle temporal", 16: "parahippocampal", 17: "paracentral", 18: "pars opercularis",
         19: "pars orbitalis", 20: "pars triangularis", 21: "pericalcarine", 22: "postcentral",
         23: "posterior cingulate", 24: "precentral", 25: "precuneus", 26: "rostral anterior cingulate",
         27: "rostral middle frontal", 28: "superior frontal", 29: "superior parietal", 30: "superior temporal",
         31: "supramarginal", 32: "frontal pole", 33: "temporal pole", 34: "transverse temporal", 35: "insula"}
ASEG = {17: "hippocampus", 53: "hippocampus", 18: "amygdala", 54: "amygdala", 10: "thalamus", 49: "thalamus",
        11: "caudate", 50: "caudate", 12: "putamen", 51: "putamen", 13: "pallidum", 52: "pallidum",
        26: "accumbens", 58: "accumbens", 8: "cerebellum", 47: "cerebellum", 16: "brainstem",
        2: "white matter", 41: "white matter", 7: "cerebellar white matter", 46: "cerebellar white matter"}


def label_name(code: int) -> str:
    code = int(code)
    if code in ASEG:
        side = "" if code == 16 else ("L " if code < 40 else "R ")
        return side + ASEG[code]
    if 1000 <= code < 3000 and code % 1000 in APARC:
        return ("L " if code < 2000 else "R ") + APARC[code % 1000]
    return "unlabelled" if code == 0 else f"label {code}"


def group_atlas(subjects, ref_img):
    """Most frequent FreeSurfer label per voxel across participants (cached)."""
    path = opj(str(step_output_dir(__file__, "atlas")), "aparcaseg_mode.nii.gz")
    if os.path.exists(path):
        return nib.load(path)
    stack = []
    for sub in subjects:
        hits = sorted(glob.glob(opj(str(DERIVATIVES_DIR), f"sub-{sub[1:]}", "anat",
                                    f"sub-{sub[1:]}_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz")) +
                      glob.glob(opj(str(DERIVATIVES_DIR), f"sub-{sub[1:]}", "ses-*", "func",
                                    f"*space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz")))
        if hits:
            img = resample_to_img(hits[0], ref_img, interpolation="nearest")
            stack.append(np.asarray(img.dataobj).astype(np.int32))
    arr = np.stack(stack, -1)
    flat = arr.reshape(-1, arr.shape[-1])
    mode = np.zeros(flat.shape[0], np.int32)
    nz = flat.max(1) > 0
    srt = np.sort(flat[nz], 1)
    for i, row in zip(np.where(nz)[0], srt):          # mode of each row
        vals, cnt = np.unique(row, return_counts=True)
        mode[i] = vals[cnt.argmax()]
    out = nib.Nifti1Image(mode.reshape(arr.shape[:3]), ref_img.affine)
    tmp = path.replace(".nii.gz", f"_tmp{os.getpid()}.nii.gz")
    out.to_filename(tmp)
    os.replace(tmp, path)
    return out


NATIVE_FIRST = os.environ.get("E4_MAPS", "") == "native_first"


def participant_maps(model, kind, contrast):
    """dict subject -> 3D array-like image of the contrast."""
    src_model = f"{model}_rt" if contrast == "per_second_rt" else model
    mdir = (str(step_input_dir("E8_native_then_standard", opj("maps", src_model))) if NATIVE_FIRST
            else str(step_input_dir("E3_wholebrain_maps", opj("maps", src_model))))
    suffix = f"{kind}_per_second_rt" if contrast == "per_second_rt" else kind
    files = {}
    for f in sorted(glob.glob(opj(mdir, f"s*_*_{suffix}.nii.gz"))):
        sub, cond, rest = os.path.basename(f)[:-len(".nii.gz")].split("_", 2)
        if rest != suffix:              # e.g. `object` must not pick up `phase_ordercue_object`
            continue
        files.setdefault(sub, {})[cond] = f
    need = {"pooled": CONDS, "pre": ("pre",), "post1": ("post1",), "post2": ("post2",),
            "post1_minus_pre": ("post1", "pre"), "post2_minus_post1": ("post2", "post1"),
            "post2_minus_pre": ("post2", "pre"), "post_minus_pre": CONDS}
    out = {}
    for sub, d in files.items():
        if contrast == "per_second_rt":
            out[sub] = image.mean_img([d[c] for c in CONDS if c in d], copy_header=True)
        elif all(c in d for c in need.get(contrast, (contrast,))):
            if contrast == "pooled":
                out[sub] = image.mean_img([d[c] for c in CONDS], copy_header=True)
            elif contrast == "post_minus_pre":
                out[sub] = image.math_img("(a + b) / 2 - c", a=d["post1"], b=d["post2"], c=d["pre"])
            elif "_minus_" in contrast:
                a, b = need[contrast]
                out[sub] = image.math_img("a - b", a=d[a], b=d[b])
            else:
                out[sub] = nib.load(d[contrast])
    return out


def main():
    if len(sys.argv) < 4 or sys.argv[3] not in CONTRASTS:
        sys.exit(f"usage: python E4_wholebrain_group.py <fixed|cue_to_response> <object|scene> <{'|'.join(CONTRASTS)}> "
                 "[n_permutations] [n_jobs]")
    model, kind, contrast = sys.argv[1:4]
    n_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    out_dir = str(step_output_dir(__file__, opj("native_first", model, kind) if NATIVE_FIRST else opj(model, kind)))

    maps = participant_maps(model, kind, contrast)
    subs = sorted(maps)
    print(f"{model} / {kind} / {contrast}: {len(subs)} participants, {n_perm} permutations", flush=True)
    mask_dir = str(step_input_dir("E3_wholebrain_maps", "masks"))
    mask = masking.intersect_masks([opj(mask_dir, f"{s}_brainmask_mni.nii.gz") for s in subs],
                                   threshold=1, connected=False)
    if NATIVE_FIRST:                                   # transformed maps are zero outside the native brain mask
        has = np.all([np.asarray(maps[s].dataobj) != 0 for s in subs], axis=0)
        mask = nib.Nifti1Image((np.asarray(mask.dataobj).astype(bool) & has).astype(np.uint8), mask.affine)
    print(f"group mask: {int(np.asarray(mask.dataobj).sum())} voxels", flush=True)

    res = non_parametric_inference([maps[s] for s in subs],
                                   design_matrix=pd.DataFrame({"intercept": np.ones(len(subs))}),
                                   mask=mask, n_perm=n_perm, two_sided_test=True, tfce=True,
                                   random_state=0, n_jobs=n_jobs, verbose=1)
    t_img, logp = res["t"], res["logp_max_tfce"]
    thr = image.math_img(f"t * (lp > {-np.log10(ALPHA)})", t=t_img, lp=logp)
    base = opj(out_dir, contrast)
    t_img.to_filename(base + "_t.nii.gz")
    logp.to_filename(base + "_tfce_logp.nii.gz")
    thr.to_filename(base + "_t_thresholded.nii.gz")
    pd.DataFrame(dict(subject=subs)).to_csv(base + "_participants.csv", index=False)

    t = np.asarray(t_img.dataobj)
    sig = np.asarray(thr.dataobj) != 0
    m = np.asarray(mask.dataobj) > 0
    print(f"significant (TFCE, corrected p < {ALPHA}): {int((sig & (t > 0)).sum())} voxels above, "
          f"{int((sig & (t < 0)).sum())} below, of {int(m.sum())}", flush=True)

    atlas = np.asarray(group_atlas(subs, t_img).dataobj)
    rows = []
    for code in np.unique(atlas[m]):
        name = label_name(code)
        if code == 0 or "white matter" in name or name.startswith("label"):
            continue
        r = m & (atlas == code)
        if r.sum() < 20:
            continue
        rows.append(dict(region=name, n_voxels=int(r.sum()), mean_t=t[r].mean(), max_t=t[r].max(), min_t=t[r].min(),
                         share_above=(sig & (t > 0))[r].mean(), share_below=(sig & (t < 0))[r].mean()))
    pd.DataFrame(rows).sort_values("mean_t", ascending=False).to_csv(base + "_regions.csv", index=False)

    try:
        if sig.any():
            peaks = get_clusters_table(thr, stat_threshold=1e-6, cluster_threshold=20, two_sided=True,
                                       min_distance=16)
            ijk = np.round(nib.affines.apply_affine(np.linalg.inv(t_img.affine),
                                                    peaks[["X", "Y", "Z"]].values.astype(float))).astype(int)
            peaks["region"] = [label_name(atlas[tuple(v)]) for v in ijk]
            peaks = peaks.rename(columns={"Peak Stat": "t"})
        else:
            peaks = pd.DataFrame(columns=["Cluster ID", "X", "Y", "Z", "t", "Cluster Size (mm3)", "region"])
        peaks.to_csv(base + "_peaks.csv", index=False)
    except Exception as err:                         # the maps are the result; the table is a convenience
        print(f"peak table not written: {err}")
    print("done")


if __name__ == "__main__":
    main()
