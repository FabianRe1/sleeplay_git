#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E8 - Check: fit in native space, then transform the maps to standard space.

E3 fits the models on data that fMRIPrep already resampled to standard space.
The other order is to fit in each participant's own anatomy (E3 with
E3_SPACE=T1w) and transform the resulting maps with fMRIPrep's T1w-to-MNI
transform. This step does that transformation for one participant (linear
interpolation, onto the grid of the participant's standard-space maps) and, in
group mode, compares the two routes.

Input   outputs/E3_wholebrain_maps/maps_T1w/<model>/     maps fitted in native space
        outputs/E3_wholebrain_maps/maps/<model>/         maps fitted in standard space
        <derivatives>/sub-XX/anat/sub-XX_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
Output  outputs/E8_native_then_standard/maps/<model>/<sub>_<time point>_<name>.nii.gz
        outputs/E8_native_then_standard/participant_agreement.csv, group_agreement.csv

The transformed maps can be given to E4 in place of the standard-space maps
(E4_MAPS=native_first), so that the whole group analysis follows the order
"fit in native space, transform, then test".

Usage:  python E8_native_then_standard.py <subject> [model]     default model: cue_search_fixbase
        python E8_native_then_standard.py group [model]

Needs the package nitransforms. Not a port of the original pipeline: added 2026-11-22.
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "B_decoding"))
from config import DERIVATIVES_DIR, step_input_dir, step_output_dir
from decoding_utils import parse_subject_arg

NAMES = ("object_search", "object_cue")
CONDS = ("pre", "post1", "post2")


def transform(sub, model):
    from nitransforms.manip import load as ntload
    from nitransforms.resampling import apply
    sub_mri = f"sub-{sub[1:]}"
    xfm = ntload(opj(str(DERIVATIVES_DIR), sub_mri, "anat",
                     f"{sub_mri}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"), fmt="itk")
    src = str(step_input_dir("E3_wholebrain_maps", opj("maps_T1w", model)))
    ref_dir = str(step_input_dir("E3_wholebrain_maps", opj("maps", model)))
    out = str(step_output_dir(__file__, opj("maps", model)))
    n = 0
    for f in sorted(glob.glob(opj(src, f"{sub}_*.nii.gz"))):      # every map of the model
        ref = opj(ref_dir, os.path.basename(f))
        if not os.path.exists(ref):
            continue
        apply(xfm, nib.load(f), reference=nib.load(ref), order=1).to_filename(opj(out, os.path.basename(f)))
        n += 1
    print(f"{sub}: {n} maps transformed")
    print("done")


def group(model):
    a_dir = str(step_input_dir("E3_wholebrain_maps", opj("maps", model)))
    b_dir = opj(str(step_output_dir(__file__, "maps")), model)
    mask_dir = str(step_input_dir("E3_wholebrain_maps", "masks"))
    rows, pooled = [], {}
    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(opj(b_dir, "*.nii.gz"))})
    masks = [np.asarray(nib.load(opj(mask_dir, f"{s}_brainmask_mni.nii.gz")).dataobj) > 0 for s in subs]
    gmask = np.all(masks, 0)
    for s in subs:
        for name in NAMES:
            both = []
            for c in CONDS:
                fa, fb = opj(a_dir, f"{s}_{c}_{name}.nii.gz"), opj(b_dir, f"{s}_{c}_{name}.nii.gz")
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                a, b = np.asarray(nib.load(fa).dataobj)[gmask], np.asarray(nib.load(fb).dataobj)[gmask]
                ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
                rows.append(dict(subject=s, name=name, cond=c, r=np.corrcoef(a[ok], b[ok])[0, 1],
                                 share_covered=ok.mean()))
                both.append((a, b))
            if len(both) == 3:
                pooled.setdefault(name, []).append((np.mean([x[0] for x in both], 0), np.mean([x[1] for x in both], 0)))
    out = str(step_output_dir(__file__))
    pd.DataFrame(rows).to_csv(opj(out, "participant_agreement.csv"), index=False)
    g = []
    for name, lst in pooled.items():
        ta = stats.ttest_1samp(np.array([x[0] for x in lst]), 0)[0]
        tb = stats.ttest_1samp(np.array([x[1] for x in lst]), 0)[0]
        thr = stats.t.ppf(1 - 0.0005, len(lst) - 1)
        sa, sb = np.abs(ta) > thr, np.abs(tb) > thr
        same = sa & sb & (np.sign(ta) == np.sign(tb))
        g.append(dict(name=name, n=len(lst), r_group_t=np.corrcoef(ta, tb)[0, 1], voxels_standard=int(sa.sum()),
                      voxels_native_then_standard=int(sb.sum()), dice=2 * same.sum() / (sa.sum() + sb.sum()),
                      max_abs_t_difference=float(np.max(np.abs(ta - tb))), median_abs_t_difference=float(np.median(np.abs(ta - tb)))))
    pd.DataFrame(g).to_csv(opj(out, "group_agreement.csv"), index=False)
    p = pd.DataFrame(rows)
    print(p.groupby("name").r.describe()[["count", "mean", "min", "50%", "max"]].round(3).to_string())
    print(pd.DataFrame(g).round(3).to_string(index=False))


if __name__ == "__main__":
    model = sys.argv[2] if len(sys.argv) > 2 else "cue_search_fixbase"
    if len(sys.argv) > 1 and sys.argv[1] == "group":
        group(model)
    else:
        transform(parse_subject_arg(sys.argv[1] if len(sys.argv) > 1 else None), model)
