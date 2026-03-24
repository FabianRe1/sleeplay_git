#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mni_plot_freesurfer_labels.py

Create an MNI-space sanity-check plot for a set of FreeSurfer label IDs
(from fMRIPrep's aparcaseg_dseg in MNI space).

Example (cortical_memory):
  1014,2014,1026,2026,1002,2002,1028,2028,1010,2010,1023,2023,1025,2025,1008,2008
  
  
                      cortical_memory = \
                        (FS==1014) | (FS==2014) | \
                        (FS==1026) | (FS==2026) | \
                        (FS==1002) | (FS==2002) | \
                        (FS==1028) | (FS==2028) | \
                        (FS==1010) | (FS==2010) | \
                        (FS==1023) | (FS==2023) | \
                        (FS==1025) | (FS==2025) | \
                        (FS==1008) | (FS==2008)
                    # Breakdown of regions included above:
                    # mPFC: Medial OFC (1014/2014), Rostral ACC (1026/2026), Caudal ACC (1002/2002), Superior Frontal (1028/2028), Frontal Pole (1032/2032)
                    # Precuneus: Precuneus (1025/2025)
                    # Parietal:  Inf Parietal (1008/2008)
"""

import os
import glob
import argparse
import numpy as np
from nilearn import image, plotting, datasets
from nilearn.image import load_img


def find_mni_aparcaseg(fmriprep_root: str, subj: str, session: str, space: str) -> str:
    func_dir = os.path.join(fmriprep_root, subj, session, "func")
    pattern = os.path.join(
        func_dir,
        f"{subj}_{session}_task-*_run-*_space-{space}_desc-aparcaseg_dseg.nii.gz",
    )
    files = sorted(glob.glob(pattern), key=os.path.basename)
    if not files:
        raise FileNotFoundError(
            f"No MNI aparcaseg_dseg found.\nPattern:\n  {pattern}\n"
            f"Tip: check that fMRIPrep produced aparcaseg in space={space}."
        )
    return files[0]  # just take first run for visualization


def make_label_mask(dseg_img, labels):
    dseg = np.rint(np.asarray(dseg_img.dataobj)).astype(np.int32)
    mask = np.isin(dseg, np.asarray(labels, dtype=np.int32)).astype(np.uint8)
    return image.new_img_like(dseg_img, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep_root", required=True)
    ap.add_argument("--subj", required=True, help='e.g., "sub-13"')
    ap.add_argument("--session", required=True, help='e.g., "ses-02"')
    ap.add_argument("--space", default="MNI152NLin2009cAsym")
    ap.add_argument(
        "--labels",
        required=True,
        help="Comma-separated label IDs, e.g. 1014,2014,1026,2026,...",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="fs_labels", help="basename for outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels = [int(x.strip()) for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise ValueError("No labels parsed from --labels.")

    dseg_path = find_mni_aparcaseg(args.fmriprep_root, args.subj, args.session, args.space)
    dseg_img = load_img(dseg_path)

    mask_img = make_label_mask(dseg_img, labels)

    # Put mask on the MNI152 template grid for clean display (nearest neighbor)
    mni = datasets.load_mni152_template()
    mask_mni = image.resample_to_img(mask_img, mni, interpolation="nearest")

    title = f"{args.subj} {args.session} {args.space} | labels: {len(labels)}"

    out_ortho = os.path.join(args.out_dir, f"{args.subj}_{args.session}_{args.name}.png")
    disp = plotting.plot_roi(
        mask_mni,
        bg_img=mni,
        title=title,
        display_mode="ortho",
        cut_coords=None,  # auto
        annotate=True,
        draw_cross=False,
    )
    disp.savefig(out_ortho, dpi=180)
    disp.close()

    out_glass = os.path.join(args.out_dir, f"{args.subj}_{args.session}_{args.name}_glass.png")
    disp = plotting.plot_glass_brain(
        mask_mni,
        title=title,
        display_mode="lyrz",
        plot_abs=False,
    )
    disp.savefig(out_glass, dpi=180)
    disp.close()

    print("Wrote:")
    print(" ", out_ortho)
    print(" ", out_glass)
    print("Using aparcaseg:", dseg_path)


if __name__ == "__main__":
    main()
