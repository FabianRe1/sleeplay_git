# GLM_for_beta_maps_grp_lvl.py
# Group-level (second-level) analysis for subject-level maps created by GLM_for_beta_maps.py
#
# Expects files like:
#   <this_dir>/GLM_for_beta_maps/glm_maps/sub-XX/sublevel_qmark_eff.nii.gz
# Writes:
#   <this_dir>/GLM_for_beta_maps_grp_lvl/group_maps/group_qmark_[t|z].nii.gz

import os
from pathlib import Path
import pandas as pd

from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import load_img
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd


from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.image import load_img, math_img
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt

def script_dir() -> Path:
    return Path(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------
# Paths (tied to filenames)
# ------------------------------------------------------------
BASE = script_dir()

firstlevel_root = BASE / "GLM_for_beta_maps" / "glm_maps"
if not firstlevel_root.exists():
    raise FileNotFoundError(
        f"Expected first-level output folder not found:\n  {firstlevel_root}\n"
        "Make sure GLM_for_beta_maps.py has been run and created GLM_for_beta_maps/glm_maps/..."
    )

sub_eff_maps = sorted(firstlevel_root.glob("sub-*/sublevel_qmark_eff_MEAN.nii.gz")) # sublevel_qmark_eff for precision weighted

if len(sub_eff_maps) == 0:
    raise FileNotFoundError(
        f"No subject maps found under:\n  {firstlevel_root}\n"
        "Expected pattern: sub-*/sublevel_qmark_eff.nii.gz"
    )

print(f"Found {len(sub_eff_maps)} subject maps.")
print("Example:", sub_eff_maps[0])

# ------------------------------------------------------------
# Second-level one-sample t-test
# ------------------------------------------------------------
design_matrix = pd.DataFrame(
    {"intercept": 1.0},
    index=[p.parent.name for p in sub_eff_maps],  # e.g., 'sub-13'
)

slm = SecondLevelModel(smoothing_fwhm=None).fit(
    [str(p) for p in sub_eff_maps],
    design_matrix=design_matrix,
)

group_z = slm.compute_contrast(output_type="z_score")
group_t = slm.compute_contrast(output_type="stat")

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
out_dir = BASE / "GLM_for_beta_maps_grp_lvl" / "group_maps"
out_dir.mkdir(parents=True, exist_ok=True)

z_path = out_dir / "group_qmark_z.nii.gz"
t_path = out_dir / "group_qmark_t.nii.gz"
group_z.to_filename(z_path)
group_t.to_filename(t_path)

print("Saved group maps:")
print("  Z:", z_path)
print("  T:", t_path)

    
from nilearn.plotting import view_img
from nilearn.image import mean_img
from pathlib import Path

# Example: interactive Z-map view
# Toggle between t and z maps
use_z_map = True  # Set to False to view t-map instead

stat_map = group_z if use_z_map else group_t
map_type = "z" if use_z_map else "t"
threshold = 2.6 if use_z_map else 3.0  # Adjust thresholds as needed

# threshold cluster size
thresholded_zmap = threshold_stats_img(
    stat_map,
    alpha=0.01,
    height_control="fpr",
    cluster_threshold=10,
    two_sided=True,
)[0]
thresholded_tmap = threshold_stats_img(
    stat_map,
    alpha=0.01,
    height_control="fpr",
    cluster_threshold=10,
    two_sided=True,
)[0]
stat_map = thresholded_zmap if use_z_map else thresholded_tmap

view = view_img(
    stat_map,
    threshold=0,
    bg_img=mean_img(load_img(sub_eff_maps[0])),
    symmetric_cmap=True,
    title=f'View of group {map_type} map',
)
view.open_in_browser()


###############
###############
################

# --- ROI extraction: hippocampus + parietal cortex (mask + save + optional mean value) ---
from pathlib import Path
import numpy as np

from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.maskers import NiftiMasker

# output folder
roi_dir = out_dir / "roi_maps"
roi_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Load an atlas in MNI space (Harvard-Oxford)
# ----------------------------
atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")  # cortex labels (MNI 2mm)
atlas_img = load_img(atlas.maps)
labels = atlas.labels  # list of label names

# for hippocampus use the subcortical atlas:
atlas_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
atlas_sub_img = load_img(atlas_sub.maps)
labels_sub = atlas_sub.labels

# ----------------------------
# 2) Build ROI masks by label name
# ----------------------------
def label_mask(atlas_img, labels, name_contains: str):
    """Return a binary mask image where atlas label contains substring."""
    hits = [i for i, lab in enumerate(labels) if isinstance(lab, str) and name_contains.lower() in lab.lower()]
    if len(hits) == 0:
        raise ValueError(f"No atlas labels matched: {name_contains}\nExample labels: {labels[:30]}")
    # Harvard-Oxford label images are integer-coded; 0 is background.
    # Keep any of the matched indices.
    expr = " + ".join([f"(img == {i})" for i in hits])
    return math_img(f"({expr}) > 0", img=atlas_img)

# hippocampus (subcortical)
hipp_mask = label_mask(atlas_sub_img, labels_sub, "Hippocampus")  # matches Left/Right Hippocampus

# parietal cortex (cortical) — pick what you mean by “parietal”
# Common parietal ROIs: Superior Parietal Lobule, Inferior Parietal Lobule, Angular Gyrus, Supramarginal Gyrus, Precuneus.
parietal_names = [
    "Precuneous Cortex",                 # posterior midline hub
    "Cingulate Gyrus, posterior division",  # PCC / retrosplenial complex
    "Angular Gyrus",                   # ventral lateral parietal (BA39)
]
parietal_masks = [label_mask(atlas_img, labels, n) for n in parietal_names]
parietal_mask = math_img(" + ".join([f"m{i}" for i in range(len(parietal_masks))]) + " > 0",
                         **{f"m{i}": parietal_masks[i] for i in range(len(parietal_masks))})

# ----------------------------
# 3) Resample masks to your group_z grid (safe even if already 2mm)
# ----------------------------
hipp_mask_rs = resample_to_img(hipp_mask, group_z, interpolation="nearest")
parietal_mask_rs = resample_to_img(parietal_mask, group_z, interpolation="nearest")

# ----------------------------
# 4) Mask the group maps and save ROI-only images
# ----------------------------
group_z_hipp = math_img("z * m", z=group_z, m=hipp_mask_rs)
group_z_par  = math_img("z * m", z=group_z, m=parietal_mask_rs)

hipp_path = roi_dir / "group_qmark_z_hippocampus.nii.gz"
par_path  = roi_dir / "group_qmark_z_parietal.nii.gz"
group_z_hipp.to_filename(hipp_path)
group_z_par.to_filename(par_path)

print("Saved ROI-masked maps:")
print(" ", hipp_path)
print(" ", par_path)

# ----------------------------
# 5) Optional: extract a single summary value (mean Z within ROI)
# ----------------------------
hipp_mean = NiftiMasker(mask_img=hipp_mask_rs).fit_transform(group_z).mean()
par_mean  = NiftiMasker(mask_img=parietal_mask_rs).fit_transform(group_z).mean()
hipp_var = NiftiMasker(mask_img=hipp_mask_rs).fit_transform(group_z).var()
par_var  = NiftiMasker(mask_img=parietal_mask_rs).fit_transform(group_z).var()
print(f"Mean group Z in hippocampus: {hipp_mean:.4f} (var: {hipp_var:.4f})")
print(f"Mean group Z in parietal ROI: {par_mean:.4f} (var: {par_var:.4f})")