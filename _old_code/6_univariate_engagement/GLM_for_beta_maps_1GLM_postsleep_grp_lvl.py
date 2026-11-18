# GLM_for_beta_maps_1GLM_postsleep_grp_lvl.py
# Group-level (second-level) analysis for subject-level maps created by GLM_for_beta_maps_1GLM_postsleep.py
#
# Expects files like:
#   <this_dir>/GLM_for_beta_maps_1GLM_postsleep/glm_maps/sub-XX/Big_GLM_remem_vs_nonremem_effect.nii.gz
# Writes:
#   <this_dir>/GLM_for_beta_maps_1GLM_postsleep_grp_lvl/group_maps/group_qmark_[t|z].nii.gz

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
import numpy as np
from nilearn.image import threshold_img

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.image import load_img, math_img
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt

from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import load_img, concat_imgs, new_img_like, mean_img
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
from pathlib import Path
import numpy as np

from nilearn.image import load_img, mean_img, math_img
from nilearn.plotting import (
    plot_stat_map,
    plot_glass_brain,
    plot_roi,
    view_img,
    view_img_on_surf,
)
from nilearn.reporting import get_clusters_table


def script_dir() -> Path:
    return Path(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------
# Paths (tied to filenames)
# ------------------------------------------------------------
BASE = script_dir()

firstlevel_root = BASE / "GLM_for_beta_maps_1GLM_postsleep" / "glm_maps"
if not firstlevel_root.exists():
    raise FileNotFoundError(
        f"Expected first-level output folder not found:\n  {firstlevel_root}\n"
        "Make sure GLM_for_beta_maps_1GLM_postsleep.py has been run and created GLM_for_beta_maps_1GLM_postsleep/glm_maps/..."
    )

sub_eff_maps = sorted(firstlevel_root.glob("sub-*/Big_GLM_remem_vs_nonremem_effect.nii.gz")) # sublevel_qmark_eff for precision weighted

if len(sub_eff_maps) == 0:
    raise FileNotFoundError(
        f"No subject maps found under:\n  {firstlevel_root}\n"
        "Expected pattern: sub-*/Big_GLM_remem_vs_nonremem_effect.nii.gz"
    )

print(f"Found {len(sub_eff_maps)} subject maps.")
print("Example:", sub_eff_maps[0])


# ------------------------------------------------------------
# Inputs you should have at group level
# ------------------------------------------------------------
# sub_eff_maps = [Path(...sub-01...effect.nii.gz), Path(...sub-02...effect.nii.gz), ...]
# (optional) sub_var_maps = [Path(...var.nii.gz), ...]  # not used by SecondLevelModel directly

subjects = [p.parent.name for p in sub_eff_maps]  # or parse however you prefer

design_matrix = pd.DataFrame({"intercept": 1.0}, index=subjects)

mni_mask = load_mni152_brain_mask()
mni_mask_resampled = resample_to_img(mni_mask, load_img(str(sub_eff_maps[0])), interpolation="nearest")

slm = SecondLevelModel(
    smoothing_fwhm=4,
    mask_img=mni_mask_resampled,   # <-- add this
).fit(
    [str(p) for p in sub_eff_maps],
    design_matrix=design_matrix,
)

group_t = slm.compute_contrast("intercept", output_type="stat")
group_z = slm.compute_contrast("intercept", output_type="z_score")
group_p = slm.compute_contrast("intercept", output_type="p_value")

# ------------------------------------------------------------
# Group Cohen's d (one-sample): d = t / sqrt(n)
# (This is the standardized effect size corresponding to your group t-test.)
# ------------------------------------------------------------
n = len(sub_eff_maps)
t_data = group_t.get_fdata()
d_data = t_data / np.sqrt(n)
group_d = new_img_like(group_t, d_data)

# ------------------------------------------------------------
# Multiple comparisons correction (parametric)
# ------------------------------------------------------------
t_fdr, thr_fdr = threshold_stats_img(
    group_t, alpha=0.05, height_control="fdr", two_sided=True, cluster_threshold=0
)

t_bonf, thr_bonf = threshold_stats_img(
    group_t, alpha=0.05, height_control="bonferroni", two_sided=True, cluster_threshold=0
)

# ------------------------------------------------------------
# Multiple comparisons correction (nonparametric permutation FWE) - recommended
# This returns a "neg_log10_pvals" style map by default (depending on nilearn version).
# ------------------------------------------------------------
perm = non_parametric_inference(
    second_level_input=[str(p) for p in sub_eff_maps],
    design_matrix=design_matrix,
    second_level_contrast="intercept",
    n_perm=5000,
    two_sided_test=True,
    smoothing_fwhm=None,
    n_jobs=-1,
)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
out_dir = Path(BASE) / "GLM_for_beta_maps_1GLM_postsleep_grp_lvl" / "group_maps"
out_dir.mkdir(parents=True, exist_ok=True)

group_t.to_filename(out_dir / "group_t.nii.gz")
group_z.to_filename(out_dir / "group_z.nii.gz")
group_p.to_filename(out_dir / "group_p.nii.gz")
group_d.to_filename(out_dir / "group_cohens_d_from_t.nii.gz")

t_fdr.to_filename(out_dir / "group_t_FDR_q05.nii.gz")
t_bonf.to_filename(out_dir / "group_t_BONF_a05.nii.gz")

# perm is often a dict-like object; nilearn typically returns an image in perm["logp_max_t"] or similar.
# Try these common keys:
if isinstance(perm, dict):
    for k in ["logp_max_t", "logp_max_stat", "logp_max_size", "logp_max_tfce", "logp"]:
        if k in perm and hasattr(perm[k], "to_filename"):
            perm[k].to_filename(out_dir / f"perm_{k}.nii.gz")
else:
    # if it directly returns an image
    perm.to_filename(out_dir / "perm_map.nii.gz")

print("Saved group outputs to:", out_dir)
print(f"FDR |t| threshold: {float(thr_fdr):.4f}")
print(f"Bonferroni |t| threshold: {float(thr_bonf):.4f}")

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------


# -------------------------
# Load maps
# -------------------------
out_dir = Path(BASE) / "GLM_for_beta_maps_1GLM_postsleep_grp_lvl" / "group_maps"

group_t = load_img(out_dir / "group_t.nii.gz")
group_z = load_img(out_dir / "group_z.nii.gz")
group_d = load_img(out_dir / "group_cohens_d_from_t.nii.gz")

t_fdr  = load_img(out_dir / "group_t_FDR_q05.nii.gz")
t_bonf = load_img(out_dir / "group_t_BONF_a05.nii.gz")

bg = mean_img(load_img(sub_eff_maps[0]))

# -------------------------
# Helpers
# -------------------------
def robust_abs_max(img, q=99.5):
    x = img.get_fdata()
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(np.abs(x), q)) if x.size else 0.0

def save_stat(img, title, fname, threshold=None, vmax=None, cut_coords=(20, 0, 0)):
    fig = plot_stat_map(
        img,
        bg_img=bg,
        title=title,
        threshold=threshold,
        display_mode="ortho",
        cut_coords=cut_coords,
        colorbar=True,
        vmax=vmax,
        symmetric_cbar=True,  # <-- correct for nilearn 0.13
    )
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    return fig

def save_glass(img, title, fname, threshold=None, vmax=None):
    fig = plot_glass_brain(
        img,
        title=title,
        threshold=threshold,
        display_mode="lyrz",
        colorbar=True,
        vmax=vmax,
        symmetric_cbar=True,  # <-- correct for nilearn 0.13
    )
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    return fig

# -------------------------
# Make figures
# -------------------------
t_vmax = robust_abs_max(group_t)
z_vmax = robust_abs_max(group_z)
d_vmax = robust_abs_max(group_d)

# # Unthresholded
# save_stat(group_t, "Group t (unthresholded)", "viz_group_t_unthr.png", vmax=t_vmax)
# save_glass(group_t, "Group t (unthresholded) - glass", "viz_group_t_unthr_glass.png", vmax=t_vmax)

# save_stat(group_z, "Group z (unthresholded)", "viz_group_z_unthr.png", vmax=z_vmax)
# save_glass(group_z, "Group z (unthresholded) - glass", "viz_group_z_unthr_glass.png", vmax=z_vmax)

# save_stat(group_d, "Group Cohen's d", "viz_group_d_unthr.png", vmax=d_vmax)
# save_glass(group_d, "Group Cohen's d - glass", "viz_group_d_unthr_glass.png", vmax=d_vmax)

# # Corrected
# save_stat(t_fdr, "Group t (FDR q=0.05)", "viz_group_t_fdr.png", threshold=0.0, vmax=t_vmax)
# save_glass(t_fdr, "Group t (FDR q=0.05) - glass", "viz_group_t_fdr_glass.png", threshold=0.0, vmax=t_vmax)

# save_stat(t_bonf, "Group t (Bonferroni a=0.05)", "viz_group_t_bonf.png", threshold=0.0, vmax=t_vmax)
# save_glass(t_bonf, "Group t (Bonferroni) - glass", "viz_group_t_bonf_glass.png", threshold=0.0, vmax=t_vmax)

# print("Saved figures to:", out_dir)

# -------------------------
# Interactive
# -------------------------
# plot t values FDR
#view_img(group_t, bg_img=bg, title="Interactive: group t", threshold=2).open_in_browser()
view_img(t_fdr, bg_img=bg, title="Interactive: group t (FDR q=0.05)", threshold=3.0).open_in_browser()

# plot t Bonferrioni
view_img(t_bonf, bg_img=bg, title="Interactive: group t (Bonferroni)", threshold=3.0).open_in_browser()

# plot z scores view_img
thresholded_z = threshold_img(
    group_z,
    threshold=3.29,
    cluster_threshold=10,
    two_sided=True,
)
view_img(thresholded_z, bg_img=bg, title="Group z (thresholded)").open_in_browser()
save_glass(thresholded_z, "Group z (thresholded) - glass", "viz_group_z_thr_glass.png", vmax=z_vmax)

# plot d values
view_img(group_d, bg_img=bg, title="Interactive: group Cohen's d", threshold=0.5).open_in_browser()


# ------------------------------------------------------------
# 5) Cluster tables (CSV) for reporting
# ------------------------------------------------------------
# For cluster tables, feed a stat map with a numeric threshold.
# Here we use the unthresholded group_t but apply a chosen |t| cutoff.
t_cut = 3.1  # e.g., ~p<0.002 uncorrected; adjust as needed
clusters_pos = get_clusters_table(group_t, stat_threshold=t_cut, cluster_threshold=10, two_sided=False)
clusters_neg = get_clusters_table(math_img("-img", img=group_t), stat_threshold=t_cut, cluster_threshold=10, two_sided=False)

clusters_pos.to_csv(out_dir / "clusters_positive_t.csv", index=False)
clusters_neg.to_csv(out_dir / "clusters_negative_t.csv", index=False)

print("Saved figures + cluster tables to:", out_dir)
print("Cluster tables:")
print(" -", out_dir / "clusters_positive_t.csv")
print(" -", out_dir / "clusters_negative_t.csv")


#################################################################
# Masking
#################################################################

# ###############
# ###############
# ################

# # --- ROI extraction: hippocampus + parietal cortex (mask + save + optional mean value) ---
# from pathlib import Path
# import numpy as np

from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.maskers import NiftiMasker

# # output folder
# roi_dir = out_dir / "roi_maps"
# roi_dir.mkdir(parents=True, exist_ok=True)

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

# parietal cortex (cortical) — pick what you mean by "parietal"
# Common parietal ROIs: Superior Parietal Lobule, Inferior Parietal Lobule, Angular Gyrus, Supramarginal Gyrus, Precuneus.
parietal_names = [
    "Precuneous Cortex",                 # posterior midline hub
    "Cingulate Gyrus, posterior division",  # PCC / retrosplenial complex
    "Angular Gyrus",                   # ventral lateral parietal (BA39)
]
parietal_masks = [label_mask(atlas_img, labels, n) for n in parietal_names]
parietal_mask = math_img(" + ".join([f"m{i}" for i in range(len(parietal_masks))]) + " > 0",
                         **{f"m{i}": parietal_masks[i] for i in range(len(parietal_masks))})

#################################################################
# Apply ROI masks to the *t-maps* (and optionally to FDR/Bonf maps)
#################################################################

# output folder for ROI-masked maps
roi_dir = out_dir / "roi_maps"
roi_dir.mkdir(parents=True, exist_ok=True)

# --- 1) Resample your ROI masks to the SAME grid as the group t-map ---
# (use group_t as target, not group_z, since you want t-maps)
hipp_mask_rs = resample_to_img(hipp_mask, group_t, interpolation="nearest")
parietal_mask_rs = resample_to_img(parietal_mask, group_t, interpolation="nearest")

# --- 2) Mask the t-maps (unthresholded) ---
group_t_hipp = math_img("t * m", t=group_t, m=hipp_mask_rs)
group_t_par  = math_img("t * m", t=group_t, m=parietal_mask_rs)

t_hipp_path = roi_dir / "group_t_hippocampus.nii.gz"
t_par_path  = roi_dir / "group_t_parietal.nii.gz"
group_t_hipp.to_filename(t_hipp_path)
group_t_par.to_filename(t_par_path)

print("Saved ROI-masked *group t* maps:")
print(" ", t_hipp_path)
print(" ", t_par_path)

# --- 3) Mask the corrected maps too (optional but often useful) ---
# These are already thresholded images; ROI-masking just restricts to ROI.
t_fdr_hipp = math_img("t * m", t=t_fdr, m=hipp_mask_rs)
t_fdr_par  = math_img("t * m", t=t_fdr, m=parietal_mask_rs)

t_bonf_hipp = math_img("t * m", t=t_bonf, m=hipp_mask_rs)
t_bonf_par  = math_img("t * m", t=t_bonf, m=parietal_mask_rs)

t_fdr_hipp.to_filename(roi_dir / "group_t_FDR_q05_hippocampus.nii.gz")
t_fdr_par.to_filename(roi_dir / "group_t_FDR_q05_parietal.nii.gz")
t_bonf_hipp.to_filename(roi_dir / "group_t_BONF_a05_hippocampus.nii.gz")
t_bonf_par.to_filename(roi_dir / "group_t_BONF_a05_parietal.nii.gz")

# --- 4) Quick interactive views (optional) ---
view_img(t_fdr_hipp, bg_img=bg, title="Group t in Hippocampus (ROI-masked)", threshold=0.0).open_in_browser()
view_img(t_fdr_par,  bg_img=bg, title="Group t in Parietal ROI (ROI-masked)", threshold=0.0).open_in_browser()

# --- 5) Optional summary values (mean t within ROI) ---
hipp_mean_t = NiftiMasker(mask_img=hipp_mask_rs).fit_transform(group_t).mean()
par_mean_t  = NiftiMasker(mask_img=parietal_mask_rs).fit_transform(group_t).mean()
hipp_max_t = NiftiMasker(mask_img=hipp_mask_rs).fit_transform(group_t).max()
par_max_t  = NiftiMasker(mask_img=parietal_mask_rs).fit_transform(group_t).max()
print(f"Mean group t in hippocampus ROI: {hipp_mean_t:.4f}")
print(f"Mean group t in parietal ROI:    {par_mean_t:.4f}")
print(f"Max group t in hippocampus ROI: {hipp_max_t:.4f}")
print(f"Max group t in parietal ROI:    {par_max_t:.4f}")
