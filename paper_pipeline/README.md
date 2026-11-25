# Sleeplay paper pipeline

One folder that recreates every analysis for the paper, from the raw behaviour
logs to the final statistics and figures. Steps are lettered by stage and
numbered in the order they run.

Status on 2026-11-18: stages A, B and C are ported. Every step was checked
against the existing results (details under each stage). The full pipeline has
not yet been run from start to end in this folder: `outputs/` holds stage A
for everyone and the stage B test runs for one participant.

Decisions taken during the step-by-step review are listed under "Decisions".
The full run has not been started; see "Open points".

## Rules

- Raw data (`behavior/`, fMRIPrep derivatives) is read-only. No step writes there.
- Every step writes to `outputs/<script name>/` and reads earlier results only from there.
- Every plot goes to `plots/<script name>/`, next to `outputs/`. The folder name is the script that drew it.
- All paths and constants come from `config.py`. To run elsewhere, set
  `SLEEPLAY_ROOT`, `SLEEPLAY_DERIVATIVES` and `SLEEPLAY_OUTPUT` (plots follow the
  output folder; `SLEEPLAY_PLOTS` overrides that).
- A port keeps the original analysis logic unchanged. Only paths and
  input/output handling are replaced, and each port is checked against the
  existing results before it counts as done.

## Setup

    source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
    cd paper_pipeline/A_behavior && python A1_add_tr_columns_main.py   # and so on

## Moving the analysis to another computer

Survey of 2026-11-23 (sizes measured on Hummel; nothing was copied). Set
`SLEEPLAY_ROOT`, `SLEEPLAY_DERIVATIVES` and `SLEEPLAY_OUTPUT` on the new
machine. How much to copy depends on how far back the analysis should be
rerunnable there:

| Level | Can be rerun there | Copy | Size |
| --- | --- | --- | --- |
| 1 | Statistics and figures (stage C, D2-D4, E1-E2) on the existing results | scripts, behaviour, existing results | about 20 GB |
| 2 | Stages A to C from preprocessed BOLD (masks, classifiers, sequentiality) | level 1 + native-space BOLD of ses-02, 03, 04, 06 | about 800 GB |
| 3 | Also D1 and the whole-brain univariate maps (E3 onwards) | level 2 + standard-space BOLD of ses-04, 06 | about 1,260 GB |

Always needed (under 1 GB):

- `sleeplay_git/paper_pipeline/` scripts (about 1 MB) and `.my_hidden_root_file`
  in the project folder (or set `SLEEPLAY_ROOT`).
- `Sleeplay_Hummel/behavior/` (85 MB).
- Python environment `/usw/u/bbc6124/u/bbc6124/venvs/myenv` (716 MB): recreate
  it from `pip freeze` instead of copying.

Level 1, existing results under `sleeplay_git/`:

- `4_decoding_claude_retrieval/` 5.2 GB (classifier output, slope, sinusoid)
- `7_rsa_exploratory/` 9.1 GB (statistics on existing results, trial patterns)
- `3_reliability_voxel_selection/` 4.2 GB (voxel masks)
- `6_univariate_engagement_claude_v2/` 1 GB
- `Sleeplay_Hummel/_old/exploration/tmp_explore_union_vs_intersection/` 1.9 GB (classifier
  accuracy by trial type; temporary folder)

Levels 2 and 3, from the derivatives folder
(`/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives`, 6.3 TB in
total, 46 participants). Only these files are read by the pipeline:

| Session | Content | Participants | `space-T1w_desc-preproc_bold` | `space-MNI152NLin2009cAsym_desc-preproc_bold` |
| --- | --- | --- | --- | --- |
| ses-02 | localizer | 46 | 219 GB | not needed (280 GB) |
| ses-03 | localizer | 44 | 207 GB | not needed (265 GB) |
| ses-04 | retrieval, evening | 44 | 191 GB | 245 GB (level 3) |
| ses-06 | retrieval, morning | 40 | 166 GB | 211 GB (level 3) |

Plus, for the same sessions, `*_desc-confounds_timeseries.tsv`,
`*_desc-brain_mask.nii.gz` and `*_desc-aparcaseg_dseg.nii.gz` (about 2 GB
together), and from `sub-XX/anat/` the transform
`*_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5` and the standard-space
`aparcaseg` file (the `anat` folders are 14.5 GB in total).

Not needed: surface BOLD files (`*.func.gii`, 2,583 GB), the sleep sessions
ses-01 and ses-05 (815 GB native, 1,045 GB standard space), fMRIPrep report
figures (19 GB), `2_get_threshold_roi/` (316 GB of old masks; B1 rebuilds
them), `_old_code/` (23 GB), `logs/` (4.4 GB).

Notes: the sizes cover all 46 participants; the analyses use 32 to 36, so
copying only those cuts levels 2 and 3 by about a quarter. The B1 output grows
to several times its current 6.6 GB in a full run. The size of the raw data
(before fMRIPrep) could not be read from this account; the pipeline does not
use it.

## Plots

All figures are in `plots/`, one folder per script. Tables, masks, classifiers
and other data stay in `outputs/`.

| Folder in `plots/` | Contains |
| --- | --- |
| `A5_analyse_behavior/` | Retrieval rate and reaction time figures (4 figures, PDF and PNG); `fig4` is reaction time by sequence position per session |
| `B2_voxel_reliability_masks/sXX/ses-0Y/` | Voxel reliability maps per participant and localizer |
| `B4_train_classifiers/` | Cross-validation accuracy per participant and ROI |
| `B5_classifier_accuracy/` | Group classifier accuracy per ROI, participant and stimulus; `decoding_per_mask_cond*_by_trial_type` splits image shown vs recalled |
| `B6a_apply_classifiers_encoding/<roi>/` | Encoding-session decoding time courses per participant |
| `B6b_encoding_decoding_group/` | The same at group level, per ROI |
| `B7_apply_classifiers_retrieval/<session>/` | Retrieval decoding time courses per participant and ROI |
| `B8_sequentiality_soda/<session>/` | Slope time courses per ROI (4 plots each) |
| `B9_sinusoid_summary/<preset>/<roi>/` | Example fits, amplitude summary, fit parameters |
| `C2_item_reactivation/` | Item-level reactivation figures |
| `C3_window_comparison/` | Heatmaps comparing the metric versions |
| `C6_sequence_vs_control/` | Retrieved and control sequence per session, four main ROIs, four measures |
| `C5_predecessor_reactivation/` | Evidence by lag to the retrieved item; predecessor and successor per session; evidence per TR |
| `C4a_figure_panels_main/full/` and `/matched/` | Panels for the main figures |
| `C4b_figure_panels_statistics/` | Panels for the statistics figures |

The other steps draw no plots. On 2026-11-18 only `A5_analyse_behavior/` is
filled; the rest fills when the full pipeline runs. Until then, the current
plots of the later steps are in the old folders under
`4_decoding_claude_retrieval/` (for example `5.0_thesis_figures/`,
`5.2_replay_statistics_figures/`).

## Stage A: behaviour (ported, verified)

| Step | Script | Does | Ported from | Runtime |
| --- | --- | --- | --- | --- |
| A1 | `A1_add_tr_columns_main.py` | HRF-shifted onsets and TR indices for the main task | `1_prepare_behavior/1_enc_ret_extract_TR.py` | about 1 min |
| A2 | `A2_add_tr_columns_localizer.py` | The same for the two localizers | `1_prepare_behavior/1_localizer_extract_TR.py` | about 1 min |
| A3 | `A3_check_completeness.py` | Lists missing or incomplete behaviour files | `1_prepare_behavior/2_check_missing.py` | seconds |
| A4 | `A4_aggregate_behavior.py` | One table with all participants and sessions | `1_prepare_behavior/3_aggregate_behav.py` | seconds |
| A5 | `A5_analyse_behavior.py` | Retrieval rate and reaction time by type, position and session; reaction time by position per session; mixed models | `1_prepare_behavior/4_analyse_behav_data.py` | seconds |
| A6 | `A6_position_cost.py` | Shape of the reaction-time profile: dip at the last position, position cost over positions 1-4 and as a share of mean reaction time, per session | new (2026-11-19) | seconds |

Verification against the existing results (2026-11-18):

- A1: all 123 files have the same values as the existing `*_w_HRF.csv` files.
  The text differs in the last printed digit of some floats (newer pandas).
- A2: all 75 files have the same values as the existing files to floating-point
  precision. 36 existing files carry two extra derived columns
  (`encoding_stimulus`, `encoding_category`); A2 leaves them out for everyone,
  and the classifier step rebuilds them itself.
- A4: the combined table has the same values as the existing one (113,607 rows).
- A5: the session models reproduce the thesis values (retrieval rate, morning 1
  vs evening: b = 2.38, z = 3.50; reaction time: b = -0.52 s, z = -9.72).

Differences from the original scripts:

- The originals wrote the `*_w_HRF.csv` files into `behavior/`. These steps
  write to `outputs/` instead, so later stages must read them from there.
- The original aggregate script had a hard-coded laptop path.
- A3 writes its findings to a table instead of only printing them.
- A5 has one addition that is not in the original (2026-11-18): reaction time
  by sequence position separately for evening, morning 1 and morning 2
  (`fig4_rt_by_position_per_session`), with per-participant slopes and a mixed
  model in `outputs/A5_analyse_behavior/rt_by_position_by_session.txt`. The
  original figures and models are unchanged.

A6 is an addition, not a port (2026-11-19); run on the pipeline's own stage A
output, 39 participants. Object trials:

- Position 5 is answered faster than position 4 in every session (-0.38 s,
  -0.46 s, -0.28 s; t = -3.1, -3.7, -2.7) and about 0.7 to 0.9 s faster than a
  straight line through positions 1 to 4 predicts (t = -5.6 to -6.4). The last
  item is not reached by stepping through the whole sequence.
- The cost per position over positions 1 to 4 falls in seconds across the
  three retrievals but stays the same share of mean reaction time (17, 17 and
  16 percent; no difference between sessions, p >= .18). Retrieval gets faster
  as a whole; the position cost does not disappear, which speaks against
  direct access ("cached" answers) at the repeated retrieval.
- Scene trials show the same profile; there the share falls slightly at the
  repeated retrieval (19 to 16 percent, p = .02).

## Stage B: decoding and replay (ported, verified)

Scripts are in `B_decoding/`. "Job" means the step runs on the cluster through
`submit_cluster_jobs.sh`; the others run directly with `python <script>`.

| Step | Script | Does | Ported from | Runs as | Runtime |
| --- | --- | --- | --- | --- | --- |
| B1 | `B1_threshold_roi_masks.py` | ROI masks from FreeSurfer labels, thresholded by signal | `2_get_threshold_roi/localizer_getThresholdedROI.py` | job per participant | about 3 min per session |
| B2 | `B2_voxel_reliability_masks.py` | Voxel reliability masks per localizer | `3_reliability_voxel_selection/generate_voxel_reliability_mask_localizer_1.py` | job per participant | about 1 hour |
| B3 | `B3_combine_localizer_masks.py` | Combine the two localizers' masks | `3_reliability_voxel_selection/combine_loc1_and_loc3_masks.py` | direct | about 10 min |
| B4 | `B4_train_classifiers.py` | Train classifiers on the localizers (encoding + retrieval trials) | `4.1_train_visual_classifier.py` | job per participant | about 10 min per ROI |
| B5 | `B5_classifier_accuracy.py` | Cross-validated classifier accuracy: group plots and `accuracy_summary.csv` (all trials, image shown, image recalled; t-test against chance, Holm across ROIs) | `4.2_analyse_best_classifier_performance.py` | direct | about 1 min |
| B6a | `B6a_apply_classifiers_encoding.py` | Apply classifiers to the encoding session (sanity check) | `4.3_transfer_visual_classifier_encoding.py` | job per participant | about 1 min per ROI |
| B6b | `B6b_encoding_decoding_group.py` | Group plots for B6a | `4.3_transfer_visual_classifier_encoding_grouplevel.py` | direct | about 1 min |
| B7 | `B7_apply_classifiers_retrieval.py` | Apply classifiers to retrieval, evening and morning | `4.4_transfer_visual_classifier_retrieval.py` | job per participant and session | about 1 min per ROI |
| B8 | `B8_sequentiality_soda.py` | Sequentiality metric (SODA slope, three windows; also with the retrieved item left out) | `4.6_SODA.py` | job per ROI and session | minutes |
| B9 | `B9_sinusoid_summary.py` | Sinusoid summary of the slope time course (three presets) | `4.7_sinus_1cycle.py` | job per ROI and session, 16 cores | about 2 min |
| B11 | `B11_searchlight_retrieval.py` | Searchlight version of B4 and B7 over the whole brain: object classifier per sphere (centres on a 4 mm grid in standard space, carried into each participant's anatomy; radii 10 to 30 mm), retrieved minus control for the retrieved item, its neighbours and the absolute slope per trial; classifier accuracy per sphere | new (2026-11-25) | `centres` once, then job per participant, 16 cores | about 1 hour |

The "ported from" scripts of B4 to B10 are in `4_decoding_claude_retrieval/`.
`decoding_utils.py` holds the shared helpers (copy of `4_decoding_claude/utils.py`).

Running stage B:

    cd paper_pipeline/B_decoding
    bash submit_cluster_jobs.sh B1            # all participants; wait until the jobs are done
    bash submit_cluster_jobs.sh B2
    python B3_combine_localizer_masks.py
    bash submit_cluster_jobs.sh B4
    python B5_classifier_accuracy.py
    bash submit_cluster_jobs.sh B6a  &&  bash submit_cluster_jobs.sh B7
    python B6b_encoding_decoding_group.py
    bash submit_cluster_jobs.sh B8
    bash submit_cluster_jobs.sh B9
    python B10_trial_level_master_table.py

The launcher uses jobber, like the original launchers: each Slurm job gets a
task list (for B4, B6a, B7 and B8 one task per ROI) and runs `N_PARALLEL`
tasks at a time (default 2). Tested with B6a for sub-14 (12 ROI tasks, no errors).

`DRY_RUN=1 bash submit_cluster_jobs.sh B4` writes the task lists and job
scripts without submitting.
Participants can be named: `bash submit_cluster_jobs.sh B1 sub-13 sub-14`.
Task lists, job scripts and logs go to `<project>/logs/paper_pipeline/<step>/`.

The decoding steps read the BOLD data from the shared derivatives folder for
every ROI again, so they are limited by disk speed, not CPU. With about 100
jobs reading at once, B6a took 1.5 hours for one participant instead of
minutes. Submitting fewer participants at a time is faster overall.

Verification against the existing results (2026-11-18). Each step was run
with the existing results of the step before it as input, so a difference
can only come from the step itself.

- B1: sub-13, all four sessions: all 12,900 masks are byte-identical to the
  existing ones (and the dropout table, checked for ses-06).
- B2: sub-13: all 200 reliability masks (20 regions, 5 thresholds, 2
  localizers) are identical to the existing ones, and the raw localizer logs
  were not touched. The combined correlation maps are identical too. For
  localizer 2 the two diagnostic maps (scenes only, objects only) are swapped
  relative to the existing files, which date from an older script version
  (3 February 2026); the masks use only the combined map, so nothing
  downstream is affected. Runtime: 1 hour, 21 GB memory.
- B3: run on all existing reliability masks. 29 of 36 participants: all masks
  identical. 7 participants differ because the existing combined masks are
  out of date (see "Open points").
- B4: sub-13, ROI vis: the three saved classifiers have identical weights and
  selected voxels. The cross-validation tables of the scene and the object
  classifier are identical. The cross-validation of the combined 20-class
  classifier differs between runs (accuracy 0.189 against 0.187), see "Open
  points". No later step uses that table.
- B5: run on the existing cross-validation tables: same 52 output files and
  the same group mean accuracies. Changed afterwards (2026-11-18): the mean is
  no longer part of the file names, and the accuracy tables and the plots by
  trial type were added; tested on the union masks of the exploration run.
- B6a: sub-13 and sub-14, ROI vis: time courses identical.
  Changed afterwards (2026-11-19): scenes restricted to the 60 scene onsets;
  sub-14, vis: the 60 time courses are a subset of the original 180, the
  object time courses are identical.
- B6b: run on the existing B6a data: all 24 group time-course files identical.
  Changed afterwards (2026-11-19): peak search restricted to 2 to 6 TRs after
  onset and `encoding_decoding_summary.csv` added; the time-course files are
  unaffected.
- B7: sub-14, ROI vis, both sessions: all time courses identical. In the
  trial table, 15 of 16 columns are identical; the reaction-time column
  differs by rounding only (at most 9e-16 s), because stage A wrote the
  behaviour files anew.
- B8: ROI vis, evening session: result table identical (23,800 rows, 91 columns).
  After adding the `*_wo_target` columns (2026-11-19), parietal_precuneus, both
  sessions: all original columns of B8 and all original amplitudes of B9
  (preset thesis) are identical to the existing results; C6 on the existing
  results reproduces all its earlier rows exactly.
- B9: ROI vis, evening session, all three presets: result tables identical.
- B10: master table byte-identical (34,300 rows).

Differences from the original scripts (file handling only):

- B2 no longer writes two derived columns back into the raw localizer logs.
- B4 saves the `_with_training_columns.csv` files to its own output folder
  instead of `behavior/`.
- B4 was restructured after the review (2026-11-18): each region is masked,
  smoothed and cleaned once instead of six times (same result, about a sixth
  of the reading); the combined 20-class classifier, which no later step
  uses, is no longer trained; the SMOTE oversampling has a fixed seed; the
  accuracy summary is written per participant with a classifier column.
  The scene and object classifiers are computed as before: for sub-13, vis,
  the restructured script gives byte-identical accuracy tables and identical
  classifier weights (20 minutes on a busy disk).
- B3 always rebuilds its masks instead of skipping existing files.
- B6b no longer copies its plots into `1_thesis_chapter_plots/`.
- B3 and B8 match the ROI name against the file name instead of the full
  path, so the result cannot depend on where the pipeline is stored.
- B10: a switched-off branch for FFT band power was removed (that step was
  archived in September 2026).
- Output folders: B7 and B8 write to `<step>/presleep` and `<step>/postsleep`;
  B9 writes to `<step>/thesis`, `<step>/trunc`, `<step>/trunc_tapered`.
- Plots are saved to `plots/<script name>/` instead of next to the tables
  (added after the verification above; checked by running every plotting
  step except B2 and B4 again).

Participants with incomplete retrieval data (checked 2026-11-19):
- sub-13, evening: now processed. Its encoding events are in the `step0`
  behaviour folder and its retrieval trials in `step6`; B6a and B7 take the
  encoding from the highest step that contains encoding events
  (`decoding_utils.find_encoding_folder`). The original took the highest step
  and stopped with "Could not build valid sequence_dict". For every other
  participant the chosen folder is unchanged. Test, vis: 70 trials, target
  minus other classes 3.5 percentage points.
- sub-13, morning: not usable. Only 4 of the required 6 long fMRI runs.
- sub-27, morning: not usable. Only 3 long fMRI runs.

## Stage C: statistics and figures (ported, verified)

Scripts are in `C_statistics/`. All run directly and need only stage B outputs.

| Step | Script | Does | Ported from |
| --- | --- | --- | --- |
| C8 | `C8_overnight_change.py` | What changes from the evening to the morning and where: retrieved item, predecessor, successor per time point; which items carry the slope; order test against all orderings of the same probabilities; sinusoid with and without the retrieved item; comparison between ROIs | new (2026-11-19) |
| C9 | `C9_item_time_courses.py` | Classifier evidence for the retrieved item and the items up to four positions before and after it, each against its matched control (the same position of the other sequence); time courses from cue onset, by time point and by retrieved position | new (2026-11-25) |
| C10 | `C10_slope_time_courses.py` | Signed and absolute slope over the trial, by retrieved position; the single-trial example of slope and sinusoid fit | new (2026-11-25) |
| C11 | `C11_slope_amplitude_by_position.py <sub>` / `group` | Sinusoid amplitude and absolute slope, retrieved minus control, with items left out; by time point and region; which items carry the effect; relation to reported retrieval and search time | new (2026-11-25) |
| C12 | `C12_searchlight_group.py` | Group maps of B11 per time point with whole-brain correction by sign-flipping; clusters; comparison with the regions (`C12_TAG=<space>_r<radius>`) | new (2026-11-25) |
| C13 | `C13_paper_figures_decoding.py [1\|2\|3]` | The three decoding figures of the paper (classifiers; retrieved item and neighbours; sequentiality at the three time points) from the tables of B5, B6b, C9 and C11, in the shared look of `figure_style.py`; writes every test shown to `figure_statistics.csv` | new (2026-11-25) |
| C14 | `C14_matched_search_time.py` | The comparison of the three time points for absolute slope, sinusoid amplitude and item evidence on object trials matched for search time (same number of trials per 0.25 s bin at every time point, 200 draws), on trials matched over two time points only, and with search time as a covariate; tables `trial_table.csv`, `matching.csv`, `group_contrasts.csv`; figure `matched_search_time` (paper: FIGURE_D4) | new (2026-11-25) |

B10 and C1 to C7 were moved to `_old_scripts/` on 2026-11-25 (see the README there); the detailed notes on them further down describe those earlier steps.

Verification (2026-11-18), each script run on the existing stage B results:

- C1: all 20 output files (results text and tables, full and matched) are
  byte-identical to the existing ones.
- C2: all 7 tables byte-identical; the same 8 figure files are produced.
- C3: all 3 tables byte-identical; the same 5 figure files are produced.
- C4a, C4b: the same panel files are produced (40 + 40 and 129). They are
  drawn from the byte-identical master table; the PDF contents themselves
  were not compared.

C5 is an addition, not a port (2026-11-19). It asks whether the item before
the retrieved one is reactivated, as forward stepping through the sequence
would predict. C2 (analysis 6) averages all preceding items and is null; C5
separates the items by their lag to the retrieved item. Evidence is P(item of
the retrieved sequence) minus P(item at the same position of the other
sequence), TRs 3 to 8, as in C2.

- Verification: run on the existing stage B results (`SLEEPLAY_OUTPUT` pointing
  at links to `4.6_SODA_based_on_singlesubj_4.4_*`). The lag profile (1,152
  rows) and the predecessor / successor table (512 rows) are identical to the
  exploratory script's tables.
- Result on the existing data, object trials at positions 2 to 4, sessions
  pooled, n = 34: the immediate predecessor is reactivated in vis (t = 5.2) and
  parietal_precuneus (t = 4.5), precuneus t = 2.9; the successor is not (0.8,
  0.4, -0.1); predecessor minus successor t = 3.1 and 2.7 (uncorrected
  p = .004 and .012; FDR across 12 ROIs p = .053 and .070; evening alone
  p = .045 in vis). Lags of two or more are at zero on both sides.
- On correct trials only the predecessor remains (vis t = 2.4,
  parietal_precuneus t = 3.0) but the asymmetry does not (t = 0.8 and 1.5).
- Session: the asymmetry is significant in the evening (vis t = 2.8) and not
  in morning pass 1 (t = 0.0), but the direct test of the change is not
  significant in any ROI (morning pass 1 minus evening: vis t = -1.5,
  parietal_precuneus t = -1.7; no FDR p below .13). The predecessor itself is
  present in the morning as well (vis: morning t = 3.8, pass 2 t = 4.1). The
  data do not show a change from evening to morning.
- Scene trials show no predecessor effect.
- Main ROIs (decided 2026-11-19): vis, parietal_precuneus, mtl, and caudal_acc
  as a control region (classifier at chance, about 10.5 percent, so no effect
  is expected). C5 writes `p_fdr_main` (FDR across these four) next to
  `p_fdr_rois` (all twelve) and draws its figures for these four. With the
  four-ROI correction, object trials, positions 2 to 4, sessions pooled:
  predecessor vis p < .001, parietal_precuneus p < .001, mtl p = .11;
  successor mtl p = .014, others p = .69; predecessor minus successor vis
  p = .018, parietal_precuneus p = .023. caudal_acc: all |t| < 0.7. Correct
  trials only: predecessor vis p = .049, parietal_precuneus p = .022;
  successor mtl p = .023; asymmetry p = .44. Session changes: all p > .19.
- Successor (analyses 5 and 6, added the same day; 2,160 rows identical to the
  exploratory table): MTL is the only one of 12 ROIs with evidence for the
  upcoming item (positions 2 to 4: t = 3.1; FDR across ROIs p = .02 to .04).
  Successor minus predecessor is larger in MTL than in vis (t = 3.1, p = .003),
  parietal_precuneus (t = 2.5) and precuneus (t = 2.2). MTL also carries the
  predecessor (t = 3.0) and its own successor-minus-predecessor difference is
  not significant (t = 1.0). Only one step ahead is seen. Preliminary: the MTL
  classifier is close to chance (12 to 13 percent) and the effects are small
  (0.4 to 0.5 percentage points); on correct trials the region difference
  weakens (t = 1.6 to 1.7).
- Until the full pipeline runs, the C5 tables and plots computed on the
  existing stage B results are in
  `7_rsa_exploratory/outputs/C5_on_existing_results/`.

C6 is an addition, not a port (2026-11-19). It returns to the original
comparison of the retrieved sequence with the control sequence and reports it
per session in the format of C5 (four main ROIs, `p_fdr_main`). Run on the
existing stage B results; tables and plots are in
`7_rsa_exploratory/outputs/C6_on_existing_results/`. Object trials, n = 34:

- Sequence evidence (mean probability of the five elements, retrieved minus
  control, all TRs): vis t = 3.3, parietal_precuneus 2.3, mtl 3.1, caudal_acc
  -0.3. No session change in any ROI (morning pass 1 minus evening: all
  p_fdr_main >= .26).
- The one significant session change is the absolute five-item slope in
  parietal_precuneus (morning pass 1 minus evening: t = 2.7, p_fdr_main = .040;
  TRs 3 to 8: t = 4.0, p = .002). It comes mainly from the CONTROL sequence,
  whose absolute slope drops (t = -3.2; TRs 3 to 8: -4.4), while the retrieved
  sequence changes little (t = 0.9 and 1.4).
- Restricted to retrieved positions 3 and 4 that change is not significant
  (t = 1.2), and the slope up to the retrieved item decreases (pass 1 minus
  evening t = -2.6, morning minus evening t = -3.3).
- The control region is flat on sequence evidence, but its slope up to the
  retrieved item reaches t = 2.7 in morning pass 2: single slope values of that
  size occur without signal.
- Fitted sinusoid amplitude (added the same day; B9 presets `thesis` and
  `trunc_tapered`, one amplitude per trial). Five items, parietal_precuneus:
  evening t = -0.2, morning pass 1 t = 3.8 (p_fdr_main = .003), pass 2 t = 0.1;
  pass 1 minus evening t = 2.6 (p_fdr_main = .056), pass 2 minus pass 1
  t = -2.8 (.038). As for the slope, the evening-to-morning change comes mainly
  from the control sequence (t = -3.1; retrieved t = 1.1), and it is absent on
  retrieved positions 3 and 4 (t = 1.1). Up to the retrieved item:
  parietal_precuneus evening t = 3.0, morning t = 2.9, no increase (morning
  minus evening t = -1.5; on positions 3 and 4 t = -2.7, p_fdr_main = .042).
  In the control region the amplitude up to the retrieved item is t = 1.9
  pooled (2.6 on correct trials), so this measure is not flat where no signal
  is expected.

C7 is an addition, not a port (2026-11-19). Run on the existing stage B
results; tables and the scatter figure are in
`7_rsa_exploratory/outputs/C7_on_existing_results/`. Object trials, n = 34,
sessions pooled:

- Between participants, related to accuracy in vis and parietal_precuneus:
  the retrieved item (r = .51 and .43; Spearman .65 and .42), the slope over
  five items (.39 and .57; Spearman .39 and .41), the slope up to the
  retrieved item (.34 and .42) and the sinusoid amplitude up to the retrieved
  item in parietal_precuneus (.51). Surviving FDR across the four ROIs:
  retrieved item (both), slope over five items (both), slope and sinusoid up
  to the retrieved item (parietal_precuneus).
- Not related to accuracy in any ROI: predecessor, successor and their
  difference (all |r| <= .20). Nothing in mtl or caudal_acc.
- Without the participant with the lowest accuracy (s41): retrieved item vis
  .51, parietal_precuneus .37; slope over five items .37 and .38 (from .57).
- Within participants (22 with at least 5 correct and 5 incorrect trials):
  only vis separates correct from incorrect trials, for the retrieved item
  (t = 2.6), the successor (2.7) and sequence evidence (2.1). The slope and
  sinusoid measures do not differ between correct and incorrect trials in any
  ROI (all |t| <= 1.6).
- Change against change: the evening-to-morning change of no measure relates
  to the change in accuracy (all |r| <= .40, none significant in vis or
  parietal_precuneus).

C8 is an addition, not a port (2026-11-19). Run on the existing stage B
results (probabilities) and on the sinusoid tables with the `wo_*` columns for
the four main ROIs (`7_rsa_exploratory/outputs/19_without_target_sinusoid/`);
tables and plots are in `7_rsa_exploratory/outputs/C8_on_existing_results/`.
Analyses 1, 2 and 4 reproduce the exploratory numbers. Object trials, n = 32
to 34, uncorrected unless stated:

- Successor (item after the retrieved one): parietal_precuneus evening
  t = -0.6, morning pass 1 t = 2.3, pass 2 t = 0.5; change pass 1 minus evening
  t = 2.1, p = .043 (not significant after correction across the four ROIs).
  No change in vis (0.1), mtl (0.1) or caudal_acc (-0.2).
- Sinusoid amplitude, change pass 1 minus evening in parietal_precuneus: five
  items t = 2.6 (p = .014, p_fdr_main = .056), without the retrieved item
  t = 2.3 (p = .030, p_fdr_main = .12). Against vis p = .045, against mtl
  p = .047, against caudal_acc p = .08 (without the retrieved item).
- The slope effect survives leaving out the predecessor (t = 2.6) and is gone
  when both neighbours are left out (t = 0.6).
- Order test (new): in no ROI and at no time point is the absolute slope of
  the retrieved sequence larger than the mean absolute slope over all
  orderings of the same probabilities (parietal_precuneus, without the
  retrieved item: t = -1.1, 0.6, -1.5). The slope and amplitude measures
  therefore show that some items of the retrieved sequence stand out, not
  that the items are arranged along the sequence. In text, "neighbour
  reactivation" describes the finding better than "sequential reactivation".
- All changes are confined to morning pass 1 and reverse in pass 2.

## Notes for the methods text

Collected during the step-by-step review (2026-11-18). No code was changed for these.

- B1: the signal threshold (T = 0.3) is relative to the mean of the whole
  image, including voxels outside the brain. It removes clear dropout only
  (most regions keep 99 to 100 percent of their voxels).
- B1: the vis region includes inferior parietal cortex (the whole parietal
  ROI), so vis and parietal are nested, not independent. Kept as is by
  decision. Until 2026-11-19 vis also included parahippocampal cortex (part of
  the MTL ROI); that was removed, see "Decisions". All results computed before
  the full run use the old vis mask with parahippocampal cortex.
- B2: the "70" masks are in effect "voxels with a positive split-half
  correlation": fewer than 70 percent of voxels have r > 0 in every region,
  so the zero floor sets the cut-off (r at the cut-off: 0.00 to 0.02).
- B2: in MTL the split-half correlation is at noise level (mean r = 0.02,
  47 percent of voxels positive), so the selection there is close to random.
- B2: the GLM uses the Glover HRF and a 0.01 Hz cosine drift model (the
  SPM HRF and 1/128 Hz written in the script are ignored by nilearn). The
  correlation is across 10 category betas per voxel; retrieval trials in the
  same runs are not modelled.
- B4: training samples are the mean of 4 TRs around stimulus onset + 5 s.
  Half of them are retrieval trials (word cue, image recalled), which are
  barely decodable in cross-validation (vis: about 47 percent on encoding
  trials, 15 percent on retrieval trials; chance 10). Report both figures,
  not only the average.
- B4: the fixed settings (4 mm smoothing, 4-TR average, top 30 percent of
  voxels) were chosen by trying several setups and taking the one with the
  best cross-validated accuracy on these same localizer data (confirmed
  2026-11-18). The reported cross-validated accuracy is therefore the
  maximum over the setups tried and is optimistic. State this, and state how
  many setups were tried. The retrieval sessions played no part in the
  choice, so the reactivation results are not affected.
- B4, B5: the classifiers are trained on trials with the image shown and on
  trials with the image recalled, because this transferred better to retrieval
  in the main task (decision of the authors). Report cross-validated accuracy
  separately for the two trial types (`outputs/B5_classifier_accuracy/accuracy_summary.csv`):
  in vis about 45 percent for shown and 15 percent for recalled images, chance 10.
- B6a, B6b: the stored time courses are relative to the onset plus 5 s
  (3.33 TRs). The peak is searched 2 to 6 TRs after the real onset.
  Of the 180 events labelled `dn+sc_cue`, only 60 are scene onsets (30 in
  run 1, 30 at the start of an object block). In the other 120 the scene is
  already on screen between two objects. B6a uses only the 60 scene onsets
  (decided 2026-11-19); the original used all 180, which raised the scene
  probability before 'onset' (vis: 15 percent). For the onsets the probability
  starts at chance and peaks 3.3 TRs after onset (vis: 24 to 27 percent).
- B7 onwards: the column `behav_correct` is NOT answer accuracy. It is the
  participant's button press at the question mark, indicating that they have
  retrieved the item (True, 85 percent of trials) or no press (False, 15
  percent, no response time). The verbal answers were recorded separately and
  are not part of this pipeline. Call it "reported retrieval" in text and
  figures, not "accuracy" or "correct" (stated by the author, 2026-11-19).
- B8, B9, B10, C5: the retrieved item has a raised classifier probability and
  is part of the retrieved sequence, not of the control sequence. This alone
  can make the slope of the retrieved sequence larger. B8 therefore also
  computes the slope over the other four items (`*_wo_target`; the control
  sequence drops the same position), B9 (preset thesis) fits the sinusoid to
  it (`wo_*_w_amplitude`), and C6 reports both next to the five-item versions
  (measures `|slope| without target` and `sinusoid without target`), per
  morning half and per sequence. The five-item analysis stays the main
  analysis; these are its control. On the existing data,
  parietal_precuneus: amplitude, morning +0.30 points with all items
  (p < .001) and +0.31 with the item left out (p = .008); evening no effect;
  morning minus evening p = .016 and p = .021. In vis the absolute-slope
  effect of the evening disappears when the item is left out.
- B2, B4, B5: voxels are selected on the same localizer runs that train and
  cross-validate the classifiers, so the cross-validated accuracies are
  slightly optimistic. The retrieval sessions are independent data.

## Decisions

- **s41, morning session: retrieved items taken from the cue (2026-11-25, decided with the user).**
  *What happened.* s41 learned on the evening of 2025-07-09 and was scanned again
  on the morning of 2025-07-10 (dates of the raw log files: evening 22:11;
  morning 08:28 for `nses_1` and 08:59 for `nses_2`). That morning the task was
  started with participant number 42 instead of 41 (`Data_overview.xlsx`: "wrong
  number entered during recording - renamed to 41"). The task builds its list of
  questions from the participant number, so s41 was asked participant 42's list.
  s42 is a different person (evening 2025-07-10, morning 2025-07-11). The two
  morning logs are in s41's folder under s41's name and were always there; no
  file had to be moved.
  *How it was found.* In s41's morning logs only 6 of 100 object targets are
  objects s41 learned at the cued scene and position; all 100 match what s42
  learned. The trial order is identical to s42's own morning session, the logs
  store number 42 (`sub_idx` 41, zero-based), and the responses differ from
  s42's, so it is a separate recording of a different person.
  *Why the session can be used.* A question shows a scene and a position. All
  participants learned the same ten scenes, so the question is valid for s41,
  who answers with the object s41 learned there; only the "correct answer"
  written in the log belongs to s42. The items asked are the same 50 (scene,
  position) questions per pass as in any session, in s42's order. The retrieved
  item of a trial is therefore taken as the object s41 learned at the cued
  position of the cued scene (scene trials: the scene s41 learned at the cued
  position of the cued context). This rule reproduces the logged target exactly
  in every other session of every participant (78 sessions). After the
  correction all 50 of s41's objects are asked once per pass. As a check on
  s41's own data: with these targets the classifier evidence for the retrieved
  item in visual cortex is +2.3 points (SE 1.7, 100 trials); with the targets
  as logged it is -3.7 (SE 1.6).
  *How it is implemented.* The session is listed in
  `config.RELABELLED_SESSIONS`. `decoding_utils.relabel_targets_from_cue` is
  applied by A1 when it writes the behaviour tables (both morning logs); the
  targets as logged are kept in `obj_ret_targ_logged` and `sc_ret_targ_logged`.
  `load_main_task_behavior` (B7) applies it again on loading, which changes
  nothing in a table A1 already corrected. The raw logs in `behavior/` are not
  changed. A copy of s41's A1 tables from before the change is in
  `logs/paper_pipeline/dec_review/A1_s41_before_relabel_2026-11-25/`.
  *Consequences.* Before this date the session was dropped (by the sequence
  validation in B8 and the hierarchy check in D1), so every result computed
  before 2026-11-25 is without it. It enters on the next run of B7 (decoding) or
  of D1 / the E steps (similarity, univariate) for s41; morning time points then
  have up to 33 participants in the decoding analyses. C9 already includes it.
  *What is not repaired.* The spoken answers of that session would have to be
  scored against s41's own items; nothing in the pipeline uses them. The
  post-task files of the two participants had been filed in each other's folder
  and were moved and renamed on 2026-11-25 after checking dates and contents
  (the typed scene answers match the other participant's learned scenes): the
  file of 2025-07-10 is now `s41/behav/post/output_post_task_sub41.csv`, the file
  of 2025-07-11 is now `s42/behav/post/output_post_task_sub42.csv`. Contents are
  unchanged; the questions and the `target` column in both still come from the
  number the task was started with (42 and 72). Details in the notes in
  `behavior/s41/` and `behavior/s42/`. The pipeline does not use these files.

- **B1: parahippocampal cortex removed from the vis mask** (2026-11-19).
  Reason: an exploration of the MTL classifier on the localizer data
  (`7_rsa_exploratory/14_mtl_decoder/`, 75 variants, 35 participants) showed
  that the decodable category signal of the MTL mask comes from its
  parahippocampal part (parahippocampal alone: objects 13.6 percent, as the
  whole MTL; hippocampus alone 10.9, entorhinal 10.5; chance 10). With that
  part in both masks, vis and mtl were not separate regions. Checked for
  sub-14, ses-02: the new vis mask has no voxel in common with the
  parahippocampal or the MTL mask and differs from the old one only by those
  voxels. Takes effect with the full run (B1 to B9 for vis); existing results
  are unchanged.
- **B4: classifiers stay trained on shown and recalled images together; the
  MTL classifier gets stronger regularisation** (2026-11-19). `CLASSIFIER_C`
  in B4 sets C = 0.001 for mtl and leaves 1.0 for all other ROIs. In the same
  exploration this raised cross-validated accuracy in MTL for objects from
  13.5 to 14.7 percent (shown images 15.0 to 17.1; recalled images 11.9 to
  12.3, not significant; scenes 12.4 to 12.8, not significant). No setting
  improved decoding of recalled images reliably, and training on shown images
  only lowered it. Not tested in other ROIs. Stronger regularisation gives
  flatter class probabilities, so probability differences in mtl will be on a
  smaller scale than in the other ROIs: compare mtl with other ROIs by test
  statistics, not by percentage points. Takes effect with the full run.
- **MTL means parahippocampal cortex in practice.** No classifier decodes
  categories from the hippocampus or entorhinal cortex. Results in the mtl ROI
  should not be described as hippocampal.

- **B4** (2026-11-18): restructure to preprocess once per region, drop the
  combined 20-class classifier, fix the SMOTE seed. Done.
- **B3** (2026-11-18): rebuild the seven out-of-date masks and always
  rebuild. Done in the script; takes effect with the full run.

- **B3 keeps the union of the two localizer masks** (decided 2026-11-18),
  unless the full union-versus-intersection exploration changes the picture
  drastically. Interim result with about two thirds of the runs done: the
  two versions differ by about one percentage point or less in
  cross-validated accuracy and show no consistent difference in reactivation
  of the retrieved item. The exploration is in
  `Sleeplay_Hummel/_old/exploration/tmp_explore_union_vs_intersection/` (temporary; results
  in its `summary/` folder) and is not part of the pipeline.
- B3 note for the methods text: the masks are a union (a voxel is kept if it
  is reliable in at least one localizer), although the function and the old
  folder name say "intersect".

## Open points

1. **Out-of-date combined masks for 7 participants (decided: rebuild).** For
   sub-32, 36, 40, 41, 42, 44 and 47 the existing combined masks
   (`3_reliability_voxel_selection/combine_loc1_and_loc3_masks/`) date from
   24 February 2026, but their reliability masks were recomputed on 2 March
   2026; the old combine script skipped existing files. All existing results
   use the old masks for these participants (overlap with rebuilt masks:
   Dice 0.74 to 0.99, lowest for sub-44). B3 now always rebuilds, so a full
   run of this pipeline gives all 36 participants consistent masks, and the
   results for these seven will differ slightly from the existing numbers.
2. **Full run.** Not started. It recomputes everything for all participants
   (B2 alone takes about 1 hour per participant).

## Stage D: pattern similarity (added 2026-11-19, verified)

Scripts are in `D_similarity/`. Retrieval trials only; encoding is never paired
with retrieval. One estimator throughout: per run, ONE model with a regressor
per retrieval trial (plus drift, 20 confounds and catch trials), on data
smoothed with 4 mm and expressed as percent signal change. One model per trial
gave the same RSA results (group t values correlate .98) and was dropped. Not a
port of the original pipeline; ported from `7_rsa_exploratory/` (the earlier
`5_rsa_claude/` was removed, its code is in `_old_code/5_rsa_claude/`).

| Step | Script | Does | Ported from |
| --- | --- | --- | --- |
| D1 | `D1_trial_patterns.py <sub>` | Trial table, brain mask, one estimate per retrieval trial for two event models (fixed 3 s; cue to response). Saves per event model the ROI patterns (reliability masks of B3) and the mean per trial in anatomical regions (B1 masks, T-0). No whole-brain files are kept | `04_single_trial_betas.py` (method `lsa`), mask part of `15_univariate_session.py` |
| D2 | `D2_binary_rsa.py [n_jobs]` | Binary models (context, scene, position, category; fitted alone and together) on trial-pair dissimilarity; object x object, object x scene, scene x scene pairs; evening, morning pass 1, pass 2 and the three between-session sets; group tests, session contrasts, session mixed model. Variant `four_rt` (added 2026-11-23) adds the response-time difference of the two trials as covariate | `02_binary_rsa.py` |
| D3 | `D3_position_and_neighbours.py [n_jobs]` | Graded position distance, scene order, neighbour advantage within a scene against other scenes, probe-order check. Parts E, F and G (added 2026-11-23): hierarchy distance without shared picture or digit; position distance across sequences (different positions, different scenes or contexts; objects and scenes separately), and the position model with response-time controls (trial-level difference, mean response time per position), on answered trials only, and for object x scene pairs | `05_position_distance.py`, `06_followups.py` (part A), `07_probe_order_check.py` |
| D4 | `D4_figures.py` | Six figures (models and data, four models, sessions, scene cue control, position distance, within-scene neighbours) | `08_paper_figures.py` (figures 1-5, 7) |

How to run (D1 needs A1, A2, B1, B3 and the fMRIPrep derivatives; about 20
minutes and 8 cores per participant; D2 to D4 take a few minutes together and
should run as a job):

    cd B_decoding && bash submit_cluster_jobs.sh D1 [sub-XX ...]
    cd ../D_similarity
    python D2_binary_rsa.py 16
    python D3_position_and_neighbours.py 16
    python D4_figures.py

D2 and D3 use the fixed 3 s events (`EVENTS` in `D2_binary_rsa.py`).

Verification (2026-11-19), run on the existing masks and behaviour tables
(`SLEEPLAY_OUTPUT` pointing at links; `7_rsa_exploratory/run_pipeline_DE_on_existing_results.sh`),
36 participants in the evening, 34 in the morning; results copied to
`7_rsa_exploratory/outputs/D_E_on_existing_results/`:

- D1: ROI patterns of five participants (124 files) are identical to the
  exploratory `patterns_lsa` (largest difference 0 at float16 precision); trial
  tables identical except response time and onset to 1e-13. The patterns for
  cue-to-response events had no stored counterpart; their region means were
  checked through E1.
- D2: all seven tables have the same rows as `outputs/02_binary_rsa/lsa`
  (51,180 participant values identical; group t values differ by at most 4e-16).
- D3: all ten tables identical to `05_position_distance/lsa`,
  `06_followups/lsa` and `07_probe_order_check` (largest difference 0).
- D4: six figures produced; drawn from the identical tables, not compared
  pixel by pixel.
- 2026-11-23: D2 gained the variant `four_rt` and D3 part E (response-time
  controls, tables `E_*`), with a minimum of 60 pairs for analyses restricted
  to answered trials. Rerun on the same data: all 17 earlier tables are
  unchanged (rows of `main` and `four` in D2; tables A, C and D in D3).

Results on the existing data (t across participants, sessions pooled, n = 35
unless stated):

1. **Graded position code.** Objects at neighbouring positions are more
   similar than distant ones, among pairs that do not share the position
   digit: distance effect in visual (t = 5.4), parietal (5.0), cortical memory
   (5.8), precuneus (4.1), superior frontal (4.1), retrosplenial (2.5); MTL 1.9.
   The binary same-position effect is present next to it in visual (4.1) and
   parietal cortex (3.7).
2. **Scene effect explained by the cue.** Same-scene pairs are more similar
   between two object trials, where the scene picture is on screen in both
   (morning pass 1: visual 4.4, parietal 3.5, cortical memory 3.4), and not
   between an object trial and the scene trial that retrieves the scene
   (-0.3, -0.5, -0.2; no ROI above 1.6 in any session).
3. **Neighbours within a scene.** The neighbour advantage is larger within a
   scene than across scenes in the evening (parietal t = 2.6, retrosplenial
   2.7, cortical memory 2.2, superior frontal 2.3) and not in the morning
   except retrosplenial cortex (2.3); morning minus evening -1.0 to -1.9, not
   significant. Probe order does not explain it (neighbours and distant
   objects are probed equally far apart, t = -0.1, 1.0, 0.8).
   **Caveat: this result depends on the estimator.** With one model per run it
   is weaker than with one model per trial (retrosplenial t = 2.7 against 3.8;
   best FDR p across ROIs .08 against .006), and it appears in retrosplenial
   cortex in the morning too. It does not survive correction here.
4. **No context effect, no scene-order effect, no reliable session change.**
   Context: no ROI consistent across sessions (pooled t between -1.2 and 0.4).
   Scene order: |t| below 1.8 in the ROIs above. Session mixed model: no
   model has an FDR-corrected session effect (smallest p = .11).

5. **Controls for the position code (D3 part E, added 2026-11-23).** Response
   time differs by position (1.3, 1.7, 2.2, 2.2, 1.9 s for positions 1 to 5)
   and 11 to 23 % of trials have no response. Graded distance effect, t across
   34-35 participants, sessions pooled:

   | Model | Visual | Parietal + precuneus | Parietal | Cortical memory | Superior frontal | Retrosplenial |
   | --- | --- | --- | --- | --- | --- | --- |
   | As in result 1 | 5.4 | 5.5 | 5.0 | 5.8 | 4.1 | 2.5 |
   | Answered trials only | 3.0 | 4.4 | 4.2 | 3.6 | 2.5 | 2.3 |
   | + response-time difference of the two trials | 2.9 | 4.3 | 4.1 | 3.4 | 2.4 | 1.9 |
   | + mean response time per position (all trials) | 4.4 | 4.2 | 3.7 | 4.9 | 3.8 | 1.8 |
   | + both covariates | 2.4 | 3.3 | 3.1 | 3.4 | 2.9 | 1.4 |

   With both covariates the effect survives FDR across regions in visual,
   parietal, cortical memory and superior frontal ROIs (p = .014 to .045), not
   in retrosplenial cortex. Trials with similar response times have more
   similar patterns in every region including the control region (t = 4.8 to
   9.1), and the profile of mean response times across positions has its own
   effect (visual 4.0, parietal + precuneus 3.0). **Report the position code
   with these controls** (t about 2.4 to 3.4, not 5 to 6).
6. **The binary same-position effect is a digit effect.** Object and scene
   trials both show a digit, with different meanings. Object x scene pairs with
   the same digit are as similar as object x object pairs (visual t = 4.5,
   parietal + precuneus 4.7, caudal ACC 3.2), and with the response-time
   controls the same-position effect between object trials is gone (|t| <= 1.5).
   A linear distance term shows no effect for object x scene pairs (0.4 to
   1.8) except in retrosplenial cortex (2.7). A neighbour contrast (distant
   minus adjacent, other context, check of 2026-11-23, not a pipeline table)
   does (2.0 to 3.4, also in caudal ACC), so part of the neighbour effect is
   general to the digit or to ordinal position at either level.
8. **Position distance across sequences (D3 part F, added 2026-11-23).** A
   model matrix per trial type with the distance between positions (1 to 4
   steps), defined only for pairs at different positions (no shared digit)
   that do not belong to the same sequence (no shared scene picture or context
   cue). t across 34-35 participants, sessions pooled:

   | Pairs | Visual | Parietal + precuneus | Parietal | Cortical memory | Superior frontal | Retrosplenial |
   | --- | --- | --- | --- | --- | --- | --- |
   | Objects, other scenes | 6.1 | 4.7 | 4.6 | 5.4 | 3.7 | 1.7 |
   | same, with response-time difference | 2.6 | 3.3 | 3.4 | 3.0 | 2.1 | 0.7 |
   | Objects, other context | 3.7 | 2.8 | 3.3 | 3.3 | 3.0 | 1.2 |
   | same, with response-time difference | 2.9 | 1.4 | 2.9 | 2.0 | 1.2 | -1.0 |
   | Scenes, other context | 0.3 | 1.0 | 0.9 | 0.2 | -0.4 | 0.4 |
   | Scenes, same context | 1.9 | 2.3 | 1.9 | 1.8 | 0.5 | 0.9 |
   | Object x scene, other context | 1.1 | 0.9 | 0.7 | 2.0 | 1.3 | 2.5 |

   For objects the effect holds across scenes (FDR p <= .002; with the
   response-time covariate .015 to .08) and is weaker across contexts. For
   scenes there is none across contexts (40 to 55 pairs per time point, so low
   power). The shape is a step, not a line: adjacent positions are similar
   (z = -0.04), two, three and four steps apart are alike (0.02 to 0.03).
   Neighbouring positions still show neighbouring digits; no pair of this
   task removes that.
   **Gradient or step.** On the same pairs, a gradient model (1 to 4 steps)
   and a step model (adjacent against all others), one parameter each, are
   compared by their partial correlation with the dissimilarity
   (`F_group_model_comparison.csv`). Objects from other scenes: the step fits
   better in visual (paired t = 2.1), parietal + precuneus (2.7), cortical
   memory (3.2) and superior frontal ROIs (2.7); mean r .025 to .030 against
   .017 to .023. With the step in the model, distance among non-adjacent pairs
   adds nothing (t = -0.7 to -1.1). With the response-time covariate the
   advantage of the step is a trend (parietal + precuneus t = 1.8). The
   position effect is a neighbour effect, not a graded distance.
9. **Hierarchy distance without a shared picture or digit (D3 part G, added
   2026-11-23).** Tree distance between two trials: 1 = an object trial and
   the scene trial that retrieves its scene, 2 = same context, 3 = other
   context; pairs of the same scene picture or the same digit and all scene x
   scene pairs are left out. No effect at any level in any region: the full
   model t = -1.4 to 0.1; object trial against its own scene trial compared
   with another scene of the same context (the step the cue does not explain,
   about 58 pairs per time point) t = -0.7 to 1.4; context step between object
   trials -0.5 to 0.3, between object and scene trials -2.0 to 0.0 (smallest
   FDR p = .54). The context is shown on every trial, so the context step is
   not free of the cue, and it still shows nothing. Pattern similarity gives
   no evidence for the context > scene > object hierarchy.
7. **Binary models with the response-time covariate (D2 variant `four_rt`).**
   On answered pairs with the covariate, scene and position shrink (visual
   scene z = 1.7 against 4.1, position 2.0 against 4.0). No binary model gains
   a session effect (smallest FDR p = .09). The response-time covariate itself
   weakens at the repeated morning retrieval (parietal + precuneus FDR
   p = .0001), when response times are shorter.

With events from the cue to the response instead of fixed 3 s events
(exploratory comparison, not a pipeline output) the binary same-position
effect is weaker outside visual cortex, while the graded distance effect is
unchanged.

## Stage E: univariate activity at retrieval and during learning (rewritten 2026-11-25)

Scripts are in `E_univariate/`. Scripts that are no longer used (E1, E2, E6, E8, E11) are in
`_old_scripts/E_univariate/`, with a note on each in `_old_scripts/README.md`.

**Main model (`ret_press`, decided 2026-11-23).** A retrieval trial is: fixation (3-7 s), trial-type
cue (about 2 s), retrieval cue (1.015 s), question mark, button press, holding the answer, speaking
screen (11-13 s after the question mark), next fixation. Regressors: `object` and `scene` = retrieval
cue until the button press (trials with a press); `wait` = press until the speaking screen;
`speak` = speaking screen until the next fixation; `noretrieval` = trials without a press, cue
until the timeout. Fixation and the trial-type cue have no regressor: they are the baseline.
Time points: pre = evening, post1 = first morning retrieval, post2 = repeated morning retrieval.
Results are session averages; single runs are not compared (they hold different items).

| Step | Script | Does |
| --- | --- | --- |
| E3 | `E3_wholebrain_maps.py` | Whole-brain maps per participant in standard space (4 mm), one model per run; `E3_MODELS=ret_press` for the main model (the script still holds the earlier models) |
| E4 | `E4_wholebrain_group.py` | Group test of one contrast with sign-flip permutations and TFCE (whole-brain corrected) |
| E5 | `E5_wholebrain_figures.py` | Slices with t values where corrected p < .05 |
| E7 | `E7_native_regions.py` | Main model in each participant's own anatomy (T1w, not smoothed), signal averaged per FreeSurfer region; `E7_MODELS=ret_press` |
| E9 | `E9_encoding_regions.py` | Learning runs of the evening, per region, by showing of the sequence |
| E10 | `E10_matched_search_time.py` | One estimate per trial; time points compared on trials matched for search time (object trials; `E10_KIND=scene` for scene trials) |
| E12 | `E12_precuneus_parietal_subregions.py` | Supporting check: E7 and E10 on finer subdivisions of precuneus and parietal cortex, also on reliable voxels only |
| E13 | `E13_response_time_course.py` | Supporting check: course of the signal over the trial without an assumed response shape; `E13_MATCHED=1` matches object trials for search time over the three time points |
| E14 | `E14_region_summary.py` | The tables and region figures quoted in the text, collected from E7, E9, E10 and E13; also the relations to behaviour across participants |
| E15 | `E15_learning_by_sequence.py` | Learning runs with one estimate per sequence and showing; relation of the change to retrieval success and reaction time within participants |
| E16 | `E16_paper_figures_univariate.py` | The five univariate figures of the paper in the shared style of `figure_style.py` (as C13 for decoding), drawn from the outputs of E4, E7, E9, E14 and E15; no model is fitted |

    cd B_decoding
    bash submit_cluster_jobs.sh E3            # one job per participant
    bash submit_cluster_jobs.sh E4            # after E3: 18 group tests (N_PERM=5000)
    bash submit_cluster_jobs.sh E7            # one job per participant, about 6 min each
    bash submit_cluster_jobs.sh E9
    bash submit_cluster_jobs.sh E10           # object trials, then scene trials
    bash submit_cluster_jobs.sh E12           # supporting checks
    bash submit_cluster_jobs.sh E13
    bash submit_cluster_jobs.sh E15
    cd ../E_univariate                        # when the jobs have finished:
    python E5_wholebrain_figures.py
    E7_MODELS=ret_press python E7_native_regions.py group
    python E9_encoding_regions.py group
    python E10_matched_search_time.py group && E10_KIND=scene python E10_matched_search_time.py group
    python E12_precuneus_parietal_subregions.py group
    python E13_response_time_course.py group && E13_MATCHED=1 python E13_response_time_course.py group
    python E14_region_summary.py
    python E15_learning_by_sequence.py group
    python E16_paper_figures_univariate.py

E7-E14 were run on 2026-11-23 to 2026-11-25 on the existing masks and behaviour tables, with
`SLEEPLAY_OUTPUT=<project>/logs/paper_pipeline/DE_check/out` (a folder of links to the existing
results of stages A and B); tables and plots are there until the full pipeline has been run.
E14 reproduces the tables computed by hand on 2026-11-24 to the last digit.

Results in short (34 participants with all three time points, uncorrected unless stated):

- Retrieval against baseline engages a control network at every time point: anterior cingulate,
  anterior insula, caudate, thalamus, cerebellum, visual and posterior parietal cortex and the
  posterior precuneus. Default-mode regions (medial prefrontal, lateral temporal cortex) and the
  hippocampus are below baseline.
- Whole brain, between time points: only the putamen is higher in the morning (and a small right
  precuneus cluster at the repeated retrieval); the putamen is most active right after the press
  and the difference is attributed to the faster responses.
- Regions, wording agreed on 2026-11-25: precuneus and retrosplenial cortex are **higher against
  their baseline** in the morning than in the evening (precuneus t = 2.0 and 2.4, retrosplenial 2.1
  and 2.0, N = 35; on trials matched for search time 2.0 to 3.0). This is not described as stronger
  retrieval activity: holding the answer and speaking rise against the baseline by the same amount
  (retrieval minus holding: t = 0.1 to 0.8), and the freely estimated time course (E13) shows no
  increase. Caudal ACC is higher at the first morning retrieval (t = 3.6) and that difference is
  gone on matched trials in E13.
- MTL does not change clearly from the evening to the first morning retrieval and falls from the
  first to the repeated morning retrieval (matched trials: MTL t = -3.1, parahippocampal -2.9;
  scene trials over the three time points F = 6.4, corrected p = .006). On object trials the
  hippocampus is below baseline throughout.
- Learning is reported for the whole sequence (scene onset to rating; decided 2026-11-25), with the
  objects-only model for comparison. Change per showing (second minus first showing is within one
  run, so the baseline cancels): hippocampus t = -4.9, parahippocampal -5.0, MTL -5.3, visual -5.2;
  precuneus 5.1, posterior cingulate 5.3, retrosplenial 2.3; entorhinal -0.5 and inferior parietal
  -1.8. Precuneus and posterior cingulate rise from below baseline to baseline level and do not
  change further at the third showing; medial temporal and visual activity keep falling (third
  minus second: hippocampus -2.7, parahippocampal -3.5).
- Learning and later retrieval, whole sequence, main contrast second minus first showing.
  Across participants (one value per person; may reflect a trait): a larger medial temporal decrease
  goes with faster retrieval that evening (parahippocampal r = .46, hippocampus .34, MTL .38) and,
  as a trend, with more items retrieved (-.32, -.32, -.28). Across runs (third minus first) the
  relation with reaction time is at least as strong (parahippocampal r = .53, p = .001). The
  participant with the lowest evening retrieval success (s41, 14 %) carries part of the success
  relation (-.24 without) but not the reaction-time relation (.41 without).
  Within participants (the ten sequences of a person compared with each other, E15, model block):
  sequences with a larger parahippocampal decrease are retrieved more completely (slope t = -3.5,
  p = .002; unchanged with scene identity in the model and without s41; also for the first morning
  retrieval), not in the hippocampus; no within-participant relation to reaction time.
  The launcher needs a login shell (`bash -lc`) when it is not started from a terminal.
- Participants in stage E: 36 with the evening retrieval, 34 with all three time points. The two with
  the evening only were s27 (no morning session) and s41. On 2026-11-25 (afternoon) E7, E10, E12,
  E13 and E15 were rerun for s41 and all group tables and E14 were redone WITH s41's morning session
  (retrieved items taken from the cue, see Decisions): 35 participants with all three time points.
  The tables from before are in logs/paper_pipeline/DE_check/out_before_s41_morning_2026-11-25/.
  s41 has too few trials with a button press to enter the matched comparisons (E10 objects, E13
  matched), so those are unchanged. NOT rerun: E3/E4 (whole-brain maps and group tests, still 34).

Open: E15 fails for one participant (sub-51, reading the learning events). E12 and E13 are exploratory (many regions and windows, no correction); the slide deck
(https://claude.ai/artifact/CtnyvkQzWKwgtWnRwKkyfF) still shows single runs and the earlier wording.

### Earlier approach (2026-11-19 to 2026-11-22; scripts now in `_old_scripts/`)

The text below describes E1, E2 and E6 and the two event models used before the main model was
decided. It is kept for the record.

E3-E5 were run on 2026-11-22 on the existing behaviour tables (object trials, all trials).
A version restricted to trials with a button press gave the same group maps (t maps correlate
at .99 across voxels, 95 % of significant voxels shared) and was removed again, with its
mirror image (trials without a button press: same layout, weaker, r = .79), to keep the
number of variants down. For the same reason E1 now reports only "all trials" and
"RT-adjusted" (the variants "both correct" and "relative to whole brain" agreed and were removed).

Decoding region check (2026-11-22): a region defined from the engaged posterior precuneus
(pooled whole-brain map within precuneus and retrosplenial cortex, about 800 voxels per
participant) shows no decoding effect (classifier 12 %, retrieved item t = 0.4, no change
between time points). Engagement and item content are in different places; the anatomical
parietal + precuneus region stays the decoding region. Tables, mask and scripts:
`7_rsa_exploratory/outputs/precuneus_functional/`.
Decision (2026-11-22): decoding is reported for the parietal + precuneus region as defined
all along; no sub-regions are reported. For the record, from the existing tables: the
retrieved item is as strong in the parietal lobule alone (t = 3.8) as in the combined
region (3.9), the predecessor is stronger in precuneus alone (2.9) than in the parietal
lobule (1.4), and the sequentiality increase at the first morning retrieval is seen only
in the combined region (sinusoid t = 3.8; parietal lobule -0.5, precuneus 0.1).
Maps, tables and figures: `7_rsa_exploratory/outputs/D_E_on_existing_results/E4_wholebrain_group/`
and `.../E5_wholebrain_figures/`; slides: https://claude.ai/artifact/CtnyvkQzWKwgtWnRwKkyfF.
Result in short: the same network (visual, posterior precuneus, medial frontal, striatum,
thalamus, parahippocampal) is engaged at all three time points; no change between time
points holds under both event models except a small increase in left putamen at the
repeated morning retrieval; restricting to button-press trials leaves the map unchanged.

Verification (2026-11-19): trial values (7,280 trials) differ from the
exploratory tables by at most 1.3e-4 (fixed) and 2.7e-4 (cue to response)
percent signal change, because the exploratory script read them from float16
files; group t values differ by at most 9e-5, all tables have the same rows
(`outputs/15_univariate_session_lsa/`, `..._lsa_rt/`). The relation to response
time per session had no stored counterpart for this estimator; the same
function applied to the exploratory trial tables gives the same values (9e-5).

**Rule: univariate engagement depends on response time.** With fixed 3 s
events, activity rises per second of response time in precuneus (t = 7.0
object trials / 6.7 scene trials), retrosplenial cortex (5.5 / 5.9), caudal ACC
(4.5 / 2.6), parahippocampal cortex (4.0 / 2.4) and parietal + precuneus
(3.5 / 2.2); not in hippocampus (-0.4 / -1.2). Response times are about 0.5 s
shorter in morning pass 1 than in the evening and another 0.4 s shorter in pass
2 (A5), so session comparisons are confounded in exactly these regions. With
events from the cue to the response (13 percent of trials have no response
time and get the participant's median) the relation disappears in precuneus
(-0.4), retrosplenial cortex (0.6) and caudal ACC (1.1), which are active for
longer, not more strongly, on slow trials, and becomes negative in visual
(-5.9) and parahippocampal cortex (-4.3). Neither model is neutral. Session
effects count only when they agree across both event models and with response
time as a covariate. The relation itself does not differ between sessions
(|t| below 2).

Object trials, t for fixed duration / cue to response:

| Region | Morning pass 1 - evening | Whole morning - evening | Pass 2 - pass 1 |
| --- | --- | --- | --- |
| Parietal + precuneus | 0.5 / 1.4 | 0.8 / 1.8 | 0.3 / 0.5 |
| Precuneus | 0.5 / 1.7 | 0.1 / 2.0 | -0.9 / 0.0 |
| Hippocampus | 0.4 / 1.5 | -1.4 / -0.3 | -2.5 / -2.7 |
| MTL | 1.0 / 2.4 | -1.1 / 0.7 | -4.3 / -3.8 |
| Visual | -0.2 / 2.8 | -1.3 / 2.4 | -2.5 / -1.3 |
| Caudal ACC | 3.8 / 3.6 | 2.4 / 2.9 | -4.2 / -3.4 |

Agreeing across models and with response time as a covariate: the rise in
caudal ACC from the evening to morning pass 1 (adjusted 3.9 / 3.6) and the drop
in hippocampus and MTL from pass 1 to pass 2. Scene trials: hippocampus is
lower in the whole morning than in the evening (-3.0 / -2.7; adjusted -3.3 /
-3.0), carried by pass 2 (pass 1 minus evening -0.6 / -0.5). Parietal +
precuneus shows a positive trend (adjusted whole morning minus evening 2.0 /
1.8; minus hippocampus 3.0 / 2.3) that does not survive correction. These
numbers replace the earlier ones from one model per trial, which were the same
in sign and nearly the same in size.

## Planned stages (no code yet)

- Applying the classifiers to the rest runs and the in-scanner sleep runs.
- **Successor and region tests** for the overnight change
  (`7_rsa_exploratory/18_supplementary_checks.py`, parts E to G, and
  `19_without_target_sinusoid.sh`) as an extension of C5 and C6.
- Sequential versus cached retrieval: reaction time by position and session.
