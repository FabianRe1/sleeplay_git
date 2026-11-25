#!/bin/bash
# ==============================================================================
# Submit one stage-B step to the cluster (Slurm + jobber, Hummel).
#
#   bash submit_cluster_jobs.sh <step> [subject ...]
#
#   step      B1 | B2 | B4 | B6a | B7 | B8 | B9 | D1 | E3 | E4 | E7 | E9 | E10 | E12 | E13 | E15
#   subject   optional, e.g. sub-13 sub-14; default: every sub-* folder in the
#             fMRIPrep derivatives (only used by the per-participant steps)
#
#   DRY_RUN=1 bash submit_cluster_jobs.sh B4     writes the task lists and job
#                                                scripts, submits nothing
#
# Uses jobber, as the original launchers do: every Slurm job gets a task list
# and works through it N_PARALLEL tasks at a time (default 2, e.g.
# N_PARALLEL=4 bash submit_cluster_jobs.sh B4).
#   B1, B2        one job per participant (one task)
#   B4, B6a, B7   one job per participant (and session for B7); tasks = ROIs
#   B8            one job per session; tasks = ROIs
#   B9            one job per session and preset; tasks = ROIs, 16 cores each
#   D1            one job per participant (trial patterns for stages D and E), 8 cores
#   E3            one job per participant (whole-brain maps in standard space, main model), 8 cores
#   E4            one job per group contrast (TFCE permutations), 8 cores; after E3. E5 runs directly.
#   E7, E9, E10   one job per participant (regions in the participant's own anatomy: retrieval,
#                 learning, single trials for matching), 8 cores; then `python <script> group`
#                 (E7: with E7_MODELS=ret_press; E10: also with E10_KIND=scene)
#   E12, E13, E15 one job per participant (supporting checks: finer subdivisions, free time course,
#                 learning per sequence);
#                 then `python <script> group` (E13: also with E13_MATCHED=1). E14 runs directly.
# B3, B5, B6b and B10 are short and run directly (python <script>), no job needed.
#
# Order: B1 -> B2 -> B3 -> B4 -> B5, B6a -> B6b, B7 -> B8 -> B9 -> B10.
# Wait until all jobs of a step have finished before starting the next one.
# Task lists, job scripts and logs: <project>/logs/paper_pipeline/<step>/
# ==============================================================================

set -euo pipefail

STEP="${1:?Usage: bash submit_cluster_jobs.sh <step> [subject ...]}"
shift || true

VENV="/usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLEEPLAY_ROOT:-$(cd "${CODE_DIR}/../../.." && pwd)}"
DERIVATIVES="${SLEEPLAY_DERIVATIVES:-/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives}"
LOG_DIR="${PROJECT_ROOT}/logs/paper_pipeline/${STEP}"
DRY_RUN="${DRY_RUN:-0}"
N_PARALLEL="${N_PARALLEL:-2}"

ROIS=(mtl parietal_precuneus parietal precuneus medial_ofc rostral_acc caudal_acc
      superior_frontal rsc posterior_cingulate vis cortical_memory)
# SLEEPLAY_ROIS="hpc somato" limits the per-ROI steps (B4, B6a, B7, B8, B9) to these ROIs
if [ -n "${SLEEPLAY_ROIS:-}" ]; then read -ra ROIS <<< "$SLEEPLAY_ROIS"; fi
SESSIONS=(presleep postsleep)
B9_PRESETS=(thesis trunc trunc_tapered)

if [ "$#" -gt 0 ]; then
    SUBJECTS=("$@")
else
    mapfile -t SUBJECTS < <(find "$DERIVATIVES" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)
fi

mkdir -p "$LOG_DIR"

# Environment for every task: virtual environment, code folder, and the
# SLEEPLAY_* settings of the submitting shell (only those that are set).
TASK_ENV="source ${VENV} && cd ${CODE_DIR} && export MPLBACKEND=Agg"
for var in SLEEPLAY_ROOT SLEEPLAY_DERIVATIVES SLEEPLAY_OUTPUT SLEEPLAY_PLOTS; do
    if [ -n "${!var:-}" ]; then TASK_ENV+=" && export ${var}=${!var}"; fi
done

if [ "$DRY_RUN" != "1" ]; then
    # `module` is a shell function that is not safe under `set -eu`
    set +eu; module load jobber; set -eu
fi

# submit <job name> <time limit> <tasks in parallel> <cpus per task> <task> [task ...]
#   Each task is "<log name>|<command>". The tasks go into a jobber task list;
#   one Slurm job works through the list, <tasks in parallel> at a time.
submit() {
    local name="$1" time="$2" parallel="$3" cpus="$4"
    shift 4
    local list="${LOG_DIR}/${name}_task.list"
    local script="${LOG_DIR}/${name}_jobber.sh"
    : > "$list"
    local task
    for task in "$@"; do
        echo "bash -c '${TASK_ENV} && ${task#*|}' > ${LOG_DIR}/${task%%|*}.out 2>&1" >> "$list"
    done
    cat > "$script" <<SLURM
#!/bin/bash
#SBATCH --job-name ${name}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${parallel}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${time}
#SBATCH --export=NONE
#SBATCH --output=${LOG_DIR}/jobber_${name}_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${list} all

exit
SLURM
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry run] ${name}: $# task(s), ${parallel} in parallel; first: ${1#*|}"
    else
        jobber "$list" cleanup init
        sbatch "$script"
    fi
}

# One task per ROI: roi_tasks "<log prefix>" "<command with ROI placeholder>"
roi_tasks() {
    ROI_TASKS=()
    local roi
    for roi in "${ROIS[@]}"; do
        ROI_TASKS+=("$1_${roi}|${2//ROI/$roi}")
    done
}

case "$STEP" in
    B1)
        for s in "${SUBJECTS[@]}"; do
            submit "B1_${s}" "02:00:00" 1 1 "B1_${s}|python B1_threshold_roi_masks.py ${s}"
        done ;;
    B2)
        for s in "${SUBJECTS[@]}"; do
            submit "B2_${s}" "04:00:00" 1 1 "B2_${s}|python B2_voxel_reliability_masks.py ${s}"
        done ;;
    B4)
        for s in "${SUBJECTS[@]}"; do
            roi_tasks "B4_${s}" "python B4_train_classifiers.py ${s} ROI"
            submit "B4_${s}" "08:00:00" "$N_PARALLEL" 1 "${ROI_TASKS[@]}"
        done ;;
    B6a)
        for s in "${SUBJECTS[@]}"; do
            roi_tasks "B6a_${s}" "python B6a_apply_classifiers_encoding.py ${s} ROI"
            submit "B6a_${s}" "04:00:00" "$N_PARALLEL" 1 "${ROI_TASKS[@]}"
        done ;;
    B7)
        for s in "${SUBJECTS[@]}"; do
            for ses in "${SESSIONS[@]}"; do
                roi_tasks "B7_${s}_${ses}" "python B7_apply_classifiers_retrieval.py ${s} ROI ${ses}"
                submit "B7_${s}_${ses}" "04:00:00" "$N_PARALLEL" 1 "${ROI_TASKS[@]}"
            done
        done ;;
    B8)
        for ses in "${SESSIONS[@]}"; do
            roi_tasks "B8_${ses}" "python B8_sequentiality_soda.py ROI ${ses}"
            submit "B8_${ses}" "06:00:00" "$N_PARALLEL" 1 "${ROI_TASKS[@]}"
        done ;;
    B9)
        # one task at a time, 16 cores each (the fits are parallel inside the script)
        for ses in "${SESSIONS[@]}"; do
            for preset in "${B9_PRESETS[@]}"; do
                roi_tasks "B9_${ses}_${preset}" \
                    "export SLURM_CPUS_PER_TASK=16 && python B9_sinusoid_summary.py ROI ${ses} ${preset}"
                submit "B9_${ses}_${preset}" "06:00:00" 1 16 "${ROI_TASKS[@]}"
            done
        done ;;
    D1)
        # stage D: one job per participant, both retrieval sessions and both event models
        for s in "${SUBJECTS[@]}"; do
            submit "D1_${s}" "03:00:00" 1 8 "D1_${s}|cd ../D_similarity && python D1_trial_patterns.py ${s}"
        done ;;
    E3)
        # stage E: whole-brain maps in standard space, main model, one job per participant
        for s in "${SUBJECTS[@]}"; do
            submit "E3_${s}" "02:00:00" 1 8 "E3_${s}|cd ../E_univariate && export E3_MODELS=ret_press && python E3_wholebrain_maps.py ${s}"
        done ;;
    E4)
        # stage E: group maps with TFCE for the main model, one job per contrast: retrieval (object
        # and scene trials) at every time point and between time points, and the phases of the trial
        # pooled over time points. N_PERM=5000 by default.
        N_PERM="${N_PERM:-5000}"
        for k in object scene; do
            for c in pooled pre post1 post2 post1_minus_pre post2_minus_pre post_minus_pre post2_minus_post1; do
                submit "E4_${k}_${c}" "08:00:00" 1 8 "E4_ret_press_${k}_${c}|cd ../E_univariate && python E4_wholebrain_group.py ret_press ${k} ${c} ${N_PERM} 8"
            done
        done
        for k in phase_wait phase_speak; do
            submit "E4_${k}_pooled" "08:00:00" 1 8 "E4_ret_press_${k}_pooled|cd ../E_univariate && python E4_wholebrain_group.py ret_press ${k} pooled ${N_PERM} 8"
        done ;;
    E7)
        # stage E: main model in each participant's own anatomy, per region
        for s in "${SUBJECTS[@]}"; do
            submit "E7_${s}" "01:00:00" 1 8 "E7_${s}|cd ../E_univariate && export E7_MODELS=ret_press && python E7_native_regions.py ${s}"
        done ;;
    E9)
        # stage E: learning runs (evening), per region
        for s in "${SUBJECTS[@]}"; do
            submit "E9_${s}" "01:00:00" 1 8 "E9_${s}|cd ../E_univariate && python E9_encoding_regions.py ${s}"
        done ;;
    E10)
        # stage E: single trials for the comparison matched for search time; object trials, then scene trials
        for s in "${SUBJECTS[@]}"; do
            submit "E10_${s}" "02:00:00" 1 8 "E10_${s}|cd ../E_univariate && python E10_matched_search_time.py ${s} && export E10_KIND=scene && python E10_matched_search_time.py ${s}"
        done ;;
    E12)
        # stage E: E7 and E10 on finer subdivisions of precuneus and parietal cortex (supporting check)
        for s in "${SUBJECTS[@]}"; do
            submit "E12_${s}" "04:00:00" 1 8 "E12_${s}|cd ../E_univariate && python E12_precuneus_parietal_subregions.py ${s}"
        done ;;
    E13)
        # stage E: time course without an assumed response shape; all trials, then matched for search time
        for s in "${SUBJECTS[@]}"; do
            submit "E13_${s}" "02:00:00" 1 8 "E13_${s}|cd ../E_univariate && python E13_response_time_course.py ${s} && export E13_MATCHED=1 && python E13_response_time_course.py ${s}"
        done ;;
    E15)
        # stage E: learning runs with one estimate per sequence and showing (within-participant relation to retrieval)
        for s in "${SUBJECTS[@]}"; do
            submit "E15_${s}" "01:00:00" 1 8 "E15_${s}|cd ../E_univariate && python E15_learning_by_sequence.py ${s}"
        done ;;
    *)
        echo "Unknown step '${STEP}'. Choose one of: B1 B2 B4 B6a B7 B8 B9 D1 E3 E4 E7 E9 E10 E12 E13 E15" >&2
        exit 1 ;;
esac

echo "Step ${STEP}: task lists, job scripts and logs in ${LOG_DIR}"
