#!/bin/bash
# ==============================================================================
# RE-RUN EVERYTHING DOWNSTREAM OF 4.4 — chained jobber stages
# ==============================================================================
# 4.4 has been re-run (cue-aligned, with the confound/run-mapping guards), so
# every output derived from it is stale. This submits the downstream stages as
# SLURM jobs chained with --dependency=afterok, so each stage only starts if the
# previous one finished cleanly.
#
#   stage 1  4.6_SODA.py            12 ROIs x 2 sessions   <- REQUIRED after 4.4
#   stage 2  4.7_sinus_1cycle.py    12 ROIs x 2 sessions x 2 presets (thesis, trunc)
#   stage 3  4.8_fft.py             ARCHIVED (_old_analyses/), off by default
#   stage 4  4.9_bayesian_replay.py ARCHIVED (_old_analyses/), off by default
#   stage 5  5.1_master_replay_dataframe.py   (single task)
#   stage 6  5.2_replay_statistics.py         (single task)
#
# 5.1/5.2 use the thesis version: full 5-element |SODA| and tapered SinAmp.
# Set the RUN_* switches below to skip them.
#
# Usage:  bash rerun_after_4.4.sh
# ==============================================================================

set -u

RUN_46=true      # SODA                     — required after any 4.4 re-run
RUN_47=true      # sinusoid fits (thesis + trunc presets) — required after any 4.6 re-run
RUN_48=false     # FFT band power      — ARCHIVED to _old_analyses/ (2026-09-21)
RUN_49=false     # Bayesian replay     — ARCHIVED to _old_analyses/ (2026-09-21)
RUN_51=true      # master dataframe
RUN_52=true      # replay statistics

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/4_decoding_claude_retrieval"
PATH_LOG="${PATH_DATA}/logs/rerun_after_4.4"
VENV="/usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

N_PARALLEL=2
TIME_LIMIT="12:00:00"

SESSIONS=(presleep postsleep)
mask_labels=(
  mtl parietal_precuneus parietal precuneus medial_ofc rostral_acc
  caudal_acc superior_frontal rsc posterior_cingulate vis cortical_memory
)

rm -rf "$PATH_LOG"; mkdir -p "$PATH_LOG"
cd "$PATH_CODE" || { echo "ERROR: cannot cd into $PATH_CODE"; exit 1; }
module load jobber

# ------------------------------------------------------------------------------
# submit_stage <name> <tasklist> <dependency-jobid|""> [pre-command]
#   builds the jobber queue + batch script, submits it, echoes the new job id
# ------------------------------------------------------------------------------
submit_stage () {
    local name="$1" tasklist="$2" dep="$3" pre="${4:-}"
    local batch="${PATH_CODE}/.rerun_${name}_batch.sh"

    jobber "$tasklist" cleanup init > /dev/null

    local dep_line=""
    [ -n "$dep" ] && dep_line="#SBATCH --dependency=afterok:${dep}"

    cat > "$batch" <<SLURM
#!/bin/bash
#SBATCH --job-name RR_${name}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/${name}_master_%j.out
${dep_line}

source /sw/batch/init.sh
module load jobber
${pre}

jobber -p \\\$SLURM_NTASKS_PER_NODE ${tasklist} all

rm -f ${tasklist} ${tasklist}.cont ${tasklist}.queue ${batch}
exit
SLURM

    sbatch --parsable "$batch"
}

# ------------------------------------------------------------------------------
# task-list builders
# ------------------------------------------------------------------------------
build_roi_session_tasks () {   # <script> <tasklist> <logprefix>
    local script="$1" tasklist="$2" prefix="$3"
    > "$tasklist"
    for session in "${SESSIONS[@]}"; do
        for roi in "${mask_labels[@]}"; do
            cat >> "$tasklist" <<EOF
bash -c 'source ${VENV} && cd ${PATH_CODE} && python ${script} ${roi} ${session}' > ${PATH_LOG}/${prefix}_${roi}_${session}.out 2>&1
EOF
        done
    done
}

build_single_task () {         # <script> <tasklist> <logprefix>
    local script="$1" tasklist="$2" prefix="$3"
    > "$tasklist"
    cat >> "$tasklist" <<EOF
bash -c 'source ${VENV} && cd ${PATH_CODE} && python ${script}' > ${PATH_LOG}/${prefix}.out 2>&1
EOF
}

DEP=""

if $RUN_46; then
    build_roi_session_tasks "4.6_SODA.py" "${PATH_CODE}/.rr_46.list" "46_SODA"
    DEP=$(submit_stage "46_SODA" "${PATH_CODE}/.rr_46.list" "$DEP")
    echo "stage 1  4.6 SODA                 -> job ${DEP}  (24 tasks)"
fi

if $RUN_47; then
    # both presets: "thesis" (full window, tapered SinAmp -> 4.7_sinus_1cycle/, read by 5.1)
    # and "trunc" (truncated window, grid fit -> 4.7_sinus_1cycle_trunc/)
    > "${PATH_CODE}/.rr_47.list"
    for preset in thesis trunc; do for session in "${SESSIONS[@]}"; do for roi in "${mask_labels[@]}"; do
        echo "bash -c 'source ${VENV} && cd ${PATH_CODE} && python 4.7_sinus_1cycle.py ${roi} ${session} ${preset}' > ${PATH_LOG}/47_${preset}_${roi}_${session}.out 2>&1" >> "${PATH_CODE}/.rr_47.list"
    done; done; done
    # the 4.7 summaries merge into existing files, so drop them first or stale
    # rows from earlier runs survive alongside the new ones
    PRE="rm -f ${PATH_CODE}/4.7_sinus_1cycle/summary/windowed_fit_summary_*.csv ${PATH_CODE}/4.7_sinus_1cycle_trunc/summary/windowed_fit_summary_*.csv"
    DEP=$(submit_stage "47_sinus" "${PATH_CODE}/.rr_47.list" "$DEP" "$PRE")
    echo "stage 2  4.7 sinusoid fits      -> job ${DEP}  (48 tasks: thesis + trunc)"
fi

if $RUN_48; then
    build_roi_session_tasks "4.8_fft.py" "${PATH_CODE}/.rr_48.list" "48_fft"
    DEP=$(submit_stage "48_fft" "${PATH_CODE}/.rr_48.list" "$DEP")
    echo "stage 3  4.8 FFT                  -> job ${DEP}  (24 tasks)"
fi

if $RUN_49; then
    build_roi_session_tasks "4.9_bayesian_replay.py" "${PATH_CODE}/.rr_49.list" "49_bayes"
    DEP=$(submit_stage "49_bayes" "${PATH_CODE}/.rr_49.list" "$DEP")
    echo "stage 4  4.9 Bayesian replay      -> job ${DEP}  (24 tasks)"
fi

if $RUN_51; then
    build_single_task "5.1_master_replay_dataframe.py" "${PATH_CODE}/.rr_51.list" "51_master"
    DEP=$(submit_stage "51_master" "${PATH_CODE}/.rr_51.list" "$DEP")
    echo "stage 5  5.1 master dataframe     -> job ${DEP}  (1 task)"
fi

if $RUN_52; then
    build_single_task "5.2_replay_statistics.py" "${PATH_CODE}/.rr_52.list" "52_stats"
    DEP=$(submit_stage "52_stats" "${PATH_CODE}/.rr_52.list" "$DEP")
    echo "stage 6  5.2 replay statistics    -> job ${DEP}  (1 task)"
fi

echo ""
echo "Chained with --dependency=afterok, so a failing stage stops the rest."
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $PATH_LOG"
