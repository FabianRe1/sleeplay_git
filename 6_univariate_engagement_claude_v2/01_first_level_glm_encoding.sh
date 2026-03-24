#!/bin/bash
set -euo pipefail

# ==============================================================================
# 01_first_level_glm_encoding.sh — jobber-based (one batch per subject)
# ==============================================================================
# Submits one SLURM job per subject for first-level GLM (encoding session).
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2"
PATH_LOG="${PATH_DATA}/logs/6_GLM_v2_encoding"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
TASK_LIST="${PATH_CODE}/glm_enc_task.list"

# ==============================================================================
# SLURM / RESOURCE SETTINGS
# ==============================================================================

N_PARALLEL=1
TIME_LIMIT="08:00:00"
BATCH_SIZE=1

# ==============================================================================
# CLEAN UP OLD LOGS
# ==============================================================================

if [ -d "$PATH_LOG" ]; then
    echo "Cleaning old logs in $PATH_LOG ..."
    rm -rf "$PATH_LOG"
fi
mkdir -p "$PATH_LOG"

# ==============================================================================
# DISCOVER SUBJECTS AUTOMATICALLY
# ==============================================================================

cd "$PATH_SHARED_DERIV" || {
    echo "ERROR: cannot cd into $PATH_SHARED_DERIV"
    exit 1
}

mapfile -t sub_list < <(find . -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)

if [ ${#sub_list[@]} -eq 0 ]; then
    echo "ERROR: No sub-* folders found in $PATH_SHARED_DERIV"
    exit 1
fi

echo "Found ${#sub_list[@]} subjects:"
printf '  %s\n' "${sub_list[@]}"
echo ""

# ==============================================================================
# BUILD JOBBER TASK LIST
# ==============================================================================

> "$TASK_LIST"

for subj in "${sub_list[@]}"; do
    cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python 01_first_level_glm.py --subject ${subj} --session encoding' > ${PATH_LOG}/GLM_enc_${subj}.out 2>&1
EOF
done

echo "Task list written to $TASK_LIST (${#sub_list[@]} tasks)"

# ==============================================================================
# SPLIT INTO BATCHES AND SUBMIT
# ==============================================================================

module load jobber

n_subs=${#sub_list[@]}
n_batches=$(( (n_subs + BATCH_SIZE - 1) / BATCH_SIZE ))

echo ""
echo "Splitting ${n_subs} subjects into ${n_batches} batch(es) of up to ${BATCH_SIZE} each."
echo ""

for (( batch=0; batch < n_batches; batch++ )); do
    start=$(( batch * BATCH_SIZE ))
    end=$(( start + BATCH_SIZE ))
    if [ $end -gt $n_subs ]; then
        end=$n_subs
    fi

    BATCH_TASK_LIST="${PATH_CODE}/glm_enc_task_batch${batch}.list"
    > "$BATCH_TASK_LIST"

    for (( i=start; i < end; i++ )); do
        subj="${sub_list[$i]}"
        cat >> "$BATCH_TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python 01_first_level_glm.py --subject ${subj} --session encoding' > ${PATH_LOG}/GLM_enc_${subj}.out 2>&1
EOF
    done

    echo "Batch ${batch}: subjects ${start}..$(( end - 1 )) -> $BATCH_TASK_LIST"

    jobber "$BATCH_TASK_LIST" cleanup init

    BATCH_SCRIPT="${PATH_CODE}/glm_enc_jobber_batch${batch}.sh"

    cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name glm_enc_b${batch}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_batch${batch}_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${BATCH_TASK_LIST} all

# Clean up jobber files
rm -f ${BATCH_TASK_LIST}
rm -f ${BATCH_SCRIPT}

exit
SLURM

    sbatch "$BATCH_SCRIPT"
    echo "  -> Submitted batch ${batch}"
done

# Clean up main task list (batch-specific lists are cleaned up by the SLURM jobs)
rm -f "$TASK_LIST"

echo ""
echo "Done. Submitted ${n_batches} batch job(s)."
echo "Monitor with:  squeue -u \$USER"
echo "Check logs in: $PATH_LOG"
