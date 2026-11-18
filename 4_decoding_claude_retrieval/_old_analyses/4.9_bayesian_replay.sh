#!/bin/bash

# ==============================================================================
# BAYESIAN REPLAY MODEL COMPARISON — jobber-based (enc+ret pipeline)
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/4_decoding_claude_retrieval"
PATH_LOG="${PATH_DATA}/logs/4.9_bayesian_replay_enc_ret"
TASK_LIST="${PATH_CODE}/bayesian_task.list"

N_PARALLEL=2
TIME_LIMIT="08:00:00"

SESSIONS=(presleep postsleep)

mask_labels=(
  mtl
  parietal_precuneus
  parietal
  precuneus
  medial_ofc
  rostral_acc
  caudal_acc
  superior_frontal
  rsc
  posterior_cingulate
  vis
  cortical_memory
)

if [ -d "$PATH_LOG" ]; then
    echo "Cleaning old logs in $PATH_LOG ..."
    rm -rf "$PATH_LOG"
fi
mkdir -p "$PATH_LOG"

cd "$PATH_CODE" || {
    echo "ERROR: cannot cd into $PATH_CODE"
    exit 1
}

> "$TASK_LIST"

for session in "${SESSIONS[@]}"; do
    for roi in "${mask_labels[@]}"; do
        cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python 4.9_bayesian_replay.py ${roi} ${session}' > ${PATH_LOG}/bayesian_${roi}_${session}.out 2>&1
EOF
    done
done

N_TASKS=$(( ${#mask_labels[@]} * ${#SESSIONS[@]} ))
echo "Task list written to $TASK_LIST ($N_TASKS tasks: ${#mask_labels[@]} ROIs × ${#SESSIONS[@]} sessions)"

module load jobber
jobber "$TASK_LIST" cleanup init

BATCH_SCRIPT="${PATH_CODE}/bayesian_jobber_batch.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name bayesER
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_master_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${TASK_LIST} all

# Clean up jobber artifacts
rm -f ${TASK_LIST} ${TASK_LIST}.cont ${TASK_LIST}.queue ${BATCH_SCRIPT}

exit
SLURM

echo ""
echo "Submitting jobber batch job ..."
sbatch "$BATCH_SCRIPT"
echo "Done. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"
