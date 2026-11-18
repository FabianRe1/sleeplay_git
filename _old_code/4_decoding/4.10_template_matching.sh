#!/bin/bash

# ==============================================================================
# SINUS 1-CYCLE FIT — jobber-based
# ==============================================================================
# Single SLURM job. Jobber runs all ROIs in parallel.
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/4_decoding"
PATH_LOG="${PATH_DATA}/logs/4.7_sinus_1cycle"
TASK_LIST="${PATH_CODE}/sinus_1cycle_task.list"

# ==============================================================================
# SLURM / RESOURCE SETTINGS
# ==============================================================================

N_PARALLEL=2          # run 2 ROIs at a time
TIME_LIMIT="08:00:00"

# ==============================================================================
# SETTINGS
# ==============================================================================

PRE_POST="presleep"   # presleep or postsleep

# ==============================================================================
# ROIs
# ==============================================================================

mask_labels=(
  # mtl
  parietal_precuneus
  parietal
  # precuneus
  # medial_ofc
  # rostral_acc
  # caudal_acc
  # superior_frontal
  # rsc
  # posterior_cingulate
  # vis
  vis
)

# ==============================================================================
# CLEAN UP OLD LOGS
# ==============================================================================

if [ -d "$PATH_LOG" ]; then
    echo "Cleaning old logs in $PATH_LOG ..."
    rm -rf "$PATH_LOG"
fi
mkdir -p "$PATH_LOG"

# ==============================================================================
# BUILD JOBBER TASK LIST (one task per ROI)
# ==============================================================================

cd "$PATH_CODE" || {
    echo "ERROR: cannot cd into $PATH_CODE"
    exit 1
}

> "$TASK_LIST"

for roi in "${mask_labels[@]}"; do
    cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python 4.10_template_matching.py ${roi} ${PRE_POST}' > ${PATH_LOG}/sinus_1cycle_${roi}.out 2>&1
EOF
done

echo "Task list written to $TASK_LIST (${#mask_labels[@]} tasks)"

# ==============================================================================
# INITIALISE JOBBER QUEUE & SUBMIT
# ==============================================================================

module load jobber
jobber "$TASK_LIST" cleanup init

BATCH_SCRIPT="${PATH_CODE}/sinus_1cycle_jobber_batch.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name template_fitter
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_master_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${TASK_LIST} all

exit
SLURM

echo ""
echo "Submitting jobber batch job ..."
sbatch "$BATCH_SCRIPT"
echo "Done. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"
