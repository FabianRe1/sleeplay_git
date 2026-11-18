#!/bin/bash

# ==============================================================================
# GLM FOR BETA MAPS — jobber-based
# ==============================================================================
# Single SLURM job. Jobber runs all subjects in parallel.
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/6_univariate_engagement"
PATH_LOG="${PATH_DATA}/logs/6_GLM"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
TASK_LIST="${PATH_CODE}/glm_task.list"

# ==============================================================================
# SLURM / RESOURCE SETTINGS
# ==============================================================================

N_PARALLEL=2
TIME_LIMIT="08:00:00"

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
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python GLM_for_beta_maps.py ${subj}' > ${PATH_LOG}/GLM_${subj}.out 2>&1
EOF
done

echo "Task list written to $TASK_LIST (${#sub_list[@]} tasks)"

# ==============================================================================
# INITIALISE JOBBER QUEUE & SUBMIT
# ==============================================================================

module load jobber
jobber "$TASK_LIST" cleanup init

BATCH_SCRIPT="${PATH_CODE}/glm_jobber_batch.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name glm_jobber
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
