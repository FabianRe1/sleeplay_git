#!/bin/bash

# ==============================================================================
# RERUN: Voxel reliability masks for chunk 40–44 only
# ==============================================================================
# N_PARALLEL=1 to avoid OOM (sub-44 was killed last time).
# The Python skip-check will automatically skip sub-42 (already done).
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/3_reliability_voxel_selection"
PATH_LOG="${PATH_DATA}/logs/reliability_masks_generation"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

N_PARALLEL=1
TIME_LIMIT="08:00:00"

mkdir -p "$PATH_LOG"

# ── Subjects in the 40–44 range ──────────────────────────────────────────
SUBS=()
for num in 40 41 42 43 44; do
    if [ -d "${PATH_SHARED_DERIV}/sub-${num}" ]; then
        SUBS+=("sub-${num}")
    fi
done

if [ ${#SUBS[@]} -eq 0 ]; then
    echo "ERROR: No sub-4[0-4] folders found in $PATH_SHARED_DERIV"
    exit 1
fi

echo "Subjects to process: ${SUBS[*]}"

# ── Task list ─────────────────────────────────────────────────────────────
TASK_LIST="${PATH_CODE}/reliability_mask_task_40_44.list"
> "$TASK_LIST"

for subj in "${SUBS[@]}"; do
    cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python generate_voxel_reliability_mask_localizer_1.py ${subj}' > ${PATH_LOG}/reliability_${subj}.out 2>&1
EOF
done

echo "Task list: $TASK_LIST (${#SUBS[@]} tasks)"

# ── Jobber init & submit ─────────────────────────────────────────────────
module load jobber
jobber "$TASK_LIST" cleanup init

BATCH_SCRIPT="${PATH_CODE}/reliability_mask_jobber_batch_40_44_rerun.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rel_40_44
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_master_40_44_rerun_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${TASK_LIST} all

exit
SLURM

echo "Submitting rerun batch job (N_PARALLEL=${N_PARALLEL}) ..."
sbatch "$BATCH_SCRIPT"
echo "Done. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"
