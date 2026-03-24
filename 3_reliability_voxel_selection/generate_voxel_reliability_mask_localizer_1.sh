#!/bin/bash

# ==============================================================================
# VOXEL RELIABILITY MASK — jobber-based
# ==============================================================================
# Separate SLURM jobs, one per chunk of 5 subjects.
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/3_reliability_voxel_selection"
PATH_LOG="${PATH_DATA}/logs/reliability_masks_generation"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

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
# SKIP SUBJECTS WITH FAILED EXPERIMENTS / MISSING DATA
# ==============================================================================
# sub-31,33,35,39,49,56 : no behavioral data at all
# sub-54,55,57           : missing localizer behavioral data
# sub-38,45              : behavior/fMRI run-count mismatch
# sub-50                 : confound/BOLD TR mismatch
SKIP_SUBS="sub-31 sub-33 sub-35 sub-39 sub-49 sub-50 sub-54 sub-55 sub-56 sub-57"

filtered_list=()
for subj in "${sub_list[@]}"; do
    if [[ " $SKIP_SUBS " =~ " $subj " ]]; then
        echo "  SKIP $subj (failed experiment)"
    else
        filtered_list+=("$subj")
    fi
done
sub_list=("${filtered_list[@]}")
echo "After filtering: ${#sub_list[@]} subjects remain."
echo ""

# ==============================================================================
# GROUP SUBJECTS INTO CHUNKS OF 5
# ==============================================================================

declare -A chunk_subjects
CHUNKS=()

for subj in "${sub_list[@]}"; do
    # Extract the numeric ID from sub-XX
    num="${subj#sub-}"
    chunk=$(( (num / 5) * 5 ))
    chunk_subjects[$chunk]+="${subj} "
    # Track unique chunk keys
    if [[ ! " ${CHUNKS[*]} " =~ " ${chunk} " ]]; then
        CHUNKS+=("$chunk")
    fi
done

# Sort chunk keys numerically
IFS=$'\n' CHUNKS=($(sort -n <<< "${CHUNKS[*]}")); unset IFS

module load jobber

# ==============================================================================
# BUILD TASK LIST, BATCH SCRIPT & SUBMIT — one per chunk of 5
# ==============================================================================

for chunk in "${CHUNKS[@]}"; do
    subjects="${chunk_subjects[$chunk]}"
    chunk_end=$(( chunk + 4 ))
    chunk_label="${chunk}_${chunk_end}"

    # Skip chunks with no subjects
    if [ -z "$subjects" ]; then
        echo "Chunk ${chunk_label}: no subjects, skipping."
        echo ""
        continue
    fi

    read -ra subs <<< "$subjects"
    echo "Chunk ${chunk_label}: ${#subs[@]} subjects — ${subjects}"

    # ── Task list ─────────────────────────────────────────────────────────
    TASK_LIST="${PATH_CODE}/reliability_mask_task_${chunk_label}.list"
    > "$TASK_LIST"

    for subj in "${subs[@]}"; do
        cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python generate_voxel_reliability_mask_localizer_1.py ${subj}' > ${PATH_LOG}/reliability_${subj}.out 2>&1
EOF
    done

    echo "  Task list: $TASK_LIST (${#subs[@]} tasks)"

    # ── Jobber init ───────────────────────────────────────────────────────
    jobber "$TASK_LIST" cleanup init

    # ── Batch script ──────────────────────────────────────────────────────
    BATCH_SCRIPT="${PATH_CODE}/reliability_mask_jobber_batch_${chunk_label}.sh"

    cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rel_${chunk_label}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_master_${chunk_label}_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${TASK_LIST} all

exit
SLURM

    # ── Submit ────────────────────────────────────────────────────────────
    echo "  Submitting ${chunk_label} batch job ..."
    sbatch "$BATCH_SCRIPT"
    echo ""
done

echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"
