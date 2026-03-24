#!/bin/bash

# ==============================================================================
# TRAIN VISUAL CLASSIFIER — jobber-based
# ==============================================================================
# One SLURM job per subject. Within each job, jobber runs all ROIs in parallel
# (2 at a time).
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/code/4_decoding"
PATH_LOG="${PATH_DATA}/logs/train_vis_classifier"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

# ==============================================================================
# SLURM / RESOURCE SETTINGS
# ==============================================================================

N_PARALLEL=2          # run 2 ROIs at a time
TIME_LIMIT="08:00:00"

# ==============================================================================
# ROIs
# ==============================================================================

mask_labels=(
#   mtl
#   parietal_precuneus
#   parietal
#   precuneus
#   medial_ofc
#   rostral_acc
#   caudal_acc
#   superior_frontal
#   rsc
  posterior_cingulate
  vis
  cortical_memory
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

# sub_list=(sub-46)  # uncomment to test with a single subject

# ==============================================================================
# SUBMIT ONE JOBBER JOB PER SUBJECT (all ROIs parallelised within)
# ==============================================================================

module load jobber

cd "$PATH_CODE" || {
    echo "ERROR: cannot cd into $PATH_CODE"
    exit 1
}

for subj in "${sub_list[@]}"; do
    echo "--- ${subj}: building task list for ${#mask_labels[@]} ROIs ---"

    # -- task list for this subject --
    TASK_LIST="${PATH_CODE}/${subj}_train_roi_task.list"
    > "$TASK_LIST"

    for roi in "${mask_labels[@]}"; do
        cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && cd ${PATH_CODE} && python 4.1_train_visual_classifier.py ${subj} ${roi}' > ${PATH_LOG}/thrROI_${subj}_${roi}.out 2>&1
EOF
    done

    # -- init jobber queue --
    jobber "$TASK_LIST" cleanup init

    # -- write batch script --
    BATCH_SCRIPT="${PATH_CODE}/${subj}_train_roi_jobber.sh"

    cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name ${subj}_train
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_${subj}_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p \$SLURM_NTASKS_PER_NODE ${TASK_LIST} all

exit
SLURM

    # -- submit --
    sbatch "$BATCH_SCRIPT"
    echo "  Submitted ${subj} (${#mask_labels[@]} ROI tasks, ${N_PARALLEL} parallel)"
    echo ""
done

echo "All subjects submitted. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"