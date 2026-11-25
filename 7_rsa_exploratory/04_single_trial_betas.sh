#!/bin/bash

# ==============================================================================
# SINGLE-TRIAL BETAS (LSS, GLMsingle joint, GLMsingle separate) — jobber-based
# ==============================================================================
# One SLURM job per subject. Within each job, jobber runs that subject's
# estimators in parallel.
#
#   bash 04_single_trial_betas.sh                              # all subjects, all estimators
#   bash 04_single_trial_betas.sh "glmsingle glmsingle_sep"    # chosen estimators, all subjects
#   bash 04_single_trial_betas.sh "lss" sub-14 sub-16          # chosen estimators and subjects
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/7_rsa_exploratory"
PATH_LOG="${PATH_DATA}/logs/7_rsa_exploratory_04"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"

N_THREADS=8           # BLAS threads per estimator
TIME_LIMIT="04:00:00"

mkdir -p "$PATH_LOG"

methods=(${1:-lss glmsingle})   # also: lss_rt, lsa, glmsingle_sep
shift
if [ $# -gt 0 ]; then
    sub_list=("$@")
else
    mapfile -t sub_list < <(find "$PATH_SHARED_DERIV" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)
fi
N_PARALLEL=${#methods[@]}

echo "Estimators: ${methods[*]}"
echo "Subjects (${#sub_list[@]}): ${sub_list[*]}"

module load jobber

cd "$PATH_CODE" || { echo "ERROR: cannot cd into $PATH_CODE"; exit 1; }

for subj in "${sub_list[@]}"; do

    # -- task list for this subject: one line per estimator --
    TASK_LIST="${PATH_CODE}/${subj}_betas_task.list"
    > "$TASK_LIST"
    for m in "${methods[@]}"; do
        cat >> "$TASK_LIST" <<EOF
bash -c 'source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate && export OMP_NUM_THREADS=${N_THREADS} OPENBLAS_NUM_THREADS=${N_THREADS} MKL_NUM_THREADS=${N_THREADS} && cd ${PATH_CODE} && python 04_single_trial_betas.py ${subj} ${m}' > ${PATH_LOG}/${subj}_${m}.out 2>&1
EOF
    done

    jobber "$TASK_LIST" cleanup init

    BATCH_SCRIPT="${PATH_CODE}/${subj}_betas_jobber.sh"
    cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name ${subj}_beta
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$((N_PARALLEL * N_THREADS))
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/jobber_${subj}_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p ${N_PARALLEL} ${TASK_LIST} all

# Clean up jobber artifacts
rm -f ${TASK_LIST} ${TASK_LIST}.cont ${TASK_LIST}.queue ${BATCH_SCRIPT}

exit
SLURM

    sbatch "$BATCH_SCRIPT"
    echo "  Submitted ${subj} (${N_PARALLEL} estimator tasks)"
done

echo "All subjects submitted. Monitor with:  squeue -u \$USER"
echo "Logs in: $PATH_LOG"
