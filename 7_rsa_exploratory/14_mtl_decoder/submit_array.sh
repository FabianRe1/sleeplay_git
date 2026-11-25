#!/bin/bash
# ==============================================================================
# Submit one script of 14_mtl_decoder as a SLURM array, one task per participant,
# at most MAX_PARALLEL tasks at a time (the shared derivatives disk is slow).
#
#   bash submit_array.sh 01_cache_roi_timeseries.py cache "" sub-13 sub-14
#   bash submit_array.sh 02_run_variants.py stage1 "stage1"          # all participants
#
# Arguments: <python script> <tag for logs> <extra arguments, quoted> [participants ...]
# Participants default to those with a current MTL cross-validation table.
# ==============================================================================

SCRIPT="$1"; TAG="$2"; EXTRA="$3"; shift 3

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="${PATH_DATA}/sleeplay_git/7_rsa_exploratory/14_mtl_decoder"
PATH_LOG="${PATH_DATA}/logs/7_rsa_exploratory_14"
PATH_CV="${PATH_DATA}/sleeplay_git/4_decoding_claude_retrieval/4.1_train_visual_classifier/df_classifier_results_csv"
MAX_PARALLEL=${MAX_PARALLEL:-12}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}

mkdir -p "$PATH_LOG"

if [ $# -gt 0 ]; then
    sub_list=("$@")
else
    mapfile -t sub_list < <(ls "$PATH_CV" | grep "both3_2_s[0-9]*_Mask_mtl_" | sed 's/.*both3_2_s\([0-9]*\)_Mask.*/sub-\1/' | sort -u)
fi
N=${#sub_list[@]}
echo "Participants ($N): ${sub_list[*]}"

BATCH="${PATH_CODE}/_job_${TAG}.sh"
cat > "$BATCH" <<SLURM
#!/bin/bash
#SBATCH --job-name mtl_${TAG}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --array=0-$((N - 1))%${MAX_PARALLEL}
#SBATCH --output=${PATH_LOG}/${TAG}_%a_%A.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

subs=(${sub_list[*]})
subj=\${subs[\$SLURM_ARRAY_TASK_ID]}
cd ${PATH_CODE}
echo "\$subj  ${SCRIPT} ${EXTRA}"
python ${SCRIPT} \$subj ${EXTRA} > ${PATH_LOG}/${TAG}_\${subj}.out 2>&1
echo "exit code \$?"
SLURM

sbatch "$BATCH"
