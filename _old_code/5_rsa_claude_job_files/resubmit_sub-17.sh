#!/bin/bash
set -euo pipefail

# Resubmit sub-17 with 64G memory (OOM at ~31.3G)

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/5_rsa_claude"
PATH_LOG="${PATH_DATA}/logs/5_rsa_rdm"
VENV="source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

TIME_LIMIT="04:00:00"
SESSIONS=("presleep" "postsleep" "postsleep_3runs")
ROIS=("vis" "hpc" "mtl")

mkdir -p "$PATH_LOG"

subj="sub-17"
JOB_SCRIPT="${PATH_CODE}/_job_rdm_${subj}.sh"

cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rsa_rdm_${subj}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/RSA_rdm_${subj}_%j.out

source /sw/batch/init.sh
${VENV}
cd ${PATH_CODE}

for session in ${SESSIONS[@]}; do
    for roi in ${ROIS[@]}; do
        echo "=========================================="
        echo "Subject: ${subj}, Session: \${session}, ROI: \${roi}"
        echo "=========================================="
        python 02_compute_rdms.py --subject ${subj} --session \${session} --roi \${roi} || echo "FAILED: ${subj} \${session} \${roi}"
    done
done

rm -f ${JOB_SCRIPT}
SLURM

sbatch "$JOB_SCRIPT"
echo "Submitted: ${subj} with 64G memory"
