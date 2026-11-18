#!/bin/bash
set -euo pipefail

# ==============================================================================
# 03_compare_sessions.sh — single job for all ROIs (group-level analysis)
# ==============================================================================
# Runs the group-level session comparison across all ROIs in one job.
# This is fast (no fMRI data loading, just RDM CSV files + statistics).
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/5_rsa_claude"
PATH_LOG="${PATH_DATA}/logs/5_rsa_compare"
VENV="source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

TIME_LIMIT="01:00:00"

mkdir -p "$PATH_LOG"

JOB_SCRIPT="${PATH_CODE}/_job_compare.sh"

cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rsa_compare
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/RSA_compare_%j.out

source /sw/batch/init.sh
${VENV}
cd ${PATH_CODE}

# Run all ROIs in a single invocation
python 03_compare_sessions.py

rm -f ${JOB_SCRIPT}
SLURM

sbatch "$JOB_SCRIPT"
echo "Submitted: compare_sessions (all ROIs in one job)"
echo "Logs: $PATH_LOG"
