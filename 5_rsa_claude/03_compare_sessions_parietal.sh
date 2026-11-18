#!/bin/bash
set -euo pipefail

# ==============================================================================
# 03_compare_sessions_parietal.sh — group-level comparison for parietal only
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/5_rsa_claude"
PATH_LOG="${PATH_DATA}/logs/5_rsa_compare_parietal"
VENV="source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

TIME_LIMIT="01:00:00"

mkdir -p "$PATH_LOG"

JOB_SCRIPT="${PATH_CODE}/_job_compare_parietal.sh"

cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rsa_cmp_par
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/RSA_compare_parietal_%j.out

source /sw/batch/init.sh
${VENV}
cd ${PATH_CODE}

python 03_compare_sessions.py --roi parietal

rm -f ${JOB_SCRIPT}
SLURM

sbatch "$JOB_SCRIPT"
echo "Submitted: compare_sessions (parietal only)"
echo "Logs: $PATH_LOG"
