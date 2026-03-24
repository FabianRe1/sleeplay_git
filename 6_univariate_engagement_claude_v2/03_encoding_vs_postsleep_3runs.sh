#!/bin/bash
set -euo pipefail

# ==============================================================================
# 03_encoding_vs_postsleep_3runs.sh — Paired comparison (run-matched)
# ==============================================================================
# Encoding vs postsleep_3runs — fair comparison with matched run count.
# ==============================================================================

PATH_CODE="$(cd "$(dirname "$0")" && pwd)"
PATH_LOG="${PATH_CODE}/logs/6_GLM_v2_paired_3runs"

TIME_LIMIT="02:00:00"

mkdir -p "$PATH_LOG"

BATCH_SCRIPT="${PATH_CODE}/paired_3runs_slurm.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name paired_3r
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/paired_3runs_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=${PATH_CODE}/.matplotlib
mkdir -p \${MPLCONFIGDIR}

cd ${PATH_CODE}
python 03_encoding_vs_postsleep.py --postsleep postsleep_3runs

exit
SLURM

sbatch "$BATCH_SCRIPT"
echo "Submitted paired comparison (3-run matched) job."
echo "Monitor with:  squeue -u \$USER"
echo "Check logs in: $PATH_LOG"
