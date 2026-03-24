#!/bin/bash
set -euo pipefail

# ==============================================================================
# 02_group_level_glm_postsleep_3runs.sh — Single SLURM job
# ==============================================================================
# Group-level analysis for postsleep_3runs (matched to encoding run count).
# ==============================================================================

PATH_CODE="$(cd "$(dirname "$0")" && pwd)"
PATH_LOG="${PATH_CODE}/logs/6_GLM_v2_grp_postsleep_3runs"

TIME_LIMIT="04:00:00"

mkdir -p "$PATH_LOG"

BATCH_SCRIPT="${PATH_CODE}/grp_post3_slurm.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name grp_post3
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/grp_post3_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=${PATH_CODE}/.matplotlib
mkdir -p \${MPLCONFIGDIR}

cd ${PATH_CODE}
python 02_group_level_glm.py --session postsleep_3runs

exit
SLURM

sbatch "$BATCH_SCRIPT"
echo "Submitted group-level postsleep_3runs job."
echo "Monitor with:  squeue -u \$USER"
echo "Check logs in: $PATH_LOG"
