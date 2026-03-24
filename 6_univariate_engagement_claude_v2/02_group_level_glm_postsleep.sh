#!/bin/bash
set -euo pipefail

# ==============================================================================
# 02_group_level_glm_postsleep.sh — Single SLURM job for group-level analysis
# ==============================================================================
# Runs after ALL first-level postsleep jobs have completed.
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_CODE="$(cd "$(dirname "$0")" && pwd)"
PATH_LOG="${PATH_CODE}/logs/6_GLM_v2_grp_postsleep"

# ==============================================================================
# SLURM SETTINGS
# ==============================================================================

TIME_LIMIT="04:00:00"

# ==============================================================================
# SETUP
# ==============================================================================

mkdir -p "$PATH_LOG"

# ==============================================================================
# SUBMIT
# ==============================================================================

BATCH_SCRIPT="${PATH_CODE}/grp_post_slurm.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name grp_post
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/grp_post_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=${PATH_CODE}/.matplotlib
mkdir -p \${MPLCONFIGDIR}

cd ${PATH_CODE}
python 02_group_level_glm.py --session postsleep

exit
SLURM

sbatch "$BATCH_SCRIPT"
echo "Submitted group-level postsleep job."
echo "Monitor with:  squeue -u \$USER"
echo "Check logs in: $PATH_LOG"
