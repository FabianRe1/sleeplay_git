#!/bin/bash
set -euo pipefail

# ==============================================================================
# 02_group_level_glm_encoding.sh — Single SLURM job for group-level analysis
# ==============================================================================
# Runs after ALL first-level encoding jobs have completed.
# ==============================================================================

# ==============================================================================
# PATHS
# ==============================================================================

PATH_CODE="$(cd "$(dirname "$0")" && pwd)"
PATH_LOG="${PATH_CODE}/logs/6_GLM_v2_grp_encoding"

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

BATCH_SCRIPT="${PATH_CODE}/grp_enc_slurm.sh"

cat > "$BATCH_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name grp_enc
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/grp_enc_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=${PATH_CODE}/.matplotlib
mkdir -p \${MPLCONFIGDIR}

cd ${PATH_CODE}
python 02_group_level_glm.py --session encoding

exit
SLURM

sbatch "$BATCH_SCRIPT"
echo "Submitted group-level encoding job."
echo "Monitor with:  squeue -u \$USER"
echo "Check logs in: $PATH_LOG"
