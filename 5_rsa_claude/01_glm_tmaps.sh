#!/bin/bash
set -euo pipefail

# ==============================================================================
# 01_glm_tmaps.sh — runs all sessions for each subject in a single job
# ==============================================================================
# Each SLURM job processes one subject across all 3 sessions (presleep,
# postsleep, postsleep_3runs). This keeps job count = number of subjects.
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/5_rsa_claude"
PATH_LOG="${PATH_DATA}/logs/5_rsa_glm"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
VENV="source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

TIME_LIMIT="12:00:00"
SESSIONS=("presleep" "postsleep" "postsleep_3runs")

# Clean old logs
rm -rf "$PATH_LOG" 2>/dev/null
mkdir -p "$PATH_LOG"

# Discover subjects
cd "$PATH_SHARED_DERIV" || { echo "ERROR: cannot cd into $PATH_SHARED_DERIV"; exit 1; }
mapfile -t sub_list < <(find . -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)

if [ ${#sub_list[@]} -eq 0 ]; then
    echo "ERROR: No sub-* folders found in $PATH_SHARED_DERIV"
    exit 1
fi

echo "Found ${#sub_list[@]} subjects. Submitting one job per subject (all sessions)."

# Submit one job per subject
for subj in "${sub_list[@]}"; do
    JOB_SCRIPT="${PATH_CODE}/_job_glm_${subj}.sh"

    cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rsa_glm_${subj}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/RSA_glm_${subj}_%j.out

source /sw/batch/init.sh
${VENV}
cd ${PATH_CODE}

for session in ${SESSIONS[@]}; do
    echo "=========================================="
    echo "Subject: ${subj}, Session: \${session}"
    echo "=========================================="
    python 01_glm_tmaps_residuals.py --subject ${subj} --session \${session} || echo "FAILED: ${subj} \${session}"
done

# Clean up this script
rm -f ${JOB_SCRIPT}
SLURM

    sbatch "$JOB_SCRIPT"
    echo "  Submitted: ${subj} (all sessions)"
done

echo ""
echo "Done. Submitted ${#sub_list[@]} jobs."
echo "Monitor: squeue -u \$USER"
echo "Logs:    $PATH_LOG"
