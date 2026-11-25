#!/bin/bash
set -euo pipefail

# ==============================================================================
# 02_compute_rdms_parietal.sh — rerun step 02 for parietal ROI only
# ==============================================================================

PATH_DATA="$BEEGFS/Sleeplay_Hummel"
PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/5_rsa_claude"
PATH_LOG="${PATH_DATA}/logs/5_rsa_rdm_parietal"
PATH_SHARED_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
VENV="source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate"

TIME_LIMIT="04:00:00"
SESSIONS=("presleep" "postsleep" "postsleep_3runs")

# Clean old logs
rm -rf "$PATH_LOG" 2>/dev/null
mkdir -p "$PATH_LOG"

# Discover subjects
cd "$PATH_SHARED_DERIV" || exit 1
mapfile -t sub_list < <(find . -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)

if [ ${#sub_list[@]} -eq 0 ]; then
    echo "ERROR: No sub-* folders found"
    exit 1
fi

echo "Found ${#sub_list[@]} subjects. Submitting one job per subject (parietal only)."

for subj in "${sub_list[@]}"; do
    JOB_SCRIPT="${PATH_CODE}/_job_rdm_parietal_${subj}.sh"

    cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name rsa_rdm_p_${subj}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/RSA_rdm_parietal_${subj}_%j.out

source /sw/batch/init.sh
${VENV}
cd ${PATH_CODE}

for session in ${SESSIONS[@]}; do
    echo "=========================================="
    echo "Subject: ${subj}, Session: \${session}, ROI: parietal"
    echo "=========================================="
    python 02_compute_rdms.py --subject ${subj} --session \${session} --roi parietal || echo "FAILED: ${subj} \${session} parietal"
done

rm -f ${JOB_SCRIPT}
SLURM

    sbatch "$JOB_SCRIPT"
    echo "  Submitted: ${subj} (parietal only)"
done

echo ""
echo "Done. Submitted ${#sub_list[@]} jobs."
echo "Logs: $PATH_LOG"
