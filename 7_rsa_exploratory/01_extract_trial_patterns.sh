#!/bin/bash
# Single-trial patterns for all subjects and both sessions, as ONE SLURM job
# that runs several subject-sessions in parallel on one node.
#
#   bash 01_extract_trial_patterns.sh            # all subjects
#   bash 01_extract_trial_patterns.sh sub-14     # only the named subjects

PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/7_rsa_exploratory"
PATH_LOG="$BEEGFS/Sleeplay_Hummel/logs/7_rsa_exploratory_01"
PATH_DERIV="/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives"
N_PARALLEL=8

mkdir -p "$PATH_LOG"

if [ $# -gt 0 ]; then
    subs=("$@")
else
    mapfile -t subs < <(find "$PATH_DERIV" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)
fi

TASKS="${PATH_LOG}/tasks_$$.txt"
> "$TASKS"
for s in "${subs[@]}"; do
    for ses in presleep postsleep; do
        echo "$s $ses" >> "$TASKS"
    done
done

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name rsa7_01
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_PARALLEL}
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/job_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd ${PATH_CODE}
xargs -P ${N_PARALLEL} -L 1 bash -c 'python 01_extract_trial_patterns.py \$0 \$1 > ${PATH_LOG}/\$0_\$1.out 2>&1' < ${TASKS}
SLURM
echo "tasks: $TASKS   logs: $PATH_LOG"
