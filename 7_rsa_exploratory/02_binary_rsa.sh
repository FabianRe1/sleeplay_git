#!/bin/bash
# Binary-model RSA (02) and its figures (03) for one or more pattern sources, as one SLURM job.
#   bash 02_binary_rsa.sh                       # raw lss glmsingle
#   bash 02_binary_rsa.sh lss

PATH_CODE="$BEEGFS/Sleeplay_Hummel/sleeplay_git/7_rsa_exploratory"
PATH_LOG="$BEEGFS/Sleeplay_Hummel/logs/7_rsa_exploratory_02"
N_JOBS=32
SOURCES=${@:-"raw lss glmsingle"}
mkdir -p "$PATH_LOG"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name rsa7_02
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_JOBS}
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=${PATH_LOG}/job_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${PATH_CODE}
for src in ${SOURCES}; do
    python 02_binary_rsa.py \$src ${N_JOBS} && python 03_figures.py \$src
done
SLURM
echo "logs: $PATH_LOG"
