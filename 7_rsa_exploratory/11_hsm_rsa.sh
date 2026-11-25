#!/bin/bash
# Hierarchical sequence model RSA (11) for one or more pattern sources, as one SLURM job.
#   bash 11_hsm_rsa.sh                       # raw lss glmsingle glmsingle_hrf
#   bash 11_hsm_rsa.sh glmsingle_sep glmsingle_sep_hrf

PATH_CODE="/beegfs/uu/bbc6124/bbc6124/Sleeplay_Hummel/sleeplay_git/7_rsa_exploratory"
PATH_LOG="/beegfs/uu/bbc6124/bbc6124/Sleeplay_Hummel/logs/7_rsa_exploratory_11"
N_JOBS=32
SOURCES=${@:-"raw lss glmsingle glmsingle_hrf"}
mkdir -p "$PATH_LOG"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name rsa7_11
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
python 10_hsm_model_rdms.py
for src in ${SOURCES}; do
    python 11_hsm_rsa.py \$src ${N_JOBS}
done
SLURM
echo "logs: $PATH_LOG"
