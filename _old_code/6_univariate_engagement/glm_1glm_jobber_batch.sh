#!/bin/bash
#SBATCH --job-name glm1_jobber
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/logs/6_GLM/jobber_master_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p $SLURM_NTASKS_PER_NODE /beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement/glm_1glm_task.list all

exit
