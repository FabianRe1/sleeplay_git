#!/bin/bash
#SBATCH --job-name SODA_post
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/logs/4.6_SODA_based_on_singlesubj_4.4_postsleep/jobber_master_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p $SLURM_NTASKS_PER_NODE /beegfs/u/bbc6124/Sleeplay_Hummel/code/4_decoding/soda_postsleep_task.list all

exit
