#!/bin/bash
#SBATCH --job-name sub-54_rois
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/logs/4.5_postsleep/jobber_sub-54_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p $SLURM_NTASKS_PER_NODE /beegfs/u/bbc6124/Sleeplay_Hummel/code/4_decoding/sub-54_roi_task.list all

exit
