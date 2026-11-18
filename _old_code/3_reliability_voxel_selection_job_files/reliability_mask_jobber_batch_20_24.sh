#!/bin/bash
#SBATCH --job-name rel_20_24
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/logs/reliability_masks_generation/jobber_master_20_24_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p $SLURM_NTASKS_PER_NODE /beegfs/u/bbc6124/Sleeplay_Hummel/code/3_reliability_voxel_selection/reliability_mask_task_20_24.list all

exit
