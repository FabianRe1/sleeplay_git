#!/bin/bash
#SBATCH --job-name tmpl_compute
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/logs/4.10_template_matching/jobber_compute_%j.out

source /sw/batch/init.sh
module load jobber

jobber -p $SLURM_NTASKS_PER_NODE /beegfs/u/bbc6124/Sleeplay_Hummel/code/4_decoding/template_compute_task.list all

exit
