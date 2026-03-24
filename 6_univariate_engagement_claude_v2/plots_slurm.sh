#!/bin/bash
#SBATCH --job-name plots_v2
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2/logs/6_GLM_v2_plots/plots_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=/beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2/.matplotlib
mkdir -p ${MPLCONFIGDIR}

cd /beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2
python 04_plot_results.py

exit
