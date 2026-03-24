#!/bin/bash
#SBATCH --job-name grp_post3
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2/logs/6_GLM_v2_grp_postsleep_3runs/grp_post3_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=/beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2/.matplotlib
mkdir -p ${MPLCONFIGDIR}

cd /beegfs/u/bbc6124/Sleeplay_Hummel/code/6_univariate_engagement_claude_v2
python 02_group_level_glm.py --session postsleep_3runs

exit
