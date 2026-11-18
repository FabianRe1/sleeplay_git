#!/bin/bash
#SBATCH --job-name grp_post
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/beegfs/u/bbc6124/Sleeplay_Hummel/sleeplay_git/6_univariate_engagement_claude_v2/logs/6_GLM_v2_grp_postsleep/grp_post_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

export MPLCONFIGDIR=/beegfs/u/bbc6124/Sleeplay_Hummel/sleeplay_git/6_univariate_engagement_claude_v2/.matplotlib
mkdir -p ${MPLCONFIGDIR}

cd /beegfs/u/bbc6124/Sleeplay_Hummel/sleeplay_git/6_univariate_engagement_claude_v2
python 02_group_level_glm.py --session postsleep

exit
