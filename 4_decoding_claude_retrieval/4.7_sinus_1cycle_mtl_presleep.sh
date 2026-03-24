#!/bin/bash
#SBATCH --job-name sin1c_mtl_pre
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

cd "$BEEGFS/Sleeplay_Hummel/code/4_decoding_claude_retrieval"

python 4.7_sinus_1cycle.py mtl presleep
