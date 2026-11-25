#!/bin/bash
# Sinusoid amplitude with and without the retrieved item for the four main ROIs.
#
# Runs the paper-pipeline steps B8 (slope, including the `*_wo_target` columns) and
# B9 (preset thesis, including `wo_*_w_amplitude`) on the EXISTING classifier output
# (4.4 tables), in a temporary work folder per ROI, and keeps the trial-level
# sinusoid tables in outputs/19_without_target_sinusoid/B9_thesis/<roi>/.
# The statistics are in 18_supplementary_checks.py (part G).
#
#   bash 19_without_target_sinusoid.sh        # one SLURM job per ROI, about 20-30 min each
#
# Each job leaves a work folder outputs/19_without_target_work_<roi>/ (about 250 MB of
# B8 tables). Delete those folders by hand once the jobs have printed FINISHED.

ROOT="$BEEGFS/Sleeplay_Hummel"
GIT="$ROOT/sleeplay_git"
DEC="$GIT/4_decoding_claude_retrieval"
DEST="$GIT/7_rsa_exploratory/outputs/19_without_target_sinusoid/B9_thesis"
LOG="$ROOT/logs/7_rsa_exploratory_19"
mkdir -p "$LOG" "$DEST"

for roi in parietal_precuneus vis mtl caudal_acc; do
W="$GIT/7_rsa_exploratory/outputs/19_without_target_work_${roi}"
sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name wo_${roi}
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=${LOG}/${roi}_%j.out
source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
mkdir -p ${W}/out/B7_apply_classifiers_retrieval
ln -sfn ${DEC}/4.4_transfer_visual_classifier_retrieval_presleep  ${W}/out/B7_apply_classifiers_retrieval/presleep
ln -sfn ${DEC}/4.4_transfer_visual_classifier_retrieval_postsleep ${W}/out/B7_apply_classifiers_retrieval/postsleep
export SLEEPLAY_OUTPUT=${W}/out SLURM_CPUS_PER_TASK=32 MPLBACKEND=Agg
cd ${GIT}/paper_pipeline/B_decoding
for ses in presleep postsleep; do
    python B8_sequentiality_soda.py ${roi} \$ses && python B9_sinusoid_summary.py ${roi} \$ses thesis
done
mkdir -p ${DEST}/${roi}
cp ${W}/out/B9_sinusoid_summary/thesis/${roi}/*.csv ${DEST}/${roi}/
echo FINISHED ${roi}
SLURM
done
echo "logs: $LOG"
