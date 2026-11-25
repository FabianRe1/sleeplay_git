#!/bin/bash
# Run the paper-pipeline steps C5, C6, C7 and C8 on the EXISTING decoding results
# (4_decoding_claude_retrieval) and copy their tables and plots to
# 7_rsa_exploratory/outputs/C{5,6,7}_on_existing_results/.
#
# The pipeline's own outputs/ folder has no stage B results until the full
# pipeline has been run. Until then this script points the pipeline at a
# temporary folder of links to the existing results (SLEEPLAY_OUTPUT), which is
# how each step was verified.
#
#   bash run_pipeline_C_on_existing_results.sh            # submits one SLURM job
#
# Logs: <project>/logs/paper_pipeline/C_on_existing/

ROOT="$BEEGFS/Sleeplay_Hummel"
GIT="$ROOT/sleeplay_git"
DEC="$GIT/4_decoding_claude_retrieval"
DEST="$GIT/7_rsa_exploratory/outputs"
LOG="$ROOT/logs/paper_pipeline/C_on_existing"
WORK="$ROOT/logs/paper_pipeline/C_on_existing/work_$$"      # link folder, removed at the end of the job
mkdir -p "$LOG"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name pp_C567
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=${LOG}/job_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate

mkdir -p ${WORK}/out/B8_sequentiality_soda ${WORK}/out/B9_sinusoid_summary
ln -sfn ${DEC}/4.6_SODA_based_on_singlesubj_4.4_presleep  ${WORK}/out/B8_sequentiality_soda/presleep
ln -sfn ${DEC}/4.6_SODA_based_on_singlesubj_4.4_postsleep ${WORK}/out/B8_sequentiality_soda/postsleep
ln -sfn ${DEC}/4.7_sinus_1cycle                           ${WORK}/out/B9_sinusoid_summary/thesis
ln -sfn ${DEC}/4.7_sinus_1cycle_trunc_tapered             ${WORK}/out/B9_sinusoid_summary/trunc_tapered
export SLEEPLAY_OUTPUT=${WORK}/out

cd ${GIT}/paper_pipeline/C_statistics
for step in C5_predecessor_reactivation C6_sequence_vs_control C7_brain_behaviour; do
    python \${step}.py || { echo "FAILED: \${step}"; continue; }
    short=\${step%%_*}
    mkdir -p ${DEST}/\${short}_on_existing_results/tables ${DEST}/\${short}_on_existing_results/plots
    cp ${WORK}/out/\${step}/*.csv ${DEST}/\${short}_on_existing_results/tables/
    cp ${WORK}/plots/\${step}/*   ${DEST}/\${short}_on_existing_results/plots/
    echo "copied \${step}"
done
# C8 needs the sinusoid tables that include the versions without the retrieved item
# (four main ROIs, made by 19_without_target_sinusoid.sh)
ln -sfn ${DEST}/19_without_target_sinusoid/B9_thesis ${WORK}/out/B9_sinusoid_summary/thesis
python C8_overnight_change.py && {
    mkdir -p ${DEST}/C8_on_existing_results/tables ${DEST}/C8_on_existing_results/plots
    cp ${WORK}/out/C8_overnight_change/*.csv ${DEST}/C8_on_existing_results/tables/
    cp ${WORK}/plots/C8_overnight_change/*   ${DEST}/C8_on_existing_results/plots/
    echo "copied C8_overnight_change"; }
echo "work folder left in place for inspection: ${WORK}"
SLURM
echo "logs: $LOG"
