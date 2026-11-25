#!/bin/bash
# Run the paper-pipeline stages D (similarity) and E (univariate) on the EXISTING
# masks and behaviour tables and copy the tables and figures to
# 7_rsa_exploratory/outputs/D_E_on_existing_results/.
#
# The pipeline's own outputs/ folder has no stage B results until the full
# pipeline has been run. Until then the stages are pointed at a folder of links
# (SLEEPLAY_OUTPUT), which is how they were verified.
#
#   bash run_pipeline_DE_on_existing_results.sh links   # make the link folder (seconds)
#   bash run_pipeline_DE_on_existing_results.sh D1      # one job per participant (about 20 min each)
#   bash run_pipeline_DE_on_existing_results.sh rest    # D2, D3, D4, E14 and the copy, one job
#                                                       # (E1, E2 were moved to paper_pipeline/_old_scripts on 2026-11-25)
#
# "D1" submits all participants at once; to limit the number of jobs, call
# submit_cluster_jobs.sh yourself with a few participants at a time (see below).
# Nothing is deleted. Logs: <project>/logs/paper_pipeline/D1/ and .../DE_check/

ROOT="$BEEGFS/Sleeplay_Hummel"
GIT="$ROOT/sleeplay_git"
PIPE="$GIT/paper_pipeline"
WORK="$ROOT/logs/paper_pipeline/DE_check"
DEST="$GIT/7_rsa_exploratory/outputs/D_E_on_existing_results"
export SLEEPLAY_OUTPUT="$WORK/out"

case "$1" in
links)
    mkdir -p "$WORK/out"
    for s in A1_add_tr_columns_main A2_add_tr_columns_localizer A4_aggregate_behavior; do
        ln -sfn "$PIPE/outputs/$s" "$WORK/out/$s"
    done
    ln -sfn "$GIT/3_reliability_voxel_selection/combine_loc1_and_loc3_masks" "$WORK/out/B3_combine_localizer_masks"
    ln -sfn "$GIT/2_get_threshold_roi/localizer_getThresholdedROI"           "$WORK/out/B1_threshold_roi_masks"
    ls -l "$WORK/out" ;;
D1)
    # all participants; or: cd $PIPE/B_decoding && SLEEPLAY_OUTPUT=$WORK/out bash submit_cluster_jobs.sh D1 sub-11 sub-12 ...
    cd "$PIPE/B_decoding" && bash submit_cluster_jobs.sh D1 ;;
rest)
    mkdir -p "$WORK"
    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name pp_DE
#SBATCH --partition std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=${WORK}/job_%j.out

source /sw/batch/init.sh
source /usw/u/bbc6124/u/bbc6124//venvs/myenv/bin/activate
export SLEEPLAY_OUTPUT=${WORK}/out

cd ${PIPE}/D_similarity
python D2_binary_rsa.py 16              || echo "FAILED: D2"
python D3_position_and_neighbours.py 16 || echo "FAILED: D3"
python D4_figures.py                    || echo "FAILED: D4"
cd ${PIPE}/E_univariate
python E14_region_summary.py            || echo "FAILED: E14"   # needs the group tables of E7, E9, E10 (and E13)

for step in D2_binary_rsa D3_position_and_neighbours E14_region_summary; do
    mkdir -p ${DEST}/\${step}
    cp -r ${WORK}/out/\${step}/. ${DEST}/\${step}/ && echo "copied \${step}"
done
for step in D4_figures E14_region_summary; do
    mkdir -p ${DEST}/\${step}
    cp ${WORK}/plots/\${step}/* ${DEST}/\${step}/ && echo "copied \${step}"
done
echo "work folder left in place: ${WORK}"
SLURM
    echo "log: $WORK/job_<id>.out" ;;
*)
    echo "usage: bash run_pipeline_DE_on_existing_results.sh links|D1|rest" ;;
esac
