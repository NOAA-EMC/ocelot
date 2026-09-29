#!/bin/bash -l
# EXPERIMENT E3 (Reviewer 2: rollout beyond the 4 trained steps)
# Roll the frozen OCELOT v1 latent processor to 48 h (16 x 3 h steps; 12 more than trained)
# on the 2025 evaluation subset. No retraining: the processor keeps its W = 4 sliding cache,
# so steps 5-16 run on a window that no longer contains the encoded initial state, which is
# exactly the out-of-distribution regime the reviewer asked about.
#
# Submit (from gnn_model/):  mkdir -p logs && sbatch evaluation/revision/run_extended_rollout.sh
# Override horizon:          sbatch --export=ALL,ROLLOUT_WINDOWS=2 evaluation/revision/run_extended_rollout.sh  (24 h)
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J ocelot_rollout48
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH --array=0-121%24
#SBATCH --output=logs/ocelot_rollout_%A_%a.out
#SBATCH --error=logs/ocelot_rollout_%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/evaluation/revision/revision_env.sh"

ROLLOUT_WINDOWS=${ROLLOUT_WINDOWS:-4}          # 4 x 12 h = 48 h
TARGET_HOURS=$(( ROLLOUT_WINDOWS * 12 ))
if [[ -n "${INIT_LIST_FILE:-}" ]]; then mapfile -t INITS < "${INIT_LIST_FILE}"; else mapfile -t INITS < <(make_subset_inits); fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
(( IDX < ${#INITS[@]} )) || { echo "index ${IDX} out of range"; exit 0; }
INIT="${INITS[$IDX]}"

OUT_ROOT="predictions/rollout_${TARGET_HOURS}h"
echo "Extended rollout: init=${INIT} horizon=${TARGET_HOURS}h -> ${OUT_ROOT}"
run_one_init "${INIT}" "${OUT_ROOT}" "${ROLLOUT_WINDOWS}"

# Afterwards (CPU):
#   python evaluation/revision/revision_metrics.py \
#     --pred_dir predictions/rollout_48h/pred_csv/obs-space \
#     --clim_dir evaluation/revision/climatology \
#     --out_dir evaluation/revision/results/rollout_48h
# GFS can be added for leads <= the longest fhr in GFS_ROOT (the default archive holds f000-f012;
# stage f015-f048 to extend the GFS curve with evaluation/scripts/compare_to_gfs.py).
