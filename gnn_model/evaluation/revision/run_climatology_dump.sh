#!/bin/bash -l
# EXPERIMENT E2 (Reviewer 1 c2, Reviewer 2 climatology baseline)
# Dump TRAINING-period (2015-2023) target observations in exactly the verification format
# (same QC, subsampling, inverse normalization) to build an observation-space climatology.
# Sampled dates: the 5th, 15th and 25th of every month, 2015-2023 (324 tasks, ~970 inits).
# Only the true_*/mask_* columns are used; the model forward pass is incidental.
#
# Submit:  sbatch evaluation/revision/run_climatology_dump.sh
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J ocelot_clim_dump
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH --array=0-323%24
#SBATCH --output=logs/ocelot_clim_dump_%A_%a.out
#SBATCH --error=logs/ocelot_clim_dump_%A_%a.err

set -euo pipefail
# Submit from the gnn_model/ directory (mkdir -p logs first).
source "${SLURM_SUBMIT_DIR:-$PWD}/evaluation/revision/revision_env.sh"

mapfile -t DAYS < <(python - <<'EOF'
for y in range(2015, 2024):
    for m in range(1, 13):
        for d in (5, 15, 25):
            print(f"{y}-{m:02d}-{d:02d}")
EOF
)
DAY="${DAYS[${SLURM_ARRAY_TASK_ID:-0}]}"
PREV=$(date -u -d "${DAY} -1 day" +%Y-%m-%d)
NEXT=$(date -u -d "${DAY} +1 day" +%Y-%m-%d)
OUT_ROOT="${OUT_ROOT:-predictions/clim_dump_2015_2023}/${DAY}"
echo "Climatology dump day=${DAY} -> ${OUT_ROOT}"

# v1.0 binning: start=DAY-1, end=DAY+1 yields the DAY-1 12Z, DAY 00Z and DAY 12Z inits.
unset PREDICT_INIT_TIME_FILTER
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python predict_gnn.py \
  --checkpoint "${CKPT}" --data_path "${DATA_PATH}" \
  --start_date "${PREV}" --end_date "${NEXT}" \
  --output_dir "${OUT_ROOT}" --eval-mode --devices 1 --num_nodes 1 --batch_size 1

# After all tasks finish (CPU is fine):
#   python evaluation/revision/build_obs_climatology.py \
#     --pred_dir predictions/clim_dump_2015_2023 --recursive \
#     --out_dir evaluation/revision/climatology
