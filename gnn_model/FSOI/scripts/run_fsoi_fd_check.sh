#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_fd_skip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=fsoi_fd_check_%j.out
#SBATCH --error=fsoi_fd_check_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ========================================================================
# FSOI: Finite-Difference Gradient Validation (standalone, 3-day window)
#
# Runs fsoi_inference.py over 2025-07-01 to 2025-07-03 (~6 pairs).
# FD check fires on pairs 0, 2, 4 (10 samples each = 30 total tests).
# Bug fix: replace_batch_inputs() now receives observation_config.
# Output: FSOI/fsoi_outputs/fd_check_rerun/evaluation/fd_validation.csv
# ========================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Finite-Difference Gradient Validation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn-env

# Resolve gnn_model root -------------------------------------------------
find_gnn_model_dir() {
    local d="$1"
    for _ in 1 2 3 4 5 6 7 8; do
        if [ -f "$d/configs/observation_config.yaml" ] && [ -d "$d/FSOI" ]; then
            echo "$d"; return 0
        fi
        d="$(dirname -- "$d")"
    done
    return 1
}

if [ -n "${GNN_MODEL_DIR:-}" ]; then
    GNN_DIR="$GNN_MODEL_DIR"
elif GNN_DIR="$(find_gnn_model_dir "${SLURM_SUBMIT_DIR:-$PWD}")"; then
    :
elif [ -d "${SLURM_SUBMIT_DIR:-}/gnn_model" ]; then
    GNN_DIR="${SLURM_SUBMIT_DIR}/gnn_model"
else
    echo "ERROR: cannot resolve gnn_model root"; exit 1
fi

cd "$GNN_DIR"
echo "[PATH] Working dir: $(pwd)"

# Checkpoint -------------------------------------------------------------
CHECKPOINT="${CHECKPOINT_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "Override with: CHECKPOINT_PATH=/path/to/model.ckpt sbatch $0"
    exit 1
fi
echo "[CKPT] $CHECKPOINT"

# Fixed settings for this FD-only run ------------------------------------
CONFIG_FILE="FSOI/configs/fsoi_config_radiosonde_all.yaml"
START_DATE="2025-07-01"
END_DATE="2025-07-03"
OUTPUT_DIR="FSOI/fsoi_outputs/fd_check_skip"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "FSOI/logs"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[CONFIG] $CONFIG_FILE"
echo "[DATES]  $START_DATE → $END_DATE  (~6 pairs; FD on pairs 0, 2, 4)"
echo "[OUTPUT] $OUTPUT_DIR"
echo ""

# ---- Run inference (FD check fires automatically from config) ----------
python FSOI/fsoi_inference.py \
    --checkpoint  "$CHECKPOINT" \
    --config      "$CONFIG_FILE" \
    --data_path   "$DATA_PATH" \
    --start_date  "$START_DATE" \
    --end_date    "$END_DATE" \
    --output_dir  "$OUTPUT_DIR" \
    --diagnostics \
    2>&1 | tee "FSOI/logs/fd_check_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================="
echo "FD validation finished: $(date)"
echo ""

FD_CSV="$OUTPUT_DIR/evaluation/fd_validation.csv"
echo "Fix applied: get_fsoi_inputs and replace_batch_inputs now use bt_start = 7 + n_meta offset"
if [ -f "$FD_CSV" ]; then
    echo "fd_validation.csv:"
    cat "$FD_CSV"
else
    echo "WARNING: $FD_CSV not found — check the log above."
    echo ""
    echo "Checking for FD output in log..."
    grep -E "PASS|FAIL|WARNING|Relative error|autograd|central" \
         "FSOI/logs/fd_check_${SLURM_JOB_ID}.log" || true
fi
echo "=================================================="
