#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_fd_enhanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=fsoi_fd_enhanced_%j.out
#SBATCH --error=fsoi_fd_enhanced_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ========================================================================
# FSOI: Enhanced Three-Tier Gradient Validation
#
# Runs three gradient tests on a 3-day window (2025-07-01 to 2025-07-03):
#
#   Step 1a — scalar float32 FD (existing; SKIP for high-N satellites)
#   Step 1b — Rademacher directional derivative (float32, all instruments)
#             Expected signal ATMS: ε × Σ|g_i| ~ 1.1e-4 ≈ 90 ULP
#   Step 1c — per-obs FD in float64 (resolves 1e-10 perturbations)
#
# Output: FSOI/fsoi_outputs/fd_check_enhanced/evaluation/
#   fd_validation.csv              ← Step 1a
#   fd_directional_validation.csv  ← Step 1b
#   fd_float64_validation.csv      ← Step 1c
# ========================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Enhanced Gradient Validation (3-tier)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gnn-env

# Resolve gnn_model root ------------------------------------------------
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

# Checkpoint ------------------------------------------------------------
CHECKPOINT="${CHECKPOINT_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "Override with: CHECKPOINT_PATH=/path/to/model.ckpt sbatch $0"
    exit 1
fi
echo "[CKPT] $CHECKPOINT"

CONFIG_FILE="FSOI/configs/fsoi_config_fd_enhanced.yaml"
OUTPUT_DIR="FSOI/fsoi_outputs/fd_check_enhanced"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "FSOI/logs"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[CONFIG] $CONFIG_FILE"
echo "[OUTPUT] $OUTPUT_DIR"
echo "[TESTS]  scalar float32 FD + directional (Rademacher) + float64 FD"
echo ""

python FSOI/fsoi_inference.py \
    --checkpoint  "$CHECKPOINT" \
    --config      "$CONFIG_FILE" \
    --data_path   "$DATA_PATH" \
    --output_dir  "$OUTPUT_DIR" \
    --diagnostics \
    2>&1 | tee "FSOI/logs/fd_enhanced_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================="
echo "Enhanced gradient validation finished: $(date)"
echo ""

for csv in \
    "$OUTPUT_DIR/evaluation/fd_validation.csv" \
    "$OUTPUT_DIR/evaluation/fd_directional_validation.csv" \
    "$OUTPUT_DIR/evaluation/fd_float64_validation.csv"
do
    if [ -f "$csv" ]; then
        echo "--- $(basename $csv) ---"
        # Print status summary
        python3 -c "
import pandas as pd, sys
df = pd.read_csv('$csv')
if 'status' in df.columns:
    print(df['status'].value_counts().to_string())
elif 'pearson_r' in df.columns:
    print(df[['inst_name','pearson_r','mean_rel_error','status']].drop_duplicates('inst_name').to_string(index=False))
" 2>/dev/null || cat "$csv" | head -5
        echo ""
    else
        echo "WARNING: $csv not found"
    fi
done
echo "=================================================="
