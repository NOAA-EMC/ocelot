#!/bin/bash
#SBATCH --job-name=fsoi_inference
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --output=fsoi_%j.out
#SBATCH --error=fsoi_%j.err

# FSOI Inference Script
# Single-GPU, deterministic, sequential inference for computing FSOI

echo "=========================================="
echo "FSOI Inference - Starting"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

set -e
set -o pipefail

# Resolve gnn_model working directory (so you can submit from gnn_model/FSOI/scripts)
find_gnn_model_dir() {
    local start_dir="$1"
    local d="$start_dir"
    for _ in 1 2 3 4 5 6 7 8; do
        if [ -f "$d/configs/observation_config.yaml" ] && [ -d "$d/FSOI" ]; then
            echo "$d"
            return 0
        fi
        d="$(dirname -- "$d")"
    done
    return 1
}

GNN_MODEL_DIR_RESOLVED=""
if [ -n "${GNN_MODEL_DIR:-}" ]; then
    GNN_MODEL_DIR_RESOLVED="$GNN_MODEL_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$SLURM_SUBMIT_DIR")"; then
        :
    else
        if [ -d "$SLURM_SUBMIT_DIR/gnn_model" ] && [ -f "$SLURM_SUBMIT_DIR/gnn_model/configs/observation_config.yaml" ]; then
            GNN_MODEL_DIR_RESOLVED="$SLURM_SUBMIT_DIR/gnn_model"
        fi
    fi
else
    if GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$PWD")"; then
        :
    fi
fi

if [ -z "$GNN_MODEL_DIR_RESOLVED" ]; then
    echo "ERROR: Could not resolve gnn_model working directory."
    echo "  Typical usage (from gnn_model/FSOI/scripts):"
    echo "    sbatch $(basename "$0") checkpoints/"
    echo "  Or provide an override:"
    echo "    GNN_MODEL_DIR=/path/to/ocelot/gnn_model sbatch $(basename "$0") checkpoints/"
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  PWD=$PWD"
    exit 1
fi

echo "[PATH] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[PATH] Using GNN_MODEL_DIR=$GNN_MODEL_DIR_RESOLVED"
cd "$GNN_MODEL_DIR_RESOLVED"
echo "[PATH] PWD=$(pwd)"

# Create logs directory (keep FSOI logs under FSOI/)
mkdir -p FSOI/logs

# Activate conda environment (match other FSOI runners)
CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate gnn-env
    echo "Activated conda environment: gnn-env"
else
    echo "ERROR: Cannot find conda at ${CONDA_BASE}"
    echo "Please update CONDA_BASE in this script to point to your conda installation"
    exit 1
fi

# Set environment variables for single-GPU deterministic inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# FSOI Configuration
# Priority: 1) --checkpoint PATH / --checkpoint=PATH, 2) first positional arg, 3) CHECKPOINT_PATH env var, 4) default checkpoints/
DEFAULT_CKPT_DIR="checkpoints"

CHECKPOINT_PATH_CLI=""
if [ $# -gt 0 ]; then
    case "$1" in
        --checkpoint)
            CHECKPOINT_PATH_CLI="${2:-}"
            shift 2 || true
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH_CLI="${1#*=}"
            shift 1 || true
            ;;
    esac
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH_CLI:-${1:-${CHECKPOINT_PATH:-$DEFAULT_CKPT_DIR}}}"

OUTPUT_DIR="${OUTPUT_DIR:-./FSOI/fsoi_outputs/$(date +%Y%m%d_%H%M%S)}"
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-01-31}"

if [ ! -f "$CHECKPOINT_PATH" ] && [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path not found: $CHECKPOINT_PATH"
    echo "Provide a checkpoint path via either:"
    echo "  - flag:          sbatch $(basename "$0") --checkpoint checkpoints/"
    echo "  - flag:          sbatch $(basename "$0") --checkpoint /path/to/model.ckpt"
    echo "  - positional arg: sbatch $(basename "$0") checkpoints/"
    echo "  - positional arg: sbatch $(basename "$0") /path/to/model.ckpt"
    echo "  - env var: CHECKPOINT_PATH=/path/to/model.ckpt sbatch $(basename "$0")"
    exit 1
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Date Range: $START_DATE to $END_DATE"
echo "=========================================="

# Run FSOI inference
python FSOI/fsoi_inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config FSOI/configs/fsoi_config.yaml \
    --obs_config configs/observation_config.yaml \
    --output_dir "$OUTPUT_DIR" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --device cuda:0

exit_code=$?

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "FSOI Inference - Completed Successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "FSOI Inference - Failed with exit code $exit_code"
fi
echo "Date: $(date)"
echo "=========================================="

exit $exit_code
