#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_radiosonde_temp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=fsoi_radiosonde_temp_%j.out
#SBATCH --error=fsoi_radiosonde_temp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ========================================================================
# FSOI: Radiosonde Temperature Impact Study
# ========================================================================
# Objective: Determine how ALL observation types impact radiosonde 
#            temperature forecast error
#
# Inputs:  All obs types (ATMS, AMSUA, surface, radiosonde, etc.)
# Target:  Radiosonde temperature only
# Output:  FSOI attribution per observation type/channel
# ========================================================================

echo "=================================================="
echo "FSOI: Radiosonde Temperature Impact Analysis"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

# Load environment
set -e
set -o pipefail

# Initialize conda (adjust path if your conda is elsewhere)
# Common locations: ~/miniconda3, ~/anaconda3, /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3
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

# Navigate to working directory (no --chdir required)
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

# Create output directories
mkdir -p FSOI/fsoi_outputs/radiosonde_temp_impact
LOG_DIR="FSOI/logs"
mkdir -p "$LOG_DIR"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Reduce CUDA memory fragmentation.  When reserved-but-unallocated memory is
# large (as shown in the OOM traceback: 22 GiB reserved but not in use) the
# allocator cannot find a contiguous block for a new 11 GiB tensor.
# expandable_segments lets the allocator grow existing segments instead of
# requiring a fresh contiguous block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print environment info
echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo ""

# Display checkpoint info
# Accept either a .ckpt file or a directory containing .ckpt files.
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

echo "Checkpoint Information:"
echo "  Path: $CHECKPOINT_PATH"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "  Type: file"
    echo "  Size: $(du -h "$CHECKPOINT_PATH" | cut -f1)"
    echo "  Modified: $(stat -c %y "$CHECKPOINT_PATH" | cut -d'.' -f1)"
elif [ -d "$CHECKPOINT_PATH" ]; then
    echo "  Type: directory"
    LATEST_CKPT=$(ls -t "$CHECKPOINT_PATH"/*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "  Latest checkpoint: $LATEST_CKPT"
        echo "  Size: $(du -h "$LATEST_CKPT" | cut -f1)"
        echo "  Modified: $(stat -c %y "$LATEST_CKPT" | cut -d'.' -f1)"
    else
        echo "  ERROR: No .ckpt files found in $CHECKPOINT_PATH"
        exit 1
    fi
else
    echo "  ERROR: Checkpoint path not found: $CHECKPOINT_PATH"
    echo "  Provide a checkpoint path via either:"
    echo "    - flag:          sbatch $(basename "$0") --checkpoint checkpoints/"
    echo "    - flag:          sbatch $(basename "$0") --checkpoint /path/to/model.ckpt"
    echo "    - positional arg: sbatch $(basename "$0") checkpoints/"
    echo "    - positional arg: sbatch $(basename "$0") /path/to/model.ckpt"
    echo "    - env var: CHECKPOINT_PATH=/path/to/model.ckpt sbatch $(basename "$0")"
    echo "  (Default is '$DEFAULT_CKPT_DIR' relative to $GNN_MODEL_DIR_RESOLVED)"
    exit 1
fi
echo ""

# Display configuration
CONFIG_FILE="FSOI/configs/fsoi_config_radiosonde_temp.yaml"
echo "Configuration: $CONFIG_FILE"
echo "Key settings:"
echo "  - Target instrument: radiosonde"
echo "  - Target variable: temperature"
echo "  - Forecast lead: +12h (step 0)"
echo "  - Finite-difference validation: ENABLED (first run)"
echo ""

# ========================================================================
# ========================================================================
# Step 1: Run basic validation tests
# ========================================================================
echo "=========================================="
echo "Step 1: Running validation tests..."
echo "=========================================="
python FSOI/test_fsoi.py --checkpoint "$CHECKPOINT_PATH" 2>&1 | tee "${LOG_DIR}/test_fsoi_${SLURM_JOB_ID}.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Validation tests failed. Check ${LOG_DIR}/test_fsoi_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "✓ Validation tests passed"
echo ""

# ========================================================================
# Step 2: Run FSOI computation
# ========================================================================
echo "=========================================="
echo "Step 2: Computing FSOI..."
echo "=========================================="
echo "This will:"
echo "  1. Load all observation types from sequential time windows"
echo "  2. Compute background prediction from previous window"
echo "  3. Compute analysis prediction from current window"
echo "  4. Calculate forecast error for RADIOSONDE TEMPERATURE"
echo "  5. Compute gradients (adjoints) w.r.t. ALL input observations"
echo "  6. Validate gradients with finite differences (first pair only)"
echo "  7. Aggregate FSOI by observation type and channel"
echo ""

python FSOI/fsoi_inference.py --checkpoint "$CHECKPOINT_PATH" --config "$CONFIG_FILE" 2>&1 | tee "${LOG_DIR}/fsoi_inference_${SLURM_JOB_ID}.log"

if [ $? -ne 0 ]; then
    echo "ERROR: FSOI computation failed. Check ${LOG_DIR}/fsoi_inference_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "✓ FSOI computation completed"
echo ""

# ========================================================================
# Step 3: Analyze and summarize results
# ========================================================================
echo "=========================================="
echo "Step 3: Analyzing results..."
echo "=========================================="

OUTPUT_DIR="FSOI/fsoi_outputs/radiosonde_temp_impact"
CSV_DIR="$OUTPUT_DIR/csv"

# Check for output files
if [ -f "$CSV_DIR/fsoi_by_instrument.csv" ]; then
    echo ""
    echo "FSOI by Observation Type (Top 10):"
    echo "===================================="
    head -n 15 "$CSV_DIR/fsoi_by_instrument.csv"
    echo ""

    echo "Total impact by observation type:"
    python -c "
import pandas as pd
df = pd.read_csv('$CSV_DIR/fsoi_by_instrument.csv')
if 'instrument' in df.columns and 'mean_impact' in df.columns:
    df_sorted = df.sort_values('mean_impact')
    print('\nMost beneficial observations (negative = helpful):')
    print(df_sorted[['instrument', 'mean_impact', 'count']].head(10).to_string(index=False))
    print('\nLeast beneficial observations:')
    print(df_sorted[['instrument', 'mean_impact', 'count']].tail(5).to_string(index=False))
" 2>/dev/null || echo "Could not parse instrument summary"
    echo ""
fi

if [ -f "$CSV_DIR/fsoi_by_channel.csv" ]; then
    echo "FSOI by Channel (Top 20 channels):"
    echo "===================================="
    head -n 25 "$CSV_DIR/fsoi_by_channel.csv"
    echo ""
fi

# ========================================================================
# Step 4: Generate visualizations (if available)
# ========================================================================
echo "=========================================="
echo "Step 4: Generating visualizations..."
echo "=========================================="

if command -v python &> /dev/null && [ -d "$OUTPUT_DIR" ]; then
    python FSOI/visualize_fsoi.py \
        --input "$CSV_DIR" \
        --output "$OUTPUT_DIR/figures" \
        2>&1 | tee "${LOG_DIR}/visualize_${SLURM_JOB_ID}.log"

    if [ $? -eq 0 ]; then
        echo "✓ Visualizations created in $OUTPUT_DIR/figures/"
    else
        echo "⚠ Visualization failed (non-critical)"
    fi
else
    echo "⚠ Skipping visualization (output directory not found or python unavailable)"
fi

echo ""
echo "=================================================="
echo "FSOI Analysis Complete!"
echo "=================================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  - fsoi_by_instrument.csv   : Impact by obs type"
echo "  - fsoi_by_channel.csv      : Impact by channel/pressure level"
echo "  - fsoi_by_time.csv         : Daily time series"
echo "  - fsoi_summary.txt         : Summary statistics"
echo "  - figures/                 : Plots and visualizations"
echo ""
echo "Next steps:"
echo "  1. Review fsoi_by_instrument.csv to see which obs types help most"
echo "  2. Check fsoi_by_channel.csv for channel-specific impacts"
echo "  3. If FD validation passed, disable it for production runs"
echo "  4. Extend date range for longer-term analysis"
echo "=================================================="
