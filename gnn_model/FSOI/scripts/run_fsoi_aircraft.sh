#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_aircraft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=fsoi_aircraft_%j.out
#SBATCH --error=fsoi_aircraft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ========================================================================
# FSOI: Aircraft Observations Impact Study
# ========================================================================
# Objective: Determine how ALL observation types impact aircraft
#            forecast error: T, specific humidity, u/v wind at flight levels.
#
# Differences from radiosonde run:
#   - target_instruments: ["aircraft"]
#   - use_area_weights: true       (flights concentrated over NH corridors)
#   - finite_difference_check: true (10 samples)
#   - feature_masks: {}            (aircraft q NOT masked — aircraft is target)
#   - Longer walltime: pressure stratification × variable stratification
#     means more backward passes than radiosonde-temperature-only run.
# ========================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Aircraft Observations Impact Analysis"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate gnn-env
    echo "Activated conda environment: gnn-env"
else
    echo "ERROR: Cannot find conda at ${CONDA_BASE}"
    exit 1
fi

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
    echo "  Typical usage: sbatch $(basename "$0") checkpoints/"
    exit 1
fi

echo "[PATH] Using GNN_MODEL_DIR=$GNN_MODEL_DIR_RESOLVED"
cd "$GNN_MODEL_DIR_RESOLVED"

CONFIG_FILE="${CONFIG_FILE:-FSOI/configs/fsoi_config_aircraft.yaml}"
read -r CONFIG_START_DATE CONFIG_END_DATE <<< "$(python -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1])); data=cfg.get("data", {}); print(data.get("start_date", ""), data.get("end_date", ""))' "$CONFIG_FILE")"
START_DATE_RUN="${FSOI_START_DATE:-$CONFIG_START_DATE}"
END_DATE_RUN="${FSOI_END_DATE:-$CONFIG_END_DATE}"
DATE_TAG="${START_DATE_RUN//-/}_${END_DATE_RUN//-/}"
OUTPUT_DIR="${FSOI_OUTPUT_DIR:-FSOI/fsoi_outputs/aircraft_impact_${DATE_TAG}}"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="FSOI/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo ""

DEFAULT_CKPT_DIR="checkpoints"
CHECKPOINT_PATH_CLI=""
if [ $# -gt 0 ]; then
    case "$1" in
        --checkpoint)  CHECKPOINT_PATH_CLI="${2:-}"; shift 2 || true ;;
        --checkpoint=*) CHECKPOINT_PATH_CLI="${1#*=}"; shift 1 || true ;;
    esac
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH_CLI:-${1:-${CHECKPOINT_PATH:-$DEFAULT_CKPT_DIR}}}"

echo "Checkpoint: $CHECKPOINT_PATH"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "  Size: $(du -h "$CHECKPOINT_PATH" | cut -f1)"
elif [ -d "$CHECKPOINT_PATH" ]; then
    LATEST_CKPT=$(ls -t "$CHECKPOINT_PATH"/*.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && echo "  Latest: $LATEST_CKPT" || { echo "ERROR: No .ckpt in $CHECKPOINT_PATH"; exit 1; }
else
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"; exit 1
fi

DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH does not exist: $DATA_PATH"; exit 1; }

echo ""
echo "Configuration: $CONFIG_FILE"
echo "  Target instrument: aircraft"
echo "  Target variables: T, q, u, v"
echo "  Pressure stratified: YES (flight levels)"
echo "  Area weighted: YES"
echo "  Finite-difference check: YES (10 samples)"
echo "  Aircraft q masking: DISABLED (aircraft is verification target)"
echo "  Date range: $START_DATE_RUN to $END_DATE_RUN"
echo "  Output dir: $OUTPUT_DIR"
echo "  Data path: $DATA_PATH"
echo ""

# ========================================================================
# Step 1: Validation tests
# ========================================================================
echo "=========================================="
echo "Step 1: Running validation tests..."
echo "=========================================="
python FSOI/test_fsoi.py --checkpoint "$CHECKPOINT_PATH" \
    2>&1 | tee "${LOG_DIR}/test_fsoi_aircraft_${SLURM_JOB_ID}.log"
[ $? -eq 0 ] || { echo "ERROR: Validation tests failed"; exit 1; }
echo "✓ Validation tests passed"
echo ""

# ========================================================================
# Step 2: FSOI computation
# ========================================================================
echo "=========================================="
echo "Step 2: Computing FSOI (aircraft targets)..."
echo "=========================================="
python FSOI/fsoi_inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE" \
    --data_path "$DATA_PATH" \
    --start_date "$START_DATE_RUN" \
    --end_date "$END_DATE_RUN" \
    --output_dir "$OUTPUT_DIR" \
    --diagnostics \
    2>&1 | tee "${LOG_DIR}/fsoi_aircraft_${SLURM_JOB_ID}.log"
[ $? -eq 0 ] || { echo "ERROR: FSOI computation failed"; exit 1; }
echo "✓ FSOI computation completed"
echo ""

# ========================================================================
# Step 3: Summaries
# ========================================================================
CSV_DIR="$OUTPUT_DIR/csv"
echo "=========================================="
echo "Step 3: Summarizing results..."
echo "=========================================="
[ -f "$CSV_DIR/fsoi_by_instrument.csv" ] && {
    echo "FSOI by Observation Type (Top 10):"
    head -n 15 "$CSV_DIR/fsoi_by_instrument.csv" || true
}

# ========================================================================
# Step 4: Visualizations
# ========================================================================
echo "=========================================="
echo "Step 4: Generating visualizations..."
echo "=========================================="
python FSOI/visualize_fsoi.py \
    --input "$CSV_DIR" \
    --output "$OUTPUT_DIR/figures" \
    2>&1 | tee "${LOG_DIR}/visualize_aircraft_${SLURM_JOB_ID}.log" || true

python FSOI/plot_instrument_channel_heatmaps.py \
    --input "$CSV_DIR" \
    --output "$OUTPUT_DIR/figures" \
    2>&1 | tee "${LOG_DIR}/plot_heatmaps_aircraft_${SLURM_JOB_ID}.log" || true

python FSOI/evaluate_fsoi_results.py \
    --input "$CSV_DIR" \
    --output "$OUTPUT_DIR/evaluation" \
    2>&1 | tee "${LOG_DIR}/evaluate_aircraft_${SLURM_JOB_ID}.log" || true

# Gridded FSOI maps (requires lat/lon in scatter_samples.csv)
if [ -f "$CSV_DIR/scatter_samples.csv" ]; then
    python FSOI/plot_fsoi_maps.py \
        --input "$CSV_DIR/scatter_samples.csv" \
        --output "$OUTPUT_DIR/figures/maps" \
        --title "Aircraft FSOI" \
        2>&1 | tee "${LOG_DIR}/plot_maps_aircraft_${SLURM_JOB_ID}.log" || true
fi

# Innovation diagnostics plots
DIAG_CSV="$OUTPUT_DIR/evaluation/innovation_diagnostics.csv"
INNO_SCATTER_ARG=""; INNO_DIAG_ARG=""
[ -f "$CSV_DIR/scatter_samples.csv" ] && INNO_SCATTER_ARG="--scatter $CSV_DIR/scatter_samples.csv"
[ -f "$DIAG_CSV" ] && INNO_DIAG_ARG="--diag $DIAG_CSV"
if [ -n "$INNO_SCATTER_ARG" ] || [ -n "$INNO_DIAG_ARG" ]; then
    python FSOI/plot_innovation_diagnostics.py \
        $INNO_SCATTER_ARG $INNO_DIAG_ARG \
        --output "$OUTPUT_DIR/figures/innovation" \
        2>&1 | tee "${LOG_DIR}/plot_innovation_aircraft_${SLURM_JOB_ID}.log" || true
fi

echo "=================================================="
echo "FSOI aircraft run finished: $(date)"
echo "Outputs: $OUTPUT_DIR"
echo "=================================================="
