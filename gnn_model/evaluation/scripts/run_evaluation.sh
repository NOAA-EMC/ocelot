#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A da-cpu
#SBATCH -p u1-service
#SBATCH -J gnn_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --output=gnn_eval_%j.out
#SBATCH --error=gnn_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL


#Script to run evaluations for multiple initialization times
#Submit with: sbatch evaluation/scripts/run_evaluation.sh [START_DATE] [END_DATE]
#Example: sbatch evaluation/scripts/run_evaluation.sh 2023010100 2023010912
#Or use defaults: sbatch evaluation/scripts/run_evaluation.sh

#set -e  # Exit on error

echo "================================================"
echo "Starting batch evaluation on $(hostname)"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OCELOT_DIR="$(cd "${GNN_MODEL_DIR}/.." && pwd)"

cd "${SLURM_SUBMIT_DIR:-${GNN_MODEL_DIR}}"

# Use the clean micromamba environment (does not depend on conda).
OCELOT_ENV_HOME="${OCELOT_ENV_HOME:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/ocelot_env}"
MM="${MM:-${OCELOT_ENV_HOME}/micromamba/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${OCELOT_ENV_HOME}/micromamba_root}"
OCELOT_ENV_NAME="${OCELOT_ENV_NAME:-ocelot-cu121}"

if [[ ! -x "${MM}" ]]; then
    echo "ERROR: micromamba not found/executable at: ${MM}"
    echo "Create it via: cd ${OCELOT_ENV_HOME} && ./create_env.sh ${OCELOT_ENV_NAME}"
    exit 2
fi

PY=("${MM}" run -n "${OCELOT_ENV_NAME}" python)

# Add PYTHONPATH
export PYTHONPATH="${GNN_MODEL_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"

# ============================================
# CONFIGURATION - Set all parameters here
# ============================================

# MODE:
#   standard    -> runs evaluations.py (pred vs truth)
#   gfs_compare -> runs plot_gfs_compare.py (pred/truth vs GFS from *_vs_gfs.csv)
MODE=${MODE:-"standard"}

# For MODE=gfs_compare
# Valid: surface_obs | radiosonde | aircraft
INSTRUMENT=${INSTRUMENT:-"surface_obs"}
CHUNKSIZE=${CHUNKSIZE:-"200000"}
VARS=${VARS:-"wind"}

EVAL_SCRIPT="evaluation/scripts/evaluations.py"
GFS_COMPARE_SCRIPT="evaluation/scripts/plot_gfs_compare.py"

# --- Training Mode Parameters ---
# Leave empty ("") for testing mode, or set for training mode:
EPOCH_TO_PLOT="2"
BATCH_IDX_TO_PLOT="0"

# --- Data Directories ---
# Training mode example:
#   DATA_DIR="val_csv"                             # Training validation (obs location; set HAS_GROUND_TRUTH = true)
#   DATA_DIR="val_mesh_csv"                        # Training validation (mesh; set HAS_GROUND_TRUTH = false)
# Testing mode examples:
#   DATA_DIR="predictions/pred_csv/obs-space/"     # observation-location outputs (with ground truth; set HAS_GROUND_TRUTH = true)
#   DATA_DIR="predictions/pred_csv/mesh-grid/"     # mesh-grid outputs (forecast only; set HAS_GROUND_TRUTH = false)
DATA_DIR="val_csv"

# Output directory for plots
PLOT_DIR="evaluation/figures"

# --- Mode Configuration ---
# Set HAS_GROUND_TRUTH=true for obs-space files, has ground truth
# Set HAS_GROUND_TRUTH=false for mesh-grid files, no ground truth
HAS_GROUND_TRUTH=true

# Generate per-pressure-level maps for radiosonde/aircraft (default: true).
# (evaluations.py defaults to --plot_pressure_level_maps; opt-out with --no-plot_pressure_level_maps)
PLOT_PRESSURE_LEVEL_MAPS=${PLOT_PRESSURE_LEVEL_MAPS:-"true"}

# --- Date Range for Batch Processing ---
START_DATE=${1:-"2024112500"}
END_DATE=${2:-"2024112500"}
FHR_LIST=("")
# Examples:
#   FHR_LIST=("")        # obs-space data (always empty)
#   FHR_LIST=(3)         # mesh-grid data: Single forecast hour
#   FHR_LIST=(3 6 9 12)  # mesh-grid data: All forecast hours

# ============================================

# Check required script exists
if [ "$MODE" == "gfs_compare" ]; then
    if [ ! -f "$GFS_COMPARE_SCRIPT" ]; then
        echo "Error: $GFS_COMPARE_SCRIPT not found!"
        exit 1
    fi
else
    if [ ! -f "$EVAL_SCRIPT" ]; then
        echo "Error: $EVAL_SCRIPT not found!"
        exit 1
    fi
fi

# Validate date format (YYYYMMDDHH)
# Note: In [[ ]], the =~ operator needs a space before and after it,
# but the regex itself should NOT have spaces in the middle of it.
if [[ ! "$START_DATE" =~ ^[0-9]{10}$ ]] || [[ ! "$END_DATE" =~ ^[0-9]{10}$ ]]; then
    echo "Error: Dates must be in format YYYYMMDDHH"
    echo "Usage: $0 START_DATE END_DATE"
    echo "Example: $0 2023010100 2023010912"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Evaluation script: $EVAL_SCRIPT"
echo "  Start date: $START_DATE"
echo "  End date: $END_DATE"
echo "  Forecast hours: ${FHR_LIST[@]}"
echo "  Data directory: $DATA_DIR"
echo "  Plot directory: $PLOT_DIR"
echo "  Has ground truth?: $HAS_GROUND_TRUTH"
echo "  Mode: $MODE"
if [ "$MODE" == "gfs_compare" ]; then
    echo "  Instrument: $INSTRUMENT"
    echo "  Chunksize: $CHUNKSIZE"
    echo "  Vars: $VARS"
else
    echo "  Plot pressure-level maps?: $PLOT_PRESSURE_LEVEL_MAPS"
fi

# Fixed: Added spaces inside [ ] brackets
if [ -n "$EPOCH_TO_PLOT" ]; then
    echo "  Epoch: $EPOCH_TO_PLOT"
fi

if [ -n "$BATCH_IDX_TO_PLOT" ]; then
    echo "  Batch index: $BATCH_IDX_TO_PLOT"
fi

echo "  Python path: $PYTHONPATH"
echo "================================================"
echo ""

# Convert YYYYMMDDHH to epoch seconds for comparison
# Fixed: Removed spaces around = and inside the date/format flags
START_TIME=$(date -d "${START_DATE:0:8} ${START_DATE:8:2}:00:00" +%s)
END_TIME=$(date -d "${END_DATE:0:8} ${END_DATE:8:2}:00:00" +%s)

# Initialize counters - Fixed: Removed spaces around =
CURRENT_TIME=$START_TIME
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

# Loop through dates with 12-hour increments
# Fixed: Added spaces inside [ ] and removed space in -le
while [ $CURRENT_TIME -le $END_TIME ]
do
    # Convert epoch back to YYYYMMDDHH format
    # Fixed: Removed spaces around =, corrected date format strings
    INIT_TIME=$(date -d "@$CURRENT_TIME" +%Y%m%d%H)
    DISPLAY_DATE=$(date -d "@$CURRENT_TIME" +"%Y-%m-%d %H:%M")

    echo "----------------------------------------"
    echo "Processing init time: $DISPLAY_DATE ($INIT_TIME)"
    echo "----------------------------------------"

    # Loop through each forecast hour
    for FHR in "${FHR_LIST[@]}"
    do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        echo ""
        if [ -n "$FHR" ]; then
            echo "  Running: init_time=$INIT_TIME, fhr=$FHR"
        else
            echo "  Running: init_time=$INIT_TIME (no fhr)"
        fi

        # Build command (array) to avoid shell quoting issues.
        cmd=()
        if [ "$MODE" == "gfs_compare" ]; then
            cmd=("${PY[@]}" "$GFS_COMPARE_SCRIPT" --init_time "$INIT_TIME" --data_dir "$DATA_DIR" --plot_dir "$PLOT_DIR" --instrument "$INSTRUMENT" --vars "$VARS" --chunksize "$CHUNKSIZE")
        else
            cmd=("${PY[@]}" "$EVAL_SCRIPT" --init_time "$INIT_TIME" --data_dir "$DATA_DIR" --plot_dir "$PLOT_DIR")
        fi

        if [ "$MODE" != "gfs_compare" ] && [ "$PLOT_PRESSURE_LEVEL_MAPS" == "false" ]; then
            cmd+=(--no-plot_pressure_level_maps)
        fi

        if [ -n "$FHR" ] && [ "$MODE" != "gfs_compare" ]; then
            cmd+=(--fhr "$FHR")
        fi

        if [ "$HAS_GROUND_TRUTH" == "true" ] && [ "$MODE" != "gfs_compare" ]; then
            cmd+=(--has_ground_truth)
        fi

        if [ -n "$EPOCH_TO_PLOT" ]; then
            cmd+=(--epoch "$EPOCH_TO_PLOT")
        fi

        if [ -n "$BATCH_IDX_TO_PLOT" ]; then
            cmd+=(--batch_idx "$BATCH_IDX_TO_PLOT")
        fi

        if "${cmd[@]}"; then
            SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
            if [ -n "$FHR" ]; then
                echo "Success: init_time=$INIT_TIME, fhr=$FHR"
            else
                echo "Success: init_time=$INIT_TIME (no fhr)"
            fi
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            if [ -n "$FHR" ]; then
                echo "Failed: init_time=$INIT_TIME, fhr=$FHR (continuing...)"
            else
                echo "Failed: init_time=$INIT_TIME (no fhr) (continuing...)"
            fi
        fi
    done

    echo ""
    # Increment by 12 hours (43200 seconds)
    CURRENT_TIME=$((CURRENT_TIME + 43200))
done

echo "================================================"
echo "Batch evaluation complete"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESS_RUNS"
echo "Failed: $FAILED_RUNS"
echo "End time: $(date)"
echo "================================================"

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    echo "WARNING: Some runs failed. Check output for details."
    exit 1
fi
