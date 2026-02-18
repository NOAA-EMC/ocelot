#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A da-cpu
#SBATCH -p u1-service
#SBATCH -J gnn_eval_M2M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --output=gnn_eval_%j.out
#SBATCH --error=gnn_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL


#Script to run evaluations for multiple initialization times
#Submit with: sbatch run_evaluation.sh [START_DATE] [END_DATE]
#Example: sbatch run_evaluation.sh 2023010100 2023010912
#Or use defaults: sbatch run_evaluation.sh

#set -e  # Exit on error

echo "================================================"
echo "Starting batch evaluation on $(hostname)"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "================================================"

# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Add PYTHONPATH
export NNJA_LOCAL_ROOT=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/nnja-ai
export PYTHONPATH=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot/gnn_model:/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot:$PYTHONPATH

# ============================================
# CONFIGURATION - Set all parameters here
# ============================================

EVAL_SCRIPT="evaluations.py"

# --- Training Mode Parameters ---
# Leave empty for testing mode, or set for training mode:
EPOCH_TO_PLOT="9"
BATCH_IDX_TO_PLOT="0"

# --- Data Directories ---
# Training mode example:
#   DATA_DIR="val_csv"                             # Training validation
# Testing mode examples:
#   DATA_DIR="predictions/pred_csv/target/"        # Target files (forecast only)
#   DATA_DIR="predictions/pred_csv/non-target/"    # Non-target files (with truth)
DATA_DIR="val_target_csv" # "predictions/pred_csv/non-target/"

# Output directory for plots
PLOT_DIR="final_fig_val/target"  # "figures/"

# --- Mode Configuration ---
# Set PRED_ONLY=true for forecast-only (target files, no ground truth)
# Set PRED_ONLY=false for comparison mode (non-target files, has ground truth)
PRED_ONLY=true

# --- Date Range for Batch Processing ---
START_DATE=${1:-"2024112500"}
END_DATE=${2:-"2024112500"}
FHR_LIST=(12)
# Examples:
#   FHR_LIST=(3 6 9 12)  # All forecast hours
#   FHR_LIST=(3)         # Single forecast hour
#   FHR_LIST=("")        # Non-target data

# ============================================

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: $EVAL_SCRIPT not found!"
    exit 1
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
echo "  Prediction only: $PRED_ONLY"

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

        # Build Python command
        # Fixed: Removed spaces around =
        CMD="python $EVAL_SCRIPT --init_time $INIT_TIME --data_dir $DATA_DIR --plot_dir $PLOT_DIR"

        if [ -n "$FHR" ]; then
            CMD="$CMD --fhr $FHR"
        fi

        # Fixed: Comparison for strings usually needs == and PRED_ONLY was set to "false" earlier
        if [ "$PRED_ONLY" == "true" ]; then
            CMD="$CMD --pred_only"
        fi

        if [ -n "$EPOCH_TO_PLOT" ]; then
            CMD="$CMD --epoch $EPOCH_TO_PLOT"
        fi
        
        if [ -n "$BATCH_IDX_TO_PLOT" ]; then
            CMD="$CMD --batch_idx $BATCH_IDX_TO_PLOT"
        fi

        # Run Python script
        # Note: eval is used here to properly interpret the string as a command
        if eval $CMD; then
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
