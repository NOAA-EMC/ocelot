#!/bin/bash
# ============================================================================
# Submit 4 seasonal aircraft-target FSOI jobs (Jan, Apr, Jul, Oct 2025).
#
# Objective: measure how ALL observation types impact forecast error scored
#            at AIRCRAFT verification targets (T, q, u, v at flight levels).
#
# This is the companion to submit_fsoi_seasonal_rerun.sh which targets
# radiosonde. Running both allows cross-verification: if an instrument is
# detrimental at radiosonde sites AND at aircraft sites, that finding is
# network-independent and much stronger for the paper.
#
# Output dirs:
#   FSOI/fsoi_outputs/aircraft_seasonal/aircraft_jan2025/
#   FSOI/fsoi_outputs/aircraft_seasonal/aircraft_apr2025/
#   FSOI/fsoi_outputs/aircraft_seasonal/aircraft_jul2025/
#   FSOI/fsoi_outputs/aircraft_seasonal/aircraft_oct2025/
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_aircraft_seasonal.sh [/path/to/checkpoint.ckpt]
#
# After all 4 jobs complete, generate paper heatmaps:
#   python FSOI/generate_paper_heatmaps.py \
#       --output FSOI/fsoi_outputs/paper_figures_aircraft
# ============================================================================

set -e

# ── Resolve gnn_model working directory ──────────────────────────────────────
# This script must be run with `bash`, not `sbatch`.
# If accidentally submitted via sbatch, SLURM_SUBMIT_DIR points to the real
# submit location; fall back to that before trying the script-relative path.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/configs/observation_config.yaml" ]; then
    # Running inside a SLURM job (accidental sbatch) — use submit directory
    GNN_MODEL_DIR="$SLURM_SUBMIT_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/gnn_model/configs/observation_config.yaml" ]; then
    GNN_MODEL_DIR="${SLURM_SUBMIT_DIR}/gnn_model"
else
    # Normal interactive use: derive from script location
    GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [ ! -f "${GNN_MODEL_DIR}/configs/observation_config.yaml" ]; then
    echo "ERROR: Could not locate gnn_model directory."
    echo "  Run this script interactively: bash FSOI/scripts/submit_fsoi_aircraft_seasonal.sh"
    echo "  Do NOT submit it with sbatch — it is a submission orchestrator, not a job script."
    exit 1
fi

cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

CHECKPOINT_PATH="${1:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
CONFIG="FSOI/configs/fsoi_config_aircraft.yaml"

[ -f "$CHECKPOINT_PATH" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH"; exit 1; }
[ -d "$DATA_PATH" ]       || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }
[ -f "$CONFIG" ]          || { echo "ERROR: config not found: $CONFIG"; exit 1; }

echo "[SUBMIT] Checkpoint : $CHECKPOINT_PATH"
echo "[SUBMIT] Data path  : $DATA_PATH"
echo "[SUBMIT] Config     : $CONFIG"
echo ""

# Verify area weighting is enabled
AW=$(grep "use_area_weights" "$CONFIG" | head -1)
echo "[CONFIG CHECK] $AW"
echo ""

mkdir -p FSOI/fsoi_outputs/aircraft_seasonal/logs

# ── Jan 2025 (winter) ─────────────────────────────────────────────────────────
echo "Submitting Jan 2025 (winter)..."
JID_JAN=$(sbatch \
    --job-name="fsoi_aircraft_jan2025" \
    --time="20:00:00" \
    --output="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_jan2025_%j.out" \
    --error="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_jan2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --nodes=1 --ntasks-per-node=1 \
    --exclude=u22g09,u22g08,u22g10 \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="2025-01-01",\
FSOI_END_DATE="2025-01-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/aircraft_seasonal/aircraft_jan2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_aircraft.sh --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_JAN"

# ── Apr 2025 (spring) ─────────────────────────────────────────────────────────
echo "Submitting Apr 2025 (spring)..."
JID_APR=$(sbatch \
    --job-name="fsoi_aircraft_apr2025" \
    --time="20:00:00" \
    --output="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_apr2025_%j.out" \
    --error="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_apr2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --nodes=1 --ntasks-per-node=1 \
    --exclude=u22g09,u22g08,u22g10 \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="2025-04-01",\
FSOI_END_DATE="2025-04-30",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/aircraft_seasonal/aircraft_apr2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_aircraft.sh --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_APR"

# ── Jul 2025 (summer) ─────────────────────────────────────────────────────────
echo "Submitting Jul 2025 (summer)..."
JID_JUL=$(sbatch \
    --job-name="fsoi_aircraft_jul2025" \
    --time="20:00:00" \
    --output="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_jul2025_%j.out" \
    --error="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_jul2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --nodes=1 --ntasks-per-node=1 \
    --exclude=u22g09,u22g08,u22g10 \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="2025-07-01",\
FSOI_END_DATE="2025-07-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/aircraft_seasonal/aircraft_jul2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_aircraft.sh --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_JUL"

# ── Oct 2025 (autumn) ─────────────────────────────────────────────────────────
echo "Submitting Oct 2025 (autumn)..."
JID_OCT=$(sbatch \
    --job-name="fsoi_aircraft_oct2025" \
    --time="20:00:00" \
    --output="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_oct2025_%j.out" \
    --error="FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_oct2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --nodes=1 --ntasks-per-node=1 \
    --exclude=u22g09,u22g08,u22g10 \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="2025-10-01",\
FSOI_END_DATE="2025-10-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/aircraft_seasonal/aircraft_oct2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_aircraft.sh --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_OCT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Aircraft-target FSOI seasonal jobs submitted"
echo "============================================================"
echo "  $JID_JAN  ->  Jan 2025 (winter)  -> aircraft_seasonal/aircraft_jan2025/"
echo "  $JID_APR  ->  Apr 2025 (spring)  -> aircraft_seasonal/aircraft_apr2025/"
echo "  $JID_JUL  ->  Jul 2025 (summer)  -> aircraft_seasonal/aircraft_jul2025/"
echo "  $JID_OCT  ->  Oct 2025 (autumn)  -> aircraft_seasonal/aircraft_oct2025/"
echo ""
echo "  Verification target : aircraft (T, q, u, v at flight levels)"
echo "  Walltime per job    : 20 h  (pressure + variable stratification)"
echo "  GPU                 : 1 x H100"
echo "  RAM                 : 250 GB"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f FSOI/fsoi_outputs/aircraft_seasonal/logs/aircraft_jul2025_*.out"
echo ""
echo "After all 4 jobs complete:"
echo "  # Evaluate results"
echo "  for season in jan2025 apr2025 jul2025 oct2025; do"
echo "    python FSOI/evaluate_fsoi_results.py \\"
echo "        --input FSOI/fsoi_outputs/aircraft_seasonal/aircraft_\${season}/csv \\"
echo "        --output FSOI/fsoi_outputs/aircraft_seasonal/aircraft_\${season}/evaluation"
echo "  done"
echo ""
echo "  # Generate paper heatmaps from aircraft runs"
echo "  python FSOI/generate_paper_heatmaps.py \\"
echo "      --output FSOI/fsoi_outputs/paper_figures_aircraft"
echo "============================================================"
