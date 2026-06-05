#!/bin/bash
# ============================================================================
# Submit 4 seasonal surface-obs-target FSOI jobs (Jan, Apr, Jul, Oct 2025).
#
# Objective: measure how ALL observation types impact forecast error scored
#            at SURFACE OBS verification targets (2-m T, q, 10-m u/v, ps).
#
# Key differences from aircraft/radiosonde runs:
#   - target_instruments: ["surface_obs"]
#   - stratify_by_pressure: false  (surface obs have no vertical structure)
#   - stratify_by_variable: true   (5 variables: T, q, u, v, ps)
#   - Faster than aircraft (no pressure stratification) -> 16h walltime
#
# Output dirs:
#   FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_jan2025/
#   FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_apr2025/
#   FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_jul2025/
#   FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_oct2025/
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_surface_obs_seasonal.sh [/path/to/checkpoint.ckpt]
# ============================================================================

set -e

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/configs/observation_config.yaml" ]; then
    GNN_MODEL_DIR="$SLURM_SUBMIT_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/gnn_model/configs/observation_config.yaml" ]; then
    GNN_MODEL_DIR="${SLURM_SUBMIT_DIR}/gnn_model"
else
    GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [ ! -f "${GNN_MODEL_DIR}/configs/observation_config.yaml" ]; then
    echo "ERROR: Could not locate gnn_model directory."
    echo "  Run this script interactively: bash FSOI/scripts/submit_fsoi_surface_obs_seasonal.sh"
    echo "  Do NOT submit it with sbatch -- it is a submission orchestrator, not a job script."
    exit 1
fi

cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

CHECKPOINT_PATH="${1:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
CONFIG="FSOI/configs/fsoi_config_surface_obs.yaml"
CKPT="$CHECKPOINT_PATH"

[ -f "$CKPT" ]       || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ]  || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }
[ -f "$CONFIG" ]     || { echo "ERROR: config not found: $CONFIG"; exit 1; }

echo "[SUBMIT] Checkpoint : $CKPT"
echo "[SUBMIT] Data path  : $DATA_PATH"
echo "[SUBMIT] Config     : $CONFIG"
echo ""

AW=$(grep "use_area_weights" "$CONFIG" | head -1)
echo "[CONFIG CHECK] $AW"
echo ""

mkdir -p FSOI/fsoi_outputs/surface_obs_seasonal/logs

# ── Jan 2025 (winter) ─────────────────────────────────────────────────────────
echo "Submitting Jan 2025 (winter)..."
JID_JAN=$(sbatch \
    --job-name="fsoi_surface_jan2025" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_jan2025_%j.out" \
    --error="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_jan2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-01-01",\
FSOI_END_DATE="2025-01-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_jan2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_surface_obs.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_JAN"

# ── Apr 2025 (spring) ─────────────────────────────────────────────────────────
echo "Submitting Apr 2025 (spring)..."
JID_APR=$(sbatch \
    --job-name="fsoi_surface_apr2025" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_apr2025_%j.out" \
    --error="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_apr2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-04-01",\
FSOI_END_DATE="2025-04-30",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_apr2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_surface_obs.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_APR"

# ── Jul 2025 (summer) ─────────────────────────────────────────────────────────
echo "Submitting Jul 2025 (summer)..."
JID_JUL=$(sbatch \
    --job-name="fsoi_surface_jul2025" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_jul2025_%j.out" \
    --error="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_jul2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-07-01",\
FSOI_END_DATE="2025-07-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_jul2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_surface_obs.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_JUL"

# ── Oct 2025 (autumn) ─────────────────────────────────────────────────────────
echo "Submitting Oct 2025 (autumn)..."
JID_OCT=$(sbatch \
    --job-name="fsoi_surface_oct2025" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_oct2025_%j.out" \
    --error="FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_oct2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-10-01",\
FSOI_END_DATE="2025-10-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_oct2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_surface_obs.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_OCT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Surface-obs-target FSOI seasonal jobs submitted"
echo "============================================================"
echo "  $JID_JAN  ->  Jan 2025 (winter)  -> surface_obs_seasonal/surface_obs_jan2025/"
echo "  $JID_APR  ->  Apr 2025 (spring)  -> surface_obs_seasonal/surface_obs_apr2025/"
echo "  $JID_JUL  ->  Jul 2025 (summer)  -> surface_obs_seasonal/surface_obs_jul2025/"
echo "  $JID_OCT  ->  Oct 2025 (autumn)  -> surface_obs_seasonal/surface_obs_oct2025/"
echo ""
echo "  Verification target : surface_obs (2-m T, q, 10-m u/v, ps)"
echo "  Walltime per job    : 16 h  (variable stratification, no pressure levels)"
echo "  GPU                 : 1 x H100"
echo "  RAM                 : 250 GB"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f FSOI/fsoi_outputs/surface_obs_seasonal/logs/surface_jul2025_*.out"
echo ""
echo "After all 4 jobs complete:"
echo "  for season in jan2025 apr2025 jul2025 oct2025; do"
echo "    python FSOI/evaluate_fsoi_results.py \\"
echo "        --input FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_\${season}/csv \\"
echo "        --output FSOI/fsoi_outputs/surface_obs_seasonal/surface_obs_\${season}/evaluation"
echo "  done"
echo "============================================================"
