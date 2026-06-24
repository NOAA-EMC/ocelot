#!/bin/bash
# ============================================================================
# Re-run 3 seasonal periods with all fixes applied:
#   - use_area_weights: true  (changes FSOI gradients — required for paper)
#   - stratified spatial subsampling  (fixes AVHRR/ATMS scale bias)
#   - --diagnostics  (innovation quality)
#   - FD check on 3 dates
#
# Target: radiosonde (T, Td, u, v) — same target as old runs for direct
#         comparison of area-weighted vs non-area-weighted rankings.
#
# Runs: Jan 2025 (winter), Apr 2025 (spring), Oct 2025 (fall)
# Jul 2025 completed as job 14152885 → full_eval_fixed_20250701_20250731/
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_seasonal_rerun.sh [--checkpoint PATH]
# ============================================================================

set -e

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

CHECKPOINT_PATH="${1:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
CONFIG="FSOI/configs/fsoi_config_radiosonde_all.yaml"
CKPT="$CHECKPOINT_PATH"

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }

echo "[SUBMIT] Checkpoint : $CKPT"
echo "[SUBMIT] Data path  : $DATA_PATH"
echo ""

# ── Verify the config now has area weighting enabled ─────────────────────────
AW=$(grep "use_area_weights" "$CONFIG" | head -1)
if echo "$AW" | grep -q "false"; then
    echo "ERROR: $CONFIG still has use_area_weights: false"
    echo "  The config must have been updated. Check git status."
    exit 1
fi
echo "[OK] $CONFIG has: $AW"
echo ""

mkdir -p FSOI/fsoi_outputs/seasonal_fixed/logs

# ── Submit 3 seasonal jobs ────────────────────────────────────────────────────

echo "Submitting Jan 2025 (winter)..."
JID_JAN=$(sbatch \
    --job-name="fsoi_jan2025_fixed" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_jan2025_%j.out" \
    --error="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_jan2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-01-01",\
FSOI_END_DATE="2025-01-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/seasonal_fixed/radiosonde_jan2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_radiosonde_all.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_JAN"

echo "Submitting Apr 2025 (spring)..."
JID_APR=$(sbatch \
    --job-name="fsoi_apr2025_fixed" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_apr2025_%j.out" \
    --error="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_apr2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-04-01",\
FSOI_END_DATE="2025-04-30",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/seasonal_fixed/radiosonde_apr2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_radiosonde_all.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_APR"

echo "Submitting Oct 2025 (fall)..."
JID_OCT=$(sbatch \
    --job-name="fsoi_oct2025_fixed" \
    --time="16:00:00" \
    --output="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_oct2025_%j.out" \
    --error="FSOI/fsoi_outputs/seasonal_fixed/logs/fsoi_oct2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-10-01",\
FSOI_END_DATE="2025-10-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/seasonal_fixed/radiosonde_oct2025",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_radiosonde_all.sh --checkpoint "$CKPT" \
    | awk '{print $NF}')
echo "  Job ID: $JID_OCT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Seasonal re-runs submitted (with all fixes)"
echo "=============================================="
echo "  $JID_JAN  →  Jan 2025 (winter)   → seasonal_fixed/radiosonde_jan2025/"
echo "  $JID_APR  →  Apr 2025 (spring)   → seasonal_fixed/radiosonde_apr2025/"
echo "  $JID_OCT  →  Oct 2025 (fall)     → seasonal_fixed/radiosonde_oct2025/"
echo ""
echo "  Fixes applied vs old runs:"
echo "    [x] use_area_weights: true  (changes FSOI gradients)"
echo "    [x] stratified spatial subsampling  (fixes AVHRR/ATMS scale bias)"
echo "    [x] --diagnostics  (innovation quality output)"
echo "    [x] FD gradient check  (10 samples × 3 dates)"
echo ""
echo "  Jul 2025 completed: job 14152885 → full_eval_fixed_20250701_20250731/"
echo ""
echo "  After all 4 seasons complete, re-compute weights:"
echo "    conda run -n gnn-env python FSOI/compute_fsoi_weights.py \\"
echo "        --fsoi_dirs FSOI/fsoi_outputs/seasonal_fixed/*/csv \\"
echo "                    FSOI/fsoi_outputs/full_eval_fixed_20250701_20250731/radiosonde/csv \\"
echo "        --obs_config configs/observation_config.yaml \\"
echo "        --output_dir FSOI/fsoi_weights/all_seasons_fixed"
echo "=============================================="
echo ""
echo "Monitor: squeue -u $USER"
