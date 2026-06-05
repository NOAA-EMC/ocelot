#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_enc_order
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=fsoi_enc_order_%j.out
#SBATCH --error=fsoi_enc_order_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# FSOI Encoding-Order Investigation (Hypothesis H1)
# ============================================================================
# Tests whether FSOI rankings depend on the order observations are encoded
# into the shared mesh. Runs the same checkpoint and date range as the
# baseline (default encoding order) but with --encoding_order reversed.
#
# Reversed order: conventional obs encode FIRST (aircraft last → aircraft first,
#                 radiosonde → radiosonde second), satellites encode LAST.
# Default order:  satellite instruments first (ATMS, AMSU-A, ...), then
#                 surface_obs, radiosonde, aircraft.
#
# See: FSOI/ENCODING_ORDER_INVESTIGATION.md — Hypothesis H1
#
# Interpretation:
#   If rankings shift substantially (e.g. radiosonde FSOI drops 2×, ATMS
#   becomes clearly detrimental), encoding position is a first-order confound
#   that needs architectural correction in future model versions.
#   If rankings are stable, the physical signal dominates.
#
# Usage:
#   # Both default and reversed order (compare directly):
#   sbatch FSOI/scripts/run_fsoi_encoding_order_test.sh
#
#   # Or with custom dates/output:
#   sbatch --export=ALL,FSOI_START_DATE=2025-01-01,FSOI_END_DATE=2025-01-31,\
#   ENCODING_ORDER=reversed FSOI/scripts/run_fsoi_encoding_order_test.sh
# ============================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Encoding-Order Investigation (H1 test)"
echo "=================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "=================================================="

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate gnn-env
else
    echo "ERROR: conda not found at ${CONDA_BASE}"; exit 1
fi

# ── Resolve gnn_model working directory ──────────────────────────────────────
find_gnn_model_dir() {
    local d="$1"
    for _ in 1 2 3 4 5 6; do
        [ -f "$d/configs/observation_config.yaml" ] && [ -d "$d/FSOI" ] && { echo "$d"; return 0; }
        d="$(dirname -- "$d")"
    done
    return 1
}

GNN_MODEL_DIR_RESOLVED=""
if [ -n "${GNN_MODEL_DIR:-}" ]; then
    GNN_MODEL_DIR_RESOLVED="$GNN_MODEL_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$SLURM_SUBMIT_DIR")" || true
    [ -z "$GNN_MODEL_DIR_RESOLVED" ] && \
        [ -f "${SLURM_SUBMIT_DIR}/gnn_model/configs/observation_config.yaml" ] && \
        GNN_MODEL_DIR_RESOLVED="${SLURM_SUBMIT_DIR}/gnn_model"
fi
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$PWD")" || true

if [ -z "$GNN_MODEL_DIR_RESOLVED" ] || [ ! -f "${GNN_MODEL_DIR_RESOLVED}/configs/observation_config.yaml" ]; then
    echo "ERROR: Could not locate gnn_model directory."; exit 1
fi
cd "$GNN_MODEL_DIR_RESOLVED"
echo "[PATH] gnn_model dir: $(pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
CONFIG="FSOI/configs/fsoi_config_radiosonde_all.yaml"

# Encoding order to test: "default" or "reversed"
# Override at submit time: sbatch --export=ALL,ENCODING_ORDER=reversed ...
ENCODING_ORDER="${ENCODING_ORDER:-reversed}"

# Date range — use July 2025 to match the primary FSOI analysis period
FSOI_START_DATE="${FSOI_START_DATE:-2025-07-01}"
FSOI_END_DATE="${FSOI_END_DATE:-2025-07-31}"

DATE_TAG="${FSOI_START_DATE//-/}_${FSOI_END_DATE//-/}"
OUTPUT_DIR="FSOI/fsoi_outputs/encoding_order_test/${ENCODING_ORDER}_${DATE_TAG}"
LOG_DIR="FSOI/logs/encoding_order_test"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

[ -f "$CHECKPOINT_PATH" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH"; exit 1; }
[ -d "$DATA_PATH" ]       || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Configuration:"
echo "  Checkpoint:     $CHECKPOINT_PATH"
echo "  Config:         $CONFIG"
echo "  Encoding order: $ENCODING_ORDER"
echo "  Date range:     $FSOI_START_DATE to $FSOI_END_DATE"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Data path:      $DATA_PATH"
echo ""
echo "H1 test: if ATMS FSOI becomes clearly negative (beneficial) and"
echo "         radiosonde FSOI drops substantially after reversal,"
echo "         encoding position is a first-order confound in the rankings."
echo ""

# ── Step 1: Validation ────────────────────────────────────────────────────────
echo "Step 1: Running validation tests..."
python FSOI/test_fsoi.py --checkpoint "$CHECKPOINT_PATH" \
    2>&1 | tee "${LOG_DIR}/test_${ENCODING_ORDER}_${SLURM_JOB_ID}.log"
echo "Validation passed."

# ── Step 2: FSOI computation ──────────────────────────────────────────────────
echo ""
echo "Step 2: Computing FSOI with encoding_order=${ENCODING_ORDER}..."
python FSOI/fsoi_inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG" \
    --data_path "$DATA_PATH" \
    --start_date "$FSOI_START_DATE" \
    --end_date   "$FSOI_END_DATE" \
    --output_dir "$OUTPUT_DIR" \
    --encoding_order "$ENCODING_ORDER" \
    --diagnostics \
    2>&1 | tee "${LOG_DIR}/fsoi_${ENCODING_ORDER}_${SLURM_JOB_ID}.log"
echo "FSOI computation complete."

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
echo ""
echo "Step 3: Evaluating results..."
python FSOI/evaluate_fsoi_results.py \
    --input "$OUTPUT_DIR/csv" \
    --output "$OUTPUT_DIR/evaluation" \
    2>&1 | tee "${LOG_DIR}/eval_${ENCODING_ORDER}_${SLURM_JOB_ID}.log" || true

# ── Step 4: Quick comparison printout ────────────────────────────────────────
echo ""
echo "=================================================="
echo "Quick comparison: $ENCODING_ORDER encoding order"
echo "=================================================="
if [ -f "$OUTPUT_DIR/evaluation/fsoi_evaluation_summary.csv" ]; then
    echo "Instrument rankings (sum_impact_scaled, negative=beneficial):"
    python -c "
import pandas as pd, sys
df = pd.read_csv('${OUTPUT_DIR}/evaluation/fsoi_evaluation_summary.csv')
df = df[['instrument','impact_total','positive_frac_mean']].sort_values('impact_total')
df['impact_total'] = df['impact_total'].round(1)
df['positive_frac_mean'] = df['positive_frac_mean'].round(3)
print(df.to_string(index=False))
print()
print('Compare with default-order baseline in:')
print('  FSOI/fsoi_outputs/full_eval_fixed_20250701_20250731/radiosonde/evaluation/')
" 2>/dev/null || head -n 20 "$OUTPUT_DIR/evaluation/fsoi_evaluation_summary.csv"
fi

echo ""
echo "=================================================="
echo "Job finished: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=================================================="
