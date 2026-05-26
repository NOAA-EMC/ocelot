#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_ose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=14:00:00
#SBATCH --output=fsoi_ose_%j.out
#SBATCH --error=fsoi_ose_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# FSOI: Observation System Experiment (OSE) — Instrument Denial
#
# Replaces xa[inst] → xb[inst] for the denied instrument, re-runs forward
# pass, and compares ea_denied vs ea_control.  fix commit 7f04963 ensures
# replace_batch_inputs writes to the correct observation columns (not geo
# encoding).  All pre-fix OSE results are invalid and have been deleted.
#
# Environment variables (override defaults):
#   OSE_INSTRUMENTS     space-separated instrument names to deny (default: atms)
#   CHECKPOINT_PATH     path to .ckpt file
#   FSOI_START_DATE     start date YYYY-MM-DD
#   FSOI_END_DATE       end date YYYY-MM-DD
#   FSOI_OUTPUT_DIR     output directory
#   DATA_PATH           path to v7 dataset
#   CONFIG_FILE         FSOI config yaml (default: fsoi_config_radiosonde_all.yaml)
#
# Usage:
#   sbatch FSOI/scripts/run_fsoi_ose.sh
#   OSE_INSTRUMENTS="atms amsua" sbatch FSOI/scripts/run_fsoi_ose.sh
# ============================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Observation System Experiment (OSE)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=================================================="

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate gnn-env

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
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && GNN_DIR="$(find_gnn_model_dir "$SLURM_SUBMIT_DIR")"; then
    :
elif GNN_DIR="$(find_gnn_model_dir "$PWD")"; then
    :
else
    echo "ERROR: cannot resolve gnn_model root"; exit 1
fi

cd "$GNN_DIR"
echo "[PATH] Working dir: $(pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT="${CHECKPOINT_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
CONFIG_FILE="${CONFIG_FILE:-FSOI/configs/fsoi_config_radiosonde_all.yaml}"

# Parse denied instruments (default: atms)
OSE_INSTRUMENTS="${OSE_INSTRUMENTS:-atms}"

# Read dates from config if not overridden
read -r CONFIG_START CONFIG_END <<< "$(python -c '
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
d = cfg.get("data", {})
print(d.get("start_date", ""), d.get("end_date", ""))
' "$CONFIG_FILE")"

START_DATE="${FSOI_START_DATE:-$CONFIG_START}"
END_DATE="${FSOI_END_DATE:-$CONFIG_END}"
DATE_TAG="${START_DATE//-/}_${END_DATE//-/}"

# Build output dir name from denied instruments
INST_TAG="${OSE_INSTRUMENTS// /_}"
OUTPUT_DIR="${FSOI_OUTPUT_DIR:-FSOI/fsoi_outputs/ose_${INST_TAG}_${DATE_TAG}}"

# ── Validation ────────────────────────────────────────────────────────────────
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }

mkdir -p "$OUTPUT_DIR" FSOI/logs

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "[CKPT]    $CKPT"
echo "[CONFIG]  $CONFIG_FILE"
echo "[DATES]   $START_DATE → $END_DATE"
echo "[DENIED]  $OSE_INSTRUMENTS"
echo "[OUTPUT]  $OUTPUT_DIR"
echo ""

# ── Step 1: Quick validation ──────────────────────────────────────────────────
echo "=== Step 1: Validation tests ==="
python FSOI/test_fsoi.py --checkpoint "$CKPT" \
    2>&1 | tee "FSOI/logs/test_fsoi_ose_${SLURM_JOB_ID}.log"
echo "✓ Validation passed"

# ── Step 2: OSE run ──────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: OSE inference ==="
# Build --ose_instruments argument (space-separated → multiple args)
OSE_ARGS=""
for inst in $OSE_INSTRUMENTS; do
    OSE_ARGS="$OSE_ARGS $inst"
done

python FSOI/fsoi_inference.py \
    --checkpoint    "$CKPT" \
    --config        "$CONFIG_FILE" \
    --data_path     "$DATA_PATH" \
    --start_date    "$START_DATE" \
    --end_date      "$END_DATE" \
    --output_dir    "$OUTPUT_DIR" \
    --diagnostics \
    --ose_instruments $OSE_ARGS \
    2>&1 | tee "FSOI/logs/fsoi_ose_${SLURM_JOB_ID}.log"

echo "✓ OSE inference complete"

# ── Step 3: Quick summary ─────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Results ==="
OSE_CSV="$OUTPUT_DIR/evaluation/ose_results.csv"
if [ -f "$OSE_CSV" ]; then
    echo "OSE results (first 20 rows):"
    head -21 "$OSE_CSV"
else
    echo "WARNING: $OSE_CSV not found — check log"
    grep -E "OSE|ose_impact|ea_denied|ea_control" \
         "FSOI/logs/fsoi_ose_${SLURM_JOB_ID}.log" | tail -30 || true
fi

echo ""
echo "=================================================="
echo "OSE run finished: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Denied: $OSE_INSTRUMENTS"
echo "=================================================="
