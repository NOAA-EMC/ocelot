#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-emc-ai # gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_ose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
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
# Set OSE_DENIAL_MODE=full_mask for whole-instrument input masking.
#
# Environment variables (override defaults):
#   OSE_INSTRUMENTS     space-separated instrument names to deny (default: atms)
#   OSE_DENIAL_MODE     background_replacement, sample_mask, or full_mask
#                        (default: background_replacement)
#   OSE_CHANNELS        optional channel selectors, e.g. "ssmis:21"
#   CHECKPOINT_PATH     path to .ckpt file
#   FSOI_START_DATE     start date YYYY-MM-DD
#   FSOI_END_DATE       end date YYYY-MM-DD
#   FSOI_OUTPUT_DIR     output directory
#   DATA_PATH           path to v7 dataset
#   CONFIG_FILE         FSOI config yaml (default: fsoi_config_radiosonde_all.yaml)
#   FSOI_VERIFICATION_TARGET  obs or mesh (default: obs)
#   GFS_ROOT            GFS analysis root for mesh verification
#   MESH_INSTRUMENT     radiosonde or surface_obs (default: radiosonde)
#   MESH_PRESSURE_LEVEL_IDX pressure index for mesh radiosonde verification (default: 4 = 500 hPa)
#   OSE_SAVE_SPATIAL_FIELDS 1 to save per-node full-minus-denied OSE fields
#   OSE_SPATIAL_PAIR_INDICES space-separated pair indices to save (default: 0)
#   OSE_PATH_INTEGRATION_PAIR_INDICES optional space-separated pair indices for
#                        multi-point matched path-integration diagnostics
#   OSE_PATH_INTEGRATION_T_VALUES optional t values (default: "0 0.25 0.5 0.75 1")
#
# Usage:
#   sbatch FSOI/scripts/run_fsoi_ose.sh
#   OSE_INSTRUMENTS="atms amsua" sbatch FSOI/scripts/run_fsoi_ose.sh
#   FSOI_VERIFICATION_TARGET=mesh OSE_SAVE_SPATIAL_FIELDS=1 sbatch FSOI/scripts/run_fsoi_ose.sh
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
FSOI_VERIFICATION_TARGET="${FSOI_VERIFICATION_TARGET:-obs}"
GFS_ROOT="${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}"
MESH_INSTRUMENT="${MESH_INSTRUMENT:-radiosonde}"
MESH_PRESSURE_LEVEL_IDX="${MESH_PRESSURE_LEVEL_IDX:-4}"
OSE_SAVE_SPATIAL_FIELDS="${OSE_SAVE_SPATIAL_FIELDS:-0}"
OSE_SPATIAL_PAIR_INDICES="${OSE_SPATIAL_PAIR_INDICES:-0}"
OSE_DENIAL_MODE="${OSE_DENIAL_MODE:-background_replacement}"
OSE_CHANNELS="${OSE_CHANNELS:-}"
OSE_PATH_INTEGRATION_PAIR_INDICES="${OSE_PATH_INTEGRATION_PAIR_INDICES:-}"
OSE_PATH_INTEGRATION_T_VALUES="${OSE_PATH_INTEGRATION_T_VALUES:-0 0.25 0.5 0.75 1}"
OSE_PATH_INTEGRATION_PAIR_INDICES="${OSE_PATH_INTEGRATION_PAIR_INDICES//,/ }"
OSE_PATH_INTEGRATION_PAIR_INDICES="${OSE_PATH_INTEGRATION_PAIR_INDICES//:/ }"
OSE_PATH_INTEGRATION_PAIR_INDICES="${OSE_PATH_INTEGRATION_PAIR_INDICES//;/ }"
OSE_PATH_INTEGRATION_T_VALUES="${OSE_PATH_INTEGRATION_T_VALUES//,/ }"
OSE_PATH_INTEGRATION_T_VALUES="${OSE_PATH_INTEGRATION_T_VALUES//:/ }"
OSE_PATH_INTEGRATION_T_VALUES="${OSE_PATH_INTEGRATION_T_VALUES//;/ }"

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
CHANNEL_TAG=""
if [ -n "$OSE_CHANNELS" ]; then
    CHANNEL_TAG="_ch_${OSE_CHANNELS}"
    CHANNEL_TAG="${CHANNEL_TAG// /_}"
    CHANNEL_TAG="${CHANNEL_TAG//:/_}"
    CHANNEL_TAG="${CHANNEL_TAG//,/_}"
    CHANNEL_TAG="${CHANNEL_TAG//=/}"
fi
OUTPUT_DIR="${FSOI_OUTPUT_DIR:-FSOI/fsoi_outputs/ose_${INST_TAG}${CHANNEL_TAG}_${DATE_TAG}}"

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
echo "[MODE]    $OSE_DENIAL_MODE"
if [ -n "$OSE_CHANNELS" ]; then
    echo "[CHANNEL] $OSE_CHANNELS"
fi
echo "[VERIFY]  $FSOI_VERIFICATION_TARGET"
if [ "$FSOI_VERIFICATION_TARGET" = "mesh" ]; then
    echo "[MESH]    instrument=$MESH_INSTRUMENT pressure_idx=$MESH_PRESSURE_LEVEL_IDX"
    echo "[GFS]     $GFS_ROOT"
fi
if [ "$OSE_SAVE_SPATIAL_FIELDS" = "1" ]; then
    echo "[SPATIAL] save pairs: $OSE_SPATIAL_PAIR_INDICES"
fi
if [ -n "$OSE_PATH_INTEGRATION_PAIR_INDICES" ]; then
    echo "[PATH]    path-integration pairs: $OSE_PATH_INTEGRATION_PAIR_INDICES"
    echo "[PATH]    t values: $OSE_PATH_INTEGRATION_T_VALUES"
fi
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
OSE_CHANNEL_ARGS=""
if [ -n "$OSE_CHANNELS" ]; then
    for ch in $OSE_CHANNELS; do
        OSE_CHANNEL_ARGS="$OSE_CHANNEL_ARGS $ch"
    done
fi

EXTRA_ARGS="--verification_target $FSOI_VERIFICATION_TARGET"
EXTRA_ARGS="$EXTRA_ARGS --ose_denial_mode $OSE_DENIAL_MODE"
if [ -n "$OSE_CHANNEL_ARGS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --ose_channels $OSE_CHANNEL_ARGS"
fi
if [ "$FSOI_VERIFICATION_TARGET" = "mesh" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --gfs_root $GFS_ROOT"
    EXTRA_ARGS="$EXTRA_ARGS --mesh_instrument $MESH_INSTRUMENT"
    EXTRA_ARGS="$EXTRA_ARGS --mesh_pressure_level_idx $MESH_PRESSURE_LEVEL_IDX"
fi
if [ "$OSE_SAVE_SPATIAL_FIELDS" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --ose_save_spatial_fields"
    EXTRA_ARGS="$EXTRA_ARGS --ose_spatial_pair_indices $OSE_SPATIAL_PAIR_INDICES"
fi
if [ -n "$OSE_PATH_INTEGRATION_PAIR_INDICES" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --ose_path_integration_pair_indices $OSE_PATH_INTEGRATION_PAIR_INDICES"
    EXTRA_ARGS="$EXTRA_ARGS --ose_path_integration_t_values $OSE_PATH_INTEGRATION_T_VALUES"
fi
echo "[ARGS]    $EXTRA_ARGS"

python FSOI/fsoi_inference.py \
    --checkpoint    "$CKPT" \
    --config        "$CONFIG_FILE" \
    --data_path     "$DATA_PATH" \
    --start_date    "$START_DATE" \
    --end_date      "$END_DATE" \
    --output_dir    "$OUTPUT_DIR" \
    --diagnostics \
    $EXTRA_ARGS \
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
    if [ -n "$OSE_PATH_INTEGRATION_PAIR_INDICES" ]; then
        if head -1 "$OSE_CSV" | grep -q "path_integrated_fsoi"; then
            echo "[PATH] Path-integration columns found in $OSE_CSV"
        else
            echo "ERROR: Path integration was requested for pairs '$OSE_PATH_INTEGRATION_PAIR_INDICES',"
            echo "       but $OSE_CSV does not contain path_integrated_fsoi."
            echo "       Check that the submitted checkout includes the latest fsoi_inference.py"
            echo "       and fsoi_ose.py, and avoid comma-separated values inside sbatch --export."
            exit 2
        fi
    fi
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
echo "Mode: $OSE_DENIAL_MODE"
if [ -n "$OSE_CHANNELS" ]; then
    echo "Channels: $OSE_CHANNELS"
fi
echo "=================================================="
