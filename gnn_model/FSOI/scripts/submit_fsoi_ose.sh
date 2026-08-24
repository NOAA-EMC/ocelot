#!/bin/bash
# ============================================================================
# Submit OSE (Observation System Experiment) ATMS denial jobs
#
# Submits two jobs:
#   1. Jul 2025 (summer)  — ATMS denial, radiosonde verification target
#   2. Jan 2025 (winter)  — ATMS denial, radiosonde verification target
#
# These use the fixed replace_batch_inputs (commit 7f04963) which writes
# denied observations to the correct observation columns.  All pre-fix OSE
# results were invalid (were scrambling geo-encoding columns, not obs).
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_ose.sh [--checkpoint PATH] [--instruments "atms amsua"] [--denial-mode full_mask] [--channels "ssmis:21"]
# ============================================================================

set -e

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────
CKPT="/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
OSE_INSTRUMENTS="atms"
OSE_DENIAL_MODE="${OSE_DENIAL_MODE:-background_replacement}"
OSE_CHANNELS="${OSE_CHANNELS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CKPT="$2"; shift 2 ;;
        --checkpoint=*)  CKPT="${1#*=}"; shift ;;
        --instruments)   OSE_INSTRUMENTS="$2"; shift 2 ;;
        --denial-mode)   OSE_DENIAL_MODE="$2"; shift 2 ;;
        --denial-mode=*) OSE_DENIAL_MODE="${1#*=}"; shift ;;
        --channels|--ose-channels) OSE_CHANNELS="$2"; shift 2 ;;
        --channels=*|--ose-channels=*) OSE_CHANNELS="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }

INST_TAG="${OSE_INSTRUMENTS// /_}"
MODE_SUFFIX=""
if [ "$OSE_DENIAL_MODE" != "background_replacement" ]; then
    MODE_SUFFIX="_${OSE_DENIAL_MODE}"
fi
CHANNEL_SUFFIX=""
if [ -n "$OSE_CHANNELS" ]; then
    CHANNEL_SUFFIX="_ch_${OSE_CHANNELS}"
    CHANNEL_SUFFIX="${CHANNEL_SUFFIX// /_}"
    CHANNEL_SUFFIX="${CHANNEL_SUFFIX//:/_}"
    CHANNEL_SUFFIX="${CHANNEL_SUFFIX//,/_}"
    CHANNEL_SUFFIX="${CHANNEL_SUFFIX//=/}"
fi
CONFIG="FSOI/configs/fsoi_config_radiosonde_all.yaml"
LOG_ROOT="FSOI/fsoi_outputs/ose_logs"
mkdir -p "$LOG_ROOT"

echo "[SUBMIT] Checkpoint   : $CKPT"
echo "[SUBMIT] Denied insts : $OSE_INSTRUMENTS"
echo "[SUBMIT] Denial mode  : $OSE_DENIAL_MODE"
if [ -n "$OSE_CHANNELS" ]; then
    echo "[SUBMIT] Channels     : $OSE_CHANNELS"
fi
echo ""

# ── Submit Jul 2025 ────────────────────────────────────────────────────────
echo "Submitting OSE Jul 2025 (summer)..."
JID_JUL=$(sbatch \
    --job-name="fsoi_ose_${INST_TAG}${CHANNEL_SUFFIX}${MODE_SUFFIX}_jul2025" \
    --time="14:00:00" \
    --output="${LOG_ROOT}/ose_${INST_TAG}_jul2025_%j.out" \
    --error="${LOG_ROOT}/ose_${INST_TAG}_jul2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-07-01",\
FSOI_END_DATE="2025-07-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/ose_${INST_TAG}${CHANNEL_SUFFIX}${MODE_SUFFIX}_jul2025_fixed",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
OSE_INSTRUMENTS="$OSE_INSTRUMENTS",\
OSE_DENIAL_MODE="$OSE_DENIAL_MODE",\
OSE_CHANNELS="$OSE_CHANNELS",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_ose.sh \
    | awk '{print $NF}')
echo "  Job ID: $JID_JUL"

# ── Submit Jan 2025 ────────────────────────────────────────────────────────
echo "Submitting OSE Jan 2025 (winter)..."
JID_JAN=$(sbatch \
    --job-name="fsoi_ose_${INST_TAG}${CHANNEL_SUFFIX}${MODE_SUFFIX}_jan2025" \
    --time="14:00:00" \
    --output="${LOG_ROOT}/ose_${INST_TAG}_jan2025_%j.out" \
    --error="${LOG_ROOT}/ose_${INST_TAG}_jan2025_%j.err" \
    -A gpu-ai4wp -p u1-h100 -q gpu --gres=gpu:h100:1 --mem=250G \
    --export=ALL,\
CHECKPOINT_PATH="$CKPT",\
FSOI_START_DATE="2025-01-01",\
FSOI_END_DATE="2025-01-31",\
FSOI_OUTPUT_DIR="FSOI/fsoi_outputs/ose_${INST_TAG}${CHANNEL_SUFFIX}${MODE_SUFFIX}_jan2025_fixed",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="$CONFIG",\
OSE_INSTRUMENTS="$OSE_INSTRUMENTS",\
OSE_DENIAL_MODE="$OSE_DENIAL_MODE",\
OSE_CHANNELS="$OSE_CHANNELS",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_ose.sh \
    | awk '{print $NF}')
echo "  Job ID: $JID_JAN"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "OSE jobs submitted"
echo "=============================================="
echo "  $JID_JUL  →  Jul 2025  →  ose_${INST_TAG}_jul2025_fixed/"
echo "  $JID_JAN  →  Jan 2025  →  ose_${INST_TAG}_jan2025_fixed/"
echo ""
echo "  Denied instruments: $OSE_INSTRUMENTS"
echo "  Denial mode: $OSE_DENIAL_MODE"
if [ -n "$OSE_CHANNELS" ]; then
    echo "  Channels: $OSE_CHANNELS"
fi
echo "  Fix applied: replace_batch_inputs writes to observation columns"
echo "               (not geo/time encoding) — commit 7f04963"
echo ""
echo "  Monitor: squeue -u $USER"
echo ""
echo "  After completion, compare with FSOI rankings:"
echo "    python FSOI/fsoi_ose.py compare \\"
echo "      --ose FSOI/fsoi_outputs/ose_${INST_TAG}${CHANNEL_SUFFIX}${MODE_SUFFIX}_jul2025_fixed \\"
echo "      --fsoi FSOI/fsoi_outputs/full_eval_fixed_20250701_20250731/radiosonde"
echo "=============================================="
