#!/bin/bash
# ============================================================================
# FSOI Full Evaluation — Master Submission Script
# ============================================================================
# Submits three independent SLURM jobs (one per verification target type):
#   1. Radiosonde  — T, Td, u, v  (pressure-stratified + variable-stratified)
#   2. Aircraft    — T, q, u, v   (pressure-stratified + variable-stratified)
#   3. Surface obs — T, q, u, v, ps (variable-stratified, no pressure)
#
# All jobs share:
#   - Area weighting (cosine-latitude)
#   - Finite-difference gradient check (10 samples × 3 dates = 30 triples)
#   - Innovation diagnostics (--diagnostics flag)
#   - Gridded FSOI maps from scatter_samples.csv
#   - Per-level closure checks with PASS/WARN/FAIL flags
#   - Sign consistency time series per instrument
#   - Stratified spatial subsampling (geographic bias fix)
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_full_evaluation.sh [--checkpoint PATH] \
#        [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--output_root DIR]
#
# Defaults:
#   checkpoint : last run checkpoint (Epoch3079_fixedval.ckpt)
#   start      : 2025-07-01
#   end        : 2025-07-31  (30 days = 60 pairs at 12h)
#   output_root: FSOI/fsoi_outputs/full_eval_<DATE_TAG>
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

# ── Parse arguments ──────────────────────────────────────────────────────────
CHECKPOINT_PATH=""
START_DATE="2025-07-01"
END_DATE="2025-07-31"
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT_PATH="$2"; shift 2 ;;
        --checkpoint=*)  CHECKPOINT_PATH="${1#*=}"; shift ;;
        --start|-s)      START_DATE="$2"; shift 2 ;;
        --start=*)       START_DATE="${1#*=}"; shift ;;
        --end|-e)        END_DATE="$2"; shift 2 ;;
        --end=*)         END_DATE="${1#*=}"; shift ;;
        --output_root)   OUTPUT_ROOT="$2"; shift 2 ;;
        --output_root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
        *) echo "[SUBMIT] Unknown argument: $1"; shift ;;
    esac
done

# ── Checkpoint resolution ─────────────────────────────────────────────────────
DEFAULT_CKPT="/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$DEFAULT_CKPT}"

if [ ! -f "$CHECKPOINT_PATH" ] && [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
    echo "  Provide via: --checkpoint /path/to/model.ckpt"
    exit 1
fi
echo "[SUBMIT] Checkpoint  : $CHECKPOINT_PATH"
echo "[SUBMIT] Date range  : $START_DATE → $END_DATE"

# ── Output directory ──────────────────────────────────────────────────────────
DATE_TAG="${START_DATE//-/}_${END_DATE//-/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-FSOI/fsoi_outputs/full_eval_${DATE_TAG}}"
mkdir -p "$OUTPUT_ROOT"
echo "[SUBMIT] Output root : $OUTPUT_ROOT"

DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] DATA_PATH not accessible: $DATA_PATH"
    exit 1
fi
echo "[SUBMIT] Data path   : $DATA_PATH"
echo ""

# ── Verify configs exist ──────────────────────────────────────────────────────
for cfg in \
    FSOI/configs/fsoi_config_radiosonde_all.yaml \
    FSOI/configs/fsoi_config_aircraft.yaml \
    FSOI/configs/fsoi_config_surface_obs.yaml; do
    [ -f "$cfg" ] || { echo "[ERROR] Config not found: $cfg"; exit 1; }
done

# ── Walltimes ─────────────────────────────────────────────────────────────────
# Radiosonde/aircraft: 64 backward passes/pair × 60 pairs + FD × 30 = ~14 h on H100
# Surface obs        :  4 backward passes/pair × 60 pairs + FD × 30 =  ~5 h on H100
WALLTIME_HEAVY="16:00:00"
WALLTIME_SURFACE="08:00:00"

# ── Job 1: Radiosonde ─────────────────────────────────────────────────────────
echo "=============================================="
echo "Submitting Job 1: Radiosonde (T, Td, u, v)"
echo "=============================================="
JID_SONDE=$(sbatch \
    --job-name="fsoi_sonde_${DATE_TAG}" \
    --time="$WALLTIME_HEAVY" \
    --output="${OUTPUT_ROOT}/logs/fsoi_radiosonde_%j.out" \
    --error="${OUTPUT_ROOT}/logs/fsoi_radiosonde_%j.err" \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="$START_DATE",\
FSOI_END_DATE="$END_DATE",\
FSOI_OUTPUT_DIR="${OUTPUT_ROOT}/radiosonde",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="FSOI/configs/fsoi_config_radiosonde_all.yaml",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_radiosonde_all.sh \
    --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_SONDE"

# ── Job 2: Aircraft ───────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Submitting Job 2: Aircraft (T, q, u, v)"
echo "=============================================="
JID_AIRCRAFT=$(sbatch \
    --job-name="fsoi_aircraft_${DATE_TAG}" \
    --time="$WALLTIME_HEAVY" \
    --output="${OUTPUT_ROOT}/logs/fsoi_aircraft_%j.out" \
    --error="${OUTPUT_ROOT}/logs/fsoi_aircraft_%j.err" \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="$START_DATE",\
FSOI_END_DATE="$END_DATE",\
FSOI_OUTPUT_DIR="${OUTPUT_ROOT}/aircraft",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="FSOI/configs/fsoi_config_aircraft.yaml",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_aircraft.sh \
    --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_AIRCRAFT"

# ── Job 3: Surface obs ────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Submitting Job 3: Surface obs (T, q, u, v, ps)"
echo "=============================================="
JID_SURFACE=$(sbatch \
    --job-name="fsoi_surface_${DATE_TAG}" \
    --time="$WALLTIME_SURFACE" \
    --output="${OUTPUT_ROOT}/logs/fsoi_surface_obs_%j.out" \
    --error="${OUTPUT_ROOT}/logs/fsoi_surface_obs_%j.err" \
    --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
FSOI_START_DATE="$START_DATE",\
FSOI_END_DATE="$END_DATE",\
FSOI_OUTPUT_DIR="${OUTPUT_ROOT}/surface_obs",\
DATA_PATH="$DATA_PATH",\
CONFIG_FILE="FSOI/configs/fsoi_config_surface_obs.yaml",\
GNN_MODEL_DIR="$(pwd)" \
    FSOI/scripts/run_fsoi_surface_obs.sh \
    --checkpoint "$CHECKPOINT_PATH" \
    | awk '{print $NF}')
echo "  Job ID: $JID_SURFACE"

# ── Summary ───────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_ROOT}/logs"
echo ""
echo "=============================================="
echo "All 3 jobs submitted"
echo "=============================================="
echo ""
echo "  Job ${JID_SONDE}    → radiosonde   (T, Td, u, v)"
echo "  Job ${JID_AIRCRAFT} → aircraft     (T, q, u, v)"
echo "  Job ${JID_SURFACE}  → surface obs  (T, q, u, v, ps)"
echo ""
echo "  Output root : $OUTPUT_ROOT"
echo "  Date range  : $START_DATE → $END_DATE (30 days)"
echo ""
echo "  Features enabled in all jobs:"
echo "    [x] Area weighting (cosine-latitude)"
echo "    [x] Finite-difference gradient check (10 samples × 3 dates)"
echo "    [x] Innovation diagnostics (--diagnostics)"
echo "    [x] Gridded FSOI maps (plot_fsoi_maps.py)"
echo "    [x] Per-level closure check (PASS/WARN/FAIL flags)"
echo "    [x] Sign consistency time series"
echo "    [x] Stratified spatial subsampling (geographic bias fix)"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f ${OUTPUT_ROOT}/logs/fsoi_radiosonde_${JID_SONDE}.out"
echo ""
echo "Expected output files per target type (e.g. radiosonde/):"
echo "  csv/fsoi_by_instrument.csv        — per-instrument impact totals"
echo "  csv/fsoi_by_channel.csv           — per-channel breakdown"
echo "  csv/scatter_samples.csv           — (lat, lon, fsoi, innovation) samples"
echo "  evaluation/fd_validation.csv      — FD gradient check per date"
echo "  evaluation/innovation_diagnostics.csv — background quality (xb vs xa)"
echo "  evaluation/fsoi_closure_per_level_summary.csv — per-(var,level) closure"
echo "  figures/maps/fsoi_*.png           — gridded global FSOI maps"
echo "  figures/positive_frac_timeseries.png — sign stability over time"
echo "=============================================="

# Save submission record
RECORD_FILE="${OUTPUT_ROOT}/submission_record.txt"
cat > "$RECORD_FILE" <<EOF
FSOI Full Evaluation Submission Record
=======================================
Submitted     : $(date)
User          : $USER
Checkpoint    : $CHECKPOINT_PATH
Date range    : $START_DATE to $END_DATE
Data path     : $DATA_PATH
Output root   : $OUTPUT_ROOT

Jobs
----
$JID_SONDE    fsoi_sonde_${DATE_TAG}    radiosonde
$JID_AIRCRAFT fsoi_aircraft_${DATE_TAG} aircraft
$JID_SURFACE  fsoi_surface_${DATE_TAG}  surface_obs
EOF
echo "[SUBMIT] Submission record: $RECORD_FILE"
