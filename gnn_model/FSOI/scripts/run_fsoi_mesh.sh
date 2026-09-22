#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH -J fsoi_mesh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=fsoi_mesh_%j.out
#SBATCH --error=fsoi_mesh_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# FSOI Mesh-Space Verification against GFS Analysis
# ============================================================================
# Verification: OCELOT mesh predictions (40,962 global nodes) vs GFS analysis
# Reference   : GFS f000 analysis (independent, no instrument-space bias)
# This removes:
#   - NH geographic concentration bias (radiosonde-based verification)
#   - Background field bias (ASCAT +0.54σ, surface_obs +0.67σ)
#
# Pressure level: 500 hPa (mid-troposphere, most sensitive to satellite obs)
# Compare output to obs-space FSOI to identify background-bias artifacts.
#
# Usage:
#   sbatch FSOI/scripts/run_fsoi_mesh.sh [--checkpoint PATH] [--pressure IDX]
#   --pressure 0=1000hPa 4=500hPa 7=250hPa 9=150hPa (default: 4=500hPa)
# ============================================================================

set -e
set -o pipefail

echo "=================================================="
echo "FSOI: Mesh-Space Verification (OCELOT vs GFS)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate gnn-env

find_gnn_model_dir() {
    local d="$1"
    for _ in 1 2 3 4 5 6 7 8; do
        [ -f "$d/configs/observation_config.yaml" ] && [ -d "$d/FSOI" ] && echo "$d" && return 0
        d="$(dirname -- "$d")"
    done
    return 1
}

GNN_MODEL_DIR_RESOLVED=""
[ -n "${GNN_MODEL_DIR:-}" ] && GNN_MODEL_DIR_RESOLVED="$GNN_MODEL_DIR"
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && \
    GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$SLURM_SUBMIT_DIR")" || true
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && \
    GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$PWD")" || true
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && { echo "ERROR: cannot find gnn_model dir"; exit 1; }

cd "$GNN_MODEL_DIR_RESOLVED"
echo "[PATH] $(pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
PRESSURE_IDX=4    # default: 500 hPa
CHECKPOINT_PATH_CLI=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT_PATH_CLI="$2"; shift 2 ;;
        --checkpoint=*)  CHECKPOINT_PATH_CLI="${1#*=}"; shift ;;
        --pressure|-p)   PRESSURE_IDX="$2"; shift 2 ;;
        --pressure=*)    PRESSURE_IDX="${1#*=}"; shift ;;
        *)               shift ;;
    esac
done

PRESSURE_LABELS=(1000 925 850 700 500 400 300 250 200 150 100 70 50 30 20 10)
PRESSURE_HPA="${PRESSURE_LABELS[$PRESSURE_IDX]}"

CONFIG="FSOI/configs/fsoi_config_mesh_radiosonde.yaml"
read -r CONFIG_START CONFIG_END <<< "$(python -c 'import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])); d=cfg.get("data",{}); print(d.get("start_date",""),d.get("end_date",""))' "$CONFIG")"
START="${FSOI_START_DATE:-$CONFIG_START}"
END="${FSOI_END_DATE:-$CONFIG_END}"
DATE_TAG="${START//-/}_${END//-/}"
OUTPUT="${FSOI_OUTPUT_DIR:-FSOI/fsoi_outputs/mesh_radiosonde_${PRESSURE_HPA}hPa_${DATE_TAG}}"

CKPT="${CHECKPOINT_PATH_CLI:-${CHECKPOINT_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
GFS_ROOT="${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}"

[ -f "$CKPT" ] || [ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH not found: $DATA_PATH"; exit 1; }
[ -d "$GFS_ROOT" ] || { echo "ERROR: GFS_ROOT not found: $GFS_ROOT"; exit 1; }

mkdir -p "$OUTPUT" FSOI/logs
LOG_DIR="FSOI/logs"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Configuration:"
echo "  Checkpoint   : $CKPT"
echo "  Date range   : $START → $END"
echo "  Pressure     : idx=${PRESSURE_IDX} = ${PRESSURE_HPA} hPa"
echo "  GFS root     : $GFS_ROOT"
echo "  Output dir   : $OUTPUT"
echo ""

# ── Step 1: Validation tests ──────────────────────────────────────────────────
echo "=== Step 1: Validation tests ==="
python FSOI/test_fsoi.py --checkpoint "$CKPT" \
    2>&1 | tee "${LOG_DIR}/test_fsoi_mesh_${SLURM_JOB_ID}.log"
[ $? -eq 0 ] && echo "✓ Tests passed" || { echo "ERROR: Tests failed"; exit 1; }

# ── Step 2: FSOI computation (mesh-space) ─────────────────────────────────────
echo ""
echo "=== Step 2: FSOI (mesh-space verification) ==="
python FSOI/fsoi_inference.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --data_path "$DATA_PATH" \
    --start_date "$START" \
    --end_date "$END" \
    --output_dir "$OUTPUT" \
    --verification_target mesh \
    --mesh_instrument radiosonde \
    --mesh_pressure_level_idx "$PRESSURE_IDX" \
    --gfs_root "$GFS_ROOT" \
    --diagnostics \
    2>&1 | tee "${LOG_DIR}/fsoi_mesh_${SLURM_JOB_ID}.log"
[ $? -eq 0 ] && echo "✓ FSOI computation complete" || { echo "ERROR: FSOI failed"; exit 1; }

# ── Step 3: Evaluation & visualization ───────────────────────────────────────
echo ""
echo "=== Step 3: Evaluation and visualization ==="
CSV_DIR="$OUTPUT/csv"

python FSOI/evaluate_fsoi_results.py \
    --input "$CSV_DIR" \
    --output "$OUTPUT/evaluation" \
    2>&1 | tee "${LOG_DIR}/evaluate_mesh_${SLURM_JOB_ID}.log" || true

python FSOI/visualize_fsoi.py \
    --input "$CSV_DIR" \
    --output "$OUTPUT/figures" \
    2>&1 | tee "${LOG_DIR}/visualize_mesh_${SLURM_JOB_ID}.log" || true

DIAG_CSV="$OUTPUT/evaluation/innovation_diagnostics.csv"
INNO_SCATTER=""; INNO_DIAG=""
[ -f "$CSV_DIR/scatter_samples.csv" ] && INNO_SCATTER="--scatter $CSV_DIR/scatter_samples.csv"
[ -f "$DIAG_CSV" ] && INNO_DIAG="--diag $DIAG_CSV"
if [ -n "$INNO_SCATTER" ] || [ -n "$INNO_DIAG" ]; then
    python FSOI/plot_innovation_diagnostics.py \
        $INNO_SCATTER $INNO_DIAG \
        --output "$OUTPUT/figures/innovation" \
        2>&1 | tee "${LOG_DIR}/plot_innov_mesh_${SLURM_JOB_ID}.log" || true
fi

echo ""
echo "=================================================="
echo "FSOI mesh run finished: $(date)"
echo "Output: $OUTPUT"
echo "Key result: $OUTPUT/evaluation/fsoi_evaluation_summary.csv"
echo ""
echo "Compare with obs-space run:"
echo "  python FSOI/plot_ose_comparison.py \\"
echo "      --input $OUTPUT \\"
echo "      --output $OUTPUT/figures/mesh_vs_obs"
echo "=================================================="