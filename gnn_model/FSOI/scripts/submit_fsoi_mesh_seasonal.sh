#!/bin/bash
# ============================================================================
# Mesh-Space FSOI — Seasonal Evaluation (4 seasons × 3 pressure levels)
# ============================================================================
# Submits mesh-based FSOI jobs for Jan / Apr / Jul / Oct 2025.
# Primary level: 500 hPa (mid-troposphere, all instruments).
# Optional additional levels: 850 hPa (lower troposphere) and 250 hPa (UTLS).
#
# Scientific purpose: Compare obs-space rankings (biased toward NH radiosondes)
# with mesh-space rankings (40,962 global nodes, GFS-independent reference).
# Expected key result: ATMS/AMSUA flip from net-detrimental (obs-space) to
# net-beneficial (mesh-space) because the background bias in observation space
# no longer contaminates the verification metric.
#
# Usage:
#   cd gnn_model
#   bash FSOI/scripts/submit_fsoi_mesh_seasonal.sh [OPTIONS]
#
# Options:
#   --checkpoint PATH   model checkpoint (default: Epoch3079_fixedval.ckpt)
#   --levels 4          pressure level indices, space-separated (default: "4")
#                       0=1000, 2=850, 4=500, 7=250, 9=150 hPa
#   --all_levels        shortcut: submit 850+500+250 hPa (indices 2 4 7)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GNN_MODEL_DIR"
echo "[SUBMIT] Working directory: $(pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────
CKPT="/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt"
PRESSURE_INDICES=(4)    # default: 500 hPa only
DATA_PATH="/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"
GFS_ROOT="/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CKPT="$2"; shift 2 ;;
        --checkpoint=*)  CKPT="${1#*=}"; shift ;;
        --levels)        IFS=' ' read -r -a PRESSURE_INDICES <<< "$2"; shift 2 ;;
        --all_levels)    PRESSURE_INDICES=(2 4 7); shift ;;
        --gfs_root)      GFS_ROOT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

[ -f "$CKPT" ] || [ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -d "$DATA_PATH" ] || { echo "ERROR: DATA_PATH not found"; exit 1; }
[ -d "$GFS_ROOT" ] || { echo "ERROR: GFS_ROOT not found: $GFS_ROOT"; exit 1; }

# Pressure level labels
PRESSURE_LABELS=(1000 925 850 700 500 400 300 250 200 150 100 70 50 30 20 10)

echo "[SUBMIT] Checkpoint  : $CKPT"
echo "[SUBMIT] Pressure idx: ${PRESSURE_INDICES[*]}"
echo "[SUBMIT] GFS root    : $GFS_ROOT"
echo ""

# ── Seasonal periods ───────────────────────────────────────────────────────
declare -A SEASONS
SEASONS["jan2025"]="2025-01-01|2025-01-31|winter"
SEASONS["apr2025"]="2025-04-01|2025-04-30|spring"
SEASONS["jul2025"]="2025-07-01|2025-07-31|summer"
SEASONS["oct2025"]="2025-10-01|2025-10-31|fall"
SEASON_ORDER=("jan2025" "apr2025" "jul2025" "oct2025")

OUTPUT_ROOT="FSOI/fsoi_outputs/mesh_seasonal"
mkdir -p "$OUTPUT_ROOT/logs"

# ── Submit jobs ────────────────────────────────────────────────────────────
ALL_JIDS=()
echo "============================================================"
echo "Submitting mesh-space FSOI seasonal jobs"
echo "============================================================"

for pidx in "${PRESSURE_INDICES[@]}"; do
    PHPA="${PRESSURE_LABELS[$pidx]}"
    echo ""
    echo "--- Pressure level: ${PHPA} hPa (idx=${pidx}) ---"

    for season_key in "${SEASON_ORDER[@]}"; do
        IFS='|' read -r start end label <<< "${SEASONS[$season_key]}"
        date_tag="${start//-/}_${end//-/}"
        out_dir="${OUTPUT_ROOT}/radiosonde_${PHPA}hPa_${season_key}"
        job_name="fsoi_mesh_${PHPA}h_${season_key}"

        JID=$(sbatch \
            --job-name="$job_name" \
            --time="22:00:00" \
            --output="${OUTPUT_ROOT}/logs/${job_name}_%j.out" \
            --error="${OUTPUT_ROOT}/logs/${job_name}_%j.err" \
            -A gpu-ai4wp -p u1-h100 -q gpu \
            --gres=gpu:h100:1 --mem=250G \
            --export=ALL,\
GNN_MODEL_DIR="$(pwd)",\
FSOI_START_DATE="$start",\
FSOI_END_DATE="$end",\
FSOI_OUTPUT_DIR="$out_dir",\
DATA_PATH="$DATA_PATH",\
GFS_ROOT="$GFS_ROOT",\
CHECKPOINT_PATH="$CKPT" \
            FSOI/scripts/run_fsoi_mesh.sh \
                --checkpoint "$CKPT" \
                --pressure "$pidx" \
            | awk '{print $NF}')

        echo "  Submitted: ${JID}  →  ${label} ${season_key}  ${PHPA} hPa  → ${out_dir}"
        ALL_JIDS+=("$JID:${season_key}:${PHPA}hPa")
    done
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All mesh-space FSOI seasonal jobs submitted"
echo "============================================================"
echo ""
echo "Jobs submitted (${#ALL_JIDS[@]} total):"
for entry in "${ALL_JIDS[@]}"; do
    IFS=':' read -r jid season phpa <<< "$entry"
    PHPA_NUM="${phpa/hPa/}"
    echo "  ${jid}  ${season}  ${phpa}"
done
echo ""
echo "Output root: $OUTPUT_ROOT"
echo ""
echo "Pressure levels:"
for pidx in "${PRESSURE_INDICES[@]}"; do
    PHPA="${PRESSURE_LABELS[$pidx]}"
    echo "  idx=${pidx} = ${PHPA} hPa"
done
echo ""
echo "Monitor: squeue -u $USER"
echo ""
echo "When complete, compare obs-space vs mesh-space rankings:"
echo "  python3 << 'EOF'"
echo "  import pandas as pd, glob"
echo "  mesh = pd.concat([pd.read_csv(f) for f in"
echo "    glob.glob('${OUTPUT_ROOT}/radiosonde_500hPa_*/evaluation/fsoi_evaluation_summary.csv')])"
echo "  obs  = pd.concat([pd.read_csv(f) for f in"
echo "    glob.glob('FSOI/fsoi_outputs/seasonal_fixed/*/evaluation/fsoi_evaluation_summary.csv')])"
echo "  print('MESH rankings:'); print(mesh.groupby('instrument')['impact_mean_per_pair'].mean().sort_values())"
echo "  print('OBS  rankings:'); print(obs.groupby('instrument')['impact_mean_per_pair'].mean().sort_values())"
echo "  EOF"

# Save record
cat > "${OUTPUT_ROOT}/submission_record.txt" << EOF
Mesh-Space FSOI Seasonal Submission
=====================================
Submitted     : $(date)
User          : $USER
Checkpoint    : $CKPT
GFS root      : $GFS_ROOT
Pressure idx  : ${PRESSURE_INDICES[*]}
Output root   : $OUTPUT_ROOT

Jobs
----
EOF
for entry in "${ALL_JIDS[@]}"; do
    IFS=':' read -r jid season phpa <<< "$entry"
    echo "${jid}  ${season}  ${phpa}" >> "${OUTPUT_ROOT}/submission_record.txt"
done
echo "[SUBMIT] Record saved: ${OUTPUT_ROOT}/submission_record.txt"
