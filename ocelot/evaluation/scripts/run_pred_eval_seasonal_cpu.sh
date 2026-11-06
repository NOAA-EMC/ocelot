#!/bin/bash -l
#SBATCH -A da-cpu
#SBATCH -p u1-service
#SBATCH -J ocelot_seasonal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --output=logs/ocelot_seasonal_%A_%a.out
#SBATCH --error=logs/ocelot_seasonal_%A_%a.err
#SBATCH --array=0-15
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# OCELOT-only multi-init seasonal prediction + evaluation (CPU).
#
# Generates predictions, OCELOT-vs-truth plots, and pointwise metrics
# for the 16 seasonal init dates used to support the manuscript figures
# (spatial_examples_panel, profiles_panel, Table 2, rollout_growth).
#
# This script does NOT run any GFS comparison (those inits are out of
# the local GFS archive window). For GFS-compare figures, use
# run_pred_eval_gfs.sh on the 2025-02-20..2025-04-01 set.
#
# Submit (array job, one init per task):
#   cd /scratch4/.../main_PR/ocelot/gnn_model
#   mkdir -p logs
#   sbatch evaluation/scripts/run_pred_eval_seasonal_cpu.sh
#
# Submit a single init for testing:
#   sbatch --array=0 evaluation/scripts/run_pred_eval_seasonal_cpu.sh
#
# Override checkpoint or experiment name:
#   sbatch --export=ALL,CKPT=/path/to.ckpt,EXP_NAME=PR_Test \
#       evaluation/scripts/run_pred_eval_seasonal_cpu.sh
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  GNN_MODEL_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  GNN_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
OCELOT_DIR="$(cd "${GNN_MODEL_DIR}/.." && pwd)"

cd "${SLURM_SUBMIT_DIR:-${GNN_MODEL_DIR}}"

# ---------------------------------------------------------------------
# 16 seasonal init dates (00Z), covering all four NH seasons in 2025.
# Order matches SBATCH --array=0-15.
# ---------------------------------------------------------------------
INIT_TIMES=(
  # Winter (NH)
  2025010800
  2025012200
  2025020500
  2025021900
  # Spring (NH) / Autumn (SH)
  2025031200
  2025032600
  2025040900
  2025042300
  # Summer (NH) / Winter (SH)
  2025060400
  2025070200
  2025073000
  2025081300
  # Autumn (NH) / Spring (SH)
  2025091000
  2025100800
  2025110500
  2025120300
)

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( ARRAY_IDX < 0 || ARRAY_IDX >= ${#INIT_TIMES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${ARRAY_IDX} out of range 0..$((${#INIT_TIMES[@]}-1))"
  exit 2
fi
INIT_TIME="${INIT_TIMES[${ARRAY_IDX}]}"

# =====================
# User-config parameters
# =====================

# All instruments used by the OCELOT-only manuscript figures.
INSTRUMENTS=${INSTRUMENTS:-"surface_obs radiosonde aircraft atms amsua ssmis seviri avhrr ascat"}

# Experiment name -> writes under predictions/<EXP_NAME>
EXP_NAME=${EXP_NAME:-PR_Test}

# Checkpoint (the 1556-epoch one used for the current manuscript).
CKPT=${CKPT:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/gnn-epoch-epoch=1556-val_loss-val_loss=0.18.ckpt}

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: CKPT not found: ${CKPT}"
  exit 2
fi

# Lead times to plot.
EVAL_FHR_DEFAULT=${EVAL_FHR_DEFAULT:-3}

# Optional extra args forwarded to evaluations.py (plots).
EVAL_EXTRA_ARGS=${EVAL_EXTRA_ARGS:-""}

# Outputs (must match run_pred_eval_gfs.sh layout so downstream
# aggregation scripts can glob across all inits transparently).
OUT_ROOT=${OUT_ROOT:-"predictions/${EXP_NAME}"}
OBS_DIR=${OUT_ROOT}/pred_csv/obs-space
PLOT_ROOT="evaluation/${OUT_ROOT}/figures"
PLOT_TRUTH_DIR=${PLOT_ROOT}/ocelot_vs_truth/init_${INIT_TIME}

EVAL_SCRIPT="evaluation/scripts/evaluations.py"

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
OCELOT_ENV_HOME="${OCELOT_ENV_HOME:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/ocelot_env}"
MM="${MM:-${OCELOT_ENV_HOME}/micromamba/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${OCELOT_ENV_HOME}/micromamba_root}"
OCELOT_ENV_NAME="${OCELOT_ENV_NAME:-ocelot-cu121}"

if [[ ! -x "${MM}" ]]; then
  echo "ERROR: micromamba not found/executable at: ${MM}"
  exit 2
fi

PY=("${MM}" run -n "${OCELOT_ENV_NAME}" python)

# Make sure we run THIS checkout, and force CPU (no GPUs visible).
export PYTHONPATH="${GNN_MODEL_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

# Thread tuning for CPU inference.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

# Derived dates for predict_gnn.py (YYYY-MM-DD).
INIT_DATE="${INIT_TIME:0:8}"
START_DATE=$(date -u -d "${INIT_DATE} -1 day" +%Y-%m-%d)
END_DATE=$(date -u -d "${INIT_DATE} +1 day" +%Y-%m-%d)

echo "===================================================================="
echo " OCELOT seasonal CPU eval"
echo "   host         : $(hostname)"
echo "   array idx    : ${ARRAY_IDX} of ${#INIT_TIMES[@]}"
echo "   INIT_TIME    : ${INIT_TIME}"
echo "   START_DATE   : ${START_DATE}"
echo "   END_DATE     : ${END_DATE}"
echo "   EXP_NAME     : ${EXP_NAME}"
echo "   CKPT         : ${CKPT}"
echo "   OUT_ROOT     : ${OUT_ROOT}"
echo "   OMP_NUM_THR  : ${OMP_NUM_THREADS}"
echo "===================================================================="

mkdir -p "${OUT_ROOT}" "${PLOT_TRUTH_DIR}"

# ---------------------------------------------------------------------
# 1) Prediction (obs-space, with truth) -- CPU
# ---------------------------------------------------------------------
echo "==== 1) Prediction (CPU) for init=${INIT_TIME} ===="
"${PY[@]}" predict_gnn.py \
  --checkpoint "${CKPT}" \
  --start_date "${START_DATE}" \
  --end_date "${END_DATE}" \
  --output_dir "${OUT_ROOT}" \
  --eval-mode \
  --devices 1 \
  --num_nodes 1 \
  --batch_size 1

# ---------------------------------------------------------------------
# 2) OCELOT-vs-truth plots (per lead hour, all instruments)
# ---------------------------------------------------------------------
echo "==== 2) OCELOT vs Truth plots ===="
"${PY[@]}" "${EVAL_SCRIPT}" --mode plots --has_ground_truth \
  --data_dir "${OBS_DIR}" \
  --plot_dir "${PLOT_TRUTH_DIR}" \
  --init_time "${INIT_TIME}" \
  --fhr "${EVAL_FHR_DEFAULT}" \
  --plot_all_fhrs \
  --plot_horizon_12h \
  ${EVAL_EXTRA_ARGS}

# ---------------------------------------------------------------------
# 3) Pointwise metrics (pred vs truth) -- one CSV per init
#     These feed the multi-init pooled Table 2 + RMSE-vs-lead figures.
# ---------------------------------------------------------------------
echo "==== 3) Pointwise metrics (pred vs truth) ===="
"${PY[@]}" "${EVAL_SCRIPT}" --mode metrics \
  --data_dir "${OBS_DIR}" \
  --metrics_pattern "pred_*init_${INIT_TIME}.csv" \
  --metrics_out "${OUT_ROOT}/metrics_pointwise_init_${INIT_TIME}.csv" \
  --metrics_groupby instrument,lead_hours_nominal

echo "DONE init=${INIT_TIME}. Outputs:"
echo "  Pred CSVs : ${OBS_DIR}"
echo "  Plots     : ${PLOT_TRUTH_DIR}"
echo "  Metrics   : ${OUT_ROOT}/metrics_pointwise_init_${INIT_TIME}.csv"
