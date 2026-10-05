#!/bin/bash -l
# Run synthetic common-support radiosonde/aircraft target diagnostic for ATMS.

#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fsoi_common_support_%j.out
#SBATCH --error=fsoi_common_support_%j.err

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-gnn-env}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt}"
CONFIG_FILE="${CONFIG_FILE:-FSOI/configs/fsoi_config_radiosonde_all.yaml}"
OBS_CONFIG="${OBS_CONFIG:-configs/observation_config.yaml}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
FSOI_OUTPUT_DIR="${FSOI_OUTPUT_DIR:-FSOI/fsoi_outputs/common_support_atms_ch1}"
FSOI_START_DATE="${FSOI_START_DATE:-2025-07-01}"
FSOI_END_DATE="${FSOI_END_DATE:-2025-07-03}"
FSOI_PAIR_INDICES="${FSOI_PAIR_INDICES:-0}"
FSOI_SOURCE_CHANNEL="${FSOI_SOURCE_CHANNEL:-1}"
FSOI_TEMPLATE_TARGET="${FSOI_TEMPLATE_TARGET:-radiosonde}"
FSOI_COMPARE_TARGETS="${FSOI_COMPARE_TARGETS:-radiosonde,aircraft}"
FSOI_TARGET_VARIABLES="${FSOI_TARGET_VARIABLES:-u_wind,v_wind}"
FSOI_TARGET_PRESSURE_LEVELS="${FSOI_TARGET_PRESSURE_LEVELS:-1000,925,850}"
FSOI_TARGET_VARIABLES="${FSOI_TARGET_VARIABLES//:/,}"
FSOI_TARGET_PRESSURE_LEVELS="${FSOI_TARGET_PRESSURE_LEVELS//:/,}"
FSOI_MAX_SOURCE_NODES="${FSOI_MAX_SOURCE_NODES:-50000}"
FSOI_MAX_TARGET_ROWS="${FSOI_MAX_TARGET_ROWS:-}"
GNN_MODEL_DIR="${GNN_MODEL_DIR:-$(pwd)}"

cd "${GNN_MODEL_DIR}"

cmd=(
  python FSOI/diagnose_common_support_targets.py
  --checkpoint "${CHECKPOINT_PATH}"
  --config "${CONFIG_FILE}"
  --obs-config "${OBS_CONFIG}"
  --data-path "${DATA_PATH}"
  --output-dir "${FSOI_OUTPUT_DIR}"
  --start-date "${FSOI_START_DATE}"
  --end-date "${FSOI_END_DATE}"
  --pair-indices "${FSOI_PAIR_INDICES}"
  --source-channel "${FSOI_SOURCE_CHANNEL}"
  --template-target "${FSOI_TEMPLATE_TARGET}"
  --compare-targets "${FSOI_COMPARE_TARGETS}"
  --variables "${FSOI_TARGET_VARIABLES}"
  --pressure-levels "${FSOI_TARGET_PRESSURE_LEVELS}"
  --max-source-nodes "${FSOI_MAX_SOURCE_NODES}"
)

if [[ -n "${FSOI_MAX_TARGET_ROWS}" ]]; then
  cmd+=(--max-target-rows "${FSOI_MAX_TARGET_ROWS}")
fi

echo "[COMMON SUPPORT] ${cmd[*]}"
"${cmd[@]}"
