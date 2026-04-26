#!/bin/bash
set -euo pipefail

# Submit a radiosonde-on-grid (mesh-grid) vs GFS run at a fixed pressure level.
# Defaults to 850 hPa, where mesh_pressure_level_idx=2.
#
# Usage:
#   bash evaluation/scripts/submit_radiosonde_mesh_850.sh
#
# Optional overrides (env vars):
#   INIT_TIME=2025030100
#   CKPT=/full/path/to/last.ckpt
#   EXP_NAME=MyExpName
#   FHR_LIST="3 6 9 12"
#   GFS_ROOT=/path/to/gfs-rt25
#   RADIOSONDE_LEVELS=1000
#   MESH_PRESSURE_LEVEL_IDX=0
#   OBS_SPACE_PRESSURE_LEVEL_IDX=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${GNN_MODEL_DIR}"

INIT_TIME=${INIT_TIME:-2025030100}
FHR_LIST=${FHR_LIST:-"3 6 9 12"}

# IMPORTANT: set this to your checkpoint if different.
CKPT=${CKPT:-"${GNN_MODEL_DIR}/checkpoints/Rand_MSELoss_Con/last.ckpt"}

# Pressure-level selection.
RADIOSONDE_LEVELS=${RADIOSONDE_LEVELS:-850}
MESH_PRESSURE_LEVEL_IDX=${MESH_PRESSURE_LEVEL_IDX:-2}
OBS_SPACE_PRESSURE_LEVEL_IDX=${OBS_SPACE_PRESSURE_LEVEL_IDX:-${MESH_PRESSURE_LEVEL_IDX}}

# Write into predictions/<EXP_NAME>
EXP_NAME=${EXP_NAME:-"Rand_MSELoss_Con_radiosonde_mesh_${RADIOSONDE_LEVELS}hPa"}

# GFS archive root
GFS_ROOT=${GFS_ROOT:-"/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25"}

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: CKPT not found: ${CKPT}" >&2
  exit 2
fi

echo "Submitting radiosonde mesh-grid vs GFS @${RADIOSONDE_LEVELS}hPa"
echo "  INIT_TIME=${INIT_TIME}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  CKPT=${CKPT}"
echo "  FHR_LIST=${FHR_LIST}"
echo "  GFS_ROOT=${GFS_ROOT}"
echo "  RADIOSONDE_LEVELS=${RADIOSONDE_LEVELS}"
echo "  MESH_PRESSURE_LEVEL_IDX=${MESH_PRESSURE_LEVEL_IDX}"
echo "  OBS_SPACE_PRESSURE_LEVEL_IDX=${OBS_SPACE_PRESSURE_LEVEL_IDX}"

sbatch --export=ALL,\
INSTRUMENT=radiosonde,\
INIT_TIME=${INIT_TIME},\
EXP_NAME=${EXP_NAME},\
CKPT=${CKPT},\
FHR_LIST="${FHR_LIST}",\
GFS_ROOT=${GFS_ROOT},\
RADIOSONDE_LEVELS=${RADIOSONDE_LEVELS},\
OBS_SPACE_PRESSURE_LEVEL_IDX=${OBS_SPACE_PRESSURE_LEVEL_IDX},\
MESH_PRESSURE_LEVEL_IDX=${MESH_PRESSURE_LEVEL_IDX} \
"${SCRIPT_DIR}/run_pred_eval_gfs.sh"
