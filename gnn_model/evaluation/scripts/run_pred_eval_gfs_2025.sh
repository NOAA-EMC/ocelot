#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J ocelot_2025_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 05:30:00
#SBATCH --array=0-729
#SBATCH --output=logs/ocelot_2025_full_%A_%a.out
#SBATCH --error=logs/ocelot_2025_full_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SOURCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${GNN_MODEL_DIR:-}" && -f "${GNN_MODEL_DIR}/predict_gnn.py" ]]; then
  GNN_MODEL_DIR="$(cd "${GNN_MODEL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/predict_gnn.py" ]]; then
  GNN_MODEL_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "${SOURCE_SCRIPT_DIR}/../../predict_gnn.py" ]]; then
  GNN_MODEL_DIR="$(cd "${SOURCE_SCRIPT_DIR}/../.." && pwd)"
else
  GNN_MODEL_DIR="/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model"
fi
SCRIPT_DIR="${GNN_MODEL_DIR}/evaluation/scripts"
cd "${GNN_MODEL_DIR}"

GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}
YEAR=${YEAR:-2025}

# Date window. Override these at sbatch time for each 3-month chunk.
INIT_START_DATE=${INIT_START_DATE:-20250101}
INIT_END_DATE=${INIT_END_DATE:-20251231}

# Which initialization times to generate:
#   all      = every available requested cycle in the date window.
#              For full 2025 with 00/12Z cycles this is 730 inits.
#   sampled  = use NUM_INITS and INIT_SAMPLE_MODE below.
INIT_SELECTION=${INIT_SELECTION:-all}

# Used only when INIT_SELECTION=sampled.
NUM_INITS=${NUM_INITS:-120}

# Cycles to sample from.
INIT_CYCLES=${INIT_CYCLES:-00,12}

# Sampling mode:
#   balanced_distinct = 30 different dates, half 00Z and half 12Z
#   paired_cycles     = 15 dates × both 00Z and 12Z = 30 init times
INIT_SAMPLE_MODE=${INIT_SAMPLE_MODE:-balanced_distinct}

REQUIRED_FHRS=${REQUIRED_FHRS:-0,1,2,3,4,5,6,7,8,9,10,11,12}

INIT_LIST_FILE="${SLURM_TMPDIR:-/tmp}/ocelot_init_times_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.txt"
INIT_LIST_ARGS=(
  --gfs-root "${GFS_ROOT}"
  --year "${YEAR}"
  --start-date "${INIT_START_DATE}"
  --end-date "${INIT_END_DATE}"
  --cycles "${INIT_CYCLES}"
  --required-fhrs "${REQUIRED_FHRS}"
)
if [[ "${INIT_SELECTION}" == "all" ]]; then
  INIT_LIST_ARGS+=(--all-inits)
else
  INIT_LIST_ARGS+=(--num-inits "${NUM_INITS}" --sample-mode "${INIT_SAMPLE_MODE}")
fi
python "${SCRIPT_DIR}/make_2025_init.py" "${INIT_LIST_ARGS[@]}" > "${INIT_LIST_FILE}"
mapfile -t INIT_TIMES < "${INIT_LIST_FILE}"

if (( ${#INIT_TIMES[@]} == 0 )); then
  echo "ERROR: no initialization times generated."
  echo "       helper: ${SCRIPT_DIR}/make_2025_init.py"
  echo "       GFS_ROOT=${GFS_ROOT}"
  echo "       range=${INIT_START_DATE}..${INIT_END_DATE} cycles=${INIT_CYCLES} required_fhrs=${REQUIRED_FHRS}"
  exit 2
fi

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( ARRAY_IDX < 0 || ARRAY_IDX >= ${#INIT_TIMES[@]} )); then
  echo "INFO: SLURM_ARRAY_TASK_ID=${ARRAY_IDX} out of range 0..$((${#INIT_TIMES[@]}-1)); skipping."
  exit 0
fi

export INIT_TIME="${INIT_TIMES[${ARRAY_IDX}]}"

# Manuscript checkpoint/current experiment defaults. Override at sbatch time if needed.
export EXP_NAME=${EXP_NAME:-paper_2025_full_Epoch3079}
export CKPT=${CKPT:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}
export INSTRUMENT=${INSTRUMENT:-surface_obs}
export GFS_ROOT
export GFS_TIME_MODE=${GFS_TIME_MODE:-obs_interp}
export GFS_FHR_STEP=${GFS_FHR_STEP:-1}
export FHR_LIST=${FHR_LIST:-"3 6 9 12"}
export ANALYSIS_TIME_MODE=${ANALYSIS_TIME_MODE:-exact}

echo "===================================================================="
echo " OCELOT 2025 GFS eval"
echo "   array idx    : ${ARRAY_IDX} of ${#INIT_TIMES[@]}"
echo "   INIT_TIME    : ${INIT_TIME}"
echo "   init select  : ${INIT_SELECTION}"
echo "   EXP_NAME     : ${EXP_NAME}"
echo "   CKPT         : ${CKPT}"
echo "   GFS_ROOT     : ${GFS_ROOT}"
echo "   date range   : ${INIT_START_DATE}..${INIT_END_DATE}"
echo "   GFS_TIME_MODE: ${GFS_TIME_MODE}"
echo "   GFS_FHR_STEP : ${GFS_FHR_STEP}"
echo "   ANALYSIS_MODE: ${ANALYSIS_TIME_MODE}"
echo "===================================================================="

bash "${SCRIPT_DIR}/run_pred_eval_gfs.sh"
