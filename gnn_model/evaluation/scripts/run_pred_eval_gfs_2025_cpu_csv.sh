#!/bin/bash -l
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-compute
#SBATCH -J ocelot_2025_csv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-489%24
#SBATCH --output=logs/ocelot_2025_csv_%A_%a.out
#SBATCH --error=logs/ocelot_2025_csv_%A_%a.err
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

# Default to the 2025 remainder window: 2025-05-01 through 2025-12-31.
# With 00/12Z cycles this is 245 days x 2 = 490 array tasks.
export INIT_START_DATE=${INIT_START_DATE:-20250501}
export INIT_END_DATE=${INIT_END_DATE:-20251231}

# CPU post-processing mode: consume existing prediction CSVs, build CSV products,
# and skip all per-init plotting.
export RUN_PREDICTION=${RUN_PREDICTION:-0}
export CSV_ONLY=${CSV_ONLY:-1}
export RUN_TRUTH_PLOTS=${RUN_TRUTH_PLOTS:-0}
export RUN_GFS_PLOTS=${RUN_GFS_PLOTS:-0}
export RUN_MESH_GFS_PLOTS=${RUN_MESH_GFS_PLOTS:-0}

# The manuscript scripts compute pooled metrics directly, so per-init metrics
# are optional. Override RUN_METRICS=1 if you still want them.
export RUN_METRICS=${RUN_METRICS:-0}
export RUN_OBS_GFS_CSV=${RUN_OBS_GFS_CSV:-1}
export RUN_MESH_GFS_CSV=${RUN_MESH_GFS_CSV:-1}

bash "${SCRIPT_DIR}/run_pred_eval_gfs_2025.sh"
