#!/bin/bash
# Shared settings for the OCELOT v1 evaluation launchers. Sourced by the Slurm scripts.
# Override any variable at submit time: sbatch --export=ALL,CKPT=/path/x.ckpt ...

SOURCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${GNN_MODEL_DIR:-}" && -f "${GNN_MODEL_DIR}/predict_gnn.py" ]]; then
  GNN_MODEL_DIR="$(cd "${GNN_MODEL_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/predict_gnn.py" ]]; then
  GNN_MODEL_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  GNN_MODEL_DIR="$(cd "${SOURCE_SCRIPT_DIR}/../.." && pwd)"
fi
export GNN_MODEL_DIR
cd "${GNN_MODEL_DIR}"   # predict_gnn.py / train_gnn.py read configs/ relative to this dir
export REV_DIR="${GNN_MODEL_DIR}/evaluation/revision"
OCELOT_DIR="$(cd "${GNN_MODEL_DIR}/.." && pwd)"
export PYTHONPATH="${GNN_MODEL_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"

# Manuscript (OCELOT v1.0) checkpoint and data archive.
export CKPT=${CKPT:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt}
export DATA_PATH=${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}
export GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}

# Use all allocated cores for CPU inference (harmless on GPU jobs).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Instrument groups for the observation-denial experiments.
declare -A DENY_GROUPS=(
  [control]=""
  [mw_sounders]="atms,amsua"
  [mw_imager]="ssmis"
  [ir_imagers]="avhrr,seviri_asr,seviri_csr"
  [scatterometer]="ascat"
  [aircraft]="aircraft"
  [radiosonde]="radiosonde"
  [surface]="surface_obs"
)
export DENY_GROUP_NAMES="control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface"

# Deterministic 2025 evaluation subset: every 3rd day, alternating 00/12 UTC (~122 inits,
# all seasons, both cycles). Set INIT_LIST_FILE to use your own list (one YYYYMMDDHH per line).
make_subset_inits() {
  python - <<'EOF'
import pandas as pd
days = pd.date_range("2025-01-01", "2025-12-31", freq="3D")
for i, d in enumerate(days):
    print(d.strftime("%Y%m%d") + ("00" if i % 2 == 0 else "12"))
EOF
}

# Run predict_gnn.py for a single init; extra args are forwarded.
#   run_one_init <INIT_YYYYMMDDHH> <OUT_ROOT> [extra predict_gnn args...]
run_one_init() {
  local init="$1" out_root="$2"; shift 2
  local d="${init:0:8}"
  local start end
  start=$(date -u -d "${d} -1 day" +%Y-%m-%d)
  end=$(date -u -d "${d} +2 day" +%Y-%m-%d)
  export PREDICT_INIT_TIME_FILTER="${init}"
  mkdir -p "${out_root}"
  srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python "${GNN_MODEL_DIR}/predict_gnn.py" \
    --checkpoint "${CKPT}" --data_path "${DATA_PATH}" \
    --start_date "${start}" --end_date "${end}" \
    --output_dir "${out_root}" --eval-mode --devices 1 --num_nodes 1 --batch_size 1 "$@"
}
