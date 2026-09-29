#!/bin/bash
# Shared settings for the AIES revision experiments. Sourced by the Slurm launchers.
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
export CKPT=${CKPT:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}
export DATA_PATH=${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}
export GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}

source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Instrument groups for the observation-denial experiments (Reviewer 1, comment 5).
declare -A DENY_GROUPS=(
  [control]=""
  [mw_sounders]="atms,amsua"
  [mw_imager]="ssmis"
  [ir_imagers]="avhrr,seviri_asr,seviri_csr"
  [scatterometer]="ascat"
  [aircraft]="aircraft"
  [radiosonde]="radiosonde"
  [surface]="surface_obs"
  [all_satellite]="atms,amsua,ssmis,avhrr,seviri_asr,seviri_csr,ascat"
  [all_conventional]="surface_obs,radiosonde,aircraft"
)
export DENY_GROUP_NAMES="control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional"

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

# Run predict_gnn.py (release/ocelot-v1.0) for a single init; extra args are forwarded.
#   run_one_init <INIT_YYYYMMDDHH> <OUT_ROOT> <ROLLOUT_WINDOWS> [extra predict_gnn args...]
# ROLLOUT_WINDOWS=1 is the trained 12-h forecast; 4 evaluates a 48-h latent rollout.
run_one_init() {
  local init="$1" out_root="$2" rollout_windows="$3"; shift 3
  local d="${init:0:8}"
  local days_after=$(( (rollout_windows * 12 + 23) / 24 + 1 ))
  local start end
  start=$(date -u -d "${d} -1 day" +%Y-%m-%d)
  end=$(date -u -d "${d} +${days_after} day" +%Y-%m-%d)
  export PREDICT_INIT_TIME_FILTER="${init}"
  mkdir -p "${out_root}"
  srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python "${GNN_MODEL_DIR}/predict_gnn.py" \
    --checkpoint "${CKPT}" --data_path "${DATA_PATH}" \
    --start_date "${start}" --end_date "${end}" \
    --output_dir "${out_root}" --eval-mode --devices 1 --num_nodes 1 --batch_size 1 \
    --rollout_windows "${rollout_windows}" "$@"
}
