#!/bin/bash -l
# EXPERIMENT E1 in parallel: per-initialization statistics for all targets as a CPU array,
# followed by a merge job that writes metrics_summary.csv and the compact table.
#
# Submit both steps (from gnn_model/):
#   bash evaluation/revision/run_metrics_array.sh submit
# Other prediction sets (e.g. a denial group):
#   PRED_DIR=predictions/denial_2025/control/pred_csv/obs-space OUT_DIR=evaluation/revision/results/denial/control \
#     CLIM_DIR= N_BOOT=0 bash evaluation/revision/run_metrics_array.sh submit
#SBATCH -A da-cpu
#SBATCH -q batch
#SBATCH -p u1-compute
#SBATCH -J e1_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --output=logs/e1_shard_%A_%a.out

set -euo pipefail
N_SHARDS=${N_SHARDS:-40}
PRED_DIR=${PRED_DIR:-predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space}
OUT_DIR=${OUT_DIR:-evaluation/revision/results/v1_2025}
CLIM_DIR=${CLIM_DIR-evaluation/revision/climatology}   # set CLIM_DIR= (empty) to skip climatology/ACC
N_BOOT=${N_BOOT:-1000}
VERIFY_QC=${VERIFY_QC-}                  # e.g. evaluation/revision/verify_qc.yaml
FLAGS_FROM_DIR=${FLAGS_FROM_DIR-}        # borrow qm_* flags from another run's CSVs (same targets)
CLIM_ARGS=()
[[ -n "${CLIM_DIR}" ]] && CLIM_ARGS=(--clim_dir "${CLIM_DIR}")
[[ -n "${VERIFY_QC}" ]] && CLIM_ARGS+=(--verify_qc "${VERIFY_QC}")
[[ -n "${FLAGS_FROM_DIR}" ]] && CLIM_ARGS+=(--flags_from_dir "${FLAGS_FROM_DIR}")

if [[ "${1:-}" == "submit" ]]; then
  mkdir -p logs
  exp="ALL,N_SHARDS=${N_SHARDS},PRED_DIR=${PRED_DIR},OUT_DIR=${OUT_DIR},CLIM_DIR=${CLIM_DIR},N_BOOT=${N_BOOT},VERIFY_QC=${VERIFY_QC},FLAGS_FROM_DIR=${FLAGS_FROM_DIR}"
  jid=$(sbatch --parsable --array=0-$((N_SHARDS - 1)) --export="${exp}" "$0")
  mid=$(sbatch --parsable --dependency=afterok:"${jid}" --export="${exp},MERGE=1" -J e1_merge -t 01:00:00 \
        --output=logs/e1_merge_%j.out "$0")
  echo "shards: ${jid} (array 0-$((N_SHARDS - 1)))   merge: ${mid}   ->  ${OUT_DIR}/metrics_summary.csv"
  exit 0
fi

source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ "${MERGE:-0}" == "1" ]]; then
  python evaluation/revision/revision_metrics.py --pred_dir "${PRED_DIR}" --out_dir "${OUT_DIR}" \
    ${CLIM_ARGS[@]+"${CLIM_ARGS[@]}"} --n_boot "${N_BOOT}" --merge_shards
else
  python evaluation/revision/revision_metrics.py --pred_dir "${PRED_DIR}" --out_dir "${OUT_DIR}" \
    ${CLIM_ARGS[@]+"${CLIM_ARGS[@]}"} --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
fi
