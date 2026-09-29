#!/bin/bash -l
# EXPERIMENT E5b: verify GraphCastGFS at the OCELOT surface / radiosonde / aircraft target
# observations for all 730 existing 2025 inits (CPU only; reuses the manuscript predictions).
#
# Submit (from gnn_model/):
#   mkdir -p logs && sbatch evaluation/revision/run_graphcastgfs_compare.sh
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-service
#SBATCH -q batch
#SBATCH -J ocelot_gcgfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH --array=0-729%40
#SBATCH --output=logs/ocelot_gcgfs_%A_%a.out
#SBATCH --error=logs/ocelot_gcgfs_%A_%a.err
# (Adjust -p/-q to your CPU partition.)

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/evaluation/revision/revision_env.sh"

GC_ROOT=${GC_ROOT:-/scratch4/NAGAPE/gpu-ai4wp/${USER}/graphcastgfs_2025/gfs_layout}
PRED_DIR=${PRED_DIR:-predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space}
OUT_DIR=${OUT_DIR:-predictions/ocelot_v1_2025_graphcastgfs/obs-space}
mkdir -p "${OUT_DIR}"

mapfile -t INITS < <(python - <<'EOF'
import pandas as pd
for d in pd.date_range("2025-01-01", "2025-12-31", freq="D"):
    for hh in ("00", "12"):
        print(d.strftime("%Y%m%d") + hh)
EOF
)
INIT="${INITS[${SLURM_ARRAY_TASK_ID:-0}]}"

for inst in surface_obs radiosonde aircraft; do
  in_csv="${PRED_DIR}/pred_${inst}_target_init_${INIT}.csv"
  [[ -f "${in_csv}" ]] || { echo "[WARN] missing ${in_csv}"; continue; }
  # Output columns are named gfs_* by the script; they hold GraphCastGFS values here.
  python evaluation/scripts/compare_to_gfs.py \
    --instrument "${inst}" --ocelot_csv "${in_csv}" --gfs_root "${GC_ROOT}" \
    --out_csv "${OUT_DIR}/pred_${inst}_target_init_${INIT}_vs_graphcastgfs.csv" \
    --init_mode from_csv --interp nearest --gfs_time_mode obs_interp --fhr_step 6 --chunk_size 200000
done

# Afterwards (CPU): tabulate OCELOT vs GraphCastGFS vs GFS vs persistence on common rows
#   python evaluation/revision/baseline_table.py \
#     --gfs_glob 'predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space/*_vs_gfs.csv' \
#     --gc_glob  'predictions/ocelot_v1_2025_graphcastgfs/obs-space/*_vs_graphcastgfs.csv' \
#     --out evaluation/revision/results/baselines_surface.csv
