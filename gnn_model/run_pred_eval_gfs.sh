#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J pred_eval_gfs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH --output=pred_eval_gfs_%j.out
#SBATCH --error=pred_eval_gfs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCELOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =====================
# User-config parameters
# =====================

# IMPORTANT:
# - INIT_TIME must match your experiment period AND your GFS archive content.
INIT_TIME=${INIT_TIME:-2025030100}
INSTRUMENT=${INSTRUMENT:-surface_obs}

# Which experiment to write under predictions/<EXP_NAME>
# Example: Rand_TenYear_nl16 | Seq_TenYear_nl16
EXP_NAME=${EXP_NAME:-Rand_TenYear_nl16}

# Which lead times to plot (hours)
FHR_LIST=${FHR_LIST:-"3 6 9 12"}

GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}

# Outputs
OUT_ROOT=${OUT_ROOT:-"predictions/${EXP_NAME}"}
OBS_DIR=${OUT_ROOT}/pred_csv/obs-space
MESH_DIR=${OUT_ROOT}/pred_csv/mesh-grid
PLOT_TRUTH_DIR=${OUT_ROOT}/figures/ocelot_vs_truth/init_${INIT_TIME}
PLOT_GFS_DIR=${OUT_ROOT}/figures/gfs_compare/init_${INIT_TIME}

# Mesh-grid plots:
# - OCELOT_on_mesh vs GFS_on_mesh (GFS interpolated to OCELOT mesh points)
PLOT_MESH_GFS_DIR=${OUT_ROOT}/figures/ocelot_on_mesh_vs_gfs_on_mesh/init_${INIT_TIME}

# Checkpoint selection:
# - If CKPT is provided, use it.
# - Otherwise, try checkpoints/<EXP_NAME>/last.ckpt, else newest *.ckpt.
CKPT=${CKPT:-""}
if [ -z "${CKPT}" ]; then
  CKPT_DIR="${SCRIPT_DIR}/checkpoints/${EXP_NAME}"
  if [ -f "${CKPT_DIR}/last.ckpt" ]; then
    CKPT="${CKPT_DIR}/last.ckpt"
  else
    CKPT_CAND=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -n 1 || true)
    if [ -n "${CKPT_CAND}" ]; then
      CKPT="${CKPT_CAND}"
    fi
  fi
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "ERROR: CKPT not found. Set CKPT=/path/to/model.ckpt or ensure checkpoints/${EXP_NAME}/last.ckpt exists."
  exit 2
fi

echo "Running on $(hostname)"
echo "INIT_TIME=${INIT_TIME}"
echo "INSTRUMENT=${INSTRUMENT}"
echo "EXP_NAME=${EXP_NAME}"
echo "CKPT=${CKPT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GFS_ROOT=${GFS_ROOT}"

# Conda
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Ensure we run the code from THIS checkout (not NNJA mirror)
export PYTHONPATH="${SCRIPT_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"

# Derived dates for predict_gnn.py (YYYY-MM-DD)
# IMPORTANT: end_date must be > start_date, otherwise the datamodule builds zero bins.
INIT_DATE="${INIT_TIME:0:8}"
START_DATE=$(date -u -d "${INIT_DATE} -1 day" +%Y-%m-%d)
END_DATE=$(date -u -d "${INIT_DATE} +1 day" +%Y-%m-%d)

echo "START_DATE=${START_DATE} END_DATE=${END_DATE}"

mkdir -p "${OUT_ROOT}" "${PLOT_TRUTH_DIR}" "${PLOT_GFS_DIR}" "${PLOT_MESH_GFS_DIR}"

echo "==== 1) Prediction (obs-space, with truth) ===="
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores \
  python predict_gnn.py \
    --checkpoint "${CKPT}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUT_ROOT}" \
    --eval-mode \
    --devices 1 \
    --num_nodes 1 \
    --batch_size 1

echo "==== 2) OCELOT vs Truth plots (per lead hour) ===="
for fhr in ${FHR_LIST}; do
  python evaluations.py --mode plots --has_ground_truth \
    --data_dir "${OBS_DIR}" \
    --plot_dir "${PLOT_TRUTH_DIR}/fhr${fhr}" \
    --init_time "${INIT_TIME}" \
    --fhr "${fhr}"
done

echo "==== 2b) Pointwise metrics (pred vs truth) ===="
python evaluations.py --mode metrics \
  --data_dir "${OBS_DIR}" \
  --metrics_pattern "pred_*init_${INIT_TIME}.csv" \
  --metrics_out "${OUT_ROOT}/metrics_pointwise_init_${INIT_TIME}.csv" \
  --metrics_groupby instrument,lead_hours_nominal

echo "==== 3) Build *_vs_gfs.csv (surface_obs: wind + t2m + sp) ===="
OCELOT_CSV="${OBS_DIR}/pred_${INSTRUMENT}_target_init_${INIT_TIME}.csv"
OUT_VS_GFS="${OBS_DIR}/pred_${INSTRUMENT}_target_init_${INIT_TIME}_vs_gfs.csv"

if [ ! -f "${OCELOT_CSV}" ]; then
  echo "ERROR: expected prediction CSV not found: ${OCELOT_CSV}"
  echo "Available files (first 50):"
  ls -1 "${OBS_DIR}" | head -n 50
  exit 3
fi

python compare_to_gfs.py \
  --instrument "${INSTRUMENT}" \
  --ocelot_csv "${OCELOT_CSV}" \
  --gfs_root "${GFS_ROOT}" \
  --out_csv "${OUT_VS_GFS}" \
  --init_mode from_csv \
  --interp nearest \
  --chunk_size 200000

echo "==== 3b) Sanity-check GFS coverage (fail fast if missing) ===="
python -c "
import pandas as pd
p='${OUT_VS_GFS}'
df=pd.read_csv(p, nrows=200000)
gfs_cols=[c for c in df.columns if c.startswith('gfs_')]
if not gfs_cols:
  raise SystemExit(f'ERROR: no gfs_* columns found in {p}')
non_null=int(df[gfs_cols].notna().any(axis=1).sum())
print(f'GFS coverage check: {non_null}/{len(df)} rows have at least one non-NaN gfs_* value')
if non_null==0:
  raise SystemExit('ERROR: GFS columns are all NaN (check INIT_TIME and GFS_ROOT layout)')
"

echo "==== 4) Plots: RMSE vs fhr + maps (OCELOT vs Truth vs GFS) ===="
python plot_gfs_compare.py \
  --init_time "${INIT_TIME}" \
  --data_dir "${OBS_DIR}" \
  --plot_dir "${PLOT_GFS_DIR}" \
  --instrument "${INSTRUMENT}" \
  --vars wind_temperature_pressure \
  --chunksize 200000 \
  --maps --fhrs ${FHR_LIST}

echo "==== 5) Mesh-grid: interpolate GFS onto OCELOT mesh grid (same points) ===="
if [ -d "${MESH_DIR}" ]; then
  # Mesh-grid outputs are one file per fhr: <instrument>_init_<INIT>_f<FFF>.csv
  for fhr in ${FHR_LIST}; do
    mesh_csv="${MESH_DIR}/${INSTRUMENT}_init_${INIT_TIME}_f$(printf '%03d' ${fhr}).csv"
    gfs_on_ocelot_mesh_csv="${MESH_DIR}/${INSTRUMENT}_init_${INIT_TIME}_f$(printf '%03d' ${fhr})_gfs_on_ocelot_mesh.csv"
    if [ -f "${mesh_csv}" ]; then
      # Fail fast: require mesh_idx so we can guarantee strict alignment across mesh products.
      if ! head -n 1 "${mesh_csv}" | tr ',' '\n' | grep -qx "mesh_idx"; then
        echo "ERROR: mesh-grid CSV is missing mesh_idx (old format): ${mesh_csv}"
        echo "Re-run prediction with the updated code that writes mesh_idx."
        exit 4
      fi

      python compare_mesh_to_gfs.py \
        --mesh_csv "${mesh_csv}" \
        --gfs_root "${GFS_ROOT}" \
        --out_csv "${gfs_on_ocelot_mesh_csv}" \
        --interp nearest

      # Plot a small set of variables if present.
      mkdir -p "${PLOT_MESH_GFS_DIR}/fhr${fhr}"
      for v in u10 v10 t2m sp; do
        python plot_mesh_vs_gfs_maps.py \
          --csv "${gfs_on_ocelot_mesh_csv}" \
          --plot_dir "${PLOT_MESH_GFS_DIR}/fhr${fhr}" \
          --var "${v}" || true
      done
    else
      echo "[WARN] Mesh-grid CSV not found for fhr=${fhr}: ${mesh_csv}"
    fi
  done
else
  echo "[WARN] Mesh-grid directory not found: ${MESH_DIR}"
fi

echo "DONE. Outputs:"
echo "  Pred CSVs:   ${OBS_DIR}"
echo "  Mesh CSVs:   ${MESH_DIR}"
echo "  Truth plots: ${PLOT_TRUTH_DIR}"
echo "  GFS plots:   ${PLOT_GFS_DIR}"
echo "  Mesh vs GFS: ${PLOT_MESH_GFS_DIR}"
