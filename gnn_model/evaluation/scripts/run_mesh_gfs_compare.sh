#!/bin/bash -l
#SBATCH -A da-cpu
#SBATCH -p u1-service
#SBATCH -J mesh_gfs_compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=mesh_gfs_compare_%j.out
#SBATCH --error=mesh_gfs_compare_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OCELOT_DIR="$(cd "${GNN_MODEL_DIR}/.." && pwd)"

cd "${SLURM_SUBMIT_DIR:-${GNN_MODEL_DIR}}"

# =====================
# User-config parameters
# =====================

INIT_TIME=${INIT_TIME:-2025030100}
INSTRUMENT_LIST=${INSTRUMENT_LIST:-"surface_obs radiosonde"}
EXP_NAME=${EXP_NAME:-test1-3years}
FHR_LIST=${FHR_LIST:-"3 6 9 12"}
GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}

OUT_ROOT=${OUT_ROOT:-"predictions/${EXP_NAME}"}
MESH_DIR=${OUT_ROOT}/pred_csv/mesh-grid
PLOT_ROOT="evaluation/figures/${EXP_NAME}"
PLOT_MESH_GFS_DIR=${PLOT_ROOT}/mesh_ocelot_gfs_gfs0/init_${INIT_TIME}

COMPARE_MESH_TO_GFS_SCRIPT=${SCRIPT_DIR}/compare_mesh_to_gfs_update.py
PLOT_MESH_VS_GFS_MAPS_SCRIPT=${SCRIPT_DIR}/plot_mesh_vs_gfs_maps_update.py

echo "Running on $(hostname)"
echo "INIT_TIME=${INIT_TIME}"
echo "INSTRUMENT_LIST=${INSTRUMENT_LIST}"
echo "EXP_NAME=${EXP_NAME}"
echo "FHR_LIST=${FHR_LIST}"
echo "GFS_ROOT=${GFS_ROOT}"
echo "MESH_DIR=${MESH_DIR}"
echo "PLOT_MESH_GFS_DIR=${PLOT_MESH_GFS_DIR}"

# Conda
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

export PYTHONPATH="${GNN_MODEL_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"

# =====================
# Mesh-grid: interpolate GFS forecast + analysis onto OCELOT mesh points
# =====================

echo "==== Mesh-grid: interpolate GFS onto OCELOT mesh grid (same points) ===="
if [ ! -d "${MESH_DIR}" ]; then
  echo "[WARN] Mesh-grid directory not found: ${MESH_DIR}"
  exit 0
fi

for INSTRUMENT in ${INSTRUMENT_LIST}; do
  echo "---- Instrument: ${INSTRUMENT} ----"

  shopt -s nullglob
  mesh_matches=("${MESH_DIR}/${INSTRUMENT}_init_${INIT_TIME}_f"*.csv)
  filtered_matches=()
  for mesh_path in "${mesh_matches[@]}"; do
    case "${mesh_path}" in
      *_gfs_on_ocelot_mesh.csv) ;;
      *) filtered_matches+=("${mesh_path}") ;;
    esac
  done
  shopt -u nullglob

  if [ ${#filtered_matches[@]} -eq 0 ]; then
    echo "[INFO] No mesh-grid prediction CSVs found for instrument=${INSTRUMENT} init=${INIT_TIME}; skipping."
    echo "[INFO] Mesh-grid files are only produced when prediction runs with enable_mesh_pred: true."
    continue
  fi

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

      echo "[INFO] Building mesh-vs-GFS CSV for instrument=${INSTRUMENT} fhr=${fhr}"
      python "${COMPARE_MESH_TO_GFS_SCRIPT}" \
        --mesh_csv "${mesh_csv}" \
        --gfs_root "${GFS_ROOT}" \
        --out_csv "${gfs_on_ocelot_mesh_csv}" \
        --interp nearest

      # Plot variables appropriate for the instrument.
      mkdir -p "${PLOT_MESH_GFS_DIR}/fhr${fhr}"
      if [ "${INSTRUMENT}" == "radiosonde" ]; then
        mesh_vars="u v temp"
      else
        mesh_vars="u10 v10 t2m sp"
      fi
      for v in ${mesh_vars}; do
        python "${PLOT_MESH_VS_GFS_MAPS_SCRIPT}" \
          --csv "${gfs_on_ocelot_mesh_csv}" \
          --plot_dir "${PLOT_MESH_GFS_DIR}/fhr${fhr}" \
          --var "${v}" \
          --gfs_root "${GFS_ROOT}" || true
      done
    else
      echo "[WARN] Mesh-grid CSV not found for fhr=${fhr}: ${mesh_csv}"
    fi
  done

done

echo "DONE. Outputs:"
echo "  Mesh CSVs:   ${MESH_DIR}"
echo "  Mesh vs GFS: ${PLOT_MESH_GFS_DIR}"
