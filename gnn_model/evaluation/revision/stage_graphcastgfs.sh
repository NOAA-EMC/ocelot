#!/bin/bash
# EXPERIMENT E5a (Reviewer 1 c4 / Reviewer 2: an existing ML forecasting system as baseline)
# Stage NOAA GraphCastGFS forecasts for the 2025 00/12 UTC cycles in the directory layout
# that evaluation/scripts/compare_to_gfs.py already reads, so GraphCastGFS is verified at
# exactly the same observation locations/times as OCELOT, GFS and persistence.
#
# GraphCastGFS writes 6-hourly output (f006, f012, ...). It is initialized from the GDAS
# analysis, so the GFS f000 analysis-time field is linked as its f000; compare_to_gfs.py then
# time-interpolates between f000/f006/f012 exactly as it does for GFS with --fhr_step 6.
#
# Run on a node with internet access (front end / data-transfer node), from gnn_model/:
#   bash evaluation/revision/stage_graphcastgfs.sh
# FIRST verify the bucket layout (it has changed across GraphCastGFS versions):
#   aws s3 ls --no-sign-request s3://noaa-nws-graphcastgfs-pds/
#   aws s3 ls --no-sign-request s3://noaa-nws-graphcastgfs-pds/graphcastgfs.20250401/00/
# and edit REMOTE_PATH below if needed.
set -euo pipefail

BUCKET=${BUCKET:-s3://noaa-nws-graphcastgfs-pds}
DL_ROOT=${DL_ROOT:-/scratch4/NAGAPE/gpu-ai4wp/${USER}/graphcastgfs_2025/raw}
GC_ROOT=${GC_ROOT:-/scratch4/NAGAPE/gpu-ai4wp/${USER}/graphcastgfs_2025/gfs_layout}
GFS_ROOT=${GFS_ROOT:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25}
START=${START:-20250101}
END=${END:-20251231}
FHRS=${FHRS:-"006 012"}

remote_path() {  # $1=YYYYMMDD $2=HH $3=FFF
  echo "${BUCKET}/graphcastgfs.$1/$2/forecasts_13_levels/graphcastgfs.t$2z.pgrb2.0p25.f$3"
}

d="${START}"
while [[ "${d}" -le "${END}" ]]; do
  for hh in 00 12; do
    mkdir -p "${DL_ROOT}/${d}" "${GC_ROOT}/${d}"
    for f in ${FHRS}; do
      local_file="${DL_ROOT}/${d}/graphcastgfs.t${hh}z.pgrb2.0p25.f${f}"
      [[ -s "${local_file}" ]] || aws s3 cp --no-sign-request --only-show-errors "$(remote_path "${d}" "${hh}" "${f}")" "${local_file}" \
        || echo "[WARN] missing $(remote_path "${d}" "${hh}" "${f}")"
      [[ -s "${local_file}" ]] && ln -sf "${local_file}" "${GC_ROOT}/${d}/gfs.${d}.t${hh}z.pgrb2.0p25.f${f}"
    done
    gfs0="${GFS_ROOT}/${d}/gfs.${d}.t${hh}z.pgrb2.0p25.f000"
    [[ -s "${gfs0}" ]] && ln -sf "${gfs0}" "${GC_ROOT}/${d}/gfs.${d}.t${hh}z.pgrb2.0p25.f000"
  done
  d=$(date -u -d "${d} +1 day" +%Y%m%d)
done
echo "GraphCastGFS staged under ${GC_ROOT}"
