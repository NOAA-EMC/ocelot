#!/bin/bash
# EXPERIMENT E5a (Reviewer 1 c4 / Reviewer 2: an existing ML forecasting system as baseline)
# Stage NOAA GraphCastGFS forecasts for the 2025 00/12 UTC cycles in the directory layout
# that evaluation/scripts/compare_to_gfs.py already reads, so GraphCastGFS is verified at
# exactly the same observation locations/times as OCELOT, GFS and persistence.
#
# Source: public bucket https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com (plain HTTPS, no AWS CLI):
#   graphcastgfs.YYYYMMDD/HH/forecasts_13_levels/graphcastgfs.tHHz.pgrb2.0p25.fFFF (+ .idx)
# Output is 6-hourly, including its own f000 (initial state from the GDAS analysis); compare_to_gfs.py
# time-interpolates between f000/f006/f012 with --fhr_step 6.
#
# Full files are ~270 MB each, so only the needed GRIB messages are fetched with HTTP byte ranges
# from the .idx index (~3.2 MB per message):
#   surface (default): UGRD/VGRD 10 m, TMP 2 m, PRMSL           ~13 MB per file, ~28 GB for 2025
#   UPPER_AIR=1 adds TMP/UGRD/VGRD on all 13 pressure levels    ~140 MB per file, ~300 GB for 2025
#
# Run on a node with internet access (front end), from gnn_model/:
#   bash evaluation/revision/stage_graphcastgfs.sh                 # whole year, surface fields
#   START=20250101 END=20250107 bash evaluation/revision/stage_graphcastgfs.sh   # quick test
# Re-running skips files already downloaded.
set -euo pipefail

BUCKET=${BUCKET:-https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com}
GC_ROOT=${GC_ROOT:-/scratch4/NAGAPE/gpu-ai4wp/${USER}/graphcastgfs_2025/gfs_layout}
START=${START:-20250101}
END=${END:-20251231}
FHRS=${FHRS:-"000 006 012"}
UPPER_AIR=${UPPER_AIR:-0}

PATTERN=':(UGRD|VGRD):10 m above ground:|:TMP:2 m above ground:|:PRMSL:mean sea level:'
if [[ "${UPPER_AIR}" == "1" ]]; then
  PATTERN="${PATTERN}|:(TMP|UGRD|VGRD):[0-9]+ mb:"
fi

remote_path() {  # $1=YYYYMMDD $2=HH $3=FFF
  echo "${BUCKET}/graphcastgfs.$1/$2/forecasts_13_levels/graphcastgfs.t$2z.pgrb2.0p25.f$3"
}

# Download the GRIB messages matching PATTERN into $2 (concatenated messages form a valid GRIB2 file).
fetch_subset() {  # $1=remote url  $2=local file
  local url="$1" out="$2" tmp="$2.part" idx ranges
  idx=$(curl -fsS --retry 3 "${url}.idx") || return 1
  # Byte range of each matching record: [offset, next record's offset - 1]; last record open-ended.
  ranges=$(printf '%s\n' "${idx}" | awk -F: -v pat="${PATTERN}" '
    { off[NR] = $2; line[NR] = $0 }
    END { for (i = 1; i <= NR; i++) if (line[i] ~ pat) print off[i] "-" ((i < NR) ? off[i+1] - 1 : "") }')
  [[ -n "${ranges}" ]] || return 1
  : > "${tmp}"
  local r
  for r in ${ranges}; do
    curl -fsS --retry 3 -r "${r}" "${url}" >> "${tmp}" || { rm -f "${tmp}"; return 1; }
  done
  mv "${tmp}" "${out}"
}

n_ok=0; n_miss=0
d="${START}"
while [[ "${d}" -le "${END}" ]]; do
  mkdir -p "${GC_ROOT}/${d}"
  for hh in 00 12; do
    for f in ${FHRS}; do
      out="${GC_ROOT}/${d}/gfs.${d}.t${hh}z.pgrb2.0p25.f${f}"   # name compare_to_gfs.py expects
      if [[ -s "${out}" ]] || fetch_subset "$(remote_path "${d}" "${hh}" "${f}")" "${out}"; then
        n_ok=$((n_ok + 1))
      else
        n_miss=$((n_miss + 1)); echo "[WARN] missing ${d} ${hh}Z f${f}"
      fi
    done
  done
  echo "[${d}] staged (ok=${n_ok}, missing=${n_miss})"
  d=$(date -u -d "${d} +1 day" +%Y%m%d)
done
echo "GraphCastGFS staged under ${GC_ROOT}: ${n_ok} files, ${n_miss} missing"
