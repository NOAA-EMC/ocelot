#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --start YYYY-MM-DD --end YYYY-MM-DD [--datasets adpsfc,adpupa] [--bucket nnja-ai]
Notes:
  - Set NNJA_LOCAL_ROOT to your local mirror root (e.g., /scratch3/.../nnja-ai)
  - Run this on a machine with egress (NOT on compute nodes).
  - Requires gsutil (preferred). If you don't have gsutil, you can use rclone by setting RCLONE_REMOTE (e.g., 'gcs').

Examples:
  export NNJA_LOCAL_ROOT=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/nnja-ai
  ./stage_nnja.sh --start 2024-04-01 --end 2024-07-01
  ./stage_nnja.sh --start 2024-04-01 --end 2024-07-01 --datasets adpsfc
EOF
}

START=""
END=""
DATASETS="adpsfc,adpupa"
BUCKET="nnja-ai"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

: "${NNJA_LOCAL_ROOT:?Set NNJA_LOCAL_ROOT, e.g. /scratch3/.../nnja-ai}"

if [[ -z "$START" || -z "$END" ]]; then usage; exit 1; fi

# Choose sync tool
if command -v gsutil >/dev/null 2>&1; then
  SYNC_TOOL="gsutil"
elif command -v rclone >/dev/null 2>&1 && [[ -n "${RCLONE_REMOTE:-}" ]]; then
  SYNC_TOOL="rclone"
else
  echo "Need gsutil OR rclone (with RCLONE_REMOTE set, e.g., 'gcs')." >&2; exit 1
fi

ROOT="$NNJA_LOCAL_ROOT/data/v1"
IFS=',' read -r -a dsarr <<< "$DATASETS"

# Dataset → hive path map
declare -A DATASET_PATHS=(
  [adpsfc]="conv/adpsfc/NC000101"
  [adpupa]="conv/adpupa/NC002001"
)

# Build inclusive date list via Python (portable)
mapfile -t DATES < <(python - <<PY
import datetime as dt
start=dt.date.fromisoformat("$START")
end=dt.date.fromisoformat("$END")
d=start
while d<=end:
    print(d.isoformat())
    d+=dt.timedelta(days=1)
PY
)

sync_one() {
  local src="$1" dest="$2"
  mkdir -p "$dest"
  if [[ "$SYNC_TOOL" == "gsutil" ]]; then
    gsutil -m rsync -r "$src" "$dest"
  else
    # rclone expects a remote name; set RCLONE_REMOTE (e.g., export RCLONE_REMOTE=gcs)
    rclone copy --transfers=32 --checkers=64 "$RCLONE_REMOTE:$(sed 's#^gs://##' <<<"$src")" "$dest"
  fi
}

for d in "${DATES[@]}"; do
  # GNU date on Linux; on macOS use 'gdate'
  y=$(date -d "$d" +%Y)
  m=$(date -d "$d" +%m)
  dd=$(date -d "$d" +%d)
  for ds in "${dsarr[@]}"; do
    path="${DATASET_PATHS[$ds]}"
    src="gs://$BUCKET/data/v1/$path/year=$y/month=$m/day=$dd"
    dest="$ROOT/$path/year=$y/month=$m/day=$dd"
    echo "[stage_nnja] $src -> $dest"
    sync_one "$src" "$dest"
  done
done

echo "Done. Local mirror root: $ROOT"

