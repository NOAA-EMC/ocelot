#!/bin/bash
# Rerun the 12 seasonal evaluations to save per-cycle 5-degree grids of each
# sampled row's contribution to the combined metric J (plots.save_combined_grid)
# and combined-metric scatter samples (plots.scatter_metric=combined).
# Each config and date range is copied from the seasonal run's own
# fsoi_config_used.yaml, so every other output reproduces the seasonal results.
#
# Usage (from gnn_model):  bash FSOI/scripts/submit_combined_maps.sh
# Then:                    python FSOI/build_combined_grid.py
set -euo pipefail

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
SEASONAL="FSOI/fsoi_outputs/seasonal_inclusion_weighted_final"
CONFIGS="FSOI/configs/generated/combined_maps"
OUT="FSOI/fsoi_outputs/seasonal_combined_maps"
mkdir -p "$CONFIGS" "$OUT/logs"

for target in aircraft radiosonde surface_obs; do
    for month in jan apr jul oct; do
        name="${target}_${month}2025"
        source_cfg="$SEASONAL/$name/logs/fsoi_config_used.yaml"
        [[ -f "$source_cfg" ]] || { echo "Missing $source_cfg" >&2; exit 1; }
        cfg="$CONFIGS/$name.yaml"
        dates=$(python - "$source_cfg" "$cfg" <<'EOF'
import sys, yaml
src, dst = sys.argv[1:]
c = yaml.safe_load(open(src))
# Reproduce the seasonal evaluation exactly: it used the all-channel background endpoint.
c['forecast']['background_endpoint'] = 'all_channels'
plots = c.setdefault('plots', {})
plots.update(save_combined_grid=True, scatter_metric='combined', save_scatter_samples=True)
v = c.setdefault('validation', {})
for flag in ('finite_difference_check', 'directional_derivative_check', 'float64_fd_check'):
    v[flag] = False                    # not needed for the maps; saves time
v['check_reproducibility'] = True
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(c['data']['start_date'], c['data']['end_date'])
EOF
)
        read -r start end <<< "$dates"
        sbatch --job-name="maps_${name}" --time=05:00:00 \
            --output="$OUT/logs/${name}_%j.out" --error="$OUT/logs/${name}_%j.err" \
            --export=ALL,GNN_MODEL_DIR="$GNN_MODEL_DIR",CONFIG_FILE="$cfg",\
FSOI_OUTPUT_DIR="$OUT/$name",FSOI_START_DATE="$start",FSOI_END_DATE="$end" \
            FSOI/scripts/run_fsoi_target_metric.sh
    done
done
echo "Submitted 12 jobs; outputs under $OUT"
