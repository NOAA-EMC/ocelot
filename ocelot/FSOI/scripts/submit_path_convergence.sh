#!/bin/bash
# Path-integration convergence study.
#
# Re-runs the SEVIRI ASR path cases with 5, 9 and 17 equally spaced points. The
# integrator selects composite Simpson automatically for any odd, equally spaced set,
# so only the t values change. Pair indices 5, 15 and 25 cover the three cases that
# stay outside 0.97-1.05 at five points (April 14, July 9, July 14) together with
# successful controls in the same months, so convergence can be compared between them.
#
# One job per month and grid: 4 months x 2 new grids = 8 jobs. The 5-point results
# already exist in fsoi_outputs/ose_frozen_metric_surface.
#
# Usage (from gnn_model):  bash FSOI/scripts/submit_path_convergence.sh
# Then:                    python FSOI/analyze_path_convergence.py
set -euo pipefail

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
SEASONAL="FSOI/fsoi_outputs/seasonal_inclusion_weighted_final"
CONFIGS="FSOI/configs/generated/path_convergence"
OUT="FSOI/fsoi_outputs/path_convergence"
mkdir -p "$CONFIGS" "$OUT/logs"

# 9 and 17 equally spaced points on [0, 1]. Colon-separated: sbatch --export
# splits its own value list on commas, so commas here are lost in transit.
T9="0:0.125:0.25:0.375:0.5:0.625:0.75:0.875:1"
T17="0:0.0625:0.125:0.1875:0.25:0.3125:0.375:0.4375:0.5:0.5625:0.625:0.6875:0.75:0.8125:0.875:0.9375:1"

for month in jan apr jul oct; do
    source_cfg="$SEASONAL/surface_obs_${month}2025/logs/fsoi_config_used.yaml"
    [[ -f "$source_cfg" ]] || { echo "Missing $source_cfg" >&2; exit 1; }
    cfg="$CONFIGS/surface_obs_${month}2025.yaml"
    dates=$(python - "$source_cfg" "$cfg" <<'EOF'
import sys, yaml
src, dst = sys.argv[1:]
c = yaml.safe_load(open(src))
c['forecast']['background_endpoint'] = 'all_channels'   # as in the reported path cases
v = c.setdefault('validation', {})
for flag in ('finite_difference_check', 'directional_derivative_check', 'float64_fd_check'):
    v[flag] = False
v['check_reproducibility'] = True
p = c.setdefault('plots', {})
p['save_scatter_samples'] = False
p['save_combined_grid'] = False
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(c['data']['start_date'], c['data']['end_date'])
EOF
)
    read -r start end <<< "$dates"
    for grid in 9 17; do
        [[ $grid == 9 ]] && tvals="$T9" || tvals="$T17"
        name="seviri_${month}_${grid}pt"
        sbatch --job-name="pathconv_${name}" --time=06:00:00 \
            --output="$OUT/logs/${name}_%j.out" --error="$OUT/logs/${name}_%j.err" \
            --export=ALL,GNN_MODEL_DIR="$GNN_MODEL_DIR",CONFIG_FILE="$cfg",\
FSOI_OUTPUT_DIR="$OUT/$name",FSOI_START_DATE="$start",FSOI_END_DATE="$end",\
OSE_INSTRUMENTS="seviri_asr",OSE_DENIAL_MODE=background_replacement,\
OSE_PATH_INTEGRATION_PAIR_INDICES="5:15:25",OSE_PATH_INTEGRATION_T_VALUES="$tvals" \
            FSOI/scripts/run_fsoi_target_metric.sh
    done
done
echo "Submitted 8 jobs; outputs under $OUT"
