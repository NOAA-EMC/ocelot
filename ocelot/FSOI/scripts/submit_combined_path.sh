#!/bin/bash
# Per-instrument residuals along the combined path.
#
# Denying every source at once with background replacement makes the denied path
# identical to the combined-replacement path of the seasonal evaluation. With path
# integration enabled, each cycle then records, for every instrument, its two-endpoint
# contribution and its directional derivative at t = 0, 1/4, 1/2, 3/4, 1 along that one
# path. The difference between the two-endpoint and five-point contributions is that
# instrument's quadrature residual, and the residuals can be summed to test directly
# whether they cancel (Section 4.1).
#
# The endpoint definition is pinned to all_channels so the totals match the combined
# closure reported in the manuscript.
#
# Usage (from gnn_model):  bash FSOI/scripts/submit_combined_path.sh
# Then:                    python FSOI/analyze_combined_path.py
set -euo pipefail

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
SEASONAL="FSOI/fsoi_outputs/seasonal_inclusion_weighted_final"
CONFIG="FSOI/configs/generated/combined_path/radiosonde_jul2025.yaml"
OUT="FSOI/fsoi_outputs/combined_path"
mkdir -p "$(dirname "$CONFIG")" "$OUT/logs"

source_cfg="$SEASONAL/radiosonde_jul2025/logs/fsoi_config_used.yaml"
[[ -f "$source_cfg" ]] || { echo "Missing $source_cfg" >&2; exit 1; }
dates=$(python - "$source_cfg" "$CONFIG" <<'EOF'
import sys, yaml
src, dst = sys.argv[1:]
c = yaml.safe_load(open(src))
c['forecast']['background_endpoint'] = 'all_channels'   # match the reported combined closure
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

# Every source is denied together, so the intervention is the combined replacement.
# Colon-separated: sbatch --export splits its own value list on commas.
INSTRUMENTS="radiosonde:aircraft:surface_obs:atms:amsua:ssmis:ascat:avhrr:seviri_asr"

sbatch --job-name="combined_path_jul" --time=04:00:00 \
    --output="$OUT/logs/combined_path_%j.out" --error="$OUT/logs/combined_path_%j.err" \
    --export=ALL,GNN_MODEL_DIR="$GNN_MODEL_DIR",CONFIG_FILE="$CONFIG",\
FSOI_OUTPUT_DIR="$OUT/radiosonde_jul2025",FSOI_START_DATE="$start",FSOI_END_DATE="$end",\
OSE_INSTRUMENTS="$INSTRUMENTS",OSE_DENIAL_MODE=background_replacement,\
OSE_PATH_INTEGRATION_PAIR_INDICES="5:15:25:35",OSE_PATH_INTEGRATION_T_VALUES="0:0.25:0.5:0.75:1" \
    FSOI/scripts/run_fsoi_target_metric.sh
echo "Submitted; output under $OUT"
