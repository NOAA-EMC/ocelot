#!/bin/bash
# Background-endpoint check for combined closure.
#
# The seasonal runs replace every channel of a sampled row at x_b, including
# entries flagged missing, while FSOI sums valid entries only. This reruns a
# few cycles per verification network with both endpoint definitions:
#   all_channels  the seasonal definition (control for code drift)
#   valid_only    missing entries kept at their control values
# Each config is copied from the seasonal run's own fsoi_config_used.yaml, so
# the metric, sampling and checkpoint are identical; only the endpoint differs.
#
# Usage (from gnn_model):  bash FSOI/scripts/submit_endpoint_check.sh
# Then:                    python FSOI/compare_endpoint_check.py
set -euo pipefail

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
SEASONAL="FSOI/fsoi_outputs/seasonal_inclusion_weighted_final"
CONFIGS="FSOI/configs/generated/endpoint_check"
OUT="FSOI/fsoi_outputs/endpoint_check_padded"
mkdir -p "$CONFIGS" "$OUT/logs"

# Four test cycles per season (curr_bin 10 12Z to 12 00Z), padded by two cycles on
# each side so neither the start-of-range background nor the end-of-range
# verification window is truncated. compare_endpoint_check.py scores only these four.
declare -A START=([jan]=2025-01-09 [jul]=2025-07-09)
declare -A END=([jan]=2025-01-13 [jul]=2025-07-13)

for target in aircraft radiosonde surface_obs; do
    for variant in all_channels valid_only; do
        source_cfg="$SEASONAL/${target}_jul2025/logs/fsoi_config_used.yaml"
        [[ -f "$source_cfg" ]] || { echo "Missing $source_cfg" >&2; exit 1; }
        cfg="$CONFIGS/${target}_${variant}.yaml"
        python - "$source_cfg" "$cfg" "$variant" <<'EOF'
import sys, yaml
src, dst, variant = sys.argv[1:]
c = yaml.safe_load(open(src))
c['forecast']['background_endpoint'] = variant
v = c.setdefault('validation', {})
for flag in ('finite_difference_check', 'directional_derivative_check', 'float64_fd_check'):
    v[flag] = False                    # not needed for closure; saves time
v['check_reproducibility'] = True      # sets the signal threshold
c.setdefault('plots', {})['save_scatter_samples'] = False
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
EOF
        for month in jan jul; do
            name="${target}_${month}_${variant}"
            sbatch --job-name="endpt_${name}" --time=01:00:00 \
                --output="$OUT/logs/${name}_%j.out" --error="$OUT/logs/${name}_%j.err" \
                --export=ALL,GNN_MODEL_DIR="$GNN_MODEL_DIR",CONFIG_FILE="$cfg",\
FSOI_OUTPUT_DIR="$OUT/$name",FSOI_START_DATE="${START[$month]}",FSOI_END_DATE="${END[$month]}" \
                FSOI/scripts/run_fsoi_target_metric.sh
        done
    done
done
echo "Submitted 12 jobs; outputs under $OUT"
