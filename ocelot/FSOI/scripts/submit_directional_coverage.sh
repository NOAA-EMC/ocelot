#!/bin/bash
# Directional gradient checks for the surface and aircraft verification metrics.
#
# The reported checks cover the radiosonde metric only, while the surface metric carries
# the SEVIRI result and most path-integration failures. Each job runs two consecutive
# cycles of one metric with:
#   - three perturbation sizes (validation.directional_epsilons), a size-sensitivity check
#   - directions restricted to the entries FSOI sums (validation.directional_valid_only),
#     so the test covers the valid-observation subspace rather than the whole tensor
# A radiosonde job repeats the published configuration with the same two options, so the
# three metrics are compared on equal terms.
#
# Usage (from gnn_model):  bash FSOI/scripts/submit_directional_coverage.sh
# Then:                    python FSOI/analyze_directional_coverage.py
set -euo pipefail

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
SEASONAL="FSOI/fsoi_outputs/seasonal_inclusion_weighted_final"
CONFIGS="FSOI/configs/generated/directional_coverage"
OUT="FSOI/fsoi_outputs/directional_coverage"
mkdir -p "$CONFIGS" "$OUT/logs"

# Two consecutive scored cycles, padded so neither sits at a range edge.
START=2025-07-09
END=2025-07-12

for target in surface_obs aircraft radiosonde; do
    for support in valid_only all_entries; do
        source_cfg="$SEASONAL/${target}_jul2025/logs/fsoi_config_used.yaml"
        [[ -f "$source_cfg" ]] || { echo "Missing $source_cfg" >&2; exit 1; }
        cfg="$CONFIGS/${target}_${support}.yaml"
        python - "$source_cfg" "$cfg" "$support" <<'EOF'
import sys, yaml
src, dst, support = sys.argv[1:]
c = yaml.safe_load(open(src))
v = c.setdefault('validation', {})
v['directional_derivative_check'] = True
v['directional_n_trials'] = 5
v['directional_epsilons'] = [0.003, 0.01, 0.03]      # perturbation-size sensitivity
v['directional_valid_only'] = support == 'valid_only'
v['finite_difference_check'] = False
v['float64_fd_check'] = False
v['check_reproducibility'] = True
p = c.setdefault('plots', {})
p['save_scatter_samples'] = False
p['save_combined_grid'] = False
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
EOF
        name="${target}_${support}"
        sbatch --job-name="dircov_${name}" --time=03:00:00 \
            --output="$OUT/logs/${name}_%j.out" --error="$OUT/logs/${name}_%j.err" \
            --export=ALL,GNN_MODEL_DIR="$GNN_MODEL_DIR",CONFIG_FILE="$cfg",\
FSOI_OUTPUT_DIR="$OUT/$name",FSOI_START_DATE="$START",FSOI_END_DATE="$END" \
            FSOI/scripts/run_fsoi_target_metric.sh
    done
done
echo "Submitted 6 jobs; outputs under $OUT"
