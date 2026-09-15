#!/usr/bin/env bash
# Submit existing target runners with inclusion-weighted source aggregation.
# From gnn_model: bash FSOI/scripts/submit_fsoi_sampling_final.sh smoke|seasonal [--dry-run]
#
# Requires frozen run configs FSOI/configs/generated/<target>_target_run.yaml, produced by
#   summarize_target_coverage.py --freeze-output  ->  make_target_metric_config.py --mode run --frozen-spec
# The base configs score all 16 levels, and one unsupported group excludes every cycle.
set -euo pipefail

mode="${1:-}"
dry_run=false
case "$mode" in
    smoke|seasonal) shift ;;
    *) echo "Usage: $0 smoke|seasonal [--dry-run]" >&2; exit 2 ;;
esac
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
if (( $# )); then echo "Unexpected arguments: $*" >&2; exit 2; fi

GNN_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNN_MODEL_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CKPT:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt}}"
DATA_PATH="${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
OUT_ROOT="${OUT_ROOT:-$GNN_MODEL_DIR/FSOI/fsoi_outputs/seasonal_inclusion_weighted_final}"
CONFIG_DIR="${CONFIG_DIR:-FSOI/configs/generated}"
# Results are written only at the end of a run, so a timeout loses the whole month.
WALLTIME="${WALLTIME:-12:00:00}"
if [[ "$mode" == smoke ]]; then OUT_ROOT="$OUT_ROOT/smoke"; fi

if ! $dry_run; then
    [[ -f "$CHECKPOINT_PATH" ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
    [[ -d "$DATA_PATH" ]] || { echo "Data directory not found: $DATA_PATH" >&2; exit 1; }
    command -v sbatch >/dev/null || { echo "Run this submission script on the HPC login node." >&2; exit 1; }
fi

targets=(radiosonde aircraft surface_obs)
months=(jan2025 apr2025 jul2025 oct2025)
starts=(2025-01-01 2025-04-01 2025-07-01 2025-10-01)
ends=(2025-01-31 2025-04-30 2025-07-31 2025-10-31)
if [[ "$mode" == smoke ]]; then
    months=(jul2025)
    starts=(2025-07-01T00:00:00)
    ends=(2025-07-02T00:00:00)
fi

# Every target must have a frozen (non-audit) run configuration before any job is submitted.
for target in "${targets[@]}"; do
    config="$CONFIG_DIR/${target}_target_run.yaml"
    [[ -f "$config" ]] || { echo "Missing frozen run config: $config" >&2; exit 1; }
    python - "$config" "$target" <<'EOF'
import sys, yaml
path, target = sys.argv[1:]
config = yaml.safe_load(open(path, encoding='utf-8'))
forecast = config['forecast']
assert forecast.get('target_instruments') == [target], f"{path}: target is not {target}"
assert not forecast['verification_metric'].get('audit_only'), f"{path}: audit config, not a run config"
assert config.get('target_metric_freeze'), f"{path}: not generated from a frozen coverage spec"
EOF
done

if [[ "$mode" == seasonal ]] && ! $dry_run; then
    python FSOI/audit_sampling_rank_sensitivity.py \
        --root "$OUT_ROOT/smoke" --require-ht --expected-runs 3
    for target in "${targets[@]}"; do
        design_dir="$OUT_ROOT/smoke/${target}_jul2025/evaluation/sampling_design"
        compgen -G "$design_dir/*.npz" >/dev/null || {
            echo "Missing smoke sampling-design files: $design_dir" >&2; exit 1;
        }
    done
fi

# Preflight every destination before submitting any jobs; never reuse a result directory.
for target in "${targets[@]}"; do
    for month in "${months[@]}"; do
        destination="$OUT_ROOT/${target}_${month}"
        [[ ! -e "$destination" ]] || { echo "Refusing existing output: $destination" >&2; exit 1; }
    done
done
if ! $dry_run; then mkdir -p "$OUT_ROOT/logs"; fi

for target in "${targets[@]}"; do
    case "$target" in
        radiosonde) name=radiosonde_all ;;
        aircraft) name=aircraft ;;
        surface_obs) name=surface_obs ;;
    esac
    config="$CONFIG_DIR/${target}_target_run.yaml"
    runner="FSOI/scripts/run_fsoi_${name}.sh"
    [[ -f "$runner" ]] || { echo "Missing runner for $target" >&2; exit 1; }
    for i in "${!months[@]}"; do
        month="${months[$i]}"
        destination="$OUT_ROOT/${target}_${month}"
        exports="ALL,CHECKPOINT_PATH=$CHECKPOINT_PATH,DATA_PATH=$DATA_PATH,CONFIG_FILE=$config"
        exports+=",FSOI_START_DATE=${starts[$i]},FSOI_END_DATE=${ends[$i]},FSOI_OUTPUT_DIR=$destination"
        exports+=",GNN_MODEL_DIR=$GNN_MODEL_DIR,FSOI_VERIFICATION_TARGET=obs"
        command=(sbatch --parsable --job-name="fsoi_${target}_${month}_ht_${mode}"
            --time="$WALLTIME"
            --output="$OUT_ROOT/logs/${target}_${month}_%j.out"
            --error="$OUT_ROOT/logs/${target}_${month}_%j.err"
            --export="$exports" "$runner" --checkpoint "$CHECKPOINT_PATH")
        if $dry_run; then
            printf '%q ' "${command[@]}"; printf '\n'
        else
            job_id="$("${command[@]}")"
            printf '%s: job %s -> %s\n' "${target}_${month}" "$job_id" "$destination"
        fi
    done
done

printf '\nAfter all jobs finish, validate and compare the saved scalings:\n'
printf 'python FSOI/audit_sampling_rank_sensitivity.py --require-ht --expected-runs %s --root %q\n' \
    "$((${#targets[@]} * ${#months[@]}))" "$OUT_ROOT"
if [[ "$mode" == smoke ]]; then
    echo "Before seasonal: check finite HT totals, sampling_design/*.npz, seconds per pair,"
    echo "and conventional innovation RMS in evaluation/innovation_diagnostics.csv."
fi
