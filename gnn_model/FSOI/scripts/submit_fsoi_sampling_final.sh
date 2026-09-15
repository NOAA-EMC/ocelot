#!/usr/bin/env bash
# Submit existing target runners with inclusion-weighted source aggregation.
# From gnn_model: bash FSOI/scripts/submit_fsoi_sampling_final.sh smoke|seasonal [--dry-run]
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
WALLTIME="${WALLTIME:-06:00:00}"
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
    ends=(2025-07-01T12:00:00)
fi

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
    config="FSOI/configs/fsoi_config_${name}.yaml"
    runner="FSOI/scripts/run_fsoi_${name}.sh"
    [[ -f "$config" && -f "$runner" ]] || { echo "Missing config or runner for $target" >&2; exit 1; }
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
    echo "Check that all three smoke runs produced finite HT totals and sampling_design/*.npz before submitting seasonal."
fi
