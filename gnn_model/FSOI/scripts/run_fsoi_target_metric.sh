#!/bin/bash -l
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=fsoi_target_metric_%j.out
#SBATCH --error=fsoi_target_metric_%j.err

# Run a coverage audit, seasonal FSOI, or optional matched/full-mask OSE.
# Explicit config and output paths prevent accidental reuse of previous results.
set -euo pipefail
source "${CONDA_BASE:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-gnn-env}"
cd "${GNN_MODEL_DIR:?Set GNN_MODEL_DIR to the gnn_model directory}"
: "${CONFIG_FILE:?Generate a target-metric config first}"
: "${FSOI_OUTPUT_DIR:?Set a new output directory}"
: "${FSOI_START_DATE:?Set the first observation-window date}"
: "${FSOI_END_DATE:?Set the last observation-window date}"
[[ ! -e "$FSOI_OUTPUT_DIR" ]] || { echo "Refusing existing output: $FSOI_OUTPUT_DIR" >&2; exit 1; }
python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); assert c["forecast"].get("verification_metric"), "Missing verification_metric"' "$CONFIG_FILE"
args=(
    --checkpoint "${CHECKPOINT_PATH:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt}"
    --config "$CONFIG_FILE"
    --data_path "${DATA_PATH:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7}"
    --start_date "$FSOI_START_DATE" --end_date "$FSOI_END_DATE"
    --output_dir "$FSOI_OUTPUT_DIR" --verification_target obs --diagnostics
)
if [[ -n "${OSE_INSTRUMENTS:-}" ]]; then
    # Accept commas and colons as well as spaces: sbatch --export splits its own
    # value list on commas, so a multi-instrument list must travel colon-separated.
    read -r -a instruments <<< "${OSE_INSTRUMENTS//[,:;]/ }"
    args+=(--ose_instruments "${instruments[@]}" --ose_denial_mode "${OSE_DENIAL_MODE:-background_replacement}")
    if [[ -n "${OSE_CHANNELS:-}" ]]; then
        read -r -a channels <<< "$OSE_CHANNELS"
        args+=(--ose_channels "${channels[@]}")
    fi
    if [[ -n "${OSE_PATH_INTEGRATION_PAIR_INDICES:-}" ]]; then
        # A space-separated value cannot survive sbatch --export, whose own
        # separator is the comma, so accept commas and colons here too.
        read -r -a pairs <<< "${OSE_PATH_INTEGRATION_PAIR_INDICES//[,:;]/ }"
        args+=(--ose_path_integration_pair_indices "${pairs[@]}")
        if [[ -n "${OSE_PATH_INTEGRATION_T_VALUES:-}" ]]; then
            read -r -a t_values <<< "${OSE_PATH_INTEGRATION_T_VALUES//[,:;]/ }"
            args+=(--ose_path_integration_t_values "${t_values[@]}")
        fi
    fi
fi
python FSOI/fsoi_inference.py "${args[@]}"
