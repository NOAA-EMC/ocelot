#!/bin/bash -l
# EXPERIMENT E4 (Reviewer 1 c5: contribution of individual observing systems)
# Inference-time observation-denial experiments with the frozen OCELOT v1 model. For one
# group (DENY_GROUP), the inputs of those instruments are withheld from the encoder while
# ALL targets are still predicted and verified, over the 2025 evaluation subset.
# Withholding = no encoder edges for that instrument, identical to the data module's
# representation of a window in which the instrument reported nothing (FSOI "drop_nodes").
#
# Submit every group (incl. the matched control) with:
#   bash evaluation/revision/submit_denial_all.sh
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J ocelot_denial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH --array=0-121%16
#SBATCH --output=logs/ocelot_denial_%x_%A_%a.out
#SBATCH --error=logs/ocelot_denial_%x_%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/evaluation/revision/revision_env.sh"

DENY_GROUP=${DENY_GROUP:?set DENY_GROUP to one of: ${DENY_GROUP_NAMES}}
[[ -v "DENY_GROUPS[${DENY_GROUP}]" ]] || { echo "unknown DENY_GROUP=${DENY_GROUP}"; exit 2; }
DENIED="${DENY_GROUPS[${DENY_GROUP}]}"

if [[ -n "${INIT_LIST_FILE:-}" ]]; then mapfile -t INITS < "${INIT_LIST_FILE}"; else mapfile -t INITS < <(make_subset_inits); fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
(( IDX < ${#INITS[@]} )) || { echo "index ${IDX} out of range"; exit 0; }
INIT="${INITS[$IDX]}"

OUT_ROOT="${OUT_ROOT:-predictions/denial_2025/${DENY_GROUP}}"
echo "Denial group=${DENY_GROUP} withheld=[${DENIED}] init=${INIT} -> ${OUT_ROOT}"
run_one_init "${INIT}" "${OUT_ROOT}" 1 --deny_input_instruments "${DENIED}"
