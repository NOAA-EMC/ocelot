#!/bin/bash -l
# EXPERIMENT E6 (OPTIONAL, retraining; Reviewer 1 c4/c5 "where design choices help or hurt")
# Train controlled variants from scratch with the OCELOT v1 recipe (run_gnn_Rand.sh on
# release/ocelot-v1.0), changing
# ONE factor at a time:
#   VARIANT=v1_budget        v1 architecture, same reduced budget as the variants (reference)
#   VARIANT=no_spatial_mix   temporal attention only (--spatial_mixing_steps 0)
#   VARIANT=interaction      GraphCast-style interaction-network processor (no temporal attention)
#   VARIANT=deny_satellite   v1, trained with ALL satellite inputs withheld (targets kept)
#   VARIANT=deny_conventional v1, trained with ALL conventional inputs withheld (targets kept)
# Budget: MAX_EPOCHS (default 1000; the manuscript model used ~3080). Every variant uses the
# same seed, data, windows and budget, so differences are attributable to the one factor.
# The job is resubmittable: it resumes from checkpoints/<RUN_NAME>/last.ckpt.
#
# Submit (from gnn_model/):
#   for v in v1_budget no_spatial_mix interaction deny_satellite deny_conventional; do
#     sbatch -J abl_$v --export=ALL,VARIANT=$v evaluation/revision/run_train_ablation.sh; done
# NOTE: do not combine VARIANT=interaction with input denial: the interaction processor also
# passes messages over observation->mesh edges.
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH --output=logs/ocelot_abl_%x_%j.out
#SBATCH --error=logs/ocelot_abl_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/evaluation/revision/revision_env.sh"
export TORCH_NCCL_BLOCKING_WAIT=1 NCCL_SHM_DISABLE=1 NCCL_NET_GDR_LEVEL=PHB NCCL_IB_DISABLE=0 OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=ib0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_LEVEL=NVL TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANT=${VARIANT:?set VARIANT}
MAX_EPOCHS=${MAX_EPOCHS:-1000}
RUN_NAME="abl_${VARIANT}_e${MAX_EPOCHS}"
EXTRA=()
case "${VARIANT}" in
  v1_budget) ;;
  no_spatial_mix) EXTRA+=(--spatial_mixing_steps 0) ;;
  interaction) EXTRA+=(--processor_type interaction) ;;
  deny_satellite) EXTRA+=(--deny_input_instruments "${DENY_GROUPS[all_satellite]}") ;;
  deny_conventional) EXTRA+=(--deny_input_instruments "${DENY_GROUPS[all_conventional]}") ;;
  *) echo "unknown VARIANT=${VARIANT}"; exit 2 ;;
esac
RESUME=()
[[ -f "checkpoints/${RUN_NAME}/last.ckpt" ]] && RESUME+=(--resume_from_latest)

srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
  --run_name "${RUN_NAME}" "${RESUME[@]}" "${EXTRA[@]}" \
  --mesh_type fixed --scan_angle_conditioning project --sampling_mode random \
  --cfg_path configs/observation_config.yaml --data_path "${DATA_PATH}" \
  --train_start_date 2015-01-01 --train_end_date 2024-01-01 \
  --val_start_date 2024-01-01 --val_end_date 2025-01-01 \
  --train_window_days 12 --val_window_days 12 --val_mode sequential --val_stride_days 12 \
  --val_update_every_n_epochs 100 \
  --lr 1.5e-4 --lr_schedule cosine_warmup --warmup_pct 0.05 --warmup_start_factor 0.01 --min_lr 1e-6 \
  --weight_decay 1e-4 --processor_dropout 0.1 --node_dropout 0.05 --encoder_dropout 0.1 --decoder_dropout 0.1 \
  --loss_type mse --seed 12345 --max_epochs "${MAX_EPOCHS}" --disable_early_stopping \
  --cache_val_windows --val_cache_max_entries 16 --disable_val_csv

# Evaluate each finished variant on the same 2025 subset as E4, e.g.:
#   sbatch -J eval_no_spatial_mix --export=ALL,CKPT=$PWD/checkpoints/abl_no_spatial_mix_e1000/last.ckpt,DENY_GROUP=control,OUT_ROOT=predictions/ablation/no_spatial_mix \
#          evaluation/revision/run_denial_ose.sh
# For deny_* variants use DENY_GROUP=all_satellite / all_conventional so inference withholds
# the same inputs as training. Score with revision_metrics.py and compare with
# summarize_denial.py (--control = the v1_budget results).
