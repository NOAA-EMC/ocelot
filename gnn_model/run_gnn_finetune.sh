#!/bin/bash -l
# Author: Azadeh Gholoubi
# Purpose: FSOI-guided fine-tuning of OCELOT baseline (M0 → M1)
#
# Changes vs. original training (run_gnn_Rand.sh):
#   1. --load_weights_only: starts from M0 checkpoint, resets optimizer
#   2. --encoding_order random: removes positional gradient-path bias
#      (identified by H1 experiment; see ENCODING_ORDER_INVESTIGATION.md)
#   3. --cfg_path observation_config_finetune.yaml: FSOI-corrected instrument
#      weights + channel exclusions (SSMIS ch21 zeroed; ATMS/AMSU-A detrimental
#      channels downweighted to 0.1; radiosonde weight 8.17x)
#   4. Lower LR (1e-5) + shorter run (1000 epochs max)
#   5. Same random-window sampling, same data path

#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -J ocelot_finetune
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH --output=ocelot_finetune_%j.out
#SBATCH --error=ocelot_finetune_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
cd "$WORKDIR"
echo "Working directory: $PWD"
echo "Running on H100 nodes..."

source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_DESYNC_DEBUG=1
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "SLURM Node List: $SLURM_NODELIST"

# ============================================================================
# Fine-tune configuration
# ============================================================================
RUN_NAME="${RUN_NAME:-ocelot_v1_finetune_fsoi}"
echo "RUN_NAME=$RUN_NAME"

# Baseline checkpoint to fine-tune from
BASELINE_CKPT="${BASELINE_CKPT:-/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt}"
[ -f "$BASELINE_CKPT" ] || { echo "ERROR: baseline checkpoint not found: $BASELINE_CKPT"; exit 1; }
echo "Baseline checkpoint: $BASELINE_CKPT"

# Fine-tune LR — ~10x lower than initial training to prevent catastrophic forgetting
LR="${LR:-1e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine_warmup}"
WARMUP_PCT="${WARMUP_PCT:-0.05}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-7}"
LOSS_TYPE="${LOSS_TYPE:-mse}"

echo "LR=$LR  LR_SCHEDULE=$LR_SCHEDULE  LOSS_TYPE=$LOSS_TYPE"

# Resume logic: if a fine-tune checkpoint already exists, resume from it;
# otherwise start from the baseline.
CKPT_DIR="checkpoints/${RUN_NAME}"
CKPT_LAST="${CKPT_DIR}/last.ckpt"

if [ -f "$CKPT_LAST" ]; then
    echo "[INFO] Resuming fine-tune from: $CKPT_LAST"
    RESUME_ARGS=(--resume_from_latest)
    WEIGHTS_ARGS=()
else
    echo "[INFO] Starting fresh fine-tune from baseline checkpoint."
    RESUME_ARGS=(--resume_from_checkpoint "$BASELINE_CKPT" --load_weights_only)
    WEIGHTS_ARGS=()
fi

srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
    --run_name "$RUN_NAME" \
    "${RESUME_ARGS[@]}" \
    \
    `# ── Architecture (must match M0 checkpoint exactly) ──` \
    --mesh_type fixed \
    --scan_angle_conditioning project \
    \
    `# ── Encoding order: random removes positional gradient-path bias ──` \
    --encoding_order random \
    \
    `# ── FSOI-corrected observation config ──` \
    --cfg_path configs/observation_config_finetune.yaml \
    \
    `# ── Data ──` \
    --data_path /scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7 \
    --train_start_date 2015-01-01 \
    --train_end_date   2024-01-01 \
    --val_start_date   2024-01-01 \
    --val_end_date     2025-01-01 \
    --train_window_days 12 \
    --val_window_days   12 \
    --sampling_mode random \
    --val_mode sequential \
    --val_stride_days 12 \
    --val_update_every_n_epochs 100 \
    \
    `# ── Fine-tune LR (lower than initial training) ──` \
    --lr "$LR" \
    --lr_schedule "$LR_SCHEDULE" \
    --warmup_pct "$WARMUP_PCT" \
    --warmup_start_factor "$WARMUP_START_FACTOR" \
    --min_lr "$MIN_LR" \
    --weight_decay 1e-4 \
    \
    `# ── Loss ──` \
    --loss_type "$LOSS_TYPE" \
    --huber_delta 0.5 \
    \
    `# ── Regularization (same as M0) ──` \
    --processor_dropout 0.1 \
    --node_dropout 0.05 \
    --encoder_dropout 0.1 \
    --decoder_dropout 0.1 \
    \
    `# ── Training length: shorter than initial training ──` \
    --seed 42 \
    --max_epochs 1000 \
    --disable_early_stopping \
    \
    `# ── Validation diagnostics ──` \
    --cache_val_windows \
    --val_cache_max_entries 16 \
    --val_csv_out_dir "val_csv/${RUN_NAME}" \
    --val_csv_num_batches 3 \
    --val_csv_every_n_epochs 10 \
    --val_csv_max_rows 50000 \
    --val_csv_sample_seed 42
