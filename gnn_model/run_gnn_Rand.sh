#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp  # gpu-ai4wp gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -J gnn_train_Random
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --output=gnn_train_Random_%j.out
#SBATCH --error=gnn_train_Random_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Ensure relative paths (checkpoints/, configs/, etc.) resolve consistently.
# In Slurm the script is copied to /var/spool/...; prefer the submit directory.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
cd "$WORKDIR"
echo "Working directory: $PWD"

echo "Running on H100 nodes..."
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# PYTHONPATH
# export PYTHONPATH=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/tmp/lib/python3.10/site-packages:$PYTHONPATH

# Debug + performance
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_BLOCKING_WAIT=1          # explicit
export NCCL_SHM_DISABLE=1                  # avoid shm edge cases
export NCCL_NET_GDR_LEVEL=PHB              # conservative GPUDirect setting
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export PYTHONFAULTHANDLER=1

# Fix distributed timeout issues
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600    # 1 hour timeout
export TORCH_NCCL_DESYNC_DEBUG=1                # Better error reporting  
export NCCL_TIMEOUT=3600                        # NCCL timeout 1 hour
export TORCH_DISTRIBUTED_DEBUG=OFF # INFO
# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# ============================================================================
# MESH CONFIGURATION: Hierarchical vs Fixed
# ============================================================================
# ARCHITECTURE NOTES:
# • Fixed mesh = GraphCast's multiscale merged mesh (single node set + multiscale edges)
# • Hierarchical = U-Net-style latent hierarchy (L0=40962, L1=10242, L2=2562, L3=642 nodes)
#   - Only L0 interfaces with observations/predictions
#   - L1-L3 are latent levels with cross-scale attention
#   - L1→L0 conditioning provides gradient supervision to coarse level
# ============================================================================

# Launch training (env is propagated to ranks)
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py

# HIERARCHICAL MODE
# Resume training from the latest checkpoint in hierarchical mode
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --mesh_type hierarchical --mesh_levels 4 --resume_from_latest

# FIXED MODE (10-year training: 2015-2023 train pool, 2024 validation pool)
# - Uses merged multi-year zarrs in v7
# - Trains on random/sequential 12-day windows sampled from 2015-01-01..2024-01-01
# - Validates on rotating 8-day windows sampled from 2024-01-01..2025-01-01
# - Updates validation window every 5 epochs to reduce overhead
# - Caches recently used validation windows so looping is fast
# - Uses safer EarlyStopping defaults for rotating validation

# If you are continuing the *same* run configuration (same sampling/stride/windows),
# do a full resume (restores optimizer + callback state).

# Run naming:
# - For a brand-new experiment, set a new RUN_NAME (or move/delete its checkpoint dir).
# - For resubmitting after time-limit, reuse the SAME RUN_NAME so --resume_from_latest finds checkpoints.
# Example:
#   sbatch --export=RUN_NAME=gnn_v7_seviri_fix run_gnn.sh
# New experiment name (override on submit if desired).
# Example:
#   sbatch --export=ALL,RUN_NAME=seq_convfocus_nl16 run_gnn_modified_sequential.sh
RUN_NAME="${RUN_NAME:-Rand_TenYear_edge_awareSpatialMixing}"
echo "RUN_NAME=$RUN_NAME"

# Resume behavior:
# - Default is AUTO: if checkpoints/$RUN_NAME/last.ckpt exists, resume; otherwise start fresh.
# - To force a fresh start even if a checkpoint exists: resubmit with RESUME_FROM_LATEST=0.
# - To force resume: resubmit with RESUME_FROM_LATEST=1.
CKPT_DIR="checkpoints/${RUN_NAME}"
CKPT_LAST="${CKPT_DIR}/last.ckpt"

if [[ -z "${RESUME_FROM_LATEST+x}" ]]; then
	# Not provided by the submit environment -> auto-detect based on last.ckpt
	if [[ -f "$CKPT_LAST" ]]; then
		RESUME_FROM_LATEST=1
		echo "[INFO] Found existing checkpoint: $CKPT_LAST"
		echo "[INFO] Auto-setting RESUME_FROM_LATEST=1"
	else
		RESUME_FROM_LATEST=0
	fi
fi

echo "RESUME_FROM_LATEST=$RESUME_FROM_LATEST"

if [[ "$RESUME_FROM_LATEST" == "0" && -f "$CKPT_LAST" ]]; then
	echo "[WARN] Found existing checkpoint: $CKPT_LAST"
	echo "[WARN] RESUME_FROM_LATEST=0, so this job will start FRESH and may overwrite last.ckpt."
	echo "[WARN] To continue training, resubmit with: sbatch --export=ALL,RUN_NAME=${RUN_NAME},RESUME_FROM_LATEST=1 run_gnn.sh"
fi
if [[ "$RESUME_FROM_LATEST" == "1" && ! -f "$CKPT_LAST" ]]; then
	echo "[WARN] RESUME_FROM_LATEST=1 but no $CKPT_LAST found; training will start fresh."
fi

RESUME_ARGS=()
if [[ "$RESUME_FROM_LATEST" == "1" ]]; then
	RESUME_ARGS+=(--resume_from_latest)
fi


srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
	--run_name "$RUN_NAME" \
	"${RESUME_ARGS[@]}" \
	--mesh_type fixed \
	--scan_angle_conditioning project \
	--sampling_mode random \
	--cfg_path configs/observation_config.yaml \
	--data_path /scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7 \
	--train_start_date 2015-01-01 \
	--train_end_date 2024-01-01 \
	--val_start_date 2024-01-01 \
	--val_end_date 2025-01-01 \
	--train_window_days 12 \
	--val_window_days 12 \
	--val_mode sequential \
	--val_stride_days 12 \
	--val_update_every_n_epochs 50 \
	--num_layers 16 \
	--lr 1.5e-4 \
	--weight_decay 1e-4 \
	--processor_dropout 0.1 \
	--node_dropout 0.05 \
	--encoder_dropout 0.1 \
	--decoder_dropout 0.1 \
	--conv_weight_mult 3.0 \
	--huber_delta 0.5 \
	--seed 12345 \
	--max_epochs  3280\
	--cache_val_windows \
	--val_cache_max_entries 16 \
	--disable_early_stopping \
	--val_csv_out_dir "val_csv/${RUN_NAME}" \
	--val_csv_num_batches 3 \
	--val_csv_every_n_epochs 10 \
	--val_csv_max_rows 50000 \
	--val_csv_sample_seed 12345

# Resume from specific checkpoint
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --mesh_type hierarchical --resume_from_checkpoint checkpoints/last.ckpt
