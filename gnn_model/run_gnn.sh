#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -J gnn_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

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
# Local NNJA mirror (shared path visible to ALL nodes)
export NNJA_LOCAL_ROOT=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/nnja-ai
export PYTHONPATH=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot/gnn_model:/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot:$PYTHONPATH

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# ==========================================================
# HOW TO LAUNCH TRAINING (CHOOSE EXACTLY ONE OPTION BELOW)
# ==========================================================

# ----------------------------------------------------------
# (A) CONTINUE TRAINING ON THE SAME YEAR / SAME ZARR
# ----------------------------------------------------------
# Use these only when you are training on the SAME dataset
# and want to continue training from the same checkpoint.
# These restore full trainer state (epoch, optimizer, windows).
#
# --- RANDOM SAMPLING CONTINUATION ---
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
    --resume_from_latest \
    --sampling_mode random \
    --window_mode random

# --- SEQUENTIAL SAMPLING CONTINUATION ---
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
#     --resume_from_latest \
#     --sampling_mode sequential \
#     --window_mode sequential
#
# NOTE:
# Do NOT use these if you are switching to another year.
# They will restore old windows (e.g., 2024 windows) and fail.


# ----------------------------------------------------------
# (B) START TRAINING ON A NEW YEAR / NEW ZARR (WARM START)
# ----------------------------------------------------------
# Use these when switching to a different dataset (e.g. 2024 → 2023).
# Loads model weights ONLY; trainer state and windows are NOT restored.
# This avoids the "restored window is outside dataset" crash.
#
# --- RANDOM SAMPLING WARM START (RECOMMENDED FOR MULTI-YEAR) ---
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
#     --init_from_ckpt checkpoints/last.ckpt \
#     --sampling_mode random \
#     --window_mode random
#
# --- SEQUENTIAL SAMPLING WARM START ---
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
#     --init_from_ckpt checkpoints/last.ckpt \
#     --sampling_mode sequential \
#     --window_mode sequential


# ----------------------------------------------------------
# IMPORTANT RULES:
# ----------------------------------------------------------
# - You MUST choose exactly ONE of:
#       --resume_from_latest   (continue same year)
#       --init_from_ckpt       (start new year)
#
# - Never use both at the same time.
# - For NEW YEAR training, always use --init_from_ckpt.
# - For SAME YEAR continuation, always use --resume_from_latest.
# ==========================================================

