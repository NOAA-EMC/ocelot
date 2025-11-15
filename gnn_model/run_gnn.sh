#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -J gnn_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 01:00:00
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

# Launch training (env is propagated to ranks)
# TRUE GraphDOP FSOI: Uses sequential batches with forecast from previous window as background
# Sequential mode ensures temporal continuity for TRUE GraphDOP background (x_b = prev forecast)

# NEW TRAINING (from scratch) - SEQUENTIAL MODE with TRUE GraphDOP FSOI
# Start FSOI early (epoch 0) to see results quickly
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
    --sampling_mode sequential \
    --window_mode sequential \
    --enable_fsoi \
    --fsoi_mode fast \
    --fsoi_conventional_only \
    --fsoi_every_n_epochs=1 \
    --fsoi_start_epoch=0 \
    --fsoi_batches=3 \
    --max_epochs 100

# NOTE: Using BalancedSequentialShard (per-rank sequential)
# Each GPU processes its bins sequentially: GPU0 does bin1→bin2→bin3, GPU1 does bin4→bin5→bin6, etc.
# Sequential background works within each GPU (bin2 uses bin1 forecast on same GPU)

# RESUME from latest checkpoint (automatic) - SEQUENTIAL MODE
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
#     --resume_from_latest \
#     --sampling_mode sequential \
#     --window_mode sequential \
#     --enable_fsoi \
#     --fsoi_conventional_only \
#     --fsoi_mode fast \
#     --fsoi_every_n_epochs=10 \
#     --fsoi_start_epoch=10 \
#     --fsoi_batches=5

# RESUME from specific checkpoint - SEQUENTIAL MODE
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
#     --resume_from_checkpoint checkpoints/last.ckpt \
#     --sampling_mode sequential \
#     --window_mode sequential \
#     --enable_fsoi \
#     --fsoi_mode fast \
#     --fsoi_every_n_epochs=10 \
#     --fsoi_start_epoch=10 \
#     --fsoi_batches=5

# Resume training from the latest checkpoint (no FSOI)
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --resume_from_latest
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --resume_from_checkpoint checkpoints/last.ckpt
