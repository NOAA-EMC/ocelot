#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J gnn_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 00:45:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Running on H100 nodes..."
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

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

# Prediction mode:
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python predict_gnn.py \
    --checkpoint checkpoints/gnn-epoch-epoch=134-val_loss-val_loss=0.03.ckpt \
    --start_date 2023-01-01 \
    --end_date 2023-01-10 \
    --output_dir predictions \
    --eval-mode
