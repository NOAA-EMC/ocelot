#!/bin/bash

#SBATCH -A da-cpu
#SBATCH -p fge
#SBATCH -q gpuwf
#SBATCH -J gnn_train
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 03:30:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err

# Load Conda environment
source /scratch1/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Debug + performance
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_LEVEL=NVL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NCCL_ALGO=Ring

echo "Visible GPUs on this node:"
nvidia-smi

srun --cpu_bind=cores python train_gnn.py
