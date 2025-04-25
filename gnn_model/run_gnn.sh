#!/bin/bash

#SBATCH -A da-cpu
#SBATCH -p fge
#SBATCH -q debug # gpuwf
#SBATCH -J gnn_train
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err

# Load Conda environment
source /scratch1/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Debug + performance
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# Launch training
srun --cpu_bind=cores python train_gnn.py