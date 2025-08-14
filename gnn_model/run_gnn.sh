#!/bin/bash -l 
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH --gres=gpu:h100:2
#SBATCH -J gnn_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL


# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# PYTHONPATH
export PYTHONPATH=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/tmp/lib/python3.10/site-packages:$PYTHONPATH

# Debug + performance
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=OFF # INFO
# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# Launch a new training run
srun --cpu-bind=map_cpu:0,1,2,3 python train_gnn.py

# Resume training from the latest checkpoint (with verbose logging)
# srun --cpu-bind=map_cpu:0,1,2,3 python train_gnn.py --verbose --resume_from_checkpoint checkpoints/last.ckpt

