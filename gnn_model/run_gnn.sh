#!/bin/bash

# Request resources with SLURM
#SBATCH -p fge
#SBATCH -q gpuwf
#SBATCH -t 01:30:00
#SBATCH --nodes=1
#SBATCH -A da-cpu
#SBATCH --job-name=gnn_train
#SBATCH --output=gnn_train_%j.out

# Load Conda environment
source /scratch1/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Avoid CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run Python script
python train_gnn.py

