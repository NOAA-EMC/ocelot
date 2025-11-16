#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J fsoi_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH --output=fsoi_eval_%j.out
#SBATCH --error=fsoi_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "FSOI Evaluation on trained checkpoint..."
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

# Environment variables
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

# Paths
export NNJA_LOCAL_ROOT=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/nnja-ai
export PYTHONPATH=/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot/gnn_model:/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot:$PYTHONPATH

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# =====================================================================
# FSOI EVALUATION ON TRAINED CHECKPOINT
# =====================================================================
# This runs ONLY validation (no training) with FSOI enabled
# Uses sequential sampling for validation to enable sequential background
# =====================================================================

CHECKPOINT_PATH="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/add_fsoi/ocelot/gnn_model/checkpoints/last.ckpt"
OUTPUT_DIR="fsoi_evaluation_test1"

echo "=========================================="
echo "FSOI Evaluation Configuration"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "FSOI batches: 5"
echo "Sequential sampling: YES (for two-state adjoint)"
echo "=========================================="

srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py \
    --resume_from_checkpoint "$CHECKPOINT_PATH" \
    --sampling_mode sequential \
    --window_mode sequential \
    --enable_fsoi \
    --fsoi_conventional_only \
    --fsoi_batches 5 \
    --fsoi_every_n_epochs 1 \
    --fsoi_start_epoch 0 \
    --default_root_dir "$OUTPUT_DIR" \
    --max_epochs 1 \
    --limit_train_batches 0 \
    --limit_val_batches 5 \
    --devices 1 \
    --num_nodes 1

# Explanation of flags:
# --resume_from_checkpoint: Load trained model weights
# --sampling_mode sequential: Use sequential bin order for validation
# --window_mode sequential: Use sequential data windows
# --enable_fsoi: Enable FSOI computation
# --fsoi_conventional_only: Only compute for conventional obs (surface + radiosonde)
# --fsoi_batches 5: Compute FSOI on first 5 validation batches per GPU
# --max_epochs 1: Only run 1 epoch (just validation, no training)
# --limit_train_batches 0: Skip training completely
# --limit_val_batches 5: Only run 5 validation batches (faster evaluation)
# --default_root_dir: Save FSOI results to separate directory

echo ""
echo "=========================================="
echo "FSOI Evaluation Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - $OUTPUT_DIR/fsoi_results_conventional/detailed/"
echo "  - $OUTPUT_DIR/fsoi_results_conventional/plots/"
echo ""
echo "To analyze results:"
echo "  cd $OUTPUT_DIR/fsoi_results_conventional/detailed"
echo "  head -20 fsoi_epoch0_batch0.csv  # First batch (climatological)"
echo "  head -20 fsoi_epoch0_batch1.csv  # Second batch (sequential background)"
echo ""
