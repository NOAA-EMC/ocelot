#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A gpu-emc-ai  # ai4ep; emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -J gnn_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH --output=gnn_pred_%j.out
#SBATCH --error=gnn_pred_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Running on H100 nodes..."
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

# Load Conda environment
source /scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh
conda activate gnn-env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OCELOT_DIR="$(cd "${GNN_MODEL_DIR}/.." && pwd)"

cd "${SLURM_SUBMIT_DIR:-${GNN_MODEL_DIR}}"

# Ensure we run the code from THIS checkout (not NNJA mirror)
export PYTHONPATH="${GNN_MODEL_DIR}:${OCELOT_DIR}:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "SLURM Node List: $SLURM_NODELIST"
echo "Visible GPUs on this node:"
nvidia-smi

# Prediction mode:
EXPT="test1-3years"
CKPT="checkpoints/$EXPT/ep181.ckpt"
OUT_DIR="predictions/$EXPT"

srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python predict_gnn.py \
    --checkpoint $CKPT \
    --start_date 2025-02-28 \
    --end_date 2025-03-03 \
    --output_dir $OUTDIR \
    --eval-mode  # comment out to run in inference mode
    # Evaluation mode: Predict on obs-space for all instruments (AMSUA, aircraft, etc.) with ground truth comparisons.
    #                  The last timebin is held as the target bin, consistent with training.
    # Inference mode: Predict on mesh-grid for the instruments specified in configs/mesh_config.yaml.
    #                 As the target bin is not used in this mode, all timebins are used as input.

