#!/bin/bash -l
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:2
#SBATCH -J gnn_train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 1-00:00:00
#SBATCH --output=gnn_train_%j.out
#SBATCH --error=gnn_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Running on H100 nodes..."
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

# Load Conda environment
if command -v module >/dev/null 2>&1; then
	module load conda >/dev/null 2>&1 || true
fi
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
export PYTHONPATH=/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/src/ocelot/gnn_model:/scratch3/NCEPDEV/da/Azadeh.Gholoubi/NNJA/ocelot:$PYTHONPATH

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

cd /scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/src/ocelot/gnn_model/

# Optional graph shard cache. The cache stores precomputed per-bin graph shards;
# in domain mode each rank stores its own already-sharded graph for the current
# world size, halo hops, config, and window settings.
RUN_NAME="${RUN_NAME:-run_gnn_3}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/graph_cache/${RUN_NAME}}"
GRAPH_CACHE_READ="${GRAPH_CACHE_READ:-1}"
GRAPH_CACHE_WRITE="${GRAPH_CACHE_WRITE:-1}"
PRECOMPUTE_GRAPH_CACHE="${PRECOMPUTE_GRAPH_CACHE:-0}"

echo "RUN_NAME=$RUN_NAME"
echo "GRAPH_CACHE_DIR=$GRAPH_CACHE_DIR"
echo "GRAPH_CACHE_READ=$GRAPH_CACHE_READ"
echo "GRAPH_CACHE_WRITE=$GRAPH_CACHE_WRITE"
echo "PRECOMPUTE_GRAPH_CACHE=$PRECOMPUTE_GRAPH_CACHE"

GRAPH_CACHE_ARGS=()
if [[ -n "$GRAPH_CACHE_DIR" ]]; then
	GRAPH_CACHE_ARGS+=(--graph_cache_dir "$GRAPH_CACHE_DIR")
fi
if [[ "$GRAPH_CACHE_READ" == "1" ]]; then
	GRAPH_CACHE_ARGS+=(--graph_cache_read)
fi
if [[ "$GRAPH_CACHE_WRITE" == "1" ]]; then
	GRAPH_CACHE_ARGS+=(--graph_cache_write)
fi
if [[ "$PRECOMPUTE_GRAPH_CACHE" == "1" ]]; then
	GRAPH_CACHE_ARGS+=(--precompute_graph_cache)
fi

# Launch domain-sharded training with mesh halo exchange.
srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python /scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/src/ocelot/gnn_model/train_gnn.py \
	--run_name="$RUN_NAME" \
	"${GRAPH_CACHE_ARGS[@]}" \
	--hidden_dim=256 \
	--parallelization_strategy=domain \
	--domain_halo_hops=1 \
	--num_nodes=8 \
	--zarr_cache_max_size_bytes=67108864 \
	--decoder_dst_chunk_size=1024 \
	--encoder_dst_chunk_size=4096 \
	--encoder_dst_chunk_threshold=4096 \
	--val_csv_out_dir=val_csv \
	--zarr_cache_max_size_bytes=67108864 \
	--train_num_workers=8 \
	--val_num_workers=4 \
	--dataloader_prefetch_factor=4 \
	--disable_pin_memory \
	# --resume_from_latest \
	# --limit_val_batches=1 

	# pin_memory is disabled to avoid the known PyTorch instability with
	# (pin_memory=True, persistent_workers=True, reload_dataloaders_every_n_epochs>0)
	# documented at https://github.com/pytorch/pytorch/issues/91252 .
	# The resample callbacks rely on reload_dataloaders_every_n_epochs=1, so
	# we keep that and trade off pinned host memory for stability.
	# --disable_val_csv

# HIERARCHICAL MODE
# Resume training from the latest checkpoint in hierarchical mode
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --mesh_type hierarchical --mesh_levels 4 --resume_from_latest

# FIXED MODE
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --mesh_type fixed --resume_from_latest

# Resume from specific checkpoint
# srun --export=ALL --kill-on-bad-exit=1 --cpu-bind=cores python train_gnn.py --mesh_type hierarchical --resume_from_checkpoint checkpoints/last.ckpt
