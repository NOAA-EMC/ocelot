"""
Example training script for HeteroObservationGraphModel using PyTorch Lightning.
"""

from pprint import pprint
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
import argparse
import os

import sys

from ocelot.graph_data_module import WeatherDataModule
from ocelot.timing_utils import timing_resource_decorator
from ocelot.observation_config import load_observation_config
from ocelot.utils import DEFAULT_DTYPE, instantiate_class_from_path
from ocelot.logger import logger
from ocelot.memory_callback import CombinedMemoryCallback
# Set default dtype to float32 to prevent dtype mismatch issues
torch.set_default_dtype(DEFAULT_DTYPE)

import torch.distributed as dist


def print_memory_summary(stage_name, rank=None):
    """
    Print detailed GPU memory usage at a specific stage of training.
    
    Args:
        stage_name: Name of the pipeline stage (e.g., "After Model Init")
        rank: GPU rank (if None, will get from dist if available)
    """
    if not torch.cuda.is_available():
        return
    
    # Get rank
    if rank is None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    
    # Only print from rank 0 to avoid spam (or print all if you want)
    if rank != 0:
        return
    
    # Get memory stats for all GPUs on this node
    num_gpus = torch.cuda.device_count()
    
    logger.info("\n" + "="*80)
    logger.info(f"🔍 MEMORY SNAPSHOT: {stage_name}")
    logger.info("="*80)
    
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)    # GB
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)  # GB
        total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
        free = total - allocated
        
        logger.info(f"\n📊 GPU {gpu_id}:")
        logger.info(f"   Total Memory:     {total:.2f} GB")
        logger.info(f"   Allocated:        {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        logger.info(f"   Reserved:         {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        logger.info(f"   Free:             {free:.2f} GB ({free/total*100:.1f}%)")
        logger.info(f"   Peak Allocated:   {max_allocated:.2f} GB ({max_allocated/total*100:.1f}%)")
        logger.info(f"   Fragmentation:    {(reserved - allocated):.2f} GB")
        
        # Get detailed memory breakdown if available
        try:
            memory_stats = torch.cuda.memory_stats(gpu_id)
            active_bytes = memory_stats.get('active_bytes.all.current', 0) / (1024**3)
            inactive_bytes = memory_stats.get('inactive_split_bytes.all.current', 0) / (1024**3)
            
            logger.info(f"   Active Tensors:   {active_bytes:.2f} GB")
            logger.info(f"   Inactive/Cached:  {inactive_bytes:.2f} GB")
        except:
            pass
    
    logger.info("="*80 + "\n")


def make_profiler(enabled=True, profile_per_module=False):
    if not enabled:
        return None  # disables profiling cleanly

    # Use a simpler schedule that's more robust for varying training lengths
    # This schedule will profile steps 3-7 (5 steps total) in each epoch
    return pl.profilers.PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./lightning_logs/profiler'),
        schedule=torch.profiler.schedule(
            skip_first=2,   # skip first 2 batches (reduce from 6)
            wait=1,         # wait 1 step before profiling starts
            warmup=1,       # collect warmup data for 1 step
            active=3,       # profile 3 steps (reduce from 5)
            repeat=0        # repeat=0 means continue cycling indefinitely (no NONE state)
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=profile_per_module,  # Enable per-module profiling
    )



class ProfilerSummaryCallback(pl.Callback):
    def __init__(self, save_dir="lightning_logs/profiler_summary", enabled=True):
        super().__init__()
        self.save_dir = save_dir
        self.enabled = enabled

    def on_fit_end(self, trainer, pl_module):
        # Only run profiler summary on rank 0 to avoid conflicts in distributed training
        if trainer.global_rank != 0:
            return
            
        if not self.enabled:
            logger.info("🟡 Profiler summary disabled.")
            return

        prof = getattr(trainer.profiler, "profiler", None)
        if prof is None:
            logger.info("⚠️ No PyTorch profiler found in this trainer.")
            return

        import os
        os.makedirs(self.save_dir, exist_ok=True)

        logger.info("\n" + "="*80)
        logger.info("📊 PROFILER SUMMARY - GPU Time")
        logger.info("="*80)
        
        # Table sorted by CUDA time
        cuda_time_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        logger.info(cuda_time_table)
        
        logger.info("\n" + "="*80)
        logger.info("💾 PROFILER SUMMARY - GPU Memory Usage (Ranked by Layer)")
        logger.info("="*80)
        
        # Get key averages grouped by stack trace (shows module names and code locations)
        key_averages_stack = prof.key_averages(group_by_stack_n=5)
        
        # Filter for operations with memory usage and sort by memory
        memory_ops = []
        for evt in key_averages_stack:
            if evt.cuda_memory_usage > 0:
                # Extract module/function name from stack trace
                stack_info = evt.key
                # Try to extract meaningful module name
                if 'nn.Module:' in stack_info:
                    # Extract module class name
                    module_name = stack_info.split('nn.Module:')[1].split('(')[0].strip()
                else:
                    module_name = stack_info.split('\n')[0] if '\n' in stack_info else stack_info
                
                memory_ops.append({
                    'name': module_name[:80],  # Truncate long names
                    'full_stack': stack_info,
                    'memory_mb': evt.cuda_memory_usage / (1024 * 1024),  # Convert to MB
                    'count': evt.count,
                    'cpu_time_ms': evt.cpu_time_total / 1000,  # Convert to ms
                    'cuda_time_ms': evt.cuda_time_total / 1000,  # Convert to ms
                })
        
        # Sort by memory usage (descending)
        memory_ops.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        # Print top memory consumers
        logger.info(f"\n{'Rank':<6} {'Memory (MB)':<15} {'Count':<8} {'CUDA Time (ms)':<18} {'Module/Location'}")
        logger.info("-" * 120)
        
        for i, op in enumerate(memory_ops[:25], 1):  # Top 25
            logger.info(f"{i:<6} {op['memory_mb']:>12.2f}   {op['count']:<8} {op['cuda_time_ms']:>15.2f}   {op['name']}")
        
        total_memory = sum(op['memory_mb'] for op in memory_ops)
        logger.info("-" * 120)
        logger.info(f"{'TOTAL':<6} {total_memory:>12.2f} MB")
        logger.info("="*80 + "\n")

        # Save all summaries to files
        with open(f"{self.save_dir}/cuda_time_summary.txt", "w") as f:
            f.write(cuda_time_table)
        
        with open(f"{self.save_dir}/memory_summary.txt", "w") as f:
            f.write(f"{'Rank':<6} {'Memory (MB)':<15} {'Count':<8} {'CUDA Time (ms)':<18} {'Module/Location'}\n")
            f.write("-" * 120 + "\n")
            for i, op in enumerate(memory_ops, 1):
                f.write(f"{i:<6} {op['memory_mb']:>12.2f}   {op['count']:<8} {op['cuda_time_ms']:>15.2f}   {op['name']}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'TOTAL':<6} {total_memory:>12.2f} MB\n")
        
        # Save detailed stack traces
        with open(f"{self.save_dir}/memory_detailed_stacks.txt", "w") as f:
            f.write("DETAILED STACK TRACES FOR TOP MEMORY CONSUMERS\n")
            f.write("=" * 120 + "\n\n")
            for i, op in enumerate(memory_ops[:25], 1):
                f.write(f"Rank {i}: {op['memory_mb']:.2f} MB | Count: {op['count']} | CUDA Time: {op['cuda_time_ms']:.2f} ms\n")
                f.write("-" * 120 + "\n")
                f.write(op['full_stack'])
                f.write("\n\n" + "=" * 120 + "\n\n")
        
        logger.info(f"✅ Profiler summaries saved to {self.save_dir}/")
        logger.info(f"   - cuda_time_summary.txt")
        logger.info(f"   - memory_summary.txt")
        logger.info(f"   - memory_detailed_stacks.txt (with code locations)")


def check_ddp_environment():
    """Verify DDP setup, CUDA, and seed synchronization before training (rank-0 only prints)."""
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"  # fallback for CPU

    # Initialize if needed
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception as e:
            logger.warning(f"[Warning] DDP init skipped: {e}")
            return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # # Sync RNG seeds
    # seed_everything(42, workers=True)

    # Quick communication test
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

    # Print only from rank 0
    if rank == 0:
        logger.info("====== DDP Environment Check ======")
        logger.info(f"✅ World size: {world_size}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        logger.info(f"✅ All-reduce OK (sum of ranks = {tensor.item()})")
        logger.info(f"✅ Device map example: cuda:{local_rank}")
        logger.info("===================================")


def parse_args(default_args=None):
    parser = argparse.ArgumentParser(description="Train HeteroObservationGraphModel with multi-GPU support")
    parser.add_argument('--debug_gradients', action='store_true',
                        help='If set, print gradient stats after each backward pass for debugging.')
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler and summary.")
    parser.add_argument("--profile_per_module", action="store_true", help="Enable per-module profiling (requires --profile).")
    parser.add_argument("--monitor_memory", action="store_true", help="Enable detailed memory monitoring during training.")
    parser.add_argument("--memory_log_interval", type=int, default=10, help="Log memory every N training steps.")
    parser.add_argument("--scatter_chunk_size", type=int, default=50000, 
                       help="Chunk size for scatter operations (smaller = less memory, slower)")
    
    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--hidden_dim", type=int, help="Hidden dimension size")
    model_group.add_argument("--hidden_layers", type=int, help="Number of hidden layers")
    model_group.add_argument("--mesh_resolution", type=float, help="Resolution of the mesh")
    model_group.add_argument("--mesh_feature_dim", type=int, help="Dimension of the static mesh features")
    model_group.add_argument("--cutoff_factor", type=float, help="Cutoff factor for mesh connections")
    model_group.add_argument("--num_neighbors", type=int, help="Number of neighbors for KNN")
    model_group.add_argument("--mesh_gnn_layers", type=int, help="Number of GNN layers in the mesh model")
    model_group.add_argument(
        "--hierarchical_processor_type",
        type=str,
        choices=["interaction", "transformer"],
        default="interaction",
        help="When --hierarchical: use InteractionNet stack (interaction) or "
        "HierarchicalSlidingWindowTransformer (transformer)",
    )
    model_group.add_argument(
        "--processor_window",
        type=int,
        default=4,
        help="Temporal cache window for hierarchical transformer (default: 4)",
    )
    model_group.add_argument(
        "--processor_depth",
        type=int,
        default=2,
        help="Temporal transformer depth per level (default: 2)",
    )
    model_group.add_argument(
        "--processor_heads",
        type=int,
        default=4,
        help="Attention heads for hierarchical transformer (must divide hidden_dim)",
    )
    model_group.add_argument(
        "--processor_dropout",
        type=float,
        default=0.0,
        help="Dropout for hierarchical transformer",
    )
    model_group.add_argument(
        "--no_processor_causal_mask",
        action="store_true",
        help="Disable causal temporal mask for hierarchical transformer (default: causal on)",
    )
    model_group.add_argument(
        "--no_processor_cross_scale",
        action="store_true",
        help="Disable cross-scale attention in hierarchical transformer (default: cross-scale on)",
    )
    model_group.add_argument(
        "--spatial_mixing_steps",
        type=int,
        default=1,
        help="Per-timestep spatial neighbor mixing steps at each level (transformer)",
    )
    model_group.add_argument(
        "--processor_attn_chunk_size",
        type=int,
        default=2048,
        help="Chunk size along mesh nodes for MultiheadAttention in hierarchical transformer "
        "(large meshes need chunking to avoid CUDA invalid configuration; default 2048)",
    )
    model_group.add_argument("--num_processor_rollout_steps", type=int, default=1,
                             help="Number of latent processor rollout steps (processor applied repeatedly on mesh state)")
    model_group.add_argument("--processor_rollout_alpha", type=float, default=None,
                             help="Blend factor for rollout: state = (1-alpha)*state + alpha*processor(state). "
                                  "Default: 1/num_processor_rollout_steps for better gradient flow")
    model_group.add_argument("--step_length", type=int, help="Step length for the model")
    model_group.add_argument("--num_heads", type=int, help="Number of heads for multi-head attention")
    model_group.add_argument("--scan_angle_embed_dim", type=int, default=8, help="Dimension of scan angle embedding")
    model_group.add_argument("--use_gat_encoder_decoder", action="store_true",
                             help="Use GAT instead of InteractionNet for encoder and decoder")
    model_group.add_argument("--gat_chunk_size", type=int, default=0,
                             help="Chunk receiver nodes during GAT inference to reduce peak GPU memory (0=disabled)")

    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--batch_size", type=int, help="Batch size per GPU")
    train_group.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    train_group.add_argument("--lr", type=float, help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    train_group.add_argument('--plot_on_test', action='store_true',
                             help='Generate and save plots during the test phase.')
    train_group.add_argument("--num_workers", type=int, help="Number of workers for dataloaders")
    train_group.add_argument("--loss", type=str, help="Loss function to use")
    train_group.add_argument("--obs_config", type=str, default="default",
                             help="Observation config variant to load (e.g. 'default', 'urma')")
    train_group.add_argument("--exp_name", type=str, default="default_experiment",
                             help="Name of the experiment for logging")
    train_group.add_argument("--version", type=str, default='version_0',
                             help="Logger version (subfolder). If omitted, Lightning auto-increments.")
    train_group.add_argument("--validation_interval", type=int, default=32,
                             help="Base number of training steps between validations (scaled inversely by world_size)")
    train_group.add_argument("--limit_val_batches", type=float, default=1.0,
                             help="Fraction (0.0-1.0) or number of validation batches to use per validation run (default: 1.0 = all batches)")
    train_group.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable EarlyStopping on val_loss (otherwise patience is --early_stopping_patience).",
    )
    train_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="EarlyStopping: epochs without val_loss improvement before stopping (default: 5).",
    )

    # Multi-GPU parameters
    gpu_group = parser.add_argument_group('Multi-GPU Configuration')
    gpu_group.add_argument("--strategy", type=str, choices=["ddp", "ddp_spawn", "deepspeed", "fsdp", "auto"],
                           help="Distributed training strategy")
    gpu_group.add_argument("--accelerator", type=str, help="Accelerator type (gpu, cpu, tpu)")
    gpu_group.add_argument("--devices", type=int, help="Number of GPUs to use")
    gpu_group.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training")
    gpu_group.add_argument("--precision", type=str, help="Training precision")

    # Data parameters
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--data_path", type=str, help="Path to data directory")
    data_group.add_argument("--start_date", type=str, help="Start date for training data")
    data_group.add_argument("--end_date", type=str, help="End date for training data")
    data_group.add_argument("--observation_config", type=dict, help="Observation_config for data selection")
    data_group.add_argument("--feature_stats", type=dict, help="Observation stats  for data selection")
    data_group.add_argument("--instrument_weights", type=dict, help="Per-instrument loss weight, keyed like observation_config")
    data_group.add_argument("--increment_stats", type=dict, help="Per-variable DA increment (anal-ges) mean/std in normalized-field space, keyed like the state feature list. See ocelot/compute_increment_stats.py.")
    data_group.add_argument(
        "--normalize_increment",
        action="store_true",
        help="Rescale the DA increment target (anal-ges) by the offline-computed "
        "per-variable increment stats (--increment_stats) instead of relying "
        "on implicit field normalization alone. See graphcast_residuals.md.",
    )
    data_group.add_argument(
        "--target_is_anal",
        action="store_true",
        help="Predict the full analysis field directly (y = anal) instead of the "
        "DA increment (y = anal - ges). Mutually exclusive with "
        "--normalize_increment, which only applies to the increment target.",
    )
    data_group.add_argument(
        "--gradient_loss_weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary edge/gradient loss on the state target: penalizes "
        "error in the value difference between neighboring grid nodes (a k-NN graph "
        "over the state grid), not just the raw value, to sharpen fronts and other "
        "discontinuities that a plain per-node loss tends to blur out. 0 disables it "
        "(default).",
    )
    data_group.add_argument(
        "--multiscale_loss_weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary multi-scale loss: bins target nodes by lon/lat at "
        "each of --multiscale_scales and scores the cell-averaged prediction against "
        "the cell-averaged target, in addition to the per-node loss, so synoptic-scale "
        "structure is scored directly. 0 disables it (default).",
    )
    data_group.add_argument(
        "--multiscale_scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
        help="Bin sizes in degrees lat/lon for --multiscale_loss_weight (default: 0.5 1.0 2.0 4.0).",
    )
    data_group.add_argument(
        "--delta_time",
        type=int,
        default=12,
        choices=[1, 6, 12],
        help="Time interval in hours between consecutive bins (1, 6, or 12). Default 12.",
    )
    data_group.add_argument(
        "--num_target_bins",
        type=int,
        default=1,
        help="Number of consecutive bins after the input bin to use as targets (k). "
        "Default 1 = single next-bin target (names stay {obs}_{inst}_target). "
        "For k>1, target node types are {obs}_{inst}_h{0..k-1}_target.",
    )
    data_group.add_argument(
        "--no_obs_autoregressive",
        action="store_true",
        help="When num_target_bins>1, disable autoregressive forward (one encode+processor, "
        "decode all horizons from the same mesh). Default is autoregressive when k>1.",
    )
    data_group.add_argument(
        "--no_obs_ar_teacher_forcing",
        action="store_true",
        help="When autoregressive, use previous-step predictions for feedback encode instead of ground-truth y.",
    )

    # Other arguments
    misc_group = parser.add_argument_group('Miscellaneous Configuration')
    misc_group.add_argument("--exp_type", type=str, default="global", help="Experiment type: 'global' or 'regional' (limited-area mesh and dataset)")
    misc_group.add_argument("--model_path", type=str, help="a full class path string for the model")
    misc_group.add_argument("--load_ckpt_path", type=str, help="Path to a checkpoint to load.")
    misc_group.add_argument(
        "--action",
        type=str,
        choices=['train', 'test', 'train_and_test', 'start_train', 'predict'],
        help="train / start_train: fit; train_and_test: fit then test; "
        "test / predict: no training, require --load_ckpt_path (test runs test_dataloader, "
        "predict runs predict_dataloader).",
    )
    misc_group.add_argument("--levels", type=int, help="Number of levels in the mesh hierarchy")
    misc_group.add_argument(
        "--mesh_splits",
        type=int,
        default=6,
        help="Icosahedral refinement splits passed to create_mesh (default: 6)",
    )
    misc_group.add_argument(
        "--x_min", type=float, default=None,
        help="Minimum projected x-coordinate of the regional grid (e.g. metres in LCC). "
        "Required when model_path points to RegionalHeteroObservationGraphModel.",
    )
    misc_group.add_argument(
        "--x_max", type=float, default=None,
        help="Maximum projected x-coordinate of the regional grid.",
    )
    misc_group.add_argument(
        "--y_min", type=float, default=None,
        help="Minimum projected y-coordinate of the regional grid.",
    )
    misc_group.add_argument(
        "--y_max", type=float, default=None,
        help="Maximum projected y-coordinate of the regional grid.",
    )
    misc_group.add_argument("--plot", action='store_true', help="Generate plots during testing")
    misc_group.add_argument(
        "--hierarchical",
        action='store_true',
        help="Use multi-level mesh with up/down edges and HierarchicalProcessor (U-Net style)",
    )
    misc_group.add_argument("--output_std", action='store_true', help="Whether to output standard deviation")
    misc_group.add_argument("--restore_opt", action='store_true', help="Restore optimizer state from checkpoint")
    misc_group.add_argument("--n_example_pred", type=int, help="Number of example predictions to generate")
    misc_group.add_argument("--graph", type=str, help="Type of graph to use (e.g., 'hetero')")
    misc_group.add_argument(
        "--predict_output",
        type=str,
        default=None,
        help="When action=predict: path to save outputs (.pt). "
        "Default: predict_outputs/<exp_name>/<version>/predictions.pt",
    )

    # Set defaults from dictionary, allowing command-line overrides
    if default_args:
        parser.set_defaults(**default_args)

    return parser.parse_args()


@timing_resource_decorator
def set_model(args_update):
    """
    Set up the model, trainer, and datamodule based on arguments.
    """

    seed_everything(42, workers=True)
    
    # Memory checkpoint 1: Before model initialization
    print_memory_summary("1. Before Model Init")

    # Initialize model and data module
    model = instantiate_class_from_path(args_update.model_path, args_update)
    
    # Memory checkpoint 2: After model initialization
    print_memory_summary("2. After Model Init")
    datamodule = WeatherDataModule(
        data_path=args_update.data_path,
        start_date=args_update.start_date,
        end_date=args_update.end_date,
        observation_config=args_update.observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=args_update.batch_size,
        num_workers=args_update.num_workers,
        args=args_update
    )

    # Memory checkpoint 3: After datamodule creation
    print_memory_summary("3. After DataModule Init")
    
    # --- Setup and dummy forward pass to initialize lazy layers for DDP ---
    datamodule.setup('fit')
    logger.info("Setting up model mesh...")
    model.setup_mesh() # This creates the mesh_x buffer
    
    # Memory checkpoint 4: After mesh setup
    print_memory_summary("4. After Mesh Setup")

    # Setup PyTorch Lightning logger (CSV)
    csv_logger = CSVLogger(
        save_dir="logs",
        name=args_update.exp_name,
        version=args_update.version
    )

    # Update args_update.version with actual version from logger
    # This ensures model's self.args.version matches the logger version
    # (important when version is None and gets auto-incremented to version_0, version_1, etc.)
    args_update.version = csv_logger.version

    # Checkpoints: align with logger layout logs/<exp_name>/<version>/ so runs don't clobber each other
    ckpt_dir = os.path.join("checkpoints", str(args_update.exp_name), str(args_update.version))
    os.makedirs(ckpt_dir, exist_ok=True)
    args_update.checkpoint_dir = ckpt_dir  # also used for train_model.ckpt after fit()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_dir,
        filename="hetero-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=3,
        mode='min',
    )
    logger.info(f"Model checkpoints will be saved under: {ckpt_dir}")

    logger.info('---cuda:')
    logger.info(torch.__version__)
    logger.info(torch.version.cuda)  # Should match CUDA 12.x
    if torch.cuda.is_available():
        logger.info('cuda available')
    # Initialize trainer with multi-GPU settings
    # Determine training strategy
    is_distributed = torch.cuda.device_count() > 1 or args_update.num_nodes > 1
    strategy_config = args_update.strategy if is_distributed else "auto"

    if strategy_config in ["ddp", "ddp_spawn"]:
        strategy_config = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,  # Reduce memory copies
            static_graph=False,  # Set True if your graph structure never changes
        )

    # Memory checkpoint 5: Before trainer initialization
    print_memory_summary("5. Before Trainer Init")

    # Calculate validation interval scaled by world size
    # With more GPUs, each GPU sees fewer steps per epoch, so we validate more frequently
    world_size = args_update.devices * args_update.num_nodes
    val_check_interval = max(1, args_update.validation_interval // world_size)
    
    # Determine number of training batches per epoch to decide validation strategy
    # IMPORTANT: Check this BEFORE creating the trainer to avoid validation errors
    try:
        train_dataloader = datamodule.train_dataloader()
        num_train_batches_total = len(train_dataloader)
        
        # When using DDP, each GPU sees a subset of the data
        # The dataloader length before trainer wraps it is the TOTAL batches
        # We need to divide by world_size to get batches per GPU
        if is_distributed:
            num_train_batches = num_train_batches_total // world_size
            logger.info(f"📊 Training batches: {num_train_batches_total} total, {num_train_batches} per GPU (world_size={world_size})")
        else:
            num_train_batches = num_train_batches_total
            logger.info(f"📊 Training batches per epoch: {num_train_batches}")
    except Exception as e:
        logger.warning(f"Could not determine number of training batches: {e}")
        logger.warning("Falling back to epoch-based validation")
        num_train_batches = 0
    
    # Check validation dataloader
    try:
        val_dataloader = datamodule.val_dataloader()
        if val_dataloader is None:
            logger.error("❌ Validation dataloader is None! Validation will not run.")
            num_val_batches = 0
        else:
            num_val_batches = len(val_dataloader)
            logger.info(f"📊 Validation batches: {num_val_batches}")
            if num_val_batches == 0:
                logger.warning("⚠️  Validation dataloader has 0 batches!")
    except Exception as e:
        logger.warning(f"Could not check validation dataloader: {e}")
        num_val_batches = 0
    
    # Adjust limit_val_batches to ensure at least 1 batch is used
    if num_val_batches > 0:
        if args_update.limit_val_batches < 1.0:
            # It's a fraction (0.0 to 1.0)
            batches_to_use = max(1, int(num_val_batches * args_update.limit_val_batches))
            adjusted_limit = batches_to_use / num_val_batches
            if adjusted_limit != args_update.limit_val_batches:
                logger.info(f"📊 Adjusted limit_val_batches from {args_update.limit_val_batches:.2f} to {adjusted_limit:.2f} to ensure at least 1 batch")
                args_update.limit_val_batches = adjusted_limit
        elif args_update.limit_val_batches >= 1.0:
            # It's an absolute number of batches
            batches_to_use = int(args_update.limit_val_batches)
            if batches_to_use < 1:
                logger.warning(f"⚠️  limit_val_batches={args_update.limit_val_batches} rounds to 0, setting to 1")
                args_update.limit_val_batches = 1
            elif batches_to_use > num_val_batches:
                logger.warning(f"⚠️  limit_val_batches={batches_to_use} exceeds total val batches ({num_val_batches}), using all batches")
                args_update.limit_val_batches = num_val_batches
    
    # Decide between step-based and epoch-based validation
    # Use step-based ONLY if validation can run at least once per epoch
    if num_train_batches > 0 and val_check_interval < num_train_batches:
        # Step-based validation is feasible - will run mid-epoch
        use_step_based_validation = True
        validation_config = {"val_check_interval": val_check_interval}
        logger.info(f"✅ Using step-based validation: every {val_check_interval} steps per GPU")
        logger.info(f"   (base={args_update.validation_interval}, world_size={world_size}, batches_per_epoch={num_train_batches})")
        logger.info(f"   Validation will run approximately {num_train_batches // val_check_interval} times per epoch")
    else:
        # Fall back to epoch-based validation
        # This ensures validation ALWAYS runs at the end of each epoch
        use_step_based_validation = False
        validation_config = {"check_val_every_n_epoch": 1}
        if num_train_batches > 0:
            logger.info(f"⚠️  Step-based validation not feasible (interval={val_check_interval} >= batches={num_train_batches})")
        logger.info(f"✅ Using epoch-based validation: every 1 epoch")
        logger.info(f"   Validation will run at the END of each epoch (after all {num_train_batches} training batches)")
        logger.info(f"   To use step-based validation, reduce --validation_interval or increase dataset size")

    # Only enable profiler on rank 0 in distributed training to reduce overhead
    use_profiler = args_update.profile
    # In distributed mode, only profile rank 0
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_profiler = use_profiler and (local_rank == 0)
    
    profile_per_module = args_update.profile_per_module if use_profiler else False
    profiler = make_profiler(enabled=use_profiler, profile_per_module=profile_per_module)
    callbacks = [checkpoint_callback]
    if not getattr(args_update, "no_early_stopping", False):
        es_patience = getattr(args_update, "early_stopping_patience", 5)
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=es_patience, mode="min")
        )
        logger.info(f"EarlyStopping enabled (monitor=val_loss, patience={es_patience})")
    else:
        logger.info("EarlyStopping disabled (--no_early_stopping)")
    if use_profiler:
        callbacks.append(ProfilerSummaryCallback(enabled=True))
    
    # Add memory monitoring callback if requested
    if args_update.monitor_memory:
        # Save memory CSV in the same logs directory as other experiment logs
        memory_csv_path = f"logs/{args_update.exp_name}/memory_log.csv"
        os.makedirs(f"logs/{args_update.exp_name}", exist_ok=True)
        
        memory_callback = CombinedMemoryCallback(
            log_every_n_steps=args_update.memory_log_interval,
            detailed=True,  # Set to False for less verbose output
            csv_path=memory_csv_path
        )
        callbacks.append(memory_callback)
        logger.info(f"✅ Memory monitoring enabled (logging every {args_update.memory_log_interval} steps)")
        logger.info(f"📊 Memory CSV will be saved to: {memory_csv_path}")
    # Initialize trainer with multi-GPU settings
    trainer_kwargs = {
        "max_epochs": args_update.max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": args_update.devices,
        "num_nodes": args_update.num_nodes,
        "strategy": strategy_config,
        "precision": args_update.precision,
        "callbacks": callbacks,
        "logger": csv_logger,
        "gradient_clip_val": 0.5,
        "sync_batchnorm": True if is_distributed else False,
        "num_sanity_val_steps": 0,
        "deterministic": True,
        "benchmark": False,
        "log_every_n_steps": 1,
        "profiler": profiler,
        "limit_val_batches": args_update.limit_val_batches,
    }
    
    # Add validation configuration (determined above based on dataset size)
    trainer_kwargs.update(validation_config)
    
    # Log the final trainer configuration for debugging
    logger.info(f"🔧 Trainer validation config: {validation_config}")
    logger.info(f"🔧 Limit val batches: {args_update.limit_val_batches}")
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Memory checkpoint 6: After trainer initialization
    print_memory_summary("6. After Trainer Init")
    
    return args_update, trainer, model, datamodule


def _move_tensors_to_cpu(obj):
    """Recursively move torch tensors to CPU for saving."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in obj)
    return obj


def _save_predict_outputs_to_disk(outputs, path: str, args) -> None:
    """Save ``trainer.predict`` return value to ``path`` (rank 0 only in DDP)."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    path = os.path.expanduser(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        'predictions': _move_tensors_to_cpu(outputs),
        'load_ckpt_path': getattr(args, 'load_ckpt_path', None),
        'exp_name': getattr(args, 'exp_name', None),
        'version': getattr(args, 'version', None),
    }
    torch.save(payload, path)
    logger.info(f"Saved predict outputs to {path}")


def _default_predict_output_path(args) -> str:
    exp = getattr(args, 'exp_name', 'default_experiment')
    ver = getattr(args, 'version', 'version_0')
    return os.path.join('predict_outputs', str(exp), str(ver), 'predictions.pt')


def _require_checkpoint_file(path: Optional[str], purpose: str) -> str:
    """Resolve and validate ``--load_ckpt_path`` for test/predict-only runs."""
    if not path or not str(path).strip():
        raise FileNotFoundError(
            f"--load_ckpt_path is required and must point to a checkpoint file for {purpose}."
        )
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found for {purpose}: {path}")
    return path


@timing_resource_decorator
def main(args_parsed):
    logger.info("--- Parsed Arguments ---")
    pprint(args_parsed)
    logger.info("------------------------")

    # Run with custom arguments
    args_parsed, trainer, model, datamodule = set_model(args_parsed)

    # 🔹 Run this check before training
    check_ddp_environment()

    action = args_parsed.action

    # --- Test only: no fit ---
    if action == 'test':
        print_memory_summary("7. Before Test")
        ckpt = _require_checkpoint_file(getattr(args_parsed, 'load_ckpt_path', None), "test")
        logger.info("--- Starting Test (no training) ---")
        logger.info(f"Loading weights from: {ckpt}")
        trainer.test(datamodule=datamodule, ckpt_path=ckpt)
        logger.info("--- Test Finished ---")
        print_memory_summary("8. After Test")
        return

    # --- Predict only: no fit ---
    if action == 'predict':
        print_memory_summary("7. Before Predict")
        ckpt = _require_checkpoint_file(getattr(args_parsed, 'load_ckpt_path', None), "predict")
        logger.info("--- Starting Predict (no training) ---")
        logger.info(f"Loading weights from: {ckpt}")
        predict_out = trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt,
            return_predictions=True,
        )
        out_path = getattr(args_parsed, 'predict_output', None) or _default_predict_output_path(
            args_parsed
        )
        _save_predict_outputs_to_disk(predict_out, out_path, args_parsed)
        logger.info("--- Predict Finished ---")
        print_memory_summary("8. After Predict")
        return

    # --- Training (and optional test after train) ---
    print_memory_summary("7. Before Training Start")
    logger.info("--- Starting Training ---")
    trainer.fit(
        model,
        datamodule,
        ckpt_path=args_parsed.load_ckpt_path if action == 'train' else None,
    )
    logger.info("--- Training Finished ---")
    print_memory_summary("8. After Training Complete")

    final_ckpt_path = os.path.join(
        getattr(args_parsed, "checkpoint_dir", "."), "train_model.ckpt"
    )
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Saved end-of-training checkpoint to {final_ckpt_path}")

    if action == 'train_and_test':
        logger.info("--- Starting Testing ---")
        ckpt_path_for_test = args_parsed.load_ckpt_path

        best_model_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                best_model_path = cb.best_model_path
                break
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Found best checkpoint from training run: {best_model_path}")
            ckpt_path_for_test = best_model_path
        else:
            logger.info(
                "Could not find the best checkpoint from the training run. "
                "Testing with the model's final weights."
            )

        if ckpt_path_for_test and os.path.exists(ckpt_path_for_test):
            logger.info(f"Loading model for testing from: {ckpt_path_for_test}")
            trainer.test(datamodule=datamodule, ckpt_path=ckpt_path_for_test)
        else:
            logger.info("No checkpoint specified or found, testing with model's current weights.")
            trainer.test(model=model, datamodule=datamodule)

        logger.info("--- Testing Finished ---")


if __name__ == "__main__":
    # Resolve exp_type and obs_config before full parse so the right config is loaded as default.
    _exp_type = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--exp_type"),
        "global",
    )
    _obs_config_name = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--obs_config"),
        "default",
    )
    observation_config, feature_stats, fill_values, instrument_weights, increment_stats = load_observation_config(
        exp_type=_exp_type, config_name=_obs_config_name
    )

    args_dict = {
        "model_path": "ocelot.hetero_observation_interaction.HeteroObservationGraphModel",
        #"data_path": "/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v4/global/",
        #"data_path": "/scratch3/NCEPDEV/da/Xin.C.Jin/my_data/ocelot/data_v5/global/dev/",
        #"data_path": "/scratch3/NCEPDEV/stmp/Xin.C.Jin/data/ocelot/data_v6/global/",
        "data_path": "/scratch3/NCEPDEV/stmp/Xin.C.Jin/data/ocelot/data_v6/diag_global/",
        "exp_type": _exp_type,
        'feature_stats': feature_stats,
        'fill_values': fill_values,
        "observation_config": observation_config,
        "instrument_weights": instrument_weights,
        "increment_stats": increment_stats,
        "start_date": "2024-04-01",
        "end_date": "2024-05-30",
        #"loss": "MSE",
        "loss": "huber",
        "lr": 0.0001,
        "hidden_dim": 128,
        "hidden_layers": 1,
        "mesh_resolution": 4.0,
        "mesh_feature_dim": 3,
        "cutoff_factor": 0.67,
        "num_neighbors": 2,
        "step_length": 6,
        "num_heads": 4,
        "accelerator": "gpu",
        "strategy": "auto",
        "devices": 1,
        "precision": "16-mixed",
        "num_workers": 2,
        "max_epochs": 50,
        "load_ckpt_path": 'checkpoint',
        "action": 'start_train',
        "batch_size": 2,
        "mesh_gnn_layers": 8,
        "weight_decay": 1e-5,
        "plot_on_test": False,
        "num_nodes": 1,
        "levels": 4,
        "mesh_splits": 6,
        "plot": False,
        "hierarchical": False,
        "hierarchical_processor_type": "interaction",
        "processor_window": 4,
        "processor_depth": 2,
        "processor_heads": 4,
        "processor_dropout": 0.0,
        "spatial_mixing_steps": 1,
        "processor_attn_chunk_size": 2048,
        "output_std": False,
        "restore_opt": False,
        "n_example_pred": 1,
        "graph": "hetero",
        "scan_angle_embed_dim": 4,
    }
        
    # Parse arguments, allowing command-line overrides
    args = parse_args(default_args=args_dict)
    main(args)
