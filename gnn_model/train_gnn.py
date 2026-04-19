import argparse
import faulthandler
import os
import socket
import sys
import time
from datetime import timedelta

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from callbacks import ResampleDataCallback, SequentialDataCallback, ValWindowCallback
from ckpt_utils import find_latest_checkpoint
from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator
from weight_utils import load_weights_from_yaml


torch.set_float32_matmul_precision("medium")


def _load_mesh_config(mesh_cfg_path: str = "configs/mesh_config.yaml") -> dict:
    try:
        with open(mesh_cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@timing_resource_decorator
def main():
    print(f"Hostname: {socket.gethostname()}")
    print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"  SLURM_LOCALID: {os.environ.get('SLURM_LOCALID')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="random",
        choices=["random", "sequential"],
        help="The data sampling strategy ('random' or 'sequential').",
    )
    parser.add_argument(
        "--switch_to_sequential_after_epochs",
        type=int,
        default=None,
        help=(
            "If set (e.g., 50), start training with RANDOM windows then switch the TRAIN window sampler to "
            "SEQUENTIAL starting at that epoch. Only applies when --sampling_mode random."
        ),
    )
    parser.add_argument(
        "--auto_switch_to_sequential",
        action="store_true",
        help=(
            "If --sampling_mode random, automatically switch TRAIN window sampling to sequential when the "
            "monitored metric plateaus (patience/min_delta)."
        ),
    )
    parser.add_argument(
        "--auto_switch_metric",
        type=str,
        default="val_loss",
        help="Metric name to monitor for plateau (default: val_loss).",
    )
    parser.add_argument(
        "--auto_switch_patience_epochs",
        type=int,
        default=10,
        help="Number of validation epochs with insufficient improvement before switching.",
    )
    parser.add_argument(
        "--auto_switch_min_delta",
        type=float,
        default=0.0,
        help="Minimum decrease in monitored metric to count as improvement.",
    )

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_latest", action="store_true")
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help=(
            "Load model weights from a checkpoint but do NOT resume trainer/callback/optimizer state. "
            "Useful when changing validation windows across runs."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional run identifier to isolate logs/checkpoints. If set, logs go to logs/<run_name>/ "
            "and checkpoints to checkpoints/<run_name>/"
        ),
    )

    # Debug / resource overrides
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)

    # Model hyperparameters (overridable)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--huber_delta", type=float, default=0.1)

    parser.add_argument(
        "--scan_angle_conditioning",
        type=str,
        default="project",
        choices=["pad", "project"],
        help=(
            "How to inject scan-angle into decoder receiver init for satellite targets. "
            "'pad' keeps scan embedding in the last 8 dims (backward compatible); "
            "'project' spreads it across hidden_dim via a learned linear projection."
        ),
    )

    parser.add_argument(
        "--pressure_level_conditioning",
        type=str,
        default="project",
        choices=["pad", "project"],
        help=(
            "How to inject pressure-level embedding into decoder receiver init for aircraft/radiosonde targets. "
            "'pad' keeps pressure embedding in the last 8 dims; "
            "'project' spreads it across hidden_dim via a learned linear projection."
        ),
    )

    parser.add_argument(
        "--disable_bipartite_edge_attr",
        action="store_true",
        help=(
            "By default, computed obs↔mesh / mesh↔target spatial edge_attr features are fed into "
            "the GAT encoders/decoders. Set this flag to disable those features and force zero edge_attr instead."
        ),
    )
    parser.add_argument(
        "--bipartite_edge_attr_dim",
        type=int,
        default=4,
        help=(
            "Input dimension of bipartite spatial edge_attr features produced by obs_mesh_conn. "
            "With current GraphCast-style features this is typically 4 (distance + relative position xyz)."
        ),
    )
    parser.add_argument(
        "--encoder_dst_chunk_size",
        type=int,
        default=None,
        help="Optional destination-node chunk size for encoder bipartite GAT layers.",
    )
    parser.add_argument(
        "--encoder_dst_chunk_threshold",
        type=int,
        default=20000,
        help="Enable encoder bipartite chunking when destination node count reaches this threshold.",
    )
    parser.add_argument(
        "--decoder_dst_chunk_size",
        type=int,
        default=2048,
        help="Destination-node chunk size for decoder bipartite GAT layers.",
    )
    parser.add_argument(
        "--decoder_dst_chunk_threshold",
        type=int,
        default=2048,
        help="Enable decoder bipartite chunking when destination node count reaches this threshold.",
    )
    parser.add_argument(
        "--disable_bipartite_activation_checkpointing",
        action="store_true",
        help="Disable activation checkpointing inside bipartite GAT layers.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/observation_config.yaml",
        help="Path to observation_config YAML.",
    )

    parser.add_argument(
        "--conv_weight_mult",
        type=float,
        default=1.0,
        help="Multiply loss weights for conventional instruments after loading cfg.",
    )

    # Regularization knobs
    parser.add_argument("--processor_dropout", type=float, default=0.1)
    parser.add_argument(
        "--spatial_edge_chunk_size",
        type=int,
        default=16384,
        help="Edge chunk size used by the sliding-window transformer's spatial mixer.",
    )
    parser.add_argument("--node_dropout", type=float, default=0.03)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)

    # Windowing / latent rollout
    parser.add_argument("--data_window_hours", type=int, default=12)
    parser.add_argument("--latent_step_hours", type=int, default=3)

    # Mesh config
    parser.add_argument(
        "--mesh_type",
        type=str,
        default="fixed",
        choices=["fixed", "hierarchical"],
    )
    parser.add_argument("--mesh_levels", type=int, default=4)
    parser.add_argument(
        "--parallelization_strategy",
        type=str,
        default="replicated",
        choices=["replicated", "domain"],
        help=(
            "How to parallelize the graph state across ranks. 'replicated' keeps the current full-graph path; "
            "'domain' partitions the mesh per rank and exchanges halo states at the shard boundary."
        ),
    )
    parser.add_argument(
        "--domain_halo_hops",
        type=int,
        default=1,
        help="Number of mesh hops to include in each rank's halo when --parallelization_strategy domain.",
    )

    # Determinism and sequential stride
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stride_days", type=int, default=1)

    # Data pools
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--full_start_date", type=str, default=None)
    parser.add_argument("--full_end_date", type=str, default=None)
    parser.add_argument("--train_val_split_ratio", type=float, default=None)

    parser.add_argument("--train_start_date", type=str, default="2015-01-01")
    parser.add_argument(
        "--train_end_date",
        type=str,
        default="2024-01-01",
        help="Exclusive end date.",
    )
    parser.add_argument("--val_start_date", type=str, default="2024-01-01")
    parser.add_argument(
        "--val_end_date",
        type=str,
        default="2025-01-01",
        help="Exclusive end date.",
    )
    parser.add_argument(
        "--use_split_ratio",
        action="store_true",
        help="Use full_start/end + train_val_split_ratio instead of explicit train/val pool dates.",
    )

    parser.add_argument("--train_window_days", type=int, default=None)
    parser.add_argument("--val_window_days", type=int, default=None)

    parser.add_argument(
        "--val_mode",
        type=str,
        default="sequential",
        choices=["fixed", "random", "sequential"],
        help="How to pick validation windows from the validation pool.",
    )
    parser.add_argument("--val_stride_days", type=int, default=8)
    parser.add_argument("--val_update_every_n_epochs", type=int, default=5)

    parser.add_argument("--cache_val_windows", action="store_true")
    parser.add_argument("--val_cache_max_entries", type=int, default=16)
    parser.add_argument(
        "--zarr_cache_max_size_bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Per-process Zarr LRU cache size in bytes. Lower this if dataloader workers are OOM-killed.",
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=2,
        help="Number of PyG dataloader workers for training.",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=1,
        help="Number of PyG dataloader workers for validation.",
    )
    parser.add_argument(
        "--predict_num_workers",
        type=int,
        default=1,
        help="Number of PyG dataloader workers for prediction and FSOI.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=1,
        help="Prefetch factor for dataloaders with worker processes.",
    )
    parser.add_argument(
        "--disable_pin_memory",
        action="store_true",
        help="Disable DataLoader pin_memory if host-memory pressure is high.",
    )
    parser.add_argument(
        "--capture_cuda_memory_snapshot",
        action="store_true",
        help="Capture CUDA memory history and dump a snapshot after training for debugging.",
    )

    # Validation CSV artifacts
    parser.add_argument("--disable_val_csv", action="store_true")
    parser.add_argument("--val_csv_out_dir", type=str, default="val_csv")
    parser.add_argument("--val_csv_num_batches", type=int, default=1)
    parser.add_argument("--val_csv_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_csv_max_rows", type=int, default=None)
    parser.add_argument("--val_csv_sample_seed", type=int, default=0)

    # Early stopping
    parser.add_argument("--es_patience", type=int, default=None)
    parser.add_argument("--es_min_delta", type=float, default=None)
    parser.add_argument("--disable_early_stopping", action="store_true")

    args = parser.parse_args()
    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")

    # Rank-aware reproducibility prints
    try:
        procid = int(os.environ.get("SLURM_PROCID", "0"))
    except Exception:
        procid = 0
    if procid == 0:
        print("[ARGS]", vars(args))

    # DDP-safe seed selection
    if args.debug_mode:
        base_seed = 42
        print("Debug mode enabled: Using fixed seed 42 for reproducibility.")
    elif args.seed is not None:
        base_seed = int(args.seed)
        print(f"Using user-specified seed: {base_seed}")
    else:
        base_seed = 12345
        print(f"Using base seed: {base_seed} (pass --seed to override)")

    pl.seed_everything(base_seed, workers=True)

    # === DATA & MODEL CONFIGURATION ===
    cfg_path = str(args.cfg_path)
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(cfg_path)
    with open(cfg_path, "r") as f:
        _raw_cfg = yaml.safe_load(f)
    pipeline_cfg = (_raw_cfg or {}).get("pipeline", {})

    if procid == 0:
        print(f"[CFG] cfg_path={cfg_path}")

    # Optional: emphasize conventional instruments in the loss.
    conv_mult = float(args.conv_weight_mult)
    if conv_mult != 1.0:
        conv_names = ("surface_obs", "radiosonde", "aircraft")
        scaled = []
        for name in conv_names:
            inst_id = name_to_id.get(name)
            if inst_id is None:
                continue
            if inst_id in instrument_weights:
                instrument_weights[inst_id] = float(instrument_weights[inst_id]) * conv_mult
                scaled.append(name)
        if procid == 0:
            print(f"[CFG] conv_weight_mult={conv_mult} scaled={scaled}")

    # Mesh-grid target config (optional)
    mesh_config = _load_mesh_config("configs/mesh_config.yaml")

    # Data path
    region = "global"
    if args.data_path:
        data_path = args.data_path
    elif region == "conus":
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        # Default to multi-year merged Zarrs
        data_path = "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"

    # --- DEFINE TRAIN/VAL POOLS ---
    TRAIN_WINDOW_DAYS = int(args.train_window_days) if args.train_window_days is not None else 12
    VALID_WINDOW_DAYS = int(args.val_window_days) if args.val_window_days is not None else 8
    WINDOW_DAYS = TRAIN_WINDOW_DAYS

    if args.use_split_ratio:
        FULL_START_DATE = args.full_start_date or "2015-01-01"
        FULL_END_DATE = args.full_end_date or "2025-01-01"  # exclusive
        TRAIN_VAL_SPLIT_RATIO = float(args.train_val_split_ratio) if args.train_val_split_ratio is not None else 0.9
        total_days = (pd.to_datetime(FULL_END_DATE) - pd.to_datetime(FULL_START_DATE)).days
        train_days = int(total_days * TRAIN_VAL_SPLIT_RATIO)

        TRAIN_START_DATE = FULL_START_DATE
        TRAIN_END_DATE = (pd.to_datetime(FULL_START_DATE) + pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")
        VAL_START_DATE = TRAIN_END_DATE
        VAL_END_DATE = FULL_END_DATE
    else:
        TRAIN_START_DATE = args.train_start_date
        TRAIN_END_DATE = args.train_end_date
        VAL_START_DATE = args.val_start_date
        VAL_END_DATE = args.val_end_date

    print(f"Training period:  {TRAIN_START_DATE} -> {TRAIN_END_DATE} (exclusive end)")
    print(f"Validation period:{VAL_START_DATE} -> {VAL_END_DATE} (exclusive end)")

    # --- Initial windows for epoch 0 (DM uses these before callbacks resample) ---
    ts = pd.to_datetime
    initial_start_date = TRAIN_START_DATE
    initial_end_date = min(ts(TRAIN_START_DATE) + pd.Timedelta(days=TRAIN_WINDOW_DAYS), ts(TRAIN_END_DATE)).strftime("%Y-%m-%d")
    initial_val_start_date = VAL_START_DATE
    initial_val_end_date = min(ts(VAL_START_DATE) + pd.Timedelta(days=VALID_WINDOW_DAYS), ts(VAL_END_DATE)).strftime("%Y-%m-%d")

    # --- Sanity checks ---
    assert ts(TRAIN_START_DATE) < ts(TRAIN_END_DATE), "Train range invalid"
    assert ts(VAL_START_DATE) < ts(VAL_END_DATE), "Val range invalid"
    assert ts(VAL_START_DATE) >= ts(TRAIN_END_DATE), "Train/Val pools should not overlap (end is exclusive)"
    assert ts(initial_start_date) >= ts(TRAIN_START_DATE) and ts(initial_end_date) <= ts(TRAIN_END_DATE), "Initial train window outside pool"
    assert ts(initial_val_start_date) >= ts(VAL_START_DATE) and ts(initial_val_end_date) <= ts(VAL_END_DATE), "Initial val window outside pool"

    # --- HYPERPARAMETERS ---
    mesh_resolution = 6
    hidden_dim = int(args.hidden_dim)
    num_layers = 10
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    huber_delta = float(args.huber_delta)
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else 100
    batch_size = 1

    # Rollout settings
    max_rollout_steps = 1
    rollout_schedule = "fixed"

    data_window_hours = int(args.data_window_hours)
    latent_step_hours = int(args.latent_step_hours)

    if data_window_hours % latent_step_hours != 0:
        raise ValueError(
            f"data_window_hours ({data_window_hours}) must be divisible by latent_step_hours ({latent_step_hours})"
        )

    processor_window = int(data_window_hours // latent_step_hours)

    start_time = time.time()

    # Checkpoint/log isolation
    run_name = args.run_name
    ckpt_dir = os.path.join("checkpoints", run_name) if run_name else "checkpoints"

    # === Checkpoint resolve (may be used for weights-only load) ===
    resume_path = None
    if args.resume_from_latest:
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[INFO] Auto-resuming from: {resume_path}")
        else:
            print("[INFO] No checkpoint found, starting fresh")
    elif args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        print(f"[INFO] Resuming from: {resume_path}")
    else:
        print("[INFO] No checkpoint, starting fresh training")

    # === INSTANTIATE MODEL ===
    model_kwargs = dict(
        observation_config=observation_config,
        hidden_dim=hidden_dim,
        mesh_config=mesh_config,
        num_layers=num_layers,
        lr=lr,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        instrument_weights=instrument_weights,
        channel_weights=channel_weights,
        mesh_resolution=mesh_resolution,
        mesh_type=str(args.mesh_type),
        mesh_levels=int(args.mesh_levels),
        verbose=bool(args.verbose),
        max_rollout_steps=max_rollout_steps,
        rollout_schedule=rollout_schedule,
        latent_step_hours=int(latent_step_hours),
        feature_stats=feature_stats,
        processor_type="sliding_transformer",
        processor_window=processor_window,
        processor_depth=4,
        processor_heads=4,
        processor_dropout=float(args.processor_dropout),
        spatial_edge_chunk_size=int(args.spatial_edge_chunk_size),
        node_dropout=float(args.node_dropout),
        encoder_type="gat",
        decoder_type="gat",
        encoder_layers=2,
        decoder_layers=2,
        encoder_heads=4,
        decoder_heads=4,
        encoder_dropout=float(args.encoder_dropout),
        decoder_dropout=float(args.decoder_dropout),
        val_csv_enabled=(not args.disable_val_csv),
        val_csv_out_dir=str(args.val_csv_out_dir),
        val_csv_num_batches=int(args.val_csv_num_batches),
        val_csv_every_n_epochs=int(args.val_csv_every_n_epochs),
        val_csv_max_rows=(int(args.val_csv_max_rows) if args.val_csv_max_rows is not None else None),
        val_csv_sample_seed=int(args.val_csv_sample_seed),
        scan_angle_conditioning=str(args.scan_angle_conditioning),
        pressure_level_conditioning=str(args.pressure_level_conditioning),
        use_bipartite_edge_attr=(not args.disable_bipartite_edge_attr),
        bipartite_edge_attr_dim=int(args.bipartite_edge_attr_dim),
        encoder_dst_chunk_size=(int(args.encoder_dst_chunk_size) if args.encoder_dst_chunk_size is not None else None),
        encoder_dst_chunk_threshold=int(args.encoder_dst_chunk_threshold),
        decoder_dst_chunk_size=(int(args.decoder_dst_chunk_size) if args.decoder_dst_chunk_size is not None else None),
        decoder_dst_chunk_threshold=int(args.decoder_dst_chunk_threshold),
        bipartite_activation_checkpointing=(not args.disable_bipartite_activation_checkpointing),
        parallelization_strategy=str(args.parallelization_strategy),
        domain_halo_hops=int(args.domain_halo_hops),
    )

    if resume_path and args.load_weights_only:
        print(f"[INFO] Loading weights only (strict=False) from: {resume_path}")
        model = GNNLightning(**model_kwargs)
        ckpt = torch.load(resume_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[INFO] Weights-only load complete. missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
    else:
        model = GNNLightning(**model_kwargs)

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=initial_start_date,
        end_date=initial_end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=batch_size,
        num_neighbors=3,
        feature_stats=feature_stats,
        verbose=bool(args.verbose),
        pipeline=pipeline_cfg,
        window_size=f"{data_window_hours}h",
        latent_step_hours=latent_step_hours,
        train_val_split_ratio=float(args.train_val_split_ratio) if args.train_val_split_ratio is not None else 0.9,
        cache_val_windows=bool(args.cache_val_windows),
        val_cache_max_entries=int(args.val_cache_max_entries),
        prediction_mode=False,
        parallelization_strategy=str(args.parallelization_strategy),
        domain_halo_hops=int(args.domain_halo_hops),
        data_loader_seed=base_seed,
        zarr_cache_max_size_bytes=int(args.zarr_cache_max_size_bytes),
        train_num_workers=int(args.train_num_workers),
        val_num_workers=int(args.val_num_workers),
        predict_num_workers=int(args.predict_num_workers),
        dataloader_prefetch_factor=int(args.dataloader_prefetch_factor),
        pin_memory=(not args.disable_pin_memory),
        # epoch-0 windows
        train_start=initial_start_date,
        train_end=initial_end_date,
        val_start=initial_val_start_date,
        val_end=initial_val_end_date,
    )

    setup_end_time = time.time()
    print(f"Initial setup time (pre-trainer): {(setup_end_time - start_time) / 60:.2f} minutes")

    logger = (
        CSVLogger(save_dir="logs", name=run_name)
        if run_name
        else CSVLogger(save_dir="logs", name=f"ocelot_gnn_{args.sampling_mode}")
    )

    es_patience = int(args.es_patience) if args.es_patience is not None else 25
    es_min_delta = float(args.es_min_delta) if args.es_min_delta is not None else 1e-5

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
    ]

    if args.disable_early_stopping:
        print("[INFO] EarlyStopping disabled (--disable_early_stopping).")
    else:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=es_patience,
                mode="min",
                min_delta=es_min_delta,
                verbose=True,
                check_finite=True,
                check_on_train_epoch_end=False,
                strict=False,
            )
        )

    strategy = DDPStrategy(
        process_group_backend="nccl",
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        timeout=timedelta(hours=1),
    )

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": int(args.devices) if args.devices is not None else 2,
        "num_nodes": int(args.num_nodes) if args.num_nodes is not None else 4,
        "strategy": strategy,
        "precision": "16-mixed" if torch.cuda.is_available() else "32-true",
        "log_every_n_steps": 1,
        "logger": logger,
        "num_sanity_val_steps": 2,
        "gradient_clip_val": 0.5,
        "enable_progress_bar": False,
        "reload_dataloaders_every_n_epochs": 1,
        "check_val_every_n_epoch": 1,
        "use_distributed_sampler": False,
    }

    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = int(args.limit_train_batches)
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = int(args.limit_val_batches)

    rank_env = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    is_global_zero = str(rank_env) == "0"

    if args.sampling_mode == "random":
        if is_global_zero:
            msg = "Using RANDOM sampling mode"
            if args.switch_to_sequential_after_epochs is not None:
                msg += f" (will switch to SEQUENTIAL at epoch {int(args.switch_to_sequential_after_epochs)})"
            if args.auto_switch_to_sequential:
                msg += f" (auto-switch on plateau of {args.auto_switch_metric})"
            print(msg + ".")
        callbacks.append(
            ResampleDataCallback(
                train_start_date=TRAIN_START_DATE,
                train_end_date=TRAIN_END_DATE,
                val_start_date=VAL_START_DATE,
                val_end_date=VAL_END_DATE,
                train_window_days=TRAIN_WINDOW_DAYS,
                val_window_days=VALID_WINDOW_DAYS,
                mode="random",
                resample_val=False,
                seq_stride_days=int(args.stride_days),
                switch_to_sequential_after_epochs=args.switch_to_sequential_after_epochs,
                auto_switch_to_sequential=args.auto_switch_to_sequential,
                auto_switch_metric=args.auto_switch_metric,
                auto_switch_patience_epochs=args.auto_switch_patience_epochs,
                auto_switch_min_delta=args.auto_switch_min_delta,
            )
        )
    else:
        if is_global_zero:
            print("Using SEQUENTIAL sampling mode.")
            print(f"  Window size: {WINDOW_DAYS} days")
            print(f"  Stride: {args.stride_days} days per epoch")
        callbacks.append(
            SequentialDataCallback(
                full_start_date=TRAIN_START_DATE,
                full_end_date=TRAIN_END_DATE,
                window_days=WINDOW_DAYS,
                stride_days=int(args.stride_days),
            )
        )

    # Optional rotating validation windows
    if args.val_mode != "fixed":
        callbacks.append(
            ValWindowCallback(
                val_start_date=VAL_START_DATE,
                val_end_date=VAL_END_DATE,
                val_window_days=VALID_WINDOW_DAYS,
                mode=str(args.val_mode),
                stride_days=int(args.val_stride_days),
                update_every_n_epochs=int(args.val_update_every_n_epochs),
            )
        )

    trainer_kwargs["callbacks"] = callbacks
    trainer = pl.Trainer(**trainer_kwargs)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ckpt_path_for_fit = None if (resume_path and args.load_weights_only) else resume_path
    trainer.fit(model, data_module, ckpt_path=ckpt_path_for_fit)

    # if args.capture_cuda_memory_snapshot and torch.cuda.is_available():
    #     torch.cuda.memory._record_memory_history()
    #     try:
    #         trainer.fit(model, data_module, ckpt_path=ckpt_path_for_fit)
    #     finally:
    #         torch.cuda.memory._dump_snapshot(f"gnn_profile_rank_{rank_env}.pickle")
    # else:
    #     trainer.fit(model, data_module, ckpt_path=ckpt_path_for_fit)

    end_time = time.time()
    print(f"Training time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time (setup + training): {(end_time - start_time) / 60:.2f} minutes")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
