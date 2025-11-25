"""
GNN training script for weather observation assimilation with DDP support.

Main training script that:
- Configures GNN model and PyTorch Lightning trainer
- Supports both random and sequential sampling modes with synchronized windows
- Handles multi-GPU/multi-node distributed training via DDP
- Implements time-windowed data loading with train/val splitting
- Provides checkpoint resumption and early stopping

Author: Azadeh Gholoubi (NOAA/EMC)
Date: April 2025
"""

import argparse
import faulthandler
import os
import socket
import sys
import time
import yaml
import pandas as pd

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from callbacks import ResampleDataCallback, SequentialDataCallback, AdvanceSamplerEpoch
from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator
from weight_utils import load_weights_from_yaml
from ckpt_utils import find_latest_checkpoint
from datetime import timedelta


torch.set_float32_matmul_precision("medium")


@timing_resource_decorator
def main():
    # Basic environment prints
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
        "--window_mode",
        type=str,
        default="sequential",
        choices=["random", "sequential"],
        help="For sequential sampling: window selection mode ('random' for random windows, 'sequential' for sliding windows).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--resume_from_latest",
        action="store_true",
        help="Resume from the most recent checkpoint found",
    )
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint to initialize model weights from (no Trainer resume).",
    )

    # Debug mode arguments
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode with minimal training")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="Limit validation batches per epoch")
    parser.add_argument("--devices", type=int, default=None, help="Override number of devices/GPUs")
    parser.add_argument("--num_nodes", type=int, default=None, help="Override number of nodes")
    parser.add_argument("--default_root_dir", type=str, default=None, help="Root directory for logs and checkpoints")
    args = parser.parse_args()
    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")

    # Use random seed for better diversity with year-long data, or fixed seed for debugging
    import random
    import numpy as np
    if args.debug_mode:
        print("Debug mode enabled: Using fixed seed 42 for reproducibility.")
        pl.seed_everything(42, workers=True)
    else:
        random_seed = random.randint(1, 1000000)
        print(f"Using random seed: {random_seed}")
        pl.seed_everything(random_seed, workers=True)
    # === DATA & MODEL CONFIGURATION ===
    cfg_path = "configs/observation_config.yaml"
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(cfg_path)
    with open(cfg_path, "r") as f:
        _raw_cfg = yaml.safe_load(f)
    pipeline_cfg = _raw_cfg.get("pipeline", {})

    # Data/region path
    region = "global"
    if region == "conus":
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        data_path = "/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v6/global"

    # --- DEFINE THE FULL DATE RANGE FOR THE EXPERIMENT ---
    FULL_START_DATE = "2023-01-01"
    FULL_END_DATE = "2023-12-31"
    TRAIN_WINDOW_DAYS = 12  # The size of the training window for each epoch
    VALID_WINDOW_DAYS = 8   # The size of the validation window for each epoch
    WINDOW_DAYS = TRAIN_WINDOW_DAYS

    # --- Compute train/val split BEFORE using VAL_START_DATE ---
    TRAIN_VAL_SPLIT_RATIO = 0.9  # 90% train, 10% val
    total_days = (pd.to_datetime(FULL_END_DATE) - pd.to_datetime(FULL_START_DATE)).days
    train_days = int(total_days * TRAIN_VAL_SPLIT_RATIO)

    TRAIN_START_DATE = FULL_START_DATE
    TRAIN_END_DATE = (pd.to_datetime(FULL_START_DATE) + pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")
    VAL_START_DATE = (pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    VAL_END_DATE = FULL_END_DATE

    print(f"Training period:  {TRAIN_START_DATE} -> {TRAIN_END_DATE}")
    print(f"Validation period:{VAL_START_DATE} -> {VAL_END_DATE}")

    # --- Initial windows for epoch 0 (DM uses these before callbacks resample) ---
    initial_start_date = FULL_START_DATE
    initial_end_date = (pd.to_datetime(FULL_START_DATE) + pd.Timedelta(days=TRAIN_WINDOW_DAYS)).strftime("%Y-%m-%d")
    initial_val_start_date = VAL_START_DATE
    initial_val_end_date = (pd.to_datetime(VAL_START_DATE) + pd.Timedelta(days=VALID_WINDOW_DAYS)).strftime("%Y-%m-%d")

    # --- Sanity checks ---
    ts = pd.to_datetime
    assert ts(FULL_START_DATE) < ts(FULL_END_DATE), "FULL date range must be positive"
    assert ts(TRAIN_START_DATE) <= ts(TRAIN_END_DATE), "Train range invalid"
    assert ts(VAL_START_DATE) <= ts(VAL_END_DATE), "Val range invalid"
    # Ensure no overlap between train and val pools
    assert ts(VAL_START_DATE) >= ts(TRAIN_END_DATE) + pd.Timedelta(days=1), "Train/Val pools should not overlap"
    # Ensure epoch-0 windows are within their pools
    assert ts(initial_start_date) >= ts(TRAIN_START_DATE) and ts(initial_end_date) <= ts(TRAIN_END_DATE), "Initial train window outside pool"
    assert ts(initial_val_start_date) >= ts(VAL_START_DATE) and ts(initial_val_end_date) <= ts(VAL_END_DATE), "Initial val window outside pool"

    # --- HYPERPARAMETERS ---
    mesh_resolution = 6
    hidden_dim = 96
    num_layers = 10
    lr = 5e-4
    max_epochs = 200
    batch_size = 1

    # Rollout settings
    max_rollout_steps = 1
    rollout_schedule = "fixed"

    # Latent rollout parameters (enable by setting integer hours)
    data_window_hours = 12  # Total window size in hours; default: 12h
    latent_step_hours = 3   # Size of each latent step (set to data_window_hours for a single latent step/no latent rollout)

    # Parameter checks
    latent_step_hours = data_window_hours if latent_step_hours is None else latent_step_hours
    if data_window_hours % latent_step_hours != 0:
        raise ValueError(f"target_hours ({data_window_hours}) must be divisible by latent_step_hours ({latent_step_hours})")

    start_time = time.time()

    # === INSTANTIATE MODEL & DATA MODULE ===
    model = GNNLightning(
        observation_config=observation_config,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        instrument_weights=instrument_weights,
        channel_weights=channel_weights,
        mesh_resolution=mesh_resolution,
        verbose=args.verbose,
        max_rollout_steps=max_rollout_steps,
        rollout_schedule=rollout_schedule,
        feature_stats=feature_stats,
        # Model options
        processor_type="sliding_transformer",   # sliding_transformer or "interaction"
        processor_window=4,     # default: 12h / 3h = 4
        processor_depth=4,
        processor_heads=4,
        processor_dropout=0.1,  # Add dropout for regularization
        # Dropout settings
        node_dropout=0.03,      # Slight node dropout for Phase 2 regularization
        # Encoder/decoder choices
        encoder_type="gat",    # gat or "interaction"
        decoder_type="gat",    # or "interaction"
        encoder_layers=2,
        decoder_layers=2,
        encoder_heads=4,
        decoder_heads=4,
        encoder_dropout=0.1,  # Add dropout for regularization
        decoder_dropout=0.1,  # Add dropout for regularization
    )

    # ---- OPTION A: Warm-start weights only, do NOT resume Trainer state ----
    if args.init_from_ckpt is not None:
        ckpt = torch.load(args.init_from_ckpt, map_location="cpu")
        # Lightning saves model weights under "state_dict"
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[INFO] Initialized model weights from {args.init_from_ckpt} (no Trainer resume).")

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=initial_start_date,
        end_date=initial_end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=batch_size,
        num_neighbors=3,
        feature_stats=feature_stats,
        pipeline=pipeline_cfg,
        window_size=f"{data_window_hours}h",
        latent_step_hours=latent_step_hours,
        train_val_split_ratio=TRAIN_VAL_SPLIT_RATIO,  # Pass the split ratio from training script
        sampling_mode=args.sampling_mode,  # Pass sampling mode to control bin distribution
        # ensure epoch 0 validation uses the val split, not the train slice
        train_start=initial_start_date,
        train_end=initial_end_date,
        val_start=initial_val_start_date,
        val_end=initial_val_end_date,
    )

    # Let Lightning handle setup() per rank at the correct time
    setup_end_time = time.time()
    print(f"Initial setup time (pre-trainer): {(setup_end_time - start_time) / 60:.2f} minutes")

    # Set up directories (use default_root_dir if provided)
    root_dir = args.default_root_dir if args.default_root_dir else "."
    log_dir = os.path.join(root_dir, "logs")
    checkpoint_dir = os.path.join(root_dir, "checkpoints")

    logger = CSVLogger(save_dir=log_dir, name=f"ocelot_gnn_{args.sampling_mode}")

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,        # Save less frequently to avoid timeout
            save_on_train_epoch_end=False,  # Only save after validation
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=25,                # Increase patience for full year training
            mode="min",
            min_delta=1e-5,             # Smaller threshold for year-long convergence
            verbose=True,
            check_finite=True,          # Check for finite values
            stopping_threshold=None,    # No absolute threshold
            divergence_threshold=None,  # No divergence threshold
            check_on_train_epoch_end=False,  # Only check after validation
            strict=False,
        ),
        AdvanceSamplerEpoch(),  # Ensure random samplers reshuffle each epoch
    ]

    strategy = DDPStrategy(
        process_group_backend="nccl",
        broadcast_buffers=False,
        find_unused_parameters=False,  # Changed from True - improves performance
        gradient_as_bucket_view=True,
        timeout=timedelta(hours=1),    # Increase timeout to 1 hour for checkpoints
    )

    # Use command-line args if provided, otherwise use defaults
    num_devices = args.devices if args.devices is not None else 2
    num_nodes = args.num_nodes if hasattr(args, 'num_nodes') and args.num_nodes is not None else 4

    # Override strategy for single device
    if num_devices == 1 and num_nodes == 1:
        strategy = "auto"  # No DDP for single GPU
        print("[INFO] Single device mode: Using strategy='auto' (no DDP)")

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": num_devices,
        "num_nodes": num_nodes,
        "strategy": strategy,
        "precision": "16-mixed",
        "log_every_n_steps": 1,
        "logger": logger,
        "num_sanity_val_steps": 2,  # Changed from 0 to run validation checks
        "gradient_clip_val": 0.5,
        "enable_progress_bar": False,
        "reload_dataloaders_every_n_epochs": 1,   # IMPORTANT for resampling
        "check_val_every_n_epoch": 1,
    }

    if args.sampling_mode == "random":
        print("Using RANDOM sampling mode (SYNCHRONIZED across ranks).")
        callbacks.append(
            ResampleDataCallback(
                train_start_date=TRAIN_START_DATE,
                train_end_date=TRAIN_END_DATE,
                val_start_date=VAL_START_DATE,
                val_end_date=VAL_END_DATE,
                train_window_days=TRAIN_WINDOW_DAYS,
                val_window_days=VALID_WINDOW_DAYS,
                mode="random",        # train windows chosen randomly (on rank-0), then broadcast
                resample_val=False,   # keep validation fixed for stable ES/CKPT
            )
        )
    else:
        print(f"Using SEQUENTIAL sampling mode with {args.window_mode} windows (synchronized across ranks).")
        callbacks.append(
            SequentialDataCallback(
                full_start_date=TRAIN_START_DATE,  # Use training pool, not full range
                full_end_date=TRAIN_END_DATE,
                window_days=TRAIN_WINDOW_DAYS,
                stride_days=TRAIN_WINDOW_DAYS,  # Non-overlapping windows
                mode=args.window_mode,  # "sequential" or "random"
                wrap_sequential=True,  # Wrap to start when reaching end
            )
        )

    trainer_kwargs["callbacks"] = callbacks
    trainer = pl.Trainer(**trainer_kwargs)

    # === TRAINING ===
    if torch.cuda.is_available():
        print(
            f"GPU {torch.cuda.current_device()} memory allocated:",
            torch.cuda.memory_allocated() / 1024**3,
            "GB",
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # === Checkpoint resume ===
    resume_path = None
    if args.init_from_ckpt is not None:
        # When warm-starting weights, do NOT resume Trainer state
        print("[INFO] init_from_ckpt is set -> ignoring any resume_from_* arguments.")
    else:
        if args.resume_from_latest:
            resume_path = find_latest_checkpoint(checkpoint_dir)
            if resume_path:
                print(f"[INFO] Auto-resuming from: {resume_path}")
            else:
                print("[INFO] No checkpoint found, starting fresh")
        elif args.resume_from_checkpoint:
            resume_path = args.resume_from_checkpoint
            print(f"[INFO] Resuming from: {resume_path}")
        else:
            print("[INFO] No checkpoint, starting fresh training")

    trainer.fit(model, data_module, ckpt_path=resume_path)

    end_time = time.time()
    print(f"Training time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time (setup + training): {(end_time - start_time) / 60:.2f} minutes")

    # === LOAD BEST MODEL AFTER TRAINING ===
    if trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"[INFO] Best model path: {best_path}")
        best_model = GNNLightning.load_from_checkpoint(best_path)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
