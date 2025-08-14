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

from callbacks import ResampleDataCallback, SequentialDataCallback
from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator
from weight_utils import load_weights_from_yaml

torch.set_float32_matmul_precision("medium")


@timing_resource_decorator
def main():
    # Corrected print statements for style
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()
    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")

    pl.seed_everything(42, workers=True)

    # === DATA & MODEL CONFIGURATION ===
    cfg_path = "configs/observation_config.yaml"
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(cfg_path)

    # Data/region path
    region = "global"
    if region == "conus":
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        data_path = "/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v5/global"

    # --- DEFINE THE FULL DATE RANGE FOR THE EXPERIMENT ---
    FULL_START_DATE = "2024-04-01"
    FULL_END_DATE = "2024-07-01"  # e.g., 3 months of data
    WINDOW_DAYS = 14  # The size of the window for each epoch

    # The initial start/end dates for the datamodule are the
    # first window of the full period. The callback will change this on subsequent epochs.
    initial_start_date = FULL_START_DATE
    initial_end_date = (pd.to_datetime(FULL_START_DATE) + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")

    # --- HYPERPARAMETERS ---
    mesh_resolution = 6
    hidden_dim = 64
    num_layers = 10
    lr = 0.001
    max_epochs = 100
    batch_size = 1
    # ----------------------------------------------------

    max_rollout_steps = 1
    rollout_schedule = "fixed"

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
    )

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=initial_start_date,
        end_date=initial_end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=batch_size,
        num_neighbors=3,
    )

    # The 'LOCAL_RANK' and 'NODE_RANK' env variables are set by PyTorch Lightning
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0 and int(os.environ.get("NODE_RANK", 0)) == 0

    if is_main_process:
        print("--- Main process is preparing data... ---")
        data_module.setup("fit")

    # All other processes will wait here until the main process is done
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if not is_main_process:
        print(f"--- Rank {int(os.environ.get('SLURM_PROCID'))} is loading data prepared by main process... ---")
        data_module.setup("fit")

    val_loader = data_module.val_dataloader()
    has_val_data = val_loader is not None and len(val_loader.dataset) > 0
    print(f"Initial validation loader has {len(val_loader.dataset)} bins")

    setup_end_time = time.time()
    print(f"Initial setup time: {(setup_end_time - start_time) / 60:.2f} minutes")

    logger = CSVLogger(save_dir="logs", name=f"ocelot_gnn_{args.sampling_mode}")

    callbacks = []
    if has_val_data:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True))
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                save_last=True,
            )
        )

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 2,
        "num_nodes": 4,
        "strategy": DDPStrategy(find_unused_parameters=True),
        "precision": "16-mixed",
        "log_every_n_steps": 1,
        "logger": logger,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 0.5,
        "enable_progress_bar": False,
    }

    if args.sampling_mode == "random":
        print("Using RANDOM sampling mode.")
        callbacks.append(
            ResampleDataCallback(
                full_start_date=FULL_START_DATE,
                full_end_date=FULL_END_DATE,
                window_days=WINDOW_DAYS,
            )
        )

    elif args.sampling_mode == "sequential":
        print("Using SEQUENTIAL sampling mode.")
        callbacks.append(
            SequentialDataCallback(
                full_start_date=FULL_START_DATE,
                full_end_date=FULL_END_DATE,
                window_days=WINDOW_DAYS,
            )
        )

    trainer_kwargs["callbacks"] = callbacks

    if has_val_data:
        trainer_kwargs["check_val_every_n_epoch"] = 1

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

    trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)

    end_time = time.time()
    print(f"Training time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time (setup + training): {(end_time - start_time) / 60:.2f} minutes")

    # === LOAD BEST MODEL AFTER TRAINING ===
    if has_val_data and trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"[INFO] Best model path: {best_path}")
        best_model = GNNLightning.load_from_checkpoint(best_path)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
