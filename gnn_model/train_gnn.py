import argparse
import faulthandler
import os
import socket
import sys
import time
import yaml

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

    # Load weights
    weights_config_path = "configs/weights_config.yaml"
    instrument_weights, channel_weights = load_weights_from_yaml(weights_config_path)

    # Data/region path
    region = "global"
    if region == "conus":
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        data_path = "/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v5/global"

    start_date = "2024-04-01"
    end_date = "2024-04-14"

    # --- Modular config, can add instruments here ---
    observation_config = {
        "satellite": {
            "atms": {
                "sat_ids": [224, 225],
                "features": [f"bt_channel_{i}" for i in range(1, 23)],
                "metadata": [
                    "sensorZenithAngle",
                    "solarZenithAngle",
                    "solarAzimuthAngle",
                ],
                "input_dim": 30,  # number of features incl. angle etc.
                "target_dim": 22,  # number of BT channels
                "encoder_hidden_layers": 2,
                "decoder_hidden_layers": 2,
            }
        },
        "conventional": {
            "surface_obs": {
                "features": ["virtualTemperature"],
                "metadata": ["height"],
                "input_dim": 7,
                "target_dim": 1,
                "encoder_hidden_layers": 2,
                "decoder_hidden_layers": 2,
            },
        }
    }

    feature_stats = {
        "stationPressure": (979.30, 52.78),
        "airTemperature": (24.16, 6.00),
        "dewPointTemperature": (11.99, 9.07),
        "latitude": (36.80, 6.44),
        "longitude": (-89.20, 9.75),
        "seaTemperature": (24.17, 5.62),
        "stationElevation": (381.52, 565.39),
        "virtualTemperature": (20.92, 8.51),
        "bt_channel_1": (259.20, 33.30),
        "bt_channel_10": (214.40, 4.22),
        "bt_channel_11": (217.72, 3.05),
        "bt_channel_12": (225.02, 2.63),
        "bt_channel_13": (234.56, 3.12),
        "bt_channel_14": (246.21, 3.41),
        "bt_channel_15": (255.89, 3.24),
        "bt_channel_16": (269.68, 20.06),
        "bt_channel_17": (276.82, 13.88),
        "bt_channel_18": (270.94, 11.07),
        "bt_channel_19": (265.76, 9.77),
        "bt_channel_2": (254.39, 39.29),
        "bt_channel_20": (260.43, 8.85),
        "bt_channel_21": (253.80, 8.09),
        "bt_channel_22": (247.68, 7.61),
        "bt_channel_3": (266.43, 18.62),
        "bt_channel_4": (267.84, 11.89),
        "bt_channel_5": (264.40, 7.48),
        "bt_channel_6": (252.46, 6.54),
        "bt_channel_7": (236.02, 5.08),
        "bt_channel_8": (225.50, 3.65),
        "bt_channel_9": (218.70, 3.33),
        "satelliteId": (225.02, 0.81),
        "sensorAzimuthAngle": (179.68, 94.52),
        "sensorZenithAngle": (31.15, 18.43),
        "solarAzimuthAngle": (125.22, 95.04),
        "solarZenithAngle": (71.94, 47.49),
    }

    mesh_resolution = 6
    hidden_dim = 64
    num_layers = 8
    lr = 0.001
    max_epochs = 80
    batch_size = 1
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
        mesh_resolution=mesh_resolution,  # Pass resolution to the model
        verbose=args.verbose,
        max_rollout_steps=max_rollout_steps,
        rollout_schedule=rollout_schedule,
        feature_stats=feature_stats,
    )

    # create the GNNDataModule and pass the model's mesh to it.
    data_module = GNNDataModule(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,  # This now works correctly
        batch_size=batch_size,
        num_neighbors=3,
    )

    # Call setup on the datamodule
    # data_module.setup("fit")
    # The 'LOCAL_RANK' environment variable is set by PyTorch Lightning's DDP strategy
    is_main_process = (
        int(os.environ.get("LOCAL_RANK", 0)) == 0
        and int(os.environ.get("NODE_RANK", 0)) == 0
    )

    if is_main_process:
        print("--- Main process is preparing data... ---")
        # Only the main process runs the expensive setup
        data_module.setup("fit")

    # All other processes will wait here until the main process is done
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if not is_main_process:
        print(
            f"--- Rank {int(os.environ.get('SLURM_PROCID'))} is loading data prepared by main process... ---"
        )
        # The other processes now run setup, but the data should be cached
        # or readily available, making this step very fast.
        data_module.setup("fit")

    val_loader = data_module.val_dataloader()
    has_val_data = val_loader is not None and len(val_loader.dataset) > 0
    print(f"Initial validation loader has {len(val_loader.dataset)} bins")

    setup_end_time = time.time()
    print(f"Initial setup time: {(setup_end_time - start_time) / 60:.2f} minutes")

    logger = CSVLogger(save_dir="logs", name=f"ocelot_gnn_{args.sampling_mode}")

    callbacks = []
    if has_val_data:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
        )
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
                full_start_date="2024-04-01", full_end_date="2024-06-01", window_days=14
            )
        )
        # trainer_kwargs['limit_train_batches'] = 500

    elif args.sampling_mode == "sequential":
        print("Using SEQUENTIAL sampling mode.")
        callbacks.append(
            SequentialDataCallback(
                full_start_date="2024-04-01", full_end_date="2024-06-01", window_days=10
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

    # trainer.fit(model, data_module)
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
