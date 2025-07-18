import yaml
import faulthandler
import sys
import time
import lightning.pytorch as pl
import argparse
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator
from weight_utils import load_weights_from_yaml
import os


@timing_resource_decorator
def main():
    # Enable fault handler for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")

    # Set global seed for reproducibility
    pl.seed_everything(42, workers=True)

    # === DATA CONFIGURATION ===
    weights_config_path = "configs/weights_config.yaml"
    instrument_weights, channel_weights = load_weights_from_yaml(weights_config_path)

    # Data parameters
    region = "global"
    if region == "conus":
        # CONUS data path:
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        # One week Global data path:
        data_path = "/scratch3/NCEPDEV/da/Azadeh.Gholoubi/data_v3/bigzarr/"
        # data_path = "/scratch3/NCEPDEV/da/Azadeh.Gholoubi/data_v3"

    start_date = "2024-04-01"
    end_date = "2024-05-15"

    # Observation configuration, will move to a config file later.
    observation_config = {
        "satellite": {
            "atms": {
                "sat_ids": [224],
                "features": [f"bt_channel_{i}" for i in range(1, 23)],
                "metadata": ["sensorZenithAngle", "solarZenithAngle", "solarAzimuthAngle"],
            },
            # "iasi": ,
            # "goes":,
            # "ascat":
        },
        # "conventional": {
        #     # "radiosonde": ,
        #     "surface_pressure": {
        #         "features": ["stationPressure", ],
        #         "metadata": ["height", ]
        #     },
        #     # "surface_marine": ,
        #     # "surface_land":
        # }
    }
    mesh_resolution = 6

    # === MODEL CONFIGURATION ===
    input_dim = 32
    hidden_dim = 128
    output_dim = 22
    num_layers = 6
    lr = 1e-3

    # === TRAINING CONFIGURATION ===
    max_epochs = 10
    batch_size = 1
    max_rollout_steps = 1  # Maximum rollout length; set 1 to have no rollout
    rollout_schedule = "fixed"  # 'graphcast', 'step', 'linear', or 'fixed'

    # # === INSTANTIATE MODEL & DATA MODULE ===
    model = GNNLightning(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        lr=lr,
        instrument_weights=instrument_weights,
        channel_weights=channel_weights,
        verbose=args.verbose,
        max_rollout_steps=max_rollout_steps,
        rollout_schedule=rollout_schedule,
    )
    # model = GNNLightning.load_from_checkpoint(
    #     checkpoint_path,
    #     input_dim=input_dim,
    #     hidden_dim=hidden_dim,
    #     output_dim=output_dim,
    #     num_layers=num_layers,
    #     lr=lr,
    #     instrument_weights=instrument_weights,
    #     channel_weights=channel_weights,
    #     verbose=args.verbose,
    # )

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        observation_config=observation_config,
        batch_size=batch_size,
        mesh_resolution=mesh_resolution,
        num_neighbors=3,
    )

    start_time = time.time()

    # Safety check: skip validation-related callbacks if no val_data exists
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()
    has_val_data = val_loader is not None and len(val_loader.dataset) > 0
    print(f"Validation loader has {len(val_loader.dataset)} bins")

    setup_end_time = time.time()
    print(f"Setup time: {(setup_end_time - start_time) / 60:.2f} minutes")

    logger = CSVLogger(save_dir="logs", name="ocelot_gnn")
    # Setup callbacks
    callbacks = []
    if has_val_data:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                save_last=True,
            ),
        ]

    # === TRAINER CONFIGURATION ===
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 2,
        "num_nodes": 4,
        "strategy": "ddp",
        "precision": "16-mixed",  # Mixed precision for memory efficiency
        "log_every_n_steps": 1,
        "callbacks": callbacks,
        "logger": logger,
        "num_sanity_val_steps": 0,  # Skip sanity check
        "gradient_clip_val": 0.5,
        "enable_progress_bar": False,
    }

    if has_val_data:
        trainer_kwargs["check_val_every_n_epoch"] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    # === TRAINING ===
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.current_device()} memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    trainer.fit(model, data_module)
    # trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    end_time = time.time()
    print(f"Training time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time (setup + training): {(end_time - start_time) / 60:.2f} minutes")

    # === LOAD BEST MODEL AFTER TRAINING ===
    if has_val_data:
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"[INFO] Best model path: {best_path}")
        best_model = GNNLightning.load_from_checkpoint(best_path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
