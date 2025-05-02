import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def main():
    import faulthandler
    import sys
    import time

    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")
    # Data parameters
    # CONUS data path:
    #data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/atms.zarr"
    # One week Global data path:
    data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v3/atms_small.zarr"

    start_date = "2024-04-01"
    end_date = "2024-04-03"
    satellite_id = None

    mesh_resolution = 6

    # Define model parameters
    input_dim = 30
    hidden_dim = 64
    output_dim = 22
    num_layers = 16
    lr = 1e-4

    # Training parameters
    max_epochs = 10
    batch_size = 1

    # Instantiate model & data module
    model = GNNLightning(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        lr=lr,
    )

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        satellite_id=satellite_id,
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
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
        ]

    # Build trainer arguments dynamically
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 2,
        "num_nodes": 4,
        "strategy": "ddp_find_unused_parameters_true",  # Use DistributedDataParallel
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

    # GPU memory debug
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.current_device()} memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")

    trainer.fit(model, data_module)
    end_time = time.time()
    print(f"Training time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time (setup + training): {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
