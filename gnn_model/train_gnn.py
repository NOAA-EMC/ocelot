import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def main():
    # Data parameters
    data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/atms.zarr"
    start_date = "2024-04-01"
    end_date = "2024-04-07"
    satellite_id = 224

    mesh_resolution = 6

    # Define model parameters
    input_dim = 25
    hidden_dim = 256
    output_dim = 20
    num_layers = 16
    lr = 1e-4

    # Training parameters
    max_epochs = 50
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
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gnn-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Train with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        # callbacks=[checkpoint_callback, early_stopping],  #TODO make this work
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
