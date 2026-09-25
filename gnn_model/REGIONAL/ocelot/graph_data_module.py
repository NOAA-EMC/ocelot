"""
PyTorch Lightning DataModule for weather data using GraphDataset.
"""
import pytorch_lightning as pl
import torch_geometric

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from typing import Dict, List, Optional
from torch.utils.data import Subset, DataLoader
from ocelot.graph_dataset import GraphDataset, make_graph_dataset


# NOTE: Avoid import-time side effects; log versions from the training entrypoint if needed.


def collate_skip_none(batch):
    batch = [d for d in batch if d is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)


class WeatherDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        observation_config: Dict[str, Dict],
        mesh_structure: Dict,
        train_val_test_split: Optional[List[float]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        args=None
    ):
        """
        DataModule for weather prediction using GraphDataset.

        Args:
            data_path: Path to data directory
            start_date: Start date for data range
            end_date: End date for data range
            observation_config: Configuration for observation types
            mesh_structure: Mesh graph structure
            train_val_test_split: Optional list of [train, val, test] fractions
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            args: Additional arguments passed to GraphDataset
        """
        super().__init__()
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.observation_config = observation_config
        self.mesh_structure = mesh_structure
        self.train_val_test_split = train_val_test_split or [0.8, 0.1, 0.1]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

        # Will be set up in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Load and split the dataset.
        """
        # Initialize the full dataset
        full_dataset = make_graph_dataset(
            data_path=self.data_path,
            start_date=self.start_date,
            end_date=self.end_date,
            observation_config=self.observation_config,
            mesh_structure=self.mesh_structure,
            args=self.args,
        )
        full_dataset.setup()

        # Split dataset into train, validation, and test sets
        n_samples = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * n_samples)
        val_size = int(self.train_val_test_split[1] * n_samples)
        test_size = n_samples - train_size - val_size

        indices = list(range(n_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        """Return the DataLoader for the training set."""
        # Lightning automatically wraps your dataloaders with a
        # DistributedSampler if shuffle=False.
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_skip_none
        )

    def val_dataloader(self):
        """Return the DataLoader for the validation set."""
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_skip_none
        )

    def test_dataloader(self):
        """Return the DataLoader for the test set."""
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_skip_none
        )

    def predict_dataloader(self):
        """Same split as test; used by ``Trainer.predict``."""
        return self.test_dataloader()
