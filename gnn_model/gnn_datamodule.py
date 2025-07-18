import os

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from zarr.storage import LRUStoreCache

from mesh_creation import create_icosahedral_mesh
from mesh_to_mesh import MeshSelfConnectivity
from mesh_to_target import MeshTargetKNNConnector
from obs_to_mesh import ObsMeshCutoffConnector
from process_timeseries import extract_features, organize_bins_times, flatten_data


def tensor_conversion(data, dtype=torch.float32, device=None):
    """
    Convert data to tensor efficiently without unnecessary copies.

    Args:
        data: Input data (tensor, numpy array, list, etc.)
        dtype: Target data type
        device: Target device (optional)

    Returns:
        torch.Tensor: Efficiently converted tensor
    """
    if isinstance(data, torch.Tensor):
        # Already a tensor - minimize operations
        result = data

        # Change dtype if needed
        if result.dtype != dtype:
            result = result.to(dtype)

        # Change device if needed
        if device is not None and result.device != device:
            result = result.to(device)

        # Always detach to avoid gradient issues
        return result.detach()
    else:
        # Not a tensor - create new one efficiently
        if device is not None:
            return torch.tensor(data, dtype=dtype, device=device)
        else:
            return torch.tensor(data, dtype=dtype)


@rank_zero_only
def log_system_info():
    """
    Logs CPU and SLURM environment info on rank 0.

    Helps verify the computational environment across distributed jobs.
    """
    import multiprocessing

    print(f"[Rank 0] SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
    print(f"[Rank 0] SLURM_NTASKS: {os.environ.get('SLURM_NTASKS')}")
    print(f"[Rank 0] Detected CPU count: {multiprocessing.cpu_count()}")


class BinDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads and processes individual bins from a Zarr dataset.

    For each bin:
    - Extracts necessary features on demand.
    - Constructs graph structure using obs-mesh-target connectors.
    - Converts to PyG Data object for GNN processing.

    Args:
        bin_names (list): List of bin identifiers (e.g., ["bin1", "bin2"]).
        data_summary (dict): Metadata containing input/target time indices per bin.
        zarr_store (zarr.Group): The loaded Zarr dataset object.
        create_graph_fn (function): Function that creates the graph structure per bin.
    """

    def __init__(self, bin_names, data_summary, zarr_store, create_graph_fn, observation_config):
        self.bin_names = bin_names
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[Rank {rank}] Fetching {bin_name}...")

        try:
            # bin = self.data_summary[bin_name]
            # read from z file only for this bin...to follow previous, save in self.data_summary format, but only for required bin
            # Extract features for each bin and observation type
            bin = extract_features(
                self.z,
                self.data_summary,
                bin_name,
                self.observation_config,
            )[bin_name]

            bin, _ = flatten_data(bin)
            data_dict = self.create_graph_fn(bin)
            data_dict['bin_name'] = bin_name  # Add bin_name to data_dict
            print(
                f"[{bin_name}] Input features shape: {bin['input_features_final'].shape}, Target features shape: {bin['target_features_final'].shape}"
            )
        except Exception as e:
            print(f"[Rank {rank}] Error in bin {bin_name}: {e}")
            raise

        return self._to_data(data_dict)

    def _to_data(self, data_dict):
        return Data(
            x=data_dict["x"],
            edge_index_encoder=data_dict["edge_index_encoder"],
            edge_attr_encoder=data_dict["edge_attr_encoder"],
            edge_index_processor=data_dict["edge_index_processor"],
            edge_index_decoder=data_dict["edge_index_decoder"],
            edge_attr_decoder=data_dict["edge_attr_decoder"],
            y=data_dict["y"],
            target_scaler_min=data_dict["target_scaler_min"],
            target_scaler_max=data_dict["target_scaler_max"],
            instrument_ids=data_dict["target_instrument_ids"],
            bin_name=data_dict["bin_name"],
            target_lat_deg=tensor_conversion(data_dict["target_lat_deg"], dtype=torch.float32),
            target_lon_deg=tensor_conversion(data_dict["target_lon_deg"], dtype=torch.float32),
            target_metadata=data_dict["target_metadata"],
        )


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        observation_config,
        batch_size=1,
        mesh_resolution=2,
        cutoff_factor=0.6,
        num_neighbors=3,
    ):
        super().__init__()
        # Store parameters
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observation_config = observation_config
        self.obs_types = list(observation_config.keys())
        self.target_types = ['temperature', 'salinity']  # Fixed target types

        # Graph parameters
        self.batch_size = batch_size

        # Mesh and graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors

        # Create mesh structure (shared across all types)
        self.hetero_data, self.mesh_graph, self.mesh_latlon_rad, self.mesh_stats = create_icosahedral_mesh(
            resolution=self.mesh_resolution
        )

        # Placeholders for data
        self.z = None
        self.data_summary = None
        self.train_bin_names = None
        self.val_bin_names = None
        self.test_bin_names = None
        self.predict_bin_names = None
        self.instrument_mapping = None

        self._printed_mesh_stats = False
        self._printed_processor_edges = False
        self.data_processed = False

    @staticmethod
    def get_num_workers():
        return min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 4)

    def setup(self, stage=None):
        """
        The setup process:
        1. Opens Zarr stores for each observation type
        2. Organizes data into time bins
        3. Splits data into train/val/test sets
        """
        if self.z is None:
            # Open Zarr stores
            self.z = {}
            for obs_type in self.obs_types:
                self.z[obs_type] = {}
                for key in self.observation_config[obs_type].keys():
                    data_path = os.path.join(self.data_path, key) + ".zarr"
                    print(f'path: {data_path}')
                    self.z[obs_type][key] = zarr.open(LRUStoreCache(zarr.DirectoryStore(data_path), max_size=2_000_000_000), mode="r")
                    rank_zero_info(f"Opened Zarr files for {data_path}.")

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        if not self.data_processed:
            # Organize time bins with type-specific data
            self.data_summary = organize_bins_times(
                self.z,
                self.start_date,
                self.end_date,
                self.observation_config,
            )

            # Split bins into train/val/test sets
            all_bin_names = sorted(list(self.data_summary.keys()))

            if stage == "fit" or stage is None:
                train_size = 1 if len(all_bin_names) == 1 else int(0.8 * len(all_bin_names))
                self.train_bin_names = all_bin_names[:train_size]
                self.val_bin_names = all_bin_names[train_size:]
                print(f"Train bins: {len(self.train_bin_names)}, Val bins: {len(self.val_bin_names)}")
                self._log_dataset_split()

            if stage == "test":
                self.test_bin_names = all_bin_names

            if stage == "predict":
                self.predict_bin_names = all_bin_names

            self.data_processed = True

    @rank_zero_only
    def _log_dataset_split(self):
        print(f"Split data into {len(self.train_bin_names)} training bins and {len(self.val_bin_names)} validation bins")

    def _create_graph_structure(self, bin_data):
        """
        Builds the PyG-compatible graph structure for a single time bin.

        - Encoder edges connect observation points to nearby mesh nodes.
        - Processor edges connect mesh nodes using pre-built connectivity.
        - Decoder edges connect mesh nodes to target prediction locations.
        - Global indexing is applied to combine observation and mesh nodes.
        - Node features are padded to maintain consistent dimensionality.

        Returns:
            dict: A dictionary of all graph components including edges, node features,
                targets, and min/max scalers for unnormalization.
        """
        # Get flattened features and metadata
        input_features = bin_data["input_features_final"]  # Already includes instrument IDs
        target_features = bin_data["target_features_final"]
        input_metadata = bin_data["input_metadata"]
        target_metadata = bin_data["target_metadata"]

        # Extract lat/lon from metadata (first two columns)
        obs_latlon_rad = input_metadata[:, :2]  # latitude (rad), longitude (rad)
        target_latlon_rad = target_metadata[:, :2]  # latitude (rad), longitude (rad)

        # Print mesh statistics once
        if not self._printed_mesh_stats:
            print("\n Mesh Statistics:")
            print(f"  - Mesh Resolution: {self.mesh_stats['resolution']}")
            print(f"  - Number of Nodes: {self.mesh_stats['finest_nodes']}")
            print(f"  - Number of Faces: {self.mesh_stats['finest_faces']}")
            print(f"  - Multilevel Edges: {self.mesh_stats['multilevel_edges']}")
            print("  - Edge Counts per Level:")
            for reso_level, count in self.mesh_stats["edge_counts_per_level"].items():
                print(f"    {reso_level}: {count}")
            print("")
            self._printed_mesh_stats = True

        # === ENCODER EDGES ===
        # Create edges from flattened observation points to mesh nodes
        cutoff_encoder = ObsMeshCutoffConnector(cutoff_factor=self.cutoff_factor)
        edge_index_encoder, edge_attr_encoder = cutoff_encoder.add_edges(
            obs_latlon_rad,
            self.mesh_latlon_rad,
            return_edge_attr=True,
            max_neighbors=1
        )

        if edge_index_encoder.numel() == 0:
            raise ValueError("No encoder edges were created. Try increasing cutoff_factor.")

        # === PROCESSOR EDGES ===
        # Create mesh-to-mesh connectivity (shared across all types)
        multi_scale_processor = MeshSelfConnectivity("hidden", "hidden")
        print_processor_edges = not self._printed_processor_edges
        hetero_data, mesh_graph = multi_scale_processor.update_graph(
            self.hetero_data,
            self.mesh_graph,
            print_once=print_processor_edges
        )
        self._printed_processor_edges = True

        mesh_graph = mesh_graph.to_undirected()
        # Get edge index for processor edges
        edge_index_processor = hetero_data["hidden", "to", "hidden"].edge_index

        # === DECODER EDGES ===
        knn_decoder = MeshTargetKNNConnector(num_nearest_neighbours=self.num_neighbors)
        edge_index_knn, edge_attr_knn = knn_decoder.add_edges(mesh_graph, target_latlon_rad, self.mesh_latlon_rad)

        # === GLOBAL INDEXING ===
        # Calculate dimensions for combined features
        num_obs_nodes = input_features.shape[0]
        num_mesh_nodes = hetero_data["hidden"].x.shape[0]
        mesh_offset = num_obs_nodes

        # Move mesh features to CPU and pad to match input dimension
        mesh_feats = hetero_data["hidden"].x.cpu()
        input_dim = input_features.shape[1]
        pad_feats = torch.zeros((num_mesh_nodes, input_dim - mesh_feats.shape[1]))
        mesh_feats_padded = torch.cat([mesh_feats, pad_feats], dim=1)

        # Stack flattened input features with padded mesh features
        stacked_x = torch.cat([input_features, mesh_feats_padded], dim=0)

        # Update edge indices with global offsets
        edge_index_encoder_global = edge_index_encoder.clone()
        edge_index_encoder_global[1] += mesh_offset
        edge_index_processor_global = edge_index_processor + mesh_offset
        edge_index_decoder_global = edge_index_knn.clone()
        edge_index_decoder_global[0] += mesh_offset

        # Return flattened graph structure
        return {
            "x": stacked_x,  # Combined features including instrument IDs
            "edge_index_encoder": edge_index_encoder_global.to(torch.long),
            "edge_attr_encoder": edge_attr_encoder,
            "edge_index_processor": edge_index_processor_global.to(torch.long),
            "edge_index_decoder": edge_index_decoder_global.to(torch.long),
            "edge_attr_decoder": edge_attr_knn,
            "y": target_features,  # Flattened target features
            "target_lat_deg": bin_data["target_lat_deg"],
            "target_lon_deg": bin_data["target_lon_deg"],
            "target_scaler_min": tensor_conversion(bin_data["target_scaler_min"], dtype=torch.float32),
            "target_scaler_max": tensor_conversion(bin_data["target_scaler_max"], dtype=torch.float32),
            "input_instrument_ids": tensor_conversion(bin_data["input_instrument_ids"], dtype=torch.long),
            "target_instrument_ids": tensor_conversion(bin_data["target_instrument_ids"], dtype=torch.long),
            "target_metadata": tensor_conversion(bin_data["target_metadata"], dtype=torch.float32),
        }

    def _create_data_object(self, data_dict):
        """
        Converts a graph dictionary into a PyG `Data` object with attached scalers.

        Adds additional fields needed for later unnormalization (e.g., for evaluation).
        """
        data_args = dict(
            x=data_dict["x"],
            edge_index_encoder=data_dict["edge_index_encoder"],
            edge_attr_encoder=data_dict["edge_attr_encoder"],
            edge_index_processor=data_dict["edge_index_processor"],
            edge_index_decoder=data_dict["edge_index_decoder"],
            edge_attr_decoder=data_dict["edge_attr_decoder"],
            y=data_dict["y"],
            target_scaler_min=data_dict["target_scaler_min"],
            target_scaler_max=data_dict["target_scaler_max"],
            target_lat_deg=tensor_conversion(data_dict["target_lat_deg"], dtype=torch.float32),
            target_lon_deg=tensor_conversion(data_dict["target_lon_deg"], dtype=torch.float32),
        )
        # Optional: add instrument_ids only if present
        if "instrument_ids" in data_dict:
            data_args["instrument_ids"] = tensor_conversion(data_dict["instrument_ids"], dtype=torch.long)

        return Data(**data_args)

    def train_dataloader(self):
        """
        Returns the training DataLoader with a DistributedSampler.

        - One bin per batch (batch_size=1).
        - Enables parallel processing across GPUs using DDP.
        - Uses persistent workers for better performance.
        """
        dataset = BinDataset(self.train_bin_names, self.data_summary, self.z, self._create_graph_structure, self.observation_config)
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        - If validation set is empty on a given rank (e.g., non-zero rank under DDP),
        an empty dataset is returned to prevent sync errors.
        """
        if self.val_bin_names is None or len(self.val_bin_names) == 0:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:

                class EmptyDataset(Dataset):
                    def __len__(self):
                        return 0

                    def __getitem__(self, idx):
                        return {}

                return DataLoader(EmptyDataset(), batch_size=4)
        dataset = BinDataset(self.val_bin_names, self.data_summary, self.z, self._create_graph_structure, self.observation_config)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Loads and evaluates all test bins, one per batch.
        """
        dataset = BinDataset(self.test_bin_names, self.data_summary, self.z, self._create_graph_structure, self.observation_config)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """
        Returns the prediction DataLoader.

        Reuses the train+val bins for inference unless otherwise specified.
        """
        dataset = BinDataset(self.predict_bin_names, self.data_summary, self.z, self._create_graph_structure, self.observation_config)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )
