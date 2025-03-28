import lightning.pytorch as pl
import pandas as pd
import torch
import zarr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mesh_creation import create_icosahedral_mesh
from mesh_to_mesh import MeshSelfConnectivity
from mesh_to_target import MeshTargetKNNConnector
from obs_to_mesh import ObsMeshCutoffConnector
from process_timeseries import extract_features, organize_bins_times


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        satellite_id,
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
        self.satellite_id = satellite_id
        self.batch_size = batch_size

        # Graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors

        # Will be set in setup()
        self.data_summary = None
        self.train_data = None
        self.val_data = None
        self.z = None

    def prepare_data(self):
        """
        Check if Zarr dataset exists.
        """
        try:
            zarr.open(self.data_path, mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr dataset at {self.data_path}: {e}")

    def setup(self, stage=None):
        """
        Prepare data for training/validation.
        """
        # Open Zarr dataset
        self.z = zarr.open(self.data_path, mode="r")

        # Process time bins and features
        self.data_summary = organize_bins_times(
            self.z, self.start_date, self.end_date, self.satellite_id
        )
        self.data_summary = extract_features(self.z, self.data_summary)

        # Create graph structure for first bin (can be extended to handle multiple bins)
        data_dict = self._create_graph_structure(self.data_summary["bin1"])

        # Split data into train/val
        total_samples = len(data_dict["x"])
        int(0.8 * total_samples)

        # Create train/val splits
        self.train_data = self._create_data_object(data_dict, slice(0, len(data_dict["x"])))
        # self.val_data = self._create_data_object(hetero_data, slice(train_size, total_samples))
        # TODO make validation work

    def _create_graph_structure(self, bin_data):
        """
        Create the graph structure for the model with global indexing and padded features.
        """
        input_features = bin_data["input_features_final"]
        target_features = bin_data["target_features_final"]
        obs_latlon_rad = input_features[:, -2:]
        target_latlon_rad = target_features[:, -2:]

        # Create icosahedral mesh
        hetero_data, mesh_graph, mesh_latlon_rad, stats = create_icosahedral_mesh(
            resolution=self.mesh_resolution
        )

        # Print mesh statistics
        print("\n Mesh Statistics:")
        print(f"  - Mesh Resolution: {stats['resolution']}")
        print(f"  - Number of Nodes: {stats['finest_nodes']}")
        print(f"  - Number of Faces: {stats['finest_faces']}")
        print(f"  - Multilevel Edges: {stats['multilevel_edges']}")
        print("  - Edge Counts per Level:")
        for reso_level, count in stats["edge_counts_per_level"].items():
            print(f"    {reso_level}: {count}")
        print("")

        # === ENCODER EDGES ===
        cutoff_encoder = ObsMeshCutoffConnector(cutoff_factor=self.cutoff_factor)
        edge_index_encoder, edge_attr_encoder = cutoff_encoder.add_edges(
            obs_latlon_rad, mesh_latlon_rad, return_edge_attr=True
        )

        if edge_index_encoder.numel() == 0:
            raise ValueError("No encoder edges were created. Try increasing cutoff_factor.")

        # === PROCESSOR EDGES ===
        multi_scale_processor = MeshSelfConnectivity(
            source_name="hidden",
            target_name="hidden",
        )
        hetero_data, mesh_graph = multi_scale_processor.update_graph(hetero_data, mesh_graph)

        mesh_graph = mesh_graph.to_undirected()

        # Get edge index for processor edges
        edge_index_processor = hetero_data["hidden", "to", "hidden"].edge_index

        # === DECODER EDGES ===
        knn_decoder = MeshTargetKNNConnector(num_nearest_neighbours=self.num_neighbors)
        edge_index_knn, edge_attr_knn = knn_decoder.add_edges(
            mesh_graph, target_latlon_rad, mesh_latlon_rad
        )

        # === GLOBAL INDEXING ===
        num_obs_nodes = input_features.shape[0]
        num_mesh_nodes = hetero_data["hidden"].x.shape[0]
        mesh_offset = num_obs_nodes

        # Pad mesh features to match input_dim
        mesh_feats = hetero_data["hidden"].x  # e.g., [40962, 2]
        input_dim = input_features.shape[1]
        pad_feats = torch.zeros((num_mesh_nodes, input_dim - mesh_feats.shape[1]))
        mesh_feats_padded = torch.cat([mesh_feats, pad_feats], dim=1)

        # Stack input and mesh features
        stacked_x = torch.cat([input_features, mesh_feats_padded], dim=0)

        # Update edge indices with global offsets
        edge_index_encoder_global = edge_index_encoder.clone()
        edge_index_encoder_global[1] += mesh_offset

        edge_index_processor_global = edge_index_processor + mesh_offset

        edge_index_decoder_global = edge_index_knn.clone()
        edge_index_decoder_global[0] += mesh_offset

        # Store everything in a dictionary (to be converted to Data later)
        data_dict = {
            "x": stacked_x,
            "edge_index_encoder": edge_index_encoder_global.to(torch.long),
            "edge_attr_encoder": edge_attr_encoder,
            "edge_index_processor": edge_index_processor_global.to(torch.long),
            "edge_index_decoder": edge_index_decoder_global.to(torch.long),
            "edge_attr_decoder": edge_attr_knn,
            "y": target_features,
        }
        return data_dict

    def _create_data_object(self, data_dict, idx_slice):
        """
        Create a PyG Data object from the combined data_dict.
        """
        return Data(
            x=data_dict["x"],
            edge_index_encoder=data_dict["edge_index_encoder"],
            edge_attr_encoder=data_dict["edge_attr_encoder"],
            edge_index_processor=data_dict["edge_index_processor"],
            edge_index_decoder=data_dict["edge_index_decoder"],
            edge_attr_decoder=data_dict["edge_attr_decoder"],
            y=data_dict["y"],
        )

    def train_dataloader(self):
        return DataLoader([self.train_data], batch_size=self.batch_size, shuffle=True)
