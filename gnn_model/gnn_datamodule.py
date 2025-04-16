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
        self.test_data = None
        self.predict_data = None
        self.z = None
        
        # New flags for setup tracking
        self.data_processed = False
        self.data_dict = None
        self.processed_data = None
        """
        [2025-04-11]
        MKo: added test_data, predict_data, and new flags
        """

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
        [2025-04-16]
        MKo: Add train/validate/test/predict states
             multi-bins: 80% bins for training, 20% bins for validation
             *remove single bin: 1 bin- use the entire bin for train, nothing for val
        """
        # Common operations for all stages - only execute once
        # data_processed flag prevents unnecessarily reading ZARR multiple times 
        if not self.data_processed:  
            # Open Zarr dataset
            self.z = zarr.open(self.data_path, mode="r")
            
            # Process time bins and features
            self.data_summary = organize_bins_times(
                self.z, self.start_date, self.end_date, self.satellite_id
            )
            self.data_summary = extract_features(self.z, self.data_summary)
            
            # Check bins number 
            num_bins = len(self.data_summary.keys())
            print(f"Found {num_bins} time bins in the dataset")
            
            # Process all available bins
            # Multi-bin case: Create a data object for each bin
            self.processed_data = []
            for bin_name in self.data_summary.keys():
                print(f"Processing {bin_name}...")
                data_dict = self._create_graph_structure(self.data_summary[bin_name])
                data_obj = self._create_data_object(data_dict)
                self.processed_data.append(data_obj)
            
            # Set flag to indicate data has been processed
            self.data_processed = True

        # Now prepare the appropriate splits based on what we have
        # Multiple bins case - split by bins
        num_bins = len(self.processed_data)
        train_size = 1 if num_bins==1 else int(0.8 * num_bins)
        
        if stage == 'fit' or stage is None:
            self.train_data = self.processed_data[:train_size]
            self.val_data = self.processed_data[train_size:]
            print(f"Split data into {len(self.train_data)} training bins and {len(self.val_data)} validation bins")
        elif stage == 'validate':
            self.val_data = self.processed_data[train_size:]
        elif stage == 'test':
            self.test_data = self.processed_data  # Use all bins for testing
        elif stage == 'predict':
            self.predict_data = self.processed_data  # Use all bins for prediction

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

    def _create_data_object(self, data_dict):
        """
        Create a PyG Data object from the combined data_dict.
        MK: remove <idx_slice>, unused param
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
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size)

