import os

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
from zarr.storage import LRUStoreCache
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mesh_creation import create_icosahedral_mesh
from mesh_to_mesh import MeshSelfConnectivity
from mesh_to_target import MeshTargetKNNConnector
from obs_to_mesh import ObsMeshCutoffConnector
from process_timeseries import extract_features, organize_bins_times
from torch.utils.data.distributed import DistributedSampler


def get_num_workers():
    return min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 4)


@rank_zero_only
def log_system_info():
    import multiprocessing
    print(f"[Rank 0] SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
    print(f"[Rank 0] SLURM_NTASKS: {os.environ.get('SLURM_NTASKS')}")
    print(f"[Rank 0] Detected CPU count: {multiprocessing.cpu_count()}")


@rank_zero_only
def print_dataset_lengths(train_data, val_data):
    print("Training dataset length:", len(train_data))
    print("Validation dataset length:", len(val_data))


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
        self._printed_mesh_stats = False
        self._printed_processor_edges = False

        # Graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors
        self.hetero_data, self.mesh_graph, self.mesh_latlon_rad, self.mesh_stats = create_icosahedral_mesh(resolution=self.mesh_resolution)

        self.data_summary = None
        self.data_dict = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        self.z = None

        # Flags for setup tracking
        self.data_processed = False

    def prepare_data(self):
        try:
            zarr.open(self.data_path, mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr dataset at {self.data_path}: {e}")

    def setup(self, stage=None):
        if self.z is None:
            self.z = zarr.open(LRUStoreCache(zarr.DirectoryStore(self.data_path), max_size=2_000_000_000), mode="r")
            rank_zero_info("Opened Zarr file.")

        if hasattr(self.trainer, "global_rank") and self.trainer.global_rank == 0:
            log_system_info()
            _ = list(self.z.array_keys())
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        if not self.data_processed:
            self.data_summary = organize_bins_times(self.z, self.start_date, self.end_date, self.satellite_id)
            for bin_name in self.data_summary:
                for key in ["input_features_final", "target_features_final"]:
                    if key in self.data_summary[bin_name]:
                        del self.data_summary[bin_name][key]

            self.data_summary = extract_features(self.z, self.data_summary)

            print(f"Found {len(self.data_summary)} time bins in the dataset")

            self.processed_data = []
            for bin_name in self.data_summary.keys():
                print(f"Processing {bin_name}...")
                data_dict = self._create_graph_structure(self.data_summary[bin_name])
                data_obj = self._create_data_object(data_dict)
                self.processed_data.extend(self._split_data_object(data_obj, chunks=16))

            self.data_processed = True

        num_bins = len(self.processed_data)
        train_size = 1 if num_bins == 1 else int(0.8 * num_bins)

        if stage == "fit" or stage is None:
            self.train_data = self.processed_data[:train_size]
            self.val_data = self.processed_data[train_size:]
            self._log_dataset_split()
            if len(self.val_data) > 0:
                assert isinstance(self.val_data[0], Data), "Validation data is not a torch_geometric Data object"
            else:
                rank_zero_info("Validation set is empty. Sanity check will be skipped.")

        elif stage == "validate":
            self.val_data = self.processed_data[train_size:]
        elif stage == "test":
            self.test_data = self.processed_data
        elif stage == "predict":
            self.predict_data = self.processed_data

    @rank_zero_only
    def _log_dataset_split(self):
        print(f"Split data into {len(self.train_data)} training bins and {len(self.val_data)} validation bins")
        print("Training dataset length:", len(self.train_data))
        print("Validation dataset length:", len(self.val_data))

    def _create_graph_structure(self, bin_data):
        input_features = bin_data["input_features_final"]
        target_features = bin_data["target_features_final"]
        obs_latlon_rad = input_features[:, -2:]
        target_latlon_rad = target_features[:, -2:]

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

        cutoff_encoder = ObsMeshCutoffConnector(cutoff_factor=self.cutoff_factor)
        edge_index_encoder, edge_attr_encoder = cutoff_encoder.add_edges(obs_latlon_rad, self.mesh_latlon_rad, return_edge_attr=True)

        if edge_index_encoder.numel() == 0:
            raise ValueError("No encoder edges were created. Try increasing cutoff_factor.")

        multi_scale_processor = MeshSelfConnectivity("hidden", "hidden")
        print_processor_edges = not self._printed_processor_edges
        hetero_data, mesh_graph = multi_scale_processor.update_graph(self.hetero_data, self.mesh_graph, print_once=print_processor_edges)
        self._printed_processor_edges = True

        mesh_graph = mesh_graph.to_undirected()
        edge_index_processor = hetero_data["hidden", "to", "hidden"].edge_index

        knn_decoder = MeshTargetKNNConnector(num_nearest_neighbours=self.num_neighbors)
        edge_index_knn, edge_attr_knn = knn_decoder.add_edges(mesh_graph, target_latlon_rad, self.mesh_latlon_rad)
        print(f"[{bin_data['input_time']}] Decoder edge count: {edge_index_knn.shape[1]}")

        num_obs_nodes = input_features.shape[0]
        num_mesh_nodes = hetero_data["hidden"].x.shape[0]
        mesh_offset = num_obs_nodes

        mesh_feats = hetero_data["hidden"].x
        input_dim = input_features.shape[1]
        pad_feats = torch.zeros((num_mesh_nodes, input_dim - mesh_feats.shape[1]))
        mesh_feats_padded = torch.cat([mesh_feats, pad_feats], dim=1)
        stacked_x = torch.cat([input_features, mesh_feats_padded], dim=0)

        edge_index_encoder_global = edge_index_encoder.clone()
        edge_index_encoder_global[1] += mesh_offset
        edge_index_processor_global = edge_index_processor + mesh_offset
        edge_index_decoder_global = edge_index_knn.clone()
        edge_index_decoder_global[0] += mesh_offset

        return {
            "x": stacked_x,
            "edge_index_encoder": edge_index_encoder_global.to(torch.long),
            "edge_attr_encoder": edge_attr_encoder,
            "edge_index_processor": edge_index_processor_global.to(torch.long),
            "edge_index_decoder": edge_index_decoder_global.to(torch.long),
            "edge_attr_decoder": edge_attr_knn,
            "y": target_features,
            "target_scaler_min": torch.tensor(bin_data["target_scaler_min"], dtype=torch.float32),
            "target_scaler_max": torch.tensor(bin_data["target_scaler_max"], dtype=torch.float32),
        }

    def _create_data_object(self, data_dict):
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
        )

    def _split_data_object(self, data_obj, chunks=8):
        from torch_geometric.data import Data
        mesh_node_ids = torch.arange(
            data_obj.x.shape[0] - data_obj.y.shape[0],
            data_obj.x.shape[0],
            dtype=torch.long
        )
        x = data_obj.x
        y = data_obj.y
        total_obs = y.shape[0]
        chunk_size = total_obs // chunks

        data_objs = []
        for i in range(chunks):
            start = i * chunk_size
            end = total_obs if i == chunks - 1 else (i + 1) * chunk_size

            sub_data = Data(
                x=x,
                edge_index_encoder=data_obj.edge_index_encoder,
                edge_attr_encoder=data_obj.edge_attr_encoder,
                edge_index_processor=data_obj.edge_index_processor,
                edge_index_decoder=data_obj.edge_index_decoder[:, start:end],
                edge_attr_decoder=data_obj.edge_attr_decoder[start:end],
                y=y[start:end],
                target_scaler_min=data_obj.target_scaler_min,
                target_scaler_max=data_obj.target_scaler_max,
            )
            sub_data.global_mesh_node_ids = mesh_node_ids
            data_objs.append(sub_data)

            # Log GPU memory usage after creating each chunk
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024 ** 3
                print(f"[Rank {dist.get_rank()}] Data chunk {i+1}/{chunks} | Memory allocated: {mem:.2f} GB")

        return data_objs

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_data, shuffle=True)
        return DataLoader(
            self.train_data,
            batch_size=1,
            sampler=sampler,
            num_workers=get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        from torch.utils.data import Dataset

        if self.val_data is None or len(self.val_data) == 0:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                class EmptyDataset(Dataset):
                    def __len__(self): return 0
                    def __getitem__(self, idx): return {}
                return DataLoader(EmptyDataset(), batch_size=4)

        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=1,
            num_workers=get_num_workers(),
            pin_memory=True,
            persistent_workers=True,
        )
