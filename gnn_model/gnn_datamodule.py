import os
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
import importlib
from nnja_adapter import build_zlike_from_df
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from zarr.storage import LRUStoreCache

from process_timeseries import extract_features, organize_bins_times
from create_mesh_graph_global import obs_mesh_conn


def _t32(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


class BinDataset(Dataset):
    def __init__(
        self,
        bin_names,
        data_summary,
        zarr_store,
        create_graph_fn,
        observation_config,
        feature_stats=None,
    ):
        self.bin_names = bin_names
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[Rank {rank}] Fetching {bin_name}...")
        try:
            bin_data = extract_features(self.z, self.data_summary, bin_name, self.observation_config, feature_stats=self.feature_stats)[bin_name]
            graph_data = self.create_graph_fn(bin_data)
            graph_data.bin_name = bin_name
            return graph_data
        except Exception as e:
            print(f"[Rank {rank}] Error processing bin {bin_name}: {e}")
            raise


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        observation_config,
        mesh_structure,
        batch_size=1,
        num_neighbors=3,
        feature_stats=None,
        latent_step_hours=None,  # NEW: Added latent rollout parameter
        window_size="12h",       # NEW: Added window_size parameter
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mesh_structure = mesh_structure
        self.feature_stats = feature_stats
        self.z = None
        self.data_summary = None
        self.train_bin_names = None
        self.val_bin_names = None

    def setup(self, stage=None):
        if self.z is None:
            self.z = {}
            for obs_type, instruments in self.hparams.observation_config.items():
                self.z[obs_type] = {}
                for inst_name, inst_cfg in instruments.items():
                    src = inst_cfg.get("source", "zarr")

                    if src == "zarr":
                        zarr_dir = inst_cfg.get("zarr_dir")
                        if zarr_dir:
                            zarr_path = zarr_dir
                        else:
                            zname = inst_cfg.get("zarr_name", inst_name)
                            if not zname.endswith(".zarr"):
                                zname += ".zarr"
                            zarr_path = os.path.join(self.hparams.data_path, zname)

                        if not os.path.isdir(zarr_path):
                            raise FileNotFoundError(f"Conventional Zarr not found: {zarr_path}")
                        self.z[obs_type][inst_name] = zarr.open(LRUStoreCache(zarr.DirectoryStore(zarr_path), max_size=2e9), mode="r")
                        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                        if rank == 0:
                            print(f"[ZARR] {obs_type}/{inst_name} -> {zarr_path}")
                            print("  has keys:", list(self.z[obs_type][inst_name].keys())[:12])

                        # Optional guard to ensure the right file is used for surface_obs
                        if obs_type == "conventional" and inst_name == "surface_obs":
                            if not os.path.basename(zarr_path).startswith("raw_surface_obs"):
                                print(f"[WARN] surface_obs expected raw_surface_obs*.zarr but got: {zarr_path}")

                    elif src == "nnja":
                        # Load DataFrame via dotted loader path
                        loader_path = inst_cfg["dataframe_loader"]
                        mod_name, fn_name = loader_path.rsplit(".", 1)
                        load_fn = getattr(importlib.import_module(mod_name), fn_name)

                        # Columns to request = var_map values + coords/time
                        need = list(inst_cfg["var_map"].values())
                        need += [inst_cfg.get("lat_col", "LAT"), inst_cfg.get("lon_col", "LON"), inst_cfg.get("time_col", "OBS_TIMESTAMP")]

                        df = load_fn(start_date=self.hparams.start_date, end_date=self.hparams.end_date, columns=need)

                        self.z[obs_type][inst_name] = build_zlike_from_df(
                            df,
                            var_map=inst_cfg["var_map"],
                            lat_col=inst_cfg.get("lat_col", "LAT"),
                            lon_col=inst_cfg.get("lon_col", "LON"),
                            time_col=inst_cfg.get("time_col", "OBS_TIMESTAMP"),
                        )
                    else:
                        raise ValueError(f"Unknown source '{src}' for {inst_name}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # MODIFIED: Pass latent_step_hours and window_size parameters
        print(f"start: {self.hparams.start_date}; end: {self.hparams.end_date}")
        self.data_summary = organize_bins_times(
            self.z,
            self.hparams.start_date,
            self.hparams.end_date,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
            window_size=self.hparams.window_size,
            latent_step_hours=self.hparams.latent_step_hours,
        )
        self.all_bin_names = sorted(list(self.data_summary.keys()), key=lambda x: int(x.replace("bin", "")))
        print(f"all bin names: {self.all_bin_names}")

        if stage == "fit" or stage is None:
            self.train_bin_names = self.all_bin_names
            self.val_bin_names = self.all_bin_names
            pass

    def _create_graph_structure(self, bin_data):
        data = HeteroData()

        # 1. Mesh node features and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        # 2. Processor edges (mesh-to-mesh)
        m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0]

        reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
        data["mesh", "to", "mesh"].edge_index = torch.cat([m2m_edge_index, reverse_edges], dim=1)
        data["mesh", "to", "mesh"].edge_attr = torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0)

        # 3. Determine mode from datamodule configuration
        is_batch_latent_mode = self.hparams.latent_step_hours is not None
        if is_batch_latent_mode:
            window_hours = int(self.hparams.window_size.replace('h', ''))
            num_latent_steps = window_hours // self.hparams.latent_step_hours
        else:
            num_latent_steps = 1

        # 4. Observation data and mesh connections
        # ALL instruments get the same node structure based on detected batch mode
        for obs_type, instruments in self.hparams.observation_config.items():
            for inst_name, inst_cfg in instruments.items():

                # Check if this instrument has data for this time bin
                if obs_type in bin_data and inst_name in bin_data[obs_type]:
                    inst_dict = bin_data[obs_type][inst_name]

                    if is_batch_latent_mode:
                        # LATENT MODE: Create input + multiple target step nodes
                        self._create_latent_nodes(data, inst_name, inst_dict, num_latent_steps)
                    else:
                        # STANDARD MODE: Create input + single target node
                        self._create_standard_nodes(data, inst_name, inst_dict)

                else:
                    # MISSING INSTRUMENT: Create empty nodes with same structure as present instruments
                    if is_batch_latent_mode:
                        self._create_empty_latent_nodes(data, inst_name, inst_cfg, num_latent_steps)
                    else:
                        self._create_empty_standard_nodes(data, inst_name, inst_cfg)

        return data

    def _create_latent_nodes(self, data, inst_name, inst_dict, num_latent_steps):
        """Create nodes for instrument with data in latent mode."""
        # Input features (same for all steps)
        node_type_input = f"{inst_name}_input"
        data[node_type_input].x = _t32(inst_dict["input_features_final"])

        # Create encoder edges (observation to mesh)
        if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
            grid_lat_deg = inst_dict["input_lat_deg"]
            grid_lon_deg = inst_dict["input_lon_deg"]
            edge_index_encoder, edge_attr_encoder = obs_mesh_conn(
                grid_lat_deg,
                grid_lon_deg,
                self.mesh_structure["m2m_graphs"],
                self.mesh_structure["mesh_lat_lon_list"],
                self.mesh_structure["mesh_list"],
                o2m=True,
            )
            data[node_type_input, "to", "mesh"].edge_index = edge_index_encoder
            data[node_type_input, "to", "mesh"].edge_attr = edge_attr_encoder

        # Handle target features for each latent step
        if "target_features_final_list" in inst_dict:
            for step in range(num_latent_steps):
                node_type_target = f"{inst_name}_target_step{step}"

                if step < len(inst_dict["target_features_final_list"]):
                    target_features = inst_dict["target_features_final_list"][step]
                    data[node_type_target].y = _t32(target_features)

                    # Add target metadata
                    if "target_metadata_list" in inst_dict and step < len(inst_dict["target_metadata_list"]):
                        data[node_type_target].target_metadata = _t32(inst_dict["target_metadata_list"][step])

                    # Add scan angle if available
                    if "scan_angle_list" in inst_dict and step < len(inst_dict["scan_angle_list"]):
                        data[node_type_target].x = _t32(inst_dict["scan_angle_list"][step])

                    # Add channel mask
                    if "target_channel_mask_list" in inst_dict and step < len(inst_dict["target_channel_mask_list"]):
                        data[node_type_target].target_channel_mask = _t32(inst_dict["target_channel_mask_list"][step])

                    # Add instrument ID
                    if "instrument_id" in inst_dict:
                        data[node_type_target].instrument_ids = torch.full(
                            (target_features.shape[0],),
                            inst_dict["instrument_id"],
                            dtype=torch.long
                        )

                    # Create decoder edges (mesh to observation) for this step
                    if ("target_lat_deg_list" in inst_dict and "target_lon_deg_list" in inst_dict
                        and step < len(inst_dict["target_lat_deg_list"])
                            and step < len(inst_dict["target_lon_deg_list"])):

                        target_lat_deg = inst_dict["target_lat_deg_list"][step]
                        target_lon_deg = inst_dict["target_lon_deg_list"][step]

                        if len(target_lat_deg) > 0 and len(target_lon_deg) > 0:
                            edge_index_decoder, edge_attr_decoder = obs_mesh_conn(
                                target_lat_deg,
                                target_lon_deg,
                                self.mesh_structure["m2m_graphs"],
                                self.mesh_structure["mesh_lat_lon_list"],
                                self.mesh_structure["mesh_list"],
                                o2m=False,  # mesh to obs
                            )
                            data["mesh", "to", node_type_target].edge_index = edge_index_decoder
                            data["mesh", "to", node_type_target].edge_attr = edge_attr_decoder
                        else:
                            # Empty decoder edges for empty target step
                            data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
                            data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 3), dtype=torch.float32)

    def _create_standard_nodes(self, data, inst_name, inst_dict):
        """Create nodes for instrument with data in standard mode."""
        node_type_input = f"{inst_name}_input"
        node_type_target = f"{inst_name}_target"

        # Input features
        if "input_features_final" in inst_dict:
            data[node_type_input].x = _t32(inst_dict["input_features_final"])

            # Create encoder edges (observation to mesh)
            if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
                grid_lat_deg = inst_dict["input_lat_deg"]
                grid_lon_deg = inst_dict["input_lon_deg"]
                edge_index_encoder, edge_attr_encoder = obs_mesh_conn(
                    grid_lat_deg,
                    grid_lon_deg,
                    self.mesh_structure["m2m_graphs"],
                    self.mesh_structure["mesh_lat_lon_list"],
                    self.mesh_structure["mesh_list"],
                    o2m=True,
                )
                data[node_type_input, "to", "mesh"].edge_index = edge_index_encoder
                data[node_type_input, "to", "mesh"].edge_attr = edge_attr_encoder

        # Target features (single target window)
        if "target_features_final" in inst_dict:
            target_features = inst_dict["target_features_final"]
            data[node_type_target].y = _t32(target_features)

            # Add target metadata
            if "target_metadata" in inst_dict:
                data[node_type_target].target_metadata = _t32(inst_dict["target_metadata"])

            # Add scan angle if available
            if "scan_angle" in inst_dict:
                data[node_type_target].x = _t32(inst_dict["scan_angle"])

            # Add channel mask
            if "target_channel_mask" in inst_dict:
                data[node_type_target].target_channel_mask = _t32(inst_dict["target_channel_mask"])

            # Add instrument ID
            if "instrument_id" in inst_dict:
                data[node_type_target].instrument_ids = torch.full(
                    (target_features.shape[0],),
                    inst_dict["instrument_id"],
                    dtype=torch.long
                )

            # Create decoder edges (mesh to observation)
            if "target_lat_deg" in inst_dict and "target_lon_deg" in inst_dict:
                target_lat_deg = inst_dict["target_lat_deg"]
                target_lon_deg = inst_dict["target_lon_deg"]

                if len(target_lat_deg) > 0 and len(target_lon_deg) > 0:
                    edge_index_decoder, edge_attr_decoder = obs_mesh_conn(
                        target_lat_deg,
                        target_lon_deg,
                        self.mesh_structure["m2m_graphs"],
                        self.mesh_structure["mesh_lat_lon_list"],
                        self.mesh_structure["mesh_list"],
                        o2m=False,  # mesh to obs
                    )
                    data["mesh", "to", node_type_target].edge_index = edge_index_decoder
                    data["mesh", "to", node_type_target].edge_attr = edge_attr_decoder

    def _create_empty_latent_nodes(self, data, inst_name, inst_cfg, num_latent_steps):
        """Create empty nodes for missing instrument in latent mode."""
        # Create empty input node
        node_type_input = f"{inst_name}_input"
        data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 3), dtype=torch.float32)

        # Create empty target nodes for all latent steps
        for step in range(num_latent_steps):
            node_type_target = f"{inst_name}_target_step{step}"
            data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
            data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
            data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
            data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
            data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)
            data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 3), dtype=torch.float32)

    def _create_empty_standard_nodes(self, data, inst_name, inst_cfg):
        """Create empty nodes for missing instrument in standard mode."""
        # Create empty input node
        node_type_input = f"{inst_name}_input"
        data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 3), dtype=torch.float32)

        # Create empty target node
        node_type_target = f"{inst_name}_target"
        data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
        data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
        data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
        data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
        data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)
        data[node_type_target].pos = torch.empty((0, 2), dtype=torch.float32)
        data[node_type_target].num_nodes = 0
        data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 3), dtype=torch.float32)

    def train_dataloader(self):
        train_dataset = BinDataset(
            self.train_bin_names,
            self.data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
        )
        return PyGDataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        if not self.val_bin_names:
            return None
        val_dataset = BinDataset(
            self.val_bin_names,
            self.data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
        )
        return PyGDataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )
