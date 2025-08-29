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
from utils import random_keep_fraction_and_reindex


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
        apply_masking=True,
    ):
        self.bin_names = bin_names
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.apply_masking = apply_masking

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
                        zarr_dir = inst_cfg.get("zarr_dir")  # full path override (absolute or relative)
                        if zarr_dir:
                            zarr_path = zarr_dir
                        else:
                            zname = inst_cfg.get("zarr_name", inst_name)  # basename or basename.zarr
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

        self.data_summary = organize_bins_times(
            self.z,
            self.hparams.start_date,
            self.hparams.end_date,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
        )
        all_bin_names = sorted(list(self.data_summary.keys()), key=lambda x: int(x.replace("bin", "")))

        if stage == "fit" or stage is None:
            val_size = min(3, len(all_bin_names) - 1)
            self.train_bin_names = all_bin_names[:-val_size] if val_size > 0 else all_bin_names
            self.val_bin_names = all_bin_names[-val_size:] if val_size > 0 else []
            self.train_dataset = BinDataset(
                self.train_bin_names,
                self.data_summary,
                self.z,
                self._create_graph_structure,
                self.hparams.observation_config,
                feature_stats=self.feature_stats,
                apply_masking=True,
            )

    def _create_graph_structure(self, bin_data, apply_masking=True):
        data = HeteroData()

        # 1. Mesh node features and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        # 2. For each instrument, set up input and target nodes and edges
        for obs_type, instruments in self.hparams.observation_config.items():
            for inst_name, inst_cfg in instruments.items():
                node_type_input = f"{inst_name}_input"
                node_type_target = f"{inst_name}_target"

                if inst_name in bin_data.get(obs_type, {}):
                    inst_data = bin_data[obs_type][inst_name]

                    # --- INPUT NODES & EDGES ---
                    data[node_type_input].x = inst_data["input_features_final"]

                    grid_lat_deg = inst_data["input_lat_deg"]
                    grid_lon_deg = inst_data["input_lon_deg"]
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

                    # --- TARGET NODES & EDGES ---
                    target_features = inst_data["target_features_final"]
                    if inst_name in ("atms", "amsua"):
                        data[node_type_target].y = target_features  # [N, target_dim]
                        data[node_type_target].x = inst_data["scan_angle"].to(torch.float32)  # [N, 1]
                        # instrument IDs (long)
                        inst_id = int(inst_data["instrument_id"])
                        num_nodes = target_features.shape[0]
                        data[node_type_target].instrument_ids = torch.full((num_nodes,), inst_id, dtype=torch.long)
                        if "target_channel_mask" in inst_data:
                            data[node_type_target].target_channel_mask = inst_data["target_channel_mask"].to(torch.bool)
                        else:
                            data[node_type_target].target_channel_mask = torch.ones_like(target_features, dtype=torch.bool)
                    else:
                        # For other obs, all features are targets in '.y'
                        data[node_type_target].y = target_features  # [N, target_dim]
                        num_nodes = target_features.shape[0]
                        # instrument IDs (long)
                        inst_id = int(inst_data["instrument_id"])
                        data[node_type_target].instrument_ids = torch.full((num_nodes,), inst_id, dtype=torch.long)
                        # No aux decoder features → placeholder '.x'
                        data[node_type_target].x = torch.zeros((num_nodes, 1), dtype=torch.float32)

                        # per-channel mask for conventional targets
                        if "target_channel_mask" in inst_data:
                            data[node_type_target].target_channel_mask = inst_data["target_channel_mask"].to(torch.bool)
                        else:
                            # fall back to all-True if not provided
                            data[node_type_target].target_channel_mask = torch.ones_like(target_features, dtype=torch.bool)

                    data[node_type_target].num_nodes = data[node_type_target].y.shape[0]

                    target_lat_deg = inst_data["target_lat_deg"]
                    target_lon_deg = inst_data["target_lon_deg"]
                    lon_tensor = _t32(inst_data["target_lon_deg"])
                    lat_tensor = _t32(inst_data["target_lat_deg"])
                    data[node_type_target].pos = torch.stack([lon_tensor, lat_tensor], dim=1)
                    data[node_type_target].target_metadata = _t32(inst_data["target_metadata"])

                    edge_index_decoder, edge_attr_decoder = obs_mesh_conn(
                        target_lat_deg,
                        target_lon_deg,
                        self.mesh_structure["m2m_graphs"],
                        self.mesh_structure["mesh_lat_lon_list"],
                        self.mesh_structure["mesh_list"],
                        o2m=False,
                    )
                    data["mesh", "to", node_type_target].edge_index = edge_index_decoder
                    data["mesh", "to", node_type_target].edge_attr = edge_attr_decoder
                else:
                    # --- Handle missing instruments ---
                    data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
                    data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
                    data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
                    data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
                    data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
                    data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)

        # 3. Processor edges (mesh-to-mesh)
        m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0]

        reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
        data["mesh", "to", "mesh"].edge_index = torch.cat([m2m_edge_index, reverse_edges], dim=1)
        data["mesh", "to", "mesh"].edge_attr = torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0)

        return data

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
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
            apply_masking=False,
        )
        return PyGDataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )
