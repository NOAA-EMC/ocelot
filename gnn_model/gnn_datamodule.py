import os
import importlib
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
from zarr.storage import LRUStoreCache
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from nnja_adapter import build_zlike_from_df
from process_timeseries import extract_features, organize_bins_times
from create_mesh_graph_global import obs_mesh_conn

# Number of columns for latitude and longitude in metadata
LAT_LON_COLUMNS = 2


def _t32(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


# -------------------------
# Dataset per bin
# -------------------------
class BinDataset(Dataset):
    def __init__(
        self,
        bin_names,
        data_summary,
        zarr_store,
        create_graph_fn,
        observation_config,
        feature_stats=None,
        require_targets=True,
        tag="TRAIN",
    ):
        self.bin_names = list(bin_names) if bin_names is not None else []
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.require_targets = require_targets
        self.tag = tag

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[Rank {rank}] [{self.tag}] fetching {bin_name} ... ds_id={id(self)} sum_id={id(self.data_summary)}")
        try:
            # Extract input_time before extract_features
            input_time = None
            for obs_type in self.data_summary[bin_name].keys():
                for inst_name in self.data_summary[bin_name][obs_type].keys():
                    input_time = self.data_summary[bin_name][obs_type][inst_name].get('input_time')
                    if input_time is not None:
                        break
                if input_time is not None:
                    break

            out = extract_features(
                self.z,
                self.data_summary,
                bin_name,
                self.observation_config,
                feature_stats=self.feature_stats,
                require_targets=self.require_targets,
            )
            bin_data = out[bin_name]
            graph_data = self.create_graph_fn(bin_data)
            graph_data.bin_name = bin_name

            # Add input_time to graph
            if input_time is not None:
                graph_data.input_time = input_time
                print(f"[Dataset] Added input_time: {input_time}")
            else:
                print(f"[Dataset] WARNING: No input_time found for {bin_name}")

            return graph_data
        except Exception as e:
            print(f"[Rank {rank}] [{self.tag}] ERROR processing {bin_name}: {e}")
            raise


# -------------------------
# DataModule
# -------------------------
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
        latent_step_hours=12,       # latent rollout support
        window_size="12h",          # binning window
        train_val_split_ratio=0.9,  # Default fallback, should be passed from training script
        prediction_mode=False,
        require_targets=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.prediction_mode = prediction_mode

        # If require_targets not specified, default based on prediction_mode
        # prediction_mode=True → require_targets=False (inference)
        # prediction_mode=False → require_targets=True (training/validation)
        if require_targets is None:
            self.require_targets = not prediction_mode
        else:
            self.require_targets = require_targets
        print(f"[DataModule] prediction_mode={prediction_mode}, require_targets={self.require_targets}")

        self.mesh_structure = mesh_structure
        self.feature_stats = feature_stats

        # Zarr handles (stable across window changes)
        self.z = None

        # Separate train/val summaries + bin name lists
        self.train_data_summary = None
        self.val_data_summary = None
        self.train_bin_names = []
        self.val_bin_names = []

        # Version counters (for debugging staleness)
        self._train_version = 0
        self._val_version = 0

        # If callbacks want separate windows, they will set these:
        # Default: create non-overlapping train/val split to prevent data leakage
        # Use split ratio passed from training script for consistency
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        train_days = int(total_days * train_val_split_ratio)

        default_train_start = pd.to_datetime(start_date)
        default_train_end = default_train_start + pd.Timedelta(days=train_days)
        default_val_start = default_train_end  # Validation starts where training ends
        default_val_end = pd.to_datetime(end_date)

        self.hparams.train_start = pd.to_datetime(kwargs.get("train_start", default_train_start))
        self.hparams.train_end = pd.to_datetime(kwargs.get("train_end", default_train_end))
        self.hparams.val_start = pd.to_datetime(kwargs.get("val_start", default_val_start))
        self.hparams.val_end = pd.to_datetime(kwargs.get("val_end", default_val_end))

        # In prediction mode, use all data (no split)
        if prediction_mode:
            self.hparams.train_start = pd.to_datetime(start_date)
            self.hparams.train_end = pd.to_datetime(end_date)
            self.hparams.val_start = pd.to_datetime(start_date)
            self.hparams.val_end = pd.to_datetime(end_date)
            print(f"[DataModule] Prediction mode: Using entire date range {start_date} to {end_date}")

        # Validate no overlap between train and validation windows to prevent data leakage (training mode only)
        if not prediction_mode and self.hparams.train_end > self.hparams.val_start:
            raise ValueError(
                f"Data leakage detected! Training window ({self.hparams.train_start} to {self.hparams.train_end}) "
                f"overlaps with validation window ({self.hparams.val_start} to {self.hparams.val_end}). "
                f"Ensure train_end <= val_start for proper temporal split."
            )

        # Log the train/val split for transparency
        train_days = (self.hparams.train_end - self.hparams.train_start).days
        val_days = (self.hparams.val_end - self.hparams.val_start).days
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        denom_days = total_days if total_days != 0 else 1
        print(
            f"[DataModule] Train/Val Split - Train: {train_days} days ({train_days / denom_days * 100:.1f}%), "
            f"Val: {val_days} days ({val_days / denom_days * 100:.1f}%)"
        )
        print(f"[DataModule] Train window: {self.hparams.train_start.date()} to {self.hparams.train_end.date()}")
        print(f"[DataModule] Val window:   {self.hparams.val_start.date()} to {self.hparams.val_end.date()}")

        # Ensure latent_step_hours has a valid value
        if self.hparams.latent_step_hours is None:
            window_hours = int(self.hparams.window_size.replace('h', ''))
            self.hparams.latent_step_hours = window_hours

    # ------------- Setup / Zarr open -------------

    def setup(self, stage=None):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # Open Zarrs once
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
                                # Try without year tag first (for v5 data or when the zarr_name field already includes the year)
                                zarr_path_test = os.path.join(self.hparams.data_path, f"{zname}.zarr")

                                if os.path.isdir(zarr_path_test):
                                    zname += ".zarr"
                                else:
                                    # Add year tag for v6 data (extracted from start_date)
                                    year = self.hparams.start_date.split('-')[0]
                                    zname += f"_{year}.zarr"

                            zarr_path = os.path.join(self.hparams.data_path, zname)

                        if not os.path.isdir(zarr_path):
                            raise FileNotFoundError(f"Zarr not found: {zarr_path}")

                        # Use LRU cache; ensure int for max_size
                        store = LRUStoreCache(zarr.DirectoryStore(zarr_path), max_size=int(2e9))
                        self.z[obs_type][inst_name] = zarr.open(store, mode="r")

                        if rank == 0:
                            print(f"[ZARR] {obs_type}/{inst_name} -> {zarr_path}")
                            try:
                                print("       keys:", list(self.z[obs_type][inst_name].keys())[:12])
                            except Exception:
                                pass

                        if obs_type == "conventional" and inst_name == "surface_obs":
                            if not os.path.basename(zarr_path).startswith("raw_surface_obs"):
                                print(f"[WARN] surface_obs expected raw_surface_obs*.zarr but got: {zarr_path}")

                    elif src == "nnja":
                        loader_path = inst_cfg["dataframe_loader"]
                        mod_name, fn_name = loader_path.rsplit(".", 1)
                        load_fn = getattr(importlib.import_module(mod_name), fn_name)

                        need = list(inst_cfg["var_map"].values())
                        need += [
                            inst_cfg.get("lat_col", "LAT"),
                            inst_cfg.get("lon_col", "LON"),
                            inst_cfg.get("time_col", "OBS_TIMESTAMP"),
                        ]

                        df = load_fn(
                            start_date=self.hparams.start_date,  # initial window used for first setup
                            end_date=self.hparams.end_date,
                            columns=need,
                        )
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

        # Build TRAIN and VAL summaries for current windows
        print(
            f"[Rank {rank}] [DM.setup stage={stage}] "
            f"train_window={self.hparams.train_start}..{self.hparams.train_end} | "
            f"val_window={self.hparams.val_start}..{self.hparams.val_end}"
        )

        self._rebuild_train_summary()
        self._rebuild_val_summary()

        if stage in (None, "fit"):
            # For now we use the full lists produced by organize_bins_times;
            # callbacks can narrow them by changing windows and triggering reload.
            pass

    # ------------- Summary (re)builders -------------
    def _rebuild_train_summary(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.train_data_summary = organize_bins_times(
            self.z,
            self.hparams.train_start,
            self.hparams.train_end,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
            window_size=self.hparams.window_size,
            latent_step_hours=self.hparams.latent_step_hours,
            require_targets=True,
        )
        self.train_bin_names = sorted(self.train_data_summary.keys(), key=lambda x: int(x.replace("bin", "")))
        print(
            f"[Rank {rank}] [DM.train_summary] v{self._train_version} sum_id={id(self.train_data_summary)} "
            f"bins={len(self.train_bin_names)} first={self.train_bin_names[0] if self.train_bin_names else None}"
        )

    def _rebuild_val_summary(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.val_data_summary = organize_bins_times(
            self.z,
            self.hparams.val_start,
            self.hparams.val_end,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
            window_size=self.hparams.window_size,
            latent_step_hours=self.hparams.latent_step_hours,
            require_targets=self.require_targets,
        )
        self.val_bin_names = sorted(self.val_data_summary.keys(), key=lambda x: int(x.replace("bin", "")))
        print(
            f"[Rank {rank}] [DM.val_summary]   v{self._val_version} sum_id={id(self.val_data_summary)} "
            f"bins={len(self.val_bin_names)} first={self.val_bin_names[0] if self.val_bin_names else None}"
        )

    # ------------- Window setters for callbacks -------------
    def set_train_window(self, start_dt, end_dt):
        self.hparams.train_start = pd.to_datetime(start_dt)
        self.hparams.train_end = pd.to_datetime(end_dt)
        self._train_version += 1
        print(f"[DM.set_train_window] v{self._train_version} -> {self.hparams.train_start} .. {self.hparams.train_end}")
        # Rebuild summary/bin names immediately so the *next* dataloader reload sees fresh objects
        self._rebuild_train_summary()

    def set_val_window(self, start_dt, end_dt):
        self.hparams.val_start = pd.to_datetime(start_dt)
        self.hparams.val_end = pd.to_datetime(end_dt)
        self._val_version += 1
        print(f"[DM.set_val_window]   v{self._val_version} -> {self.hparams.val_start} .. {self.hparams.val_end}")
        self._rebuild_val_summary()

    # ------------- Graph builder -------------

    def _create_graph_structure(self, bin_data):
        data = HeteroData()

        # 1) Mesh nodes and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0]
        reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
        data["mesh", "to", "mesh"].edge_index = torch.cat([m2m_edge_index, reverse_edges], dim=1)
        data["mesh", "to", "mesh"].edge_attr = torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0)

        window_hours = int(self.hparams.window_size.replace('h', ''))

        # Sanity check: ensure window_hours is divisible by latent_step_hours
        if window_hours % self.hparams.latent_step_hours != 0:
            raise ValueError(f"window_size ({window_hours}h) must be divisible by latent_step_hours ({self.hparams.latent_step_hours}h)")

        num_latent_steps = window_hours // self.hparams.latent_step_hours

        # 3) Observation data and mesh connections
        # ALL instruments get the same node structure based on detected batch mode
        for obs_type, instruments in self.hparams.observation_config.items():
            for inst_name, inst_cfg in instruments.items():

                # Check if this instrument has data for this time bin
                if obs_type in bin_data and inst_name in bin_data[obs_type]:
                    inst_dict = bin_data[obs_type][inst_name]
                    self._create_latent_nodes(data, inst_name, inst_dict, num_latent_steps)
                else:
                    # MISSING INSTRUMENT: Create empty nodes with same structure as present instruments
                    self._create_empty_latent_nodes(data, inst_name, inst_cfg, num_latent_steps)

        return data

    def _create_latent_nodes(self, data, inst_name, inst_dict, num_latent_steps):
        """Create nodes for instrument with data in latent mode."""
        # Input features (same for all steps)
        node_type_input = f"{inst_name}_input"
        if "input_features_final" in inst_dict:
            data[node_type_input].x = _t32(inst_dict["input_features_final"])

            # Store pressure level index for radiosonde and aircraft (if available)
            if "input_pressure_level" in inst_dict:
                data[node_type_input].pressure_level = inst_dict["input_pressure_level"].long()
                print(
                    f"[DATAMODULE] Stored pressure_level for {node_type_input}: "
                    f"shape={data[node_type_input].pressure_level.shape}, "
                    f"range=[{data[node_type_input].pressure_level.min()}, {data[node_type_input].pressure_level.max()}]"
                )
            elif inst_name in ["radiosonde", "aircraft"]:
                print(f"[DATAMODULE] WARNING: No pressure_level found for {node_type_input}! Data may not be preprocessed with new code.")

            # Create encoder edges (observation to mesh)
            if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
                grid_lat_deg = inst_dict["input_lat_deg"]
                grid_lon_deg = inst_dict["input_lon_deg"]

                # Keep lat/lon on the observation nodes (used by FSOI matching)
                data[node_type_input].lat = _t32(grid_lat_deg)
                data[node_type_input].lon = _t32(grid_lon_deg)

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
        if "target_features_final_list" not in inst_dict:
            return

        for step in range(num_latent_steps):
            if step >= len(inst_dict["target_features_final_list"]):
                continue

            node_type_target = f"{inst_name}_target_step{step}"
            target_features = inst_dict["target_features_final_list"][step]

            # Get channel mask and check validity
            target_channel_mask = inst_dict.get("target_channel_mask_list", [None])[step] if step < len(
                inst_dict.get("target_channel_mask_list", [])) else None

            if target_channel_mask is not None:
                target_channel_mask = target_channel_mask.to(torch.bool)
                keep_t = target_channel_mask.any(dim=1)  # Keep rows with ANY valid channel
            else:
                keep_t = torch.ones((target_features.shape[0],), dtype=torch.bool)

            # Handle empty case
            if keep_t.sum() == 0:
                data[node_type_target].y = torch.empty((0, target_features.shape[1]), dtype=torch.float32)
                data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
                data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
                data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
                data[node_type_target].target_channel_mask = torch.empty((0, target_features.shape[1]), dtype=torch.bool)
                data[node_type_target].target_pressure_hpa = torch.empty((0,), dtype=torch.float32)
                continue

            keep_np = keep_t.cpu().numpy()

            # Filter all data
            y_t = target_features[keep_t]
            mask_t = target_channel_mask[keep_t] if target_channel_mask is not None else torch.ones_like(y_t, dtype=torch.bool)

            data[node_type_target].y = _t32(y_t)
            data[node_type_target].target_channel_mask = _t32(mask_t)

            # Metadata
            if "target_metadata_list" in inst_dict and step < len(inst_dict["target_metadata_list"]):
                tgt_meta = inst_dict["target_metadata_list"][step][keep_t]
                data[node_type_target].target_metadata = _t32(tgt_meta)

            # Scan angle handling per-instrument (config-driven)
            # Determine observation type to look up config
            obs_type = "satellite" if inst_name in self.hparams.observation_config.get("satellite", {}) else "conventional"
            scan_angle_cols = self.hparams.observation_config[obs_type][inst_name].get("scan_angle_channels", 1)

            if "scan_angle_list" in inst_dict and step < len(inst_dict["scan_angle_list"]):
                x_aux = inst_dict["scan_angle_list"][step][keep_t]

                # Validate and pad/truncate to expected dimensions
                if x_aux.shape[-1] != scan_angle_cols:
                    if x_aux.shape[-1] > scan_angle_cols:
                        x_aux = x_aux[:, :scan_angle_cols]
                    else:
                        pad_cols = scan_angle_cols - x_aux.shape[-1]
                        padding = torch.zeros((x_aux.shape[0], pad_cols), dtype=x_aux.dtype, device=x_aux.device)
                        x_aux = torch.cat([x_aux, padding], dim=-1)
            else:
                x_aux = torch.zeros((y_t.shape[0], scan_angle_cols), dtype=torch.float32)
            data[node_type_target].x = _t32(x_aux)

            # Instrument ID
            if "instrument_id" in inst_dict:
                data[node_type_target].instrument_ids = torch.full(
                    (y_t.shape[0],),
                    inst_dict["instrument_id"],
                    dtype=torch.long
                )

            # Pressure data for radiosonde and aircraft (used for evaluation CSV)
            if "target_pressure_hpa_list" in inst_dict and step < len(inst_dict["target_pressure_hpa_list"]):
                pressure_hpa = inst_dict["target_pressure_hpa_list"][step][keep_np]
                data[node_type_target].target_pressure_hpa = _t32(torch.tensor(pressure_hpa, dtype=torch.float32))

            # Store pressure level index for radiosonde and aircraft (if available)
            if "target_pressure_level_list" in inst_dict and step < len(inst_dict["target_pressure_level_list"]):
                pressure_level_idx = inst_dict["target_pressure_level_list"][step][keep_t]
                data[node_type_target].pressure_level = pressure_level_idx.long()
                print(
                    f"[DATAMODULE] Stored pressure_level for {node_type_target}: "
                    f"shape={data[node_type_target].pressure_level.shape}, "
                    f"range=[{data[node_type_target].pressure_level.min()}, {data[node_type_target].pressure_level.max()}]"
                )
            elif inst_name in ["radiosonde", "aircraft"]:
                print(f"[DATAMODULE] WARNING: No pressure_level found for {node_type_target}! Data may not be preprocessed with new code.")

            # Edges - filter lat/lon too
            if ("target_lat_deg_list" in inst_dict and "target_lon_deg_list" in inst_dict):
                target_lat_deg = inst_dict["target_lat_deg_list"][step][keep_np]
                target_lon_deg = inst_dict["target_lon_deg_list"][step][keep_np]

                # Keep lat/lon on the target nodes (used by FSOI matching)
                data[node_type_target].lat = _t32(target_lat_deg)
                data[node_type_target].lon = _t32(target_lon_deg)

                if len(target_lat_deg) > 0:
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

    def _create_empty_latent_nodes(self, data, inst_name, inst_cfg, num_latent_steps):
        """Create empty nodes for missing instrument in latent mode."""
        # Create empty input node
        node_type_input = f"{inst_name}_input"
        data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
        data[node_type_input].lat = torch.empty((0,), dtype=torch.float32)
        data[node_type_input].lon = torch.empty((0,), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 3), dtype=torch.float32)

        # Create empty target nodes for all latent steps
        for step in range(num_latent_steps):
            node_type_target = f"{inst_name}_target_step{step}"
            data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
            # Get scan angle dimension from config
            obs_type = "satellite" if inst_name in self.hparams.observation_config.get("satellite", {}) else "conventional"
            scan_angle_dim = self.hparams.observation_config[obs_type][inst_name].get("scan_angle_channels", 1)
            data[node_type_target].x = torch.empty((0, scan_angle_dim), dtype=torch.float32)
            metadata_dim = len(inst_cfg.get("metadata", [])) + LAT_LON_COLUMNS  # lat/lon + metadata columns
            data[node_type_target].target_metadata = torch.empty((0, metadata_dim), dtype=torch.float32)
            data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
            data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)
            data[node_type_target].target_pressure_hpa = torch.empty((0,), dtype=torch.float32)
            data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 3), dtype=torch.float32)
            data[node_type_target].pos = torch.empty((0, LAT_LON_COLUMNS), dtype=torch.float32)  # from standard mode, seems unused
            data[node_type_target].num_nodes = 0  # from standard mode, seems unused
            data[node_type_target].lat = torch.empty((0,), dtype=torch.float32)
            data[node_type_target].lon = torch.empty((0,), dtype=torch.float32)

    # ------------- DataLoaders -------------
    def _worker_init(self, worker_id):
        import numpy as np
        base_seed = int(torch.initial_seed()) % 2**31
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(
            f"[WorkerInit] rank={rank} worker={worker_id} pid={os.getpid()} seed={base_seed} "
            f"train_sum_id={id(self.train_data_summary)} val_sum_id={id(self.val_data_summary)}"
        )

    def train_dataloader(self):
        ds = BinDataset(
            self.train_bin_names,
            self.train_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            require_targets=True,  # Training always requires targets
            tag="TRAIN",
        )
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,   # safer while debugging stale refs
            worker_init_fn=self._worker_init,
        )
        print(f"[DL] TRAIN v{self._train_version} loader_id={id(loader)} ds_id={id(ds)} "
              f"sum_id={id(self.train_data_summary)} bins={len(self.train_bin_names)}")
        return loader

    def val_dataloader(self):
        if not self.val_bin_names:
            return None
        ds = BinDataset(
            self.val_bin_names,
            self.val_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            require_targets=True,  # Validation requires targets for comparison
            tag="VAL",
        )
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=self._worker_init,
        )
        print(f"[DL] VAL   v{self._val_version} loader_id={id(loader)} ds_id={id(ds)} "
              f"sum_id={id(self.val_data_summary)} bins={len(self.val_bin_names)}")
        return loader

    def predict_dataloader(self):
        """Create dataloader for prediction/inference mode."""
        print("\n[PREDICT] Setting up prediction dataloader")

        # Use val_data_summary for prediction
        if not hasattr(self, 'val_data_summary') or not self.val_data_summary:
            print("[PREDICT] Building prediction data summary...")
            self._rebuild_val_summary()

        if not self.val_bin_names:
            print("[WARN] No bins found for prediction!")
            return None

        # Create dataset
        ds = BinDataset(
            self.val_bin_names,
            self.val_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            require_targets=self.require_targets,  # Use datamodule's require_targets setting
            tag="PREDICT",
        )

        # Create dataloader
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=1,  # Single worker for sequential processing
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=self._worker_init,
        )

        print(f"[PREDICT] Dataloader created: {len(self.val_bin_names)} bins")
        print(f"[PREDICT] require_targets={self.require_targets}")

        return loader

    def fsoi_dataloader(self):
        """Deterministic dataloader for FSOI.

        Uses the same bin ordering as prediction, but enforces batch_size=1.
        """
        if not hasattr(self, 'val_data_summary') or not self.val_data_summary:
            self._rebuild_val_summary()

        if not self.val_bin_names:
            return None

        ds = BinDataset(
            self.val_bin_names,
            self.val_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            require_targets=self.require_targets,
            tag="FSOI",
        )

        return PyGDataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=self._worker_init,
        )
