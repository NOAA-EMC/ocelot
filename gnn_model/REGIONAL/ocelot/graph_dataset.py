import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.spatial
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ocelot.constants import FEATURES_AUX_DIM, STATIC_CONTEXT_DIM
from ocelot.logger import logger
from ocelot.dataset_timeseries import generate_binned_timestamp_list, ParquetDataManager
from ocelot.create_mesh_graph_global import create_mesh, obs_mesh_conn
from ocelot.create_mesh_graph_regional import obs_mesh_conn_regional, project_coords as project_coords_regional
from ocelot.create_mesh_graph_diatom import create_obs_conn_mesh
from ocelot.graph_utils import (
    ENCODE_OBS_TO_MESH_REL,
    build_conventional_input_x,
    build_satellite_input_x,
    target_node_type,
)


class GraphDataset(Dataset, ABC):
    """
    Each sample uses one input time bin and ``k = num_target_bins`` consecutive target bins.

    When ``k > 1``, for every target horizon we also build **encoder-style** edges
    ``(target_node_type, ENCODE_OBS_TO_MESH_REL, mesh)`` (see ``ocelot.graph_utils``)
    with the same ``obs_mesh_conn(..., o2m=True)`` construction as
    ``(obs_input, "to", mesh)``, using that horizon's target lat/lon. This is
    for later use when predictions at target locations are fed back as pseudo-inputs.
    """

    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        observation_config: Dict[str, Dict],
        mesh_structure: Dict = None,
        args=None
    ):
        super().__init__()

        self.loader = None
        self.binned_samples = None
        self.z = None
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observation_config = observation_config
        self.mesh_structure = mesh_structure
        self.args = args

    @abstractmethod
    def _obs_to_mesh_edges(self, lat_deg, lon_deg, o2m: bool = True) -> Tuple:
        """Build observation↔mesh edges. Returns (edge_index, edge_attr)."""
        ...

    @abstractmethod
    def _mesh_num_nodes(self) -> int:
        """Return the number of mesh nodes for this domain."""
        ...

    @property
    def num_target_bins(self) -> int:
        return max(1, int(getattr(self.args, "num_target_bins", 1)))

    def setup(self):
        """
        Prepare data for training/validation using time-binned zarr structure.
        """
        start_mmdd = self.start_date.strftime('%m/%d')
        end_mmdd = self.end_date.strftime('%m/%d')
        year = self.start_date.year
        delta_time = getattr(self.args, 'delta_time', 12)
        self.binned_samples = generate_binned_timestamp_list(
            start_mmdd, end_mmdd, year, delta_time)

        print(f'{self.data_path}')
        self.loader = ParquetDataManager(
            data_dir=self.data_path,
            observation_config=self.observation_config,
            feature_stats=self.args.feature_stats,
            fill_values=getattr(self.args, 'fill_values', None),
            delta_time=delta_time,
        )

    def __len__(self):
        k = self.num_target_bins
        return max(0, len(self.binned_samples) - k)

    def __getitem__(self, idx: int) -> Optional[HeteroData]:
        if isinstance(idx, list):
            raise RuntimeError(
                f"idx is a list {idx}! This means you are not using PyG DataLoader."
            )
        logger.debug(f'__getitem__ called with index: {idx}')
        k = self.num_target_bins
        if idx + k >= len(self.binned_samples):
            logger.warning(
                f"Skipping index {idx}: need {k} target bins after input.")
            return None

        input_bin_name = self.binned_samples[idx]
        target_bin_names = [self.binned_samples[idx + 1 + h] for h in range(k)]
        logger.debug(
            f'Input bin: {input_bin_name}, Target bins ({k}): {target_bin_names}')
        input_bin_data = self.loader.get_data_for_bin(input_bin_name)
        target_bin_datas = [self.loader.get_data_for_bin(
            bn) for bn in target_bin_names]

        logger.debug(
            f"For idx {idx}, input_bin_data is {'None' if input_bin_data is None else 'Not None'}")
        for h, (bn, tb) in enumerate(zip(target_bin_names, target_bin_datas)):
            logger.debug(
                f"For idx {idx}, target_bin_data[{h}] ({bn}) is {'None' if tb is None else 'Not None'}"
            )

        if input_bin_data is None or any(
                tb is None for tb in target_bin_datas):
            logger.warning(f"Skipping index {idx} due to missing bin data.")
            return None

        data = HeteroData()

        any_data_missing_or_empty = False
        missing_or_empty_summary = {}

        for obs_type in self.observation_config.keys():
            all_inst_names_exist_and_nonempty = True
            missing_or_empty_inst_names = []
            for inst_name in self.observation_config[obs_type].keys():
                logger.debug(f'Processing {obs_type} for {inst_name}')

                targets_present = all(
                    obs_type in tb
                    and inst_name in tb[obs_type]
                    and tb[obs_type][inst_name] is not None
                    for tb in target_bin_datas
                )
                if (
                    obs_type in input_bin_data
                    and inst_name in input_bin_data[obs_type]
                    and input_bin_data[obs_type][inst_name] is not None
                    and targets_present
                ):
                    input_inst_data = input_bin_data[obs_type][inst_name]
                    target_inst_list = [
                        tb[obs_type][inst_name] for tb in target_bin_datas
                    ]

                    input_features_norm = input_inst_data.get(
                        "features_norm", None)
                    inst_nonempty = (
                        input_features_norm is not None and (
                            not torch.is_tensor(input_features_norm) or input_features_norm.numel() > 0) and len(
                            input_inst_data.get(
                                "lat_deg", [])) > 0 and len(
                            input_inst_data.get(
                                "lon_deg", [])) > 0)
                    for target_inst_data in target_inst_list:
                        target_features_norm = target_inst_data.get(
                            "features_norm", None)
                        inst_nonempty = inst_nonempty and (
                            target_features_norm is not None
                            and (
                                not torch.is_tensor(target_features_norm)
                                or target_features_norm.numel() > 0
                            )
                            and len(target_inst_data.get("lat_deg", [])) > 0
                            and len(target_inst_data.get("lon_deg", [])) > 0
                        )
                    if not inst_nonempty:
                        all_inst_names_exist_and_nonempty = False
                        missing_or_empty_inst_names.append(inst_name)
                        logger.debug(
                            f"Creating empty nodes for empty instrument: {obs_type}_{inst_name}"
                        )
                        inst_cfg = self.observation_config[obs_type][inst_name]
                        self._create_empty_nodes(
                            data, obs_type, inst_name, inst_cfg)
                        continue

                    for key, tensor in input_inst_data.items():
                        if torch.is_tensor(
                                tensor) and torch.isnan(tensor).any():
                            logger.warning(
                                f"Found NaN in input tensor '{key}' for {obs_type}_{inst_name}.")

                    for h, target_inst_data in enumerate(target_inst_list):
                        for key, tensor in target_inst_data.items():
                            if torch.is_tensor(
                                    tensor) and torch.isnan(tensor).any():
                                logger.warning(
                                    f"Found NaN in target tensor '{key}' for {obs_type}_{inst_name} (h={h})."
                                )

                    node_type_input = f"{obs_type}_{inst_name}_input"
                    k_bins = self.num_target_bins

                    # --- 1. Process Input Node Features ---
                    input_features = input_inst_data["features_norm"].float()
                    input_aux = input_inst_data["features_aux"].float()
                    if obs_type == "satellite":
                        features_valid_mask = input_inst_data["features_valid_mask"]
                        metadata_valid_mask = input_inst_data.get(
                            "metadata_valid_mask", None)

                        rows_to_keep = torch.all(features_valid_mask, dim=1)
                        if metadata_valid_mask is not None:
                            rows_to_keep = rows_to_keep & torch.all(
                                metadata_valid_mask, dim=1)

                        logger.debug(
                            f"  rows_to_keep: {rows_to_keep.shape}, sum={rows_to_keep.sum()}")

                        input_features = input_features[rows_to_keep]
                        input_aux = input_aux[rows_to_keep]
                        input_meta = input_inst_data["features_meta"].float()[
                            rows_to_keep]

                        rows_to_keep_np = rows_to_keep.cpu().numpy()
                        input_lat_deg_filtered = input_inst_data["lat_deg"][rows_to_keep_np]
                        input_lon_deg_filtered = input_inst_data["lon_deg"][rows_to_keep_np]

                        data[node_type_input].x = torch.cat(
                            [input_features, input_meta, input_aux], dim=1)
                    else:
                        input_lat_deg_filtered = input_inst_data["lat_deg"]
                        input_lon_deg_filtered = input_inst_data["lon_deg"]
                        input_valid_mask = input_inst_data["features_valid_mask"].float(
                        )
                        input_meta_conv = input_inst_data.get("features_meta")
                        meta_for_input = (
                            input_meta_conv.float()
                            if input_meta_conv is not None and torch.is_tensor(input_meta_conv)
                            else None
                        )
                        data[node_type_input].x = build_conventional_input_x(
                            input_features,
                            input_aux,
                            input_valid_mask,
                            features_meta=meta_for_input,
                        )

                    # --- 2. Process Target Node Features (k horizons) ---
                    for h, target_inst_data in enumerate(target_inst_list):
                        node_type_target = target_node_type(
                            obs_type, inst_name, h, k_bins)
                        self._populate_target_nodes(
                            data, obs_type, inst_name, target_inst_data, node_type_target)

                    # --- 3. Build encoder edges: obs_input → mesh ---
                    edge_index_encoder, edge_attr_encoder = self._obs_to_mesh_edges(
                        input_lat_deg_filtered, input_lon_deg_filtered, o2m=True)

                    if edge_index_encoder.numel() == 0:
                        inst_cfg = self.observation_config[obs_type][inst_name]
                        self._create_empty_nodes(
                            data, obs_type, inst_name, inst_cfg)
                        all_inst_names_exist_and_nonempty = False
                        missing_or_empty_inst_names.append(inst_name)
                        continue

                    data[node_type_input, "to",
                         "mesh"].edge_index = edge_index_encoder
                    data[node_type_input, "to",
                         "mesh"].edge_attr = edge_attr_encoder

                    # --- 4. Build decoder + encode-feedback edges (k horizons) ---
                    for h, target_inst_data in enumerate(target_inst_list):
                        node_type_target = target_node_type(
                            obs_type, inst_name, h, k_bins)
                        (target_lat_deg_filtered,
                         target_lon_deg_filtered,
                         ) = self._target_lat_lon_for_edges(obs_type,
                                                            target_inst_data)

                        edge_index_knn, edge_attr_knn = self._obs_to_mesh_edges(
                            target_lat_deg_filtered, target_lon_deg_filtered, o2m=False
                        )

                        data["mesh", "to",
                             node_type_target].edge_index = edge_index_knn
                        data["mesh", "to", node_type_target].edge_attr = edge_attr_knn

                        lat = torch.tensor(
                            target_lat_deg_filtered, dtype=torch.float)
                        lon = torch.tensor(
                            target_lon_deg_filtered, dtype=torch.float)
                        data[node_type_target].pos = torch.stack(
                            [lon, lat], dim=1)

                        if k_bins > 1:
                            enc_ei, enc_ea = self._obs_to_mesh_edges(
                                target_lat_deg_filtered, target_lon_deg_filtered, o2m=True)
                            if enc_ei.numel() == 0:
                                raise ValueError(
                                    "No encode_obs_to_mesh edges at target locations; "
                                    "try increasing cutoff_factor.")
                            et = (
                                node_type_target,
                                ENCODE_OBS_TO_MESH_REL,
                                "mesh")
                            data[et].edge_index = enc_ei
                            data[et].edge_attr = enc_ea

                else:
                    all_inst_names_exist_and_nonempty = False
                    missing_or_empty_inst_names.append(inst_name)
                    logger.debug(
                        f'Creating empty nodes for missing instrument: {obs_type}_{inst_name}')
                    inst_cfg = self.observation_config[obs_type][inst_name]
                    self._create_empty_nodes(
                        data, obs_type, inst_name, inst_cfg)

            if not all_inst_names_exist_and_nonempty:
                any_data_missing_or_empty = True
                missing_or_empty_summary[obs_type] = missing_or_empty_inst_names
                logger.warning(
                    f"For idx {idx}, obs_type '{obs_type}' has missing/empty instruments: {missing_or_empty_inst_names}"
                )

        # MESH
        data["mesh"].num_nodes = self._mesh_num_nodes()
        data["mesh"].x = torch.zeros(data["mesh"].num_nodes, 1)
        logger.debug(f"Mesh node features shape: {data['mesh'].x.shape}")
        if any_data_missing_or_empty:
            logger.warning(
                f"Finished processing bin: {input_bin_name} -> {target_bin_names} "
                f"(missing/empty: {missing_or_empty_summary})")
        else:
            logger.info(
                f'Finished processing bin: {input_bin_name} -> {target_bin_names}')
        return data

    @staticmethod
    def _target_lat_lon_for_edges(
            obs_type: str,
            target_inst_data: Dict) -> tuple:
        """Lat/lon arrays aligned with rows used for mesh→target edges (after satellite row filter)."""
        if obs_type == "satellite":
            target_features_valid_mask = target_inst_data["features_valid_mask"].bool(
            )
            target_metadata_valid_mask = target_inst_data.get(
                "metadata_valid_mask", None)
            target_rows_to_keep = torch.all(target_features_valid_mask, dim=1)
            if target_metadata_valid_mask is not None:
                target_rows_to_keep = target_rows_to_keep & torch.all(
                    target_metadata_valid_mask, dim=1)
            target_rows_to_keep_np = target_rows_to_keep.cpu().numpy()
            return (
                target_inst_data["lat_deg"][target_rows_to_keep_np],
                target_inst_data["lon_deg"][target_rows_to_keep_np],
            )
        return target_inst_data["lat_deg"], target_inst_data["lon_deg"]

    def _populate_target_nodes(
        self,
        data: HeteroData,
        obs_type: str,
        inst_name: str,
        target_inst_data: Dict,
        node_type_target: str,
    ) -> None:
        """Ground truth and decoder context (``x``) for one target horizon node type."""
        target_features = target_inst_data["features_norm"].float()
        target_features_valid_mask = target_inst_data["features_valid_mask"].bool(
        )

        if obs_type == "satellite":
            target_metadata_valid_mask = target_inst_data.get(
                "metadata_valid_mask", None)
            target_rows_to_keep = torch.all(target_features_valid_mask, dim=1)
            if target_metadata_valid_mask is not None:
                target_rows_to_keep = target_rows_to_keep & torch.all(
                    target_metadata_valid_mask, dim=1)

            target_features = target_features[target_rows_to_keep]
            target_features_valid_mask = target_features_valid_mask[target_rows_to_keep]

            target_meta = target_inst_data["features_meta"].float()[
                target_rows_to_keep]
            target_aux = target_inst_data["features_aux"].float()[
                target_rows_to_keep]
            data[node_type_target].x = torch.cat(
                [target_meta, target_aux], dim=1)
        else:
            target_aux = target_inst_data["features_aux"].float()
            target_meta_t = target_inst_data.get("features_meta")
            if target_meta_t is not None and torch.is_tensor(target_meta_t):
                data[node_type_target].x = torch.cat(
                    [target_meta_t.float(), target_aux], dim=1
                )
            else:
                data[node_type_target].x = target_aux

        data[node_type_target].y = target_features
        data[node_type_target].valid_mask = target_features_valid_mask
        data[node_type_target].num_nodes = target_features.shape[0]

    def _create_empty_nodes(
            self,
            data: HeteroData,
            obs_type: str,
            inst_name: str,
            inst_cfg: Dict):
        """
        Create empty nodes for missing instrument to maintain consistent graph structure.
        This ensures the model can handle samples where some instruments have no data.
        """
        node_type_input = f"{obs_type}_{inst_name}_input"
        k = self.num_target_bins

        target_dim = inst_cfg.get('target_dim', 0)
        features_aux_dim = FEATURES_AUX_DIM
        num_metadata = len(inst_cfg.get('metadata', []))

        if obs_type == "satellite":
            input_dim = target_dim + num_metadata + features_aux_dim
        else:
            input_dim = target_dim + num_metadata + features_aux_dim + target_dim

        data[node_type_input].x = torch.empty(
            (0, input_dim), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty(
            (2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty(
            (0, 4), dtype=torch.float32)

        target_context_dim = num_metadata + features_aux_dim
        for h in range(k):
            node_type_target = target_node_type(obs_type, inst_name, h, k)
            data[node_type_target].y = torch.empty(
                (0, target_dim), dtype=torch.float32)
            data[node_type_target].valid_mask = torch.empty(
                (0, target_dim), dtype=torch.bool)
            data[node_type_target].num_nodes = 0
            data[node_type_target].x = torch.empty(
                (0, target_context_dim), dtype=torch.float32)
            data["mesh", "to", node_type_target].edge_index = torch.empty(
                (2, 0), dtype=torch.long)
            data["mesh", "to", node_type_target].edge_attr = torch.empty(
                (0, 4), dtype=torch.float32)
            data[node_type_target].pos = torch.empty(
                (0, 2), dtype=torch.float32)
            if k > 1:
                et = (node_type_target, ENCODE_OBS_TO_MESH_REL, "mesh")
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[et].edge_attr = torch.empty((0, 4), dtype=torch.float32)


class RegionalGraphDataset(GraphDataset):
    """GraphDataset for a regional (limited-area) mesh domain."""

    def _obs_to_mesh_edges(self, lat_deg, lon_deg, o2m: bool = True) -> Tuple:
        if len(lat_deg) == 0:
            return torch.empty(
                (2, 0), dtype=torch.long), torch.empty(
                (0, 4), dtype=torch.float32)
        G_bottom = self.mesh_structure["G_bottom_mesh"]
        xy = project_coords_regional(lat_deg, lon_deg)
        if o2m:
            conn = 'g2m'
        else:
            conn = 'm2g'
        obs2grid_diatom = create_obs_conn_mesh(
            xy, G_bottom, self.args, conn=conn)
        dist_t = obs2grid_diatom['graph'].edge_weights
        vdiff_t = obs2grid_diatom['graph'].edge_vdiffs
        dm = obs2grid_diatom['mesh_distance']
        zeros = torch.zeros(dist_t.shape[0], 1)
        edge_attr = torch.cat(
            [(dist_t / dm).unsqueeze(1), vdiff_t / dm, zeros], dim=1
        )
        edge_index = obs2grid_diatom['graph'].edge_index
        return edge_index, edge_attr
        # return obs_mesh_conn_regional(xy, G_bottom, self.args, o2m=o2m)

    def _mesh_num_nodes(self) -> int:
        return self.mesh_structure["m2m_graphs"][0].pos.shape[0]


class GlobalGraphDataset(GraphDataset):
    """GraphDataset for a global (icosahedral) mesh domain."""

    def _obs_to_mesh_edges(self, lat_deg, lon_deg, o2m: bool = True) -> Tuple:
        return obs_mesh_conn(
            lat_deg,
            lon_deg,
            self.mesh_structure['m2m_graphs'],
            self.mesh_structure['mesh_lat_lon_list'],
            self.mesh_structure['mesh_list'],
            o2m=o2m,
        )

    def _mesh_num_nodes(self) -> int:
        return self.mesh_structure['mesh_lat_lon_list'][0].shape[0]


_STATE_GES = "ges"
_STATE_ANAL = "anal"

_URMA_STATIC_DATA_DIR = '/scratch3/NCEPDEV/da/Xin.C.Jin/my_data/urma'


class DARegionalGraphDataset(RegionalGraphDataset):
    """
    Regional dataset where the target time bin equals the input time bin.

    Adds a ``state`` obs_type with two sub-instruments:

    * ``ges``  – first-guess / background: encoded as input nodes.
    * ``anal`` – analysis / corrected state: decoded as target nodes (``y``).

    For all other obs_types the same bin is used for both input and target.
    """

    @property
    def num_target_bins(self) -> int:
        return 1

    def setup(self):
        super().setup()
        # Build a private loader for anal data using anal_zarr_name from the ges config.
        # "anal" never appears as a top-level observation_config key, so nothing else
        # in the codebase needs to know about it.
        ges_cfg = self.observation_config.get("state", {}).get(_STATE_GES, {})
        anal_zarr_name = ges_cfg.get("anal_zarr_name", _STATE_ANAL)
        anal_obs_config = {
            "state": {
                _STATE_ANAL: {**ges_cfg, "zarr_name": anal_zarr_name}
            }
        }
        delta_time = getattr(self.args, "delta_time", 12)
        # anal uses the same variable stats as ges; map 'anal' -> ges stats so
        # ParquetDataManager doesn't fall back to mean=0/std=1 defaults.
        anal_feature_stats = {
            **self.args.feature_stats,
            _STATE_ANAL: self.args.feature_stats.get(_STATE_GES, {}),
        }
        self.anal_loader = ParquetDataManager(
            data_dir=self.data_path,
            observation_config=anal_obs_config,
            feature_stats=anal_feature_stats,
            fill_values=getattr(self.args, "fill_values", None),
            delta_time=delta_time,
        )

        # Static per-gridpoint context (terrain height, land/sea mask) for the anal
        # decoder target nodes. Row order must match anal's grid-flatten order
        # 1:1.
        static_data_dir = getattr(
            self.args,
            "static_data_dir",
            _URMA_STATIC_DATA_DIR)
        terrain = np.load(
            os.path.join(
                static_data_dir,
                'urma2p5_terrain.npy')).astype(
            np.float32).flatten()
        slmask = np.load(
            os.path.join(
                static_data_dir,
                'urma2p5_slmask_nolakes.npz'))['slmask'].flatten()
        # z-score terrain height so its scale matches the other (already normalized)
        # input/target columns; slmask is already binary {0, 1} and needs no
        # scaling.
        terrain = (terrain - terrain.mean()) / terrain.std()
        self.terrain = torch.from_numpy(terrain)
        self.slmask = torch.from_numpy(slmask.astype(np.float32))

        # Optional: predict the full analysis field directly (y = anal) instead of
        # the DA increment (y = anal - ges). Mutually exclusive with
        # --normalize_increment, which only makes sense for the increment target.
        self.target_is_anal = bool(getattr(self.args, "target_is_anal", False))

        # Optional: rescale the DA increment (anal - ges) by offline-computed
        # per-variable increment stats (mean/std of anal_norm - ges_norm), instead
        # of leaving it implicitly normalized by field std alone. See
        # graphcast_residuals.md and ocelot/compute_increment_stats.py.
        self.normalize_increment = bool(
            getattr(self.args, "normalize_increment", False))
        if self.target_is_anal and self.normalize_increment:
            raise ValueError(
                "--target_is_anal and --normalize_increment are mutually exclusive.")
        if self.normalize_increment:
            increment_stats = getattr(self.args, "increment_stats", None) or {}
            feature_keys = ges_cfg["features"]
            missing = [k for k in feature_keys if k not in increment_stats]
            if missing:
                raise ValueError(
                    f"--normalize_increment is set but increment_stats is missing "
                    f"entries for: {missing}. Compute them with "
                    f"ocelot/compute_increment_stats.py and add a STATE_INCREMENT_STATS "
                    f"dict to the obs_config module.")
            mean = torch.tensor([increment_stats[k][0]
                                for k in feature_keys], dtype=torch.float32)
            std = torch.tensor([increment_stats[k][1]
                               for k in feature_keys], dtype=torch.float32)
            self.increment_mean = mean
            self.increment_std = torch.where(
                std > 0, std, torch.ones_like(std))

    def __len__(self) -> int:
        return len(self.binned_samples)

    def __getitem__(self, idx: int) -> Optional[HeteroData]:
        logger.debug(f"DARegionalGraphDataset.__getitem__({idx})")

        bin_name = self.binned_samples[idx]
        bin_data = self.loader.get_data_for_bin(bin_name)
        if bin_data is None:
            logger.warning(f"Skipping index {idx}: missing bin '{bin_name}'.")
            return None

        data = HeteroData()
        any_missing = False
        missing_summary: Dict = {}

        for obs_type in self.observation_config.keys():
            if obs_type == "state":
                ok = self._process_state_nodes(data, bin_data, bin_name, idx)
                if not ok:
                    any_missing = True
                    missing_summary["state"] = [_STATE_GES, _STATE_ANAL]
                continue

            all_inst_ok = True
            missing_insts = []
            for inst_name in self.observation_config[obs_type].keys():
                inst_cfg = self.observation_config[obs_type][inst_name]
                inst_data = bin_data.get(obs_type, {}).get(inst_name)
                fn = inst_data.get("features_norm") if inst_data else None
                nonempty = (
                    fn is not None
                    and (not torch.is_tensor(fn) or fn.numel() > 0)
                    and len((inst_data or {}).get("lat_deg", [])) > 0
                )
                if not nonempty:
                    all_inst_ok = False
                    missing_insts.append(inst_name)
                    self._create_empty_nodes(
                        data, obs_type, inst_name, inst_cfg)
                    continue

                self._process_same_time_instrument(
                    data, obs_type, inst_name, inst_data)

            if not all_inst_ok:
                any_missing = True
                missing_summary[obs_type] = missing_insts
                logger.warning(
                    f"idx={idx}: '{obs_type}' missing/empty: {missing_insts}")

        data["mesh"].num_nodes = self._mesh_num_nodes()
        data["mesh"].x = torch.zeros(data["mesh"].num_nodes, 1)
        if any_missing:
            logger.warning(
                f"Finished bin {bin_name} with missing: {missing_summary}")
        else:
            logger.info(f"Finished bin {bin_name}")
        return data

    def _process_same_time_instrument(
        self,
        data: HeteroData,
        obs_type: str,
        inst_name: str,
        inst_data: Dict,
    ) -> None:
        """Build encoder input and decoder target nodes from the same time-bin data."""
        node_type_input = f"{obs_type}_{inst_name}_input"
        node_type_target = target_node_type(obs_type, inst_name, 0, 1)

        input_features = inst_data["features_norm"].float()
        input_aux = inst_data["features_aux"].float()

        if obs_type == "satellite":
            fvm = inst_data["features_valid_mask"]
            mvm = inst_data.get("metadata_valid_mask")
            keep = torch.all(fvm, dim=1)
            if mvm is not None:
                keep = keep & torch.all(mvm, dim=1)
            keep_np = keep.cpu().numpy()
            input_features = input_features[keep]
            input_aux = input_aux[keep]
            input_meta = inst_data["features_meta"].float()[keep]
            lat = inst_data["lat_deg"][keep_np]
            lon = inst_data["lon_deg"][keep_np]
            data[node_type_input].x = torch.cat(
                [input_features, input_meta, input_aux], dim=1)
        else:
            lat = inst_data["lat_deg"]
            lon = inst_data["lon_deg"]
            valid_mask = inst_data["features_valid_mask"].float()
            meta = inst_data.get("features_meta")
            meta = meta.float() if meta is not None and torch.is_tensor(meta) else None
            data[node_type_input].x = build_conventional_input_x(
                input_features, input_aux, valid_mask, features_meta=meta
            )

        ei_enc, ea_enc = self._obs_to_mesh_edges(lat, lon, o2m=True)
        if ei_enc.numel() == 0:
            raise ValueError(
                "No encoder edges created; try increasing cutoff_factor.")
        data[node_type_input, "to", "mesh"].edge_index = ei_enc
        data[node_type_input, "to", "mesh"].edge_attr = ea_enc

        # target == same time: reuse inst_data
        self._populate_target_nodes(
            data,
            obs_type,
            inst_name,
            inst_data,
            node_type_target)

        target_lat, target_lon = self._target_lat_lon_for_edges(
            obs_type, inst_data)
        ei_dec, ea_dec = self._obs_to_mesh_edges(
            target_lat, target_lon, o2m=False)
        data["mesh", "to", node_type_target].edge_index = ei_dec
        data["mesh", "to", node_type_target].edge_attr = ea_dec
        data[node_type_target].pos = torch.stack(
            [torch.tensor(target_lon, dtype=torch.float),
             torch.tensor(target_lat, dtype=torch.float)],
            dim=1,
        )

    def _process_state_nodes(
        self, data: HeteroData, bin_data: Dict, bin_name: str, idx: int
    ) -> bool:
        """
        Handle ``state`` obs_type: ges → encoder input, anal → decoder target.
        Returns True when both sub-instruments are present and non-empty.
        """
        ges_data = bin_data.get("state", {}).get(_STATE_GES)
        anal_bin = self.anal_loader.get_data_for_bin(bin_name)
        anal_data = (anal_bin or {}).get("state", {}).get(_STATE_ANAL)

        # observation_config["state"] only needs one key ("ges"); anal shares
        # the same dims.
        ges_cfg = self.observation_config.get("state", {}).get(_STATE_GES, {})

        def _nonempty(d):
            if d is None:
                return False
            fn = d.get("features_norm")
            return (
                fn is not None
                and (not torch.is_tensor(fn) or fn.numel() > 0)
                and len(d.get("lat_deg", [])) > 0
            )

        if not _nonempty(ges_data) or not _nonempty(anal_data):
            logger.warning(
                f"idx={idx}: state ges/anal missing or empty; inserting empty nodes.")
            self._create_empty_state_nodes(data, ges_cfg)
            return False

        node_type_input = f"state_{_STATE_GES}_input"
        node_type_target = target_node_type("state", _STATE_GES, 0, 1)

        # Static per-gridpoint context (terrain height, land/sea mask). ges and anal
        # share the same URMA grid/row order (required for the elementwise increment
        # below), so the same static context applies to both the encoder input and
        # the decoder target.
        ges_features = ges_data["features_norm"].float()
        anal_features = anal_data["features_norm"].float()
        n_nodes = ges_features.shape[0]
        if anal_features.shape[0] != n_nodes or self.terrain.shape[0] != n_nodes:
            raise ValueError(
                f"Static grid size ({self.terrain.shape[0]}) does not match state node "
                f"count (ges={n_nodes}, anal={anal_features.shape[0]}); terrain/slmask "
                f"grid no longer lines up with state rows.")
        static_ctx = torch.stack([self.terrain, self.slmask], dim=1)

        # Encoder input from ges
        ges_aux = ges_data["features_aux"].float()
        ges_valid_mask = ges_data["features_valid_mask"].float()
        ges_meta = ges_data.get("features_meta")
        ges_meta = ges_meta.float() if ges_meta is not None and torch.is_tensor(ges_meta) else None
        data[node_type_input].x = torch.cat([build_conventional_input_x(
            ges_features, ges_aux, ges_valid_mask, features_meta=ges_meta), static_ctx, ], dim=1, )

        ges_lat = ges_data["lat_deg"]
        ges_lon = ges_data["lon_deg"]
        ei_enc, ea_enc = self._obs_to_mesh_edges(ges_lat, ges_lon, o2m=True)
        if ei_enc.numel() == 0:
            raise ValueError(
                "No state encoder edges; try increasing cutoff_factor.")
        data[node_type_input, "to", "mesh"].edge_index = ei_enc
        data[node_type_input, "to", "mesh"].edge_attr = ea_enc

        # Decoder target from anal. Default: residual formulation, y = anal - ges (DA
        # increment); at inference, full_prediction = ges_background + model_output.
        # With --target_is_anal: y = anal directly, no residual/background
        # add-back.
        anal_valid_mask = anal_data["features_valid_mask"].bool()
        anal_aux = anal_data["features_aux"].float()
        anal_meta = anal_data.get("features_meta")

        if anal_meta is not None and torch.is_tensor(anal_meta):
            data[node_type_target].x = torch.cat(
                [anal_meta.float(), anal_aux, static_ctx], dim=1)
        else:
            data[node_type_target].x = torch.cat([anal_aux, static_ctx], dim=1)
        if self.target_is_anal:
            # full analysis field, no residual formulation
            target = anal_features
        else:
            target = anal_features - ges_features
            logger.debug(
                f"DA increment stats — mean: {target.mean():.4f}, "
                f"std: {target.std():.4f}, "
                f"min: {target.min():.4f}, "
                f"max: {target.max():.4f} "
                f"(if std << 0.1, consider normalizing by std(increment) instead of std(field))")
            if self.normalize_increment:
                target = (target - self.increment_mean) / self.increment_std
                # Stored so downstream code (e.g. analyze_outputs.py) can invert:
                # anal_norm = ges + (y * increment_std + increment_mean)
                data[node_type_target].increment_mean = self.increment_mean
                data[node_type_target].increment_std = self.increment_std
            # background; add to model output at inference
            data[node_type_target].ges = ges_features
        # anal, or increment (anal - ges) optionally rescaled
        data[node_type_target].y = target
        data[node_type_target].valid_mask = anal_valid_mask
        data[node_type_target].num_nodes = anal_features.shape[0]

        anal_lat = anal_data["lat_deg"]
        anal_lon = anal_data["lon_deg"]
        ei_dec, ea_dec = self._obs_to_mesh_edges(anal_lat, anal_lon, o2m=False)
        data["mesh", "to", node_type_target].edge_index = ei_dec
        data["mesh", "to", node_type_target].edge_attr = ea_dec
        data[node_type_target].pos = torch.stack(
            [torch.tensor(anal_lon, dtype=torch.float),
             torch.tensor(anal_lat, dtype=torch.float)],
            dim=1,
        )

        # Optional: k-NN neighbor graph among state target nodes themselves (not
        # mesh <-> target), used only by the auxiliary --gradient_loss_weight term
        # to score value differences across neighboring grid cells. Gated behind
        # the flag so the KDTree build cost is only paid when the loss is
        # enabled.
        if float(getattr(self.args, "gradient_loss_weight", 0.0) or 0.0) > 0:
            grad_ei = self._build_grid_gradient_edges(anal_lon, anal_lat)
            data[node_type_target, "grad", node_type_target].edge_index = grad_ei
        return True

    def _build_grid_gradient_edges(
            self,
            lon_deg,
            lat_deg,
            k: int = 4) -> torch.Tensor:
        """
        k-NN neighbor graph over the state target grid's own nodes (lon/lat),
        used by the --gradient_loss_weight auxiliary loss. Cached per dataset
        instance (keyed by node count) since the regional grid is the same
        physical set of points for every sample.
        """
        n = len(lon_deg)
        cache = getattr(self, "_grad_edge_cache", None)
        if cache is not None and cache[0] == n:
            return cache[1]
        if n == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            xy = np.stack([np.asarray(lon_deg), np.asarray(lat_deg)], axis=1)
            tree = scipy.spatial.KDTree(xy)
            k_query = min(k + 1, n)
            _, neighbor_idx = tree.query(xy, k=k_query)
            if k_query == 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                neighbor_idx = np.atleast_2d(neighbor_idx)
                src = np.repeat(np.arange(n), k_query - 1)
                dst = neighbor_idx[:, 1:].reshape(-1)
                edge_index = torch.tensor(
                    np.stack([src, dst]), dtype=torch.long)
        self._grad_edge_cache = (n, edge_index)
        return edge_index

    def _create_empty_state_nodes(
            self,
            data: HeteroData,
            ges_cfg: Dict) -> None:
        """Placeholder empty nodes for state when ges/anal data is unavailable."""
        node_type_input = f"state_{_STATE_GES}_input"
        node_type_target = target_node_type("state", _STATE_GES, 0, 1)

        # ges and anal share the same field dimensions
        ges_dim = ges_cfg.get("target_dim", 0)
        anal_dim = ges_dim
        anal_num_meta = len(ges_cfg.get("metadata", []))
        # input: features + valid_mask + aux + static context  (mirrors
        # build_conventional_input_x + static_ctx)
        input_dim = ges_dim + ges_dim + FEATURES_AUX_DIM + STATIC_CONTEXT_DIM
        target_context_dim = anal_num_meta + FEATURES_AUX_DIM + STATIC_CONTEXT_DIM

        data[node_type_input].x = torch.empty(
            (0, input_dim), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty(
            (2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty(
            (0, 4), dtype=torch.float32)
        data[node_type_target].y = torch.empty(
            (0, anal_dim), dtype=torch.float32)    # anal/increment placeholder
        if not self.target_is_anal:
            data[node_type_target].ges = torch.empty(
                (0, ges_dim), dtype=torch.float32)   # background placeholder
        if self.normalize_increment:
            data[node_type_target].increment_mean = self.increment_mean
            data[node_type_target].increment_std = self.increment_std
        data[node_type_target].valid_mask = torch.empty(
            (0, anal_dim), dtype=torch.bool)
        data[node_type_target].num_nodes = 0
        data[node_type_target].x = torch.empty(
            (0, target_context_dim), dtype=torch.float32)
        data["mesh", "to", node_type_target].edge_index = torch.empty(
            (2, 0), dtype=torch.long)
        data["mesh", "to", node_type_target].edge_attr = torch.empty(
            (0, 4), dtype=torch.float32)
        data[node_type_target].pos = torch.empty((0, 2), dtype=torch.float32)
        if float(getattr(self.args, "gradient_loss_weight", 0.0) or 0.0) > 0:
            data[node_type_target, "grad", node_type_target].edge_index = torch.empty(
                (2, 0), dtype=torch.long)


def make_graph_dataset(
    data_path: str,
    start_date: str,
    end_date: str,
    observation_config: Dict,
    mesh_structure: Dict,
    args=None,
) -> GraphDataset:
    """Factory: returns a GraphDataset subclass selected by ``args.exp_type``.

    Supported values:
    * ``"regional"``           – RegionalGraphDataset (target at t+1 … t+k)
    * ``"regional_da"`` – DARegionalGraphDataset (target at same t, with state ges/anal)
    * ``"global"``      – GlobalGraphDataset
    """
    _MAP = {
        "regional": RegionalGraphDataset,
        "regional_rrfs": RegionalGraphDataset,
        "regional_da": DARegionalGraphDataset,
        "global": GlobalGraphDataset,
    }
    cls = _MAP.get(args.exp_type)
    if cls is None:
        raise ValueError(
            f"Unknown exp_type '{args.exp_type}'. Choose from: {list(_MAP)}")
    return cls(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        observation_config=observation_config,
        mesh_structure=mesh_structure,
        args=args,
    )
