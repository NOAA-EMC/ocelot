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
        self.binned_samples = generate_binned_timestamp_list(start_mmdd, end_mmdd, year, delta_time)

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
            logger.warning(f"Skipping index {idx}: need {k} target bins after input.")
            return None

        input_bin_name = self.binned_samples[idx]
        target_bin_names = [self.binned_samples[idx + 1 + h] for h in range(k)]
        logger.debug(f'Input bin: {input_bin_name}, Target bins ({k}): {target_bin_names}')
        input_bin_data = self.loader.get_data_for_bin(input_bin_name)
        target_bin_datas = [self.loader.get_data_for_bin(bn) for bn in target_bin_names]

        logger.debug(f"For idx {idx}, input_bin_data is {'None' if input_bin_data is None else 'Not None'}")
        for h, (bn, tb) in enumerate(zip(target_bin_names, target_bin_datas)):
            logger.debug(
                f"For idx {idx}, target_bin_data[{h}] ({bn}) is {'None' if tb is None else 'Not None'}"
            )

        if input_bin_data is None or any(tb is None for tb in target_bin_datas):
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

                    input_features_norm = input_inst_data.get("features_norm", None)
                    inst_nonempty = (
                        input_features_norm is not None
                        and (not torch.is_tensor(input_features_norm) or input_features_norm.numel() > 0)
                        and len(input_inst_data.get("lat_deg", [])) > 0
                        and len(input_inst_data.get("lon_deg", [])) > 0
                    )
                    for target_inst_data in target_inst_list:
                        target_features_norm = target_inst_data.get("features_norm", None)
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
                        self._create_empty_nodes(data, obs_type, inst_name, inst_cfg)
                        continue

                    for key, tensor in input_inst_data.items():
                        if torch.is_tensor(tensor) and torch.isnan(tensor).any():
                            logger.warning(f"Found NaN in input tensor '{key}' for {obs_type}_{inst_name}.")

                    for h, target_inst_data in enumerate(target_inst_list):
                        for key, tensor in target_inst_data.items():
                            if torch.is_tensor(tensor) and torch.isnan(tensor).any():
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
                        metadata_valid_mask = input_inst_data.get("metadata_valid_mask", None)

                        rows_to_keep = torch.all(features_valid_mask, dim=1)
                        if metadata_valid_mask is not None:
                            rows_to_keep = rows_to_keep & torch.all(metadata_valid_mask, dim=1)

                        logger.debug(f"  rows_to_keep: {rows_to_keep.shape}, sum={rows_to_keep.sum()}")

                        input_features = input_features[rows_to_keep]
                        input_aux = input_aux[rows_to_keep]
                        input_meta = input_inst_data["features_meta"].float()[rows_to_keep]

                        rows_to_keep_np = rows_to_keep.cpu().numpy()
                        input_lat_deg_filtered = input_inst_data["lat_deg"][rows_to_keep_np]
                        input_lon_deg_filtered = input_inst_data["lon_deg"][rows_to_keep_np]

                        data[node_type_input].x = torch.cat([input_features, input_meta, input_aux], dim=1)
                    else:
                        input_lat_deg_filtered = input_inst_data["lat_deg"]
                        input_lon_deg_filtered = input_inst_data["lon_deg"]
                        input_valid_mask = input_inst_data["features_valid_mask"].float()
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
                        node_type_target = target_node_type(obs_type, inst_name, h, k_bins)
                        self._populate_target_nodes(
                            data, obs_type, inst_name, target_inst_data, node_type_target
                        )

                    # --- 3. Build encoder edges: obs_input → mesh ---
                    edge_index_encoder, edge_attr_encoder = self._obs_to_mesh_edges(
                        input_lat_deg_filtered, input_lon_deg_filtered, o2m=True
                    )

                    if edge_index_encoder.numel() == 0:
                        inst_cfg = self.observation_config[obs_type][inst_name]
                        self._create_empty_nodes(data, obs_type, inst_name, inst_cfg)
                        all_inst_names_exist_and_nonempty = False
                        missing_or_empty_inst_names.append(inst_name)
                        continue

                    data[node_type_input, "to", "mesh"].edge_index = edge_index_encoder
                    data[node_type_input, "to", "mesh"].edge_attr = edge_attr_encoder

                    # --- 4. Build decoder + encode-feedback edges (k horizons) ---
                    for h, target_inst_data in enumerate(target_inst_list):
                        node_type_target = target_node_type(obs_type, inst_name, h, k_bins)
                        (
                            target_lat_deg_filtered,
                            target_lon_deg_filtered,
                        ) = self._target_lat_lon_for_edges(obs_type, target_inst_data)

                        edge_index_knn, edge_attr_knn = self._obs_to_mesh_edges(
                            target_lat_deg_filtered, target_lon_deg_filtered, o2m=False
                        )

                        data["mesh", "to", node_type_target].edge_index = edge_index_knn
                        data["mesh", "to", node_type_target].edge_attr = edge_attr_knn

                        lat = torch.tensor(target_lat_deg_filtered, dtype=torch.float)
                        lon = torch.tensor(target_lon_deg_filtered, dtype=torch.float)
                        data[node_type_target].pos = torch.stack([lon, lat], dim=1)

                        if k_bins > 1:
                            enc_ei, enc_ea = self._obs_to_mesh_edges(
                                target_lat_deg_filtered, target_lon_deg_filtered, o2m=True
                            )
                            if enc_ei.numel() == 0:
                                raise ValueError(
                                    "No encode_obs_to_mesh edges at target locations; "
                                    "try increasing cutoff_factor."
                                )
                            et = (node_type_target, ENCODE_OBS_TO_MESH_REL, "mesh")
                            data[et].edge_index = enc_ei
                            data[et].edge_attr = enc_ea

                else:
                    all_inst_names_exist_and_nonempty = False
                    missing_or_empty_inst_names.append(inst_name)
                    logger.debug(f'Creating empty nodes for missing instrument: {obs_type}_{inst_name}')
                    inst_cfg = self.observation_config[obs_type][inst_name]
                    self._create_empty_nodes(data, obs_type, inst_name, inst_cfg)

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
                f"(missing/empty: {missing_or_empty_summary})"
            )
        else:
            logger.info(f'Finished processing bin: {input_bin_name} -> {target_bin_names}')
        return data

    @staticmethod
    def _target_lat_lon_for_edges(obs_type: str, target_inst_data: Dict) -> tuple:
        """Lat/lon arrays aligned with rows used for mesh→target edges (after satellite row filter)."""
        if obs_type == "satellite":
            target_features_valid_mask = target_inst_data["features_valid_mask"].bool()
            target_metadata_valid_mask = target_inst_data.get("metadata_valid_mask", None)
            target_rows_to_keep = torch.all(target_features_valid_mask, dim=1)
            if target_metadata_valid_mask is not None:
                target_rows_to_keep = target_rows_to_keep & torch.all(target_metadata_valid_mask, dim=1)
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
        target_features_valid_mask = target_inst_data["features_valid_mask"].bool()

        if obs_type == "satellite":
            target_metadata_valid_mask = target_inst_data.get("metadata_valid_mask", None)
            target_rows_to_keep = torch.all(target_features_valid_mask, dim=1)
            if target_metadata_valid_mask is not None:
                target_rows_to_keep = target_rows_to_keep & torch.all(target_metadata_valid_mask, dim=1)

            target_features = target_features[target_rows_to_keep]
            target_features_valid_mask = target_features_valid_mask[target_rows_to_keep]

            target_meta = target_inst_data["features_meta"].float()[target_rows_to_keep]
            target_aux = target_inst_data["features_aux"].float()[target_rows_to_keep]
            data[node_type_target].x = torch.cat([target_meta, target_aux], dim=1)
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

    def _create_empty_nodes(self, data: HeteroData, obs_type: str, inst_name: str, inst_cfg: Dict):
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

        data[node_type_input].x = torch.empty((0, input_dim), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 4), dtype=torch.float32)

        target_context_dim = num_metadata + features_aux_dim
        for h in range(k):
            node_type_target = target_node_type(obs_type, inst_name, h, k)
            data[node_type_target].y = torch.empty((0, target_dim), dtype=torch.float32)
            data[node_type_target].valid_mask = torch.empty((0, target_dim), dtype=torch.bool)
            data[node_type_target].num_nodes = 0
            data[node_type_target].x = torch.empty((0, target_context_dim), dtype=torch.float32)
            data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 4), dtype=torch.float32)
            data[node_type_target].pos = torch.empty((0, 2), dtype=torch.float32)
            if k > 1:
                et = (node_type_target, ENCODE_OBS_TO_MESH_REL, "mesh")
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[et].edge_attr = torch.empty((0, 4), dtype=torch.float32)


class RegionalGraphDataset(GraphDataset):
    """GraphDataset for a regional (limited-area) mesh domain."""

    def _obs_to_mesh_edges(self, lat_deg, lon_deg, o2m: bool = True) -> Tuple:
        if len(lat_deg) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)
        G_bottom = self.mesh_structure["G_bottom_mesh"]
        xy = project_coords_regional(lat_deg, lon_deg)
        if o2m:
            conn = 'g2m'
        else:
            conn = 'm2g'
        obs2grid_diatom = create_obs_conn_mesh(xy, G_bottom, self.args, conn=conn)
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
    * ``"global"``      – GlobalGraphDataset
    """
    _MAP = {
        "regional": RegionalGraphDataset,
        "regional_rrfs": RegionalGraphDataset,
    }
    cls = _MAP.get(args.exp_type)
    if cls is None:
        raise ValueError(f"Unknown exp_type '{args.exp_type}'. Choose from: {list(_MAP)}")
    return cls(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        observation_config=observation_config,
        mesh_structure=mesh_structure,
        args=args,
    )
