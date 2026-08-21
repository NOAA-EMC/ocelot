"""Core Lightning model for Ocelot graph neural network training and inference.

This module defines the end-to-end GNN model, including observation encoders,
latent mesh processors, target decoders, rollout logic, losses, and diagnostic
output utilities used during training, validation, and prediction.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from logger import log

from ocelot.model.coder.attn_bipartite import BipartiteGAT
from ocelot.model.coder.interaction_net import InteractionNet
from ocelot.model.processor.processor_factory import ProcessorFactory
from ocelot.configs.model_config import ModelConfig
from ocelot.configs.observation_config import ObservationConfig
from ocelot.utils import make_mlp
from ocelot.process_timeseries import _encode_target_time_features
from ocelot.model.mesh.mesh_factory import MeshFactory


def _build_instrument_map(observation_config: ObservationConfig) -> dict[str, int]:
    obs = observation_config.observation_config

    # QUESTION: Does the ordering of the instruments matter? Using dictionary order isn't going to work...
    order = []
    order += sorted(obs['satellite'].keys())
    order += sorted(obs['conventional'].keys())
    return {name: i for i, name in enumerate(order)}


def _canonical_variable_name(feature_name: str) -> str:
    """Map raw feature names to canonical variable names used by FSOI filters."""
    if not feature_name:
        return ""

    key = feature_name.strip().lower().replace("-", "_")

    mapping = {
        # Temperatures
        "airtemperature": "temperature",
        "temperature": "temperature",
        "dewpointtemperature": "dewpoint_temperature",
        "dew_point_temperature": "dewpoint_temperature",

        # Winds
        "wind_u": "u_wind",
        "windu": "u_wind",
        "wind_v": "v_wind",
        "windv": "v_wind",

        # Humidity
        "specifichumidity": "specific_humidity",
        "specific_humidity": "specific_humidity",

        # Pressure
        "airpressure": "pressure",
        "airpressure_prepbufr_event_1": "pressure",
        "pressuremeansealevel_pb": "pressure",
        "pressuremeansealevel_prepbufr": "pressure",
    }

    return mapping.get(key, feature_name)


class Ocelot(nn.Module):
    """
    A Graph Neural Network (GNN) model for processing structured spatiotemporal data.
    Key Features:
    - Encoder and decoder use distance information (as edge attributes).
    - Decoder output is aggregated using inverse-distance weighted averaging.
    - Includes LayerNorm and Dropout in both encoder and decoder for regularization.

    Methods:
        forward(data):
            Runs the forward pass, including encoding, message passing, decoding, and
            weighted aggregation to produce target predictions.
    """

    def __init__(
        self,
        model_config : ModelConfig,
        observation_config: ObservationConfig,
        verbose=False,
    ):
        """
        Initializes the Ocelot GNN model with an encoder, processor, and decoder.

        Parameters:
        model_config (ModelConfig): Configuration object for the model architecture and hyperparameters.
        observation_config (ObservationConfig): Configuration object for observation features.
        verbose (bool, optional): If True, enables verbose logging (default: False).
        """
        super().__init__()

        self.verbose = verbose

        hidden_dim = model_config.hidden_dim
        mesh_arch_config = model_config.mesh
        encoder_config = model_config.encoder
        processor_config = model_config.processor
        decoder_config = model_config.decoder
        embeddings_config = model_config.embeddings

        mesh_type = mesh_arch_config.type
        mesh_levels = mesh_arch_config.levels
        mesh_resolution = int(mesh_arch_config.splits if hasattr(mesh_arch_config, 'splits') else mesh_arch_config.resolution)

        # Normalize to int so Lightning hparams merge is stable across module/datamodule.
        self.observation_config = observation_config
        self.feature_stats = self.observation_config.feature_stats
        self.instrument_weights = self.observation_config.instrument_weights
        self.channel_weights = self.observation_config.channel_weights
        self.latent_step_hours = model_config.latent_step_hours
        self.scan_angle_conditioning = embeddings_config.scan_angle_conditioning
        self.pressure_level_conditioning = embeddings_config.pressure_level_conditioning

        edge_dims = [
            config.edge_dim
            for config in (encoder_config, decoder_config)
            if config.type == 'gat' and config.edge_dim is not None
        ]
        
        self.use_bipartite_edge_attr = bool(edge_dims)
        self.bipartite_edge_attr_dim = int(edge_dims[0]) if edge_dims else 4

        # Load mesh-grid variable config
        self.obs_mesh_config = self.observation_config.mesh_config
        self.enable_mesh_pred = self.obs_mesh_config.enable_mesh_pred
        self.mesh_instruments = list(self.obs_mesh_config.variables.keys())
        self.mesh_pressure_level_idx = self.obs_mesh_config.mesh_pressure_level_idx
        
        if self.verbose:
            print(f"[DEBUG CONFIG] enable_mesh_pred: {self.enable_mesh_pred}")
            print(f"[DEBUG CONFIG] mesh_config: {self.obs_mesh_config}")
            # print(f"[DEBUG CONFIG] variables in config: {self.obs_mesh_config.variables}")
            print(f"[DEBUG CONFIG] Instruments for mesh prediction: {self.mesh_instruments}")
            print(f"[DEBUG CONFIG] mesh_pressure_level_index: {self.mesh_pressure_level_idx}")

        # Mirror process_timeseries._name2id()
        self.instrument_name_to_id = _build_instrument_map(self.observation_config)
        self.instrument_id_to_name = {v: k for k, v in self.instrument_name_to_id.items()}

        # Channel metadata used by FSOI variable filtering.
        # Format: {instrument_name: [ {"channel": int, "feature": str, "variable_name": str}, ... ]}
        self.instrument_channels: Dict[str, List[Dict]] = {}
        for _, instruments in (self.observation_config.observation_config or {}).items():
            for inst_name, cfg in (instruments or {}).items():
                features = cfg.get("features", []) or []
                ch_info = []
                for ch_idx, feat in enumerate(features):
                    canonical = _canonical_variable_name(str(feat))
                    ch_info.append(
                        {
                            "channel": ch_idx,
                            "feature": str(feat),
                            "variable": str(feat),
                            "variable_name": canonical,
                        }
                    )
                self.instrument_channels[inst_name] = ch_info

        # Normalize user-provided weights (accept names or ids)
        self.instrument_weights = self._normalize_inst_weights(self.instrument_weights)
        self.channel_weights = self._normalize_channel_weights(self.channel_weights)

        # Boolean masks per instrument for valid channels (weights > 0)
        self.channel_masks = {inst_id: (w > 0) for inst_id, w in self.channel_weights.items()}

        if self.verbose:
            print("[MODEL] instrument map:", self.instrument_name_to_id)
            print("[MODEL] instrument_weights:", {self.instrument_id_to_name[k]: float(v) for k, v in self.instrument_weights.items()})

        self.hidden_dim = hidden_dim
        self.mesh_type = mesh_type
        self.mesh_levels = mesh_levels

        # bipartite GATs consume the computed spatial edge_attr
        # directly, with edge_dim=bipartite_edge_attr_dim (GraphCast-style features are 4-dim).
        if self.bipartite_edge_attr_dim <= 0:
            raise ValueError(
                f"bipartite_edge_attr_dim must be > 0 (got: {self.bipartite_edge_attr_dim})"
            )
        print(f"\n{'='*70}")
        print(f"[GNN MODEL] Initializing with configuration:")
        print(f"  - Mesh type: {mesh_type}")
        print(f"  - Mesh levels: {mesh_levels}")
        print(f"  - Mesh resolution (splits): {mesh_resolution}")
        print(f"  - Processor type: {processor_config.type}")
        print(f"  - Encoder type: {encoder_config.type}")
        print(f"  - Decoder type: {decoder_config.type}")
        print(f"{'='*70}\n")

        self.mesh_resolution = mesh_resolution
        self.mesh = MeshFactory.build(mesh_type, mesh_levels, mesh_resolution)

        self.is_hierarchical = (mesh_type == "hierarchical")  # TODO: Delete this once hierarchical-specific logic is fully integrated

        # # --- Initialize Network Dictionaries ---
        self.observation_embedders = nn.ModuleDict()  # For initial feature projection
        self.observation_encoders = nn.ModuleDict()  # For obs -> mesh GNNs
        self.observation_decoders = nn.ModuleDict()
        self.output_mappers = nn.ModuleDict()  # For final prediction MLPs

        first_instrument_config = next(iter(next(iter(self.observation_config.observation_config.values())).values()))
        hidden_layers = first_instrument_config.get("encoder_hidden_layers", 2)

        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        
        # Get mesh feature dimension from the first mesh
        mesh_feature_dim = self.mesh.mesh_features_torch[0].shape[1]

        self.mesh_embedder = make_mlp([mesh_feature_dim] + self.mlp_blueprint_end)

        # Create scan-angle embedders once to avoid loop-order surprises
        # These embeddings are used ONLY for decoder initialization
        self.scan_angle_embed_dim = int(embeddings_config.scan_angle_dim)
        self.scan_angle_embedder = make_mlp([1, self.scan_angle_embed_dim])
        self.ascat_scan_angle_embedder = make_mlp([3, self.scan_angle_embed_dim])

        # Optional: project scan-angle embedding across the full hidden_dim so it can't be confined
        # to a small trailing slice of the receiver representation.d
        if self.scan_angle_conditioning == "project":
            self.scan_angle_projector = nn.Linear(self.scan_angle_embed_dim, self.hidden_dim)
        else:
            self.scan_angle_projector = None

        # Create pressure-level embedding for radiosonde and aircraft (16 standard levels)
        self.pressure_level_embed_dim = int(embeddings_config.pressure_level_dim)
        self.pressure_level_embedder = nn.Embedding(
            num_embeddings=int(embeddings_config.num_pressure_levels),
            embedding_dim=self.pressure_level_embed_dim
        )

        # Optional: project pressure-level embedding across the full hidden_dim.
        if self.pressure_level_conditioning == "project":
            self.pressure_level_projector = nn.Linear(self.pressure_level_embed_dim, self.hidden_dim)
        else:
            self.pressure_level_projector = None

        # Target valid-time + local solar time conditioning lives in the last 5 target_metadata columns.
        self.target_time_feature_dim = 5
        self.target_time_embed_dim = embeddings_config.target_time_dim
        self.target_time_embedder = make_mlp([self.target_time_feature_dim, self.target_time_embed_dim])
        self.target_time_projector = nn.Linear(self.target_time_embed_dim, self.hidden_dim)

        node_types = ["mesh"]
        edge_types = [("mesh", "to", "mesh")]


        for obs_type, instruments in self.observation_config.observation_config.items():
            for inst_name, cfg in instruments.items():
                node_type_input = f"{inst_name}_input"
                node_type_target = f"{inst_name}_target"

                node_types.extend([node_type_input, node_type_target])
                edge_types.extend([(node_type_input, "to", "mesh"), ("mesh", "to", node_type_target)])

                input_dim = cfg.get("input_dim")
                target_dim = cfg.get("target_dim")

                # Encoder GNN (obs -> mesh)
                edge_type_tuple_enc = (node_type_input, "to", "mesh")
                enc_key = self.mesh.edge_key(edge_type_tuple_enc)

                if encoder_config.type == "gat":
                    self.observation_encoders[enc_key] = BipartiteGAT(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=encoder_config.layers,
                        heads=encoder_config.heads,
                        dropout=encoder_config.dropout,
                        edge_dim=getattr(encoder_config, 'edge_dim', None),
                        dst_chunk_size=getattr(encoder_config, 'dst_chunk_size', None),
                        dst_chunk_threshold=encoder_config.dst_chunk_threshold,
                        use_activation_checkpointing=encoder_config.use_activation_checkpointing,
                    )
                else:
                    self.observation_encoders[enc_key] = InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=encoder_config.hidden_layers,
                        update_edges=encoder_config.update_edges,
                        edge_chunk_sizes=getattr(encoder_config, 'edge_chunk_sizes', None),
                        aggr_chunk_sizes=getattr(encoder_config, 'aggr_chunk_sizes', None),
                        aggr=encoder_config.aggr,
                    )
                # Decoder GNN (mesh -> target)
                edge_type_tuple_dec = ("mesh", "to", node_type_target)
                dec_key = self.mesh.edge_key(edge_type_tuple_dec)

                if decoder_config.type == "gat":
                    self.observation_decoders[dec_key] = BipartiteGAT(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=decoder_config.layers,
                        heads=decoder_config.heads,
                        dropout=decoder_config.dropout,
                        edge_dim=getattr(decoder_config, 'edge_dim', None),
                        dst_chunk_size=getattr(decoder_config, 'dst_chunk_size', None),
                        dst_chunk_threshold=decoder_config.dst_chunk_threshold,
                        use_activation_checkpointing=decoder_config.use_activation_checkpointing,
                    )
                else:
                    self.observation_decoders[dec_key] = InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=decoder_config.hidden_layers,
                        update_edges=decoder_config.update_edges,
                        edge_chunk_sizes=getattr(decoder_config, 'edge_chunk_sizes', None),
                        aggr_chunk_sizes=getattr(decoder_config, 'aggr_chunk_sizes', None),
                        aggr=decoder_config.aggr,
                    )

                # Initial MLP to project raw features to hidden_dim
                # Add pressure-level embedding dimensions for radiosonde and aircraft input
                embedder_input_dim = input_dim
                if inst_name in ["radiosonde", "aircraft"]:
                    embedder_input_dim += self.pressure_level_embed_dim
                self.observation_embedders[node_type_input] = make_mlp([embedder_input_dim] + self.mlp_blueprint_end)

                # Output mapper takes ONLY decoded features (hidden_dim)
                # Geometry conditioning happens at decoder initialization, not in output mapper
                input_dim_for_mapper = hidden_dim

                output_map_layers = [input_dim_for_mapper] + [hidden_dim] * hidden_layers + [target_dim]
                self.output_mappers[node_type_target] = make_mlp(output_map_layers, layer_norm=False)
                # Geometry dependence is enforced solely through decoder conditioning

        self.processor = ProcessorFactory.build(self.mesh, 
                                                hidden_dim=self.hidden_dim, 
                                                processor_config=processor_config)



    def _safe_trainer(self):
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _is_global_zero_safe(self) -> bool:
        trainer = self._safe_trainer()
        return getattr(trainer, "is_global_zero", True)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # PyG Data/HeteroData implements .to()
        if hasattr(batch, "to"):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _coerce_edge_attr_dim(self, edge_attr: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.size(-1) == dim:
            return edge_attr
        if edge_attr.size(-1) < dim:
            pad = dim - edge_attr.size(-1)
            return torch.cat(
                [
                    edge_attr,
                    torch.zeros(
                        edge_attr.size(0),
                        pad,
                        device=edge_attr.device,
                        dtype=edge_attr.dtype,
                    ),
                ],
                dim=-1,
            )
        return edge_attr[:, :dim]

    def _edge_features(
        self,
        data: HeteroData,
        edge_type,
        edge_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns per-edge features in bipartite_edge_attr_dim (raw spatial edge_attr)."""
        E = int(edge_index.size(1))

        # Debug printing: show whether we used real edge_attr or fell back to zeros.
        # Gated by verbose + global_zero and printed at most once per (edge_type, reason).
        def _maybe_print(reason: str, edge_rep_tensor: torch.Tensor | None = None) -> None:
            if not getattr(self, "verbose", False):
                return
            if not self._is_global_zero_safe():
                return
            if not hasattr(self, "_edge_attr_debug_seen") or self._edge_attr_debug_seen is None:
                self._edge_attr_debug_seen = set()
            key = (tuple(edge_type) if isinstance(edge_type, (list, tuple)) else str(edge_type), str(reason))
            if key in self._edge_attr_debug_seen:
                return
            self._edge_attr_debug_seen.add(key)

            msg = f"[EDGE_ATTR] edge_type={edge_type} E={E} used={'edge_attr' if reason == 'ok' else 'zeros'} reason={reason}"
            if edge_rep_tensor is not None and torch.is_tensor(edge_rep_tensor) and edge_rep_tensor.numel() > 0:
                try:
                    t = edge_rep_tensor.detach()
                    mean_v = t.mean().item()
                    std_v = t.std(unbiased=False).item()
                    min_v = t.min().item()
                    max_v = t.max().item()
                    msg += (
                        f" edge_attr_shape={tuple(t.shape)} "
                        f"mean={mean_v:.4g} std={std_v:.4g} min={min_v:.4g} max={max_v:.4g}"
                    )
                except Exception:
                    msg += f" edge_attr_shape={tuple(edge_rep_tensor.shape)}"
            print(msg)

        if not self.use_bipartite_edge_attr:
            _maybe_print("disabled")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        edge_rep = None
        try:
            if "edge_attr" in data[edge_type]:
                edge_rep = data[edge_type].edge_attr
        except Exception:
            edge_rep = None

        if edge_rep is None:
            _maybe_print("missing")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if torch.is_tensor(edge_rep) and edge_rep.numel() == 0:
            _maybe_print("empty")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if not torch.is_tensor(edge_rep):
            _maybe_print("non_tensor")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if edge_rep.size(0) != E:
            _maybe_print(f"edge_count_mismatch(edge_attr={int(edge_rep.size(0))})")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        edge_rep = edge_rep.to(device=device, dtype=dtype)
        edge_rep = self._coerce_edge_attr_dim(edge_rep, self.bipartite_edge_attr_dim)

        if edge_rep.size(-1) != self.bipartite_edge_attr_dim:
            _maybe_print(f"dim_mismatch(edge_attr={int(edge_rep.size(-1))})", edge_rep)
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        _maybe_print("ok", edge_rep)
        return edge_rep

    def _load_mesh_prediction_edges(self, edges_file='mesh_pred_edges.npz'):
        """
        Load pre-computed mesh prediction edges from file.

        This avoids calling obs_mesh_conn at runtime, which causes rtree
        multiprocessing issues when workers fork.

        Args:
            edges_file: Path to .npz file with pre-computed edges
        """
        import numpy as np

        # Many SLURM jobs run from a different working directory than this module.
        # If a relative path was provided, also try resolving it next to this file.
        edges_path = edges_file
        if not os.path.isabs(edges_path):
            here = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(here, edges_path)
            if os.path.exists(candidate):
                edges_path = candidate

        print(f"[MESH PRED] Loading pre-computed edges from {edges_path}...")

        if not os.path.exists(edges_path):
            raise FileNotFoundError(
                "Mesh prediction edges file not found.\n"
                f"  Requested: {edges_file}\n"
                f"  Tried:     {edges_path}\n"
                "Please run: python precompute_mesh_edges.py --config configs/mesh_config.yaml"
            )

        # Load pre-computed data
        data = np.load(edges_path)

        # Get coordinates (same for all instruments)
        mesh_lats = data['lats']
        mesh_lons = data['lons']
        num_nodes = int(data['num_nodes'])

        print(f"[MESH PRED] Loaded grid:")
        print(f"  Lat range: [{mesh_lats.min():.2f}, {mesh_lats.max():.2f}]")
        print(f"  Lon range: [{mesh_lons.min():.2f}, {mesh_lons.max():.2f}]")
        print(f"  Grid points: {num_nodes}")

        # Build edges dict for each mesh instrument
        mesh_pred_edges = {}

        for inst_name in self.mesh_instruments:
            if not self._is_mesh_pred_variable(inst_name):
                continue

            edge_index_key = f'{inst_name}_edge_index'
            if edge_index_key not in data:
                print(f"[MESH PRED] WARNING: No edges found for {inst_name}, skipping...")
                continue

            # Load edge_index (keep on CPU for now)
            edge_index = torch.from_numpy(data[edge_index_key]).long()

            print(f"[MESH PRED] {inst_name}: {edge_index.shape[1]} edges")

            edge_attr_key = f'{inst_name}_edge_attr'
            if edge_attr_key in data:
                edge_attr = torch.from_numpy(data[edge_attr_key]).float()
                if edge_attr.shape[0] != edge_index.shape[1]:
                    raise ValueError(
                        f"[MESH PRED] {inst_name}: edge_attr row count ({edge_attr.shape[0]}) "
                        f"!= edge_index edge count ({edge_index.shape[1]})"
                    )
                if edge_attr.shape[1] != self.bipartite_edge_attr_dim:
                    raise ValueError(
                        f"[MESH PRED] {inst_name}: edge_attr dim ({edge_attr.shape[1]}) "
                        f"!= bipartite_edge_attr_dim ({self.bipartite_edge_attr_dim})"
                    )
            else:
                print(
                    f"[MESH PRED] WARNING: No edge_attr found for {inst_name} in {edges_path}. "
                    f"Falling back to zeros — re-run precompute_mesh_edges.py to fix this."
                )
                edge_attr = torch.zeros((edge_index.size(1), self.bipartite_edge_attr_dim))

            # Store on CPU - will move to device when used
            mesh_pred_edges[inst_name] = {
                'edge_index': edge_index,  # CPU tensor (no .to(device))
                'edge_attr': edge_attr,
                'lats': torch.from_numpy(mesh_lats).float(),  # CPU
                'lons': torch.from_numpy(mesh_lons).float(),  # CPU
                'num_nodes': num_nodes
            }

        if not mesh_pred_edges:
            raise ValueError(
                f"No valid edges loaded for mesh instruments: {self.mesh_instruments}\n"
                f"Available in file: {[k for k in data.files if k.endswith('_edge_index')]}"
            )

        print(f"[MESH PRED] Loaded edges for {list(mesh_pred_edges.keys())}")
        return mesh_pred_edges

    def _get_mesh_pred_edges(self):
        """Load and cache mesh prediction edges once for the entire run."""
        if not hasattr(self, '_cached_mesh_pred_edges') or self._cached_mesh_pred_edges is None:
            self._cached_mesh_pred_edges = self._load_mesh_prediction_edges()
        return self._cached_mesh_pred_edges

    def _is_mesh_pred_variable(self, inst_name: str) -> bool:
        """Check if instrument has mesh variables configured."""
        if not self.enable_mesh_pred:
            return False
        return inst_name in self.mesh_instruments

    def _normalize_inst_weights(self, weights_in):
        out = {}
        if not weights_in:
            return out
        for k, v in weights_in.items():
            if isinstance(k, str):
                if k in self.instrument_name_to_id:
                    out[self.instrument_name_to_id[k]] = float(v)
            else:
                out[int(k)] = float(v)
        return out

    def _normalize_channel_weights(self, ch_in):
        """
        Accepts {name_or_id: sequence/tensor} and returns {id: torch.tensor}
        sized to that instrument's target_dim (slice/pad with 1.0 as needed).
        """
        out = {}
        if not ch_in:
            return out
        for k, v in ch_in.items():
            # resolve id and name
            if isinstance(k, str):
                if k not in self.instrument_name_to_id:
                    continue
                inst_name, inst_id = k, self.instrument_name_to_id[k]
            else:
                inst_id = int(k)
                inst_name = getattr(self, "instrument_id_to_name", {}).get(inst_id, None)

            # find expected target_dim from config
            target_dim = None
            for group, instruments in self.observation_config.observation_config.items():
                if inst_name in instruments:
                    target_dim = instruments[inst_name]["target_dim"]
                    break
            if target_dim is None:
                continue

            w = torch.as_tensor(v, dtype=torch.float32)
            if w.numel() > target_dim:
                w = w[:target_dim]
            elif w.numel() < target_dim:
                w = torch.cat([w, torch.ones(target_dim - w.numel(), dtype=torch.float32)], dim=0)
            out[inst_id] = w
        return out

    def _feature_names_for_node(self, node_type: str):
        """Return ordered feature names for this target node."""
        # Latent mode: target_step0, target_step1, etc
        if "_target_step" in node_type:
            inst_name = node_type.split("_target_step")[0]
        else:
            inst_name = node_type.replace("_target", "")
        for obs_type, instruments in self.observation_config.observation_config.items():
            if inst_name in instruments:
                return instruments[inst_name].get("features", None)
        return None


    def unnormalize_standardscaler(self, tensor, node_type, mean=None, std=None):
        """
        Reverse a per-channel standardization: x = x * std + mean.

        - If `mean` and `std` are provided, they are used directly.
        - Otherwise we look up the instrument from `node_type` (expects "<instrument>_target"),
        get the feature order from `self.observation_config`, and pull means/stds
        from `self.feature_stats[instrument][feature] = [mean, std]`.

        Args:
            tensor:  (..., C) torch.Tensor — standardized values
            node_type: str — e.g., "atms_target", "amsua_target", "surface_obs_target", "snow_cover_target"
            mean, std: optional sequences/ndarrays/torch tensors of shape (C,)

        Returns:
            torch.Tensor with the same shape as `tensor`, un-normalized per channel.
        """
        # If explicit stats are provided, use them
        if mean is not None and std is not None:
            device = tensor.device if torch.is_tensor(tensor) else getattr(self, "device", "cpu")
            dtype = tensor.dtype if torch.is_tensor(tensor) else torch.float32
            mean = torch.as_tensor(mean, dtype=dtype, device=device)
            std = torch.as_tensor(std, dtype=dtype, device=device)
            return tensor * std + mean

        # Parse "<instrument>_target" (also tolerate "<instrument>_input" just in case)
        if not isinstance(node_type, str) or "_" not in node_type:
            raise ValueError(f"node_type must look like '<instrument>_target', got: {node_type!r}")
        inst_name = node_type.rsplit("_", 1)[0]  # drop trailing _target/_input/etc.

        # Find instrument block and feature order from the config
        feats = None
        found_in_obs_type = None
        for obs_type, instruments in self.observation_config.observation_config.items():
            if inst_name in instruments:
                feats = instruments[inst_name].get("features")
                found_in_obs_type = obs_type
                break
        if not feats:
            raise ValueError(f"Features for instrument '{inst_name}' not found in observation_config.")

        # Pull stats for this instrument
        if not hasattr(self, "feature_stats") or self.feature_stats is None:
            raise ValueError("self.feature_stats is not set; cannot unnormalize without stats.")

        if inst_name not in self.feature_stats:
            # Some configs store stats under category keys; try a second chance lookup
            cand = self.observation.feature_stats.get(found_in_obs_type, {})
            if inst_name in cand:
                stats_block = cand[inst_name]
            else:
                raise KeyError(f"feature_stats has no entry for instrument '{inst_name}'.")
        else:
            stats_block = self.feature_stats[inst_name]

        # Build mean/std vectors following the feature order exactly
        try:
            mean_vec = [stats_block[f][0] for f in feats]
            std_vec = [stats_block[f][1] for f in feats]
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Missing statistics for '{inst_name}.{missing}'. " f"Expected keys: {feats}. Have: {list(stats_block.keys())}") from e

        device = tensor.device if torch.is_tensor(tensor) else getattr(self, "device", "cpu")
        dtype = tensor.dtype if torch.is_tensor(tensor) else torch.float32
        mean_vec = torch.tensor(mean_vec, dtype=dtype, device=device)
        std_vec = torch.tensor(std_vec, dtype=dtype, device=device)

        # Basic shape check: last dim must match number of features
        if tensor.size(-1) != mean_vec.numel():
            raise ValueError(
                f"Channel mismatch for '{inst_name}': tensor last-dim={tensor.size(-1)} "
                f"but have {mean_vec.numel()} feature stats. Feature order={feats}"
            )

        return tensor * std_vec + mean_vec

    def forward(self, data: HeteroData, step_data_list=None):  # -> Dict[str, torch.Tensor]:

        num_graphs = data.num_graphs
        num_mesh_nodes = self.mesh.x.shape[0]

        # Inject and batch static mesh data
        # For hierarchical mode, we use the finest mesh level for encoding/decoding
        data["mesh"].x = self.mesh.x.repeat(num_graphs, 1)
        data["mesh", "to", "mesh"].edge_attr = self.mesh.mesh_edge_attr.repeat(num_graphs, 1)

        edge_indices = [self.mesh.mesh_edge_index + i * num_mesh_nodes for i in range(num_graphs)]
        data["mesh", "to", "mesh"].edge_index = torch.cat(edge_indices, dim=1)

        # --------------------------------------------------------------------
        # STAGE 1: EMBED (Initial feature projection for all input nodes)
        # --------------------------------------------------------------------
        embedded_features = {}
        # Embed static mesh features
        for node_type, x in data.x_dict.items():
            print(f"embed: [node_type] {node_type}: {x.shape}")
            if node_type == "mesh":
                embedded_features[node_type] = self.mesh_embedder(x)
            elif node_type.endswith("_input"):
                # Apply pressure-level embedding for radiosonde and aircraft if available
                needs_pressure_level = node_type in ["radiosonde_input", "aircraft_input"]
                if needs_pressure_level:
                    if "pressure_level" in data[node_type] and data[node_type].pressure_level.shape[0] > 0:
                        pressure_level_idx = data[node_type].pressure_level  # [N]
                        pressure_embed = self.pressure_level_embedder(pressure_level_idx)
                    else:
                        pressure_embed = torch.zeros(
                            x.shape[0],
                            self.pressure_level_embed_dim,
                            device=x.device,
                            dtype=x.dtype,
                        )

                    # Concatenate with original features
                    x_with_embed = torch.cat([x, pressure_embed], dim=-1)
                    print(
                        f"PRESSURE-LEVEL EMBEDDING APPLIED: {node_type} | "
                        f"orig={x.shape} + embed={pressure_embed.shape} → combined={x_with_embed.shape}"
                    )
                    embedded_features[node_type] = self.observation_embedders[node_type](x_with_embed)

                else:
                    embedded_features[node_type] = self.observation_embedders[node_type](x)

        # --------------------------------------------------------------------
        # STAGE 2: ENCODE (Pass information from observations TO the mesh)
        # --------------------------------------------------------------------
        encoded_mesh_features = embedded_features["mesh"]

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            print(f"encode: [edge_type] {edge_type}: {edge_index.shape}")
            if dst_type == "mesh" and src_type != "mesh":  # This is an obs -> mesh edge
                obs_features = embedded_features[src_type]
                # Use device from input data instead of self.device to avoid checkpoint loading issues
                device = obs_features.device if obs_features.numel() > 0 else encoded_mesh_features.device

                encoder = self.observation_encoders[self.mesh.edge_key(edge_type)]
                encoder.edge_index = edge_index

                edge_features = self._edge_features(
                    data=data,
                    edge_type=edge_type,
                    edge_index=edge_index,
                    device=device,
                    dtype=obs_features.dtype,
                )

                # --- Debugging ---
                log.debug(f"\n[ENC] edge type: {edge_type}")
                log.debug(f"  send_rep (obs) {obs_features.shape} | rec_rep (mesh) {encoded_mesh_features.shape}")
                log.debug(f"  edge_index {edge_index.shape}")
                # --- End Debugging ---

                encoded_mesh_features = encoder(
                    send_rep=obs_features,
                    rec_rep=encoded_mesh_features,
                    edge_rep=edge_features,
                )

        # --------------------------------------------------------------------
        # STAGE 3: PREPARE FOR PROCESSOR
        # --------------------------------------------------------------------
        encoded_features = embedded_features
        encoded_features["mesh"] = encoded_mesh_features

        # For hierarchical processor, we don't need to prepare node types
        # For standard processor, ensure all node types exist
        if not self.is_hierarchical and hasattr(self.processor, 'norms'):
            for node_type in self.processor.norms[0].keys():
                print(f"prep: [node_type] ", node_type)
                if node_type not in encoded_features:
                    if node_type in data.node_types:
                        num_nodes = data[node_type].num_nodes
                        # Use device from existing encoded features to avoid checkpoint loading issues
                        reference_device = encoded_mesh_features.device
                        encoded_features[node_type] = torch.zeros(num_nodes, self.hidden_dim, device=reference_device)

        # --------------------------------------------------------------------
        # STAGE 4: DETECT MODE AND PROCESS
        # --------------------------------------------------------------------
        predictions, mesh_features_per_step = self._forward_latent_rollout(data, encoded_features)

        if not self.training:
            return predictions, mesh_features_per_step

        return predictions

    def _forward_latent_rollout(self, data: HeteroData, encoded_features: dict) -> Dict[str, List[torch.Tensor]]:
        """
        Latent rollout forward pass: Sequential processor → decoder → next processor

        Architecture:
        Input [T-12 to T) → Encoder → mesh_state_T
             ↓
        Processor₁ → mesh_state₁ → Decoder₁ → Predictions [T to T+3)
             ↓
        Processor₂ → mesh_state₂ → Decoder₂ → Predictions [T+3 to T+6)
             ↓
        Processor₃ → mesh_state₃ → Decoder₃ → Predictions [T+6 to T+9)
             ↓
        Processor₄ → mesh_state₄ → Decoder₄ → Predictions [T+9 to T+12)
        """

        step_info = self._get_latent_step_info(data)
        step_mapping = step_info["step_mapping"]
        num_latent_steps = step_info["num_steps"]
        edge_mapping = self.mesh.map_step_edges(data, step_mapping)
        
        log.debug(f"[LATENT] {num_latent_steps} latent steps detected")
        log.debug(f"[LATENT] Step mapping: {step_mapping}")

        self.processor.reset()  # Ensure processor state is reset before rollout

        # Initialize predictions dict with lists for each base instrument
        predictions = {}
        for base_type in step_mapping.keys():
            predictions[base_type] = []

        for step in range(num_latent_steps):
            encoded_features['mesh'] = self.processor(step, step_info, encoded_features['mesh'])
            self._generate_predictions(data, step, step_info['step_mapping'], edge_mapping, encoded_features['mesh'], predictions)

        return predictions, None #, mesh_features_per_step
    
    def _generate_predictions(self, 
                              data: HeteroData, 
                              step: int, 
                              step_mapping: dict, 
                              edge_mapping: dict,
                              mesh_features_processed: torch.Tensor, 
                              predictions: dict) -> None:

        # Process all instruments for this step
        for base_type, steps_dict in step_mapping.items():
            if step in steps_dict:
                step_node_type = steps_dict[step]  # e.g., "atms_target_step0"

                # Find the corresponding edge
                step_edge_type = None
                step_edge_index = None
                for edge_type, edge_index in data.edge_index_dict.items():
                    src_type, _, dst_type = edge_type
                    if src_type == "mesh" and dst_type == step_node_type:
                        step_edge_type = edge_type
                        step_edge_index = edge_index
                        print(f"decode: [edge_type] {edge_type}: {edge_index.shape}")
                        break

                if step_edge_type is None or step_edge_index is None:
                    log.debug(f"[LATENT] Warning: No edge found for {step_node_type}")
                    continue

                # Get the decoder (mapped to base instrument)
                decoder_key = edge_mapping.get(step_edge_type)
                if decoder_key not in self.observation_decoders:
                    log.debug(f"[LATENT] Warning: No decoder found for {decoder_key}")
                    continue

                decoder = self.observation_decoders[decoder_key]
                decoder.edge_index = step_edge_index

                # Condition decoder on viewing geometry at initialization
                # - For satellites: viewing zenith angle (scan angle)
                # - For radiosonde/aircraft: pressure level (vertical viewing geometry)
                reference_device = mesh_features_processed.device
                N = data[step_node_type].num_nodes

                # Embed viewing geometry information FIRST (before decoder initialization)
                sa_emb = None
                pressure_emb = None
                time_emb = None
                if base_type == "ascat_target":
                    scan_angle = data[step_node_type].x  # [N,3] for ASCAT
                    sa_emb = self.ascat_scan_angle_embedder(scan_angle)  # [N, scan_embed_dim]
                elif base_type in ("atms_target", "amsua_target", "avhrr_target", "cris_pca_target", "seviri_asr_target", "seviri_csr_target"):
                    scan_angle = data[step_node_type].x  # [N,1] for ATMS/AMSU-A/AVHRR/CrIS-PCA
                    sa_emb = self.scan_angle_embedder(scan_angle)  # [N, scan_embed_dim]

                    # Diagnostic: verify scan angle varies
                    if base_type == "atms_target" and step % 200 == 0:
                        sa = data[step_node_type].x
                        if sa.numel() == 0:
                            print(f"[SCAN DIAG] scan_angle: shape={sa.shape} (empty)")
                        else:
                            sa_f = sa.float()
                            mean_v = sa_f.mean().item()
                            std_v = sa_f.std(unbiased=False).item()
                            min_v = sa_f.min().item()
                            max_v = sa_f.max().item()
                            print(
                                f"[SCAN DIAG] scan_angle: shape={sa.shape}, mean={mean_v:.4f}, "
                                f"std={std_v:.4f}, min={min_v:.4f}, max={max_v:.4f}"
                            )
                elif base_type in ["radiosonde_target", "aircraft_target"] and "pressure_level" in data[step_node_type]:
                    # For radiosonde and aircraft: condition on pressure level (vertical geometry)
                    pressure_level_idx = data[step_node_type].pressure_level  # [N]
                    pressure_emb = self.pressure_level_embedder(pressure_level_idx)  # [N, pressure_embed_dim=8]

                if hasattr(data[step_node_type], "target_metadata"):
                    target_metadata = data[step_node_type].target_metadata
                    if (
                        target_metadata is not None
                        and target_metadata.numel() > 0
                        and target_metadata.size(1) >= (2 + self.target_time_feature_dim)
                    ):
                        time_feat = target_metadata[:, -self.target_time_feature_dim:].to(reference_device)
                        time_emb = self.target_time_embedder(time_feat)

                # Decoder initialization: CONDITION on viewing geometry
                # Instead of zeros, initialize decoder WITH geometry information
                if sa_emb is not None:
                    # Satellite: condition decoder on scan angle (viewing zenith angle)
                    if self.scan_angle_projector is not None:
                        target_features_initial = self.scan_angle_projector(sa_emb)
                    else:
                        # Backward-compatible behavior: scan info only in the last dims.
                        padding_dim = self.hidden_dim - self.scan_angle_embed_dim
                        target_features_initial = torch.cat([
                            torch.zeros(N, padding_dim, device=reference_device),
                            sa_emb
                        ], dim=-1)  # [N, hidden_dim] with scan info in last 8 dims
                elif pressure_emb is not None:
                    # Radiosonde/Aircraft: condition decoder on pressure level (vertical viewing geometry)
                    # Make prediction explicitly depend on geometry
                    if self.pressure_level_projector is not None:
                        target_features_initial = self.pressure_level_projector(pressure_emb)
                    else:
                        padding_dim = self.hidden_dim - self.pressure_level_embed_dim
                        target_features_initial = torch.cat([
                            torch.zeros(N, padding_dim, device=reference_device),
                            pressure_emb
                        ], dim=-1)  # [N, hidden_dim] with pressure info in last 8 dims
                else:
                    # Conventional obs without viewing geometry: use zeros
                    target_features_initial = torch.zeros(N, self.hidden_dim, device=reference_device)

                # Add target-time conditioning as an additive bias over the full hidden_dim.
                if time_emb is not None:
                    target_features_initial = target_features_initial + self.target_time_projector(time_emb)

                edge_attr = self._edge_features(
                    data=data,
                    edge_type=step_edge_type,
                    edge_index=step_edge_index,
                    device=reference_device,
                    dtype=mesh_features_processed.dtype,
                )

                # Decoder now receives GEOMETRY-CONDITIONED initialization
                # This ensures the model CANNOT make predictions without knowing viewing geometry
                decoded_target_features = decoder(
                    send_rep=mesh_features_processed,
                    rec_rep=target_features_initial,  # NOW conditioned on viewing geometry!
                    edge_rep=edge_attr,
                )

                # Decoder output goes directly to output mapper
                # The model learns to use the geometry information that's embedded in target_features_initial

                # Diagnostic logging for radiosonde
                if base_type == "radiosonde_target" and pressure_emb is not None and step % 200 == 0:
                    print(f"[GRAPHDOP] Radiosonde: decoder conditioned on pressure (decoded shape={decoded_target_features.shape})")

                # Diagnostic logging for satellites
                if base_type == "atms_target" and sa_emb is not None and step % 200 == 0:
                    print(f"ATMS: decoder conditioned on scan angle (decoded shape={decoded_target_features.shape})")

                # Safety: verify mapper exists before using
                assert base_type in self.output_mappers, f"Missing output mapper for {base_type}"
                step_prediction = self.output_mappers[base_type](decoded_target_features)

                # Store prediction for this step
                predictions[base_type].append(step_prediction)
                print(f"predict: [node_type] {base_type}: {step_prediction.shape}")

                log.debug(f"[LATENT] Step {step} - {base_type}: {step_prediction.shape}")

        return predictions


    @staticmethod
    def _get_latent_step_info(data: HeteroData) -> dict:
        """
        Extract information about latent steps from the batch.
        Returns dict with step mapping and number of steps.
        """
        step_info = {}
        max_step = -1

        # Find all step-specific target nodes and map them to base instruments
        for node_type in data.node_types:
            if "_target_step" in node_type:
                # Extract: atms_target_step0 -> (atms_target, 0)
                parts = node_type.split("_step")
                if len(parts) == 2:
                    base_type = parts[0]  # e.g., "atms_target"
                    try:
                        step_num = int(parts[1])
                        if base_type not in step_info:
                            step_info[base_type] = {}
                        step_info[base_type][step_num] = node_type
                        max_step = max(max_step, step_num)
                    except ValueError:
                        continue

        return {
            "step_mapping": step_info,
            "num_steps": max_step + 1 if max_step >= 0 else 0
        }

    def _extract_ground_truths_and_metadata(self, batch, all_predictions):
        """
        Extract ground truth data and metadata for both latent and standard rollout modes.
        Returns dict structured for easy loss computation.
        """
        results = {}

        # LATENT ROLLOUT: Extract from step-specific nodes
        step_info = self._get_latent_step_info(batch)
        step_mapping = step_info["step_mapping"]

        for base_type, steps_dict in step_mapping.items():
            if base_type not in all_predictions:
                continue

            results[base_type] = {
                "gts_list": [],
                "instrument_ids_list": [],
                "valid_mask_list": []
            }

            # Extract ground truths for each step
            for step in sorted(steps_dict.keys()):
                step_node_type = steps_dict[step]  # e.g., "atms_target_step0"

                if step_node_type in batch.node_types:
                    y_true = batch[step_node_type].y
                    instrument_ids = getattr(batch[step_node_type], "instrument_ids", None)
                    valid_mask = getattr(batch[step_node_type], "target_channel_mask", None)

                    results[base_type]["gts_list"].append(y_true)
                    results[base_type]["instrument_ids_list"].append(instrument_ids)
                    results[base_type]["valid_mask_list"].append(valid_mask)
                else:
                    # Handle missing step data
                    results[base_type]["gts_list"].append(None)
                    results[base_type]["instrument_ids_list"].append(None)
                    results[base_type]["valid_mask_list"].append(None)

        return results


    def _resolve_init_ts(self, batch):
        """Return raw init timestamp (int/float unix, pd.Timestamp, or datetime), or None."""
        def _pick_attr(name):
            if not hasattr(batch, name):
                return None
            v = getattr(batch, name)
            if isinstance(v, (list, tuple)):
                v = v[0] if len(v) > 0 else None
            if v is None:
                return None

            if hasattr(v, 'item'):
                try:
                    if hasattr(v, 'numel') and v.numel() > 1:
                        flat = v.reshape(-1)
                        if not torch.all(flat == flat[0]):
                            import warnings
                            warnings.warn(
                                f"[_pick_attr] '{name}' has {v.numel()} non-identical "
                                f"values; using first ({flat[0].item()!r}) for timestamp.",
                                RuntimeWarning, stacklevel=2,
                            )
                        v = flat[0]
                    vv = v.item()
                    return None if float(vv) < 0 else vv
                except Exception:
                    return None

            return None if isinstance(v, (int, float)) and float(v) < 0 else v

        init_ts = _pick_attr('init_time')
        input_ts = _pick_attr('input_time')
        time_ts = _pick_attr('time')

        if init_ts is not None:
            return init_ts
        elif input_ts is not None:
            try:
                window_h = self.hparams.get('data_window_hours', None)
                if window_h is not None and isinstance(window_h, (int, float)):
                    return float(input_ts) + float(window_h) * 3600.0
                else:
                    return input_ts
            except Exception:
                return input_ts
        else:
            return time_ts

    def _extract_init_time_str(self, batch):
        """
        Extract initialization time string from batch in YYYYMMDDHH format.
        Args:
            batch: Batch data containing input_time or time attribute

        Returns:
            str: Init time as 'YYYYMMDDHH' or 'unknown' if unavailable
        """
        if batch is None:
            return 'unknown'
        ts = self._resolve_init_ts(batch)
        if ts is None:
            return 'unknown'
        try:
            # Handle pandas Timestamp
            if isinstance(ts, pd.Timestamp):
                return ts.strftime('%Y%m%d%H')

            # Handle Unix timestamp (float/int) as UTC
            if isinstance(ts, (int, float)):
                dt = datetime.utcfromtimestamp(float(ts))
                return dt.strftime('%Y%m%d%H')

            # Handle datetime object directly
            if isinstance(ts, datetime):
                # Ensure UTC-ish formatting (drop tz conversion here; upstream should be UTC)
                return ts.strftime('%Y%m%d%H')

            print(f"[INIT_TIME] Warning: Unsupported time type: {type(ts)}")
            return 'unknown'

        except Exception as e:
            print(f"[INIT_TIME] Error converting time: {e}, type: {type(ts)}")
            return 'unknown'

    def _extract_init_time_unix(self, batch) -> int:
        ts = self._resolve_init_ts(batch)
        if ts is None:
            raise ValueError(
                "[MESH PRED] Cannot determine init_time_unix from batch — "
                "batch has no valid init_time or input_time. "
                "Target-time conditioning requires a valid analysis time."
            )
        if isinstance(ts, pd.Timestamp):
            return int(ts.timestamp())
        if isinstance(ts, datetime):
            return int(ts.timestamp())
        return int(float(ts))


    def _decode_all_steps_to_mesh(self, mesh_features_per_step, mesh_pred_edges, init_time_unix):
        """Decode all forecast steps to mesh grid."""

        if not mesh_features_per_step:  # Check if the list is empty
            print("[MESH PRED] No mesh features available")
            return {}

        predictions = {}

        with torch.no_grad():
            for inst_name in self.mesh_instruments:
                if not self._is_mesh_pred_variable(inst_name):
                    continue

                if inst_name not in mesh_pred_edges:
                    continue

                predictions[inst_name] = []

                # Decode each step
                for step_idx, mesh_feat in enumerate(mesh_features_per_step):
                    pred = self._decode_one_step_to_mesh(mesh_feat, inst_name, mesh_pred_edges[inst_name],
                                                         step_idx=step_idx,
                                                         init_time_unix=init_time_unix
                                                         )
                    predictions[inst_name].append(pred)

        return predictions

    def _decode_one_step_to_mesh(self, mesh_features, inst_name, edges, step_idx, init_time_unix=None):
        """Decode one step's mesh features to mesh grid."""
        # Get decoder
        decoder_key = f"mesh__to__{inst_name}_target"
        decoder = self.observation_decoders[decoder_key]

        # Move edges to correct device (they're stored on CPU)
        device = mesh_features.device
        edge_index = edges['edge_index'].to(device)
        edge_attr = edges['edge_attr'].to(device)

        original_edge_index = decoder.edge_index
        decoder.edge_index = edge_index

        # Fix for pressure info in inference mode
        # Set it to level = 0 (1000mb) by default.
        N = edges['num_nodes']

        # Condition decoder on viewing geometry — mirrors the regular forward pass logic.
        # Only radiosonde/aircraft use pressure level conditioning; all other instruments
        # (satellites, surface obs, etc.) use zeros, consistent with their training setup.
        base_inst = inst_name.replace('_target', '')
        if base_inst in ['radiosonde', 'aircraft']:
            fixed_idx = torch.full((N,), self.mesh_pressure_level_idx, dtype=torch.long, device=device)
            pressure_emb = self.pressure_level_embedder(fixed_idx)  # [N, 8]
            if self.pressure_level_projector is not None:
                rec_rep = self.pressure_level_projector(pressure_emb)
            else:
                padding_dim = self.hidden_dim - self.pressure_level_embed_dim
                rec_rep = torch.cat([
                    torch.zeros(N, padding_dim, device=device),
                    pressure_emb
                ], dim=-1)  # [N, hidden_dim]

            print(f"[MESH PRED] Decoding {inst_name} conditioned on pressure level "
                  f"{self.mesh_pressure_level_idx} "
                  f"({[1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10][self.mesh_pressure_level_idx]} hPa)")
        else:
            # Satellites, surface obs etc.: no pressure conditioning (same as training)
            rec_rep = torch.zeros(N, self.hidden_dim, device=device)
            print(f"[MESH PRED] Decoding {inst_name} with zero initialization (no pressure conditioning)")

        # --- Target-time conditioning (mirrors regular decoder) ---
        if init_time_unix is None:
            raise ValueError(
                f"[MESH PRED] init_time_unix is required for target-time conditioning "
                f"but was not provided for instrument '{inst_name}'. "
                f"Pass the analysis time as a Unix timestamp when calling mesh prediction."
            )

        # Compute valid time = init + lead
        lead_seconds = int(round((step_idx + 0.5) * self.latent_step_hours * 3600))
        target_time_unix = int(int(init_time_unix) + lead_seconds)

        # Build time features using the same convention as _encode_target_time_features()
        # LST is estimated from mesh node longitude
        lons_deg = edges['lons'].cpu().numpy()  # [N]
        target_times_unix = np.full(N, target_time_unix, dtype=np.int64)
        time_feat_np = _encode_target_time_features(target_times_unix, lons_deg)  # [N, 5]
        time_feat = torch.from_numpy(time_feat_np).float().to(device)
        time_emb = self.target_time_embedder(time_feat)  # [N, 8]
        rec_rep = rec_rep + self.target_time_projector(time_emb)  # additive bias

        # Decode
        decoded = decoder(
            send_rep=mesh_features,
            rec_rep=rec_rep,
            edge_rep=edge_attr
        )

        decoder.edge_index = original_edge_index

        # Apply output mapper
        output_key = f"{inst_name}_target"
        predictions = self.output_mappers[output_key](decoded)

        return predictions

    def _save_latent_concatenated_csv(self, batch, node_type, preds_list, gts_list,
                                      valid_mask_list, out_dir, batch_idx, mode='val'):
        """
        Save latent rollout as concatenated observations.

        Args:
            mode: 'val' or 'predict' - determines output filename format
        """

        step_info = self._get_latent_step_info(batch)
        step_mapping = step_info["step_mapping"]

        # Collect all observations from all steps
        all_lat = []
        all_lon = []
        all_ts = []  # per-observation valid times (unix seconds); -1 if missing
        all_obs_ts = []  # per-observation observation times (unix seconds) inside 3h window; -1 if missing
        all_latent_step = []
        all_lead_hours_nominal = []
        all_pred = []
        all_true = []
        all_mask = []
        all_pressure = []  # Pressure in hPa for radiosonde/aircraft evaluation
        all_pressure_level = []  # Pressure level index (0-15) for stratified analysis
        all_persist = []

        # Persist scan-angle conditioning inputs for satellite-style targets.
        # For these node types, batch[step_node_type].x stores scan angle(s).
        scan_angle_expected_dim = 0
        if node_type == "ascat_target":
            scan_angle_expected_dim = 3
        elif node_type in ("atms_target", "amsua_target", "avhrr_target", "cris_pca_target", "seviri_asr_target", "seviri_csr_target"):
            scan_angle_expected_dim = 1

        all_scan_angle_cols = [list() for _ in range(scan_angle_expected_dim)] if scan_angle_expected_dim > 0 else []

        def _rounded_loc_keys(lat_deg, lon_deg, pressure_level=None):
            lat_key = np.round(np.asarray(lat_deg, dtype=np.float64), 4)
            lon_key = np.round(np.asarray(lon_deg, dtype=np.float64), 4)
            if pressure_level is None:
                return list(zip(lat_key.tolist(), lon_key.tolist()))
            pressure_key = np.asarray(pressure_level, dtype=np.int64)
            return list(zip(lat_key.tolist(), lon_key.tolist(), pressure_key.tolist()))

        def _build_persistence_lookup():
            input_node = node_type.replace("_target", "_input")
            if input_node not in batch.node_types:
                return None
            store = batch[input_node]
            required = ("input_features_raw", "input_times", "lat", "lon")
            if not all(hasattr(store, name) for name in required):
                return None

            vals = store.input_features_raw.detach().cpu().numpy()
            times = store.input_times.detach().cpu().numpy().astype(np.int64)
            lat = store.lat.detach().cpu().numpy()
            lon = store.lon.detach().cpu().numpy()
            mask = None
            if hasattr(store, "input_channel_mask"):
                mask = store.input_channel_mask.detach().cpu().numpy().astype(bool)
            pressure = None
            if hasattr(store, "pressure_level"):
                pressure = store.pressure_level.detach().cpu().numpy().astype(np.int64)

            if vals.size == 0 or times.size != vals.shape[0]:
                return None

            cutoff = init_unix if init_unix >= 0 else None
            latest = {}
            for row, key in enumerate(_rounded_loc_keys(lat, lon, pressure)):
                if cutoff is not None and times[row] > cutoff:
                    continue
                prev = latest.get(key)
                if prev is None or times[row] > prev[0]:
                    v = vals[row].astype(np.float64, copy=True)
                    if mask is not None and mask.shape == vals.shape:
                        v[~mask[row]] = np.nan
                    latest[key] = (int(times[row]), v)
            return latest, pressure is not None

        # Init time columns (constant per file/batch when available)
        init_time_str = self._extract_init_time_str(batch)
        init_dt_str = ""
        init_unix = -1
        if init_time_str not in (None, '', 'unknown'):
            try:
                init_dt = pd.to_datetime(init_time_str, format='%Y%m%d%H', utc=True)
                init_dt_str = init_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                init_unix = int(init_dt.timestamp())
            except Exception:
                init_dt_str = ""
                init_unix = -1

        persistence_lookup = _build_persistence_lookup()

        for step in range(len(preds_list)):
            if step >= len(preds_list) or step >= len(gts_list):
                continue

            y_pred = preds_list[step]
            y_true = gts_list[step]
            valid_mask = valid_mask_list[step] if step < len(valid_mask_list) else None

            if y_pred is None or y_true is None:
                continue

            # Unnormalize
            y_pred_unnorm = self.unnormalize_standardscaler(y_pred, node_type)
            y_true_unnorm = self.unnormalize_standardscaler(y_true, node_type)

            # Get metadata for this step
            if node_type in step_mapping and step in step_mapping[node_type]:
                step_node_type = step_mapping[node_type][step]
                if hasattr(batch[step_node_type], 'target_metadata'):
                    target_metadata = batch[step_node_type].target_metadata
                    lat = target_metadata[:, 0].cpu().numpy()
                    lon = target_metadata[:, 1].cpu().numpy()
                    lat_deg = np.degrees(lat)
                    lon_deg = np.degrees(lon)

                else:
                    n = y_pred_unnorm.shape[0]
                    lat_deg = np.zeros(n)
                    lon_deg = np.zeros(n)

                # Per-observation timestamps (epoch seconds) if present
                if hasattr(batch[step_node_type], 'target_times'):
                    ts = batch[step_node_type].target_times.detach().cpu().numpy()
                    ts = np.asarray(ts, dtype=np.int64)
                else:
                    ts = np.full(y_pred_unnorm.shape[0], -1, dtype=np.int64)

                # Per-observation real obs time (epoch seconds) if present
                if hasattr(batch[step_node_type], 'obs_time_unix'):
                    obs_ts = batch[step_node_type].obs_time_unix.detach().cpu().numpy()
                    obs_ts = np.asarray(obs_ts, dtype=np.int64)
                else:
                    obs_ts = np.full(y_pred_unnorm.shape[0], -1, dtype=np.int64)

                # Get pressure data if available (for radiosonde and aircraft)
                if hasattr(batch[step_node_type], 'target_pressure_hpa'):
                    pressure_hpa = batch[step_node_type].target_pressure_hpa.cpu().numpy()
                else:
                    pressure_hpa = np.full(y_pred_unnorm.shape[0], np.nan)

                # Get pressure level index if available (for stratified analysis)
                if hasattr(batch[step_node_type], 'pressure_level'):
                    pressure_level_idx = batch[step_node_type].pressure_level.cpu().numpy()
                else:
                    pressure_level_idx = np.full(y_pred_unnorm.shape[0], -1, dtype=np.int32)

                # Scan-angle export for satellite-style targets.
                if all_scan_angle_cols:
                    try:
                        sa = getattr(batch[step_node_type], "x", None)
                        if sa is None:
                            raise ValueError("missing x")
                        sa_np = sa.detach().cpu().numpy()
                        sa_np = np.asarray(sa_np)
                        if sa_np.ndim == 1:
                            sa_np = sa_np[:, None]
                        if sa_np.shape[0] != int(y_pred_unnorm.shape[0]):
                            raise ValueError(f"row mismatch: x has {sa_np.shape[0]} rows, preds have {int(y_pred_unnorm.shape[0])}")
                        # Use the first expected columns; pad with NaN if fewer are present.
                        for j in range(scan_angle_expected_dim):
                            if j < sa_np.shape[1]:
                                all_scan_angle_cols[j].extend(sa_np[:, j].astype(np.float64).tolist())
                            else:
                                all_scan_angle_cols[j].extend([float('nan')] * int(y_pred_unnorm.shape[0]))
                    except Exception:
                        for j in range(scan_angle_expected_dim):
                            all_scan_angle_cols[j].extend([float('nan')] * int(y_pred_unnorm.shape[0]))
            else:
                n = y_pred_unnorm.shape[0]
                lat_deg = np.zeros(n)
                lon_deg = np.zeros(n)
                ts = np.full(n, -1, dtype=np.int64)
                obs_ts = np.full(n, -1, dtype=np.int64)
                pressure_hpa = np.full(n, np.nan)
                pressure_level_idx = np.full(n, -1, dtype=np.int32)

                if all_scan_angle_cols:
                    for j in range(scan_angle_expected_dim):
                        all_scan_angle_cols[j].extend([float('nan')] * int(n))

            # Collect data from this step
            all_lat.extend(lat_deg)
            all_lon.extend(lon_deg)
            all_ts.extend(ts.tolist())
            all_obs_ts.extend(obs_ts.tolist())
            all_latent_step.extend([int(step)] * int(len(ts)))
            lead_nom = np.nan
            try:
                lead_nom = float(step + 1) * float(self.latent_step_hours)
            except Exception:
                lead_nom = np.nan
            all_lead_hours_nominal.extend([lead_nom] * int(len(ts)))
            all_pred.append(y_pred_unnorm.detach().cpu().numpy())
            all_true.append(y_true_unnorm.detach().cpu().numpy())
            all_pressure.extend(pressure_hpa)
            all_pressure_level.extend(pressure_level_idx)

            persist = np.full(
                y_true_unnorm.detach().cpu().numpy().shape,
                np.nan,
                dtype=np.float64,
            )
            if persistence_lookup is not None:
                latest, needs_pressure = persistence_lookup
                keys = _rounded_loc_keys(
                    lat_deg,
                    lon_deg,
                    pressure_level_idx if needs_pressure else None,
                )
                for row, key in enumerate(keys):
                    item = latest.get(key)
                    if item is not None:
                        vals = item[1]
                        width = min(persist.shape[1], vals.shape[0])
                        persist[row, :width] = vals[:width]
            all_persist.append(persist)

            if valid_mask is not None:
                all_mask.append(valid_mask.detach().cpu().numpy().astype(bool))
            else:
                all_mask.append(np.ones_like(y_pred_unnorm.detach().cpu().numpy(), dtype=bool))

        if not all_pred:
            print(f"[WARN] No valid predictions for {node_type}, skipping CSV save")
            return

        # Concatenate all steps
        # If there is no ground truth at all, treat this as inference mode and skip saving
        if not all_true:
            print(f"[PREDICT] latent csv: Skipping {node_type} - no ground truth data (inference mode)")
            return

        all_pred_concat = np.vstack(all_pred)
        all_true_concat = np.vstack(all_true)
        all_mask_concat = np.vstack(all_mask)
        all_persist_concat = np.vstack(all_persist) if all_persist else np.full_like(all_true_concat, np.nan, dtype=np.float64)

        # Skip saving if no real ground truth data
        if all_true_concat.size == 0:
            print(f"[PREDICT] latent csv: Skipping {node_type} - empty ground truth array")
            return

        n = all_pred_concat.shape[0]
        n_ch = all_pred_concat.shape[1]

        # Get feature names
        feats = self._feature_names_for_node(node_type)
        if not feats:
            feats = [f"ch{i+1}" for i in range(n_ch)]
        if len(feats) > n_ch:
            feats = feats[:n_ch]
        elif len(feats) < n_ch:
            feats = feats + [f"ch{i+1}" for i in range(len(feats) + 1, n_ch + 1)]

        def _safe_col_name(s: str) -> str:
            return str(s).replace(" ", "_")

        # Build DataFrame in EXACT same format as standard rollout
        df = pd.DataFrame({"lat": all_lat, "lon": all_lon})

        # Scan-angle columns (when applicable)
        if all_scan_angle_cols and len(all_scan_angle_cols[0]) == len(df):
            for j in range(scan_angle_expected_dim):
                df[f"scan_angle_{j}"] = np.asarray(all_scan_angle_cols[j], dtype=np.float64)

        if init_dt_str:
            df.insert(0, 'init_datetime', init_dt_str)
            df.insert(1, 'init_time_unix', init_unix)

        insert_pos = 2 if 'init_datetime' in df.columns else 0

        # Per-observation valid times (if present)
        ts_arr = np.asarray(all_ts, dtype=np.int64)
        if ts_arr.size == len(df):
            # Fallback: if per-observation times are missing/invalid, compute valid time from init + nominal lead.
            if init_unix >= 0:
                try:
                    lead_seconds = (np.asarray(all_lead_hours_nominal, dtype=np.float64) * 3600.0).round().astype(np.int64)
                    computed_ts = init_unix + lead_seconds
                    ts_arr = np.where(ts_arr >= 0, ts_arr, computed_ts)
                except Exception:
                    pass

            dt = pd.to_datetime(pd.Series(ts_arr).replace(-1, pd.NA), unit='s', utc=True, errors='coerce')
            df.insert(insert_pos, 'datetime', dt.dt.strftime('%Y-%m-%dT%H:%M:%SZ').fillna(''))
            df.insert(insert_pos + 1, 'valid_time_unix', ts_arr)

        # Real obs timestamps (inside the target sub-window), if present
        obs_ts_arr = np.asarray(all_obs_ts, dtype=np.int64)
        if obs_ts_arr.size == len(df):
            df.insert(insert_pos + 2, 'obs_time_unix', obs_ts_arr)

        # Step/lead metadata so rows can be grouped per forecast hour
        step_arr = np.asarray(all_latent_step, dtype=np.int64)
        if step_arr.size == len(df):
            df['latent_step'] = step_arr
        lead_arr = np.asarray(all_lead_hours_nominal, dtype=np.float64)
        if lead_arr.size == len(df):
            df['lead_hours_nominal'] = lead_arr

        for i, fname in enumerate(feats):
            col = _safe_col_name(fname)
            df[f"pred_{col}"] = all_pred_concat[:, i]
            df[f"persist_{col}"] = all_persist_concat[:, i]
            df[f"true_{col}"] = all_true_concat[:, i]
            df[f"mask_{col}"] = all_mask_concat[:, i]

        # Add pressure columns for radiosonde and aircraft evaluation
        all_pressure_arr = np.array(all_pressure)
        all_pressure_level_arr = np.array(all_pressure_level)

        # Define standard pressure levels for labeling
        STANDARD_PRESSURE_LEVELS = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10])

        if not np.all(np.isnan(all_pressure_arr)):
            df["pressure_hPa"] = all_pressure_arr
            # Compute log_pressure_height: z = -8000 * ln(P/1013.25) meters
            pressure_clipped = np.clip(all_pressure_arr, 1.0, 1100.0)
            log_pressure_height = -8000.0 * np.log(pressure_clipped / 1013.25)
            df["log_pressure_height_m"] = log_pressure_height
            # Also add normalized version for reference
            df["log_pressure_height_norm"] = log_pressure_height / 20000.0

            # Add pressure level index and label for stratified analysis
            df["pressure_level_idx"] = all_pressure_level_arr
            # Create human-readable labels (e.g., "850hPa", "500hPa")
            pressure_level_labels = []
            for idx in all_pressure_level_arr:
                if 0 <= idx < len(STANDARD_PRESSURE_LEVELS):
                    pressure_level_labels.append(f"{STANDARD_PRESSURE_LEVELS[idx]:.0f}hPa")
                else:
                    pressure_level_labels.append("unknown")
            df["pressure_level_label"] = pressure_level_labels

            print(
                f"  Added pressure columns: pressure_hPa, log_pressure_height_m, "
                f"log_pressure_height_norm, pressure_level_idx, pressure_level_label"
            )
            print(f"  Pressure range: {np.nanmin(all_pressure_arr):.1f} - {np.nanmax(all_pressure_arr):.1f} hPa")
            # Show distribution by pressure level
            valid_levels = all_pressure_level_arr[all_pressure_level_arr >= 0]
            if len(valid_levels) > 0:
                print(f"  Pressure level distribution: {np.unique(valid_levels, return_counts=True)}")
        elif np.any(all_pressure_level_arr >= 0):
            # Even if pressure_hPa is not available, save pressure_level if it exists
            df["pressure_level_idx"] = all_pressure_level_arr
            pressure_level_labels = []
            for idx in all_pressure_level_arr:
                if 0 <= idx < len(STANDARD_PRESSURE_LEVELS):
                    pressure_level_labels.append(f"{STANDARD_PRESSURE_LEVELS[idx]:.0f}hPa")
                else:
                    pressure_level_labels.append("unknown")
            df["pressure_level_label"] = pressure_level_labels
            print(f"  Added pressure_level_idx and pressure_level_label columns")
            valid_levels = all_pressure_level_arr[all_pressure_level_arr >= 0]
            if len(valid_levels) > 0:
                print(f"  Pressure level distribution: {np.unique(valid_levels, return_counts=True)}")

        # Optional subsampling to bound I/O and file size (validation diagnostics)
        if mode != 'predict' and self.val_csv_max_rows is not None and len(df) > self.val_csv_max_rows:
            try:
                node_seed = abs(hash(str(node_type))) % 1000003
                seed = (
                    int(self.val_csv_sample_seed)
                    + int(self.current_epoch) * 1000003
                    + int(batch_idx) * 9176
                    + int(node_seed)
                ) % (2**32 - 1)
                df = df.sample(n=int(self.val_csv_max_rows), random_state=int(seed))
            except Exception:
                pass

        # Save with appropriate filename based on mode
        if mode == 'predict':
            if init_time_str != 'unknown':
                filename = f"{out_dir}/pred_{node_type}_init_{init_time_str}.csv"
            else:
                filename = f"{out_dir}/pred_{node_type}_batch{batch_idx}.csv"
        else:  # validation mode
            if init_time_str != 'unknown':
                filename = f"{out_dir}/val_{node_type}_init_{init_time_str}_epoch{self.current_epoch}_batch{batch_idx}.csv"
            else:
                filename = f"{out_dir}/val_{node_type}_epoch{self.current_epoch}_batch{batch_idx}_step0.csv"
        df.to_csv(filename, index=False)
        print(f"Saved latent concatenated CSV: {filename}")
        print(f"  Total observations from all steps: {len(df)}")
        print(f"  Steps combined: {len(all_pred)}")
