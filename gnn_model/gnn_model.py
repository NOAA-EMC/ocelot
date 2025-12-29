import lightning.pytorch as pl
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import is_initialized
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
from processor import Processor
from utils import make_mlp
from interaction_net import InteractionNet
from create_mesh_graph_global import create_mesh
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List
from torch_geometric.utils import scatter
from loss import weighted_huber_loss
from processor_transformer import SlidingWindowTransformerProcessor
from attn_bipartite import BipartiteGAT


def _build_instrument_map(observation_config: dict) -> dict[str, int]:
    order = []
    for group in ("satellite", "conventional"):
        if group in observation_config:
            order += sorted(observation_config[group].keys())
    return {name: i for i, name in enumerate(order)}


class GNNLightning(pl.LightningModule):
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
        observation_config,
        target_config,
        hidden_dim,
        mesh_resolution=6,
        num_layers=4,
        lr=1e-4,
        instrument_weights=None,
        channel_weights=None,
        verbose=False,
        detect_anomaly=False,
        max_rollout_steps=1,
        rollout_schedule="step",
        feature_stats=None,
        processor_type: str = "interaction",  # "interaction" | "sliding_transformer"
        processor_window: int = 4,
        processor_depth: int = 2,
        processor_heads: int = 4,
        processor_dropout: float = 0.0,
        encoder_type: str = "interaction",     # "interaction" | "gat"
        decoder_type: str = "interaction",     # "interaction" | "gat"
        encoder_heads: int = 4,
        decoder_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        # MK: Target decoder parameters
        target_decoder_type: str = "interaction",  # "interaction" | "gat"
        target_decoder_layers: int = 2,
        target_decoder_heads: int = 4,
        target_decoder_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the GNNLightning model with an encoder, processor, and decoder.

        Parameters:
        input_dim (int): Number of input features per observation node (before encoding).
        hidden_dim (int): Size of the hidden representation used in all layers.
        target_dim (int): Number of features to predict at each target node.
        lr (float, optional): Learning rate for the optimizer (default: 1e-4).
        """
        super().__init__()
        self.verbose = verbose
        self.detect_anomaly = detect_anomaly
        self.feature_stats = feature_stats
        self.save_hyperparameters()
        self.lr = lr
        self.instrument_weights = instrument_weights or {}
        self.channel_weights = channel_weights or {}
        self.max_rollout_steps = max_rollout_steps
        self.rollout_schedule = rollout_schedule

        self.observation_config = observation_config

        # MK: Load target variable config
        self.target_variables_enabled = target_config.get('enabled', False)
        self.target_variable_config = target_config
        self.target_instruments = list(target_config.get('variables', {}).keys())
        print(f"[DEBUG CONFIG] target_variables_enabled: {self.target_variables_enabled}")
        print(f"[DEBUG CONFIG] target_variable_config: {self.target_variable_config}")
        print(f"[DEBUG CONFIG] variables in config: {self.target_variable_config.get('variables', {})}")
        print(f"[DEBUG CONFIG] Target instruments for mesh prediction: {self.target_instruments}")

        # Mirror process_timeseries._name2id()
        self.instrument_name_to_id = _build_instrument_map(self.observation_config)
        self.instrument_id_to_name = {v: k for k, v in self.instrument_name_to_id.items()}

        # Normalize user-provided weights (accept names or ids)
        self.instrument_weights = self._normalize_inst_weights(instrument_weights)
        self.channel_weights = self._normalize_channel_weights(channel_weights)

        # Boolean masks per instrument for valid channels (weights > 0)
        self.channel_masks = {inst_id: (w > 0) for inst_id, w in self.channel_weights.items()}

        if self.verbose:
            print("[MODEL] instrument map:", self.instrument_name_to_id)
            print("[MODEL] instrument_weights:", {self.instrument_id_to_name[k]: float(v) for k, v in self.instrument_weights.items()})

        self.hidden_dim = hidden_dim

        # --- Create and store the mesh structure as part of the model ---
        self.mesh_structure = create_mesh(splits=mesh_resolution, levels=4, hierarchical=False, plot=False)

        # MK: Store mesh coordinates for saving mesh predictions
        mesh_lat_lon = self.mesh_structure["mesh_lat_lon_torch"][0]  # Mesh coordinates
        self.register_buffer("mesh_lats", mesh_lat_lon[:, 0])  # Latitude in radians
        self.register_buffer("mesh_lons", mesh_lat_lon[:, 1])  # Longitude in radians
        print(f"[MESH] Registered {len(mesh_lat_lon)} mesh coordinates (radians)")

        mesh_feature_dim = self.mesh_structure["mesh_features_torch"][0].shape[1]
        # --- Register the static mesh data as model buffers ---
        mesh_x = self.mesh_structure["mesh_features_torch"][0]
        mesh_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        mesh_edge_attr = self.mesh_structure["m2m_features_torch"][0]

        # --- Initialize Network Dictionaries ---
        self.observation_embedders = nn.ModuleDict()  # For initial feature projection
        self.observation_encoders = nn.ModuleDict()  # For obs -> mesh GNNs
        self.observation_decoders = nn.ModuleDict()
        self.output_mappers = nn.ModuleDict()  # For final prediction MLPs
        # MK
        self.target_decoders = nn.ModuleDict()      # For target variable transformers
        self.target_projections = nn.ModuleDict()   # For target variable projections

        first_instrument_config = next(iter(next(iter(observation_config.values())).values()))
        hidden_layers = first_instrument_config.get("encoder_hidden_layers", 2)

        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.mesh_embedder = make_mlp([mesh_feature_dim] + self.mlp_blueprint_end)

        # Create scan-angle embedders once to avoid loop-order surprises
        self.scan_angle_embed_dim = 8
        self.scan_angle_embedder = make_mlp([1, self.scan_angle_embed_dim])
        self.ascat_scan_angle_embedder = make_mlp([3, self.scan_angle_embed_dim])

        # MK: ADD: Mesh edge embedder for target decoders
        # Mesh edges have 4 features (distance, dx, dy, dz or similar)
        # mesh_edge_input_dim = self.mesh_structure["m2m_features_torch"][0].shape[1]
        # self.mesh_edge_embedder = make_mlp([mesh_edge_input_dim, hidden_dim])
        # print(f"[TARGET DECODER] Created mesh edge embedder: {mesh_edge_input_dim} -> {hidden_dim}")

        node_types = ["mesh"]
        edge_types = [("mesh", "to", "mesh")]

        # --- wire processor choice ---
        self.processor_type = processor_type  # "interaction" | "sliding_transformer"

        if self.processor_type == "sliding_transformer":
            self.swt = SlidingWindowTransformerProcessor(
                hidden_dim=self.hidden_dim,
                window=processor_window,
                depth=processor_depth,
                num_heads=processor_heads,
                dropout=processor_dropout,
                use_causal_mask=True,
            )
        elif self.processor_type == "interaction":
            pass  # already built self.processor above
        else:
            raise ValueError(f"Unknown processor_type: {processor_type!r}")

        for obs_type, instruments in observation_config.items():
            for inst_name, cfg in instruments.items():
                node_type_input = f"{inst_name}_input"
                node_type_target = f"{inst_name}_target"

                node_types.extend([node_type_input, node_type_target])
                edge_types.extend([(node_type_input, "to", "mesh"), ("mesh", "to", node_type_target)])

                input_dim = cfg.get("input_dim")
                target_dim = cfg.get("target_dim")

                # Encoder GNN (obs -> mesh)
                edge_type_tuple_enc = (node_type_input, "to", "mesh")
                enc_key = self._edge_key(edge_type_tuple_enc)

                if encoder_type == "gat":
                    enc_edge_dim = hidden_dim   # <- match the zeros you already pass in forward
                    self.observation_encoders[enc_key] = BipartiteGAT(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=encoder_layers,
                        heads=encoder_heads,
                        dropout=encoder_dropout,
                        edge_dim=enc_edge_dim,   # <- use edge_attr exactly like InteractionNet path
                    )
                else:
                    self.observation_encoders[enc_key] = InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        update_edges=False,
                    )
                # Decoder GNN (mesh -> target)
                edge_type_tuple_dec = ("mesh", "to", node_type_target)
                dec_key = self._edge_key(edge_type_tuple_dec)

                if decoder_type == "gat":
                    dec_edge_dim = hidden_dim   # <- same idea for decoder
                    self.observation_decoders[dec_key] = BipartiteGAT(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=decoder_layers,
                        heads=decoder_heads,
                        dropout=decoder_dropout,
                        edge_dim=dec_edge_dim,
                    )
                else:
                    self.observation_decoders[dec_key] = InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        update_edges=False,
                    )

                # Initial MLP to project raw features to hidden_dim
                self.observation_embedders[node_type_input] = make_mlp([input_dim] + self.mlp_blueprint_end)

                # Final MLP: add scan-angle embedder for ATMS, AMSU-A, AVHRR, and ASCAT targets
                targets_with_scan = {"atms_target", "amsua_target", "avhrr_target", "ascat_target"}
                if node_type_target in targets_with_scan:
                    # Embedders already created above; just use them
                    input_dim_for_mapper = hidden_dim + self.scan_angle_embed_dim
                else:
                    input_dim_for_mapper = hidden_dim

                output_map_layers = [input_dim_for_mapper] + [hidden_dim] * hidden_layers + [target_dim]
                self.output_mappers[node_type_target] = make_mlp(output_map_layers, layer_norm=False)

        self.processor = Processor(
            hidden_dim=hidden_dim,
            node_types=node_types,
            edge_types=edge_types,
            num_message_passing_steps=num_layers,
        )

        # --- wire processor choice ---
        self.processor_type = processor_type  # "interaction" | "sliding_transformer"

        if self.processor_type == "sliding_transformer":
            self.swt = SlidingWindowTransformerProcessor(
                hidden_dim=self.hidden_dim,
                window=processor_window,
                depth=processor_depth,
                num_heads=processor_heads,
                dropout=processor_dropout,
                use_causal_mask=True,
            )
        elif self.processor_type == "interaction":
            pass  # already built self.processor above
        else:
            raise ValueError(f"Unknown processor_type: {processor_type!r}")

        # MK
        # --- Build target decoders for target variables ---
        if self.target_variables_enabled:
            print(f"[TARGET CONFIG] Validating target variables configuration...")

            for inst_name, var_list in self.target_variable_config.get('variables', {}).items():
                # Verify instrument exists in observation config
                inst_config = self._get_instrument_config(inst_name)
                if inst_config is None:
                    raise ValueError(
                        f"[TARGET CONFIG ERROR] Instrument '{inst_name}' specified in target_config.yaml "
                        f"not found in observation_config.yaml"
                    )

                # Verify each variable exists in the instrument's features
                all_features = inst_config.get('features', [])
                for var_name in var_list:
                    if var_name not in all_features:
                        raise ValueError(
                            f"[TARGET CONFIG ERROR] Variable '{var_name}' specified in target_config.yaml "
                            f"not found in features for '{inst_name}'. Available: {all_features}"
                        )
                
                print(f"[TARGET CONFIG] Validated {inst_name}: {len(var_list)} variables")

            # NOW BUILD DECODERS: ONE PER INSTRUMENT (not per variable!)
            for inst_name, var_list in self.target_variable_config.get('variables', {}).items():
                # Get number of target variables for this instrument
                num_target_vars = len(var_list)

                # Build ONE decoder per instrument (like regular decoders)
                if target_decoder_type == "gat":
                    self.target_decoders[inst_name] = BipartiteGAT(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        layers=target_decoder_layers,
                        heads=target_decoder_heads,
                        dropout=target_decoder_dropout,
                        edge_dim=hidden_dim,
                    )
                else:  # interaction
                    self.target_decoders[inst_name] = InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=target_decoder_layers,
                        update_edges=False,
                    )

                # Build ONE projection per instrument: hidden_dim → num_target_vars
                projection_layers = [hidden_dim] + [hidden_dim] * target_decoder_layers + [num_target_vars]
                self.target_projections[inst_name] = make_mlp(projection_layers, layer_norm=False)

                if self.verbose:
                    print(f"[TARGET DECODER] Built {inst_name}: 1 decoder + 1 projection ({hidden_dim} → {num_target_vars})")

            print(f"[TARGET CONFIG] Successfully built {len(self.target_decoders)} target decoders")

            ## comprehensive debug output after building all decoders
            print("\n" + "="*80)
            print("[TARGET DECODER SUMMARY]")
            print("="*80)
            print(f"Target variables enabled: {self.target_variables_enabled}")
            print(f"Target instruments: {self.target_instruments}")
            print(f"\nDecoders built: {len(self.target_decoders)}")

            for inst_name in self.target_instruments:
                var_list = self.target_variable_config['variables'][inst_name]
                print(f"\n{inst_name}:")
                print(f"  Variables: {var_list}")

                # ✅ NEW: Use per-instrument key
                decoder_type = type(self.target_decoders[inst_name]).__name__
                projection_output_dim = len(var_list)

                print(f"  Decoder: {inst_name} ({decoder_type})")
                print(f"  Projection: {inst_name} (MLP: {self.hidden_dim} → {projection_output_dim})")
                print(f"  Processes {len(var_list)} variables in one forward pass")

            print("="*80)


        def _as_f32(x):
            import torch

            return x.clone().detach().to(torch.float32) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

        def _as_i64(x):
            import torch

            return x.clone().detach().to(torch.long) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

        self.register_buffer("mesh_x", _as_f32(mesh_x))
        self.register_buffer("mesh_edge_index", _as_i64(mesh_edge_index))
        self.register_buffer("mesh_edge_attr", _as_f32(mesh_edge_attr))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # PyG Data/HeteroData implements .to()
        if hasattr(batch, "to"):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _is_target_variable(self, inst_name: str) -> bool:
        """Check if instrument has target variables configured."""
        if not self.target_variables_enabled:
            return False

        base_inst_name = inst_name.replace('_target', '')

        return base_inst_name in self.target_instruments

    def _simple_3nn_interpolation(self, variable_mesh_values, edge_index,
                                  edge_attr, num_target_nodes):
        """
        Simple 3NN interpolation using distances from existing edge_attr.
        NO learnable parameters - just inverse-distance weighted averaging.

        Args:
            variable_mesh_values: Physical values on mesh [N_mesh, 1]
            edge_index: 3NN connectivity [2, N_edges] (from existing connector)
            edge_attr: Distances [N_edges, 1 or more] (from existing connector)
            num_target_nodes: Number of observation locations

        Returns:
            Interpolated values [N_obs, 1]
        """
        # Get source mesh values for each edge
        source_values = variable_mesh_values[edge_index[0]]  # [N_edges, 1]

        # Extract distances from edge_attr (reuse existing!)
        if edge_attr.dim() > 1:
            distances = edge_attr[:, 0:1]  # [N_edges, 1]
        else:
            distances = edge_attr.unsqueeze(-1)  # [N_edges, 1]

        # Convert distances to weights (closer = higher weight)
        weights = 1.0 / (distances + 1e-8)

        # Weight source values
        weighted_values = source_values * weights  # [N_edges, 1]

        # Aggregate to target nodes (sum weighted values)
        target_values = scatter(
            weighted_values,
            edge_index[1],  # target node indices
            dim=0,
            dim_size=num_target_nodes,
            reduce='sum'
        )

        # Normalize by sum of weights
        weight_sums = scatter(
            weights,
            edge_index[1],
            dim=0,
            dim_size=num_target_nodes,
            reduce='sum'
        )

        target_values = target_values / (weight_sums + 1e-8)

        return target_values  # [N_obs, 1]

    def _get_instrument_config(self, inst_name: str):
        """
        Get configuration for an instrument.
        Follows same pattern as _feature_names_for_node().
        
        Args:
            inst_name: e.g., 'surface_obs', 'radiosonde'
        
        Returns:
            Config dict for the instrument
        """
        for obs_type, instruments in self.observation_config.items():
            if inst_name in instruments:
                return instruments[inst_name]
        return None

    def _decode_target_variables(self, base_type, mesh_features, step_node_type,
                                 step_edge_index, data, return_mesh_predictions=False):
        """
        Decode target variables using: decoder → projection → 3NN interpolation.
        Uses ONE decoder per instrument (like regular decoders).
        
        Args:
            base_type: e.g., 'surface_obs_target', 'radiosonde_target'
            mesh_features: Processed mesh features [N_mesh, hidden_dim]
            step_node_type: e.g., 'surface_obs_target_step0'
            step_edge_index: 3NN edges from mesh to observation locations
            data: HeteroData batch
            return_mesh_predictions: If True, return (predictions, mesh_dict)
        
        Returns:
            If return_mesh_predictions=False: predictions [N_obs, num_features]
            If return_mesh_predictions=True: (predictions, mesh_predictions_dict)
        """
        # Extract instrument name (remove '_target' suffix)
        inst_name = base_type.replace('_target', '')

        # Get config
        inst_config = self._get_instrument_config(inst_name)
        if inst_config is None:
            raise ValueError(f"Instrument {inst_name} not found in config")

        all_features = inst_config['features']
        target_vars = self.target_variable_config['variables'].get(inst_name, [])
        regular_vars = [f for f in all_features if f not in target_vars]

        # Get mesh-to-mesh edges for decoder
        mesh_edge_index = data[("mesh", "to", "mesh")].edge_index
        
        # Create zeros ONCE
        num_mesh_edges = mesh_edge_index.size(1)
        mesh_edge_attr_zeros = torch.zeros(
            (num_mesh_edges, self.hidden_dim), 
            device=mesh_features.device
        )

        # Get decoder edge attributes (distances from existing 3NN)
        decoder_edge_attr = data[("mesh", "to", step_node_type)].edge_attr
        num_target_nodes = data[step_node_type].num_nodes
        reference_device = mesh_features.device

        # Storage for outputs by feature name
        feature_outputs_dict = {}

        # === Process Target Variables ===
        if target_vars:
            # ONE decoder call for ALL target variables
            decoder = self.target_decoders[inst_name]
            decoder.edge_index = mesh_edge_index
            
            # Part 1: Decoder (mesh → mesh)
            transformed = decoder(
                send_rep=mesh_features,
                rec_rep=mesh_features,
                edge_rep=mesh_edge_attr_zeros,
            )  # [N_mesh, hidden_dim]

            # Part 2: Projection (hidden_dim → num_target_vars)
            physical_values_all = self.target_projections[inst_name](transformed)  # [N_mesh, num_target_vars]

            # Part 3: Split by variable and interpolate
            mesh_predictions_dict = {} if return_mesh_predictions else None
            
            for i, var_name in enumerate(target_vars):
                # Extract this variable's values
                physical_values = physical_values_all[:, i:i+1]  # [N_mesh, 1]
                
                # Store mesh predictions (DETACHED to prevent memory leak)
                if return_mesh_predictions:
                    mesh_predictions_dict[var_name] = physical_values.detach().clone()
                
                # 3NN interpolation (mesh → obs)
                interpolated = self._simple_3nn_interpolation(
                    physical_values,
                    step_edge_index,
                    decoder_edge_attr,
                    num_target_nodes,
                )  # [N_obs, 1]

                feature_outputs_dict[var_name] = interpolated

        # === Process Regular Variables (if any) ===
        if regular_vars:
            # Use existing decoder path for non-target variables
            target_features_initial = torch.zeros(num_target_nodes, self.hidden_dim, device=reference_device)
            edge_attr = torch.zeros((step_edge_index.size(1), self.hidden_dim), device=reference_device)

            # FIX: Use correct decoder key
            decoder_key = self._edge_key(("mesh", "to", base_type))
            decoder = self.observation_decoders[decoder_key]
            decoder.edge_index = step_edge_index

            decoded_features = decoder(
                send_rep=mesh_features,
                rec_rep=target_features_initial,
                edge_rep=edge_attr,
            )

            # Apply scan angle embedding if needed
            scan_angle = data[step_node_type].x

            if base_type == "ascat_target":
                scan_angle_embedded = self.ascat_scan_angle_embedder(scan_angle)
                final_features = torch.cat([decoded_features, scan_angle_embedded], dim=-1)
                all_predictions = self.output_mappers[base_type](final_features)
            elif base_type in ("atms_target", "amsua_target", "avhrr_target"):
                scan_angle_embedded = self.scan_angle_embedder(scan_angle)
                final_features = torch.cat([decoded_features, scan_angle_embedded], dim=-1)
                all_predictions = self.output_mappers[base_type](final_features)
            else:
                all_predictions = self.output_mappers[base_type](decoded_features)

            # Extract only the non-target feature columns
            for i, feature_name in enumerate(all_features):
                if feature_name in regular_vars:
                    feature_outputs_dict[feature_name] = all_predictions[:, i:i+1]

        # === Combine in correct feature order ===
        output_list = []
        for feature_name in all_features:
            output_list.append(feature_outputs_dict[feature_name])

        output = torch.cat(output_list, dim=1)  # [N_obs, num_features]

        # Return with or without mesh predictions
        if return_mesh_predictions:
            return output, mesh_predictions_dict
        else:
            return output

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
            for group, instruments in self.observation_config.items():
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
        for obs_type, instruments in self.observation_config.items():
            if inst_name in instruments:
                return instruments[inst_name].get("features", None)
        return None

    def debug(self, *args, **kwargs):
        if getattr(self, "verbose", False) and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
            print(*args, **kwargs)

    def on_fit_start(self):
        if getattr(self, "detect_anomaly", False):
            # enable once per run, not every batch
            torch.autograd.set_detect_anomaly(True)
            if self.trainer.is_global_zero:
                self.debug("[ANOMALY] torch.autograd anomaly mode enabled once at fit start.")

    def _edge_key(self, edge_type: Tuple[str, str, str]) -> str:
        """Converts an edge_type tuple to a string key for ModuleDict."""
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        rank = int(os.environ.get("RANK", "0"))

        # One concise banner (only once on global zero)
        if getattr(self.trainer, "is_global_zero", True):
            print(f"=== Starting Epoch {self.current_epoch} ===")

        print(f"[Rank {rank}] === TRAIN EPOCH {self.current_epoch} START ===")

        dm = self.trainer.datamodule
        train_start = getattr(dm.hparams, "train_start", None)
        train_end = getattr(dm.hparams, "train_end", None)
        sum_id = id(getattr(dm, "train_data_summary", None))
        print(f"[TrainWindow] {train_start} .. {train_end} (sum_id={sum_id})")

        # reset first-batch flag for this epoch
        self._printed_first_train_batch = False

        # learning rate tracking
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, (list, tuple)) else opts
        current_lr = opt.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False, on_epoch=True, on_step=False)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] === VAL EPOCH {self.current_epoch} START ===")
        dm = self.trainer.datamodule
        print(f"[ValWindow]   {getattr(dm.hparams, 'val_start', None)} .. {getattr(dm.hparams, 'val_end', None)} "
              f"(sum_id={id(getattr(dm, 'val_data_summary', None))})")
        self._printed_first_val_batch = False

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
        for obs_type, instruments in self.observation_config.items():
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
            cand = self.feature_stats.get(found_in_obs_type, {})
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

    def forward(self, data: HeteroData, step_data_list=None) -> Dict[str, torch.Tensor]:

        num_graphs = data.num_graphs
        num_mesh_nodes = self.mesh_x.shape[0]

        # Inject and batch static mesh data
        data["mesh"].x = self.mesh_x.repeat(num_graphs, 1)
        data["mesh", "to", "mesh"].edge_attr = self.mesh_edge_attr.repeat(num_graphs, 1)

        edge_indices = [self.mesh_edge_index + i * num_mesh_nodes for i in range(num_graphs)]
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
                edge_attr = torch.zeros((edge_index.size(1), self.hidden_dim), device=device)

                encoder = self.observation_encoders[self._edge_key(edge_type)]
                encoder.edge_index = edge_index

                use_edge_attr = getattr(encoder, "expects_edge_attr", False)  # set on init, see below
                edge_rep = None
                if use_edge_attr:
                    edge_rep = data[edge_type].edge_attr if "edge_attr" in data[edge_type] else None

                # --- Debugging ---
                self.debug(f"\n[ENC] edge type: {edge_type}")
                self.debug(f"  send_rep (obs) {obs_features.shape} | rec_rep (mesh) {encoded_mesh_features.shape}")
                self.debug(f"  edge_index {edge_index.shape}")
                # --- End Debugging ---

                encoded_mesh_features = encoder(
                    send_rep=obs_features,
                    rec_rep=encoded_mesh_features,
                    edge_rep=edge_attr,
                )

        # --------------------------------------------------------------------
        # STAGE 3: PREPARE FOR PROCESSOR
        # --------------------------------------------------------------------
        encoded_features = embedded_features
        encoded_features["mesh"] = encoded_mesh_features

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
        return self._forward_latent_rollout(data, encoded_features)

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

        # Get latent step information
        step_info = self._get_latent_step_info(data)
        num_latent_steps = step_info["num_steps"]
        step_mapping = step_info["step_mapping"]
        edge_mapping = self._map_step_edges(data, step_mapping)

        self.debug(f"[LATENT] {num_latent_steps} latent steps detected")
        self.debug(f"[LATENT] Step mapping: {step_mapping}")

        # Initialize predictions dict with lists for each base instrument
        predictions = {}
        for base_type in step_mapping.keys():
            predictions[base_type] = []

        # Initialize mesh state for latent rollout
        current_mesh_features = encoded_features["mesh"]

        # --------------------------------------------------------------------
        # LATENT ROLLOUT LOOP: Sequential processor → decoder steps
        # --------------------------------------------------------------------
        if self.processor_type == "sliding_transformer":
            self.swt.reset()

        for step in range(num_latent_steps):
            self.debug(f"[LATENT] Processing step {step+1}/{num_latent_steps}")
            # --- PROCESS: evolve mesh one step ---
            if self.processor_type == "sliding_transformer":
                current_mesh_features = self.swt(current_mesh_features)
            else:  # interaction processor
                # Remove decoder edges (mesh → target), but keep encoder edges (input → mesh)
                processor_edges = {et: ei for et, ei in data.edge_index_dict.items()
                                   if "_target" not in et[2]}

                # STAGE 4A: PROCESS - Evolve mesh state forward one latent step
                step_features = encoded_features.copy()
                step_features["mesh"] = current_mesh_features
                processed = self.processor(step_features, processor_edges)
                current_mesh_features = processed["mesh"]

            self.debug(f"[LATENT] Step {step} - mesh after processor: {current_mesh_features.shape}")

            # STAGE 4B: DECODE - Generate predictions for this latent step
            mesh_features_processed = current_mesh_features

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
                        self.debug(f"[LATENT] Warning: No edge found for {step_node_type}")
                        continue

                    # === BRANCHING: Check if this is a target variable instrument ===
                    if self._is_target_variable(base_type):
                        # NEW PATH: Target Variable Decoder
                        save_mesh = not self.training and hasattr(self, '_save_mesh_predictions_enabled')

                        if save_mesh:
                            step_prediction, mesh_preds = self._decode_target_variables(
                                base_type=base_type,
                                mesh_features=mesh_features_processed,
                                step_node_type=step_node_type,
                                step_edge_index=step_edge_index,
                                data=data,
                                return_mesh_predictions=True,  # Request mesh predictions
                            )
                    
                            # Store mesh predictions for later saving
                            if not hasattr(self, '_mesh_predictions_buffer'):
                                self._mesh_predictions_buffer = {}
                            if base_type not in self._mesh_predictions_buffer:
                                self._mesh_predictions_buffer[base_type] = []
                            self._mesh_predictions_buffer[base_type].append(mesh_preds)
                        else:
                            step_prediction = self._decode_target_variables(
                                base_type=base_type,
                                mesh_features=mesh_features_processed,
                                step_node_type=step_node_type,
                                step_edge_index=step_edge_index,
                                data=data,
                                return_mesh_predictions=False,
                            )

                    else:
                        # Get the decoder (mapped to base instrument)
                        decoder_key = edge_mapping.get(step_edge_type)
                        if decoder_key not in self.observation_decoders:
                            self.debug(f"[LATENT] Warning: No decoder found for {decoder_key}")
                            continue

                        decoder = self.observation_decoders[decoder_key]
                        decoder.edge_index = step_edge_index

                        # Decode mesh features to target predictions
                        # Use device from mesh features to avoid checkpoint loading issues
                        reference_device = mesh_features_processed.device
                        target_features_initial = torch.zeros(data[step_node_type].num_nodes, self.hidden_dim, device=reference_device)
                        edge_attr = torch.zeros((step_edge_index.size(1), self.hidden_dim), device=reference_device)

                        decoded_target_features = decoder(
                            send_rep=mesh_features_processed,
                            rec_rep=target_features_initial,
                            edge_rep=edge_attr,
                        )

                        # Apply scan angle embedding if needed
                        scan_angle = data[step_node_type].x

                        if base_type == "ascat_target":
                            scan_angle_embedded = self.ascat_scan_angle_embedder(scan_angle)
                            final_features = torch.cat([decoded_target_features, scan_angle_embedded], dim=-1)
                            step_prediction = self.output_mappers[base_type](final_features)
                        elif base_type in ("atms_target", "amsua_target", "avhrr_target"):
                            scan_angle_embedded = self.scan_angle_embedder(scan_angle)
                            final_features = torch.cat([decoded_target_features, scan_angle_embedded], dim=-1)
                            step_prediction = self.output_mappers[base_type](final_features)
                        else:
                            step_prediction = self.output_mappers[base_type](decoded_target_features)

                    # Store prediction for this step
                    predictions[base_type].append(step_prediction)
                    print(f"predict: [node_type] {base_type}: {step_prediction.shape}")

                    self.debug(f"[LATENT] Step {step} - {base_type}: {step_prediction.shape}")

        # Verify all instruments have correct number of predictions
        for base_type, pred_list in predictions.items():
            expected_steps = len(step_mapping[base_type])
            if len(pred_list) != expected_steps:
                self.debug(f"[LATENT] Warning: {base_type} has {len(pred_list)} predictions, expected {expected_steps}")

        self.debug(f"[LATENT] Completed {num_latent_steps} sequential processor steps")
        return predictions

    def get_current_rollout_steps(self):
        """
        Determines the current number of rollout steps based on training progress.
        Implements curriculum learning where rollout length increases over time.
        """
        if not hasattr(self, "max_rollout_steps"):
            return 1  # Default to single step

        if not hasattr(self, "rollout_schedule"):
            return self.max_rollout_steps

        current_epoch = self.current_epoch
        current_step = self.global_step  # This tracks gradient descent updates

        if self.rollout_schedule == "graphcast":
            # GraphCast schedule based on gradient descent updates
            # Graphcast: 300,000 gradient descent updates - 1 autoregressive
            #            300,001 to 311,000: add 1 per 1000 updates
            #           (i.e., use 1000 steps for each autoregressive step)
            # testing functionality: train 1 rollout for 5 epochs [0-4], add 1 for every epoch
            threshold = 5  # 300000 # MK: using 5 for testing
            interval = 1  # 1000
            if current_step < threshold:
                return 1
            else:
                additional_steps = 2 + (current_step - threshold) // interval
                return min(additional_steps, self.max_rollout_steps)

        elif self.rollout_schedule == "linear":
            # Linearly increase from 1 to max_rollout_steps over training
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
            progress = min(current_epoch / max_epochs, 1.0)
            current_steps = 1 + int(progress * (self.max_rollout_steps - 1))
            return current_steps

        elif self.rollout_schedule == "step":
            # Step-wise increase (GraphCast style)
            if current_epoch < 10:
                return 1
            elif current_epoch < 20:
                return 2
            else:
                return min(self.max_rollout_steps, 3 + (current_epoch - 20) // 10)

        else:
            # "fixed"
            return self.max_rollout_steps

    def _get_latent_step_info(self, data: HeteroData) -> dict:
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

    def _map_step_edges(self, data: HeteroData, step_mapping: dict) -> dict:
        """
        Create mapping from step-specific edges to base decoder keys.
        Returns dict mapping step edges to decoder keys.
        """
        edge_mapping = {}

        for edge_type in data.edge_index_dict.keys():
            src_type, rel, dst_type = edge_type
            if "_target_step" in dst_type and src_type == "mesh":
                # Find the base target type for this step
                for base_type, steps in step_mapping.items():
                    for step_num, step_node_type in steps.items():
                        if step_node_type == dst_type:
                            base_edge_key = self._edge_key(("mesh", "to", base_type))
                            edge_mapping[edge_type] = base_edge_key
                            break

        return edge_mapping

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

    def training_step(self, batch, batch_idx):
        print("[DIAG] Entered training_step()")
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        # Print first-batch info for window validation
        if not getattr(self, "_printed_first_train_batch", False):
            bt = getattr(batch, "input_time", None) or getattr(batch, "time", None)
            print(f"[FirstTrainBatch] batch_idx=0 time={bt}")
            self._printed_first_train_batch = True

        print(f"[training_step] batch: {getattr(batch, 'bin_name', 'N/A')}")

        # ---- Forward pass and loss calculation ----
        all_predictions = self(batch)

        # Extract ground truths based on rollout mode
        ground_truth_data = self._extract_ground_truths_and_metadata(batch, all_predictions)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_predictions = 0

        # Calculate loss for each observation type and add it to the total
        for node_type, preds_list in all_predictions.items():
            if node_type not in ground_truth_data:
                continue

            # Get the base instrument name (e.g., "atms" from "atms_target")
            # Add handling for target_step in latent mode
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            inst_id = self.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            instrument_ids_list = gt_data["instrument_ids_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            for step, (y_pred, y_true, instrument_ids, valid_mask) in enumerate(
                zip(preds_list, gts_list, instrument_ids_list, valid_mask_list)
            ):
                # Skip if either prediction or ground truth is None or empty
                if y_pred is None or y_true is None or y_pred.numel() == 0 or y_true.numel() == 0:
                    continue

                # Ensure finite tensors
                if not torch.isfinite(y_pred).all():
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(y_true).all():
                    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

                # Skip if mask exists but nothing valid
                if valid_mask is not None and valid_mask.sum() == 0:
                    continue

                # Shape validation before loss calculation
                if y_pred.shape[0] != y_true.shape[0]:
                    print(f"[ERROR] Shape mismatch for {node_type} step {step}:")
                    print(f"  y_pred: {y_pred.shape} ({y_pred.shape[0]} obs)")
                    print(f"  y_true: {y_true.shape} ({y_true.shape[0]} obs)")
                    print(f"  Skipping this prediction to avoid crash")
                    continue

                channel_loss = weighted_huber_loss(
                    y_pred,
                    y_true,
                    instrument_ids=instrument_ids,
                    channel_weights=self.channel_weights,  # dict keyed by int ids
                    delta=0.1,
                    rebalancing=True,
                    valid_mask=valid_mask,
                )

                if not torch.isfinite(channel_loss):
                    if self.trainer.is_global_zero:
                        print(f"[WARN] Non-finite channel_loss for {node_type} at step {step}; skipping this term.")
                    continue

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                # Add the loss for this instrument to the total
                total_loss = total_loss + weighted_loss
                num_predictions += 1

        dummy_loss = 0.0
        for param in self.parameters():
            dummy_loss += param.sum() * 0.0
        # Average the loss over all observation types that had predictions
        avg_loss = total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)
        avg_loss = avg_loss + dummy_loss

        # Log rollout steps appropriately
        step_info = self._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        print(f"[DEBUG] latent rollout steps: {latent_rollout_steps}")

        self.log(
            "train_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log("rollout_steps", float(latent_rollout_steps), on_step=True, sync_dist=False)
        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[TRAIN] Epoch {self.current_epoch} - train_loss: {avg_loss.cpu().item():.6f}")

        return avg_loss

    def validation_step(self, batch, batch_idx):
        print(f"VALIDATION STEP batch: {batch.bin_name}")

        if batch_idx == 0:
            self._save_mesh_predictions_enabled = True
            self._mesh_predictions_buffer = {}
            print(f"[MESH PRED] Enabled for epoch {self.current_epoch}, batch {batch_idx}")
        else:
            self._save_mesh_predictions_enabled = False

        # Build decoder names from config (all possible node_types with targets)
        decoder_names = [f"{inst_name}_target" for obs_type, instruments in self.observation_config.items() for inst_name in instruments]

        # Prepare metrics storage
        all_step_rmse = {name: [] for name in decoder_names}
        all_step_mae = {name: [] for name in decoder_names}
        all_step_bias = {name: [] for name in decoder_names}
        all_losses = []

        # Determine rollout steps based on mode
        step_info = self._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        print(f"[validation_step] latent rollout steps: {latent_rollout_steps}")

        # Forward pass: Dict[node_type, List[Tensor]] per step
        all_predictions = self(batch)
        if isinstance(all_predictions, tuple):
            all_predictions, _ = all_predictions

        # Extract ground truths based on rollout mode
        ground_truth_data = self._extract_ground_truths_and_metadata(batch, all_predictions)

        total_loss = torch.tensor(0.0, device=self.device)
        num_predictions = 0

        # --- Loop over all node_types/decoders ---
        for node_type, preds_list in all_predictions.items():
            print(f"[validation_step] Processing node_type: {node_type}")
            if node_type not in ground_truth_data:
                continue

            feats = None
            # Latent mode: target_step0, target_step1, etc
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            inst_id = self.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            instrument_ids_list = gt_data["instrument_ids_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            n_steps = min(len(preds_list), len(gts_list))

            for step, (y_pred, y_true, instrument_ids, valid_mask) in enumerate(
                zip(preds_list, gts_list, instrument_ids_list, valid_mask_list)
            ):
                print(f"[validation_step] {node_type} - step {step+1}/{n_steps}")
                # Skip if either prediction or ground truth is None or empty
                if y_pred is None or y_true is None or y_pred.numel() == 0 or y_true.numel() == 0:
                    continue
                if y_pred.shape != y_true.shape:
                    continue

                if not torch.isfinite(y_pred).all():
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(y_true).all():
                    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

                if valid_mask is not None:
                    valid_mask = valid_mask.to(dtype=torch.bool, device=y_pred.device)
                    if valid_mask.numel() == 0 or valid_mask.sum() == 0:
                        continue  # nothing valid for this node_type/step

                # Get the channel-weighted loss
                channel_loss = weighted_huber_loss(
                    y_pred,
                    y_true,
                    instrument_ids=instrument_ids,
                    channel_weights=self.channel_weights,
                    delta=0.1,
                    rebalancing=True,
                    valid_mask=valid_mask,
                )

                if not torch.isfinite(channel_loss):
                    if self.trainer.is_global_zero:
                        print(f"[WARN] Non-finite channel_loss for {node_type} at step {step}; skipping this term.")
                    continue

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                total_loss = total_loss + weighted_loss
                num_predictions += 1
                self.log(
                    f"val_loss_{node_type}",
                    weighted_loss.detach(),
                    sync_dist=False,
                    on_epoch=True,
                    batch_size=1,
                    prog_bar=False,
                    logger=True,
                    rank_zero_only=True,
                )

                # --- Metrics Calculation ---
                y_pred_unnorm = self.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.unnormalize_standardscaler(y_true, node_type)

                if valid_mask is not None:
                    # reduce only over valid elements
                    vm = valid_mask
                    # RMSE
                    mse_elems = (y_pred_unnorm - y_true_unnorm).pow(2)
                    rmse = torch.sqrt((mse_elems[vm]).mean() + 1e-12)
                    # MAE
                    mae = (y_pred_unnorm - y_true_unnorm).abs()
                    mae = (mae[vm]).mean()
                    # Bias
                    bias = y_pred_unnorm - y_true_unnorm
                    bias = (bias[vm]).mean()

                    # Keep per-channel vectors to match the logging format
                    # (compute channelwise means with masking)
                    # shape handling:
                    vm_f = vm.float()
                    denom_ch = vm_f.sum(dim=0).clamp_min(1.0)
                    rmse_ch = torch.sqrt((mse_elems * vm_f).sum(dim=0) / denom_ch + 1e-12)
                    mae_ch = (mae := ((y_pred_unnorm - y_true_unnorm).abs() * vm_f).sum(dim=0) / denom_ch)
                    bias_ch = ((y_pred_unnorm - y_true_unnorm) * vm_f).sum(dim=0) / denom_ch

                    step_rmse = rmse_ch
                    step_mae = mae_ch
                    step_bias = bias_ch
                else:
                    step_rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
                    step_mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
                    step_bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)

                all_step_rmse[node_type].append(step_rmse)
                all_step_mae[node_type].append(step_mae)
                all_step_bias[node_type].append(step_bias)

                if (
                    self.trainer.is_global_zero  # only main process
                    and step == 0  # only concatenate latent rollout once
                    and batch_idx == 0  # only first batch
                ):
                    # --- CSV save block ---
                    out_dir = "val_csv"
                    os.makedirs(out_dir, exist_ok=True)

                    # LATENT ROLLOUT: Concatenate all steps into standard format
                    self._save_latent_concatenated_csv(
                        batch, node_type, preds_list, gts_list,
                        valid_mask_list, out_dir, batch_idx
                    )

            # Placeholder logging for missing steps (to ensure stable CSV shape for loggers)
            num_channels = all_step_rmse[node_type][0].shape[0] if all_step_rmse[node_type] else 1
            for step in range(n_steps, self.max_rollout_steps):
                placeholder_metric = torch.tensor(float("nan"), device=self.device)

        # --- Average metrics across steps for each node_type ---
        for node_type in decoder_names:
            if all_step_rmse[node_type]:
                avg_rmse = torch.stack(all_step_rmse[node_type]).mean(dim=0)
                avg_mae = torch.stack(all_step_mae[node_type]).mean(dim=0)
                avg_bias = torch.stack(all_step_bias[node_type]).mean(dim=0)

        if self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if all_step_rmse[node_type]:
                    print(f"[VAL] {node_type} RMSE (avg): {torch.stack(all_step_rmse[node_type]).mean().item():.4f}")

        if self.verbose and self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if node_type not in all_predictions or not all_predictions[node_type]:
                    continue
                y_pred = all_predictions[node_type][0]
                y_true = ground_truth_data[node_type]["gts_list"][0]
                y_pred_unnorm = self.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.unnormalize_standardscaler(y_true, node_type)

                n_channels = y_pred_unnorm.shape[1]
                for i in range(min(5, n_channels)):
                    try:
                        plt.figure()
                        # Get data and remove any NaN/inf values
                        y_true_data = y_true_unnorm[:, i].cpu().numpy()
                        y_pred_data = y_pred_unnorm[:, i].cpu().numpy()

                        # Filter out non-finite values
                        y_true_finite = y_true_data[np.isfinite(y_true_data)]
                        y_pred_finite = y_pred_data[np.isfinite(y_pred_data)]

                        # Skip if no finite data
                        if len(y_true_finite) == 0 or len(y_pred_finite) == 0:
                            plt.close()
                            continue

                        # Use auto bins or limit to reasonable number
                        n_bins = min(50, max(10, len(y_true_finite) // 20))

                        plt.hist(
                            y_true_finite,
                            bins=n_bins,
                            alpha=0.6,
                            color="blue",
                            label="y_true",
                        )
                        plt.hist(
                            y_pred_finite,
                            bins=n_bins,
                            alpha=0.6,
                            color="orange",
                            label="y_pred",
                        )
                        plt.xlabel(f"{node_type} - Channel {i+1}")
                        plt.ylabel("Frequency")
                        plt.title(f"Histogram - {node_type} Channel {i+1} (Epoch {self.current_epoch})")
                        plt.legend()
                        plt.tight_layout()
                    except Exception as e:
                        print(f"Warning: Could not create histogram for {node_type} channel {i+1}: {e}")
                        plt.close()
                        continue
                    plt.savefig(f"hist_{node_type}_ch_{i+1}_epoch{self.current_epoch}.png")
                    plt.close()

        # --- Final loss calculation for the entire validation step ---
        avg_loss = total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)

        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        if self.trainer.is_global_zero:
            print(f"--- Epoch {self.current_epoch} Validation ---")
            print(f"val_loss: {avg_loss.item():.6f}")

        # Save mesh predictions if collected
        if hasattr(self, '_mesh_predictions_buffer') and self._mesh_predictions_buffer:
            if batch_idx == 0 and self.trainer.is_global_zero:
                self._save_mesh_predictions(
                    self._mesh_predictions_buffer,
                    batch_idx,
                    self.current_epoch
                )
            
            # Clear buffer
            self._mesh_predictions_buffer = {}

        # Always disable and cleanup
        if hasattr(self, '_save_mesh_predictions_enabled'):
            self._save_mesh_predictions_enabled = False

        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference mode.

        This method:
        1. Runs forward pass to generate predictions
        2. Saves predictions to CSV files
        3. Does NOT compute loss or gradients

        Args:
            batch: Input batch data
            batch_idx: Batch index

        Returns:
            dict: Predictions for all node types
        """
        print(f"[PREDICT] Processing batch {batch_idx}: {batch.bin_name}")

        # Forward pass
        all_predictions = self(batch)
        if isinstance(all_predictions, tuple):
            all_predictions, _ = all_predictions

        # Extract ground truths and metadata
        ground_truth_data = self._extract_ground_truths_and_metadata(batch, all_predictions)

        # Determine rollout steps
        step_info = self._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        print(f"[PREDICT] Latent rollout steps: {latent_rollout_steps}")

        # Save predictions for each instrument
        for node_type, preds_list in all_predictions.items():
            print(f"[PREDICT] Processing node_type: {node_type}")

            if node_type not in ground_truth_data:
                continue

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            # Save to CSV (first 10 batches)
            # if batch_idx < 10:
            self._save_prediction_csv(
                batch=batch,
                node_type=node_type,
                preds_list=preds_list,
                gts_list=gts_list,
                valid_mask_list=valid_mask_list,
                batch_idx=batch_idx,
                mode='predict'
            )

        # Save mesh predictions (target variables on grid)
        if hasattr(self, '_mesh_predictions_buffer') and self._mesh_predictions_buffer:
            self._save_mesh_predictions(
                self._mesh_predictions_buffer,
                batch_idx,
                epoch=0,
                mode='predict',
                batch=batch,
            )
            self._mesh_predictions_buffer = {}

        return all_predictions


    def on_predict_epoch_start(self):
        """Setup before prediction epoch starts."""
        print("[PREDICT] Starting prediction epoch")
        self._save_mesh_predictions_enabled = True
        self._mesh_predictions_buffer = {}
        self._prediction_output_dir = getattr(self, 'prediction_output_dir', 'predictions')
        os.makedirs(self._prediction_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self._prediction_output_dir, 'non-target'), exist_ok=True)
        os.makedirs(os.path.join(self._prediction_output_dir, 'target'), exist_ok=True)
        print(f"[PREDICT] Output directory: {self._prediction_output_dir}")


    def on_predict_batch_end(self, outputs, batch, batch_idx):
        """Cleanup after each prediction batch."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def on_predict_epoch_end(self):
        """Cleanup and summary after prediction epoch ends."""
        print("[PREDICT] Prediction epoch completed")

        if hasattr(self, '_save_mesh_predictions_enabled'):
            self._save_mesh_predictions_enabled = False

        # Generate summary statistics
        if hasattr(self, '_prediction_output_dir'):
            obs_dir = os.path.join(self._prediction_output_dir, 'non-target')
            csv_files = [f for f in os.listdir(obs_dir) if f.endswith('.csv')]
            print(f"[PREDICT] Generated {len(csv_files)} observation CSV files")

            mesh_dir = os.path.join(self._prediction_output_dir, 'target')
            mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.csv')]
            print(f"[PREDICT] Generated {len(mesh_files)} mesh CSV files")

    def _extract_init_time_str(self, batch):
        """
        Extract initialization time string from batch in YYYYMMDDHH format.
        Handles multiple formats: pandas Timestamp, list/tuple, Unix timestamp, tensor.
        
        Args:
            batch: Batch data containing input_time or time attribute
            
        Returns:
            str: Init time as 'YYYYMMDDHH' or 'unknown' if unavailable
        """
        from datetime import datetime
        import pandas as pd

        if batch is None:
            return 'unknown'

        # Try input_time first
        ts = None
        if hasattr(batch, 'input_time') and batch.input_time is not None:
            # Handle if input_time is a list/tuple - take the first element
            if isinstance(batch.input_time, (list, tuple)):
                ts = batch.input_time[0] if len(batch.input_time) > 0 else None
            else:
                ts = batch.input_time

        # Fallback to batch.time if input_time not available
        elif hasattr(batch, 'time') and batch.time is not None:
            if isinstance(batch.time, (list, tuple)):
                ts = batch.time[0] if len(batch.time) > 0 else None
            else:
                ts = batch.time

        # Now convert ts to string based on its type
        if ts is None:
            return 'unknown'

        try:
            # Handle pandas Timestamp
            if isinstance(ts, pd.Timestamp):
                return ts.strftime('%Y%m%d%H')

            # Handle Unix timestamp (float/int)
            elif isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
                return dt.strftime('%Y%m%d%H')

            # Handle tensor (PyTorch/numpy with .item() method)
            elif hasattr(ts, 'item'):
                dt = datetime.fromtimestamp(ts.item())
                return dt.strftime('%Y%m%d%H')

            # Handle datetime object directly
            elif isinstance(ts, datetime):
                return ts.strftime('%Y%m%d%H')

            else:
                print(f"[INIT_TIME] Warning: Unsupported time type: {type(ts)}")
                return 'unknown'

        except Exception as e:
            print(f"[INIT_TIME] Error converting time: {e}, type: {type(ts)}")
            return 'unknown'

    def _save_prediction_csv(self, batch, node_type, preds_list, gts_list,
                             valid_mask_list, batch_idx, mode='predict'):
        """
        Save predictions to CSV file.

        Args:
            batch: Input batch
            node_type: Type of node (e.g., 'atms_target')
            preds_list: List of prediction tensors for each step
            gts_list: List of ground truth tensors (may be None)
            valid_mask_list: List of validity masks
            batch_idx: Batch index
            mode: 'predict' or 'val'
        """
        step_info = self._get_latent_step_info(batch)
        step_mapping = step_info["step_mapping"]

        # Set output directory
        if mode == 'predict':
            out_dir = os.path.join(self._prediction_output_dir, 'non-target')
        else:
            out_dir = "val_csv"

        os.makedirs(out_dir, exist_ok=True)

        # Collect all observations from all steps
        all_lat = []
        all_lon = []
        all_pred = []
        all_true = []
        all_mask = []

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
            else:
                n = y_pred_unnorm.shape[0]
                lat_deg = np.zeros(n)
                lon_deg = np.zeros(n)
            
            # Collect data from this step
            all_lat.extend(lat_deg)
            all_lon.extend(lon_deg)
            all_pred.append(y_pred_unnorm.detach().cpu().numpy())
            all_true.append(y_true_unnorm.detach().cpu().numpy())
            
            if valid_mask is not None:
                all_mask.append(valid_mask.detach().cpu().numpy().astype(bool))
            else:
                all_mask.append(np.ones_like(y_pred_unnorm.detach().cpu().numpy(), dtype=bool))

        if not all_pred:
            print(f"[WARN] No valid predictions for {node_type}, skipping CSV save")
            return

        # Concatenate all steps
        all_pred_concat = np.vstack(all_pred)
        all_true_concat = np.vstack(all_true)
        all_mask_concat = np.vstack(all_mask)

        # Skip saving if no real ground truth data (inference mode)
        if all_true_concat.size == 0 or np.sum(np.abs(all_true_concat)) == 0:
            print(f"[PREDICT] latent csv: Skipping {node_type} - no ground truth data (inference mode)")
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

        for i, fname in enumerate(feats):
            col = _safe_col_name(fname)
            df[f"pred_{col}"] = all_pred_concat[:, i]
            df[f"true_{col}"] = all_true_concat[:, i]
            df[f"mask_{col}"] = all_mask_concat[:, i]

        # Extract init time for filename
        init_time_str = self._extract_init_time_str(batch)

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

    def _save_latent_concatenated_csv(self, batch, node_type, preds_list, gts_list,
                                      valid_mask_list, out_dir, batch_idx):
        """Save latent rollout as concatenated observations (validation mode)."""
        self._save_prediction_csv(
            batch=batch,
            node_type=node_type,
            preds_list=preds_list,
            gts_list=gts_list,
            valid_mask_list=valid_mask_list,
            batch_idx=batch_idx,
            mode='val'
        )

    def _save_mesh_predictions(self, predictions_buffer, batch_idx, epoch, mode='val', batch=None):
        """Save predictions on mesh grid - one file per forecast hour."""
        
        # os.makedirs('mesh_predictions', exist_ok=True)
        if mode == 'predict':
            output_dir = os.path.join(self._prediction_output_dir, 'target')
        else:
            output_dir = 'val_pred_csv'
        os.makedirs(output_dir, exist_ok=True)

        # Get mesh coordinates
        mesh_lats = self.mesh_lats.cpu().numpy()
        mesh_lons = self.mesh_lons.cpu().numpy()

        # Get Input Time
        init_time_str = self._extract_init_time_str(batch)

        # Convert radians to degrees if needed
        if mesh_lats.max() <= np.pi:  # Likely in radians
            mesh_lats = np.degrees(mesh_lats)
            mesh_lons = np.degrees(mesh_lons)

        for inst_name, step_predictions in predictions_buffer.items():
            # Remove '_target' suffix to get base instrument name
            base_inst_name = inst_name.replace('_target', '')
            
            # Get target variables
            target_vars = self.target_variable_config.get('variables', {}).get(base_inst_name, [])
            
            if not target_vars:
                print(f"[MESH PRED] No target variables for {base_inst_name}, skipping")
                continue

            # Calculate forecast hours
            num_steps = len(step_predictions)
            latent_step_hours = self.hparams.get('latent_step_hours', 3)
            forecast_hours = [(i + 1) * latent_step_hours for i in range(num_steps)]
            
            print(f"[MESH PRED] Saving {base_inst_name}: {forecast_hours}h forecast")

            for step_idx, (mesh_preds_dict, fhr) in enumerate(zip(step_predictions, forecast_hours)):
                # Create DataFrame with coordinates
                df = pd.DataFrame({
                    'lat': mesh_lats,
                    'lon': mesh_lons,
                })

                # Add each target variable's mesh prediction
                for var_name, pred_tensor in mesh_preds_dict.items():
                    # Unnormalize if needed
                    pred_np = pred_tensor.detach().cpu().numpy().squeeze()  # [N_mesh]

                    # ADD UNNORMALIZATION
                    # Get stats for this variable
                    if base_inst_name in self.feature_stats:
                        stats_block = self.feature_stats[base_inst_name]
                        if var_name in stats_block:
                            mean, std = stats_block[var_name]
                            # Unnormalize: value = normalized * std + mean
                            pred_np = pred_np * std + mean
                            print(f"[MESH PRED] Unnormalized {var_name}: range [{pred_np.min():.2f}, {pred_np.max():.2f}]")
                        else:
                            print(f"[MESH PRED] WARNING: No stats found for {base_inst_name}.{var_name}, saving normalized")
                    else:
                        print(f"[MESH PRED] WARNING: No feature_stats for {base_inst_name}, saving normalized")

                    df[f'pred_{var_name}'] = pred_np

                # Use init_time if available, otherwise fall back to batch_idx
                if init_time_str != 'unknown':
                    if mode == 'predict':
                        filepath = f'{output_dir}/{base_inst_name}_init_{init_time_str}_f{fhr:03d}.csv'
                    else:
                        filepath = f'{output_dir}/{base_inst_name}_init_{init_time_str}_f{fhr:03d}_epoch{epoch}_batch{batch_idx}.csv'
                else:
                    filepath = f'{output_dir}/{base_inst_name}_f{fhr:03d}_epoch{epoch}_batch{batch_idx}.csv'
                df.to_csv(filepath, index=False)
                print(f"[MESH PRED] Saved {filepath}: {len(df)} mesh points, {len(target_vars)} variables")

    def on_after_backward(self):
        # Check if encoded gradient is available
        if hasattr(self, "_encoded_ref"):
            if self._encoded_ref is not None:
                if self._encoded_ref.grad is not None:
                    self.debug(f"[DEBUG] encoded.grad norm: {self._encoded_ref.grad.norm().item():.6f}")
                else:
                    self.debug("[DEBUG] encoded.grad is still None after backward.")
            else:
                self.debug("[DEBUG] _encoded_ref is None")

        # x_hidden grad
        if hasattr(self, "_x_hidden_ref"):
            if self._x_hidden_ref is not None and self._x_hidden_ref.grad is not None:
                self.debug(f"[DEBUG] x_hidden.grad norm: {self._x_hidden_ref.grad.norm().item():.6f}")
            else:
                self.debug("[DEBUG] x_hidden.grad is still None after backward.")

        # Print all parameter gradients
        if self.trainer.is_global_zero:
            total_grad_norm = 0.0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm(2)
                    self.debug(f"[DEBUG] Grad for {name}: {norm:.6f}")
                    total_grad_norm += norm.item() ** 2
                else:
                    self.debug(f"[DEBUG] Grad for {name}: None")
            total_grad_norm = total_grad_norm**0.5
            self.debug(f"[DEBUG] Total Gradient Norm: {total_grad_norm:.6f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        # This scheduler monitors the validation loss and reduces the LR when it plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Reduce LR when the monitored metric has stopped decreasing
            factor=0.5,  # new_lr = lr * factor (conservative decay; lower factor is more aggressive)
            patience=3,  # Number of epochs with no improvement after which LR will be reduced
            verbose=True,  # Print a message when the LR is changed
            min_lr=1e-6,  # safeguard against vanishing lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # The metric to monitor
                "interval": "epoch",
                "frequency": 1,
            },
        }
