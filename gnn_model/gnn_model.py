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

        first_instrument_config = next(iter(next(iter(observation_config.values())).values()))
        hidden_layers = first_instrument_config.get("encoder_hidden_layers", 2)

        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.mesh_embedder = make_mlp([mesh_feature_dim] + self.mlp_blueprint_end)

        # Create scan-angle embedders once to avoid loop-order surprises
        self.scan_angle_embed_dim = 8
        self.scan_angle_embedder = make_mlp([1, self.scan_angle_embed_dim])
        self.ascat_scan_angle_embedder = make_mlp([3, self.scan_angle_embed_dim])

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
        return avg_loss

    def _save_latent_concatenated_csv(self, batch, node_type, preds_list, gts_list,
                                      valid_mask_list, out_dir, batch_idx):
        """Save latent rollout as concatenated observations (same format as standard)."""

        step_info = self._get_latent_step_info(batch)
        step_mapping = step_info["step_mapping"]

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
        all_pred_concat = np.vstack(all_pred)  # Shape: (total_obs, n_ch)
        all_true_concat = np.vstack(all_true)  # Shape: (total_obs, n_ch)
        all_mask_concat = np.vstack(all_mask)  # Shape: (total_obs, n_ch)

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

        # Save with same filename format as standard rollout
        filename = f"{out_dir}/val_{node_type}_epoch{self.current_epoch}_batch{batch_idx}_step0.csv"
        df.to_csv(filename, index=False)
        print(f"Saved latent concatenated CSV: {filename}")
        print(f"  Total observations from all steps: {len(df)}")
        print(f"  Steps combined: {len(all_pred)}")

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
