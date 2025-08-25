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
from typing import Dict, Tuple
from torch_geometric.utils import scatter
from loss import weighted_huber_loss


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
        self.channel_masks = {}
        for inst_id, weights in self.channel_weights.items():
            if isinstance(weights, torch.Tensor):
                weights_tensor = weights.clone().detach()
            else:
                weights_tensor = torch.tensor(weights)
            mask = weights_tensor > 0
            self.channel_masks[int(inst_id)] = mask
        self.max_rollout_steps = max_rollout_steps
        self.rollout_schedule = rollout_schedule

        self.observation_config = observation_config
        self.instrument_name_to_id = {"atms": 0, "surface_obs": 1, "radiosonde": 2}
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

        node_types = ["mesh"]
        edge_types = [("mesh", "to", "mesh")]

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
                self.observation_encoders[self._edge_key(edge_type_tuple_enc)] = InteractionNet(
                    edge_index=None,
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )

                # Decoder GNN (mesh -> target)
                edge_type_tuple_dec = ("mesh", "to", node_type_target)
                self.observation_decoders[self._edge_key(edge_type_tuple_dec)] = InteractionNet(
                    edge_index=None,
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )

                # Initial MLP to project raw features to hidden_dim
                self.observation_embedders[node_type_input] = make_mlp([input_dim] + self.mlp_blueprint_end)

                # Final MLP to map from hidden dim to output dim
                if node_type_target == "atms_target":
                    self.scan_angle_embed_dim = 8
                    self.scan_angle_embedder = make_mlp([1, self.scan_angle_embed_dim])
                    input_dim_for_mapper = hidden_dim + self.scan_angle_embed_dim
                else:
                    input_dim_for_mapper = hidden_dim

                # Add one more hidden layer to the output mapper as requested
                output_map_layers = [input_dim_for_mapper] + [hidden_dim] * hidden_layers + [target_dim]
                self.output_mappers[node_type_target] = make_mlp(output_map_layers, layer_norm=False)

        self.processor = Processor(
            hidden_dim=hidden_dim,
            node_types=node_types,
            edge_types=edge_types,
            num_message_passing_steps=num_layers,
        )

        def _as_f32(x):
            import torch

            return x.clone().detach().to(torch.float32) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

        def _as_i64(x):
            import torch

            return x.clone().detach().to(torch.long) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

        self.register_buffer("mesh_x", _as_f32(mesh_x))
        self.register_buffer("mesh_edge_index", _as_i64(mesh_edge_index))
        self.register_buffer("mesh_edge_attr", _as_f32(mesh_edge_attr))

    def _feature_names_for_node(self, node_type: str):
        """Return ordered feature names for this target node."""
        inst_name = node_type.replace("_target", "")
        for obs_type, instruments in self.observation_config.items():
            if inst_name in instruments:
                return instruments[inst_name].get("features", None)
        return None

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
        if self.trainer.is_global_zero:
            print(f"=== Starting Epoch {self.current_epoch} ===")

        # Call set_epoch only if the distributed group is initialized
        if is_initialized():
            train_loaders = self.trainer.train_dataloader
            if isinstance(train_loaders, DataLoader):
                train_loaders = [train_loaders]
            for loader in train_loaders:
                if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.current_epoch)

    def unnormalize_standardscaler(self, tensor, node_type, mean=None, std=None):
        """
        Undo standardization (z-score) for predictions/targets.

        - If node_type is "atms_target": uses bt_channel_1..22 from feature_stats.
        - If node_type is "surface_obs_target": uses feature order defined in observation_config["surface_obs"]["features"].
        - Otherwise raises ValueError.
        """
        if node_type == "atms_target":
            features = [f"bt_channel_{i}" for i in range(1, 23)]
            mean_vec = torch.tensor(
                [self.feature_stats[f][0] for f in features],
                dtype=torch.float32,
                device=self.device,
            )
            std_vec = torch.tensor(
                [self.feature_stats[f][1] for f in features],
                dtype=torch.float32,
                device=self.device,
            )
            return tensor * std_vec + mean_vec

        elif node_type == "surface_obs_target":
            # Find the configured features for surface_obs
            feats = None
            for obs_type, instruments in self.observation_config.items():
                if "surface_obs" in instruments:
                    feats = instruments["surface_obs"]["features"]
                    break
            if feats is None:
                raise ValueError("surface_obs config with 'features' not found for unnormalization")

            # Build mean/std vectors for the configured order
            mean_vec = torch.tensor(
                [self.feature_stats[f][0] for f in feats],
                dtype=torch.float32,
                device=self.device,
            )
            std_vec = torch.tensor(
                [self.feature_stats[f][1] for f in feats],
                dtype=torch.float32,
                device=self.device,
            )

            # Broadcast to prediction shape if needed
            return tensor * std_vec + mean_vec

        else:
            raise ValueError(f"Un-normalization not supported for node_type: {node_type}")

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
            if dst_type == "mesh" and src_type != "mesh":  # This is an obs -> mesh edge
                obs_features = embedded_features[src_type]
                edge_attr = torch.empty((edge_index.size(1), self.hidden_dim), device=self.device)

                encoder = self.observation_encoders[self._edge_key(edge_type)]
                encoder.edge_index = edge_index

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
            if node_type not in encoded_features:
                if node_type in data.node_types:
                    num_nodes = data[node_type].num_nodes
                    encoded_features[node_type] = torch.zeros(num_nodes, self.hidden_dim, device=self.device)

        # --------------------------------------------------------------------
        # STAGE 4: PROCESS (Deep message passing on the graph)
        # --------------------------------------------------------------------
        processed_features = self.processor(encoded_features, data.edge_index_dict)

        self.debug(f"[PROCESSOR] mesh after {self.hparams.num_layers} layers -> {processed_features['mesh'].shape}")

        # --------------------------------------------------------------------
        # STAGE 5: DECODE (Identical to Ocelot2's `decode` method)
        # --------------------------------------------------------------------
        predictions = {}
        mesh_features_processed = processed_features["mesh"]

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type == "mesh" and dst_type.endswith("_target"):
                target_features_initial = torch.zeros(data[dst_type].num_nodes, self.hidden_dim, device=self.device)

                decoder = self.observation_decoders[self._edge_key(edge_type)]
                decoder.edge_index = edge_index

                edge_attr = torch.empty((edge_index.size(1), self.hidden_dim), device=self.device)

                decoded_target_features = decoder(
                    send_rep=mesh_features_processed,
                    rec_rep=target_features_initial,
                    edge_rep=edge_attr,
                )

                if dst_type == "atms_target":
                    scan_angle = data[dst_type].x
                    scan_angle_embedded = self.scan_angle_embedder(scan_angle)
                    final_features = torch.cat([decoded_target_features, scan_angle_embedded], dim=-1)
                    predictions[dst_type] = self.output_mappers[dst_type](final_features)
                else:
                    predictions[dst_type] = self.output_mappers[dst_type](decoded_target_features)

        # Wrap predictions in a list to be compatible with rollout logic
        for node_type, pred_tensor in predictions.items():
            predictions[node_type] = [pred_tensor]

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

    def _get_sequential_step_data(self, batch, n_steps):
        decoder_names = [f"{inst_name}_target" for obs_type, instruments in self.observation_config.items() for inst_name in instruments]

        data_module = self.trainer.datamodule
        current_bin_name = batch.bin_name[0] if isinstance(batch.bin_name, list) else batch.bin_name
        bin_num = int(current_bin_name.replace("bin", ""))

        step_data_list = []
        actual_step = 0

        # Choose bins based on mode
        if self.training:
            available_bins = data_module.train_bin_names
            use_padding = False
        else:
            available_bins = data_module.val_bin_names
            use_padding = True

        max_bin_num = max(int(name.replace("bin", "")) for name in available_bins)

        for step in range(n_steps):
            target_bin_num = bin_num + step
            target_bin_name = f"bin{target_bin_num}"
            step_data = {}

            if step == 0:
                print(f"[ROLL DEBUG] batch keys: {list(batch.keys())}")
                for decoder_name in decoder_names:
                    if decoder_name in batch.node_types:
                        print(f"[ROLL DEBUG] batch[{decoder_name}].y shape: {getattr(batch[decoder_name], 'y', None).shape}")
                        step_data[decoder_name] = {
                            "y": batch[decoder_name].y,
                            "x": batch[decoder_name].x,
                            "target_metadata": batch[decoder_name].target_metadata,
                            "instrument_ids": batch[decoder_name].instrument_ids,
                            "target_channel_mask": getattr(batch[decoder_name], "target_channel_mask", None),
                        }
                        step_data[("mesh", "to", decoder_name)] = {
                            "edge_index": batch[("mesh", "to", decoder_name)].edge_index,
                            "edge_attr": batch[("mesh", "to", decoder_name)].edge_attr,
                        }
                if step_data:
                    step_data_list.append(step_data)
                    actual_step += 1

            elif target_bin_num <= max_bin_num and target_bin_name in data_module.data_summary and target_bin_name in available_bins:

                target_bin_data = data_module.data_summary[target_bin_name]
                temp_graph_data = data_module._create_graph_structure(target_bin_data)
                for decoder_name in decoder_names:
                    if decoder_name in temp_graph_data:
                        device = batch[decoder_name].y.device
                        step_data[decoder_name] = {
                            "y": temp_graph_data[decoder_name].y.to(device),
                            "x": temp_graph_data[decoder_name].x.to(device),
                            "target_metadata": temp_graph_data[decoder_name].target_metadata.to(device),
                            "instrument_ids": temp_graph_data[decoder_name].instrument_ids.to(device),
                            "target_channel_mask": (
                                getattr(temp_graph_data[decoder_name], "target_channel_mask", None)
                                if hasattr(temp_graph_data[decoder_name], "target_channel_mask")
                                else None
                            ),
                        }
                        step_data[("mesh", "to", decoder_name)] = {
                            "edge_index": temp_graph_data[("mesh", "to", decoder_name)].edge_index.to(device),
                            "edge_attr": temp_graph_data[("mesh", "to", decoder_name)].edge_attr.to(device),
                        }
                if step_data:
                    step_data_list.append(step_data)
                    actual_step += 1

            else:
                # fallback: repeat last valid, or construct only if batch contains
                if step_data_list and use_padding:
                    step_data_list.append(step_data_list[-1])
                elif not step_data_list:
                    for decoder_name in decoder_names:
                        if decoder_name in batch:
                            step_data[decoder_name] = {
                                "y": batch[decoder_name].y,
                                "scan_angle": batch[decoder_name].scan_angle,
                                "target_metadata": batch[decoder_name].target_metadata,
                            }
                            step_data[("mesh", "to", decoder_name)] = {
                                "edge_index": batch[("mesh", "to", decoder_name)].edge_index,
                                "edge_attr": batch[("mesh", "to", decoder_name)].edge_attr,
                            }
                    if step_data:
                        step_data_list.append(step_data)
                        self.debug(f"Warning: No data for {target_bin_name}, using original batch")
                else:
                    break

        return step_data_list, actual_step

    def training_step(self, batch, batch_idx):
        print("[DIAG] Entered training_step()")
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        current_rollout_steps = self.get_current_rollout_steps()
        print(f"[training_step] batch: {batch.bin_name}")
        step_data_list, actual_rollout_steps = self._get_sequential_step_data(batch, current_rollout_steps)
        print(f"[DEBUG] actual_rollout_steps: {actual_rollout_steps}")

        all_predictions = self(batch, step_data_list=step_data_list)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_predictions = 0

        # Calculate loss for each observation type and add it to the total
        for node_type, preds_list in all_predictions.items():
            # Get the base instrument name (e.g., "atms" from "atms_target")
            inst_name = node_type.replace("_target", "")
            inst_id = self.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gts_list = [step_data.get(node_type, {}).get("y", None) for step_data in step_data_list]

            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts_list)):
                if y_pred is None or y_true is None or y_pred.numel() == 0:
                    continue

                instrument_ids = step_data_list[step].get(node_type, {}).get("instrument_ids", None)
                valid_mask = step_data_list[step].get(node_type, {}).get("target_channel_mask", None)
                channel_loss = weighted_huber_loss(
                    y_pred,
                    y_true,
                    instrument_ids=instrument_ids,
                    channel_weights=self.channel_weights,  # dict keyed by int ids
                    delta=0.1,
                    rebalancing=True,
                    valid_mask=valid_mask,
                )

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                # Add the loss for this instrument to the total
                total_loss = total_loss + weighted_loss
                num_predictions += 1
                # Log the individual loss for debugging
                self.log(
                    f"train_loss_{node_type}",
                    weighted_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=1,
                )

        dummy_loss = 0.0
        for param in self.parameters():
            dummy_loss += param.sum() * 0.0
        # Average the loss over all observation types that had predictions
        avg_loss = total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)
        avg_loss = avg_loss + dummy_loss

        self.log(
            "train_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log("rollout_steps", float(actual_rollout_steps), on_step=True, sync_dist=False)
        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[TRAIN] Epoch {self.current_epoch} - train_loss: {avg_loss.cpu().item():.6f}")

        return avg_loss

    def validation_step(self, batch, batch_idx):
        print(f"VALIDATION STEP batch: {batch.bin_name}")
        current_rollout_steps = self.max_rollout_steps

        # Build decoder names from config (all possible node_types with targets)
        decoder_names = [f"{inst_name}_target" for obs_type, instruments in self.observation_config.items() for inst_name in instruments]

        # Prepare metrics storage
        all_step_rmse = {name: [] for name in decoder_names}
        all_step_mae = {name: [] for name in decoder_names}
        all_step_bias = {name: [] for name in decoder_names}
        all_losses = []

        # Prepare rollout step data
        step_data_list, actual_rollout_steps = self._get_sequential_step_data(batch, current_rollout_steps)
        print(f"[validation_step] current_rollout_steps: {current_rollout_steps}")
        print(f"[validation_step] actual_rollout_steps: {actual_rollout_steps}")

        # Forward pass: Dict[node_type, List[Tensor]] per step
        all_predictions = self(batch, step_data_list=step_data_list)
        if isinstance(all_predictions, tuple):
            all_predictions, _ = all_predictions

        total_loss = torch.tensor(0.0, device=self.device)
        num_predictions = 0

        # --- Loop over all node_types/decoders ---
        for node_type, preds_list in all_predictions.items():
            feats = None
            inst_name = node_type.replace("_target", "")
            inst_id = self.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gts = [step_data.get(node_type, {}).get("y", None) for step_data in step_data_list]

            n_steps = min(len(preds_list), len(gts))

            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts)):
                if y_pred is None or y_true is None or y_pred.numel() == 0:
                    continue
                if y_pred.shape != y_true.shape:
                    continue

                instrument_ids = step_data_list[step].get(node_type, {}).get("instrument_ids", None)
                valid_mask = step_data_list[step].get(node_type, {}).get("target_channel_mask", None)
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

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                total_loss = total_loss + weighted_loss
                num_predictions += 1
                self.log(
                    f"val_loss_{node_type}",
                    weighted_loss,
                    sync_dist=True,
                    on_epoch=True,
                    batch_size=1,
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
                    and step == 0  # only first step of rollout
                    and batch_idx == 0  # only first batch
                ):
                    # --- CSV save block ---
                    out_dir = "val_csv"
                    os.makedirs(out_dir, exist_ok=True)
                    n = y_pred_unnorm.shape[0]
                    n_ch = y_pred_unnorm.shape[1]
                    # Extract lat/lon from target_metadata
                    current_step_data = step_data_list[step][node_type]
                    lat = current_step_data["target_metadata"][:, 0].cpu().numpy()
                    lon = current_step_data["target_metadata"][:, 1].cpu().numpy()
                    lat_deg = np.degrees(lat)
                    lon_deg = np.degrees(lon)
                    # Get feature names and make sure length matches n_ch
                    feats = self._feature_names_for_node(node_type)
                    if not feats:
                        feats = [f"ch{i+1}" for i in range(n_ch)]
                    # guard against mismatch (slice or pad)
                    if len(feats) > n_ch:
                        feats = feats[:n_ch]
                    elif len(feats) < n_ch:
                        feats = feats + [f"ch{i+1}" for i in range(len(feats) + 1, n_ch + 1)]

                    # sanitize any odd names for column safety (optional)
                    def _safe_col_name(s: str) -> str:
                        return str(s).replace(" ", "_")

                    # Build DataFrame
                    df = pd.DataFrame({"lat": lat_deg, "lon": lon_deg})

                    for i, fname in enumerate(feats):
                        col = _safe_col_name(fname)
                        df[f"pred_{col}"] = y_pred_unnorm[:, i].detach().cpu().numpy()
                        df[f"true_{col}"] = y_true_unnorm[:, i].detach().cpu().numpy()

                    # include mask columns when available
                    if valid_mask is not None:
                        vm_cpu = valid_mask.detach().cpu().numpy().astype(bool)
                        for i, fname in enumerate(feats):
                            col = _safe_col_name(fname)
                            df[f"mask_{col}"] = vm_cpu[:, i]

                    filename = f"{out_dir}/val_{node_type}_epoch{self.current_epoch}_batch{batch_idx}_step{step}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved: {filename}")
            # Placeholder logging for missing steps (to ensure stable CSV shape for loggers)
            num_channels = all_step_rmse[node_type][0].shape[0] if all_step_rmse[node_type] else 1
            for step in range(n_steps, self.max_rollout_steps):
                placeholder_metric = torch.tensor(float("nan"), device=self.device)
                for i in range(num_channels):
                    self.log(
                        f"val_rmse_{node_type}_step{step+1}_ch_{i+1}",
                        placeholder_metric,
                        sync_dist=True,
                        on_epoch=True,
                        batch_size=1,
                    )
                    self.log(
                        f"val_mae_{node_type}_step{step+1}_ch_{i+1}",
                        placeholder_metric,
                        sync_dist=True,
                        on_epoch=True,
                        batch_size=1,
                    )
                    self.log(
                        f"val_bias_{node_type}_step{step+1}_ch_{i+1}",
                        placeholder_metric,
                        sync_dist=True,
                        on_epoch=True,
                        batch_size=1,
                    )

        # --- Average metrics across steps for each node_type ---
        for node_type in decoder_names:
            if all_step_rmse[node_type]:
                avg_rmse = torch.stack(all_step_rmse[node_type]).mean(dim=0)
                avg_mae = torch.stack(all_step_mae[node_type]).mean(dim=0)
                avg_bias = torch.stack(all_step_bias[node_type]).mean(dim=0)
                for i in range(avg_rmse.shape[0]):
                    self.log(
                        f"val_rmse_{node_type}_ch_{i+1}",
                        avg_rmse[i].item(),
                        sync_dist=True,
                        on_epoch=True,
                        batch_size=1,
                    )
                    self.log(
                        f"val_mae_{node_type}_ch_{i+1}",
                        avg_mae[i].item(),
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=1,
                    )
                    self.log(
                        f"val_bias_{node_type}_ch_{i+1}",
                        avg_bias[i].item(),
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=1,
                    )

        if self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if all_step_rmse[node_type]:
                    print(f"[VAL] {node_type} RMSE (avg): {torch.stack(all_step_rmse[node_type]).mean().item():.4f}")

        if self.verbose and self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if node_type not in all_predictions or not all_predictions[node_type]:
                    continue
                y_pred = all_predictions[node_type][0]
                y_true = step_data_list[0][node_type]["y"]
                y_pred_unnorm = self.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.unnormalize_standardscaler(y_true, node_type)

                n_channels = y_pred_unnorm.shape[1]
                for i in range(min(5, n_channels)):
                    plt.figure()
                    plt.hist(
                        y_true_unnorm[:, i].cpu().numpy(),
                        bins=100,
                        alpha=0.6,
                        color="blue",
                        label="y_true",
                    )
                    plt.hist(
                        y_pred_unnorm[:, i].cpu().numpy(),
                        bins=100,
                        alpha=0.6,
                        color="orange",
                        label="y_pred",
                    )
                    plt.xlabel(f"{node_type} - Channel {i+1}")
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram - {node_type} Channel {i+1} (Epoch {self.current_epoch})")
                    plt.legend()
                    plt.tight_layout()
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
            factor=0.2,  # new_lr = lr * factor (a more aggressive decay can be good)
            patience=5,  # Number of epochs with no improvement after which LR will be reduced
            verbose=True,  # Print a message when the LR is changed
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

    def debug(self, *args, **kwargs):
        if getattr(self, "verbose", False) and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
            print(*args, **kwargs)
