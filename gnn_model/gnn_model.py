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
        self.hidden_dim = hidden_dim

        # --- Create and store the mesh structure as part of the model ---
        self.mesh_structure = create_mesh(
            splits=mesh_resolution, levels=4, hierarchical=False, plot=False
        )
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

        first_instrument_config = next(
            iter(next(iter(observation_config.values())).values())
        )
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
                edge_types.extend(
                    [(node_type_input, "to", "mesh"), ("mesh", "to", node_type_target)]
                )

                input_dim = cfg.get("input_dim")
                target_dim = cfg.get("target_dim")

                # Encoder GNN (obs -> mesh)
                edge_type_tuple_enc = (node_type_input, "to", "mesh")
                self.observation_encoders[self._edge_key(edge_type_tuple_enc)] = (
                    InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        update_edges=False,
                    )
                )

                # Decoder GNN (mesh -> target)
                edge_type_tuple_dec = ("mesh", "to", node_type_target)
                self.observation_decoders[self._edge_key(edge_type_tuple_dec)] = (
                    InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        update_edges=False,
                    )
                )

                # Initial MLP to project raw features to hidden_dim
                self.observation_embedders[node_type_input] = make_mlp(
                    [input_dim] + self.mlp_blueprint_end
                )

                # Final MLP to map from hidden dim to output dim
                if node_type_target == "atms_target":
                    self.scan_angle_embed_dim = 8
                    self.scan_angle_embedder = make_mlp([1, self.scan_angle_embed_dim])
                    input_dim_for_mapper = hidden_dim + self.scan_angle_embed_dim
                else:
                    input_dim_for_mapper = hidden_dim

                # Add one more hidden layer to the output mapper as requested
                output_map_layers = (
                    [input_dim_for_mapper] + [hidden_dim] * hidden_layers + [target_dim]
                )
                self.output_mappers[node_type_target] = make_mlp(
                    output_map_layers, layer_norm=False
                )

        self.processor = Processor(
            hidden_dim=hidden_dim,
            node_types=node_types,
            edge_types=edge_types,
            num_message_passing_steps=num_layers,
        )

        self.register_buffer("mesh_x", torch.tensor(mesh_x, dtype=torch.float32))
        self.register_buffer(
            "mesh_edge_index", torch.tensor(mesh_edge_index, dtype=torch.long)
        )
        self.register_buffer(
            "mesh_edge_attr", torch.tensor(mesh_edge_attr, dtype=torch.float32)
        )

        self.mse = nn.MSELoss(reduction="none")
        self.huber = nn.HuberLoss(delta=0.1, reduction="none")

        # Automatically decide whether to use weighted_huber_loss based on weight config
        self.use_weighted_huber_loss = not (
            all(w == 1.0 for w in self.instrument_weights.values())
            and all(
                torch.allclose(
                    (
                        w.to(torch.float32)
                        if isinstance(w, torch.Tensor)
                        else torch.tensor(w, dtype=torch.float32)
                    ),
                    torch.ones(len(w)),
                )
                for w in self.channel_weights.values()
            )
        )

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
                if hasattr(loader, "sampler") and isinstance(
                    loader.sampler, DistributedSampler
                ):
                    loader.sampler.set_epoch(self.current_epoch)

    def unnormalize_standardscaler(self, tensor, node_type, mean=None, std=None):
        # Use config values if node_type matches
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
            # Get the mean and std for the single stationPressure feature
            mean_val = self.feature_stats["virtualTemperature"][0]
            std_val = self.feature_stats["virtualTemperature"][1]
            return tensor * std_val + mean_val

        elif mean is not None and std is not None:
            return tensor * std + mean
        else:
            raise ValueError(
                f"Un-normalization not supported for node_type: {node_type}"
            )

    def forward(self, data: HeteroData, step_data_list=None) -> Dict[str, torch.Tensor]:

        num_graphs = data.num_graphs
        num_mesh_nodes = self.mesh_x.shape[0]

        # Inject and batch static mesh data
        data["mesh"].x = self.mesh_x.repeat(num_graphs, 1)
        data["mesh", "to", "mesh"].edge_attr = self.mesh_edge_attr.repeat(num_graphs, 1)

        edge_indices = [
            self.mesh_edge_index + i * num_mesh_nodes for i in range(num_graphs)
        ]
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
                edge_attr = torch.empty(
                    (edge_index.size(1), self.hidden_dim), device=self.device
                )

                encoder = self.observation_encoders[self._edge_key(edge_type)]
                encoder.edge_index = edge_index

                # --- Debugging ---
                print(f"\n[DEBUG] Encoding for edge type: {edge_type}")
                print(f"  - send_rep (obs) shape: {obs_features.shape}")
                print(f"  - rec_rep (mesh) shape: {encoded_mesh_features.shape}")
                print(f"  - edge_index shape: {edge_index.shape}")
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
                    encoded_features[node_type] = torch.zeros(
                        num_nodes, self.hidden_dim, device=self.device
                    )

        # --------------------------------------------------------------------
        # STAGE 4: PROCESS (Deep message passing on the graph)
        # --------------------------------------------------------------------
        processed_features = self.processor(encoded_features, data.edge_index_dict)

        print(
            f"[PROCESSOR] Processed mesh shape after {self.hparams.num_layers} "
            f"layers: {processed_features['mesh'].shape}"
        )

        # --------------------------------------------------------------------
        # STAGE 5: DECODE (Identical to Ocelot2's `decode` method)
        # --------------------------------------------------------------------
        predictions = {}
        mesh_features_processed = processed_features["mesh"]

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type == "mesh" and dst_type.endswith("_target"):
                target_features_initial = torch.zeros(
                    data[dst_type].num_nodes, self.hidden_dim, device=self.device
                )

                decoder = self.observation_decoders[self._edge_key(edge_type)]
                decoder.edge_index = edge_index

                edge_attr = torch.empty(
                    (edge_index.size(1), self.hidden_dim), device=self.device
                )

                decoded_target_features = decoder(
                    send_rep=mesh_features_processed,
                    rec_rep=target_features_initial,
                    edge_rep=edge_attr,
                )

                if dst_type == "atms_target":
                    scan_angle = data[dst_type].x
                    scan_angle_embedded = self.scan_angle_embedder(scan_angle)
                    final_features = torch.cat(
                        [decoded_target_features, scan_angle_embedded], dim=-1
                    )
                    predictions[dst_type] = self.output_mappers[dst_type](
                        final_features
                    )
                else:
                    predictions[dst_type] = self.output_mappers[dst_type](
                        decoded_target_features
                    )

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
        decoder_names = [
            f"{inst_name}_target"
            for obs_type, instruments in self.observation_config.items()
            for inst_name in instruments
        ]

        data_module = self.trainer.datamodule
        current_bin_name = (
            batch.bin_name[0] if isinstance(batch.bin_name, list) else batch.bin_name
        )
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
                        print(
                            f"[ROLL DEBUG] batch[{decoder_name}].y shape: {getattr(batch[decoder_name], 'y', None).shape}"
                        )
                        step_data[decoder_name] = {
                            "y": batch[decoder_name].y,
                            "x": batch[decoder_name].x,
                            "target_metadata": batch[decoder_name].target_metadata,
                        }
                        step_data[("mesh", "to", decoder_name)] = {
                            "edge_index": batch[
                                ("mesh", "to", decoder_name)
                            ].edge_index,
                            "edge_attr": batch[("mesh", "to", decoder_name)].edge_attr,
                        }
                if step_data:
                    step_data_list.append(step_data)
                    actual_step += 1

            elif (
                target_bin_num <= max_bin_num
                and target_bin_name in data_module.data_summary
                and target_bin_name in available_bins
            ):

                target_bin_data = data_module.data_summary[target_bin_name]
                temp_graph_data = data_module._create_graph_structure(target_bin_data)
                for decoder_name in decoder_names:
                    if decoder_name in temp_graph_data:
                        device = batch[decoder_name].y.device
                        step_data[decoder_name] = {
                            "y": temp_graph_data[decoder_name].y.to(device),
                            "x": temp_graph_data[decoder_name].scan_angle.to(device),
                            "target_metadata": temp_graph_data[
                                decoder_name
                            ].target_metadata.to(device),
                        }
                        step_data[("mesh", "to", decoder_name)] = {
                            "edge_index": temp_graph_data[
                                ("mesh", "to", decoder_name)
                            ].edge_index.to(device),
                            "edge_attr": temp_graph_data[
                                ("mesh", "to", decoder_name)
                            ].edge_attr.to(device),
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
                                # 'scaler_mean': batch[decoder_name].scaler_mean,
                                # 'scaler_std': batch[decoder_name].scaler_std,
                                "target_metadata": batch[decoder_name].target_metadata,
                            }
                            step_data[("mesh", "to", decoder_name)] = {
                                "edge_index": batch[
                                    ("mesh", "to", decoder_name)
                                ].edge_index,
                                "edge_attr": batch[
                                    ("mesh", "to", decoder_name)
                                ].edge_attr,
                            }
                    if step_data:
                        step_data_list.append(step_data)
                        self.debug(
                            f"Warning: No data for {target_bin_name}, using original batch"
                        )
                else:
                    break

        return step_data_list, actual_step

    def training_step(self, batch, batch_idx):
        print("[DIAG] Entered training_step()")
        if torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(
                f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB"
            )

        current_rollout_steps = self.get_current_rollout_steps()
        print(f"[training_step] batch: {batch.bin_name}")
        step_data_list, actual_rollout_steps = self._get_sequential_step_data(
            batch, current_rollout_steps
        )
        print(f"[DEBUG] actual_rollout_steps: {actual_rollout_steps}")

        all_predictions = self(batch, step_data_list=step_data_list)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_predictions = 0

        # Calculate loss for each observation type and add it to the total
        for node_type, preds_list in all_predictions.items():
            gts_list = [
                step_data.get(node_type, {}).get("y", None)
                for step_data in step_data_list
            ]

            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts_list)):
                if y_pred is None or y_true is None or y_pred.numel() == 0:
                    continue

                loss = self.mse(y_pred, y_true).mean()

                # Add the loss for this instrument to the total
                total_loss = total_loss + loss
                num_predictions += 1
                # Log the individual loss for debugging
                self.log(
                    f"train_loss_{node_type}",
                    loss,
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
        avg_loss = (
            total_loss / num_predictions
            if num_predictions > 0
            else torch.tensor(0.0, device=self.device)
        )
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
        self.log(
            "rollout_steps", float(actual_rollout_steps), on_step=True, sync_dist=False
        )
        if self.trainer.is_global_zero and batch_idx == 0:
            print(
                f"[TRAIN] Epoch {self.current_epoch} - train_loss: {avg_loss.cpu().item():.6f}"
            )

        return avg_loss

    def validation_step(self, batch, batch_idx):
        print(f"VALIDATION STEP batch: {batch.bin_name}")
        current_rollout_steps = self.max_rollout_steps

        # Build decoder names from config (all possible node_types with targets)
        decoder_names = [
            f"{inst_name}_target"
            for obs_type, instruments in self.observation_config.items()
            for inst_name in instruments
        ]

        # Prepare metrics storage
        all_step_rmse = {name: [] for name in decoder_names}
        all_step_mae = {name: [] for name in decoder_names}
        all_step_bias = {name: [] for name in decoder_names}
        all_losses = []

        # Prepare rollout step data
        step_data_list, actual_rollout_steps = self._get_sequential_step_data(
            batch, current_rollout_steps
        )
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
            inst_name = node_type.replace("_target", "")
            # instrument_weight = self.instrument_weights.get(inst_name, 1.0)
            gts = [
                step_data.get(node_type, {}).get("y", None)
                for step_data in step_data_list
            ]

            n_steps = min(len(preds_list), len(gts))

            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts)):
                if y_pred is None or y_true is None or y_pred.numel() == 0:
                    continue
                if y_pred.shape != y_true.shape:
                    continue

                # # 2. Apply the overall instrument weight
                loss = self.mse(y_pred, y_true).mean()

                total_loss = total_loss + loss
                num_predictions += 1
                self.log(
                    f"val_loss_{node_type}",
                    loss,
                    sync_dist=True,
                    on_epoch=True,
                    batch_size=1,
                )

                # --- Metrics Calculation ---
                y_pred_unnorm = self.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.unnormalize_standardscaler(y_true, node_type)

                step_rmse = torch.sqrt(
                    F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")
                ).mean(dim=0)
                step_mae = F.l1_loss(
                    y_pred_unnorm, y_true_unnorm, reduction="none"
                ).mean(dim=0)
                step_bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)
                all_step_rmse[node_type].append(step_rmse)
                all_step_mae[node_type].append(step_mae)
                all_step_bias[node_type].append(step_bias)

                if (
                    self.trainer.is_global_zero  # only main process
                    and step == 0  # only first step of rollout
                    and batch_idx == 0  # only first batch
                ):
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
                    # Build DataFrame
                    df = pd.DataFrame(
                        {
                            "lat": lat_deg,
                            "lon": lon_deg,
                            **{
                                f"pred_ch{i+1}": y_pred_unnorm[:, i].cpu().numpy()
                                for i in range(n_ch)
                            },
                            **{
                                f"true_ch{i+1}": y_true_unnorm[:, i].cpu().numpy()
                                for i in range(n_ch)
                            },
                        }
                    )
                    filename = f"{out_dir}/val_{node_type}_epoch{self.current_epoch}_batch{batch_idx}_step{step}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved: {filename}")
            # Placeholder logging for missing steps (to ensure stable CSV shape for loggers)
            num_channels = (
                all_step_rmse[node_type][0].shape[0] if all_step_rmse[node_type] else 1
            )
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
                    print(
                        f"[VAL] {node_type} RMSE (avg): {torch.stack(all_step_rmse[node_type]).mean().item():.4f}"
                    )

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
                    plt.title(
                        f"Histogram - {node_type} Channel {i+1} (Epoch {self.current_epoch})"
                    )
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        f"hist_{node_type}_ch_{i+1}_epoch{self.current_epoch}.png"
                    )
                    plt.close()

        # --- Final loss calculation for the entire validation step ---
        avg_loss = (
            total_loss / num_predictions
            if num_predictions > 0
            else torch.tensor(0.0, device=self.device)
        )

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
                    self.debug(
                        f"[DEBUG] encoded.grad norm: {self._encoded_ref.grad.norm().item():.6f}"
                    )
                else:
                    self.debug("[DEBUG] encoded.grad is still None after backward.")
            else:
                self.debug("[DEBUG] _encoded_ref is None")

        # x_hidden grad
        if hasattr(self, "_x_hidden_ref"):
            if self._x_hidden_ref is not None and self._x_hidden_ref.grad is not None:
                self.debug(
                    f"[DEBUG] x_hidden.grad norm: {self._x_hidden_ref.grad.norm().item():.6f}"
                )
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

    def ocelot_loss(self, y_pred, y_true, instrument_ids, check_grad=True):
        """
        Custom weighted loss that applies per-instrument and per-channel weights.

        Args:
            y_pred (Tensor): Predicted outputs of shape [N, C]
            y_true (Tensor): Ground truth targets of shape [N, C]
            instrument_ids (Tensor): Tensor of shape [N] with instrument ID per observation

        Returns:
            loss (Tensor): Scalar loss value
        """
        device = y_pred.device
        assert (
            y_pred.device == self.device
        ), f"y_pred not on model device: {y_pred.device} != {self.device}"
        assert y_true.device.type == "cuda", "y_true is not on GPU"
        if check_grad:
            assert y_pred.requires_grad, "y_pred does not require grad"

        loss = 0.0
        total = 0

        for inst_id in instrument_ids.unique():
            inst_id_int = int(inst_id.item())
            inst_mask = instrument_ids == inst_id

            y_p = y_pred[inst_mask]
            y_t = y_true[inst_mask]
            # Normalize y_pred and y_true using per-channel stats
            mean = self.channel_mean.to(device)
            std = self.channel_std.to(device)
            y_p = (y_p - mean) / std
            y_t = (y_t - mean) / std

            # Instrument weight fallback
            w_i = self.instrument_weights.get(inst_id_int, 1.0)

            # Channel weight fallback
            w_c = self.channel_weights.get(
                inst_id_int, torch.ones(y_p.shape[1], device=y_p.device)
            )
            if w_c.shape[0] != y_p.shape[1]:
                pad_len = y_p.shape[1] - w_c.shape[0]
                if pad_len > 0:
                    w_c = torch.cat([w_c, torch.zeros(pad_len, device=w_c.device)])

            # Apply channel mask (default = keep all)
            channel_mask = self.channel_masks.get(
                inst_id_int, torch.ones_like(w_c, dtype=torch.bool)
            )
            y_p_masked = y_p[:, channel_mask]
            y_t_masked = y_t[:, channel_mask]
            w_c_masked = w_c[channel_mask]

            # Per-channel MSE, then weighted sum
            per_channel_mse = ((y_p_masked - y_t_masked) ** 2).mean(dim=0)
            weighted_loss = (per_channel_mse * w_c_masked).sum()

            if self.trainer.is_global_zero:
                self.debug(
                    f"[DEBUG] Instrument {inst_id_int}: weight={w_i:.3f}, active_channels={channel_mask.sum().item()}"
                )

            loss += w_i * weighted_loss
            total += y_p.shape[0]

        return loss / (total + 1e-8)

    def weighted_huber_loss(self, pred, target, instrument_ids=None):
        """
        Weighted Huber loss, supporting per-instrument and per-channel weights.
        Args:
            pred:      [N, C] predictions
            target:    [N, C] targets
            instrument_ids: [N] or None (if None, applies uniform channel weighting)
        Returns:
            Scalar loss
        """
        device = pred.device
        huber_loss = self.huber(pred, target)  # [N, C]

        # If no instrument_ids, just mean-weight across channels (default)
        if instrument_ids is None:
            if hasattr(self, "channel_weights") and self.channel_weights:
                # If you want to apply some global channel weighting
                weights = self.channel_weights.get(
                    "global", torch.ones(pred.shape[1], device=device)
                )
                weights = weights.to(device)
                return (huber_loss * weights).mean()
            else:
                return huber_loss.mean()

        # Otherwise, per-instrument weighting
        unique_ids = instrument_ids.unique()
        total_loss = 0.0
        count = 0
        for inst_id in unique_ids:
            mask = instrument_ids == inst_id
            if not mask.any():
                continue
            inst_id_int = int(inst_id.item())
            # Accept both int and str keys (if channel_weights uses names)
            weights = self.channel_weights.get(
                inst_id_int,
                self.channel_weights.get(
                    str(inst_id_int), torch.ones(pred.shape[1], device=device)
                ),
            )
            weights = weights.to(device)
            masked_loss = huber_loss[mask] * weights  # [num_samples, C]
            total_loss += masked_loss.mean()
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=device)

    def debug(self, *args, **kwargs):
        if getattr(self, "verbose", False) and (
            not hasattr(self, "trainer") or self.trainer.is_global_zero
        ):
            print(*args, **kwargs)
