import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import is_initialized
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import GATConv
from torch_scatter import scatter, scatter_add
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import pandas as pd

# MK
from loss import level_weighted_mse


class GNNLightning(pl.LightningModule):
    """
    A Graph Neural Network (GNN) model for processing structured spatiotemporal data.

    The model consists of:
    - An MLP-based encoder that embeds observation nodes (with edge attributes) into hidden mesh nodes.
    - A processor that applies multiple GATConv layers for message passing between mesh nodes.
    - An MLP-based decoder that maps hidden mesh node features to target nodes via KNN-based edges.

    Key Features:
    - Encoder and decoder use distance information (as edge attributes).
    - Decoder output is aggregated using inverse-distance weighted averaging.
    - Includes LayerNorm and Dropout in both encoder and decoder for regularization.

    Methods:
        forward(data):
            Runs the forward pass, including encoding, message passing, decoding, and
            weighted aggregation to produce target predictions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16, lr=1e-4,
                 instrument_weights=None, channel_weights=None, verbose=False,
                 max_rollout_steps=1, rollout_schedule='step'):
        """
        Initializes the GNNLightning model with an encoder, processor, and decoder.

        Parameters:
        input_dim (int): Number of input features per observation node (before encoding).
        hidden_dim (int): Size of the hidden representation used in all layers.
        output_dim (int): Number of features to predict at each target node.
        num_layers (int, optional): Number of GATConv layers in the processor block (default: 16).
        lr (float, optional): Learning rate for the optimizer (default: 1e-4).
        """
        super().__init__()
        self.verbose = verbose
        self.save_hyperparameters()
        self.lr = lr
        self.instrument_weights = instrument_weights or {}
        self.channel_weights = channel_weights or {}
        self.channel_masks = {}
        for inst_id, weights in self.channel_weights.items():
            weights_tensor = torch.tensor(weights)
            mask = weights_tensor > 0
            self.channel_masks[int(inst_id)] = mask
        self.max_rollout_steps = max_rollout_steps  # MK: goal is 3 days
        self.rollout_schedule = rollout_schedule    # MK

        # Encoder: Maps data nodes to hidden nodes
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # <== Add this extra layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim, add_self_loops=True, bias=True) for _ in range(num_layers)])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            # nn.Sigmoid()  # MK: Add this if targets are 0-1
        )

        # Optional: Initialize processor layer biases to small values
        for layer in self.processor_layers:
            if hasattr(layer, "bias") and layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.1)

        # Define loss function
        # self.loss_fn = nn.MSELoss()
        self.huber = torch.nn.HuberLoss(delta=0.1, reduction="none")

        # Automatically decide whether to use ocelot_loss based on weight config
        self.use_ocelot_loss = not (
            all(w == 1.0 for w in self.instrument_weights.values()) and
            all(torch.allclose(torch.tensor(w, dtype=torch.float32), torch.ones(len(w))) for w in self.channel_weights.values())
        )

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            print(f"[INFO] Using {'weighted Huber loss' if self.use_ocelot_loss else 'HuberLoss'} for training.")

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

    @staticmethod
    def unnormalize(tensor, min_vals, max_vals):
        return tensor * (max_vals - min_vals) + min_vals

    # MK: add latent rollout
    def forward(self, data, n_steps=1):
        self.debug(f"[DEBUG] [forward] self.training = {self.training}, grad_enabled = {torch.is_grad_enabled()}")
        if torch.cuda.is_available():
            self.debug(f"[Forward] Batch x shape: {data.x.shape}")
        if self.trainer.is_global_zero and self.current_epoch == 0:
            for name, param in self.named_parameters():
                if "decoder" in name and "weight" in name:
                    self.debug(f"[DEBUG] Param {name} mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
                    self.debug(f"[DEBUG] Param {name} requires_grad: {param.requires_grad}")

        x = data.x
        edge_feats = data.edge_attr_encoder.unsqueeze(1)  # Shape [E, 1]

        # it makes sure gradients will propagate from the loss through the observations(x) and the edge attributes.
        if self.training:
            x.requires_grad_(True)
            edge_feats.requires_grad_(True)

        if torch.cuda.is_available():
            self.debug(f"[DEBUG] x mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
            self.debug(f"[DEBUG] x requires_grad: {x.requires_grad}")
            self.debug(f"[DEBUG] edge_feats requires_grad: {edge_feats.requires_grad}")
            self.debug(f"[Forward] CUDA Mem: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        edge_index_encoder = data.edge_index_encoder
        edge_index_processor = data.edge_index_processor

        # MK: Get both ground truth and target locations for all steps
        if n_steps > 1:
            # ground_truths, target_locations_list = self._get_sequential_targets_and_locations(data, n_steps)
            step_data_list = self._get_sequential_step_data(data, n_steps)
        else:
            # Single step: use original data
            # ground_truths = [data.y]
            # target_locations_list = [data.target_metadata[:, :2]]
            step_data_list = [{
                'y': data.y,
                'edge_index_decoder': data.edge_index_decoder,
                'edge_attr_decoder': data.edge_attr_decoder,
                'target_scaler_min': data.target_scaler_min,
                'target_scaler_max': data.target_scaler_max
            }]

        # === Encoding: obs → mesh ===
        src_encoder = edge_index_encoder[0]
        tgt_encoder = edge_index_encoder[1]

        used_obs = src_encoder.unique()
        self.debug(f"[DEBUG] Number of obs connected to mesh: {used_obs.shape[0]} / {x.shape[0]}")
        self.debug(f"[DEBUG] src_encoder shape: {src_encoder.shape}")
        self.debug(f"[DEBUG] tgt_encoder shape: {tgt_encoder.shape}")
        self.debug(f"[DEBUG] src_encoder max: {src_encoder.max().item()}, min: {src_encoder.min().item()}")
        assert src_encoder.max().item() < x.shape[0], "[ERROR] src_encoder has out-of-bounds indices"

        # === Get decoder edges for first step to determine used mesh nodes
        # MK:
        # edge_index_decoder = data.edge_index_decoder
        # Get all unique mesh nodes used across all steps for proper aggregation
        all_decoder_mesh_nodes = set()
        for step_data in step_data_list:
            decoder_mesh_nodes = step_data['edge_index_decoder'][0].unique()
            all_decoder_mesh_nodes.update(decoder_mesh_nodes.cpu().numpy())

        # MK: Get all unique processor nodes
        processor_mesh_nodes = edge_index_processor.flatten().unique()

        # === Mask edges whose target mesh nodes are not used downstream
        # MK: put all unique node IDs together:
        # used_mesh_nodes = torch.cat([edge_index_decoder[0], edge_index_processor[0], edge_index_processor[1]]).unique()
        all_used_mesh_nodes = torch.cat([
            torch.tensor(list(all_decoder_mesh_nodes), device=data.x.device),
            processor_mesh_nodes
        ]).unique()

        # MK: Filter encoder edges to only connect to mesh nodes we'll actually use
        # edge_mask = torch.isin(tgt_encoder, used_mesh_nodes)
        edge_mask = torch.isin(tgt_encoder, all_used_mesh_nodes)

        src_encoder = src_encoder[edge_mask]
        tgt_encoder = tgt_encoder[edge_mask]
        edge_feats = edge_feats[edge_mask]

        # Encode observations to mesh
        obs_feats = x[src_encoder]
        encoder_input = torch.cat([obs_feats, edge_feats], dim=1)
        encoded = self.encoder_mlp(encoder_input)
        self.debug("[DEBUG] encoder_input mean/std:", encoder_input.mean().item(), encoder_input.std().item())
        self.debug("[DEBUG] encoded min/max:", encoded.min().item(), encoded.max().item())

        # === Aggregate to mesh nodes
        # MK: DEBUG and OPTIMIZATION
        # Original code had bug: scatter_add_aggregate used reduce="add" despite claiming "mean"
        # Also had inefficient branching logic that computed scatter twice in some cases
        #       x_hidden = self.scatter_add_aggregate(encoded, tgt_encoder, num_targets) got computed twice for non missing case
        # Since satellite data always has missing mesh nodes (sparse coverage),
        # we always took the "len(missing_mesh_nodes) > 0" path (manually compute index_add and mean) -> this can be replaced by scatter

        # === OPTIMIZED: Use scatter with reduce="mean" ===
        max_node_id = all_used_mesh_nodes.max().item() + 1
        if tgt_encoder.max() >= max_node_id:
            raise RuntimeError("[BUG] tgt_encoder index out of range")
        x_hidden = scatter(encoded, tgt_encoder, dim=0, dim_size=max_node_id, reduce="mean")

        # === Set up node mapping for processor edges ===
        unique_tgt_nodes_sorted = torch.arange(0, max_node_id, device=encoded.device, dtype=torch.long)

        # === Retain gradients for training ===
        if self.training:
            assert encoded.requires_grad, "[ERROR] encoded must require grad"
            encoded.retain_grad()
            x_hidden.retain_grad()
            self._encoded_ref = encoded
            self._x_hidden_ref = x_hidden

        # === Optional debugging ===
        if self.trainer.is_global_zero:
            self.debug(f"[DEBUG] x_hidden shape: {x_hidden.shape}")
            self.debug(f"[DEBUG] x_hidden.requires_grad: {x_hidden.requires_grad}")
            self.debug(f"[DEBUG] Unified aggregation method: scatter with reduce='mean'")
            self.debug(f"[DEBUG] encoded.requires_grad: {encoded.requires_grad}")

        # === Remap processor edges to local indices ===
        processor_nodes = edge_index_processor.flatten().unique()
        missing_nodes = processor_nodes[~torch.isin(processor_nodes, unique_tgt_nodes_sorted)]
        if len(missing_nodes) > 0:
            print(f"[ERROR] These processor mesh node IDs are missing in encoded nodes: {missing_nodes}")
            raise ValueError("edge_index_processor contains mesh node IDs not in x_hidden!")
        else:
            self.debug(f"[DEBUG] All {len(processor_nodes)} processor nodes covered in x_hidden")
        edge_index_processor = torch.searchsorted(unique_tgt_nodes_sorted, edge_index_processor)

        predictions = []  # MK: add empty list to store predictions

        # Rollout loop # MK: add for loop to go through steps
        for step in range(n_steps):  # MK: indentation, code no change
            # === Processor: mesh ↔ mesh ===
            if self.trainer.is_global_zero:
                print(f"\n[PROCESSOR DEBUG] Step {step}")
                print(f"x_hidden input: mean={x_hidden.mean():.4f}, std={x_hidden.std():.4f}")
                print(f"x_hidden input: min={x_hidden.min():.4f}, max={x_hidden.max():.4f}")
                self.debug(f"[DEBUG] encoded.requires_grad: {encoded.requires_grad}")
                self.debug(f"[DEBUG] encoded.grad_fn: {encoded.grad_fn}")
                self.debug(f"[DEBUG] x_hidden.requires_grad BEFORE processor: {x_hidden.requires_grad}")
                self.debug(f"[DEBUG] x_hidden.grad_fn BEFORE processor: {x_hidden.grad_fn}")
                self.debug(f"[DEBUG] x_hidden.is_leaf: {x_hidden.is_leaf}")

            self.debug(f"[DEBUG] x_hidden shape before processor: {x_hidden.shape}")
            self.debug(f"[DEBUG] edge_index_processor shape: {edge_index_processor.shape}")
            self.debug(f"[DEBUG] edge_index_processor max: {edge_index_processor.max().item()}, min: {edge_index_processor.min().item()}")
            assert edge_index_processor.max().item() < x_hidden.size(0), "[ERROR] edge_index_processor has out-of-bounds indices"

            # MK debugging (1 line)
            x_hidden_step_start = x_hidden.clone()

            try:
                for i, layer in enumerate(self.processor_layers):
                    self.debug(f"[DEBUG] Processor layer {i} input shape: {x_hidden.shape}")
                    x_before_processor = x_hidden.detach().clone()

                    if i == 0:
                        x_new, (edge_idx, attn_weights) = layer(x_hidden, edge_index_processor, return_attention_weights=True)
                        self.debug(f"[DEBUG] GAT attention weights layer {i} min/max: {attn_weights.min().item()} / {attn_weights.max().item()}")
                        for name, param in layer.named_parameters():
                            self.debug(f"[DEBUG] GATConv param {name} norm: {param.norm().item()}")
                    else:
                        x_new = layer(x_hidden, edge_index_processor)

                    x_hidden = x_hidden + x_new

                    delta = (x_hidden - x_before_processor).abs().mean().item()
                    self.debug(f"[DEBUG] Mean change in x_hidden from processor: {delta:.6e}")

            except Exception as e:
                print(f"[ERROR] Processor failed at layer {i}: {e}")
                raise

            if self.trainer.is_global_zero:
                total_change = (x_hidden - x_hidden_step_start).abs().mean()
                self.debug(f"Step {step+1} processor change: {total_change:.6f}")
                self.debug(f"[DEBUG] x_hidden.requires_grad AFTER processor: {x_hidden.requires_grad}")
                self.debug(f"[DEBUG] x_hidden.grad_fn AFTER processor: {x_hidden.grad_fn}")
                self.debug(f"[DEBUG] x_hidden.grad is None AFTER processor: {x_hidden.grad is None}")

            # MK === Decoder: Hidden → Target ===
            # MK === Dynamic Decoder: Use target locations for this timestep ===
            current_step_data = step_data_list[step]
            edge_index_decoder = current_step_data['edge_index_decoder']
            edge_attr_decoder = current_step_data['edge_attr_decoder']

            src_decoder = edge_index_decoder[0]
            tgt_decoder = edge_index_decoder[1]

            # MK: core code below remains the same unless labeled

            # Sanity check: mesh ↔ mesh overlap
            used_by_decoder = src_decoder.unique()
            used_by_encoder = edge_index_encoder[1].unique()
            overlap_ratio = torch.isin(used_by_decoder, used_by_encoder).float().mean()
            self.debug(f"[DEBUG] Decoder/Encoder mesh overlap ratio: {overlap_ratio:.4f}")

            # Remap src_decoder to local x_hidden indices
            src_decoder_local = torch.searchsorted(unique_tgt_nodes_sorted, src_decoder)
            tgt_decoder_local = tgt_decoder

            # Get current target size
            current_target_size = current_step_data['y'].shape[0]

            # Mask invalid decoder edges
            valid_src = src_decoder_local < x_hidden.shape[0]
            valid_tgt = tgt_decoder_local < current_target_size
            valid_mask = valid_src & valid_tgt

            src_decoder_local = src_decoder_local[valid_mask]
            tgt_decoder_local = tgt_decoder_local[valid_mask]
            dist_feats = edge_attr_decoder[valid_mask].unsqueeze(1)

            if len(src_decoder_local) > 0 and src_decoder_local.max().item() >= x_hidden.size(0):
                raise IndexError(f"[ERROR] src_decoder_local index {src_decoder_local.max().item()} exceeds x_hidden size {x_hidden.size(0)}")

            # MK: Fix target size validation to use current target size
            # Validate target indices
            max_idx = tgt_decoder_local.max().item() if len(tgt_decoder_local) > 0 else -1
            if max_idx >= 0:
                assert max_idx < current_target_size, f"[DEBUG] max={max_idx} >= target_size={current_target_size}"

            # === Build decoder input AFTER masking
            mesh_feats = x_hidden[src_decoder_local]
            decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)

            # === Simple, non-chunked decoding (no gate, no scatter) ===
            # decoded = self.decoder_mlp(decoder_input)
            chunk_size = 1_500_000
            if decoder_input.size(0) > chunk_size:
                out_dim = self.decoder_mlp[-1].out_features
                num_rows = decoder_input.size(0)
                decoded_chunks = []

                for start in range(0, num_rows, chunk_size):
                    end = min(start + chunk_size, num_rows)
                    chunk = decoder_input[start:end]  # stays on GPU
                    decoded_chunks.append(self.decoder_mlp(chunk))

                decoded = torch.cat(decoded_chunks, dim=0)
            else:
                decoded = self.decoder_mlp(decoder_input)
            out = decoded  # No gate, no norm, no scatter

            # Optional: log stats for first 5 channels
            if self.trainer.is_global_zero:
                self.debug("[DEBUG] Decoded shape:", out.shape)
                for ch in range(min(5, out.shape[1])):
                    self.debug(
                        f"[DEBUG] Ch{ch+1}: μ={out[:, ch].mean().item():.4f}, σ={out[:, ch].std().item():.4f}, "
                        f"min={out[:, ch].min().item():.4f}, max={out[:, ch].max().item():.4f}"
                    )

            tgt_decoder_local = torch.clamp(tgt_decoder_local, max=current_target_size - 1)  # prevent out-of-bounds after remapping
            bincounts = torch.bincount(tgt_decoder_local.cpu(), minlength=current_target_size)
            self.debug("[DEBUG] Max decoder count per target:", bincounts.max().item())
            self.debug("[DEBUG] out std before scatter:", out.std().item())

            # Aggregate to targets
            x_out = scatter(out, tgt_decoder_local, dim=0, dim_size=current_target_size, reduce="mean")
            self.debug("[DEBUG] scatter_mean variance:", x_out.var().item())
            self.debug(f"[DEBUG] Unique targets: {tgt_decoder_local.unique().numel()} / {current_target_size}")
            self.debug(f"[DEBUG] x_out requires_grad: {x_out.requires_grad}, grad_fn: {x_out.grad_fn}")

            if self.training and self.trainer.is_global_zero:
                try:
                    from torch.autograd import grad

                    probe = grad(self._x_hidden_ref.sum(), self._encoded_ref, retain_graph=True, allow_unused=True)[0]
                    if probe is None:
                        self.debug("[DEBUG] (Sanity) x_hidden → encoded: no grad")
                    else:
                        self.debug("[DEBUG] (Sanity) x_hidden → encoded grad norm:", probe.norm().item())
                except Exception as e:
                    self.debug("[DEBUG] grad probe failed:", e)

            if self.training:
                mesh_feats.retain_grad()
                self._mesh_feats_ref = mesh_feats
                self._x_hidden_ref.retain_grad()
                # Optional L2 regularization term to encourage small norms (acts as a probe for debugging)

            # MK: store predictions
            predictions.append(x_out)
            self.debug(f"[DEBUG] Step {step+1}: prediction shape {x_out.shape}")

        # MK: Handle return value
        if self.training:
            return predictions, 1e-6 * x_hidden.norm()  # Return tuple like main
        else:
            return predictions

    def get_current_rollout_steps(self):
        """
        Determines the current number of rollout steps based on training progress.
        Implements curriculum learning where rollout length increases over time.
        """
        if not hasattr(self, 'max_rollout_steps'):
            return 1  # Default to single step

        if not hasattr(self, 'rollout_schedule'):
            return self.max_rollout_steps

        # Progressive rollout schedule
        current_epoch = self.current_epoch
        current_step = self.global_step  # This tracks gradient descent updates

        if self.rollout_schedule == 'graphcast':
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
                # Add 1 rollout step per 1000 updates after 300k
                additional_steps = 2 + (current_step - threshold) // interval
                return min(additional_steps, self.max_rollout_steps)

        elif self.rollout_schedule == 'linear':
            # Linearly increase from 1 to max_rollout_steps over training
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
            progress = min(current_epoch / max_epochs, 1.0)
            current_steps = 1 + int(progress * (self.max_rollout_steps - 1))
            return current_steps

        elif self.rollout_schedule == 'step':
            # Step-wise increase (GraphCast style)
            if current_epoch < 10:
                return 1
            elif current_epoch < 20:
                return 2
            else:
                return min(self.max_rollout_steps, 3 + (current_epoch - 20) // 10)

        else:
            return self.max_rollout_steps

    # MK
    def _get_sequential_step_data(self, batch, n_steps):
        """
        Gets preprocessed data (y, edge_index_decoder, edge_attr_decoder) for each rollout step
        by accessing the corresponding time bins from the data module.

        Returns:
            List[Dict]: List of data dictionaries for each step containing:
                - 'y': ground truth targets
                - 'edge_index_decoder': decoder edge indices
                - 'edge_attr_decoder': decoder edge attributes
                - 'target_scaler_min', 'target_scaler_max': for unnormalization
        """
        data_module = self.trainer.datamodule
        current_bin_name = batch.bin_name[0] if isinstance(batch.bin_name, list) else batch.bin_name
        bin_num = int(current_bin_name.replace("bin", ""))

        step_data_list = []

        for step in range(n_steps):  # ← FIX: This was not properly indented
            target_bin_name = f"bin{bin_num + step}"

            if target_bin_name in data_module.data_summary:
                # Get the preprocessed bin data
                target_bin_data = data_module.data_summary[target_bin_name]

                # Create a temporary data object to get the preprocessed graph structure
                temp_graph_data = data_module._create_graph_structure(target_bin_data)

                step_data = {
                    'y': temp_graph_data['y'].to(batch.y.device),
                    'edge_index_decoder': temp_graph_data['edge_index_decoder'].to(batch.y.device),
                    'edge_attr_decoder': temp_graph_data['edge_attr_decoder'].to(batch.y.device),
                    'target_scaler_min': temp_graph_data['target_scaler_min'].to(batch.y.device),
                    'target_scaler_max': temp_graph_data['target_scaler_max'].to(batch.y.device)
                }

                step_data_list.append(step_data)

            else:
                # Fallback: repeat last available data
                if step_data_list:
                    step_data_list.append(step_data_list[-1])
                    self.debug(f"Warning: No data for {target_bin_name}, using previous step")
                else:
                    # Ultimate fallback: use original batch data
                    step_data = {
                        'y': batch.y,
                        'edge_index_decoder': batch.edge_index_decoder,
                        'edge_attr_decoder': batch.edge_attr_decoder,
                        'target_scaler_min': batch.target_scaler_min,
                        'target_scaler_max': batch.target_scaler_max
                    }
                    step_data_list.append(step_data)
                    self.debug(f"Warning: No data for {target_bin_name}, using original batch")

        return step_data_list

    def training_step(self, batch, batch_idx):
        # === Enable anomaly detection (only once, for debugging backward passes)
        if torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            # Prints how much GPU memory is being used before you run the forward pass.
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        # === Run forward pass (predict outputs from input batch)
        # MK: update with rollout steps

        current_rollout_steps = self.get_current_rollout_steps()
        # MK: Log the tracking info here
        self.log("MK_global_step", self.global_step)
        self.log("MK_rollout_steps", float(current_rollout_steps))

        # Handle return format from forward method
        forward_result = self(batch, n_steps=current_rollout_steps)
        if isinstance(forward_result, tuple):
            # Training mode: returns (predictions, loss_probe)
            predictions, loss_probe = forward_result
        else:
            # Eval mode: returns just predictions
            predictions = forward_result
            loss_probe = 0.0

        # Get ground truths - now much simpler since we use preprocessed data
        ground_truths = []
        step_data_list = self._get_sequential_step_data(batch, current_rollout_steps)
        for step_data in step_data_list:
            ground_truths.append(step_data['y'])

        # === Sanity checks (shape match and NaN checks)
        assert len(predictions) == len(ground_truths), f"Number of predictions {len(predictions)} != number of ground truths {len(ground_truths)}"

        # Check shapes of individual predictions
        for step, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
            assert prediction.shape == ground_truth.shape, f"Step {step+1}: pred shape {prediction.shape} != truth shape {ground_truth.shape}"

        rank = self.trainer.global_rank if hasattr(self.trainer, "global_rank") else 0
        print(f"[Rank {rank}] Number of rollout steps: {len(predictions)}")
        print(f"[Rank {rank}] Each prediction shape: {predictions[0].shape}, Each ground truth shape: {ground_truths[0].shape}")

        # Check for NaNs
        has_nan_pred = any(torch.isnan(pred).any() for pred in predictions)
        has_nan_truth = any(torch.isnan(truth).any() for truth in ground_truths)

        if has_nan_pred or has_nan_truth:
            print(f"NaNs detected at batch {batch_idx} on rank {self.trainer.global_rank}")
            for step, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                if torch.isnan(pred).any():
                    print(f"  Step {step+1} prediction has NaNs: min={pred.min()}, max={pred.max()}")
                if torch.isnan(truth).any():
                    print(f"  Step {step+1} ground truth has NaNs: min={truth.min()}, max={truth.max()}")

        # === Compute loss and log it
        # MK: update for rollout loss
        # Compute loss for all steps
        total_loss = 0.0
        for step, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
            if prediction.shape[0] == ground_truth.shape[0]:
                if self.use_ocelot_loss:
                    # step_loss = self.ocelot_loss(prediction, ground_truth, batch.instrument_ids)
                    step_loss = self.weighted_huber_loss(prediction, ground_truth, batch.instrument_ids)
                else:
                    # step_loss = self.loss_fn(prediction, ground_truth)
                    loss = self.huber(prediction, ground_truth).mean()  # fallback
                total_loss += step_loss
                self.log(f"train_loss_step{step+1}", step_loss)
            else:
                print(f"Skipping step {step+1}: pred shape {prediction.shape} != truth shape {ground_truth.shape}")

        # Add the loss probe from forward method
        if self.training and isinstance(loss_probe, torch.Tensor):
            total_loss += loss_probe

        # Log final loss and metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("rollout_steps", float(current_rollout_steps))

        # Set loss for return
        loss = total_loss

        if self.trainer.is_global_zero:
            self.debug(f"[DEBUG] Total loss at batch {batch_idx}: {loss.item():.6f}")
            # Get first prediction for logging
            y_pred = predictions[0] if predictions else None
            y_true = ground_truths[0] if ground_truths else None
            if y_pred is not None and y_true is not None:
                with torch.no_grad():
                    self.debug(f"[DEBUG] y_pred → min: {y_pred.min():.4f}, max: {y_pred.max():.4f}, std: {y_pred.std():.6f}")
                    self.debug(f"[DEBUG] y_true → min: {y_true.min():.4f}, max: {y_true.max():.4f}, std: {y_true.std():.6f}")

        # === Gradient probe for encoded
        from torch.autograd import grad

        if self.training and hasattr(self, "_encoded_ref") and self._encoded_ref is not None:
            encoded_probe = self._encoded_ref
            try:
                probe = grad(loss, encoded_probe, retain_graph=True, allow_unused=True)[0]
                if probe is None:
                    self.debug("[DEBUG] Loss → encoded: NO GRADIENT FLOW")
                else:
                    self.debug("[DEBUG] Loss → encoded grad norm:", probe.norm().item())
            except Exception as e:
                self.debug("[DEBUG] grad() check failed:", e)

        # Check _x_hidden_ref grad state
        if self.trainer.is_global_zero and hasattr(self, "_x_hidden_ref") and self._x_hidden_ref is not None:
            self.debug(f"[DEBUG] _x_hidden_ref requires_grad: {self._x_hidden_ref.requires_grad}")
            self.debug(f"[DEBUG] _x_hidden_ref is leaf: {self._x_hidden_ref.is_leaf}")

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[TRAIN] Epoch {self.current_epoch} - train_loss: {loss.item():.6f}")

        if self.trainer.is_global_zero and self.current_epoch % 1 == 0:
            with torch.no_grad():
                norm = self.processor_layers[0].bias.norm().item()
                self.debug(f"[DEBUG] Epoch {self.current_epoch} - Processor Layer 0 bias norm: {norm:.6f}")

        if self.trainer.is_global_zero and predictions:
            y_pred = predictions[0]
            for ch in range(min(5, y_pred.shape[1])):
                self.debug(f"[DEBUG] Predicted ch{ch+1}: mean={y_pred[:, ch].mean().item():.4f}, std={y_pred[:, ch].std().item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        # Use current rollout steps to match training
        current_rollout_steps = self.get_current_rollout_steps()
        # self.max_rollout_steps

        # Get predictions for all rollout steps
        y_pred_list = self(batch, n_steps=current_rollout_steps)

        # Get ground truths for all rollout steps
        ground_truths = []
        step_data_list = self._get_sequential_step_data(batch, current_rollout_steps)
        for step_data in step_data_list:
            ground_truths.append(step_data['y'])

        # Compute loss for all steps
        total_loss = 0.0
        for step, (prediction, ground_truth) in enumerate(zip(y_pred_list, ground_truths)):
            if prediction.shape[0] == ground_truth.shape[0]:
                if self.use_ocelot_loss:
                    step_loss = self.ocelot_loss(prediction, ground_truth, batch.instrument_ids, check_grad=False)
                else:
                    step_loss = self.loss_fn(prediction, ground_truth)

                total_loss += step_loss
                self.log(f"val_loss_step{step+1}", step_loss, sync_dist=True, on_epoch=True)
            else:
                print(f"Skipping validation step {step+1}: pred shape {prediction.shape} != truth shape {ground_truth.shape}")

        # Average loss across all steps
        loss = total_loss / len(y_pred_list)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_rollout_steps", float(current_rollout_steps), sync_dist=True, on_epoch=True)

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] Epoch {self.current_epoch} - val_loss: {loss.item():.6f} (across {current_rollout_steps} steps)")

        # Use first step for instrument-specific logging and metrics (for consistency)
        y_pred = y_pred_list[0]
        y_true = ground_truths[0]
        instrument_ids = batch.instrument_ids

        # Per-instrument loss logging (using first step)
        for inst_id in instrument_ids.unique():
            mask = instrument_ids == inst_id
            if mask.sum() == 0:
                continue
            inst_loss = F.mse_loss(y_pred[mask], y_true[mask])
            self.log(f"val_loss_inst_{inst_id.item()}", inst_loss.item(), prog_bar=False, sync_dist=True, on_epoch=True)

        if self.verbose and self.trainer.is_global_zero and batch_idx == 0:
            import matplotlib.pyplot as plt

            for i in range(min(5, y_pred.shape[1])):  # Plot only first 5 channels
                y_pred_i = y_pred[:, i]
                y_true_i = y_true[:, i]
                self.debug("[DEBUG] y_pred min:", y_pred.min().item(), "max:", y_pred.max().item())
                self.debug("[DEBUG] y_pred std:", y_pred.std().item())
                self.debug(
                    f"[DEBUG] Channel {i+1} → y_pred mean: {y_pred_i.mean():.6f}, std: {y_pred_i.std():.6f}, "
                    f"min: {y_pred_i.min():.6f}, max: {y_pred_i.max():.6f}"
                )
                plt.figure()
                # Convert to float32 before moving to CPU to avoid BFloat16 issues with matplotlib
                y_true_plot = y_true[:, i].float().detach().cpu().numpy()
                y_pred_plot = y_pred[:, i].float().detach().cpu().numpy()
                plt.hist(y_true_plot, bins=100, alpha=0.6, color="blue", label="Normalized y_true")
                plt.hist(y_pred_plot, bins=100, alpha=0.6, color="orange", label="Normalized y_pred")
                plt.xlabel("Normalized Brightness Temperature")
                plt.ylabel("Frequency")
                plt.title(f"Normalized BT Histogram - Channel {i+1}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"hist_norm_ytrue_ypred_ch_{i+1}_epoch{self.current_epoch}.png")
                plt.close()

        # === UPDATED: Compute metrics averaged across ALL steps ===
        step_rmse_list = []
        step_mae_list = []
        step_bias_list = []

        for step, (y_pred, y_true) in enumerate(zip(y_pred_list, ground_truths)):
            # Get scaler info for this step
            step_data = step_data_list[step]
            min_vals = step_data['target_scaler_min'].to(self.device)
            max_vals = step_data['target_scaler_max'].to(self.device)

            # Unnormalize for this step
            y_pred_unnorm = self.unnormalize(y_pred, min_vals, max_vals)
            y_true_unnorm = self.unnormalize(y_true, min_vals, max_vals)

            # Compute per-channel metrics for this step
            step_rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
            step_mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
            step_bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)

            step_rmse_list.append(step_rmse)
            step_mae_list.append(step_mae)
            step_bias_list.append(step_bias)

            # Log per-step metrics
            for i in range(step_rmse.shape[0]):
                self.log(f"val_rmse_step{step+1}_ch_{i+1}", step_rmse[i].item(), sync_dist=True, on_epoch=True)
                self.log(f"val_mae_step{step+1}_ch_{i+1}", step_mae[i].item(), sync_dist=True, on_epoch=True)
                self.log(f"val_bias_step{step+1}_ch_{i+1}", step_bias[i].item(), sync_dist=True, on_epoch=True)

        # Average metrics across all steps
        avg_rmse = torch.stack(step_rmse_list).mean(dim=0)  # Average across steps
        avg_mae = torch.stack(step_mae_list).mean(dim=0)
        avg_bias = torch.stack(step_bias_list).mean(dim=0)

        # Log averaged metrics (these are the main validation metrics)
        for i in range(avg_rmse.shape[0]):
            self.log(f"val_rmse_ch_{i+1}", avg_rmse[i].item(), sync_dist=True, on_epoch=True)
            self.log(f"val_mae_ch_{i+1}", avg_mae[i].item(), on_epoch=True, sync_dist=True)
            self.log(f"val_bias_ch_{i+1}", avg_bias[i].item(), on_epoch=True, sync_dist=True)

        # Also log step-wise degradation metrics
        if current_rollout_steps > 1:
            # Compare last step to first step to see degradation
            first_step_rmse = step_rmse_list[0].mean()  # Average across channels
            last_step_rmse = step_rmse_list[-1].mean()
            rmse_degradation = last_step_rmse / first_step_rmse

            self.log("val_rmse_degradation", rmse_degradation.item(), sync_dist=True, on_epoch=True)
            self.log("val_first_step_rmse", first_step_rmse.item(), sync_dist=True, on_epoch=True)
            self.log("val_last_step_rmse", last_step_rmse.item(), sync_dist=True, on_epoch=True)

        # Save CSV for visual inspection (use first step)
        if self.trainer.is_global_zero and not hasattr(self, f"_saved_csv_epoch_{self.current_epoch}"):
            setattr(self, f"_saved_csv_epoch_{self.current_epoch}", True)
            import pandas as pd

            # Use first step for CSV
            y_pred_csv = y_pred_list[0]
            y_true_csv = ground_truths[0]
            first_step_data = step_data_list[0]
            min_vals_csv = first_step_data['target_scaler_min'].to(self.device)
            max_vals_csv = first_step_data['target_scaler_max'].to(self.device)

            y_pred_unnorm_csv = self.unnormalize(y_pred_csv, min_vals_csv, max_vals_csv)
            y_true_unnorm_csv = self.unnormalize(y_true_csv, min_vals_csv, max_vals_csv)

            df = pd.DataFrame({
                "lat_deg": batch.target_lat_deg.cpu().numpy(),
                "lon_deg": batch.target_lon_deg.cpu().numpy(),
                **{f"true_bt_{i+1}": y_true_unnorm_csv[:, i].cpu().numpy() for i in range(22)},
                **{f"pred_bt_{i+1}": y_pred_unnorm_csv[:, i].cpu().numpy() for i in range(22)},
            })
            df.to_csv(f"bt_predictions_epoch{self.current_epoch}.csv", index=False)

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] Averaged metrics across {current_rollout_steps} steps:")
            print(f"  RMSE (avg): {avg_rmse.mean().item():.4f}")
            print(f"  MAE (avg): {avg_mae.mean().item():.4f}")
            if current_rollout_steps > 1:
                print(f"  RMSE degradation: {rmse_degradation.item():.2f}x")

        self.log("val_lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)
        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] y_pred mean: {y_pred.mean().item():.6f}, std: {y_pred.std().item():.6f}")

        # Cleanup
        del y_pred, y_true, y_pred_unnorm, y_true_unnorm, batch
        return loss

    def validation_step_original(self, batch, batch_idx):
        # MK: small change for a quick test
        y_pred_list = self(batch, n_steps=1)
        y_pred = y_pred_list[0]
        y_true = batch.y
        # Choose appropriate loss
        if self.use_ocelot_loss:
            # loss = self.ocelot_loss(y_pred, y_true, batch.instrument_ids, check_grad=False)
            loss = self.weighted_huber_loss(y_pred, y_true, batch.instrument_ids)
        else:
            # loss = self.loss_fn(y_pred, y_true)
            loss = self.huber(y_pred, y_true).mean()

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] Epoch {self.current_epoch} - val_loss: {loss.item():.6f}")

        instrument_ids = batch.instrument_ids

        # Per-instrument loss logging
        for inst_id in instrument_ids.unique():
            mask = instrument_ids == inst_id
            if mask.sum() == 0:
                continue
            inst_loss = F.mse_loss(y_pred[mask], y_true[mask])
            self.log(f"val_loss_inst_{inst_id.item()}", inst_loss.item(), prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        if self.verbose and self.trainer.is_global_zero and batch_idx == 0:

            # === Masks ===
            bt_true = y_true[:, :22]
            bt_pred = y_pred[:, :22]

            # Rows where only bt1 (index 0) is non-zero → pressure
            pressure_mask = (bt_true[:, 1:] == 0).all(dim=1)
            bt_mask = ~pressure_mask  # everything else is ATMS

            # === Plot Histograms ===
            for i in range(22):  # All channels
                label = f"BT Channel {i+1}" if i < 22 else "Surface Pressure"
                if i == 0:
                    bt_y_true = y_true[bt_mask, i]
                    bt_y_pred = y_pred[bt_mask, i]
                    pr_y_true = y_true[pressure_mask, i]
                    pr_y_pred = y_pred[pressure_mask, i]

                    # --- BT Histogram for ch0 (skip pressure rows) ---
                    plt.figure()
                    plt.hist(bt_y_true.detach().cpu().numpy(), bins=100, alpha=0.6, color="blue", label="BT y_true")
                    plt.hist(bt_y_pred.detach().cpu().numpy(), bins=100, alpha=0.6, color="orange", label="BT y_pred")
                    plt.xlabel("Normalized Brightness Temperature")
                    plt.ylabel("Frequency")
                    plt.title("Normalized Histogram - BT Channel 1")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"hist_bt_ch_1_bt_only_epoch{self.current_epoch}.png")
                    plt.close()

                    # --- Pressure Histogram (only pressure rows) ---
                    plt.figure()
                    plt.hist(pr_y_true.detach().cpu().numpy(), bins=100, alpha=0.6, color="blue", label="Pressure y_true")
                    plt.hist(pr_y_pred.detach().cpu().numpy(), bins=100, alpha=0.6, color="orange", label="Pressure y_pred")
                    plt.xlabel("Normalized Surface Pressure")
                    plt.ylabel("Frequency")
                    plt.title("Normalized Histogram - Surface Pressure")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"hist_pressure_epoch{self.current_epoch}.png")
                    plt.close()

                else:
                    y_pred_i = y_pred[bt_mask, i]
                    y_true_i = y_true[bt_mask, i]
                    plt.figure()
                    plt.hist(y_true_i.detach().cpu().numpy(), bins=100, alpha=0.6, color="blue", label="Normalized y_true")
                    plt.hist(y_pred_i.detach().cpu().numpy(), bins=100, alpha=0.6, color="orange", label="Normalized y_pred")
                    plt.xlabel("Normalized Brightness Temperature")
                    plt.ylabel("Frequency")
                    plt.title(f"Normalized Histogram - BT Channel {i+1}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"hist_bt_ch_{i+1}_epoch{self.current_epoch}.png")
                    plt.close()

        # Unnormalize
        min_vals = batch.target_scaler_min.to(self.device)
        max_vals = batch.target_scaler_max.to(self.device)
        if self.trainer.is_global_zero and batch_idx == 0:
            print("Min per channel:", min_vals.cpu().numpy())
            print("Max per channel:", max_vals.cpu().numpy())

        y_pred_unnorm = self.unnormalize(y_pred, min_vals, max_vals)
        y_true_unnorm = self.unnormalize(y_true, min_vals, max_vals)

        # === Metrics Logging (same) ===
        rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
        mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
        bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)
        for i in range(rmse.shape[0]):
            self.log(f"val_rmse_ch_{i+1}", rmse[i].item(), sync_dist=True, on_epoch=True)
            self.log(f"val_mae_ch_{i+1}", mae[i].item(), on_epoch=True, sync_dist=True)
            self.log(f"val_bias_ch_{i+1}", bias[i].item(), on_epoch=True, sync_dist=True)

        # === CSV Outputs ===
        if self.trainer.is_global_zero and not hasattr(self, f"_saved_csv_epoch_{self.current_epoch}"):
            setattr(self, f"_saved_csv_epoch_{self.current_epoch}", True)

            # Save BT predictions
            df_bt = pd.DataFrame({
                "lat_deg": batch.target_lat_deg[bt_mask].cpu().numpy(),
                "lon_deg": batch.target_lon_deg[bt_mask].cpu().numpy(),
                **{f"true_bt_{i+1}": y_true_unnorm[bt_mask, i].cpu().numpy() for i in range(22)},
                **{f"pred_bt_{i+1}": y_pred_unnorm[bt_mask, i].cpu().numpy() for i in range(22)},
            })
            df_bt.to_csv(f"bt_predictions_epoch{self.current_epoch}.csv", index=False)

            # Save Pressure predictions
            df_pressure = pd.DataFrame({
                "lat_deg": batch.target_lat_deg[pressure_mask].cpu().numpy(),
                "lon_deg": batch.target_lon_deg[pressure_mask].cpu().numpy(),
                "true_pressure": y_true_unnorm[pressure_mask, 0].cpu().numpy(),
                "pred_pressure": y_pred_unnorm[pressure_mask, 0].cpu().numpy(),
            })
            df_pressure.to_csv(f"pressure_predictions_epoch{self.current_epoch}.csv", index=False)

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] y_pred mean: {y_pred.mean().item():.6f}, std: {y_pred.std().item():.6f}")

        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[VAL] Averaged metrics across {current_rollout_steps} steps:")
            print(f"  RMSE (avg): {avg_rmse.mean().item():.4f}")
            print(f"  MAE (avg): {avg_mae.mean().item():.4f}")
            if current_rollout_steps > 1:
                print(f"  RMSE degradation: {rmse_degradation.item():.2f}x")

        self.log("val_lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)

        # Cleanup - use correct variable names
        del y_pred_list, ground_truths, step_data_list
        return loss

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

        if hasattr(self, "_mesh_feats_ref") and self._mesh_feats_ref.grad is not None:
            self.debug("[DEBUG] mesh_feats.grad norm:", self._mesh_feats_ref.grad.norm().item())
        else:
            self.debug("[DEBUG] mesh_feats.grad is None → x_hidden is NOT contributing to loss")

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
        # return Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
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
        assert y_pred.device == self.device, f"y_pred not on model device: {y_pred.device} != {self.device}"
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
            w_c = self.channel_weights.get(inst_id_int, torch.ones(y_p.shape[1], device=y_p.device))
            if w_c.shape[0] != y_p.shape[1]:
                pad_len = y_p.shape[1] - w_c.shape[0]
                if pad_len > 0:
                    w_c = torch.cat([w_c, torch.zeros(pad_len, device=w_c.device)])

            # Apply channel mask (default = keep all)
            channel_mask = self.channel_masks.get(inst_id_int, torch.ones_like(w_c, dtype=torch.bool))
            y_p_masked = y_p[:, channel_mask]
            y_t_masked = y_t[:, channel_mask]
            w_c_masked = w_c[channel_mask]

            # Per-channel MSE, then weighted sum
            per_channel_mse = ((y_p_masked - y_t_masked) ** 2).mean(dim=0)
            weighted_loss = (per_channel_mse * w_c_masked).sum()

            if self.trainer.is_global_zero:
                self.debug(f"[DEBUG] Instrument {inst_id_int}: weight={w_i:.3f}, active_channels={channel_mask.sum().item()}")

            loss += w_i * weighted_loss
            total += y_p.shape[0]

        return loss / (total + 1e-8)

    def weighted_huber_loss(self, pred, target, instrument_ids):
        """
        pred, target: [N, C]
        instrument_ids: [N], integer instrument IDs (e.g., 0 for ATMS)
        """
        device = pred.device
        huber_loss = self.huber(pred, target)  # shape [N, C]

        total_loss = 0.0
        for inst_id in instrument_ids.unique():
            mask = instrument_ids == inst_id
            if mask.sum() == 0:
                continue

            inst_id_int = int(inst_id.item())
            weights = self.channel_weights.get(inst_id_int, torch.ones(pred.shape[1], device=device))
            weights = weights.to(device)

            masked_loss = huber_loss[mask] * weights  # shape [M, C]
            total_loss += masked_loss.mean()

        return total_loss / instrument_ids.unique().numel()

    @staticmethod
    def scatter_add_aggregate(values, indices, dim_size):
        """
        Aggregates features using scatter-add along the specified target indices.

        Args:
            values (Tensor): Values to be aggregated, typically encoded features.
            indices (Tensor): Target indices for aggregation (e.g., mesh or target nodes).
            dim_size (int): Size of the output dimension.

        Returns:
            Tensor: Aggregated features (sum).
        """
        return scatter(values, indices, dim=0, dim_size=dim_size, reduce="add")

    def debug(self, *args, **kwargs):
        if self.verbose and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
            print(*args, **kwargs)
