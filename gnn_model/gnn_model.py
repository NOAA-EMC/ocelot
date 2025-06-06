import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_mean

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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16, lr=1e-4, max_rollout_steps=1, rollout_schedule='step'):
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
        self.save_hyperparameters()
        self.lr = lr
        self.max_rollout_steps = max_rollout_steps  # MK: goal is 3 days
        self.rollout_schedule = rollout_schedule    # MK

        # Encoder: Maps data nodes to hidden nodes
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"=== Starting Epoch {self.current_epoch} ===")
        train_loader = self.trainer.train_dataloader
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(self.current_epoch)

    @staticmethod
    def unnormalize(tensor, min_vals, max_vals):
        return tensor * (max_vals - min_vals) + min_vals

    # MK: add latent rollout function
    def forward(self, data, n_steps=1):  # MK add n_steps=1
        if torch.cuda.is_available():
            print(f"[Forward] Batch x shape: {data.x.shape}")
            print(f"[Forward] CUDA Mem: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

        x = data.x
        edge_index_encoder = data.edge_index_encoder
        edge_index_processor = data.edge_index_processor
        #edge_index_decoder = data.edge_index_decoder
        y = data.y

        # === Encoding: obs → mesh ===
        src_encoder = edge_index_encoder[0]
        tgt_encoder = edge_index_encoder[1]
        obs_feats = x[src_encoder]
        edge_feats = data.edge_attr_encoder.unsqueeze(1)  # Shape [E, 1]
        encoder_input = torch.cat([obs_feats, edge_feats], dim=1)

        # Pass through MLP
        encoded = self.encoder_mlp(encoder_input)
        x_hidden = scatter_mean(encoded, tgt_encoder, dim=0, dim_size=x.shape[0])
        x_hidden = F.relu(x_hidden)
        if self.trainer.is_global_zero:
            print("[After Encoding]", torch.cuda.memory_allocated() / 1e9, "GB")

        # MK: Get both ground truth and target locations for all steps
        if n_steps > 1:
            ground_truths, target_locations_list = self._get_sequential_targets_and_locations(data, n_steps)
        else:
            # Single step: use original data
            ground_truths = [data.y]
            target_locations_list = [data.target_metadata[:, :2]]

        predictions = []  # MK: add empty list to store predictions

        # Rollout loop # MK: add for loop to go through steps
        for step in range(n_steps):
            if step > 0 and self.trainer.is_global_zero:
                print(f"[Rollout Step {step+1}] Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            # MK: indentation; code stays the same
            # === Processor: mesh ↔ mesh ===
            x_hidden.requires_grad_(True)
            # We process two processor layers at a time to improve memory efficiency with checkpointing.
            # Therefore, the number of processor layers must be even to avoid leaving a layer unpaired.
            # Assert here ensures that each iteration handles exactly two layers.
            for i in range(0, len(self.processor_layers), 2):
                assert len(self.processor_layers) % 2 == 0, "Processor layers must be even for paired execution"
                layer1 = self.processor_layers[i]
                if self.training:
                    x_hidden = checkpoint(layer1, x_hidden, edge_index_processor)
                else:
                    x_hidden = layer1(x_hidden, edge_index_processor)
                x_hidden = F.relu(x_hidden)

                if i + 1 < len(self.processor_layers):
                    layer2 = self.processor_layers[i + 1]
                    if self.training:
                        x_hidden = checkpoint(layer2, x_hidden, edge_index_processor)
                    else:
                        x_hidden = layer2(x_hidden, edge_index_processor)
                    x_hidden = F.relu(x_hidden)

            x_hidden = x_hidden.detach().requires_grad_()
            if self.trainer.is_global_zero:
                print("[After Processor]", torch.cuda.memory_allocated() / 1e9, "GB")

            # === Decoder: Hidden → Target ===
            # MK === Dynamic Decoder: Use target locations for this timestep ===
            target_latlon_rad = target_locations_list[step]

            if step == 0:
                # First step: use original decoder edges
                edge_index_decoder = data.edge_index_decoder
                edge_attr_decoder = data.edge_attr_decoder
                print(f"[DEBUG] Step {step}: Using original decoder, target nodes: {torch.unique(edge_index_decoder[1]).shape}")
            else:
                # Later steps: create new decoder edges for this timestep
                edge_index_decoder, edge_attr_decoder = self._create_decoder_for_step(
                    x_hidden, target_latlon_rad
                )
                print(f"[DEBUG] Step {step}: Using new decoder, target nodes: {torch.unique(edge_index_decoder[1]).shape}")

            src_decoder = edge_index_decoder[0]  # mesh node indices
            tgt_decoder = edge_index_decoder[1]  # target node indices

            # Make local copies of src and tgt decoder indices to safely apply masking.
            # We will filter out invalid edges (e.g., edges pointing to non-existent target indices)
            # without modifying the original global decoder edge_index stored in the data object.
            src_decoder_local = src_decoder
            tgt_decoder_local = tgt_decoder

            # Decoder input and aggregation
            mesh_feats = x_hidden[src_decoder_local]  # Features of mesh nodes sending messages
            # dist_feats = data.edge_attr_decoder.unsqueeze(1)  # Haversine distance
            # MK: update
            dist_feats = edge_attr_decoder.unsqueeze(1)  # Haversine distance

            decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)
            decoded = self.decoder_mlp(decoder_input)
            if self.trainer.is_global_zero:
                print("[After Decoder]", torch.cuda.memory_allocated() / 1e9, "GB")

            # === Weighted aggregation using inverse distance ===
            # Aggregate per target using scatter mean
            weights = 1.0 / (dist_feats + 1e-8)  # Avoid division by zero
            weighted = decoded * weights

            # Sanity check
            # assert tgt_decoder.max().item() < y.shape[0], "Decoder index out of bounds"
            # MK: update for rollout
            current_target_size = target_locations_list[step].shape[0]
            assert tgt_decoder.max().item() < current_target_size, "Decoder index out of bounds"


            # Remove any edges that point to invalid target indices before aggregation.
            mask = tgt_decoder_local < current_target_size  # MK: [original] y.shape[0]
            tgt_decoder_local = tgt_decoder_local[mask]
            src_decoder_local = src_decoder_local[mask]
            dist_feats = dist_feats[mask]
            decoded = decoded[mask]
            # MK: add these to avoid any tensor mismatch
            weighted = weighted[mask]
            weights = weights[mask]

            # Aggregate using scatter
            # norm_weights = scatter_add(weights, tgt_decoder_local, dim=0, dim_size=y.shape[0])
            # x_out = scatter_add(weighted, tgt_decoder_local, dim=0, dim_size=y.shape[0])
            # MK: update for rollout
            current_target_size = target_locations_list[step].shape[0]
            norm_weights = scatter_add(weights, tgt_decoder_local, dim=0, dim_size=current_target_size)
            x_out = scatter_add(weighted, tgt_decoder_local, dim=0, dim_size=current_target_size)

            x_out = x_out / (norm_weights + 1e-8)

            if self.trainer.is_global_zero:
                print(torch.cuda.memory_summary())

            # MK: store predictions
            print(f"[DEBUG] Step {step}: Final prediction shape: {x_out.shape}")
            predictions.append(x_out)

        return predictions  # MK: return predictions (list) instead of x_out

    # MK: simple rollout steps for now
    # Graphcast: 300,000 gradient descent updates - 1 autoregressive
    #            300,001 to 311,000: add 1 per 1000 updates
    #           (i.e., use 1000 steps for each autoregressive step)
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
            # testing functionality: train 1 rollout for 5 epochs [0-4], add 1 for every epoch
            threshold = 5  # 300000 # MK: using 5 for testing
            interval  = 1  # 1000
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
    def _create_decoder_for_step(self, mesh_state, target_latlon_rad):
        """
        Create decoder edges connecting mesh nodes to target locations for this step.
        """
        from mesh_to_target import MeshTargetKNNConnector

        # Get mesh locations (you'll need to store this from the data module)
        data_module = self.trainer.datamodule
        mesh_latlon_rad = data_module.mesh_latlon_rad

        # Debug: verify we have access to what we need
        print(f"[DEBUG DECODER] mesh_latlon_rad shape: {mesh_latlon_rad.shape}")
        print(f"[DEBUG DECODER] target_latlon_rad shape: {target_latlon_rad.shape}")
        print(f"[DEBUG DECODER] mesh_graph nodes: {data_module.mesh_graph.number_of_nodes()}")
        print(f"[DEBUG DECODER] mesh_state shape: {mesh_state.shape}")

        # Create KNN connector (same parameters as in your data module)
        knn_decoder = MeshTargetKNNConnector(num_nearest_neighbours=3)

        # Create edges for this timestep's target locations
        edge_index, edge_attr = knn_decoder.add_edges(
            mesh_graph=None,  # not used: data_module.mesh_graph,  # Pass the mesh graph
            target_latlon_rad=target_latlon_rad.cpu().numpy(),  # Convert to numpy
            mesh_latlon_rad=mesh_latlon_rad  # Already numpy from data module
        )
 
        # Calculate mesh offset
        num_mesh_nodes = len(mesh_latlon_rad)  # 642
        num_obs_nodes = mesh_state.shape[0] - num_mesh_nodes  # 28294 - 642 = 27652
        mesh_offset = num_obs_nodes
        
        # Apply mesh offset to point to actual mesh nodes in mesh_state/x_hidden
        edge_index_global = edge_index.clone()
        edge_index_global[0] += mesh_offset  # Shift mesh indices

        print(f"[DEBUG] Applied mesh offset {mesh_offset}")
        print(f"[DEBUG] Mesh indices after offset: {edge_index_global[0].min()} to {edge_index_global[0].max()}")
        print(f"[DEBUG] Expected mesh range: {mesh_offset} to {mesh_state.shape[0]-1}")

        return edge_index_global.to(mesh_state.device), edge_attr.to(mesh_state.device)

    # MK
    def _get_sequential_ground_truth(self, batch, n_steps):
        # Debug: Let's see what bin_name actually is
        print(f"[DEBUG] batch.bin_name type: {type(batch.bin_name)}")
        print(f"[DEBUG] batch.bin_name value: {batch.bin_name}")

        if hasattr(batch.bin_name, '__len__'):
            print(f"[DEBUG] batch.bin_name length: {len(batch.bin_name)}")
            if len(batch.bin_name) > 0:
                print(f"[DEBUG] batch.bin_name[0] type: {type(batch.bin_name[0])}")
                print(f"[DEBUG] batch.bin_name[0] value: {batch.bin_name[0]}")

        data_module = self.trainer.datamodule
        current_bin_name = batch.bin_name
#        if isinstance(current_bin_name, torch.Tensor):
#            current_bin_name = current_bin_name.item() if current_bin_name.numel() == 1 else str(current_bin_name)
        # bin_name is a list
        if isinstance(current_bin_name, list):
            current_bin_name = current_bin_name[0]  # Take the first element: 'bin1'

        bin_num = int(current_bin_name.replace("bin", ""))

        ground_truths = []
        for step in range(n_steps):
            target_bin_name = f"bin{bin_num + step}"

            if target_bin_name in data_module.data_summary:
                # Get target features for this time step
                target_bin_data = data_module.data_summary[target_bin_name]
                target_features = target_bin_data["target_features_final"]

                # Move to same device as batch
                target_features = target_features.to(batch.y.device)
                ground_truths.append(target_features)
            else:
                if ground_truths:
                    ground_truths.append(ground_truths[-1])
                else:
                    # This shouldn't happen in normal training, but just in case
                    ground_truths.append(batch.y)
                    print(f"Warning: No sequential ground truth found for {target_bin_name}")

        return ground_truths

    # MK:
    def _get_sequential_targets_and_locations(self, batch, n_steps):
        """
        Gets sequential ground truth targets AND target locations for rollout training.

        Returns:
            tuple: (ground_truths, target_locations_list)
                - ground_truths: List of ground truth tensors for each step
                - target_locations_list: List of target location arrays for each step
        """
        data_module = self.trainer.datamodule
        current_bin_name = batch.bin_name[0] if isinstance(batch.bin_name, list) else batch.bin_name
        bin_num = int(current_bin_name.replace("bin", ""))

        ground_truths = []
        target_locations_list = []

        for step in range(n_steps):
            target_bin_name = f"bin{bin_num + step}"

            if target_bin_name in data_module.data_summary:
                target_bin_data = data_module.data_summary[target_bin_name]

                # Get ground truth features
                target_features = target_bin_data["target_features_final"].to(batch.y.device)
                ground_truths.append(target_features)

                # Get target locations (lat, lon in radians)
                target_metadata = target_bin_data["target_metadata"]
                target_latlon_rad = target_metadata[:, :2]  # lat, lon
                target_locations_list.append(target_latlon_rad)

            else:
                # Fallback: use last available data
                # TODO: Edge handling? update this to be stricker?
                # If we don't have ground truth for this step, repeat the last one
                if ground_truths:
                    ground_truths.append(ground_truths[-1])
                    target_locations_list.append(target_locations_list[-1])
                else:
                    # Ultimate fallback: use original batch data
                    ground_truths.append(batch.y)
                    target_locations_list.append(batch.target_metadata[:, :2])
                    print(f"Warning: No sequential data found for {target_bin_name}")

            # Debug: Check shapes
            print(f"[DEBUG] Step {step}: ground truth shape {target_features.shape}")
            print(f"[DEBUG] Step {step}: target locations shape {target_latlon_rad.shape}")

        return ground_truths, target_locations_list

    def training_step(self, batch, batch_idx):
        # === Step 1: Enable anomaly detection (only once, for debugging backward passes)
        if torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            # Prints how much GPU memory is being used before you run the forward pass.
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        # === Step 2: Run forward pass (predict outputs from input batch)
        # MK: update with rollout steps
        # y_pred = self(batch)
        current_rollout_steps = self.get_current_rollout_steps()

        # MK: Log the tracking info here
        self.log("MK: global_step", self.global_step)
        self.log("MK: rollout_steps", float(current_rollout_steps))

        predictions = self(batch, n_steps=current_rollout_steps)
        # ground_truths = self._get_sequential_ground_truth(batch, current_rollout_steps)

        # Get ground truths (already computed in forward method)
        if current_rollout_steps > 1:
            ground_truths, _ = self._get_sequential_targets_and_locations(batch, current_rollout_steps)
        else:
            ground_truths = [batch.y]

        # y_pred_list = self(batch)
        # y_pred = predictions[0]  # TODO: remove this; keeping it for now so assert OK
        # y_true = batch.y  # TODO: need to update y, for now make y all the same

        # === Step 3: Sanity checks (shape match and NaN checks)
        # assert y_pred.shape == y_true.shape, f"Mismatch: {y_pred.shape} vs {y_true.shape}"

        # rank = self.trainer.global_rank if hasattr(self.trainer, "global_rank") else 0
        # print(f"[Rank {rank}] y shape: {y_true.shape}, pred shape: {y_pred.shape}")

        #if torch.isnan(y_pred).any() or torch.isnan(batch.y).any():
        #    print(f"NaNs detected at batch {batch_idx} on rank {self.trainer.global_rank}")
        #    print(f"y_pred min: {y_pred.min()}, max: {y_pred.max()}")
        #    print(f"y_true min: {batch.y.min()}, max: {batch.y.max()}")


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


        # === Step 4: Compute loss and log it
        # loss = self.loss_fn(y_pred, y_true)
        # MK: weighted mse
        # pressure_levels = None
        # loss = level_weighted_mse(y_pred, y_true, pressure_levels)

        # MK: rollout loss 
        total_loss = 0.0
        for step, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
            if prediction.shape[0] == ground_truth.shape[0]:
                step_loss = self.loss_fn(prediction, ground_truth)
                total_loss += step_loss
                self.log(f"train_loss_step{step+1}", step_loss)
            else:
                print(f"Skipping step {step+1}: pred shape {prediction.shape} != truth shape {ground_truth.shape}")
    
        self.log("train_loss", total_loss)
        self.log("rollout_steps", float(current_rollout_steps))

        # === Step 5: Optional memory usage print (every 2 batches by rank 0 only)
        if self.trainer.is_global_zero and batch_idx % 2 == 0:
            print(
                f"[Rank 0 GPU Mem] Step {batch_idx} | "
                f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | "
                f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        # === Step 6: Clean up memory (free GPU memory manually)
        #del y_pred, y_true, batch
        del batch, predictions, ground_truths

        # === Step 7: Return loss for Lightning to perform backward + optimizer step
        return {"loss": total_loss}  # MK: update to total_loss

    def validation_step(self, batch, batch_idx):
        # MK: small change for a quick test
        #y_pred = self(batch)
        y_pred_list = self(batch, n_steps=1)
        y_pred = y_pred_list[0]
        y_true = batch.y
        loss = self.loss_fn(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        # Unnormalize
        min_vals = batch.target_scaler_min.to(self.device)
        max_vals = batch.target_scaler_max.to(self.device)
        y_pred_unnorm = self.unnormalize(y_pred, min_vals, max_vals)
        y_true_unnorm = self.unnormalize(y_true, min_vals, max_vals)

        # Metrics
        rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
        mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
        bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)
        for i in range(rmse.shape[0]):
            self.log(f"val_rmse_ch_{i+1}", rmse[i].item(), sync_dist=True, on_epoch=True)
            self.log(f"val_mae_ch_{i+1}", mae[i].item(), on_epoch=True, sync_dist=True)
            self.log(f"val_bias_ch_{i+1}", bias[i].item(), on_epoch=True, sync_dist=True)

        # Save CSV for visual inspection
        if self.trainer.is_global_zero and not hasattr(self, "_saved_csv"):
            import pandas as pd

            df = pd.DataFrame(
                {
                    "lat_deg": batch.target_lat_deg.cpu().numpy(),
                    "lon_deg": batch.target_lon_deg.cpu().numpy(),
                    **{f"true_bt_{i+1}": y_true_unnorm[:, i].cpu().numpy() for i in range(22)},
                    **{f"pred_bt_{i+1}": y_pred_unnorm[:, i].cpu().numpy() for i in range(22)},
                }
            )
            df.to_csv(f"bt_predictions_epoch{self.current_epoch}.csv", index=False)
            self._saved_csv = True

        # Cleanup
        del y_pred, y_true, y_pred_unnorm, y_true_unnorm, batch
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
