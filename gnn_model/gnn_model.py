import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_mean


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

    def __init__(self, input_dim, hidden_dim, output_dim, num_instruments, num_layers=16, lr=1e-4):
        """
        Initializes the GNNLightning model with an encoder, processor, and decoder.

        Parameters:
        input_dim (int): Number of input features per observation node (including instrument IDs).
        hidden_dim (int): Size of the hidden representation used in all layers.
        output_dim (int): Number of features to predict at each target node.
        num_instruments (int): Number of unique instruments in the dataset.
        num_layers (int, optional): Number of GATConv layers in the processor block (default: 16).
        lr (float, optional): Learning rate for the optimizer (default: 1e-4).
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_instruments = num_instruments

        # Encoder: Maps flattened data nodes to hidden nodes
        # Input includes features and instrument IDs
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for edge distance
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processor: Message passing between mesh nodes
        # Uses the same hidden dimension for all nodes
        self.processor_layers = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Decoder: Maps hidden nodes to target predictions
        # Separate decoders for temperature and salinity
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for edge distance
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Loss function
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

    def forward(self, data):
        if torch.cuda.is_available():
            print(f"[Forward] Batch x shape: {data.x.shape}")
            print(f"[Forward] CUDA Mem: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

        # Get flattened input features (includes instrument IDs)
        x = data.x
        edge_index_encoder = data.edge_index_encoder
        edge_index_processor = data.edge_index_processor
        edge_index_decoder = data.edge_index_decoder
        y = data.y

        # === Encoding: obs → mesh ===
        # Extract source and target indices for encoder edges
        src_encoder = edge_index_encoder[0]
        tgt_encoder = edge_index_encoder[1]
        
        # Get observation features at source nodes
        obs_feats = x[src_encoder]  # Already includes instrument IDs
        
        # Add edge distance features
        edge_feats = data.edge_attr_encoder.unsqueeze(1)  # Shape [E, 1]
        encoder_input = torch.cat([obs_feats, edge_feats], dim=1)

        # Encode observations into hidden space
        encoded = self.encoder_mlp(encoder_input)
        mesh_feats = scatter_mean(encoded, tgt_encoder, dim=0, dim_size=data.num_mesh_nodes)

        # === Processing: mesh ↔ mesh ===
        # Apply graph attention layers for message passing
        hidden = mesh_feats
        for layer in self.processor_layers:
            hidden = layer(hidden, edge_index_processor)
            hidden = F.relu(hidden)
        # === Processing: mesh ↔ mesh ===
        # Process layers in pairs for memory efficiency with checkpointing
        assert len(self.processor_layers) % 2 == 0, "Number of processor layers must be even."
        hidden.requires_grad_(True)

        for i in range(0, len(self.processor_layers), 2):
            layer1 = self.processor_layers[i]
            layer2 = self.processor_layers[i + 1]

            def processor_block(x, edge_index):
                x = layer1(x, edge_index)
                x = F.relu(x)
                x = layer2(x, edge_index)
                return F.relu(x)

            hidden = checkpoint(processor_block, hidden, edge_index_processor)

        if self.trainer.is_global_zero:
            print("[After Processing]", torch.cuda.memory_allocated() / 1e9, "GB")

        # === Decoding: mesh → target ===
        # Extract source mesh nodes and target nodes
        src_decoder = edge_index_decoder[0]
        tgt_decoder = edge_index_decoder[1]
        
        # Get hidden features at source mesh nodes
        mesh_hidden = hidden[src_decoder]
        
        # Add edge distance features for decoding
        edge_feats_decoder = data.edge_attr_decoder.unsqueeze(1)  # Shape [E, 1]
        decoder_input = torch.cat([mesh_hidden, edge_feats_decoder], dim=1)

        # Decode to target predictions
        decoded = self.decoder_mlp(decoder_input)

        # === Aggregation: mesh → target ===
        # Use inverse distance weighted aggregation
        edge_weights = 1.0 / (data.edge_attr_decoder + 1e-6)
        weighted_preds = decoded * edge_weights.unsqueeze(1)
        
        # Sum weighted predictions and normalize
        x_out = scatter_add(weighted_preds, tgt_decoder, dim=0)
        weight_sum = scatter_add(edge_weights, tgt_decoder, dim=0).unsqueeze(1)
        x_out = x_out / weight_sum  # [num_targets, output_dim]

        # Make local copies of src and tgt decoder indices to safely apply masking.
        # We will filter out invalid edges (e.g., edges pointing to non-existent target indices)
        # without modifying the original global decoder edge_index stored in the data object.
        src_decoder_local = src_decoder
        tgt_decoder_local = tgt_decoder

        # Decoder input and aggregation
        mesh_feats = x_hidden[src_decoder_local]  # Features of mesh nodes sending messages
        dist_feats = data.edge_attr_decoder.unsqueeze(1)  # Haversine distance

        decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)
        decoded = self.decoder_mlp(decoder_input)
        if self.trainer.is_global_zero:
            print("[After Decoder]", torch.cuda.memory_allocated() / 1e9, "GB")

        # === Weighted aggregation using inverse distance ===
        # Aggregate per target using scatter mean
        weights = 1.0 / (dist_feats + 1e-8)  # Avoid division by zero
        weighted = decoded * weights

        # Sanity check
        assert tgt_decoder.max().item() < y.shape[0], "Decoder index out of bounds"

        # Remove any edges that point to invalid target indices before aggregation.
        mask = tgt_decoder_local < y.shape[0]
        tgt_decoder_local = tgt_decoder_local[mask]
        src_decoder_local = src_decoder_local[mask]
        dist_feats = dist_feats[mask]
        decoded = decoded[mask]

        # Aggregate using scatter
        norm_weights = scatter_add(weights, tgt_decoder_local, dim=0, dim_size=y.shape[0])
        x_out = scatter_add(weighted, tgt_decoder_local, dim=0, dim_size=y.shape[0])
        x_out = x_out / (norm_weights + 1e-8)

        if self.trainer.is_global_zero:
            print(torch.cuda.memory_summary())

        return x_out

    def training_step(self, batch, batch_idx):
        # === Step 1: Enable anomaly detection and monitor memory
        if torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        # === Step 2: Run forward pass with flattened data
        y_pred = self(batch)
        y_true = batch.y

        # === Step 3: Verify predictions
        # Check shape compatibility
        assert y_pred.shape == y_true.shape, (
            f"Shape mismatch - Predictions: {y_pred.shape}, Targets: {y_true.shape}"
        )

        # Log shapes for debugging
        rank = self.trainer.global_rank if hasattr(self.trainer, "global_rank") else 0
        print(f"[Rank {rank}] Target shape: {y_true.shape}, Prediction shape: {y_pred.shape}")

        # Check for NaN values
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print(f"NaNs detected in batch {batch_idx} on rank {self.trainer.global_rank}")
            print(f"Predictions - Min: {y_pred.min()}, Max: {y_pred.max()}")
            print(f"Targets - Min: {y_true.min()}, Max: {y_true.max()}")

        # === Step 4: Compute loss and log it
        loss = self.loss_fn(y_pred, y_true)
        self.log("train_loss", loss)

        # === Step 5: Optional memory usage print (every 2 batches by rank 0 only)
        if self.trainer.is_global_zero and batch_idx % 2 == 0:
            print(
                f"[Rank 0 GPU Mem] Step {batch_idx} | "
                f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | "
                f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        # === Step 6: Clean up memory (free GPU memory manually)
        del y_pred, y_true, batch

        # === Step 7: Return loss for Lightning to perform backward + optimizer step
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Forward pass with flattened data
        y_pred = self(batch)
        y_true = batch.y
        loss = self.loss_fn(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        # Unnormalize predictions and targets
        min_vals = batch.target_scaler_min.to(self.device)
        max_vals = batch.target_scaler_max.to(self.device)
        y_pred_unnorm = self.unnormalize(y_pred, min_vals, max_vals)
        y_true_unnorm = self.unnormalize(y_true, min_vals, max_vals)

        # Calculate metrics for each target variable
        rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
        mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
        bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)

        # Log metrics per target variable
        for i in range(rmse.shape[0]):
            self.log(f"val_rmse_ch_{i+1}", rmse[i].item(), sync_dist=True, on_epoch=True)
            self.log(f"val_mae_ch_{i+1}", mae[i].item(), on_epoch=True, sync_dist=True)
            self.log(f"val_bias_ch_{i+1}", bias[i].item(), on_epoch=True, sync_dist=True)

        # Save predictions for visualization (once per epoch)
        if self.trainer.is_global_zero and not hasattr(self, "_saved_csv"):
            import pandas as pd

            # Create DataFrame with geographical coordinates and predictions
            df = pd.DataFrame(
                {
                    "lat_deg": batch.target_lat_deg.cpu().numpy(),
                    "lon_deg": batch.target_lon_deg.cpu().numpy(),
                    **{f"true_bt_{i+1}": y_true_unnorm[:, i].cpu().numpy() for i in range(y_true_unnorm.shape[1])},
                    **{f"pred_bt_{i+1}": y_pred_unnorm[:, i].cpu().numpy() for i in range(y_pred_unnorm.shape[1])},
                }
            )
            df.to_csv(f"predictions_epoch{self.current_epoch}.csv", index=False)
            self._saved_csv = True

        # Cleanup
        del y_pred, y_true, y_pred_unnorm, y_true_unnorm, batch
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
