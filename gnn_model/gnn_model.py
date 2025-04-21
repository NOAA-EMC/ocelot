import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_mean
from torch.utils.data.distributed import DistributedSampler

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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16, lr=1e-4):
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

        # Encoder: Maps data nodes to hidden nodes
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.processor_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self.loss_fn = nn.MSELoss()

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            print("Starting new training epoch...")
        train_loader = self.trainer.train_dataloader
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(self.current_epoch)

    @staticmethod
    def unnormalize(tensor, min_vals, max_vals):
        return tensor * (max_vals - min_vals) + min_vals

    def forward(self, data):
        x, edge_index_encoder, edge_attr_encoder, edge_index_processor, edge_index_decoder, y = (
            data.x, data.edge_index_encoder, data.edge_attr_encoder,
            data.edge_index_processor, data.edge_index_decoder, data.y)

        # === Encoding: obs → mesh ===
        src_encoder = edge_index_encoder[0]
        tgt_encoder = edge_index_encoder[1]
        obs_feats = x[src_encoder]
        edge_feats = data.edge_attr_encoder.unsqueeze(1)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        encoder_input = torch.cat([obs_feats, edge_feats], dim=1)
        encoder_input.requires_grad_(True)
        encoded = self.encoder_mlp(encoder_input)

        x_hidden = scatter_mean(encoded, tgt_encoder, dim=0, dim_size=x.shape[0])
        x_hidden = F.relu(x_hidden)

        # === Processor: mesh ↔ mesh ===
        x_hidden.requires_grad_(True)

        for i in range(0, len(self.processor_layers), 2):
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

        # === Decoder: Hidden → Target ===
        src_decoder = edge_index_decoder[0]
        tgt_decoder = edge_index_decoder[1]
        global_ids = data.global_mesh_node_ids

        # Ensure all decoder targets are present in this batch's mesh node IDs
        id_map = {g.item(): i for i, g in enumerate(global_ids)}
        missing_ids = [g.item() for g in tgt_decoder if g.item() not in id_map]
        assert not missing_ids, f"Some decoder target IDs not in global_mesh_node_ids: {missing_ids[:5]}"

        # Map global target IDs to local mesh node indices
        tgt_decoder_local = torch.tensor([id_map[g.item()] for g in tgt_decoder], device=tgt_decoder.device)

        mesh_feats = x_hidden[src_decoder]
        dist_feats = data.edge_attr_decoder.unsqueeze(1)
        decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)
        decoder_input.requires_grad_(True)
        decoded = self.decoder_mlp(decoder_input)

        weights = 1.0 / (dist_feats + 1e-8)
        weighted = decoded * weights
        assert tgt_decoder.max().item() < y.shape[0], "Decoder index out of bounds"

        norm_weights = scatter_add(weights, tgt_decoder_local, dim=0, dim_size=y.shape[0])
        x_out = scatter_add(weighted, tgt_decoder_local, dim=0, dim_size=y.shape[0])
        x_out = x_out / (norm_weights + 1e-8)

        return x_out

    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        y_pred = self(batch)
        lat_lon_meta = batch.y[: y_pred.shape[0], :2].cpu().numpy()
        y_pred.detach().cpu().numpy()
        y_true = batch.y
        loss = self.loss_fn(y_pred, y_true)

        self.log("train_loss", loss)

        if self.trainer.is_global_zero and batch_idx % 2 == 0:
            print(
                f"[Rank 0 GPU Mem] Step {batch_idx} | "
                f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | "
                f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch.y
        loss = self.loss_fn(y_pred, y_true)
        self.log("val_loss", loss.detach(), prog_bar=True, sync_dist=True, on_epoch=True)

        min_vals = batch.target_scaler_min.to(self.device)
        max_vals = batch.target_scaler_max.to(self.device)
        y_pred_unnorm = self.unnormalize(y_pred, min_vals, max_vals)
        y_true_unnorm = self.unnormalize(y_true, min_vals, max_vals)

        rmse = torch.sqrt(F.mse_loss(y_pred, y_true, reduction="none")).mean(dim=0)
        mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
        bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)
        for i, rmse_val in enumerate(rmse):
            self.log(f"val_rmse_ch_{i+1}", rmse_val.item(), sync_dist=True, on_epoch=True)
            self.log(f"val_mae_ch_{i+1}", mae[i], on_epoch=True, sync_dist=True)
            self.log(f"val_bias_ch_{i+1}", bias[i], on_epoch=True, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            import pandas as pd
            df = pd.DataFrame(
                {
                    "lat": batch.y[:, 0].cpu().numpy(),
                    "lon": batch.y[:, 1].cpu().numpy(),
                    **{f"true_bt_{i+1}": y_true_unnorm[:, i].cpu().numpy() for i in range(22)},
                    **{f"pred_bt_{i+1}": y_pred_unnorm[:, i].cpu().numpy() for i in range(22)},
                }
            )
            df.to_csv("bt_predictions.csv", index=False)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)