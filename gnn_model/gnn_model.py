import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
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

        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

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

    def forward(self, data):
        (
            x,
            edge_index_encoder,
            edge_attr_encoder,
            edge_index_processor,
            edge_index_decoder,
            y,
        ) = (
            data.x,
            data.edge_index_encoder,
            data.edge_attr_encoder,
            data.edge_index_processor,
            data.edge_index_decoder,
            data.y,
        )

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

        # === Processor: mesh ↔ mesh ===
        for layer in self.processor_layers:
            x_hidden = layer(x_hidden, edge_index_processor)
            x_hidden = F.relu(x_hidden)

        # Decoding: Hidden → Target (MLP using mesh → target edges with edge_attr)
        src_decoder = edge_index_decoder[0]  # mesh node indices
        tgt_decoder = edge_index_decoder[1]  # target node indices
        mesh_feats = x_hidden[src_decoder]  # Features of mesh nodes sending messages
        dist_feats = data.edge_attr_decoder.unsqueeze(1)  # Haversine distance

        decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)
        decoded = self.decoder_mlp(decoder_input)

        # === Weighted aggregation using inverse distance ===
        # Aggregate per target using scatter mean
        weights = 1.0 / (dist_feats + 1e-8)  # Avoid division by zero
        weighted = decoded * weights

        # Normalize by total weights per target
        norm_weights = scatter_add(weights, tgt_decoder, dim=0, dim_size=y.shape[0])
        x_out = scatter_add(weighted, tgt_decoder, dim=0, dim_size=y.shape[0])
        x_out = x_out / (norm_weights + 1e-8)

        return x_out

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        # === Extract metadata (lat/lon) and predictions ===
        lat_lon_meta = batch.y[: y_pred.shape[0], :2].cpu().numpy()    # for evaluation or trace-back
        pred_bt = y_pred.detach().cpu().numpy()
        y_true = batch.y[: y_pred.shape[0], 2:] # Skip lat/lon, use BTs only
        loss = self.loss_fn(y_pred, y_true)
        self.log("train_loss", loss)  # prog_bar=True)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     y_pred = self(batch)
    #     y_true = batch.y  # batch.y[:y_hat.shape[0], :]
    #     loss = self.loss_fn(y_pred, y_true)
    #     self.log("val_loss", loss)
    #     return {'val_loss': loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
