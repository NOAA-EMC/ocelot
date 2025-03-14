import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
import torch
import torch.nn.functional as F


class GNNModel(nn.Module):
    """
    A Graph Neural Network (GNN) model for processing structured spatial data.

    The model consists of:
    - An encoder that maps input data nodes to a hidden representation.
    - A processor with multiple GATConv layers for message passing between hidden nodes.
    - A decoder that maps hidden node embeddings back to target node predictions.

    Attributes: input_dim (int): Dimension of the input node features.
                hidden_dim (int): Dimension of the hidden layer representations.
                output_dim (int): Dimension of the output predictions.
                num_layers (int): Number of message-passing layers (default: 16).
                encoder (GCNConv): Graph convolution layer for initial feature encoding.
                processor_layers (nn.ModuleList): List of GATConv layers for message passing.
                decoder (GCNConv): Graph convolution layer for decoding hidden node features to output space.

    Methods:
        forward(data):
            Defines the forward pass of the model, applying the encoder, processor, and decoder.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16):
        """
        Initializes the GNN model.

        Parameters:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of hidden dimensions in the graph layers.
            output_dim (int): Number of output features per node.
            num_layers (int, optional): Number of GATConv message-passing layers (default: 16).
        """
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Encoder: Maps data nodes to hidden nodes
        self.encoder = GCNConv(input_dim, hidden_dim)

        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Hidden node connections
    
        # Encoding: Input → Hidden
        x_hidden = self.encoder(x, edge_index)
        x_hidden = F.relu(x_hidden)
    
        # Message Passing: Hidden ↔ Hidden
        for layer in self.processor_layers:
            x_hidden = layer(x_hidden, edge_index)
            x_hidden = F.relu(x_hidden)
    
        # Decoding: Hidden → Target (Use Target Edge Index)
        target_indices = torch.unique(data.edge_index_target[1])  # Get target node indices
        x_out = self.decoder(x_hidden, data.edge_index_target)  # Decode using target edges
        x_out = x_out[target_indices]  # Extract only target predictions
    
        return x_out
