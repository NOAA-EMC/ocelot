# Third-party
import torch
import torch_geometric as pyg
from torch import nn
from torch_geometric.utils import scatter

# Local
from . import utils


class InteractionNet(nn.Module):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        send_dim,
        rec_dim,
        edge_dim=None,
        update_edges=False,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        """
        Create a new InteractionNet

        edge_index: (2,M), Edges in pyg format
        input_dim: Dimensionality of input representations,
            for both nodes and edges
        edge_dim: Dimensionality of edge features. If provided, edge features are
            concatenated into the message MLP input. If None, edges are ignored.
        update_edges: If new edge representations should be computed
            and returned
        hidden_layers: Number of hidden layers in MLPs
        hidden_dim: Dimensionality of hidden layers, if None then same
            as input_dim
        edge_chunk_sizes: List of chunks sizes to split edge representation
            into and use separate MLPs for (None = no chunking, same MLP)
        aggr_chunk_sizes: List of chunks sizes to split aggregated node
            representation into and use separate MLPs for
            (None = no chunking, same MLP)
        aggr: Message aggregation method (sum/mean)
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__()
        self.aggr = aggr
        self.update_edges = update_edges
        self.use_edge_attr_in_message = edge_dim is not None

        if hidden_dim is None:
            # Default to receiver dim if not explicitly given
            hidden_dim = rec_dim

        if edge_dim is None:
            edge_dim = hidden_dim  # used only when update_edges=True

        # The edge_index is now passed dynamically to the forward method.

        # Create MLPs
        # Edge MLP input: [edge_attr (optional), x_j, x_i]
        if update_edges or self.use_edge_attr_in_message:
            edge_mlp_input_dim = edge_dim + send_dim + rec_dim
        else:
            edge_mlp_input_dim = send_dim + rec_dim
        # Output of edge_mlp is the message, which has hidden_dim
        edge_mlp_recipe = [edge_mlp_input_dim] + [hidden_dim] * (hidden_layers + 1)

        # Aggregation MLP input: [rec_rep, edge_rep_aggr]
        # Output of aggr_mlp is the residual update, which has rec_dim
        aggr_mlp_input_dim = rec_dim + hidden_dim
        aggr_mlp_recipe = [aggr_mlp_input_dim] + [hidden_dim] * hidden_layers + [rec_dim]

        if edge_chunk_sizes is None:
            self.edge_mlp = utils.make_mlp(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [utils.make_mlp(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = utils.make_mlp(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [utils.make_mlp(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        # Add normalization for receiver nodes
        self.rec_norm = nn.LayerNorm(rec_dim)

    def forward(self, send_rep, rec_rep, edge_rep, edge_index):
        """
        Apply interaction network to update the representations of receiver
        nodes, and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations
            of edges
        """
        row, col = edge_index
        x_j = send_rep[row]
        x_i = rec_rep[col]

        if self.update_edges or self.use_edge_attr_in_message:
            edge_inputs = torch.cat([edge_rep, x_j, x_i], dim=-1)
        else:
            edge_inputs = torch.cat([x_j, x_i], dim=-1)

        messages = self.edge_mlp(edge_inputs)

        # Aggregate messages
        edge_rep_aggr = scatter(messages, col, dim=0, dim_size=rec_rep.size(0), reduce=self.aggr)

        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff
        rec_rep = self.rec_norm(rec_rep)  # Normalize updated receiver nodes

        if self.update_edges:
            edge_rep = edge_rep + messages
            return rec_rep, edge_rep

        return rec_rep


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
        """
        Chunk up input and feed through MLPs

        x: (..., N, d), where N = sum(chunk_sizes)

        Returns:
        joined_output: (..., N, d), concatenated results from the MLPs
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
