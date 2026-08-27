"""Generic interaction-network message passing layer for OCELOT.

Author: Azadeh Gholoubi
"""

# Third-party
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor
from torch import nn

from ocelot.configs.model_config import InteractionCoderConfig
from ocelot.model import mlp_block


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, coder_config: InteractionCoderConfig):
        """
        Create a new InteractionNet

        coder_config: instance of InteractionCoderConfig containing all necessary hyperparameters.
        """
        assert coder_config.aggr in ("sum", "mean"), f"Unknown aggregation method: {coder_config.aggr}"
        super().__init__(aggr=coder_config.aggr)

        if coder_config.hidden_dim is None:
            # Default to receiver dim if not explicitly given
            hidden_dim = rec_dim

        # Register the edge_index buffer. It will be set dynamically.
        self.register_buffer("edge_index", edge_index, persistent=False)

        # Create MLPs
        # Edge MLP input: [edge_attr (optional), x_j, x_i]
        if update_edges:
            # This assumes edge features have the same dim as receiver nodes
            edge_mlp_input_dim = send_dim + rec_dim + rec_dim
        else:
            edge_mlp_input_dim = send_dim + rec_dim
        # Output of edge_mlp is the message, which has hidden_dim
        edge_mlp_recipe = [edge_mlp_input_dim] + [hidden_dim] * (hidden_layers + 1)

        # Aggregation MLP input: [rec_rep, edge_rep_aggr]
        # Output of aggr_mlp is the residual update, which has rec_dim
        aggr_mlp_input_dim = rec_dim + hidden_dim
        aggr_mlp_recipe = (
            [aggr_mlp_input_dim] + [hidden_dim] * hidden_layers + [rec_dim]
        )

        if edge_chunk_sizes is None:
            self.edge_mlp = mlp_block.make(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [mlp_block.make(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = mlp_block.make(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [mlp_block.make(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        # Add normalization for receiver nodes
        self.rec_norm = nn.LayerNorm(rec_dim)

        self.update_edges = update_edges

    def forward(self, send_rep, rec_rep, edge_rep):
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
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        # By passing a tuple of tensors as `x`, we tell PyG to treat this as a
        # bipartite graph, where edge_index[0] maps to x[0] (send_rep) and
        # edge_index[1] maps to x[1] (rec_rep). This is the correct way to
        # handle message passing between two distinct sets of nodes.
        # Convert edge_index to a SparseTensor to ensure propagate calls the
        # message_and_aggregate method. This is the modern PyG API.
        adj_t = SparseTensor.from_edge_index(
            self.edge_index, sparse_sizes=(send_rep.shape[0], rec_rep.shape[0])
        )

        # propagate now uses the SparseTensor and dispatches to message_and_aggregate
        edge_rep_aggr, edge_diff = self.propagate(adj_t, x=(send_rep, rec_rep))
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff
        rec_rep = self.rec_norm(rec_rep)  # Normalize updated receiver nodes

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message_and_aggregate(self, adj_t, x):
        """
        Fused message and aggregation step.
        This is called by propagate and is more efficient than separate
        message and aggregate calls.
        """
        # In a bipartite graph, x_j is from the source nodes (x[0]) and x_i is from the destination nodes (x[1]).
        # We need to gather the features for each edge based on the indices in the sparse tensor.
        x_j = x[0][adj_t.storage.row()]
        x_i = x[1][adj_t.storage.col()]
        # --- Debugging ---
        print(f"  - Inside InteractionNet:")
        print(f"    - x_j (sender features per edge) shape: {x_j.shape}")
        print(f"    - x_i (receiver features per edge) shape: {x_i.shape}")
        # --- End Debugging ---
        # We don't have edge_attr in this specific bipartite case, but this is where it would be accessed.

        # Create messages
        if self.update_edges:
            # This path is not used by our current model but is kept for completeness
            messages = self.edge_mlp(
                torch.cat((torch.zeros_like(x_j), x_j, x_i), dim=-1)
            )  # Assuming dummy edge_attr
        else:
            messages = self.edge_mlp(torch.cat((x_j, x_i), dim=-1))

        # Aggregate messages
        # Aggregate messages at the destination nodes, which correspond to `col` in the SparseTensor.
        # The output size of the aggregation should be the number of receiver nodes.
        aggregated_messages = self.aggr_module(
            messages, adj_t.storage.col(), dim_size=x[1].size(0)
        )
        return aggregated_messages, messages  # Return both for node and edge updates

    def edge_features(
        self,
        data: HeteroData,
        edge_type,
        edge_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns per-edge features in bipartite_edge_attr_dim (raw spatial edge_attr)."""
        E = int(edge_index.size(1))

        # Debug printing: show whether we used real edge_attr or fell back to zeros.
        # Gated by verbose + global_zero and printed at most once per (edge_type, reason).
        def _maybe_print(reason: str, edge_rep_tensor: torch.Tensor | None = None) -> None:
            if not getattr(self, "verbose", False):
                return
            if not self._is_global_zero_safe():
                return
            if not hasattr(self, "_edge_attr_debug_seen") or self._edge_attr_debug_seen is None:
                self._edge_attr_debug_seen = set()
            key = (tuple(edge_type) if isinstance(edge_type, (list, tuple)) else str(edge_type), str(reason))
            if key in self._edge_attr_debug_seen:
                return
            self._edge_attr_debug_seen.add(key)

            msg = f"[EDGE_ATTR] edge_type={edge_type} E={E} used={'edge_attr' if reason == 'ok' else 'zeros'} reason={reason}"
            if edge_rep_tensor is not None and torch.is_tensor(edge_rep_tensor) and edge_rep_tensor.numel() > 0:
                try:
                    t = edge_rep_tensor.detach()
                    mean_v = t.mean().item()
                    std_v = t.std(unbiased=False).item()
                    min_v = t.min().item()
                    max_v = t.max().item()
                    msg += (
                        f" edge_attr_shape={tuple(t.shape)} "
                        f"mean={mean_v:.4g} std={std_v:.4g} min={min_v:.4g} max={max_v:.4g}"
                    )
                except Exception:
                    msg += f" edge_attr_shape={tuple(edge_rep_tensor.shape)}"
            print(msg)

        _maybe_print("disabled")
        return torch.zeros((E, 4), device=device, dtype=dtype)


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
