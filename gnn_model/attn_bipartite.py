from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class BipartiteGAT(nn.Module):
    """
    Multi-layer GATv2 for bipartite edges (src -> dst).
    Mirrors InteractionNet's interface: forward(send_rep, rec_rep, edge_rep, edge_index)

    Args:
      send_dim: feature dim on source nodes (obs or mesh)
      rec_dim:  feature dim on destination nodes (mesh or target)
      hidden_dim: internal/out dim (kept constant across layers)
      layers: number of stacked GAT layers
      heads: attention heads per layer
      dropout: dropout inside attention/FFN
      edge_dim: dimension of per-edge attributes (optional); if given, used by GATv2
    """
    def __init__(
        self,
        send_dim: int,
        rec_dim: int,
        hidden_dim: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        in_src = send_dim
        in_dst = rec_dim
        for li in range(layers):
            conv = GATv2Conv(
                in_channels=(in_src, in_dst),   # bipartite (src,dst)
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                concat=False,                   # shape = [N_dst, hidden_dim]
                edge_dim=edge_dim,              # use edge_attr in attention if provided
                share_weights=False,
                add_self_loops=False,           # we are bipartite; no self loops
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

            # after first layer, both sides live in hidden_dim
            in_src = hidden_dim
            in_dst = hidden_dim

        # if the very first dst dim != hidden_dim, build a projection for residual
        self.res_proj = (
            nn.Linear(rec_dim, hidden_dim)
            if rec_dim != hidden_dim else nn.Identity()
        )

    @property
    def edge_index(self):
        # kept only for API parity with your InteractionNet usage
        # where you set encoder.edge_index = ...
        return getattr(self, "_edge_index", None)

    @edge_index.setter
    def edge_index(self, ei):
        self._edge_index = ei

    def forward(
        self,
        send_rep: torch.Tensor,   # [N_src, F_src]
        rec_rep: torch.Tensor,    # [N_dst, F_dst]
        edge_rep: Optional[torch.Tensor] = None,  # [E, edge_dim] or None
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = self._edge_index

        x_src, x_dst = send_rep, rec_rep
        res0 = self.res_proj(x_dst)

        for conv, norm in zip(self.layers, self.norms):
            x_dst_new = conv((x_src, x_dst), edge_index, edge_rep)
            x_dst_new = norm(x_dst_new + res0)      # pre-norm residual
            x_dst_new = self.dropout(x_dst_new)
            # next layer: both sides are hidden_dim
            x_src, x_dst, res0 = x_src, x_dst_new, x_dst_new

        return x_dst
