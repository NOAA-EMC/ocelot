"""Sliding-window transformer processor for latent mesh temporal evolution.

This module defines temporal attention, spatial mixing, and the fixed-mesh
transformer processor used to evolve latent mesh states across rollout steps.

Author: Azadeh Gholoubi
"""

from collections import deque
from typing import Callable, List, Optional
import torch
import torch.nn as nn
import math


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H]
        T = x.size(1)
        return x + self.pe[:, :T, :]


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # allow attend to <= current time only
    # [T, T] with True = -inf mask positions
    # (nn.MultiheadAttention expects attn_mask additive or boolean)
    # Using boolean mask (True = mask)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x_seq: [N, T, H]  (N nodes = batch)
        attn_out, _ = self.attn(
            x_seq, x_seq, x_seq, attn_mask=attn_mask, need_weights=False
        )
        y = self.norm1(x_seq + self.drop(attn_out))
        ff_out = self.ff(y)
        y = self.norm2(y + self.drop(ff_out))
        return y


class SpatialMixBlock(nn.Module):
    """One explicit mesh-neighbor mixing step using `edge_index`.

    This is intentionally simple and fast:
    - aggregate neighbor features with mean
    - transform with an MLP
    - residual + LayerNorm

    Shapes:
      x:         [N, H]
      edge_index:[2, E] (src, dst)
      edge_attr: [E] or [E, F] (optional)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0, edge_chunk_size: int = 16384):
        super().__init__()
        # When `edge_attr` comes from GraphCast-style spatial features, the first
        # columns are normalized spatial features (typically 4 dims: dist + relative xyz).
        # We turn the full feature vector into a bounded positive weight and compute
        # a weighted-mean aggregation.
        # The edge-weighting rule itself is parameter-free; the aggregated message
        # is then processed by the learned MLP below before the residual update.
        self.distance_scale: float = 4.0
        self.edge_chunk_size = max(1, int(edge_chunk_size))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _edge_ranges(self, num_edges: int):
        for start in range(0, int(num_edges), self.edge_chunk_size):
            yield start, min(start + self.edge_chunk_size, int(num_edges))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
        owned_node_count: Optional[int] = None,
        halo_exchange: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        # IMPORTANT: When ``halo_exchange`` is provided, every rank must call
        # it the same number of times per forward pass even if this rank has
        # nothing to mix. Otherwise the collective desyncs and NCCL hangs.
        no_edges = (
            edge_index is None
            or not torch.is_tensor(edge_index)
            or edge_index.numel() == 0
        )
        if no_edges:
            if halo_exchange is not None:
                return halo_exchange(x)
            return x

        # edge_index: [2, E]
        src = edge_index[0]
        dst = edge_index[1]

        N = int(x.size(0))
        owned_nodes = N if owned_node_count is None else int(owned_node_count)
        device = x.device
        dtype = x.dtype

        if edge_attr is None:
            # Aggregate: mean of neighbor source features into dst nodes.
            agg = torch.zeros((owned_nodes, x.size(1)), device=device, dtype=dtype)
            deg = torch.zeros((owned_nodes,), device=device, dtype=dtype)
            for start, end in self._edge_ranges(dst.numel()):
                src_chunk = src[start:end]
                dst_chunk = dst[start:end]
                agg.index_add_(0, dst_chunk, x.index_select(0, src_chunk))
                deg.index_add_(0, dst_chunk, torch.ones((dst_chunk.numel(),), device=device, dtype=dtype))
            agg = agg / deg.clamp(min=1).unsqueeze(-1)
        else:
            # Edge-aware aggregate: weighted mean of neighbor source features.
            # - If edge_attr is [E] or [E,1]: treat as precomputed positive weights.
            # - Else: use the full edge feature vector (all columns) to compute weights.
            agg = torch.zeros((owned_nodes, x.size(1)), device=device, dtype=dtype)
            wsum = torch.zeros((owned_nodes,), device=device, dtype=dtype)
            for start, end in self._edge_ranges(dst.numel()):
                src_chunk = src[start:end]
                dst_chunk = dst[start:end]
                edge_attr_chunk = edge_attr[start:end]

                if edge_attr_chunk.dim() == 1:
                    w_chunk = edge_attr_chunk
                elif edge_attr_chunk.dim() == 2 and edge_attr_chunk.size(1) == 1:
                    w_chunk = edge_attr_chunk[:, 0]
                else:
                    # Compute in fp32 for stability under AMP/fp16.
                    feat = edge_attr_chunk.to(device=device, dtype=torch.float32)
                    feat_norm = torch.sqrt((feat * feat).sum(dim=1) + 1e-12)
                    w_chunk = torch.exp(-float(self.distance_scale) * feat_norm).to(dtype=dtype)

                w_chunk = w_chunk.to(device=device, dtype=dtype).clamp(min=0)
                src_features = x.index_select(0, src_chunk)
                agg.index_add_(0, dst_chunk, src_features * w_chunk.unsqueeze(-1))
                wsum.index_add_(0, dst_chunk, w_chunk)
            agg = agg / wsum.clamp(min=1e-6).unsqueeze(-1)

        msg = self.mlp(agg)
        if owned_nodes == N:
            updated = self.norm(x + self.drop(msg))
            return halo_exchange(updated) if halo_exchange is not None else updated

        updated_owned = self.norm(x[:owned_nodes] + self.drop(msg))
        updated = torch.cat([updated_owned, x[owned_nodes:]], dim=0)
        return halo_exchange(updated) if halo_exchange is not None else updated


class SlidingWindowTransformerProcessor(nn.Module):
    """
    Temporal transformer over a rolling window of latent mesh states.
    Call reset() at the start of each new sequence/bin;
    then call forward() each rollout step.
    """
    def __init__(self,
                 hidden_dim: int,
                 window: int = 4,
                 depth: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 use_causal_mask: bool = True,
                 spatial_mixing_steps: int = 1,
                 spatial_edge_chunk_size: int = 16384):
        super().__init__()
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.spatial_mixing_steps = int(spatial_mixing_steps)
        self.spatial_edge_chunk_size = max(1, int(spatial_edge_chunk_size))
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.posenc = TemporalPositionalEncoding(hidden_dim, max_len=window)
        self.spatial_mix = SpatialMixBlock(
            hidden_dim,
            dropout=dropout,
            edge_chunk_size=self.spatial_edge_chunk_size,
        )
        self.register_buffer("_dummy", torch.empty(0))  # for device inference
        self.cache: deque[torch.Tensor] = deque(maxlen=window)

    def reset(self):
        self.cache.clear()

    @torch.no_grad()
    def warm_start(self, states: List[torch.Tensor]):
        """Optionally pre-fill with historical mesh states
        (no gradient through history)."""
        self.cache.clear()
        for s in states[-self.window:]:
            self.cache.append(s.detach())

    def forward(
        self,
        x_mesh: torch.Tensor,
        mesh_edge_index: Optional[torch.Tensor] = None,
        mesh_edge_attr: Optional[torch.Tensor] = None,
        owned_node_count: Optional[int] = None,
        halo_exchange: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x_mesh: [N_mesh, H] current latent mesh state
        returns: [N_mesh, H] updated latent mesh state
        """
        # ensure device consistency
        device = x_mesh.device
        dtype = x_mesh.dtype

        self.cache.append(x_mesh)
        x_seq = torch.stack(list(self.cache), dim=1).to(
            device=device, dtype=dtype
        )  # [N, T, H]

        # add temporal positional encoding
        x_seq = self.posenc(x_seq)

        # causal mask (time x time), broadcasted across batch
        attn_mask = (
            _causal_mask(x_seq.size(1), device)
            if self.use_causal_mask else None
        )

        for blk in self.blocks:
            x_seq = blk(x_seq, attn_mask)

            # Interleave explicit spatial mixing with temporal layers.
            # Apply mixing per time-slice (T is small: window size).
            if self.spatial_mixing_steps > 0:
                mixed = []
                for t in range(x_seq.size(1)):
                    xt = x_seq[:, t, :]
                    for _ in range(self.spatial_mixing_steps):
                        xt = self.spatial_mix(
                            xt,
                            mesh_edge_index,
                            mesh_edge_attr,
                            owned_node_count=owned_node_count,
                            halo_exchange=halo_exchange,
                        )
                    mixed.append(xt)
                x_seq = torch.stack(mixed, dim=1)

        return x_seq[:, -1, :]
