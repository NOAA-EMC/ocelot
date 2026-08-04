# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Hierarchical sliding-window transformer for multi-scale mesh processing.
# Ported from gnn_model/processor_transformer_hierarchical.py for ocelot.

from __future__ import annotations

from collections import deque
from typing import List, Optional

import torch
import torch.nn as nn

from ocelot.logger import logger


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        return x + self.pe[:, :t, :]


def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            t,
            t,
            dtype=torch.bool,
            device=device),
        diagonal=1)


def _mha_self_chunked(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    """Self-attention with ``batch_first`` on ``x`` shaped ``[N, T, H]``.

    PyTorch treats ``N`` as batch; very large ``N`` (many mesh nodes) can trigger
    ``CUDA error: invalid configuration argument`` in SDPA. Chunking along ``N``
    fixes this without changing semantics (each node is an independent length-``T`` series).
    """
    n = x.size(0)
    if chunk_size <= 0 or n <= chunk_size:
        out, _ = attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out
    parts = []
    for start in range(0, n, chunk_size):
        xc = x[start: start + chunk_size]
        oc, _ = attn(xc, xc, xc, attn_mask=attn_mask, need_weights=False)
        parts.append(oc)
    return torch.cat(parts, dim=0)


def _mha_cross_chunked(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Cross-attention when ``query`` and ``key_value`` share the same leading size ``N``."""
    n = query.size(0)
    if chunk_size <= 0 or n <= chunk_size:
        out, _ = attn(query, key_value, key_value, need_weights=False)
        return out
    parts = []
    for start in range(0, n, chunk_size):
        q = query[start: start + chunk_size]
        kv = key_value[start: start + chunk_size]
        oc, _ = attn(q, kv, kv, need_weights=False)
        parts.append(oc)
    return torch.cat(parts, dim=0)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_chunk_size: int = 2048,
    ):
        super().__init__()
        self.attn_chunk_size = int(attn_chunk_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x_seq: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        attn_out = _mha_self_chunked(
            self.attn, x_seq, attn_mask, self.attn_chunk_size
        )
        y = self.norm1(x_seq + self.drop(attn_out))
        ff_out = self.ff(y)
        return self.norm2(y + self.drop(ff_out))


class CrossScaleAttention(nn.Module):
    """Cross-attention between mesh scales."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_chunk_size: int = 2048,
    ):
        super().__init__()
        self.attn_chunk_size = int(attn_chunk_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key_value: torch.Tensor) -> torch.Tensor:
        attn_out = _mha_cross_chunked(
            self.cross_attn, query, key_value, self.attn_chunk_size
        )
        return self.norm(query + self.drop(attn_out))


class SpatialMixBlock(nn.Module):
    """One within-level neighbor mixing step using ``edge_index``."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.distance_scale: float = 4.0
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index is None:
            return x
        if not torch.is_tensor(edge_index) or edge_index.numel() == 0:
            return x

        src = edge_index[0]
        dst = edge_index[1]
        n = int(x.size(0))
        device = x.device
        dtype = x.dtype

        if edge_attr is None:
            agg = torch.zeros((n, x.size(1)), device=device, dtype=dtype)
            agg.index_add_(0, dst, x[src])
            deg = torch.zeros((n,), device=device, dtype=dtype)
            deg.index_add_(
                0, dst, torch.ones(
                    (dst.numel(),), device=device, dtype=dtype))
            agg = agg / deg.clamp(min=1).unsqueeze(-1)
        else:
            if edge_attr.dim() == 1:
                w = edge_attr
            elif edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                w = edge_attr[:, 0]
            else:
                feat = edge_attr.to(device=device, dtype=torch.float32)
                feat_norm = torch.sqrt((feat * feat).sum(dim=1) + 1e-12)
                w = torch.exp(-float(self.distance_scale)
                              * feat_norm).to(dtype=dtype)

            w = w.to(device=device, dtype=dtype).clamp(min=0)
            agg = torch.zeros((n, x.size(1)), device=device, dtype=dtype)
            agg.index_add_(0, dst, x[src] * w.unsqueeze(-1))
            wsum = torch.zeros((n,), device=device, dtype=dtype)
            wsum.index_add_(0, dst, w)
            agg = agg / wsum.clamp(min=1e-6).unsqueeze(-1)

        msg = self.mlp(agg)
        return self.norm(x + self.drop(msg))


class HierarchicalSlidingWindowTransformer(nn.Module):
    """Temporal transformer over a rolling window at each mesh level + cross-scale attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int = 4,
        window: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        use_cross_scale: bool = True,
        spatial_mixing_steps: int = 1,
        attn_chunk_size: int = 2048,
        spatial_edge_dims: Optional[List[int]] = None,
        up_edge_dims: Optional[List[int]] = None,
        down_edge_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.use_cross_scale = use_cross_scale
        self.spatial_mixing_steps = int(spatial_mixing_steps)
        self.attn_chunk_size = int(attn_chunk_size)

        self.level_transformers = nn.ModuleList()
        for _ in range(num_levels):
            blocks = nn.ModuleList(
                [
                    TemporalBlock(
                        hidden_dim,
                        num_heads,
                        dropout,
                        attn_chunk_size=self.attn_chunk_size,
                    )
                    for _ in range(depth)
                ]
            )
            self.level_transformers.append(blocks)

        self.level_posenc = nn.ModuleList(
            [
                TemporalPositionalEncoding(hidden_dim, max_len=window)
                for _ in range(num_levels)
            ]
        )

        if use_cross_scale:
            self.up_cross_attn = nn.ModuleList(
                [
                    CrossScaleAttention(
                        hidden_dim,
                        num_heads,
                        dropout,
                        attn_chunk_size=self.attn_chunk_size,
                    )
                    for _ in range(num_levels - 1)
                ]
            )
            self.down_cross_attn = nn.ModuleList(
                [
                    CrossScaleAttention(
                        hidden_dim,
                        num_heads,
                        dropout,
                        attn_chunk_size=self.attn_chunk_size,
                    )
                    for _ in range(num_levels - 1)
                ]
            )

        self.up_pool = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_levels - 1)
            ]
        )
        self.down_unpool = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_levels - 1)
            ]
        )

        self.register_buffer("_dummy", torch.empty(0))
        self.caches: List[deque] = [
            deque(maxlen=window) for _ in range(num_levels)]
        self.level_spatial_mix = nn.ModuleList(
            [SpatialMixBlock(hidden_dim, dropout=dropout) for _ in range(num_levels)]
        )

    def reset(self) -> None:
        for cache in self.caches:
            cache.clear()

    @torch.no_grad()
    def warm_start(self, states_per_level: List[List[torch.Tensor]]) -> None:
        for level_idx, level_states in enumerate(states_per_level):
            self.caches[level_idx].clear()
            for state in level_states[-self.window:]:
                self.caches[level_idx].append(state.detach())

    def _pool_features(
        self,
        fine_features: torch.Tensor,
        up_edge_index: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        from torch_geometric.utils import scatter

        n_fine, t, h = fine_features.shape
        fine_idx = up_edge_index[0]
        coarse_idx = up_edge_index[1]
        n_coarse = int(coarse_idx.max().item()) + 1
        fine_flat = fine_features.reshape(n_fine * t, h)
        fine_idx_expanded = fine_idx.unsqueeze(1).expand(-1, t).reshape(-1)
        coarse_idx_expanded = coarse_idx.unsqueeze(1).expand(-1, t).reshape(-1)
        edge_features = fine_flat[fine_idx_expanded]
        edge_features = self.up_pool[level](edge_features)
        coarse_flat = scatter(
            edge_features,
            coarse_idx_expanded,
            dim=0,
            dim_size=n_coarse * t,
            reduce="mean",
        )
        return coarse_flat.reshape(n_coarse, t, h)

    def _unpool_features(
        self,
        coarse_features: torch.Tensor,
        down_edge_index: torch.Tensor,
        level: int,
        n_fine_nodes: int,
    ) -> torch.Tensor:
        from torch_geometric.utils import scatter

        n_coarse, t, h = coarse_features.shape
        coarse_idx = down_edge_index[0]
        fine_idx = down_edge_index[1]
        coarse_flat = coarse_features.reshape(n_coarse * t, h)
        coarse_idx_expanded = coarse_idx.unsqueeze(1).expand(-1, t).reshape(-1)
        fine_idx_expanded = fine_idx.unsqueeze(1).expand(-1, t).reshape(-1)
        edge_features = coarse_flat[coarse_idx_expanded]
        edge_features = self.down_unpool[level](edge_features)
        fine_flat = scatter(
            edge_features,
            fine_idx_expanded,
            dim=0,
            dim_size=n_fine_nodes * t,
            reduce="mean",
        )
        return fine_flat.reshape(n_fine_nodes, t, h)

    def forward(
        self,
        mesh_features_list: List[torch.Tensor],
        up_edge_index_list: Optional[List[torch.Tensor]] = None,
        down_edge_index_list: Optional[List[torch.Tensor]] = None,
        mesh_edge_index_list: Optional[List[torch.Tensor]] = None,
        mesh_edge_attr_list: Optional[List[torch.Tensor]] = None,
        up_edge_attr_list: Optional[List[torch.Tensor]] = None,
        down_edge_attr_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        device = mesh_features_list[0].device
        dtype = mesh_features_list[0].dtype

        for level, x_mesh in enumerate(mesh_features_list):
            self.caches[level].append(x_mesh)

        x_seq_list = []
        for level in range(self.num_levels):
            x_seq = torch.stack(list(self.caches[level]), dim=1).to(
                device=device, dtype=dtype
            )
            x_seq = self.level_posenc[level](x_seq)
            x_seq_list.append(x_seq)

        t = x_seq_list[0].size(1)
        attn_mask = _causal_mask(t, device) if self.use_causal_mask else None

        processed_list = []
        for level in range(self.num_levels):
            x_seq = x_seq_list[level]
            for block in self.level_transformers[level]:
                x_seq = block(x_seq, attn_mask)
                if (
                    self.spatial_mixing_steps > 0
                    and mesh_edge_index_list is not None
                    and len(mesh_edge_index_list) == self.num_levels
                ):
                    ei = mesh_edge_index_list[level]
                    ea = None
                    if (
                        mesh_edge_attr_list is not None
                        and len(mesh_edge_attr_list) == self.num_levels
                    ):
                        ea = mesh_edge_attr_list[level]
                    mixed = []
                    for ti in range(x_seq.size(1)):
                        xt = x_seq[:, ti, :]
                        for _ in range(self.spatial_mixing_steps):
                            xt = self.level_spatial_mix[level](xt, ei, ea)
                        mixed.append(xt)
                    x_seq = torch.stack(mixed, dim=1)
            processed_list.append(x_seq)
            logger.debug(
                "Hierarchical transformer level %s: nodes=%s temporal=%s",
                level,
                x_seq.shape[0],
                x_seq.shape,
            )

        if (
            self.use_cross_scale
            and up_edge_index_list is not None
            and down_edge_index_list is not None
            and len(up_edge_index_list) == self.num_levels - 1
            and len(down_edge_index_list) == self.num_levels - 1
        ):
            for level in range(self.num_levels - 1):
                pooled_fine = self._pool_features(
                    processed_list[level],
                    up_edge_index_list[level],
                    level,
                )
                processed_list[level + 1] = self.up_cross_attn[level](
                    query=processed_list[level + 1],
                    key_value=pooled_fine,
                )
            for level in range(self.num_levels - 2, -1, -1):
                unpooled_coarse = self._unpool_features(
                    processed_list[level + 1],
                    down_edge_index_list[level],
                    level,
                    n_fine_nodes=processed_list[level].size(0),
                )
                processed_list[level] = self.down_cross_attn[level](
                    query=processed_list[level],
                    key_value=unpooled_coarse,
                )

        return [x_seq[:, -1, :] for x_seq in processed_list]


class SlidingWindowTransformerProcessor(nn.Module):
    """Single-level temporal transformer over a rolling mesh state window."""

    def __init__(
        self,
        hidden_dim: int,
        window: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        attn_chunk_size: int = 2048,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    hidden_dim, num_heads, dropout, attn_chunk_size=attn_chunk_size
                )
                for _ in range(depth)
            ]
        )
        self.posenc = TemporalPositionalEncoding(hidden_dim, max_len=window)
        self.register_buffer("_dummy", torch.empty(0))
        self.cache: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.cache.clear()

    @torch.no_grad()
    def warm_start(self, states: List[torch.Tensor]) -> None:
        self.cache.clear()
        for s in states[-self.window:]:
            self.cache.append(s.detach())

    def forward(self, x_mesh: torch.Tensor) -> torch.Tensor:
        device = x_mesh.device
        dtype = x_mesh.dtype
        self.cache.append(x_mesh)
        x_seq = torch.stack(
            list(
                self.cache),
            dim=1).to(
            device=device,
            dtype=dtype)
        x_seq = self.posenc(x_seq)
        attn_mask = (_causal_mask(x_seq.size(1), device)
                     if self.use_causal_mask else None)
        for blk in self.blocks:
            x_seq = blk(x_seq, attn_mask)
        return x_seq[:, -1, :]
