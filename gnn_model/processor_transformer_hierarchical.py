from collections import deque
from typing import List, Optional
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
            batch_first=True,
            dropout=dropout
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


class CrossScaleAttention(nn.Module):
    """
    Cross-attention between different mesh scales.
    Allows information flow between coarse and fine levels.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query: [N_query, T, H] - features at target scale
        key_value: [N_kv, T, H] - features at source scale
        returns: [N_query, T, H] - updated features at target scale
        """
        attn_out, _ = self.cross_attn(
            query, key_value, key_value, need_weights=False
        )
        return self.norm(query + self.drop(attn_out))


class HierarchicalSlidingWindowTransformer(nn.Module):
    """
    Hierarchical temporal transformer that processes multiple mesh resolution levels.

    Architecture:
    1. Each level has its own temporal transformer (intra-level processing)
    2. Cross-scale attention allows information flow between levels
    3. Coarse levels capture large-scale temporal patterns
    4. Fine levels capture local temporal evolution
    5. Bidirectional cross-scale attention (up and down)

    This creates a spatiotemporal U-Net where:
    - Spatial hierarchy: coarse to fine mesh levels
    - Temporal processing: transformer over time at each level
    """
    def __init__(self,
                 hidden_dim: int,
                 num_levels: int = 4,
                 window: int = 4,
                 depth: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 use_causal_mask: bool = True,
                 use_cross_scale: bool = True):
        """
        Args:
            hidden_dim: Hidden dimension for all levels
            num_levels: Number of mesh hierarchy levels
            window: Temporal window size
            depth: Number of transformer blocks per level
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking (for autoregressive)
            use_cross_scale: Whether to use cross-scale attention between levels
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.use_cross_scale = use_cross_scale

        # Temporal transformers for each level (intra-level)
        self.level_transformers = nn.ModuleList()
        for _ in range(num_levels):
            blocks = nn.ModuleList([
                TemporalBlock(hidden_dim, num_heads, dropout)
                for _ in range(depth)
            ])
            self.level_transformers.append(blocks)

        # Positional encodings for each level
        self.level_posenc = nn.ModuleList([
            TemporalPositionalEncoding(hidden_dim, max_len=window)
            for _ in range(num_levels)
        ])

        # Cross-scale attention (if enabled)
        if use_cross_scale:
            # Upward cross-attention (fine -> coarse)
            self.up_cross_attn = nn.ModuleList([
                CrossScaleAttention(hidden_dim, num_heads, dropout)
                for _ in range(num_levels - 1)
            ])

            # Downward cross-attention (coarse -> fine)
            self.down_cross_attn = nn.ModuleList([
                CrossScaleAttention(hidden_dim, num_heads, dropout)
                for _ in range(num_levels - 1)
            ])

        # Spatial pooling for upward information flow (fine -> coarse)
        # Use learnable aggregation
        self.up_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for _ in range(num_levels - 1)
        ])

        # Spatial unpooling for downward information flow (coarse -> fine)
        self.down_unpool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for _ in range(num_levels - 1)
        ])

        self.register_buffer("_dummy", torch.empty(0))

        # Cache for each level (stores temporal history)
        self.caches: List[deque] = [deque(maxlen=window) for _ in range(num_levels)]

    def reset(self):
        """Clear all temporal caches"""
        for cache in self.caches:
            cache.clear()

    @torch.no_grad()
    def warm_start(self, states_per_level: List[List[torch.Tensor]]):
        """
        Pre-fill caches with historical states
        states_per_level: List of length num_levels, each containing historical states
        """
        for level_idx, level_states in enumerate(states_per_level):
            self.caches[level_idx].clear()
            for state in level_states[-self.window:]:
                self.caches[level_idx].append(state.detach())

    def _pool_features(self, fine_features: torch.Tensor,
                       up_edge_index: torch.Tensor,
                       level: int) -> torch.Tensor:
        """
        Pool fine-level features to coarse level using edge connections.

        Args:
            fine_features: [N_fine, T, H]
            up_edge_index: [2, E] edges from fine to coarse
            level: which level (for selecting pooling layer)

        Returns:
            coarse_features: [N_coarse, T, H]
        """
        from torch_geometric.utils import scatter

        N_fine, T, H = fine_features.shape
        fine_idx = up_edge_index[0]  # source (fine) node indices
        coarse_idx = up_edge_index[1]  # target (coarse) node indices

        N_coarse = coarse_idx.max().item() + 1

        # Reshape for processing: [N_fine*T, H]
        fine_flat = fine_features.reshape(N_fine * T, H)

        # Expand indices for temporal dimension
        fine_idx_expanded = fine_idx.unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]
        coarse_idx_expanded = coarse_idx.unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]

        # Gather fine features using edges: [E*T, H]
        edge_features = fine_flat[fine_idx_expanded]

        # Apply pooling transformation
        edge_features = self.up_pool[level](edge_features)

        # Aggregate to coarse nodes using mean
        coarse_flat = scatter(edge_features, coarse_idx_expanded, dim=0,
                              dim_size=N_coarse * T, reduce='mean')

        # Reshape back: [N_coarse, T, H]
        coarse_features = coarse_flat.reshape(N_coarse, T, H)

        return coarse_features

    def _unpool_features(self, coarse_features: torch.Tensor,
                         down_edge_index: torch.Tensor,
                         level: int) -> torch.Tensor:
        """
        Unpool coarse-level features to fine level using edge connections.

        Args:
            coarse_features: [N_coarse, T, H]
            down_edge_index: [2, E] edges from coarse to fine
            level: which level (for selecting unpooling layer)

        Returns:
            fine_features: [N_fine, T, H]
        """
        from torch_geometric.utils import scatter

        N_coarse, T, H = coarse_features.shape
        coarse_idx = down_edge_index[0]  # source (coarse) node indices
        fine_idx = down_edge_index[1]  # target (fine) node indices

        N_fine = fine_idx.max().item() + 1

        # Reshape for processing: [N_coarse*T, H]
        coarse_flat = coarse_features.reshape(N_coarse * T, H)

        # Expand indices for temporal dimension
        coarse_idx_expanded = coarse_idx.unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]
        fine_idx_expanded = fine_idx.unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]

        # Gather coarse features using edges: [E*T, H]
        edge_features = coarse_flat[coarse_idx_expanded]

        # Apply unpooling transformation
        edge_features = self.down_unpool[level](edge_features)

        # Aggregate to fine nodes using mean
        fine_flat = scatter(edge_features, fine_idx_expanded, dim=0,
                            dim_size=N_fine * T, reduce='mean')

        # Reshape back: [N_fine, T, H]
        fine_features = fine_flat.reshape(N_fine, T, H)

        return fine_features

    def forward(self,
                mesh_features_list: List[torch.Tensor],
                up_edge_index_list: Optional[List[torch.Tensor]] = None,
                down_edge_index_list: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Forward pass through hierarchical temporal transformer.

        Args:
            mesh_features_list: List of [N_level, H] current mesh states per level
            up_edge_index_list: List of [2, E] edge indices for fine->coarse (optional)
            down_edge_index_list: List of [2, E] edge indices for coarse->fine (optional)

        Returns:
            List of [N_level, H] updated mesh states per level
        """
        device = mesh_features_list[0].device
        dtype = mesh_features_list[0].dtype

        # Update caches with current states
        for level, x_mesh in enumerate(mesh_features_list):
            self.caches[level].append(x_mesh)

        # Stack temporal sequences for each level: [N_level, T, H]
        x_seq_list = []
        for level in range(self.num_levels):
            x_seq = torch.stack(list(self.caches[level]), dim=1).to(device=device, dtype=dtype)
            x_seq = self.level_posenc[level](x_seq)  # Add positional encoding
            x_seq_list.append(x_seq)

        # Causal mask (shared across all levels)
        T = x_seq_list[0].size(1)
        attn_mask = _causal_mask(T, device) if self.use_causal_mask else None

        # ========================================================================
        # Phase 1: Intra-level temporal processing
        # ========================================================================
        print(f"[HIERARCHICAL TRANSFORMER] Phase 1: Intra-level temporal processing")
        print(f"  - Processing {self.num_levels} levels with window size {T}")
        processed_list = []
        for level in range(self.num_levels):
            x_seq = x_seq_list[level]
            for block in self.level_transformers[level]:
                x_seq = block(x_seq, attn_mask)
            processed_list.append(x_seq)
            print(f"  - Level {level}: {x_seq.shape[0]} nodes, temporal shape {x_seq.shape}")

        # ========================================================================
        # Phase 2: Cross-scale attention (if enabled)
        # ========================================================================
        if self.use_cross_scale and up_edge_index_list is not None and down_edge_index_list is not None:
            print(f"[HIERARCHICAL TRANSFORMER] Phase 2: Upward cross-scale attention (fine→coarse)")

            # Upward pass: incorporate fine-scale info into coarse levels
            for level in range(self.num_levels - 1):
                # Pool fine features to coarse level
                pooled_fine = self._pool_features(
                    processed_list[level],
                    up_edge_index_list[level],
                    level
                )

                print(f"  - Level {level}→{level+1}: Pooled {processed_list[level].shape[0]} → {pooled_fine.shape[0]} nodes")

                # Cross-attention: coarse attends to pooled fine
                processed_list[level + 1] = self.up_cross_attn[level](
                    query=processed_list[level + 1],
                    key_value=pooled_fine
                )

            print(f"[HIERARCHICAL TRANSFORMER] Phase 3: Downward cross-scale attention (coarse→fine)")
            # Downward pass: incorporate coarse-scale info into fine levels
            for level in range(self.num_levels - 2, -1, -1):
                # Unpool coarse features to fine level
                unpooled_coarse = self._unpool_features(
                    processed_list[level + 1],
                    down_edge_index_list[level],
                    level
                )

                print(f"  - Level {level+1}→{level}: Unpooled {processed_list[level + 1].shape[0]} → {unpooled_coarse.shape[0]} nodes")

                # Cross-attention: fine attends to unpooled coarse
                processed_list[level] = self.down_cross_attn[level](
                    query=processed_list[level],
                    key_value=unpooled_coarse
                )

        # ========================================================================
        # Extract current timestep (last in sequence) for each level
        # ========================================================================
        output_list = [x_seq[:, -1, :] for x_seq in processed_list]

        return output_list


class SlidingWindowTransformerProcessor(nn.Module):
    """
    Temporal transformer over a rolling window of latent mesh states.
    Call reset() at the start of each new sequence/bin;
    then call forward() each rollout step.

    NOTE: This is the single-level version. For hierarchical meshes,
    use HierarchicalSlidingWindowTransformer instead.
    """
    def __init__(self,
                 hidden_dim: int,
                 window: int = 4,
                 depth: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 use_causal_mask: bool = True):
        super().__init__()
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.posenc = TemporalPositionalEncoding(hidden_dim, max_len=window)
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

    def forward(self, x_mesh: torch.Tensor) -> torch.Tensor:
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

        return x_seq[:, -1, :]
