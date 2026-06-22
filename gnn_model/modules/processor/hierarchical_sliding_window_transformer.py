"""Hierarchical sliding-window transformer processor for multi-scale mesh rollout.

This module defines temporal attention, cross-scale interactions, spatial
mixing, and the hierarchical transformer processor used to evolve latent mesh
states across multiple resolution levels.

Author: Azadeh Gholoubi
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .processor_base import ProcessorBase
from ..mesh.hierarchical_mesh import HierarchicalMesh


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


class SpatialMixBlock(nn.Module):
    """One explicit within-level neighbor mixing step using `edge_index`."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # See processor_transformer.SpatialMixBlock for rationale.
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
        N = int(x.size(0))
        device = x.device
        dtype = x.dtype

        if edge_attr is None:
            agg = torch.zeros((N, x.size(1)), device=device, dtype=dtype)
            agg.index_add_(0, dst, x[src])

            deg = torch.zeros((N,), device=device, dtype=dtype)
            deg.index_add_(0, dst, torch.ones((dst.numel(),), device=device, dtype=dtype))
            agg = agg / deg.clamp(min=1).unsqueeze(-1)
        else:
            if edge_attr.dim() == 1:
                w = edge_attr
            elif edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                w = edge_attr[:, 0]
            else:
                feat = edge_attr.to(device=device, dtype=torch.float32)
                feat_norm = torch.sqrt((feat * feat).sum(dim=1) + 1e-12)
                w = torch.exp(-float(self.distance_scale) * feat_norm).to(dtype=dtype)

            w = w.to(device=device, dtype=dtype).clamp(min=0)

            agg = torch.zeros((N, x.size(1)), device=device, dtype=dtype)
            agg.index_add_(0, dst, x[src] * w.unsqueeze(-1))

            wsum = torch.zeros((N,), device=device, dtype=dtype)
            wsum.index_add_(0, dst, w)
            agg = agg / wsum.clamp(min=1e-6).unsqueeze(-1)

        msg = self.mlp(agg)
        return self.norm(x + self.drop(msg))


class HierarchicalSlidingWindowTransformer(ProcessorBase):
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
                 mesh: HierarchicalMesh,
                 hidden_dim: int,
                 num_levels: int = 4,
                 window: int = 4,
                 depth: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 use_causal_mask: bool = True,
                 use_cross_scale: bool = True,
                 spatial_mixing_steps: int = 1):
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
        super().__init__(mesh)
        
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.window = window
        self.use_causal_mask = use_causal_mask
        self.use_cross_scale = use_cross_scale
        self.spatial_mixing_steps = int(spatial_mixing_steps)

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

        # Optional within-level spatial neighbor mixing after temporal+cross-scale attention.
        self.level_spatial_mix = nn.ModuleList(
            [SpatialMixBlock(hidden_dim, dropout=dropout) for _ in range(num_levels)]
        )

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

    def forward(self, data: HeteroData, encoded_mesh_features) -> List[torch.Tensor]:
        """
        Forward pass through hierarchical temporal transformer.

        Args:
            data: HeteroData containing mesh edge indices and attributes for all levels
            encoded_mesh_features: tensor of encoded features for the finest level (level 0)

        Returns:
            List of [N_level, H] updated mesh states per level
        """

        step_info = self._get_latent_step_info(data)
        num_latent_steps = step_info["num_steps"]
        step_mapping = step_info["step_mapping"]
        edge_mapping = self._map_step_edges(data, step_mapping)

        self.debug(f"[LATENT] {num_latent_steps} latent steps detected")
        self.debug(f"[LATENT] Step mapping: {step_mapping}")
        
        for step in range(num_latent_steps):
            self._do_forward_step(step, num_latent_steps, encoded_mesh_features)

        return output_list
    

    def _do_forward_step(self, step: int, num_latent_steps: int, current_mesh_features: torch.Tensor) -> None:
        meshData = self._prep_mesh_data(step, num_latent_steps, current_mesh_features)

        device = meshData.mesh_features_list[0].device
        dtype = meshData.mesh_features_list[0].dtype

        # Update caches with current states
        for level, x_mesh in enumerate(meshData.mesh_features_list):
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

                # Interleave explicit within-level spatial mixing with temporal layers.
                if (
                    self.spatial_mixing_steps > 0
                    and meshData.mesh_edge_index_list is not None
                    and isinstance(meshData.mesh_edge_index_list, (list, tuple))
                    and len(meshData.mesh_edge_index_list) == self.num_levels
                ):
                    ei = meshData.mesh_edge_index_list[level]
                    ea = None
                    if (
                        meshData.mesh_edge_attr_list is not None
                        and isinstance(meshData.mesh_edge_attr_list, (list, tuple))
                        and len(meshData.mesh_edge_attr_list) == self.num_levels
                    ):
                        ea = meshData.mesh_edge_attr_list[level]
                    mixed = []
                    for t in range(x_seq.size(1)):
                        xt = x_seq[:, t, :]
                        for _ in range(self.spatial_mixing_steps):
                            xt = self.level_spatial_mix[level](xt, ei, ea)
                        mixed.append(xt)
                    x_seq = torch.stack(mixed, dim=1)
            processed_list.append(x_seq)
            print(f"  - Level {level}: {x_seq.shape[0]} nodes, temporal shape {x_seq.shape}")

        # ========================================================================
        # Phase 2: Cross-scale attention (if enabled)
        # ========================================================================
        if self.use_cross_scale and meshData.up_edge_index_list is not None and meshData.down_edge_index_list is not None:
            print(f"[HIERARCHICAL TRANSFORMER] Phase 2: Upward cross-scale attention (fine→coarse)")

            # Upward pass: incorporate fine-scale info into coarse levels
            for level in range(self.num_levels - 1):
                # Pool fine features to coarse level
                pooled_fine = self._pool_features(
                    processed_list[level],
                    meshData.up_edge_index_list[level],
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
                    meshData.down_edge_index_list[level],
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

        self._gather_node_features(step, processed_list)
    

    @dataclass
    class MeshData:
        mesh_features_list: List[torch.Tensor]
        up_edge_index_list: List[torch.Tensor]
        down_edge_index_list: List[torch.Tensor]
        mesh_edge_index_list: List[torch.Tensor]
        mesh_edge_attr_list: List[torch.Tensor]
    

    def _prep_mesh_data(self, step: int, num_latent_steps: int, current_mesh_features: torch.Tensor) -> MeshData:
        # Hierarchical transformer: process all mesh levels with cross-scale attention
        print(f"[FORWARD] Step {step+1}/{num_latent_steps}: Using HIERARCHICAL transformer")
        # Prepare mesh features for all levels
        # NOTE: Level ordering is [finest, ..., coarsest] (level 0 = finest, level -1 = coarsest)
        mesh_features_list = []

        for level in range(self.num_mesh_levels):
            level_mesh_x = getattr(self.mesh, f"mesh_x_level_{level}")

            # Only the FINEST level (level 0) receives encoded features
            # Coarser levels start with zeros
            # TODO: Could distribute encoded features across levels based on spatial scale
            if level == 0:  # Finest level
                mesh_features_list.append(current_mesh_features)
            else:
                # Initialize coarser levels with zeros
                num_nodes_this_level = level_mesh_x.shape[0]
                mesh_features_list.append(
                    torch.zeros(num_nodes_this_level, self.hidden_dim,
                                device=current_mesh_features.device)
                )

        print(f"[FORWARD]   - Mesh features per level: {[m.shape for m in mesh_features_list]}")

        # Prepare up/down edge indices for cross-scale attention
        up_edge_index_list = [None] * (self.num_mesh_levels - 1)
        down_edge_index_list = [None] * (self.num_mesh_levels - 1)

        for level in range(self.num_mesh_levels - 1):
            up_edge_index_list[level] =  getattr(self.mesh, f"mesh_up_edge_index_{level}")
            down_edge_index_list[level] = getattr(self.mesh, f"mesh_down_edge_index_{level}")

        print(f"[FORWARD]   - Cross-scale connections: {len(up_edge_index_list)} up/down pairs")

        # Process through hierarchical transformer
        mesh_edge_index_list = [
            getattr(self.mesh, f"mesh_edge_index_level_{lvl}")
            for lvl in range(self.num_mesh_levels)
        ]
        mesh_edge_attr_list = [
            getattr(self.mesh, f"mesh_edge_attr_level_{lvl}")
            for lvl in range(self.num_mesh_levels)
        ]

        return MeshData(
            mesh_features_list=mesh_features_list,
            up_edge_index_list=up_edge_index_list,
            down_edge_index_list=down_edge_index_list,
            mesh_edge_index_list=mesh_edge_index_list,
            mesh_edge_attr_list=mesh_edge_attr_list
        )
    
    
    def _gather_node_features(self, step: int, processed_levels: list) -> torch.Tensor:
        # COARSE→FINE CONDITIONING: Add hierarchical information flow
        # Gather coarse features (L1) to fine nodes (L0) for better multi-scale learning

        current_mesh_features: torch.Tensor = None  # Will hold the final features for this step

        if self.num_mesh_levels > 1:
            fine_features = processed_levels[0]  # [N_fine, H] - finest level (L0)
            coarse_features = processed_levels[1]  # [N_coarse, H] - coarse level (L1)

            # DIRECTION CHECK: down_edges should be coarse→fine
            # mesh_down_edge_index_0: L1→L0 (coarse to fine)
            # Shape: [2, E] where [0, :] = source (coarse), [1, :] = target (fine)
            down_edge_index = getattr(self.mesh, "mesh_down_edge_index_0")

            # Verify directionality: source indices should be < N_coarse
            if step == 0 and self.global_step == 0:
                src_max = down_edge_index[0].max().item()
                dst_max = down_edge_index[1].max().item()
                print(f"[COARSE→FINE] Edge direction check: src_max={src_max} (expect <{coarse_features.shape[0]}), "
                        f"dst_max={dst_max} (expect <{fine_features.shape[0]})")

            # Gather: each edge gets coarse features from source
            coarse_gathered = coarse_features[down_edge_index[0]]  # [E, H]

            # Aggregate to fine nodes using mean (stable across variable degree)
            fine_conditioned = torch.zeros_like(fine_features)
            fine_conditioned.scatter_reduce_(
                0,
                down_edge_index[1].unsqueeze(-1).expand(-1, self.hidden_dim),
                coarse_gathered,
                reduce='mean'  # Mean is safest - keeps scale stable
            )

            # Normalize coarse signal before projection
            fine_conditioned_norm = self.coarse_to_fine_norm(fine_conditioned)

            # Project to delta
            delta = self.coarse_to_fine_proj(fine_conditioned_norm)

            # Gated residual: model learns how much coarse info to use
            gate_input = torch.cat([fine_features, fine_conditioned_norm], dim=-1)  # [N, 2H]
            gate = self.coarse_to_fine_gate(gate_input)  # [N, H] in [0, 1]

            # Final: fine + gated coarse contribution
            current_mesh_features = fine_features + gate * delta

            if step == 0:  # Diagnostics once per batch
                delta_norm = delta.norm(dim=-1).mean().item()
                gate_mean = gate.mean().item()
                print(f"[COARSE→FINE] L1({coarse_features.shape[0]})→L0({fine_features.shape[0]}) | "
                        f"δ_norm={delta_norm:.4f}, gate_μ={gate_mean:.4f}")
        else:
            # Use the finest level output (level 0)
            current_mesh_features = processed_levels[0]

        return current_mesh_features
