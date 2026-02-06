import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from interaction_network import InteractionNetwork
from interaction_net import InteractionNet
import torch.utils.checkpoint as checkpoint


class HierarchicalProcessor(nn.Module):
    """
    A hierarchical processor that performs message passing across multiple mesh levels.

    WARNING: This uses InteractionNet which may cause OOM (Out of Memory) issues
    on large meshes. For transformer-based hierarchical processing (recommended),
    use HierarchicalSlidingWindowTransformer from processor_transformer_hierarchical.py

    Architecture:
    1. At each level, perform local message passing within that level
    2. Pass information UP from finer to coarser levels (aggregation)
    3. Pass information DOWN from coarser to finer levels (refinement)

    This creates a U-Net like structure where information flows:
    Fine → Medium → Coarse (encoding/aggregation)
    Coarse → Medium → Fine (decoding/refinement)

    Use Cases:
    - Research: Comparing GNN vs Transformer hierarchical processing
    - Future: When OOM issues are resolved with better optimization
    - Small meshes: Works fine on lower resolution meshes

    For Production Use: Use processor_transformer_hierarchical.py instead
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        num_message_passing_steps: int = 4,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden features
            num_levels: Number of mesh hierarchy levels
            num_message_passing_steps: Number of message passing steps per level
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_message_passing_steps = num_message_passing_steps

        # Intra-level message passing (within each mesh level)
        self.intra_level_layers = nn.ModuleList()
        for level in range(num_levels):
            level_layers = nn.ModuleList()
            for _ in range(num_message_passing_steps):
                level_layers.append(
                    InteractionNet(
                        edge_index=None,
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        hidden_layers=2,
                        update_edges=False,
                    )
                )
            self.intra_level_layers.append(level_layers)

        # Up connections (fine → coarse)
        self.up_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            self.up_layers.append(
                InteractionNet(
                    edge_index=None,
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    hidden_layers=2,
                    update_edges=False,
                )
            )

        # Down connections (coarse → fine)
        self.down_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            self.down_layers.append(
                InteractionNet(
                    edge_index=None,
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    hidden_layers=2,
                    update_edges=False,
                )
            )

        # Layer normalization for each level
        self.level_norms = nn.ModuleList()
        for level in range(num_levels):
            self.level_norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        mesh_features_list: List[torch.Tensor],
        mesh_edge_index_list: List[torch.Tensor],
        mesh_edge_attr_list: List[torch.Tensor],
        up_edge_index_list: List[torch.Tensor],
        up_edge_attr_list: List[torch.Tensor],
        down_edge_index_list: List[torch.Tensor],
        down_edge_attr_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward pass through hierarchical mesh levels.

        Args:
            mesh_features_list: List of mesh node features for each level [coarse → fine]
            mesh_edge_index_list: List of edge indices for intra-level connections
            mesh_edge_attr_list: List of edge attributes for intra-level connections
            up_edge_index_list: List of edge indices for up connections (fine → coarse)
            up_edge_attr_list: List of edge attributes for up connections
            down_edge_index_list: List of edge indices for down connections (coarse → fine)
            down_edge_attr_list: List of edge attributes for down connections

        Returns:
            List of updated mesh features for each level
        """

        # Store features at each level
        level_features = [feat.clone() for feat in mesh_features_list]

        # ============================================================
        # PHASE 1: Intra-level message passing
        # ============================================================
        for level in range(self.num_levels):
            residual = level_features[level]

            # Perform message passing within this level
            for step in range(self.num_message_passing_steps):
                self.intra_level_layers[level][step].edge_index = mesh_edge_index_list[level]
                level_features[level] = self.intra_level_layers[level][step](
                    send_rep=level_features[level],
                    rec_rep=level_features[level],
                    edge_rep=mesh_edge_attr_list[level] if mesh_edge_attr_list[level] is not None else None,
                )

            # Residual connection and normalization
            level_features[level] = self.level_norms[level](
                level_features[level] + residual
            )

        # ============================================================
        # PHASE 2: Upward pass (fine → coarse aggregation)
        # ============================================================
        for level in range(self.num_levels - 1):
            # level is the finer level, level+1 is the coarser level
            self.up_layers[level].edge_index = up_edge_index_list[level]

            # Aggregate information from finer to coarser level
            coarse_update = self.up_layers[level](
                send_rep=level_features[level],  # from finer level
                rec_rep=level_features[level + 1],  # to coarser level
                edge_rep=up_edge_attr_list[level] if up_edge_attr_list[level] is not None else None,
            )

            # Add to coarser level with residual
            level_features[level + 1] = level_features[level + 1] + coarse_update

        # ============================================================
        # PHASE 3: Downward pass (coarse → fine refinement)
        # ============================================================
        for level in reversed(range(self.num_levels - 1)):
            # level+1 is the coarser level, level is the finer level
            self.down_layers[level].edge_index = down_edge_index_list[level]

            # Refine finer level with information from coarser level
            fine_update = self.down_layers[level](
                send_rep=level_features[level + 1],  # from coarser level
                rec_rep=level_features[level],  # to finer level
                edge_rep=down_edge_attr_list[level] if down_edge_attr_list[level] is not None else None,
            )

            # Add to finer level with residual
            level_features[level] = level_features[level] + fine_update

        return level_features
