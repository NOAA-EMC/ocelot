"""Hierarchical interaction-network processor for OCELOT mesh levels.

Author: Azadeh Gholoubi

This module keeps the message-passing hierarchical processor used for
experiments and compatibility. The transformer-based hierarchical processor is
preferred for large OCELOT v1 runs.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.data import HeteroData

from logger import log

from .hierarchical_processor_base import HierarchicalProcessorBase
from ..mesh.hierarchical_mesh import HierarchicalMesh
from ..coder.interaction_net import InteractionNet


class HierarchicalInteractionProcessor(HierarchicalProcessorBase):
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
        mesh: HierarchicalMesh,
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
        super().__init__(mesh, hidden_dim, num_levels, num_message_passing_steps)

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

    def forward(self, data: HeteroData, encoded_features: dict) -> List[torch.Tensor]:
        """
        Forward pass through hierarchical temporal transformer.

        Args:
            data: HeteroData containing mesh edge indices and attributes for all levels
            encoded_features: dict of encoded features for the finest level (level 0)

        Returns:
            List of [N_level, H] updated mesh states per level
        """

        step_info = self._get_latent_step_info(data)
        num_latent_steps = step_info["num_steps"]
        step_mapping = step_info["step_mapping"]
        edge_mapping = self._map_step_edges(data, step_mapping)

        log.debug(f"[LATENT] {num_latent_steps} latent steps detected")
        log.debug(f"[LATENT] Step mapping: {step_mapping}")
        
        for step in range(num_latent_steps):
            self._do_forward_step(step, num_latent_steps, encoded_features["mesh"])

        return level_features
    
    
    def _do_forward_step(self, step: int, processed_levels: list) -> torch.Tensor:
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
                
        mesh_data = self._prep_mesh_data(step, num_latent_steps, current_mesh_features)

        # Store features at each level
        level_features = [feat.clone() for feat in mesh_data.mesh_features_list]

        # ============================================================
        # PHASE 1: Intra-level message passing
        # ============================================================
        for level in range(self.num_levels):
            residual = level_features[level]

            # Perform message passing within this level
            for step in range(self.num_message_passing_steps):
                self.intra_level_layers[level][step].edge_index = mesh_data.mesh_edge_index_list[level]
                level_features[level] = self.intra_level_layers[level][step](
                    send_rep=level_features[level],
                    rec_rep=level_features[level],
                    edge_rep=mesh_data.mesh_edge_attr_list[level] if mesh_data.mesh_edge_attr_list[level] is not None else None,
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
            self.up_layers[level].edge_index = mesh_data.up_edge_index_list[level]

            # Aggregate information from finer to coarser level
            coarse_update = self.up_layers[level](
                send_rep=level_features[level],  # from finer level
                rec_rep=level_features[level + 1],  # to coarser level
                edge_rep=mesh_data.up_edge_attr_list[level] if mesh_data.up_edge_attr_list[level] is not None else None,
            )

            # Add to coarser level with residual
            level_features[level + 1] = level_features[level + 1] + coarse_update

        # ============================================================
        # PHASE 3: Downward pass (coarse → fine refinement)
        # ============================================================
        for level in reversed(range(self.num_levels - 1)):
            # level+1 is the coarser level, level is the finer level
            self.down_layers[level].edge_index = mesh_data.down_edge_index_list[level]

            # Refine finer level with information from coarser level
            fine_update = self.down_layers[level](
                send_rep=level_features[level + 1],  # from coarser level
                rec_rep=level_features[level],  # to finer level
                edge_rep=mesh_data.down_edge_attr_list[level] if mesh_data.down_edge_attr_list[level] is not None else None,
            )

            # Add to finer level with residual
            level_features[level] = level_features[level] + fine_update

    @dataclass
    class MeshData:
        mesh_features_list: List[torch.Tensor]
        mesh_edge_index_list: List[torch.Tensor]
        mesh_edge_attr_list: List[torch.Tensor]
        up_edge_index_list: List[torch.Tensor]
        up_edge_attr_list: List[torch.Tensor]
        down_edge_index_list: List[torch.Tensor]
        down_edge_attr_list: List[torch.Tensor]

    def _prep_mesh_data(self, step: int, num_latent_steps: int, current_mesh_features: torch.Tensor) -> MeshData:
            # Hierarchical processor with InteractionNet: process across multiple mesh levels
        # Prepare mesh features for all levels (replicate for batch)
        mesh_features_list = []
        mesh_edge_index_list = []
        mesh_edge_attr_list = []

        for level in range(self.num_mesh_levels):
            level_mesh_x = getattr(self, f"mesh_x_level_{level}")
            level_mesh_ei = getattr(self, f"mesh_edge_index_level_{level}")
            level_mesh_ea = getattr(self, f"mesh_edge_attr_level_{level}")

            # Only the FINEST level (level 0) receives encoded features
            # Future: distribute features across levels
            if level == 0:
                mesh_features_list.append(current_mesh_features)
            else:
                # Initialize coarser levels with zeros for now
                num_nodes_this_level = level_mesh_x.shape[0]
                mesh_features_list.append(
                    torch.zeros(num_nodes_this_level, self.hidden_dim,
                                device=current_mesh_features.device)
                )

            # Batch the edge indices
            num_nodes_this_level = level_mesh_x.shape[0]
            batched_ei = [level_mesh_ei + i * num_nodes_this_level for i in range(num_graphs)]
            mesh_edge_index_list.append(torch.cat(batched_ei, dim=1))
            mesh_edge_attr_list.append(level_mesh_ea.repeat(num_graphs, 1))

        # Prepare up/down connections
        up_edge_index_list = []
        up_edge_attr_list = []
        down_edge_index_list = []
        down_edge_attr_list = []

        for level in range(self.num_mesh_levels - 1):
            up_ei = getattr(self, f"mesh_up_edge_index_{level}")
            up_ea = getattr(self, f"mesh_up_edge_attr_{level}")
            down_ei = getattr(self, f"mesh_down_edge_index_{level}")
            down_ea = getattr(self, f"mesh_down_edge_attr_{level}")

            # Batch the hierarchical edges
            num_nodes_fine = getattr(self, f"mesh_x_level_{level}").shape[0]
            num_nodes_coarse = getattr(self, f"mesh_x_level_{level+1}").shape[0]

            batched_up_ei = []
            batched_down_ei = []
            for i in range(num_graphs):
                batched_up_ei.append(up_ei + torch.tensor([[i * num_nodes_fine], [i * num_nodes_coarse]], device=up_ei.device))
                batched_down_ei.append(down_ei + torch.tensor([[i * num_nodes_coarse], [i * num_nodes_fine]], device=down_ei.device))

            up_edge_index_list.append(torch.cat(batched_up_ei, dim=1))
            up_edge_attr_list.append(up_ea.repeat(num_graphs, 1))
            down_edge_index_list.append(torch.cat(batched_down_ei, dim=1))
            down_edge_attr_list.append(down_ea.repeat(num_graphs, 1))

        return self.MeshData(
            mesh_features_list=mesh_features_list,
            mesh_edge_index_list=mesh_edge_index_list,
            mesh_edge_attr_list=mesh_edge_attr_list,
            up_edge_index_list=up_edge_index_list,
            up_edge_attr_list=up_edge_attr_list,
            down_edge_index_list=down_edge_index_list,
            down_edge_attr_list=down_edge_attr_list,
        )
    
    def _gather_node_features(self, step: int, processed_levels: list) -> torch.Tensor:

        current_mesh_features: torch.Tensor = None  # Will hold the final features for this step

        # COARSE→FINE CONDITIONING: Add hierarchical information flow (InteractionNet path)
        if self.num_mesh_levels > 1:
            fine_features = processed_levels[0]  # [N_fine * batch, H]
            coarse_features = processed_levels[1]  # [N_coarse * batch, H]

            # Use batched down edges (L1→L0) for conditioning
            # Already batched for multiple graphs
            down_edge_index = down_edge_index_list[0]  # Already batched

            # Direction check (only once at start)
            if step == 0 and self.global_step == 0:
                src_max = down_edge_index[0].max().item()
                dst_max = down_edge_index[1].max().item()
                print(f"[COARSE→FINE] InteractionNet edge check: src_max={src_max}, dst_max={dst_max}")

            # Gather coarse features to fine nodes
            coarse_gathered = coarse_features[down_edge_index[0]]  # [E, H]

            # Aggregate to fine nodes (mean for stability)
            fine_conditioned = torch.zeros_like(fine_features)
            fine_conditioned.scatter_reduce_(
                0,
                down_edge_index[1].unsqueeze(-1).expand(-1, self.hidden_dim),
                coarse_gathered,
                reduce='mean'
            )

            # Normalize → Project → Gate
            fine_conditioned_norm = self.coarse_to_fine_norm(fine_conditioned)
            delta = self.coarse_to_fine_proj(fine_conditioned_norm)
            gate_input = torch.cat([fine_features, fine_conditioned_norm], dim=-1)
            gate = self.coarse_to_fine_gate(gate_input)

            # Gated residual
            current_mesh_features = fine_features + gate * delta

            if step == 0:  # Diagnostics
                delta_norm = delta.norm(dim=-1).mean().item()
                gate_mean = gate.mean().item()
                print(f"[COARSE→FINE] InteractionNet: δ_norm={delta_norm:.4f}, gate_μ={gate_mean:.4f}")
        else:
            # Use the finest level output (level 0)
            current_mesh_features = processed_levels[0]

        return current_mesh_features

