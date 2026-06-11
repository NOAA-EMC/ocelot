"""Fixed-mesh interaction processor for OCELOT.

Author: Azadeh Gholoubi
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.data import HeteroData

from ..mesh.fixed_mesh import FixedMesh

from .interaction_network import InteractionNetwork
from .flat_processor_base import FlatProcessorBase


class InteractionProcessor(FlatProcessorBase):
    """
    A Processor module that applies multiple steps of message passing using
    InteractionNetwork blocks, inspired by graphcast's processor.
    This module handles the core GNN processing, including the message-passing
    loop and residual connections.
    """

    def __init__(
        self,
        mesh: FixedMesh,
        hidden_dim: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        num_message_passing_steps: int,
    ):
        super().__init__(mesh)

        if not isinstance(mesh, FixedMesh):
            raise ValueError("InteractionProcessor requires a FixedMesh instance")

        self.num_message_passing_steps = num_message_passing_steps

        self.layers = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            # This is now the simple, original InteractionNetwork call
            self.layers.append(InteractionNetwork(hidden_dim, node_types, edge_types))

        self.norms = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            self.norms.append(
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_dim) for node_type in node_types}
                )
            )


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

        self.debug(f"[LATENT] {num_latent_steps} latent steps detected")
        self.debug(f"[LATENT] Step mapping: {step_mapping}")
        
        for step in range(num_latent_steps):
            processed_x_dict = self._do_forward_step(step, num_latent_steps, encoded_features["mesh"])

        current_mesh_features = processed_x_dict["mesh"]

        return output_list
    
    
    def _do_forward_step(self, step, num_latent_steps, x_seq):
        """
        Processes the graph through multiple message-passing steps.
        """
        # Remove decoder edges (mesh → target), but keep encoder edges (input → mesh)
        processor_edges = {et: ei for et, ei in data.edge_index_dict.items()
                           if "_target" not in et[2]}

        # STAGE 4A: PROCESS - Evolve mesh state forward one latent step
        step_features = encoded_features.copy()
        step_features["mesh"] = current_mesh_features
        
        processed_x_dict = x_dict
        for i in range(self.num_message_passing_steps):
            residual_x_dict = processed_x_dict

            # Apply one step of message passing using gradient checkpointing
            processed_x_dict = checkpoint.checkpoint(
                self.layers[i], processed_x_dict, edge_index_dict, use_reentrant=False
            )

            # Add residual connection and apply layer norm
            for node_type in processed_x_dict:
                processed_x_dict[node_type] = self.norms[i][node_type](
                    processed_x_dict[node_type] + residual_x_dict[node_type]
                )
        return processed_x_dict
