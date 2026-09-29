"""Fixed-mesh interaction processor for OCELOT.

Author: Azadeh Gholoubi
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.data import HeteroData

from ocelot.logger import log
from ocelot.model.mesh.fixed_mesh import FixedMesh
from ocelot.configs.model_config import ProcessorConfig
from ocelot.model.graph_schema import GraphSchema
from ocelot.model.processor.interaction_network import InteractionNetwork
from ocelot.model.processor.flat_processor_base import FlatProcessorBase


class InteractionProcessor(FlatProcessorBase):
    """
    A Processor module that applies multiple steps of message passing using
    InteractionNetwork blocks, inspired by graphcast's processor.
    This module handles the core GNN processing, including the message-passing
    loop and residual connections.
    """

    def __init__(self,
                 mesh: FixedMesh,
                 processor_config: ProcessorConfig,
                 graph_schema: GraphSchema):
        super().__init__(mesh)

        if not isinstance(mesh, FixedMesh):
            raise ValueError("InteractionProcessor requires a FixedMesh instance")

        self.num_message_passing_steps = processor_config.num_message_passing_steps
        node_types = graph_schema.node_types
        edge_types = graph_schema.edge_types

        self.layers = nn.ModuleList()
        for _ in range(processor_config.num_message_passing_steps):
            # This is now the simple, original InteractionNetwork call
            self.layers.append(
                InteractionNetwork(
                    processor_config.hidden_dim,
                    node_types,
                    edge_types
                )
            )

        self.norms = nn.ModuleList()
        for _ in range(processor_config.num_message_passing_steps):
            self.norms.append(
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(processor_config.hidden_dim) for node_type in node_types}
                )
            )

    def forward(self, step: int, step_info: dict, encoded_mesh_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the interaction processor.

        Args:
            step: Current step index
            step_info: Dictionary containing information about latent steps and step mapping
            encoded_mesh_features: tensor of encoded features for the finest level (level 0)

        Returns:
            List of [N_level, H] updated mesh states per level
        """

        """
        Processes the graph through multiple message-passing steps.
        """

        processor_edges = {et: ei for et, ei in data.edge_index_dict.items()
                    if "_target" not in et[2]}

        processed_x_dict = encoded_mesh_features
        for i in range(self.num_message_passing_steps):
            residual_x_dict = processed_x_dict

            # Apply one step of message passing using gradient checkpointing
            processed_x_dict = checkpoint.checkpoint(
                self.layers[i], processed_x_dict, step_info["edge_index_dict"], use_reentrant=False
            )

            # Add residual connection and apply layer norm
            for node_type in processed_x_dict:
                processed_x_dict[node_type] = self.norms[i][node_type](
                    processed_x_dict[node_type] + residual_x_dict[node_type]
                )

        return processed_x_dict
