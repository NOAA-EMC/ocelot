"""Fixed-mesh interaction processor for OCELOT.

Author: Azadeh Gholoubi
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ocelot.model.mesh.fixed_mesh import FixedMesh
from ocelot.configs.model_config import InteractionProcessorConfig
from ocelot.model.graph_schema import GraphSchema
from ocelot.model.processor.interaction_network import InteractionNetwork
from ocelot.model.processor.flat_processor_base import FlatProcessorBase
from ocelot.model.processor.processor_base import ProcessorContext


class InteractionProcessor(FlatProcessorBase):
    """
    A Processor module that applies multiple steps of message passing using
    InteractionNetwork blocks, inspired by graphcast's processor.
    This module handles the core GNN processing, including the message-passing
    loop and residual connections.
    """

    def __init__(self,
                 mesh: FixedMesh,
                 processor_config: InteractionProcessorConfig,
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

    def forward(self, step: int, encoded_mesh_features: torch.Tensor, context: ProcessorContext) -> torch.Tensor:
        """
        Evolve the mesh state one latent step.

        Input nodes are re-injected unchanged each step and evolve only within it;
        only the mesh state is carried to the next step.

        Returns:
            [num_graphs * N_mesh, H] updated mesh state
        """
        x_dict = dict(context.node_features)
        x_dict[GraphSchema.MESH] = encoded_mesh_features

        for layer, norms in zip(self.layers, self.norms):
            residual_x_dict = x_dict
            x_dict = checkpoint.checkpoint(layer, x_dict, context.edge_index_dict, use_reentrant=False)
            x_dict = {nt: norms[nt](x_dict[nt] + residual_x_dict[nt]) for nt in x_dict}

        return x_dict[GraphSchema.MESH]
