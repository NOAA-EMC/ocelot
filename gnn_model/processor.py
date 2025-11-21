import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from interaction_network import InteractionNetwork
import torch.utils.checkpoint as checkpoint


class Processor(nn.Module):
    """
    A Processor module that applies multiple steps of message passing using
    InteractionNetwork blocks, inspired by graphcast's processor.
    This module handles the core GNN processing, including the message-passing
    loop and residual connections.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        num_message_passing_steps: int,
    ):
        super().__init__()
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

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Processes the graph through multiple message-passing steps.
        """
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
