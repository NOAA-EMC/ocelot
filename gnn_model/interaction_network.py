import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from typing import Dict, List, Tuple
from utils import make_mlp


class InteractionNetwork(nn.Module):
    """
    A PyTorch implementation of an Interaction Network for heterogeneous graphs,
    inspired by the Graph Nets library and DeepMind's graphcast model.

    This network performs one step of message passing:
    1. Updates all edge features based on the features of the connected nodes.
    2. Aggregates messages from edges at the destination nodes.
    3. Updates all node features based on their own features and aggregated messages.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_types: List[str],
        edge_types: List[tuple[str, str, str]],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Create MLPs for updating each edge type
        self.edge_models = nn.ModuleDict()
        for edge_type in edge_types:
            src, _, dst = edge_type
            # Input to edge MLP: edge_features + src_node_features + dst_node_features
            # Note: We assume edge features are handled by the caller for simplicity here,
            # focusing on node features influencing the message.
            input_dim = 2 * hidden_dim  # src_features + dst_features
            self.edge_models[self._edge_key(edge_type)] = make_mlp(
                [input_dim] + [hidden_dim, hidden_dim]
            )

        # Create MLPs for updating each node type
        self.node_models = nn.ModuleDict()
        for node_type in node_types:
            # Input to node MLP: node_features + aggregated_messages
            input_dim = 2 * hidden_dim
            self.node_models[node_type] = make_mlp(
                [input_dim] + [hidden_dim, hidden_dim]
            )

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Performs one step of message passing.

        Args:
            x_dict: A dictionary of node features for each node type.
            edge_index_dict: A dictionary of edge indices for each edge type.

        Returns:
            A dictionary of updated node features for each node type.
        """
        # 1. Update edge features (compute messages)
        messages = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            src_nodes = x_dict[src_type][edge_index[0]]
            dst_nodes = x_dict[dst_type][edge_index[1]]

            # Concatenate features to form message input
            message_input = torch.cat([src_nodes, dst_nodes], dim=-1)
            messages[edge_type] = self.edge_models[self._edge_key(edge_type)](
                message_input
            )

        # 2. Aggregate messages at destination nodes
        aggregated_messages = {node_type: [] for node_type in x_dict}
        for edge_type, msg in messages.items():
            src_type, _, dst_type = edge_type
            dst_index = edge_index_dict[edge_type][1]
            # The scatter function from PyG is the correct way to do this.
            # It aggregates messages at the destination nodes.
            aggregated_messages[dst_type].append(
                scatter(msg, dst_index, dim=0, dim_size=x_dict[dst_type].shape[0])
            )

        # Combine aggregated messages for each node type
        final_aggregated = {}
        for node_type, msg_list in aggregated_messages.items():
            if msg_list:
                # Sum messages from different edge types arriving at the same node type
                final_aggregated[node_type] = sum(msg_list)
            else:
                # If no messages arrive, use a zero tensor of the correct shape
                final_aggregated[node_type] = torch.zeros_like(x_dict[node_type])

        # 3. Update node features
        updated_x_dict = {}
        for node_type, x in x_dict.items():
            node_input = torch.cat([x, final_aggregated[node_type]], dim=-1)
            updated_x_dict[node_type] = self.node_models[node_type](node_input)

        return updated_x_dict

    def _edge_key(self, edge_type: Tuple[str, str, str]) -> str:
        """Converts an edge_type tuple to a string key for ModuleDict."""
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
