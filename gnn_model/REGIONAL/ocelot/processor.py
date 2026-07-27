import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from torch.utils.checkpoint import checkpoint

from ocelot.interaction_network import InteractionNetwork


class Processor(nn.Module):
    """
    A Processor module that applies multiple steps of message passing using
    InteractionNetwork blocks, inspired by graphcast's processor.

    This module handles the core GNN processing, including the message-passing
    loop and residual connections.
    """

    def __init__(self,
                 hidden_dim: int,
                 node_types: List[str],
                 edge_types: List[Tuple[str,
                                        str,
                                        str]],
                 num_message_passing_steps: int,
                 use_gradient_checkpointing: bool = True,
                 scatter_chunk_size: int = 50000,
                 edge_chunk_size: int = 10000,
                 share_weights: bool = False,
                 edge_attr_dim: int = 0):
        """
        Args:
            hidden_dim: Hidden dimension size
            node_types: List of node type names
            edge_types: List of edge type tuples (src, edge_type, dst)
            num_message_passing_steps: Number of message passing iterations
            use_gradient_checkpointing: Whether to use gradient checkpointing to save memory
            scatter_chunk_size: Chunk size for scatter operations
            share_weights: If True, share the same InteractionNetwork across all steps.
                          If False, use separate networks for each step.
        """
        super().__init__()
        self.num_message_passing_steps = num_message_passing_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.share_weights = share_weights

        if share_weights:
            self.shared_layer = InteractionNetwork(
                hidden_dim, node_types, edge_types,
                scatter_chunk_size=scatter_chunk_size,
                edge_chunk_size=edge_chunk_size,
                edge_attr_dim=edge_attr_dim,
            )
            self.layers = None
        else:
            self.layers = nn.ModuleList()
            for _ in range(num_message_passing_steps):
                self.layers.append(
                    InteractionNetwork(hidden_dim, node_types, edge_types,
                                       scatter_chunk_size=scatter_chunk_size,
                                       edge_chunk_size=edge_chunk_size,
                                       edge_attr_dim=edge_attr_dim)
                )
            self.shared_layer = None

        # Always use separate norms for each step (even with shared weights)
        # This allows each step to normalize differently
        self.norms = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            self.norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim) for node_type in node_types
            }))

    def forward(self,
                x_dict: Dict[str,
                             torch.Tensor],
                edge_index_dict: Dict[str,
                                      torch.Tensor],
                edge_attr_dict: Dict[str,
                                     torch.Tensor] = None) -> Dict[str,
                                                                   torch.Tensor]:
        """
        Processes the graph through multiple message-passing steps.

        With gradient checkpointing enabled, intermediate activations are NOT stored
        during forward pass. They are recomputed during backward pass, trading
        compute for memory (30-40% memory savings).

        Args:
            x_dict: A dictionary of node features for each node type.
            edge_index_dict: A dictionary of edge indices for each edge type.

        Returns:
            A dictionary of processed node features for each node type.
        """
        processed_x_dict = x_dict

        for i in range(self.num_message_passing_steps):
            # Select which layer to use
            layer = self.shared_layer if self.share_weights else self.layers[i]
            norms = self.norms[i]

            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing: don't store activations.
                # Bind layer/norms via default args to avoid a closure over
                # loop variables — the lambda would otherwise capture the name
                # binding rather than the value, causing the wrong layer to be
                # used during checkpoint recomputation.
                processed_x_dict = checkpoint(
                    lambda x_d,
                    e_d,
                    ea_d,
                    _l=layer,
                    _n=norms: self._process_step(
                        _l,
                        _n,
                        x_d,
                        e_d,
                        ea_d),
                    processed_x_dict,
                    edge_index_dict,
                    edge_attr_dict,
                    use_reentrant=False)
            else:
                # Normal forward pass: store all activations
                processed_x_dict = self._process_step(
                    layer,
                    norms,
                    processed_x_dict,
                    edge_index_dict,
                    edge_attr_dict,
                )

        return processed_x_dict

    def _process_step(
            self,
            layer,
            norms,
            x_dict,
            edge_index_dict,
            edge_attr_dict=None):
        """
        Single message passing step - extracted for gradient checkpointing.

        Args:
            layer: The InteractionNetwork layer
            norms: Dictionary of LayerNorm modules for each node type
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
            edge_attr_dict: Optional edge attribute dictionary

        Returns:
            Processed node features dictionary
        """
        residual_x_dict = x_dict

        # Apply message passing
        processed_x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        # Add residual and normalize
        for node_type in processed_x_dict:
            processed_x_dict[node_type] = norms[node_type](
                processed_x_dict[node_type] + residual_x_dict[node_type]
            )

        return processed_x_dict
