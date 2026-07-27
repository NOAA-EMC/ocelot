import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ocelot import utils


class InteractionNetwork(nn.Module):
    """Memory-optimized Interaction Network for heterogeneous graphs."""

    def __init__(self, hidden_dim: int, node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 scatter_chunk_size: int = 50000,
                 edge_chunk_size: int = 50000,
                 process_edge_types_sequentially: bool = True,
                 edge_attr_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.scatter_chunk_size = scatter_chunk_size
        self.edge_chunk_size = edge_chunk_size
        self.process_edge_types_sequentially = process_edge_types_sequentially

        # Edge models
        self.edge_models = nn.ModuleDict()
        for edge_type in edge_types:
            input_dim = 2 * hidden_dim + edge_attr_dim
            self.edge_models[self._edge_key(edge_type)] = utils.make_mlp(
                [input_dim] + [hidden_dim, hidden_dim]
            )

        # Node models
        self.node_models = nn.ModuleDict()
        for node_type in node_types:
            input_dim = 2 * hidden_dim
            self.node_models[node_type] = utils.make_mlp(
                [input_dim] + [hidden_dim, hidden_dim]
            )

    def forward(self,
                x_dict: Dict[str,
                             torch.Tensor],
                edge_index_dict: Dict[Tuple[str,
                                            str,
                                            str],
                                      torch.Tensor],
                edge_attr_dict: Dict[Tuple[str,
                                           str,
                                           str],
                                     torch.Tensor] = None) -> Dict[str,
                                                                   torch.Tensor]:

        if self.process_edge_types_sequentially:
            return self._forward_sequential(
                x_dict, edge_index_dict, edge_attr_dict)
        else:
            return self._forward_parallel(
                x_dict, edge_index_dict, edge_attr_dict)

    def _forward_sequential(
            self,
            x_dict,
            edge_index_dict,
            edge_attr_dict=None):
        """Process edge types one at a time to minimize peak memory."""
        aggregated_messages = {node_type: torch.zeros_like(x_dict[node_type])
                               for node_type in x_dict}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_key = self._edge_key(edge_type)
            edge_attr = edge_attr_dict.get(
                edge_type) if edge_attr_dict is not None else None
            msg = self._compute_and_aggregate_edge_type(
                x_dict[src_type],
                x_dict[dst_type],
                edge_index,
                edge_key,
                x_dict[dst_type].shape[0],
                edge_attr,
            )
            aggregated_messages[dst_type] = aggregated_messages[dst_type] + msg

        # Update nodes — pop each aggregated message to free it immediately
        # after use
        updated_x_dict = {}
        for node_type, x in x_dict.items():
            node_input = torch.cat(
                [x, aggregated_messages.pop(node_type)], dim=-1)
            updated_x_dict[node_type] = self.node_models[node_type](node_input)
            del node_input

        return updated_x_dict

    def _compute_and_aggregate_edge_type(
            self,
            src_x,
            dst_x,
            edge_index,
            edge_key,
            num_dst_nodes,
            edge_attr=None):
        """Compute messages and aggregate for a single edge type.

        Processes edges in chunks to keep peak memory at chunk_size*2*hidden_dim
        rather than num_edges*2*hidden_dim during gradient checkpoint recomputation.
        """
        num_edges = edge_index.shape[1]
        output = torch.zeros(num_dst_nodes, self.hidden_dim,
                             dtype=src_x.dtype, device=src_x.device)

        for start in range(0, num_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, num_edges)
            idx = edge_index[:, start:end]
            parts = [src_x[idx[0]], dst_x[idx[1]]]
            if edge_attr is not None:
                parts.append(edge_attr[start:end])
            chunk_input = torch.cat(parts, dim=-1)
            chunk_msg = self.edge_models[edge_key](chunk_input)
            output.index_add_(0, idx[1], chunk_msg)

        return output

    def _forward_parallel(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Original parallel processing (higher memory usage)."""
        messages = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_nodes = x_dict[src_type][edge_index[0]]
            dst_nodes = x_dict[dst_type][edge_index[1]]
            parts = [src_nodes, dst_nodes]
            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                parts.append(edge_attr_dict[edge_type])
            message_input = torch.cat(parts, dim=-1)
            messages[edge_type] = self.edge_models[self._edge_key(
                edge_type)](message_input)

        # Aggregate
        aggregated_messages = {node_type: [] for node_type in x_dict}
        for edge_type, msg in messages.items():
            _, _, dst_type = edge_type
            dst_index = edge_index_dict[edge_type][1]
            aggregated_messages[dst_type].append(
                self._scatter_chunked(
                    msg, dst_index, x_dict[dst_type].shape[0]))

        final_aggregated = {}
        for node_type, msg_list in aggregated_messages.items():
            if msg_list:
                final_aggregated[node_type] = sum(msg_list)
            else:
                final_aggregated[node_type] = torch.zeros_like(
                    x_dict[node_type])

        # Update nodes
        updated_x_dict = {}
        for node_type, x in x_dict.items():
            node_input = torch.cat([x, final_aggregated[node_type]], dim=-1)
            updated_x_dict[node_type] = self.node_models[node_type](node_input)

        return updated_x_dict

    def _scatter_chunked(self, msg: torch.Tensor, dst_index: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Memory-efficient chunked scatter."""
        from torch_geometric.utils import scatter

        num_edges = msg.shape[0]

        if num_edges <= self.scatter_chunk_size:
            return scatter(
                msg,
                dst_index,
                dim=0,
                dim_size=num_nodes,
                reduce='sum')

        output = torch.zeros(num_nodes, msg.shape[1],
                             dtype=msg.dtype, device=msg.device)

        for start_idx in range(0, num_edges, self.scatter_chunk_size):
            end_idx = min(start_idx + self.scatter_chunk_size, num_edges)
            msg_chunk = msg[start_idx:end_idx]
            idx_chunk = dst_index[start_idx:end_idx]
            output.index_add_(0, idx_chunk, msg_chunk)

        return output

    def _edge_key(self, edge_type: Tuple[str, str, str]) -> str:
        return f'{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'
