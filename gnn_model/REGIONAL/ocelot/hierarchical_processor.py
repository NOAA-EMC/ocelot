# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Hierarchical mesh processor: intra-level message passing plus up/down
# between icosahedral hierarchy levels (U-Net style). Ported from gnn_model
# interaction_hierarchical_processor.py for use with ocelot.interaction_net.

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ocelot.interaction_net import InteractionNet


def _edge_placeholder(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Returns edge_attr if provided, otherwise a zero placeholder for the edge MLP."""
    if edge_attr is not None:
        return edge_attr
    e = int(edge_index.shape[1])
    return torch.zeros((e, dim), device=edge_index.device, dtype=torch.float32)


class HierarchicalProcessor(nn.Module):
    """Message passing across multiple mesh levels (fine → coarse → fine).

    Level ordering matches ``create_mesh(..., hierarchical=True)``:
    index 0 = finest, index ``num_levels - 1`` = coarsest.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        num_message_passing_steps: int = 4,
        hidden_layers: int = 2,
        intra_edge_dims: Optional[List[int]] = None,
        up_edge_dims: Optional[List[int]] = None,
        down_edge_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_message_passing_steps = num_message_passing_steps

        self.intra_level_layers = nn.ModuleList()
        for level in range(num_levels):
            edge_dim = intra_edge_dims[level] if intra_edge_dims is not None else None
            level_layers = nn.ModuleList()
            for _ in range(num_message_passing_steps):
                level_layers.append(
                    InteractionNet(
                        send_dim=hidden_dim,
                        rec_dim=hidden_dim,
                        edge_dim=edge_dim,
                        hidden_layers=hidden_layers,
                    )
                )
            self.intra_level_layers.append(level_layers)

        self.up_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            edge_dim = up_edge_dims[level] if up_edge_dims is not None else None
            self.up_layers.append(
                InteractionNet(
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_layers=hidden_layers,
                )
            )

        self.down_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            edge_dim = down_edge_dims[level] if down_edge_dims is not None else None
            self.down_layers.append(
                InteractionNet(
                    send_dim=hidden_dim,
                    rec_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_layers=hidden_layers,
                )
            )

        self.level_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_levels)]
        )

    def forward(
        self,
        mesh_features_list: List[torch.Tensor],
        mesh_edge_index_list: List[torch.Tensor],
        mesh_edge_attr_list: List[Optional[torch.Tensor]],
        up_edge_index_list: List[torch.Tensor],
        up_edge_attr_list: List[Optional[torch.Tensor]],
        down_edge_index_list: List[torch.Tensor],
        down_edge_attr_list: List[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        level_features = [feat.clone() for feat in mesh_features_list]
        hd = self.hidden_dim

        for level in range(self.num_levels):
            residual = level_features[level]
            ei = mesh_edge_index_list[level]
            er = _edge_placeholder(ei, mesh_edge_attr_list[level], hd).to(
                dtype=level_features[level].dtype
            )
            for step in range(self.num_message_passing_steps):
                level_features[level] = self.intra_level_layers[level][step](
                    send_rep=level_features[level],
                    rec_rep=level_features[level],
                    edge_rep=er,
                    edge_index=ei,
                )
            level_features[level] = self.level_norms[level](
                level_features[level] + residual
            )

        for level in range(self.num_levels - 1):
            uei = up_edge_index_list[level]
            ua = _edge_placeholder(uei, up_edge_attr_list[level], hd).to(
                dtype=level_features[level].dtype
            )
            coarse_update = self.up_layers[level](
                send_rep=level_features[level],
                rec_rep=level_features[level + 1],
                edge_rep=ua,
                edge_index=uei,
            )
            level_features[level +
                           1] = level_features[level + 1] + coarse_update

        for level in reversed(range(self.num_levels - 1)):
            dei = down_edge_index_list[level]
            da = _edge_placeholder(dei, down_edge_attr_list[level], hd).to(
                dtype=level_features[level].dtype
            )
            fine_update = self.down_layers[level](
                send_rep=level_features[level + 1],
                rec_rep=level_features[level],
                edge_rep=da,
                edge_index=dei,
            )
            level_features[level] = level_features[level] + fine_update

        return level_features
