"""
Base class for hierarchical mesh processor modules.

Author: Azadeh Gholoubi

"""

import torch.nn as nn

from .processor_base import ProcessorBase
from ..mesh.hierarchical_mesh import HierarchyMesh


class HierarchicalProcessorBase(ProcessorBase):
    def __init__(
        self,
        mesh: HierarchyMesh,
        hidden_dim: int,
        num_levels: int,
        num_message_passing_steps: int = 4,
    ):
        super().__init__(hidden_dim, num_levels, num_message_passing_steps)

        # Coarse→fine conditioning: project coarse features to fine level
        # This gives coarse levels indirect supervision through fine level's loss
        self.coarse_to_fine_norm = nn.LayerNorm(hidden_dim)  # Normalize coarse features
        self.coarse_to_fine_proj = nn.Linear(hidden_dim, hidden_dim)  # Project to delta
        # Gating: allows model to control how much coarse info to use
        self.coarse_to_fine_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [fine; coarse] → gate
            nn.Sigmoid()  # Gate values in [0, 1]
        )
