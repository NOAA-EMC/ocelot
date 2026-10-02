"""
Base class for hierarchical mesh processor modules.

Author: Azadeh Gholoubi

"""

import torch.nn as nn
from ocelot.model.processor.processor_config import ProcessorConfig
from ocelot.model.graph.graph_schema import GraphSchema

from ocelot.model.processor.processor_base import ProcessorBase
from ocelot.model.mesh.hierarchical_mesh import HierarchicalMesh


class HierarchicalProcessorBase(ProcessorBase):
    def __init__(self,
                 mesh: HierarchicalMesh,
                 processor_config: ProcessorConfig,
                 graph_schema: GraphSchema):
        if not isinstance(mesh, HierarchicalMesh):
            raise TypeError(f"Expected HierarchicalMesh, got {type(mesh).__name__}")

        super().__init__(mesh)

        # Coarse→fine conditioning: project coarse features to fine level
        # This gives coarse levels indirect supervision through fine level's loss
        self.coarse_to_fine_norm = nn.LayerNorm(processor_config.hidden_dim)  # Normalize coarse features
        self.coarse_to_fine_proj = nn.Linear(processor_config.hidden_dim, processor_config.hidden_dim)  # Project to delta
        # Gating: allows model to control how much coarse info to use
        self.coarse_to_fine_gate = nn.Sequential(
            nn.Linear(processor_config.hidden_dim * 2, processor_config.hidden_dim),  # [fine; coarse] → gate
            nn.Sigmoid()  # Gate values in [0, 1]
        )
