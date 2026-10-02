from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ocelot.logger import log
from ocelot.model.graph_schema import EdgeType


@dataclass(frozen=True)
class ProcessorContext:
    """Per-batch inputs shared by every latent step of a rollout."""

    num_steps: int
    node_features: Dict[str, torch.Tensor]  # encoded non-mesh nodes, fixed for the rollout
    edge_index_dict: Dict[EdgeType, torch.Tensor]  # batched processor edges, incl. mesh-to-mesh


class ProcessorBase(nn.Module):
    def __init__(self, mesh, enable_mesh_pred: bool = False):
        super().__init__()
        self.mesh = mesh
        self.enable_mesh_pred = enable_mesh_pred

    def reset(self):
        """Reset the processor state if needed."""
        pass

    @staticmethod
    def _get_latent_step_info(data: HeteroData) -> dict:
        """
        Extract information about latent steps from the batch.
        Returns dict with step mapping and number of steps.
        """
        step_info = {}
        max_step = -1

        # Find all step-specific target nodes and map them to base instruments
        for node_type in data.node_types:
            if "_target_step" in node_type:
                # Extract: atms_target_step0 -> (atms_target, 0)
                parts = node_type.split("_step")
                if len(parts) == 2:
                    base_type = parts[0]  # e.g., "atms_target"
                    try:
                        step_num = int(parts[1])
                        if base_type not in step_info:
                            step_info[base_type] = {}
                        step_info[base_type][step_num] = node_type
                        max_step = max(max_step, step_num)
                    except ValueError:
                        continue

        return {
            "step_mapping": step_info,
            "num_steps": max_step + 1 if max_step >= 0 else 0
        }
