"""
Regional domain variant of HeteroObservationGraphModel.

Overrides create_mesh_structures() and setup_mesh() to work in 2D Cartesian
(projected) space instead of the global icosahedral/spherical approach.

Required args:
    x_min, x_max, y_min, y_max (float): lon/lat domain bounds in degrees
        (projected to Cartesian (x, y) internally; see create_regional_mesh_from_corners).
Optional args (same as global model):
    levels (int|None): max mesh levels
    hierarchical (bool): hierarchical mesh
    plot (bool): plot mesh during creation
"""

import torch
from torch_geometric.data import Data

from ocelot.hetero_observation_interaction import HeteroObservationGraphModel
from ocelot.create_mesh_graph_regional import create_regional_mesh_from_corners


class RegionalHeteroObservationGraphModel(HeteroObservationGraphModel):
    """
    Drop-in replacement for HeteroObservationGraphModel for regional domains.

    Mesh coordinates are 2D Cartesian (map-projected), so mesh node features
    are (x, y) positions and edge features are Euclidean distances/differences.
    Set args.mesh_feature_dim = 2 and args.x_min/x_max/y_min/y_max accordingly.
    """

    def create_mesh_structures(self) -> None:
        for attr in ("x_min", "x_max", "y_min", "y_max"):
            if getattr(self.args, attr, None) is None:
                raise ValueError(
                    f"args.{attr} must be set for RegionalHeteroObservationGraphModel"
                )

        x_min = float(self.args.x_min)
        x_max = float(self.args.x_max)
        y_min = float(self.args.y_min)
        y_max = float(self.args.y_max)

        mesh_structure = create_regional_mesh_from_corners(
            x_min, x_max, y_min, y_max, self.args)

        self.mesh_structure = mesh_structure
        self.is_hierarchical = bool(getattr(self.args, "hierarchical", False))
        self.num_mesh_levels = len(mesh_structure["m2m_graphs"])

        # mesh_graph is not used by setup_mesh() below, but kept for API
        # parity.
        self.mesh_graph = Data(
            pos=mesh_structure["m2m_graphs"][0].pos,
            m2m_graphs=mesh_structure["m2m_graphs"],
            edge_index=mesh_structure["m2m_graphs"][0].edge_index,
            edge_attr=mesh_structure["m2m_graphs"][0].edge_attr,
        )

        # setup_mesh() must run before setup_observation_networks() accesses
        # self.mesh_edge_attr to size the Processor's edge MLP.  In the global
        # model this is fine because hierarchical mode (HierarchicalProcessor)
        # never reads self.mesh_edge_attr, but the flat Processor does.
        self.setup_mesh()

    def setup_mesh(self):
        """
        Register static mesh tensors as buffers using Cartesian node positions.

        For a flat mesh all levels are combined into one m2m_graphs[0].
        For a hierarchical mesh each level is registered separately.
        """
        all_graphs = self.mesh_structure["m2m_graphs"]

        # Node features = projected (x, y) positions; cat across all levels.
        mesh_x = torch.cat([g.pos for g in all_graphs],
                           dim=0).to(torch.float32)
        mesh_edge_index = torch.cat([g.edge_index for g in all_graphs], dim=1)
        mesh_edge_attr = torch.cat(
            [g.edge_attr for g in all_graphs], dim=0).to(torch.float32)

        # Normalize coordinates to [-1, 1] using global min/max across all levels.
        # Raw Lambert Conformal x/y coords are in meters (O(1e6)), which overflow
        # fp16's max (~65504) and produce NaN in the mesh embedder.
        pos_min = mesh_x.min(0).values
        pos_max = mesh_x.max(0).values
        pos_scale = (pos_max - pos_min).clamp(min=1e-8)
        mesh_x = 2.0 * (mesh_x - pos_min) / pos_scale - 1.0

        self.register_buffer("mesh_x", mesh_x)
        self.register_buffer("mesh_edge_index", mesh_edge_index)
        self.register_buffer("mesh_edge_attr", mesh_edge_attr)

        if self.is_hierarchical:
            ms = self.mesh_structure
            for i, g in enumerate(all_graphs):
                pos_i = g.pos.clone().to(torch.float32)
                pos_i = 2.0 * (pos_i - pos_min) / pos_scale - 1.0
                self.register_buffer(f"mesh_x_level_{i}", pos_i)
                self.register_buffer(
                    f"mesh_edge_index_level_{i}",
                    g.edge_index.clone())
                self.register_buffer(
                    f"mesh_edge_attr_level_{i}",
                    g.edge_attr.clone().to(
                        torch.float32))
            for i, up_g in enumerate(ms.get("mesh_up_graphs", [])):
                self.register_buffer(
                    f"mesh_up_edge_index_{i}",
                    up_g.edge_index.clone())
                self.register_buffer(
                    f"mesh_up_edge_attr_{i}",
                    up_g.edge_attr.clone().to(
                        torch.float32))
            for i, dn_g in enumerate(ms.get("mesh_down_graphs", [])):
                self.register_buffer(
                    f"mesh_down_edge_index_{i}",
                    dn_g.edge_index.clone())
                self.register_buffer(
                    f"mesh_down_edge_attr_{i}",
                    dn_g.edge_attr.clone().to(
                        torch.float32))
