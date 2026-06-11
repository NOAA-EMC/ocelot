import matplotlib.pyplot as plt

from .mesh import Mesh
from .fixed_mesh import FixedMesh
from .hierarchical_mesh import HierarchicalMesh


MeshTypes = {
    "fixed": FixedMesh,
    "hierarchical": HierarchicalMesh
}

class MeshFactory():
    def build(self, mesh_type: str, splits: int, mesh_levels: int, plot: bool = False) -> Mesh:
        mesh = MeshTypes[mesh_type](levels=mesh_levels, splits=splits)

        if plot:
            mesh.plot()

        return mesh
