
from .mesh import Mesh
from .fixed_mesh import FixedMesh
from .hierarchical_mesh import HierarchicalMesh


MeshTypes = {
    "fixed": FixedMesh,
    "hierarchical": HierarchicalMesh
}

class MeshFactory:
    @staticmethod
    def build(mesh_type: str, levels: int, splits: int, plot: bool = False) -> Mesh:
        mesh = MeshTypes[mesh_type](levels=levels, splits=splits)

        if plot:
            mesh.plot()

        return mesh
