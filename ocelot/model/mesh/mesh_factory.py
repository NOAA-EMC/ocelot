
from ocelot.model.mesh.mesh import Mesh
from ocelot.model.mesh.fixed_mesh import FixedMesh
from ocelot.model.mesh.hierarchical_mesh import HierarchicalMesh


MeshTypes = {
    "fixed": FixedMesh,
    "hierarchical": HierarchicalMesh
}

class MeshFactory:
    @staticmethod
    def build(mesh_type: str, levels: int, resolution: int, plot: bool = False) -> Mesh:
        if mesh_type not in MeshTypes:
            raise ValueError(f"Unknown mesh_type: {mesh_type}")

        if mesh_type == "fixed":
            mesh = FixedMesh(levels=levels, splits=resolution)
        else:
            mesh = HierarchicalMesh(levels=levels, resolution=resolution)

        if plot:
            mesh.plot()

        return mesh
