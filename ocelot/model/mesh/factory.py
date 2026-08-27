from ocelot.configs.model_config import MeshConfig
from ocelot.model.mesh.mesh import Mesh
from ocelot.model.mesh.fixed_mesh import FixedMesh
from ocelot.model.mesh.hierarchical_mesh import HierarchicalMesh


MeshTypes = {
    "fixed": FixedMesh,
    "hierarchical": HierarchicalMesh
}

def make(mesh_config: MeshConfig, plot: bool = False) -> Mesh:
        if mesh_config.type not in MeshTypes:
            raise ValueError(f"Unknown mesh_type: {mesh_config.type}")

        mesh = MeshTypes[mesh_config.type](mesh_config)

        if plot:
            mesh.plot()

        return mesh
