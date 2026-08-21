import numpy as np

from ocelot.model.mesh.mesh import Mesh
from ocelot.model.mesh.deepmind import icosahedral_mesh as gc_im


class FixedMesh(Mesh):
    def __init__(self, levels: int, splits: int, plot: bool = False):
        super().__init__(levels, splits)
        self._register_buffers()

    @property
    def mesh_structure(self) -> dict:
        structure = {
            "m2m_graphs": self.m2m_graphs,
            "mesh_lat_lon_list": self.mesh_lat_lon_list,
            "mesh_list": self.mesh_list,
            "m2m_edge_index_torch": self.m2m_edge_index_torch,
            "m2m_features_torch": self.m2m_features_torch,
            "mesh_features_torch": self.mesh_features_torch,
            "mesh_lat_lon_torch": self.mesh_lat_lon_torch,
        }
        return structure

    def _create_m2m_graph(self, mesh_list):

        # Merge meshes
        # Modify gc code, as this uses some python 3.10 things
        for mesh_i, mesh_ip1 in zip(mesh_list[:-1], mesh_list[1:]):
            # itertools.pairwise(mesh_list):
            num_nodes_mesh_i = mesh_i.vertices.shape[0]
            assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

        merged_mesh = gc_im.TriangularMesh(
            vertices=mesh_list[-1].vertices,
            faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
        )

        m2m_graphs = [merged_mesh]  # Should be list of len 1

        return m2m_graphs


