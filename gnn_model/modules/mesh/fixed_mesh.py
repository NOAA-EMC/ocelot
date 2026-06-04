import numpy as np

from .mesh import Mesh
from . import icosahedral_mesh_builder as gc_im


class FixedMesh(Mesh):
    def __init__(self, levels: int, splits: int, plot: bool = False):
        super().__init__(levels, splits)
        self._register_buffers()

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
