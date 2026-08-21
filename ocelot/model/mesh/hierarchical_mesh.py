import numpy as np
import torch

from ocelot.model.mesh.mesh import Mesh, GC_SPATIAL_FEATURES_KWARGS, DEFAULT_DTYPE
from ocelot.model.mesh.deepmind import icosahedral_mesh as gc_im
from ocelot.model.mesh.deepmind import model_utils as gc_mu


class HierarchicalMesh(Mesh):
    def __init__(self, levels: int, resolution: int, plot: bool = False):
        super().__init__(levels, resolution)

        self.mesh_up_ei_list = []
        self.mesh_down_ei_list = []
        self.mesh_up_features_list = []
        self.mesh_down_features_list = []

        self._init_hierarchical_edges()
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
            "mesh_up_ei_list": self.mesh_up_ei_list,
            "mesh_down_ei_list": self.mesh_down_ei_list,
            "mesh_up_features_list": self.mesh_up_features_list,
            "mesh_down_features_list": self.mesh_down_features_list,
        }
        return structure

    def _init_hierarchical_edges(self):
        # Up and down edges for hierarchy
        # Reuse code for connecting grid to mesh?
        for from_mesh, to_mesh in zip(self.m2m_graphs[:-1], self.m2m_graphs[1:]):
            mesh_up_ei = self._inter_mesh_connection(from_mesh, to_mesh)
            # Down is opposite direction of up
            mesh_down_ei = np.stack((mesh_up_ei[1, :], mesh_up_ei[0, :]), axis=0)
            self.mesh_up_ei_list.append(torch.tensor(mesh_up_ei, dtype=torch.long))
            self.mesh_down_ei_list.append(torch.tensor(mesh_down_ei, dtype=torch.long))

            from_mesh_lat_lon = self._vertice_cart_to_lat_lon(from_mesh.vertices)  # (N, 2)
            to_mesh_lat_lon = self._vertice_cart_to_lat_lon(to_mesh.vertices)  # (N, 2)

            # Extract features for hierarchical edges
            _, _, mesh_up_features = gc_mu.get_bipartite_graph_spatial_features(
                senders_node_lat=from_mesh_lat_lon[:, 0],
                senders_node_lon=from_mesh_lat_lon[:, 1],
                senders=mesh_up_ei[0, :],
                receivers_node_lat=to_mesh_lat_lon[:, 0],
                receivers_node_lon=to_mesh_lat_lon[:, 1],
                receivers=mesh_up_ei[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            _, _, mesh_down_features = gc_mu.get_bipartite_graph_spatial_features(
                senders_node_lat=to_mesh_lat_lon[:, 0],
                senders_node_lon=to_mesh_lat_lon[:, 1],
                senders=mesh_down_ei[0, :],
                receivers_node_lat=from_mesh_lat_lon[:, 0],
                receivers_node_lon=from_mesh_lat_lon[:, 1],
                receivers=mesh_down_ei[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            self.mesh_up_features_list.append(
                torch.tensor(mesh_up_features, dtype=DEFAULT_DTYPE)
            )
            self.mesh_down_features_list.append(
                torch.tensor(mesh_down_features, dtype=DEFAULT_DTYPE)
            )


    def _register_buffers(self):
        super()._register_buffers()

        for i, (mx, mei, mea) in enumerate(zip(self.mesh_features_torch,
                                               self.m2m_edge_index_torch,
                                               self.m2m_features_torch)):
            self.register_buffer(f"mesh_x_level_{i}", self._as_f32(mx))
            self.register_buffer(f"mesh_edge_index_level_{i}", self._as_i64(mei))
            self.register_buffer(f"mesh_edge_attr_level_{i}", self._as_f32(mea))

        if self.mesh_up_ei_list is not None:
            for i, (mue, muea, mde, mdea) in enumerate(zip(self.mesh_up_ei_list,
                                                           self.mesh_up_features_list,
                                                           self.mesh_down_ei_list,
                                                           self.mesh_down_features_list)):
                self.register_buffer(f"mesh_up_edge_index_{i}", self._as_i64(mue))
                self.register_buffer(f"mesh_up_edge_attr_{i}", self._as_f32(muea))
                self.register_buffer(f"mesh_down_edge_index_{i}", self._as_i64(mde))
                self.register_buffer(f"mesh_down_edge_attr_{i}", self._as_f32(mdea))


    def _create_m2m_graph(self, mesh_list):
        mesh_list_rev = list(reversed(mesh_list))  # 0 is finest graph now
        m2m_graphs = mesh_list_rev  # list of num_splitgraphs

        return m2m_graphs
    
    @staticmethod
    def _inter_mesh_connection(from_mesh, to_mesh):
        """
        Connect from_mesh to to_mesh
        """
        kd_tree = scipy.spatial.cKDTree(to_mesh.vertices)

        # Each node on lower (from) mesh will connect to 1 or 2 on level above
        # pylint: disable-next=protected-access
        radius = 1.1 * HierarchicalMesh._get_max_edge_distance(from_mesh)
        query_indices = kd_tree.query_ball_point(x=from_mesh.vertices, r=radius)

        from_edge_indices = []
        to_edge_indices = []
        for from_index, to_neighbors in enumerate(query_indices):
            from_edge_indices.append(np.repeat(from_index, len(to_neighbors)))
            to_edge_indices.append(to_neighbors)

        from_edge_indices = np.concatenate(from_edge_indices, axis=0).astype(int)
        to_edge_indices = np.concatenate(to_edge_indices, axis=0).astype(int)

        edge_index = np.stack((from_edge_indices, to_edge_indices), axis=0)  # (2, M)
        return edge_index
    
    @staticmethod
    def _vertice_cart_to_lat_lon(vertices):
        """
        Convert vertice positions to lat-lon

        vertices: (N_vert, 3), cartesian coordinates
        Returns: (N_vert, 2), lat-lon coordinates
        """
        phi, theta = gc_mu.cartesian_to_spherical(
            vertices[:, 0], vertices[:, 1], vertices[:, 2]
        )
        (
            nodes_lat,
            nodes_lon,
        ) = gc_mu.spherical_to_lat_lon(phi=phi, theta=theta)
        return np.stack((nodes_lat, nodes_lon), axis=1)  # (N, 2)
    
    @staticmethod
    def _get_max_edge_distance(mesh):
        """Return the maximum Euclidean edge length in a triangular mesh."""
        senders, receivers = gc_im.faces_to_edges(mesh.faces)
        edge_distances = np.linalg.norm(
            mesh.vertices[senders] - mesh.vertices[receivers], axis=-1
        )
        return edge_distances.max()
