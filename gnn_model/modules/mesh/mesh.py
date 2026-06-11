from matplotlib import pyplot as plt

import torch


DEFAULT_DTYPE = torch.float32

GC_SPATIAL_FEATURES_KWARGS = {
    "add_node_positions": False,
    "add_node_latitude": True,
    "add_node_longitude": True,
    "add_relative_positions": True,
    "relative_longitude_local_coordinates": True,
    "relative_latitude_local_coordinates": True,
}

def obs_mesh_conn(
    grid_lat, grid_lon, m2m_graphs, mesh_lat_lon_list, mesh_list, o2m=True
):
    """Build observation-to-mesh or mesh-to-observation connectivity.

    Args:
        grid_lat: Observation/target latitudes in degrees.
        grid_lon: Observation/target longitudes in degrees.
        m2m_graphs: Mesh graph list from `create_mesh`.
        mesh_lat_lon_list: Mesh latitude/longitude arrays from `create_mesh`.
        mesh_list: Full mesh hierarchy from `create_mesh`.
        o2m: If true, build radius-query obs-to-mesh edges. If false, build
            mesh-to-target edges using containing mesh triangles.

    Returns:
        Tuple of `(edge_index, edge_attr)` as torch tensors.
    """

    # Create lat-lon grid
    grid_lat_lon_flat = np.stack((grid_lat, grid_lon), axis=1)  # shape (N, 2)
    num_grid_nodes = grid_lat_lon_flat.shape[0]
    # flattened, (num_grid_nodes, 2)

    # Because GC code returns indexes into flattened lat-lon matrix, we have to
    # re-map grid indices. We always work with lon-lat order, to be consistent
    # with WB2 data.
    # This creates the correct mapping for the grid indices

    # Grid2Mesh: Radius-based
    grid_con_mesh = m2m_graphs[0]  # Mesh graph that should be connected to grid
    grid_con_mesh_lat_lon = mesh_lat_lon_list[0]

    if o2m:
        # Compute maximum edge distance in finest mesh
        # pylint: disable-next=protected-access
        max_mesh_edge_len = _get_max_edge_distance(mesh_list[-1])
        g2m_connect_radius = 0.6 * max_mesh_edge_len
        g2m_grid_mesh_indices = gc_gm.radius_query_indices(
            grid_latitude=grid_lat,
            grid_longitude=grid_lon,
            mesh=grid_con_mesh,
            radius=g2m_connect_radius,
        )
        # Returns two arrays of node indices, each [num_edges]

        g2m_edge_index = np.stack(g2m_grid_mesh_indices, axis=0)
        g2m_edge_index_torch = torch.tensor(g2m_edge_index, dtype=torch.long)

        if g2m_edge_index.shape[1] == 0:
            g2m_features_torch = torch.empty((0, 4), dtype=DEFAULT_DTYPE)
        else:
            # Only care about edge features here
            _, _, g2m_features = gc_mu.get_bipartite_graph_spatial_features(
                senders_node_lat=grid_lat_lon_flat[:, 0],
                senders_node_lon=grid_lat_lon_flat[:, 1],
                senders=g2m_edge_index[0, :],
                receivers_node_lat=grid_con_mesh_lat_lon[:, 0],
                receivers_node_lon=grid_con_mesh_lat_lon[:, 1],
                receivers=g2m_edge_index[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            g2m_features_torch = torch.tensor(g2m_features, dtype=DEFAULT_DTYPE)

    else:

        # Mesh2Grid: Connect to containing mesh triangle
        m2g_grid_mesh_indices = gc_gm.in_mesh_triangle_indices(
            grid_latitude=grid_lat,
            grid_longitude=grid_lon,
            mesh=mesh_list[-1],
        )  # Note: Still returned in order (grid, mesh), need to inverse
        m2g_edge_index = np.stack(m2g_grid_mesh_indices[::-1], axis=0)
        m2g_edge_index_torch = torch.tensor(m2g_edge_index, dtype=torch.long)

        if m2g_edge_index.shape[1] == 0:
            m2g_features_torch = torch.empty((0, 4), dtype=DEFAULT_DTYPE)
        else:
            # Only care about edge features here
            _, _, m2g_features = gc_mu.get_bipartite_graph_spatial_features(
                senders_node_lat=grid_con_mesh_lat_lon[:, 0],
                senders_node_lon=grid_con_mesh_lat_lon[:, 1],
                senders=m2g_edge_index[0, :],
                receivers_node_lat=grid_lat_lon_flat[:, 0],
                receivers_node_lon=grid_lat_lon_flat[:, 1],
                receivers=m2g_edge_index[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            m2g_features_torch = torch.tensor(m2g_features, dtype=DEFAULT_DTYPE)

    num_mesh_nodes = grid_con_mesh_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes:.2f}")
    if o2m:
        return (g2m_edge_index_torch, g2m_features_torch)
    else:
        return (m2g_edge_index_torch, m2g_features_torch)

class Mesh(torch.nn.Module):
    def __init__(self, levels: int, resolution: int):
        super().__init__()

        self.m2m_graphs: torch.tensor = None
        self.mesh_lat_lon_list: list = None
        self.mesh_list: torch.tensor = None
        self.m2m_edge_index_torch: torch.tensor = None
        self.m2m_features_torch: torch.tensor = None
        self.mesh_features_torch: torch.tensor = None
        self.mesh_lat_lon_torch: torch.tensor = None

        self._create_mesh(levels, resolution)

    def plot():
        """
        Plot flattened global graph
        """
        fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H

        edge_index = self.m2m_edge_index_torch[0]  # (2, M)
        pos_lat_lon = self.mesh_lat_lon_torch[0]  # (N,

        # Fix for re-indexed edge indices only containing mesh nodes at
        # higher levels in hierarchy
        edge_index = edge_index - edge_index.min()

        if pyg.utils.is_undirected(edge_index):
            # Keep only 1 direction of edge_index
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

        # Move all to cpu and numpy, compute (in)-degrees
        degrees = (
            pyg.utils.degree(edge_index[1], num_nodes=pos_lat_lon.shape[0]).cpu().numpy()
        )
        edge_index = edge_index.cpu().numpy()
        # Make lon x-axis
        pos = torch.stack((pos_lat_lon[:, 1], pos_lat_lon[:, 0]), dim=1)
        pos = pos.cpu().numpy()

        # Plot edges
        from_pos = pos[edge_index[0]]  # (M/2, 2)
        to_pos = pos[edge_index[1]]  # (M/2, 2)
        edge_lines = np.stack((from_pos, to_pos), axis=1)
        axis.add_collection(
            matplotlib.collections.LineCollection(
                edge_lines, lw=0.4, colors="black", zorder=1
            )
        )

        # Plot nodes
        node_scatter = axis.scatter(
            pos[:, 0],
            pos[:, 1],
            c=degrees,
            s=3,
            marker="o",
            zorder=2,
            cmap="viridis",
            clim=None,
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")

        plt.colorbar(node_scatter, aspect=50)

        if title is not None:
            axis.set_title(title)

        return fig, axis
    

    def _register_buffers(self):
        self.register_buffer("mesh_x", self._as_f32(self.mesh_features_torch[0]))
        self.register_buffer("mesh_edge_index", self._as_i64(self.m2m_edge_index_torch[0]))
        self.register_buffer("mesh_edge_attr", self._as_f32(self.m2m_features_torch[0]))


    def _create_mesh(self, levels: int, splits: int):
        mesh_list = gc_im.get_hierarchy_of_triangular_meshes_for_sphere(splits)
        if levels is not None:
            assert (levels <= splits + 1), \
                    f"Can not keep {levels} levels when doing {splits} splits"
            mesh_list = mesh_list[-levels:]

        self.m2m_graphs = self._create_m2m_graph(mesh_list)

        m2m_edge_index_list = []
        m2m_features_list = []
        mesh_features_list = []
        mesh_lat_lon_list = []
        for mesh_graph in m2m_graphs:
            mesh_edge_index = np.stack(gc_im.faces_to_edges(mesh_graph.faces), axis=0)
            m2m_edge_index_list.append(mesh_edge_index)

            # Compute features
            mesh_lat_lon = vertice_cart_to_lat_lon(mesh_graph.vertices)  # (N, 2)
            mesh_features, m2m_features = gc_mu.get_graph_spatial_features(
                node_lat=mesh_lat_lon[:, 0],
                node_lon=mesh_lat_lon[:, 1],
                senders=mesh_edge_index[0, :],
                receivers=mesh_edge_index[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            mesh_features_list.append(mesh_features)
            m2m_features_list.append(m2m_features)
            mesh_lat_lon_list.append(mesh_lat_lon)

            # Check that indexing is correct
            _, mesh_theta = gc_mu.lat_lon_deg_to_spherical(
                mesh_lat_lon[:, 0],
                mesh_lat_lon[:, 1],
            )
            assert np.sum(np.abs(mesh_features[:, 0] - np.cos(mesh_theta))) <= 1e-10

        # Convert to torch
        self.m2m_edge_index_torch = [
            torch.tensor(mesh_ei, dtype=torch.long) for mesh_ei in m2m_edge_index_list
        ]
        self.m2m_features_torch = [
            torch.tensor(m2m_features, dtype=DEFAULT_DTYPE)
            for m2m_features in m2m_features_list
        ]
        self.mesh_features_torch = [
            torch.tensor(mesh_features, dtype=DEFAULT_DTYPE)
            for mesh_features in mesh_features_list
        ]
        self.mesh_lat_lon_torch = [
            torch.tensor(mesh_lat_lon, dtype=DEFAULT_DTYPE)
            for mesh_lat_lon in mesh_lat_lon_list
        ]

        self.mesh_lat_lon_list = mesh_lat_lon_list
        self.mesh_list = mesh_list

    # This should be implemented by subclasses, as the way we create the m2m graph depends on the mesh type
    def _create_m2m_graph(self, mesh_list):
        raise NotImplementedError("This should be implemented by subclasses")

    @staticmethod
    def _as_f32(x):
        import torch

        return x.clone().detach().to(torch.float32) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _as_i64(x):
        import torch

        return x.clone().detach().to(torch.long) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
    