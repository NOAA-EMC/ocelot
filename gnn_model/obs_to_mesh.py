import torch
from sklearn.neighbors import NearestNeighbors

from timing_utils import timing_resource_decorator


class ObsMeshCutoffConnector:
    """
    Computes cutoff-based edges to connect observation nodes to mesh nodes
    based on a geodesic distance threshold.

    Attributes:
        cutoff_factor (float): Scaling factor to adjust the cutoff radius.
        radius (float or None): Computed cutoff radius based on the mesh node distances.

    Methods:
        compute_cutoff_radius(mesh_latlon_rad):
            Computes the cutoff radius using the maximum geodesic neighbor distance.

        add_edges(graph, obs_latlon_rad, mesh_latlon_rad):
            Establishes directional edges from observations to mesh nodes based
            on the computed cutoff radius.
    """

    def __init__(self, cutoff_factor: float, metric: str = "haversine"):
        """
        Initializes the ObsMeshCutoffConnector class with a cutoff factor.

        Parameters:
            cutoff_factor (float): Scaling factor for determining the cutoff radius.
            metric (str): Distance metric to use for nearest neighbor search (default: 'haversine').
        """
        self.cutoff_factor = cutoff_factor
        self.metric = metric
        self.radius = None

    @timing_resource_decorator
    def compute_cutoff_radius(self, mesh_latlon_rad):
        """
        Computes the cutoff radius using the Haversine metric, based on the
        maximum distance between mesh node neighbors.

        Parameters:
            mesh_latlon_rad (numpy.ndarray): Array of shape (N, 2) containing
                                             mesh node coordinates (latitude, longitude in radians).

        Returns:
            float: The computed cutoff radius.
        """
        knn = NearestNeighbors(n_neighbors=2, metric=self.metric)
        knn.fit(mesh_latlon_rad)
        dists, _ = knn.kneighbors(mesh_latlon_rad)
        self.radius = dists[dists > 0].max() * self.cutoff_factor
        return self.radius

    def add_edges(self, obs_latlon_rad, mesh_latlon_rad, return_edge_attr=False):
        """
        Adds edges from observation nodes to mesh nodes based on a cutoff radius.

        Parameters:
            obs_latlon_rad (numpy.ndarray): Array of shape (M, 2) containing
                                            observation node coordinates (lat, lon in radians).
            mesh_latlon_rad (numpy.ndarray): Array of shape (N, 2) containing
                                             mesh node coordinates (lat, lon in radians).
            return_edge_attr (bool): If True, return geodesic distances as edge attributes.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - edge_index (torch.Tensor): Tensor of shape (2, E), where E is the number of edges.
                - edge_attr (torch.Tensor, optional): Edge weights if `return_edge_attr=True`.
        """
        if self.radius is None:
            self.compute_cutoff_radius(mesh_latlon_rad)

        knn = NearestNeighbors(metric=self.metric)
        knn.fit(mesh_latlon_rad)

        # Find mesh nodes within the cutoff radius for each observation node
        distances, indices = knn.radius_neighbors(obs_latlon_rad, radius=self.radius)

        obs_to_mesh_edges = []
        edge_weights = []

        for obs_idx, mesh_neighbors in enumerate(indices):
            for mesh_idx, dist in zip(mesh_neighbors, distances[obs_idx]):
                obs_to_mesh_edges.append([obs_idx, mesh_idx])
                if return_edge_attr:
                    edge_weights.append(dist)

        if len(obs_to_mesh_edges) == 0:
            print("Warning: No obs-to-mesh edges were created. Check cutoff radius or input coordinates.")

        edge_index_obs_to_mesh = torch.tensor(obs_to_mesh_edges, dtype=torch.long).t().contiguous()

        if return_edge_attr:
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            return edge_index_obs_to_mesh, edge_attr

        return edge_index_obs_to_mesh
