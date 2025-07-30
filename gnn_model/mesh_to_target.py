import torch
from sklearn.neighbors import NearestNeighbors

import torch
import numpy as np
import trimesh
from icosahedral_mesh import TriangularMesh


class MeshTargetTriangleConnector:
    """
    Computes decoder edges by connecting each target point to the three
    vertices of the mesh triangle that contains it. This matches the
    method used in the original Ocelot2/GraphCast implementation.

    Methods:
        add_edges(mesh, target_latlon_rad):
            Connects each target location to the vertices of its containing mesh triangle.
    """

    def __init__(self):
        """Initializes the MeshTargetTriangleConnector class."""
        pass

    def add_edges(
        self,
        mesh: TriangularMesh,
        target_latlon_rad: np.ndarray,
        mesh_latlon_rad: np.ndarray,
    ):
        """
        Connects each target observation node to the three vertices of the
        mesh triangle it is contained within.

        Parameters:
            mesh (TriangularMesh): An object with `.vertices` and `.faces` attributes,
                                   as defined in `icosahedral_mesh.py`.
            target_latlon_rad (np.ndarray): Array of shape (M, 2) containing target node
                                            coordinates (latitude, longitude in radians).
            mesh_latlon_rad (np.ndarray): Not directly used in this method but kept for
                                          API consistency with other connectors.

        Returns:
            tuple:
                - torch.Tensor: Edge index tensor of shape (2, E), where E = num_targets * 3.
                                Each column represents an edge from a mesh node to a target node.
                - torch.Tensor: Edge attribute tensor containing distance weights.
        """
        # Convert lat/lon coordinates to 3D Cartesian coordinates for trimesh
        target_positions_3d = self._lat_lon_to_cartesian(target_latlon_rad)

        # Use the trimesh library to find the closest face for each target point
        mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        _, distances, query_face_indices = trimesh.proximity.closest_point(
            mesh_trimesh, target_positions_3d
        )

        # Get the 3 vertex indices for each of the found faces
        # Shape: [num_targets, 3]
        mesh_vertex_indices = mesh.faces[query_face_indices]

        # Create target indices, repeating each target_idx 3 times
        # Shape: [num_targets, 3]
        target_indices = np.repeat(np.arange(target_positions_3d.shape[0]), 3).reshape(
            -1, 3
        )

        # Create the edge list and attributes
        # Edges go from the mesh vertex (source) to the target point (destination)
        edge_list = np.stack(
            [mesh_vertex_indices.flatten(), target_indices.flatten()], axis=0
        )

        # Edge attributes are the distances to each of the 3 vertices
        edge_attr = np.repeat(distances, 3)

        return (
            torch.tensor(edge_list, dtype=torch.long),
            torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1),
        )

    def _lat_lon_to_cartesian(self, lat_lon_rad: np.ndarray) -> np.ndarray:
        """Converts latitude and longitude (in radians) to 3D Cartesian coordinates."""
        lat, lon = lat_lon_rad[:, 0], lat_lon_rad[:, 1]
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.stack([x, y, z], axis=1)


class MeshTargetKNNConnector:
    """
    Computes K-Nearest Neighbors (KNN)-based edges for decoding
    (connecting hidden mesh nodes to target data nodes).

    Attributes:
        num_nearest_neighbours (int): Number of nearest neighbors to consider for each target node.

    Methods:
        add_edges(mesh_graph, target_latlon_rad, mesh_latlon_rad):
            Connects each target observation location to its K-nearest mesh nodes.

        create_edge_index(edge_list, edge_weights):
            Converts edge list to PyTorch Geometric `edge_index` format with associated edge weights.
    """

    def __init__(self, num_nearest_neighbours: int):
        """
        Initializes the MeshTargetKNNConnector class.

        Parameters:
            num_nearest_neighbours (int): Number of nearest neighbors to connect each target node to.

        Raises:
            AssertionError: If `num_nearest_neighbours` is not a positive integer.
        """
        assert isinstance(
            num_nearest_neighbours, int
        ), "num_nearest_neighbours must be an integer"
        assert num_nearest_neighbours > 0, "num_nearest_neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def add_edges(self, mesh_graph, target_latlon_rad, mesh_latlon_rad):
        """
        Connects each target observation node to its K-nearest mesh nodes using the Haversine metric.

        Parameters:
            mesh_graph (networkx.Graph): The mesh graph containing node connectivity.
            target_latlon_rad (numpy.ndarray): Array of shape (M, 2) containing target node
                                               coordinates (latitude, longitude in radians).
            mesh_latlon_rad (numpy.ndarray): Array of shape (N, 2) containing mesh node
                                             coordinates (latitude, longitude in radians).

        Returns:
            tuple:
                - torch.Tensor: Edge index tensor of shape (2, E), where each column represents
                                an edge from a mesh node to a target node.
                - torch.Tensor: Edge attribute tensor containing distance weights for each edge.
        """
        knn = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbours, metric="haversine"
        )
        knn.fit(mesh_latlon_rad)

        distances, indices = knn.kneighbors(target_latlon_rad)

        edge_list = []
        edge_weights = []

        # Create directed edges from mesh nodes to target nodes
        for target_idx, mesh_neighbors in enumerate(indices):
            for i, neighbor in enumerate(mesh_neighbors):
                edge_list.append([neighbor, target_idx])  # Mesh node → Target node
                edge_feat = [distances[target_idx, i]]
                edge_weights.append(edge_feat)

        return self.create_edge_index(edge_list, edge_weights)

    def create_edge_index(self, edge_list, edge_weights):
        """
        Converts an edge list to PyTorch Geometric `edge_index` format with edge attributes.

        Parameters:
            edge_list (list): List of edges, where each entry is a pair [source, target].
            edge_weights (list): List of distance weights corresponding to the edges.

        Returns:
            tuple:
                - torch.Tensor: Edge index tensor (2, E) in COO format.
                - torch.Tensor: Edge attribute tensor containing distance weights.
        """
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(
            edge_weights, dtype=torch.float32
        )  # shape: [N_edges, edge_dim]
        return edge_index, edge_attr
