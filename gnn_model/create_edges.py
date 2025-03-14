import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import networkx as nx
from torch_geometric.data import HeteroData

from utils import timing_decorator


class CutOffEdges:
    """
    Computes cutoff-based edges to connect observation nodes to mesh nodes 
    based on a geodesic distance threshold.

    Attributes:
        cutoff_factor (float): Scaling factor to adjust the cutoff radius.
        radius (float or None): Computed cutoff radius based on the mesh node distances.

    Methods:
        get_cutoff_radius(mesh_latlon_rad):
            Computes the cutoff radius using the maximum geodesic neighbor distance.
        
        add_edges(graph, obs_latlon_rad, mesh_latlon_rad):
            Establishes directional edges from observations to mesh nodes based 
            on the computed cutoff radius.
    """

    def __init__(self, cutoff_factor: float):
        """
        Initializes the CutOffEdges class with a cutoff factor.

        Parameters:
            cutoff_factor (float): Scaling factor for determining the cutoff radius.
        """
        self.cutoff_factor = cutoff_factor
        self.radius = None
    
    @timing_decorator
    def get_cutoff_radius(self, mesh_latlon_rad):
        """
        Computes the cutoff radius using the Haversine metric, based on the 
        maximum distance between mesh node neighbors.

        Parameters:
            mesh_latlon_rad (numpy.ndarray): Array of shape (N, 2) containing 
                                             mesh node coordinates (latitude, longitude in radians).

        Returns:
            float: The computed cutoff radius.
        """
        knn = NearestNeighbors(n_neighbors=2, metric="haversine")
        knn.fit(mesh_latlon_rad)
        dists, _ = knn.kneighbors(mesh_latlon_rad)
        self.radius = dists[dists > 0].max() * self.cutoff_factor  
        return self.radius

    def add_edges(self, graph, obs_latlon_rad, mesh_latlon_rad):
        """
        Adds edges from observation nodes to mesh nodes based on a cutoff radius.

        Parameters:
            graph (networkx.DiGraph): The directed graph where edges will be added.
            obs_latlon_rad (numpy.ndarray): Array of shape (M, 2) containing 
                                            observation node coordinates (lat, lon in radians).
            mesh_latlon_rad (numpy.ndarray): Array of shape (N, 2) containing 
                                             mesh node coordinates (lat, lon in radians).

        Returns:
            torch.Tensor: Edge index tensor in PyTorch Geometric COO format (2, E), 
                          where each column represents a directed edge (obs → mesh).
        """
        if self.radius is None:
            self.get_cutoff_radius(mesh_latlon_rad)

        knn = NearestNeighbors(metric="haversine")
        knn.fit(mesh_latlon_rad)

        # Find mesh nodes within the cutoff radius for each observation node
        distances, indices = knn.radius_neighbors(obs_latlon_rad, radius=self.radius)

        obs_to_mesh_edges = []

        for obs_idx, mesh_neighbors in enumerate(indices):
            for mesh_idx in mesh_neighbors:
                # Only allow directional edges: observation → mesh
                obs_to_mesh_edges.append([obs_idx, mesh_idx])

        # Convert edges to PyTorch tensor in COO format
        edge_index_obs_to_mesh = torch.tensor(obs_to_mesh_edges, dtype=torch.long).t().contiguous()

        # Use `DiGraph()` to enforce directional edges
        for obs_idx, mesh_idx in obs_to_mesh_edges:
            graph.add_edge(f"obs_{obs_idx}", f"mesh_{mesh_idx}")  # No reverse edges added

        return edge_index_obs_to_mesh  # PyTorch Geometric-compatible adjacency matrix


class MultiScaleEdges:
    """
    Defines multi-scale edges in the icosahedral mesh by connecting nodes that are `x_hops` apart.
    This helps in propagating information across different scales within the mesh.

    Attributes:
        source_name (str): The source node type in the graph (e.g., "hidden").
        target_name (str): The target node type in the graph (must be the same as `source_name`).
        x_hops (int): The number of hops to define connectivity between nodes.

    Methods:
        add_edges_from_mesh(mesh_graph):
            Connects nodes that are exactly `x_hops` apart while preserving the original edges for `x_hops=1`.
        
        get_adjacency_matrix(mesh_graph):
            Converts the updated mesh graph to a sparse adjacency matrix.
        
        update_graph(graph, mesh_graph):
            Updates the input graph with multi-scale edges and returns the modified PyG `HeteroData` object.
    """

    VALID_NODES = ["hidden"]  # Valid node types for this edge class

    def __init__(self, source_name: str, target_name: str, x_hops: int):
        """
        Initializes the MultiScaleEdges class for defining connectivity in the icosahedral mesh.

        Parameters:
            source_name (str): Name of the source node type (must match `target_name`).
            target_name (str): Name of the target node type (must match `source_name`).
            x_hops (int): Number of hops for defining multi-scale connectivity.
        
        Raises:
            AssertionError: If source and target names do not match or if `x_hops` is not a positive integer.
        """
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "x_hops must be an integer"
        assert x_hops > 0, "x_hops must be positive"

        self.source_name = source_name
        self.target_name = target_name
        self.x_hops = x_hops

    def add_edges_from_mesh(self, mesh_graph):
        """
        Adds edges between nodes that are exactly `x_hops` apart in the mesh graph.
        The original edges are preserved if `x_hops=1`.

        Parameters:
            mesh_graph (networkx.Graph): The input mesh graph.

        Returns:
            networkx.Graph: The updated mesh graph with new edges added for multi-scale connectivity.
        """
        new_edges = []
        existing_edges = mesh_graph.edges
        print(f"Before x_hops={self.x_hops}, edges: {mesh_graph.number_of_edges()}")  # Debugging
        
        for node in mesh_graph.nodes:
            # Get nodes that are `x_hops` away
            neighbors = nx.single_source_shortest_path_length(mesh_graph, node, cutoff=self.x_hops)
    
            for neighbor, hops in neighbors.items():
                # Preserve the original 960 edges for x_hops=1
                if self.x_hops == 1 and (node, neighbor) in existing_edges:
                    continue  # Do not modify the initial edges
    
                # Add only new edges for x_hops > 1
                if hops == self.x_hops and (node, neighbor) not in existing_edges:
                    new_edges.append((node, neighbor))
        print(f"Filtered new edges to add: {len(new_edges)}")  # Debug info
        mesh_graph.add_edges_from(new_edges)
        print(f"Final edge count in mesh_graph: {mesh_graph.number_of_edges()}")
        return mesh_graph

    def get_adjacency_matrix(self, mesh_graph):
        """
        Converts the updated mesh graph into a sparse adjacency matrix.

        Parameters:
            mesh_graph (networkx.Graph): The mesh graph with added multi-scale edges.

        Returns:
            scipy.sparse.coo_matrix: The adjacency matrix in COO format.
        """
        adj_matrix = nx.to_scipy_sparse_array(mesh_graph, format="coo")
        return adj_matrix

    def update_graph(self, graph: HeteroData, mesh_graph) -> tuple[HeteroData, nx.Graph]:
        """
        Updates the graph with multi-scale edges by computing adjacency and edge indices.

        Parameters:
            graph (torch_geometric.data.HeteroData): The PyTorch Geometric heterogeneous data object.
            mesh_graph (networkx.Graph): The input mesh graph.

        Returns:
            tuple:
                - HeteroData: The updated PyG heterogeneous data object with multi-scale edges.
                - networkx.Graph: The modified mesh graph with multi-scale connectivity.
        
        Raises:
            AssertionError: If `source_name` is missing in the `HeteroData` object.
        """
        assert self.source_name in graph, f"{self.source_name} is missing in graph."
    
        # Convert mesh to adjacency matrix
        graph[self.source_name]["_nx_graph"] = self.add_edges_from_mesh(mesh_graph)
    
        adj_matrix = self.get_adjacency_matrix(mesh_graph)
    
        # Convert adjacency matrix to PyG edge index
        edge_index = torch.tensor(np.vstack([adj_matrix.row, adj_matrix.col]), dtype=torch.long)
    
        # Assign edges to graph
        graph[self.source_name, "to", self.target_name].edge_index = edge_index
    
        return graph, mesh_graph


class KNNEdges:
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
        Initializes the KNNEdges class.

        Parameters:
            num_nearest_neighbours (int): Number of nearest neighbors to connect each target node to.

        Raises:
            AssertionError: If `num_nearest_neighbours` is not a positive integer.
        """
        assert isinstance(num_nearest_neighbours, int), "num_nearest_neighbours must be an integer"
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
        knn = NearestNeighbors(n_neighbors=self.num_nearest_neighbours, metric="haversine")
        knn.fit(mesh_latlon_rad)

        distances, indices = knn.kneighbors(target_latlon_rad)

        edge_list = []
        edge_weights = []

        # Create directed edges from mesh nodes to target nodes
        for target_idx, mesh_neighbors in enumerate(indices):
            for i, neighbor in enumerate(mesh_neighbors):
                edge_list.append([neighbor, target_idx])  # Mesh node → Target node
                edge_weights.append(distances[target_idx, i])  # Distance weight

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
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        return edge_index, edge_attr
