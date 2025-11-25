import networkx as nx
import numpy as np
import torch
from networkx import Graph
from scipy.sparse import coo_matrix
from torch_geometric.data import HeteroData


class MeshSelfConnectivity:
    """
    Handles intra-mesh connectivity by extracting processor edges from a mesh graph.

    Attributes:
        source_name (str): Node type in the graph (e.g., "hidden").
        target_name (str): Must match `source_name`.

    Methods:
        get_adjacency_matrix(mesh_graph):
            Converts the mesh graph to a sparse adjacency matrix.

        update_graph(graph, mesh_graph):
            Updates the PyTorch Geometric HeteroData object with edge indices from the mesh graph.
    """

    VALID_NODES = ["hidden"]

    def __init__(self, source_name: str, target_name: str, relation: str = "to"):
        """
        Initializes the MeshSelfConnectivity class.

        Parameters:
            source_name (str): Node type in the graph.
            target_name (str): Must match `source_name`.

        Raises:
            AssertionError: If source and target names don't match.
        """
        assert (
            source_name in self.VALID_NODES
        ), f"Invalid source_name: {source_name}. Must be one of: {self.VALID_NODES}"
        assert (
            target_name in self.VALID_NODES
        ), f"Invalid target_name: {target_name}. Must be one of: {self.VALID_NODES}"
        assert (
            source_name == target_name
        ), f"{self.__class__.__name__} requires source and target names to be the same."

        self.source_name = source_name
        self.target_name = target_name
        self.relation = relation

    def get_adjacency_matrix(self, mesh_graph: Graph) -> coo_matrix:
        """
        Converts the updated mesh graph into a sparse adjacency matrix.

        Parameters:
            mesh_graph (networkx.Graph): The mesh graph with added multi-scale edges.

        Returns:
            scipy.sparse.coo_matrix: The adjacency matrix in COO format.
        """
        adj_matrix = nx.to_scipy_sparse_array(mesh_graph, format="coo")
        return adj_matrix

    def update_graph(
        self, graph: HeteroData, mesh_graph: Graph, print_once=False
    ) -> tuple[HeteroData, Graph]:
        """
        Updates the graph with intra-mesh edges based on existing mesh connectivity.

        Parameters:
            graph (HeteroData): The PyTorch Geometric heterogeneous data object.
            mesh_graph (networkx.Graph): The mesh graph.

        Returns:
            tuple:
                - HeteroData: Updated PyG graph with intra-mesh edges.
                - nx.Graph: The original mesh graph (unchanged).
        """
        assert self.source_name in graph, f"{self.source_name} is missing in graph."

        # Compute edge index from adjacency
        adj_matrix = self.get_adjacency_matrix(mesh_graph)
        edge_index = torch.tensor(
            np.vstack([adj_matrix.row, adj_matrix.col]), dtype=torch.long
        )

        if print_once:
            print(
                f"Added {edge_index.shape[1]} intra-mesh edges to graph: {self.source_name} -> {self.target_name}"
            )

        # Assign edges to graph
        graph[self.source_name, self.relation, self.target_name].edge_index = edge_index

        return graph, mesh_graph
