import networkx as nx
import numpy as np
import torch
import trimesh
from torch_geometric.data import Data, HeteroData

from timing_utils import timing_resource_decorator


def cartesian_to_latlon_rad(cartesian_coords):
    """
    Converts 3D Cartesian coordinates to latitude and longitude in radians.

    Parameters:
        cartesian_coords (numpy.ndarray): Array of shape (N, 3) representing (x, y, z) coordinates.

    Returns:
        numpy.ndarray: Array of shape (N, 2) containing latitude and longitude in radians.
    """
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    lat = np.arcsin(z)  # Latitude in radians
    lon = np.arctan2(y, x)  # Longitude in radians
    return np.column_stack((lat, lon))


@timing_resource_decorator
def create_icosahedral_mesh(resolution=2):
    """
    Generates an icosahedral mesh, converts its nodes from Cartesian coordinates to latitude/longitude,
    and returns it as a PyTorch Geometric HeteroData graph.

    Parameters:
        resolution (int, optional): The number of subdivisions to refine the icosahedral mesh.
                                    Higher values increase node density. Default is 2.

    Returns:
        tuple:
            - HeteroData: PyTorch Geometric HeteroData object containing the graph representation.
            - networkx.DiGraph: The directed graph representation of the mesh.
            - numpy.ndarray: Array of mesh node coordinates in (latitude, longitude) radians.
            - numpy.ndarray: Array of mesh node coordinates in Cartesian (x, y, z) format.
    """
    mesh_graph = nx.DiGraph()

    #  Generate an icosahedral mesh with highest given resolution
    finest_sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
    finest_mesh_coords = finest_sphere.vertices
    finest_mesh_latlon_rad = cartesian_to_latlon_rad(finest_mesh_coords)

    for i, coord in enumerate(finest_mesh_latlon_rad):
        mesh_graph.add_node(i, pos=tuple(coord))  # Store lat/lon as node attributes

    # Stats tracking
    stats = {
        "resolution": resolution,
        "finest_nodes": len(finest_mesh_coords),
        "finest_faces": len(finest_sphere.faces),
        "multilevel_edges": 0,
        "edge_counts_per_level": {},
    }

    # Collect all edges from resolution 0 to current resolution
    all_edges = set()
    for reso in range(0, resolution + 1):
        this_sphere = trimesh.creation.icosphere(subdivisions=reso, radius=1.0)
        this_mesh_faces = this_sphere.faces  # Triangular faces

        # Add edges based on triangular faces
        edge_set = set()
        for face in this_mesh_faces:
            edge_set.update(
                [
                    (face[0], face[1]),
                    (face[1], face[0]),
                    (face[1], face[2]),
                    (face[2], face[1]),
                    (face[2], face[0]),
                    (face[0], face[2]),
                ]
            )

        # Store edge count for this level
        stats["edge_counts_per_level"][f"resolution_{reso}"] = len(edge_set)
        all_edges.update(edge_set)

    # Keep only edges where both nodes exist in the finest resolution
    max_node_id = len(finest_mesh_coords)
    valid_edges = [(i, j) for (i, j) in all_edges if i < max_node_id and j < max_node_id]
    stats["multilevel_edges"] = len(valid_edges)

    mesh_graph.add_edges_from(valid_edges)

    # Convert to PyTorch Geometric HeteroData format
    hetero_data = HeteroData()
    node_coords = torch.tensor(finest_mesh_latlon_rad, dtype=torch.float32)
    hetero_data["hidden"] = Data(x=node_coords)  # Store node features (lat/lon)

    # Convert edges from NetworkX graph to PyG format
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    hetero_data["hidden", "to", "hidden"].edge_index = edge_index

    return hetero_data, mesh_graph, finest_mesh_latlon_rad, stats
