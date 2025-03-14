import networkx as nx
import numpy as np
import time
import torch
import trimesh
from torch_geometric.data import HeteroData, Data


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper


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

    
@timing_decorator
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

    # Step 1: Generate an icosahedral mesh with given resolution
    sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
    mesh_coords = sphere.vertices  # Cartesian coordinates
    mesh_faces = sphere.faces  # Triangular faces

    # step 2: convert Cartesian coordinates to latitude/longitude (radians)
    mesh_latlon_rad = cartesian_to_latlon_rad(mesh_coords)

    # Step 3: Create a directed NetworkX graph
    mesh_graph = nx.DiGraph()

    # Add nodes with lat/lon positions
    for i, coord in enumerate(mesh_latlon_rad):
        mesh_graph.add_node(i, pos=tuple(coord))  # Store lat/lon as node attributes
    
    # Add edges based on triangular faces
    for face in mesh_faces:
        edges = [(face[0], face[1]), (face[1], face[0]),
                 (face[1], face[2]), (face[2], face[1]),
                 (face[2], face[0]), (face[0], face[2])]
        mesh_graph.add_edges_from(edges)

    # Step 4: Convert to PyTorch Geometric HeteroData format
    hetero_data = HeteroData()
    node_coords = torch.tensor(mesh_latlon_rad, dtype=torch.float32)
    hetero_data["hidden"] = Data(x=node_coords)  # Store node features (lat/lon)

    # Convert edges from NetworkX graph to PyG format
    edge_index = torch.tensor(list(mesh_graph.edges), dtype=torch.long).t().contiguous()
    hetero_data["hidden", "to", "hidden"].edge_index = edge_index

    return hetero_data, mesh_graph, mesh_latlon_rad, mesh_coords 
