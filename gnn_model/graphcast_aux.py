import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_utils as gc_mu
import torch_geometric as pyg
import torch

import icosahedral_mesh

GC_SPATIAL_FEATURES_KWARGS = {
    "add_node_positions": False,
    "add_node_latitude": True,
    "add_node_longitude": True,
    "add_relative_positions": True,
    "relative_longitude_local_coordinates": True,
    "relative_latitude_local_coordinates": True,
}


def vertice_cart_to_lat_lon(vertices):
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


def plot_graph(edge_index, pos_lat_lon, title=None):
    """
    Plot flattened global graph
    """
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H

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


def inter_mesh_connection(from_mesh, to_mesh):
    """
    Connect from_mesh to to_mesh
    """
    kd_tree = scipy.spatial.cKDTree(to_mesh.vertices)

    # Each node on lower (from) mesh will connect to 1 or 2 on level above
    # pylint: disable-next=protected-access
    radius = 1.1 * gc_gc._get_max_edge_distance(from_mesh)
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


def _get_max_edge_distance(mesh):
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(
        mesh.vertices[senders] - mesh.vertices[receivers], axis=-1
    )
    return edge_distances.max()
