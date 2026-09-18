# Standard library
from pyproj import Proj, CRS, Transformer
import os
from argparse import ArgumentParser

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import networkx as netwx
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# Local
from .utils import DEFAULT_DTYPE


def plot_graph(graph, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )
    edge_index = edge_index.cpu().numpy()
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

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = netwx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def save_edges(graph, name, base_path):
    torch.save(
        graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt")
    )
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        DEFAULT_DTYPE
    )
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            DEFAULT_DTYPE
        )
        for graph in graphs
    ]

    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def create_boundary_mask(G, coords):
    """
    Create a mask for mesh nodes where boundary nodes are 0 and interior nodes are 1.

    Args:
        G: NetworkX graph with node positions
        coords: Array of shape [N, 2] containing [x, y] coordinates
               or dictionary with {'x': x_array, 'y': y_array}

    Returns:
        torch.Tensor: Binary mask where 0 indicates boundary nodes and 1 indicates interior nodes
    """
    if isinstance(coords, dict):
        x_min, x_max = coords['x'].min(), coords['x'].max()
        y_min, y_max = coords['y'].min(), coords['y'].max()
    else:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Add small epsilon to handle floating point comparisons
    eps = 1e-6
    x_min -= eps
    x_max += eps
    y_min -= eps
    y_max += eps

    # Create mask tensor
    mask = torch.ones(len(G.nodes))

    # Identify boundary nodes
    for node in G.nodes:
        pos = G.nodes[node]['pos']
        if (
            abs(pos[0] - x_min) < eps or
            abs(pos[0] - x_max) < eps or
            abs(pos[1] - y_min) < eps or
            abs(pos[1] - y_max) < eps
        ):
            mask[node] = 0.0

    return mask


def mk_2d_graph(xy, nx, ny):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = netwx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    # add diagonal edges
    g.add_edges_from([((x, y), (x +
                                1, y +
                                1)) for x in range(nx -
                                                   1) for y in range(ny -
                      1)] +
                     [((x +
                        1, y), (x, y +
                                1)) for x in range(nx -
                                                   1) for y in range(ny -
                                                                     1)])

    # turn into directed graph
    dg = netwx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


# def prepend_node_index(graph, new_index):
#     # Relabel node indices in graph, insert (graph_level, i, j)
#     ijk = [tuple((new_index,) + tuple(graph.nodes[n]['pos'])) for n in graph.nodes]
#     to_mapping = dict(zip(graph.nodes, ijk))
#     return netwx.relabel_nodes(graph, to_mapping, copy=True)

def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return netwx.relabel_nodes(graph, to_mapping, copy=True)


def prepend_node_index_int(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + (x,)) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return netwx.relabel_nodes(graph, to_mapping, copy=True)


def create_mesh_structure(xy, args, graph_dir_path):
    """
    Create multi-resolution mesh structure with optional hierarchical organization.

    Args:
        xy: Grid coordinates array
        args: Arguments containing:
            - levels: Maximum number of mesh levels
            - hierarchical: Whether to create hierarchical mesh
            - plot: Whether to plot graphs
        graph_dir_path: Path to save graph data

    Returns:
        dict: Contains mesh graphs, positions, and related data
    """
    import networkx as netwx
    import numpy as np
    import torch
    from scipy import spatial

    # Graph geometry parameters
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    # Determine mesh levels
    mesh_levels = nlev - 1
    if args.levels:
        mesh_levels = min(mesh_levels, args.levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # Create multi-resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n)
        if args.plot:
            plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
            plt.show()
        G.append(g)

    if args.hierarchical:
        return _create_hierarchical_mesh(G, mesh_levels, graph_dir_path, args)
    else:
        return _create_flat_mesh(G, nx, graph_dir_path, args)


def _create_hierarchical_mesh(G, mesh_levels, graph_dir_path, args):
    """Create hierarchical mesh structure with inter-level connections."""
    # Relabel nodes with level index
    G = [prepend_node_index(graph, level_i)
         for level_i, graph in enumerate(G)]

    # Calculate level indices
    num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
    first_index_level = np.concatenate(
        (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
    )

    # Create inter-level connections
    up_graphs, down_graphs = _create_interlevel_connections(
        G, mesh_levels, first_index_level, args
    )

    # Save up and down edges
    save_edges_list(up_graphs, "mesh_up", graph_dir_path)
    save_edges_list(down_graphs, "mesh_down", graph_dir_path)

    # Create m2m graphs
    m2m_graphs = [
        from_networkx_with_start_index(
            netwx.convert_node_labels_to_integers(
                level_graph, first_label=start_index, ordering="sorted"
            ),
            start_index,
        )
        for level_graph, start_index in zip(G, first_index_level)
    ]

    # Add edge attributes to each mesh graph
    for graph in m2m_graphs:
        vdiff = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
        lengths = torch.linalg.vector_norm(vdiff, dim=1).unsqueeze(-1)
        graph.edge_attr = torch.cat((lengths, vdiff), dim=1).float()

    mesh_pos = [graph.pos.to(DEFAULT_DTYPE) for graph in m2m_graphs]

    # Create combined mesh structure
    G_bottom_mesh = G[0]

    joint_mesh_graph = netwx.union_all([graph for graph in G])
    return {
        'm2m_graphs': m2m_graphs,
        'mesh_pos': mesh_pos,
        'G_bottom_mesh': G_bottom_mesh,
        'all_mesh_nodes': joint_mesh_graph.nodes(data=True)
    }


def _create_flat_mesh(G, nx, graph_dir_path, args):
    """Create flat mesh structure combining all levels."""
    G_tot = G[0]

    # Combine all levels
    for lev in range(1, len(G)):
        nodes = list(G[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::nx, 1::nx, :]
            .reshape(int(n / nx) ** 2, 2)
        )
        ij = [tuple(x) for x in ij]
        G[lev] = netwx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
        G_tot = netwx.compose(G_tot, G[lev])

    # Relabel and convert to integers
    G_tot = prepend_node_index(G_tot, 0)
    G_int = netwx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted"
    )

    # Create PyG graph
    pyg_m2m = from_networkx(G_int)

    # Add edge attributes
    vdiff = pyg_m2m.pos[pyg_m2m.edge_index[1]] - \
        pyg_m2m.pos[pyg_m2m.edge_index[0]]
    lengths = torch.linalg.vector_norm(vdiff, dim=1).unsqueeze(-1)
    pyg_m2m.edge_attr = torch.cat((lengths, vdiff), dim=1).float()

    m2m_graphs = [pyg_m2m]
    mesh_pos = [pyg_m2m.pos.to(DEFAULT_DTYPE)]

    if args.plot:
        plot_graph(pyg_m2m, title="Mesh-to-mesh")
        plt.show()

    return {
        'm2m_graphs': m2m_graphs,
        'mesh_pos': mesh_pos,
        'G_bottom_mesh': G_int,
        'all_mesh_nodes': G_int.nodes(data=True)
    }


def _create_level_connections(G_from, G_to, start_index):
    # start out from graph at from level
    G_down = G_from.copy()
    G_down.clear_edges()
    G_down = netwx.DiGraph(G_down)

    # Add nodes of to level
    G_down.add_nodes_from(G_to.nodes(data=True))

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    v_to_list = list(G_to.nodes)
    v_from_list = list(G_from.nodes)
    v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
    kdt_m = scipy.spatial.KDTree(v_from_xy)

    # add edges from mesh to grid
    for v in v_to_list:
        # find 1(?) nearest neighbours (index to vm_xy)
        neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
        u = v_from_list[neigh_idx]

        # add edge from mesh to grid
        G_down.add_edge(u, v)
        d = np.sqrt(
            np.sum(
                (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
            )
        )
        G_down.edges[u, v]["len"] = d
        G_down.edges[u, v]["vdiff"] = (
            G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
        )

    # relabel nodes to integers (sorted)
    G_down_int = netwx.convert_node_labels_to_integers(
        G_down, first_label=start_index, ordering="sorted"
    )  # Issue with sorting here
    G_down_int = sort_nodes_internally(G_down_int)
    pyg_down = from_networkx_with_start_index(G_down_int, start_index)
    return pyg_down


def _create_interlevel_connections(G, mesh_levels, first_index_level, args):
    """Create connections between different mesh levels."""
    up_graphs = []
    down_graphs = []

    for from_level, to_level, G_from, G_to, start_index in zip(
        range(1, mesh_levels),
        range(0, mesh_levels - 1),
        G[1:],
        G[:-1],
        first_index_level[: mesh_levels - 1],
    ):
        # Create downward connections
        G_down = _create_level_connections(
            G_from, G_to, start_index
        )

        # Create upward connections by inverting downward edges
        up_edges = torch.stack(
            (G_down.edge_index[1], G_down.edge_index[0]), dim=0
        )
        pyg_up = G_down.clone()
        pyg_up.edge_index = up_edges

        up_graphs.append(pyg_up)
        down_graphs.append(G_down)

        if args.plot:
            plot_graph(
                pyg_up, title=f"Down graph, {from_level} -> {to_level}"
            )
            plt.show()

    return up_graphs, down_graphs


def print_pos(vm):
    """Calculate distance between mesh nodes."""
    # print(vm.data('pos'))
    pos_data = dict(vm.data("pos"))
    print("Available keys in pos_data:", list(pos_data.keys()))
    return


def create_obs_conn_mesh(coords, G_bottom_mesh, args, conn='g2m'):
    """
    Create Grid-to-Mesh (g2m) or Mesh-to-Grid (m2g) graph structure for heterogeneous observations.

    Args:
        coords: Array of shape [N, 2] containing [x, y] coordinates
               or dictionary with {'x': x_array, 'y': y_array}
               or dictionary with observation types as keys and coordinates as values
        G_bottom_mesh: Bottom level mesh graph
        args: Arguments containing:
            - cutoff: Radius scale for grid-mesh association (default: 0.67)
            - num_neighbors: Number of neighbors for KNN fallback
            - plot: Whether to plot the graph
            - obs_type: Optional, observation type for type-specific parameters
            - include_boundary_mask: Optional, whether to include boundary mask (default: False)
        conn: Connection type, either 'g2m' (grid-to-mesh) or 'm2g' (mesh-to-grid)

    Returns:
        dict: Contains:
            - graph: PyTorch Geometric graph for grid-to-mesh or mesh-to-grid
            - grid_graph: Base grid graph
            - mesh_distance: Distance between mesh nodes
            - edge_weights: Optional weights for heterogeneous observations
            - boundary_mask: Optional tensor marking boundary (0) and interior (1) nodes
    """
    import networkx as netwx
    import numpy as np
    from scipy.spatial import KDTree

    # Constants
    DM_SCALE = 0.67  # radius scale for grid-mesh association

    def _euclidean_distance(p1, p2):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def _calculate_mesh_distance(vm):
        """Calculate distance between mesh nodes."""
        # print(vm.data('pos'))
        # pos_data = dict(vm.data("pos"))
        # print("Available keys in pos_data:", list(pos_data.keys()))
        # pos1 = vm.data("pos")[(0, 1, 0)]
        # pos2 = vm.data("pos")[(0, 0, 0)]
        pos1 = vm.data("pos")[1]
        pos2 = vm.data("pos")[0]
        return _euclidean_distance(pos1, pos2)

    def _get_coordinates(coords):
        """Get coordinates from input format."""
        if isinstance(coords, dict):
            x, y = coords['x'].flatten(), coords['y'].flatten()
            return np.column_stack((x, y))
        return coords

    def _create_base_grid(points):
        """Create base grid graph from coordinates."""
        G_grid = netwx.Graph()
        for i, pos in enumerate(points):
            G_grid.add_node(i, pos=pos)
        return G_grid

    try:
        args_dict = vars(args)
        # 1. Get mesh nodes and their positions
        vm = G_bottom_mesh.nodes
        vm_xy = np.array([pos for _, pos in vm.data("pos")])
        dm = _calculate_mesh_distance(vm)

        # 2. Get grid points and create grid
        grid_points = _get_coordinates(coords)
        G_grid = _create_base_grid(grid_points)

        # 3. Build KD-tree for grid points
        vg_list = list(G_grid.nodes)
        vg_coords = np.array([G_grid.nodes[n]['pos'] for n in vg_list])
        print(
            f"[DEBUG] vg_coords shape: {vg_coords.shape}, dtype: {vg_coords.dtype}, n_nodes: {len(vg_list)}")
        kdt_g = KDTree(vg_coords)

        # 4. Create edge connections between grid and mesh
        grid_to_mesh_edges = []
        edge_weights = []
        edge_vdiffs = []

        # Process each mesh node
        for mesh_idx, v in enumerate(vm):
            v_pos = vm[v]["pos"]

            # Try radius-based neighbors first
            neigh_idxs = kdt_g.query_ball_point(v_pos, dm * args.cutoff_factor)

            # Fallback to KNN if no neighbors found
            if not neigh_idxs:
                k = min(args.num_neighbors, len(vg_coords))
                if k == 0:
                    continue
                distances, indices = kdt_g.query(v_pos, k=k)
                indices = np.atleast_1d(indices)
                # scipy pads with n_points when k > n; filter those out
                neigh_idxs = [int(idx)
                              for idx in indices if idx < len(vg_coords)]

            for i in neigh_idxs:
                grid_idx = i  # Grid indices already start from 0

                # Add connection
                grid_to_mesh_edges.append((grid_idx, mesh_idx))

                # Calculate edge properties
                grid_pos = G_grid.nodes[i]["pos"]
                d = _euclidean_distance(grid_pos, v_pos)
                edge_weights.append(d)
                edge_vdiffs.append(v_pos - grid_pos)

        # Convert to PyTorch tensors
        edge_index = torch.tensor(
            grid_to_mesh_edges,
            dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        edge_vdiffs = torch.tensor(edge_vdiffs, dtype=torch.float)

        # Create PyG graph with separate grid and mesh nodes
        pyg_g2m = Data(
            grid_pos=torch.tensor([G_grid.nodes[n]['pos'] for n in vg_list], dtype=torch.float),
            mesh_pos=torch.tensor([vm[n]['pos'] for n in vm], dtype=torch.float),
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_vdiffs=edge_vdiffs,
            num_grid_nodes=len(vg_list),
            num_mesh_nodes=len(vm)
        )

        # Create the appropriate graph based on connection type
        if conn == 'g2m':
            # Grid-to-mesh: Use edges as is
            graph = pyg_g2m
        elif conn == 'm2g':
            # Mesh-to-grid: Flip the edge indices
            edge_index = pyg_g2m.edge_index
            m2g_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

            # Create new graph with flipped edges but keep other attributes
            graph = Data(
                edge_index=m2g_edge_index,
                grid_pos=pyg_g2m.grid_pos,
                mesh_pos=pyg_g2m.mesh_pos,
                edge_weights=pyg_g2m.edge_weights,
                edge_vdiffs=pyg_g2m.edge_vdiffs
            )
        else:
            raise ValueError(
                f"Unknown connection type: {conn}. Must be 'g2m' or 'm2g'")

        # Create result dictionary
        result = {
            'graph': graph,
            'grid_graph': G_grid,
            'mesh_distance': dm,
        }

        # Add boundary mask if requested
        if args_dict.get('include_boundary_mask', False):
            boundary_mask = create_boundary_mask(G_bottom_mesh, coords)
            result['boundary_mask'] = boundary_mask
            # Add mask to PyG graph as well
            pyg_g2m.boundary_mask = boundary_mask

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to create grid-to-mesh graph: {e}")

# Example usage in another function


def some_function(coords, proj_params=None):
    if proj_params is not None:
        # Setup projection if needed
        proj = setup_lambert_projection(proj_params)
        # Project coordinates
        projected_coords = project_coordinates(coords, proj)
    else:
        # Use coordinates as is
        projected_coords = coords


def create_transformer():
    # projection:
    #   class: LambertConformal
    #   kwargs:
    #     central_longitude: -97.5  # Converted from LoV 262.5 degrees as 360 - 262.5 = 97.5
    #     central_latitude: 38.5  # Directly from LatD (Latitude of origin)
    #     standard_parallels: [38.5, 38.5]  # From Latin1 and Latin2

    # Lambert Conformal Conic projection parameters
    kwargs = {
        "proj": "lcc",
        "central_longitude": -97.5,
        "central_latitude": 38.5,
        "standard_parallels": [38.5, 38.5],
        "ellps": "WGS84",
        "x_0": 0,
        "y_0": 0,
    }

    # Create CRS and Transformer
    lcc_crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={kwargs['standard_parallels'][0]} "
        f"+lat_2={kwargs['standard_parallels'][1]} "
        f"+lat_0={kwargs['central_latitude']} "
        f"+lon_0={kwargs['central_longitude']} "
        f"+ellps={kwargs['ellps']} +x_0={kwargs['x_0']} +y_0={kwargs['y_0']}")

    transformer = Transformer.from_crs("EPSG:4326", lcc_crs, always_xy=True)
    return transformer


def project_coords(lat, lon):
    transformer = create_transformer()
    lon_deg = lon
    lat_deg = lat
    lon_lcc, lat_lcc = transformer.transform(lon_deg, lat_deg)
    coords = np.column_stack((lon_lcc, lat_lcc))
    return coords


def setup_lambert_projection(params=None):
    """
    Setup Lambert Conformal projection with default or custom parameters.

    Args:
        params: Optional dictionary containing Lambert projection parameters:
                - lat_1: First standard parallel (default: 30.0)
                - lat_2: Second standard parallel (default: 60.0)
                - lat_0: Latitude of origin (default: 40.0)
                - lon_0: Central meridian (default: -97.0)
                - earth_radius: Earth radius in meters (default: 6371000)

    Returns:
        pyproj.Proj: Configured Lambert Conformal projection object
    """
    import pyproj

    default_params = {
        'lat_1': 30.0,
        'lat_2': 60.0,
        'lat_0': 40.0,
        'lon_0': -97.0,
        'earth_radius': 6371000
    }

    if params is not None:
        default_params.update(params)

    return pyproj.Proj(
        proj='lcc',
        lat_1=default_params['lat_1'],
        lat_2=default_params['lat_2'],
        lat_0=default_params['lat_0'],
        lon_0=default_params['lon_0'],
        R=default_params['earth_radius']
    )


def project_coordinates(coords, proj):
    """
    Project latitude/longitude coordinates to Lambert projection space.

    Args:
        coords: Array of shape [N, 2] containing [lat, lon] pairs
               or dictionary with {'lat': lat_array, 'lon': lon_array}
        proj: pyproj.Proj object for Lambert projection

    Returns:
        numpy.ndarray: Array of shape [N, 2] containing projected [x, y] coordinates
    """
    if isinstance(coords, dict):
        lats, lons = coords['lat'].flatten(), coords['lon'].flatten()
    else:
        lats, lons = coords[:, 0], coords[:, 1]

    x, y = proj(lons, lats)
    return np.column_stack((x, y))


def check_nan(coords):
    if np.isnan(coords).any() or np.isinf(coords).any():
        print("coords contains NaN or Inf")


if __name__ == "__main__":
    main()
