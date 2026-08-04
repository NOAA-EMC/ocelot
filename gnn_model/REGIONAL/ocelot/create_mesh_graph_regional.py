"""
Regional (limited-area) mesh creation and obs-mesh connectivity.

Mirrors create_mesh_graph_global.py but works in 2D Cartesian (projected) space
instead of spherical lat/lon, which is appropriate for limited-area domains
where a map projection (e.g. Lambert Conformal) is nearly flat.

Main API:
    create_regional_mesh(xy, args)
        Build multi-resolution 2D mesh from projected grid coordinates.
        Returns a dict with keys: m2m_graphs, mesh_pos, G_bottom_mesh,
        all_mesh_nodes.

    obs_mesh_conn_regional(obs_xy, G_bottom_mesh, args, o2m=True)
        Build obs-to-mesh (o2m=True) or mesh-to-obs (o2m=False) edges.
        Returns (edge_index, edge_attr) where edge_attr is shape [E, 4]:
            [dist/dm, dx/dm, dy/dm, 0]  (zero-pads missing z-component)

    project_coords(lat, lon)
        Project lat/lon degrees to RRFS 15 km Lambert Conformal x/y [m].
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
import scipy.spatial
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from ocelot.utils import DEFAULT_DTYPE
from ocelot.logger import logger


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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mk_2d_graph(xy, nx_pts, ny_pts):
    """Create a directed 2D grid graph with diagonal edges in Cartesian space."""
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    dx = (xM - xm) / nx_pts
    dy = (yM - ym) / ny_pts
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx_pts)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny_pts)

    mg = np.meshgrid(lx, ly)
    g = nx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    g.add_edges_from([((x, y), (x +
                                1, y +
                                1)) for x in range(nx_pts -
                                                   1) for y in range(ny_pts -
                      1)] +
                     [((x +
                        1, y), (x, y +
                                1)) for x in range(nx_pts -
                                                   1) for y in range(ny_pts -
                                                                     1)])

    dg = nx.DiGraph(g)
    for u, v in g.edges():
        d = float(
            np.sqrt(
                np.sum(
                    (g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2)))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


def _prepend_node_index(graph, level_index):
    ijk = [tuple((level_index,) + x) for x in graph.nodes]
    return nx.relabel_nodes(graph, dict(zip(graph.nodes, ijk)), copy=True)


def _sort_nodes_internally(graph):
    H = nx.DiGraph()
    H.add_nodes_from(sorted(graph.nodes(data=True)))
    H.add_edges_from(graph.edges(data=True))
    return H


def _from_networkx_offset(nx_graph, start_index):
    pyg = from_networkx(nx_graph)
    # Compute edge_attr using 0-based local pos BEFORE offsetting edge_index.
    vdiff = pyg.pos[pyg.edge_index[1]] - pyg.pos[pyg.edge_index[0]]
    lengths = torch.linalg.vector_norm(vdiff, dim=1, keepdim=True)
    lengths_norm = lengths / (lengths.mean() + 1e-8)
    pyg.edge_attr = torch.cat(
        (lengths_norm, vdiff / (lengths + 1e-8)), dim=1).to(DEFAULT_DTYPE)
    pyg.edge_index = pyg.edge_index + start_index
    return pyg


def _add_edge_attr(pyg_graph):
    """Compute edge_attr = [length, vdiff_x, vdiff_y] for a 0-indexed PyG graph."""
    vdiff = pyg_graph.pos[pyg_graph.edge_index[1]] - \
        pyg_graph.pos[pyg_graph.edge_index[0]]
    lengths = torch.linalg.vector_norm(vdiff, dim=1, keepdim=True)
    lengths_norm = lengths / (lengths.mean() + 1e-8)
    pyg_graph.edge_attr = torch.cat(
        (lengths_norm, vdiff / (lengths + 1e-8)), dim=1).to(DEFAULT_DTYPE)
    return pyg_graph


def _mesh_spacing(G_nx):
    """Return minimum edge length in networkx mesh graph."""
    lens = [d["len"] for _, _, d in G_nx.edges(data=True) if "len" in d]
    return float(min(lens)) if lens else 1.0


# ---------------------------------------------------------------------------
# Flat mesh
# ---------------------------------------------------------------------------

def _create_flat_mesh(G, nx_stride, graph_dir_path, args):
    G_tot = G[0]
    for lev in range(1, len(G)):
        nodes = list(G[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::nx_stride, 1::nx_stride, :]
            .reshape(int(n / nx_stride) ** 2, 2)
        )
        G[lev] = nx.relabel_nodes(G[lev], dict(
            zip(G[lev].nodes, [tuple(x) for x in ij])))
        G_tot = nx.compose(G_tot, G[lev])

    G_tot = _prepend_node_index(G_tot, 0)
    G_int = nx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted")

    pyg_m2m = from_networkx(G_int)
    pyg_m2m = _add_edge_attr(pyg_m2m)

    m2m_graphs = [pyg_m2m]
    mesh_pos = [pyg_m2m.pos.to(DEFAULT_DTYPE)]

    if args.plot:
        plot_graph(pyg_m2m, title="Mesh-to-mesh")
        plt.show()

    return {
        "m2m_graphs": m2m_graphs,
        "mesh_pos": mesh_pos,
        "G_bottom_mesh": G_int,
        "all_mesh_nodes": G_int.nodes(data=True),
    }


# ---------------------------------------------------------------------------
# Hierarchical mesh
# ---------------------------------------------------------------------------

def _create_level_connections(G_from, G_to, start_index):
    G_down = G_from.copy()
    G_down.clear_edges()
    G_down = nx.DiGraph(G_down)
    G_down.add_nodes_from(G_to.nodes(data=True))

    v_to_list = list(G_to.nodes)
    v_from_list = list(G_from.nodes)
    v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
    kdt = scipy.spatial.KDTree(v_from_xy)

    for v in v_to_list:
        neigh_idx = kdt.query(G_down.nodes[v]["pos"], 1)[1]
        u = v_from_list[neigh_idx]
        d = float(
            np.sqrt(
                np.sum(
                    (G_down.nodes[u]["pos"] -
                     G_down.nodes[v]["pos"]) ** 2)))
        G_down.add_edge(
            u,
            v,
            len=d,
            vdiff=G_down.nodes[u]["pos"] -
            G_down.nodes[v]["pos"])

    G_down_int = nx.convert_node_labels_to_integers(
        G_down, first_label=start_index, ordering="sorted")
    G_down_int = _sort_nodes_internally(G_down_int)
    return _from_networkx_offset(G_down_int, start_index)


def _create_hierarchical_mesh(G, mesh_levels, graph_dir_path, args):
    G = [_prepend_node_index(graph, level_i)
         for level_i, graph in enumerate(G)]

    num_nodes_level = np.array([len(g.nodes) for g in G])
    first_index_level = np.concatenate(
        (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
    )

    up_graphs, down_graphs = [], []
    for G_from, G_to, start_index in zip(
        G[1:], G[:-1], first_index_level[: mesh_levels - 1]
    ):
        G_down = _create_level_connections(G_from, G_to, start_index)
        G_up = G_down.clone()
        G_up.edge_index = torch.stack(
            (G_down.edge_index[1], G_down.edge_index[0]), dim=0)
        up_graphs.append(G_up)
        down_graphs.append(G_down)
        if args.plot:
            plot_graph(
                G_up,
                title=f"Down graph, {first_index_level} -> {first_index_level + 1}")
            plt.show()

    m2m_graphs = [
        _from_networkx_offset(
            nx.convert_node_labels_to_integers(g, first_label=si, ordering="sorted"),
            si,
        )
        for g, si in zip(G, first_index_level)
    ]

    mesh_pos = [graph.pos.to(DEFAULT_DTYPE) for graph in m2m_graphs]

    joint = nx.union_all(list(G))
    return {
        "m2m_graphs": m2m_graphs,
        "mesh_pos": mesh_pos,
        "G_bottom_mesh": G[0],
        "all_mesh_nodes": joint.nodes(data=True),
        "mesh_up_graphs": up_graphs,
        "mesh_down_graphs": down_graphs,
    }


# ---------------------------------------------------------------------------
# Public mesh creation
# ---------------------------------------------------------------------------

def create_regional_mesh(xy, args):
    """
    Build a multi-resolution 2D Cartesian mesh from projected grid coordinates.

    Args:
        xy: 2-element array/list [X_2d, Y_2d] where X_2d[i, j] and Y_2d[i, j]
            are the projected x/y coordinates at grid point (i, j).
        args: Namespace with:
            levels (int|None): max mesh levels (None = use all)
            hierarchical (bool): build hierarchical mesh
            plot (bool): whether to plot

    Returns dict with keys:
        m2m_graphs      list[Data]  PyG graphs per level
        mesh_pos        list[Tensor]  node positions per level
        G_bottom_mesh   networkx DiGraph  finest-level mesh (for obs connectivity)
        all_mesh_nodes  networkx NodeView
        mesh_up_graphs  list[Data]  (hierarchical only)
        mesh_down_graphs list[Data] (hierarchical only)
    """
    NX = 3  # branching factor
    nlev = int(np.log(max(xy[0].shape)) / np.log(NX))
    nleaf = NX ** nlev

    mesh_levels = nlev - 1
    levels_arg = getattr(args, "levels", None)
    if levels_arg is not None:
        mesh_levels = min(mesh_levels, int(levels_arg))
    logger.info(
        f"[regional mesh] nlev={nlev}, nleaf={nleaf}, mesh_levels={mesh_levels}")

    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (NX ** lev))
        G.append(_mk_2d_graph(xy, n, n))
        if args.plot:
            plot_graph(from_networkx(G[-1]), title=f"Mesh graph, level {lev}")
            plt.show()

    hierarchical = bool(getattr(args, "hierarchical", False))
    if hierarchical:
        return _create_hierarchical_mesh(G, mesh_levels, None, args)
    else:
        return _create_flat_mesh(G, NX, None, args)


# ---------------------------------------------------------------------------
# Convenience entry point: domain corners instead of a full meshgrid
# ---------------------------------------------------------------------------

def create_regional_mesh_from_corners(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        args,
        mesh_levels=None):
    """
    Build a regional mesh from domain corner coordinates in degrees.

    Args:
        lon_min, lon_max: longitude extent (degrees)
        lat_min, lat_max: latitude extent (degrees)
        args: same Namespace as create_regional_mesh
        mesh_levels: desired number of mesh levels (overrides args.levels).
            The grid shape is set to 3**mesh_levels so the full hierarchy fits.

    Returns: same dict as create_regional_mesh.
    """
    NX = 3
    levels = mesh_levels if mesh_levels is not None else getattr(
        args, "mesh_splits", 2)
    if levels is None:
        levels = 2
    # Shape must be >= 3^(levels+1) so nlev >= levels+1 and mesh_levels >=
    # levels
    n = NX ** (levels + 1)
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)
    LON, LAT = np.meshgrid(lons, lats)
    xy_proj = project_coords(LAT.ravel(), LON.ravel())
    X = xy_proj[:, 0].reshape(LON.shape)
    Y = xy_proj[:, 1].reshape(LON.shape)
    return create_regional_mesh([X, Y], args)


# ---------------------------------------------------------------------------
# Obs-mesh connectivity
# ---------------------------------------------------------------------------

def obs_mesh_conn_regional(obs_xy, G_bottom_mesh, args, o2m=True):
    """
    Build obs-to-mesh (o2m=True) or mesh-to-obs (o2m=False) edge tensors.

    Args:
        obs_xy: np.ndarray [N, 2] projected (x, y) coordinates of observations
        G_bottom_mesh: networkx DiGraph of the finest mesh level
        args: Namespace with cutoff_factor (float) and num_neighbors (int)
        o2m: if True build obs→mesh edges; if False build mesh→obs edges

    Returns:
        edge_index: LongTensor [2, E]
        edge_attr:  FloatTensor [E, 4]  = [dist/dm, dx/dm, dy/dm, 0]
            The 4th feature is zero (no z-component in 2D Cartesian).
    """
    cutoff_factor = float(getattr(args, "cutoff_factor", 0.67))
    num_neighbors = int(getattr(args, "num_neighbors", 4))

    # Mesh nodes and their positions
    vm = G_bottom_mesh.nodes
    vm_list = list(vm)
    vm_xy = np.array([vm[n]["pos"] for n in vm_list])  # [M, 2]
    dm = _mesh_spacing(G_bottom_mesh)

    # KD-tree on obs points
    kdt_obs = scipy.spatial.KDTree(obs_xy)

    edges = []
    edge_dist = []
    edge_vdiff = []

    for mesh_idx, n in enumerate(vm_list):
        v_pos = vm[n]["pos"]
        neigh_idxs = kdt_obs.query_ball_point(v_pos, dm * cutoff_factor)
        if not neigh_idxs:
            _, indices = kdt_obs.query(v_pos, k=num_neighbors)
            neigh_idxs = [
                int(indices)] if num_neighbors == 1 else indices.tolist()

        for obs_idx in neigh_idxs:
            obs_pos = obs_xy[obs_idx]
            diff = v_pos - obs_pos        # vector from obs to mesh
            d = float(np.linalg.norm(diff))
            edges.append((obs_idx, mesh_idx))
            edge_dist.append(d)
            edge_vdiff.append(diff)

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=DEFAULT_DTYPE)
        return edge_index, edge_attr

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    dist_t = torch.tensor(edge_dist, dtype=DEFAULT_DTYPE)
    vdiff_t = torch.tensor(edge_vdiff, dtype=DEFAULT_DTYPE)          # [E, 2]

    # Normalise by mesh spacing; zero-pad to 4 features (global API parity)
    zeros = torch.zeros(dist_t.shape[0], 1, dtype=DEFAULT_DTYPE)
    edge_attr = torch.cat(
        [(dist_t / dm).unsqueeze(1), vdiff_t / dm, zeros], dim=1
    )  # [E, 4]

    if not o2m:
        # Flip to mesh→obs
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Coordinate projection  (RRFS 15 km Lambert Conformal)
# ---------------------------------------------------------------------------

def project_coords(lat, lon):
    """
    Project lat/lon (degrees) to RRFS 15 km Lambert Conformal x/y (metres).

    Returns np.ndarray [N, 2] of (x, y) in the LCC projection.
    """
    from pyproj import CRS, Transformer

    lcc_crs = CRS.from_proj4(
        "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 "
        "+lon_0=-97.5 +ellps=WGS84 +x_0=0 +y_0=0"
    )
    transformer = Transformer.from_crs("EPSG:4326", lcc_crs, always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack((x, y))
