import psutil
import os
import time
import zarr
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import torch
import trimesh
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import joblib
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch_geometric.data import HeteroData, Data, DataLoader
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


def timing_resource_decorator(func):
    """
    A decorator that tracks execution time, memory usage, disk usage, and CPU usage
    before and after function execution.
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        # Record system resource usage BEFORE execution
        mem_before = process.memory_info().rss / (1024**3)  # GB
        disk_before = psutil.disk_usage('/').used / (1024**3)
        cpu_before = psutil.cpu_percent(interval=None)  # Snapshot before execution

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Record system resource usage AFTER execution
        mem_after = process.memory_info().rss / (1024**3)
        disk_after = psutil.disk_usage('/').used / (1024**3)
        cpu_after = psutil.cpu_percent(interval=None)  # Snapshot after execution

        execution_time = end_time - start_time

        # Print execution time and resource usage
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        print(f"  Memory Usage: {mem_before:.2f} GB → {mem_after:.2f} GB")
        print(f"  CPU Usage: {cpu_before:.2f}% → {cpu_after:.2f}%")
        print(f"  Disk Usage: {disk_before:.2f} GB → {disk_after:.2f} GB\n")
        return result

    return wrapper


@timing_resource_decorator
def organize_bins_times(z, start_date, end_date, selected_satelliteId):
    """
    Organizes satellite observation times into 12-hour bins and creates input-target pairs
    for time-series prediction.

    - Reads satellite observation times and filters data for a specific week (April 1-7, 2024)
      and a selected satellite (ID 224).
    - Groups observations into 12-hour time bins.
    - Creates a mapping of input and target time indices for each bin, forming sequential
      input-target pairs for model training.

    Returns:
        dict: A dictionary where each key represents a time bin (e.g., 'bin1', 'bin2') and
              contains input-target time indices and corresponding timestamps.
    """
    # Read time and convert to pandas datetime
    time = pd.to_datetime(z["time"][:], unit="s")
    satellite_ids = z["satelliteId"][:]

    # Select data based on the given time range and satellite ID
    selected_times = np.where((time >= start_date) & (time <= end_date) & (satellite_ids == selected_satelliteId))[0]

    # Filter data for the specified week and satellite
    df = pd.DataFrame({"time": time[selected_times], "zar_time": z["time"][selected_times]})
    df["index"] = np.where((time >= start_date) & (time <= end_date) & (satellite_ids == selected_satelliteId))[0]
    df["time_bin"] = df["time"].dt.floor("12h")

    # Sort by time
    df = df.sort_values(by="zar_time")
    data_summary = {}

    # Iterate over the time bins and shift them to form input-target pairs
    unique_bins = df["time_bin"].unique()
    print("Unique time bins:", df["time_bin"].unique())

    for i in range(len(unique_bins) - 1):  # Exclude last bin (no target)
        input_times = df[df["time_bin"] == unique_bins[i]]["zar_time"].values
        target_times = df[df["time_bin"] == unique_bins[i + 1]]["zar_time"].values
        data_summary[f"bin{i+1}"] = {
            'input_time': unique_bins[i],
            'target_time': unique_bins[i+1],
            'input_time_index': input_times,
            'target_time_index': target_times
        }

    return data_summary


@timing_resource_decorator
def extract_features(z, data_summary):
    """
    Extracts and normalizes input and target features for each time bin in the dataset.

    Parameters:
        z (zarr.Group): The Zarr dataset containing satellite observation data.
        data_summary (dict): Dictionary containing time bins and corresponding input/target time indices.

    Returns:
        dict: Updated data_summary with additional keys:
            - 'input_features_final': Normalized input features including latitude, longitude, and sensor measurements.
            - 'target_features_final': Normalized target features including latitude, longitude, and brightness temperatures.

    Notes:
        - Uses MinMax scaling for normalization.
        - Adds latitude and longitude (converted to radians) to both input and target features.
    """

    # Initialize scalers
    minmax_scaler_input = MinMaxScaler()
    minmax_scaler_target = MinMaxScaler()

    # Extract all necessary data at once (reduces repeated Zarr indexing)
    all_times = z["time"][:]
    latitude_rad = np.radians(z["latitude"][:])[:, None]  # Convert once, reshape for stacking
    longitude_rad = np.radians(z["longitude"][:])[:, None]

    # Extract all sensor angles at once (batch indexing)
    sensor_zenith = z["sensorZenithAngle"][:]
    solar_zenith = z["solarZenithAngle"][:]
    solar_azimuth = z["solarAzimuthAngle"][:]

    # Extract all 22 BT channels efficiently
    bt_channels = np.stack([z[f"bt_channel_{i}"][:] for i in range(1, 23)], axis=1)

    for bin_name in data_summary.keys():  # Process all bins
        # Find indices for input and target times
        input_mask = np.isin(all_times, data_summary[bin_name]['input_time_index'])
        target_mask = np.isin(all_times, data_summary[bin_name]['target_time_index'])

        # Prepare input features (batch extraction, avoid repeated indexing)
        input_features_orig = np.column_stack([
            sensor_zenith[input_mask],
            solar_zenith[input_mask],
            solar_azimuth[input_mask],
            bt_channels[input_mask]
        ])
        input_features_normalized = minmax_scaler_input.fit_transform(input_features_orig)

        input_features_final = np.hstack([
            latitude_rad[input_mask],
            longitude_rad[input_mask],
            input_features_normalized
        ])

        # Prepare target features
        target_features_orig = bt_channels[target_mask]
        target_features_normalized = minmax_scaler_target.fit_transform(target_features_orig)

        target_features_final = np.hstack([
            latitude_rad[target_mask],
            longitude_rad[target_mask],
            target_features_normalized
        ])

        # Convert to tensors at the end
        data_summary[bin_name]['input_features_final'] = torch.tensor(input_features_final, dtype=torch.float32)
        data_summary[bin_name]['target_features_final'] = torch.tensor(target_features_final, dtype=torch.float32)

        # Store min/max values for later unnormalization
        data_summary[bin_name]['target_scaler_min'] = minmax_scaler_target.data_min_
        data_summary[bin_name]['target_scaler_max'] = minmax_scaler_target.data_max_
    return data_summary


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

    @timing_resource_decorator
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


class GNNLightning(pl.LightningModule):
    """
    A Graph Neural Network (GNN) model for processing structured spatial data.

    The model consists of:
    - An encoder that maps input data nodes to a hidden representation.
    - A processor with multiple GATConv layers for message passing between hidden nodes.
    - A decoder that maps hidden node embeddings back to target node predictions.

    Attributes: input_dim (int): Dimension of the input node features.
                hidden_dim (int): Dimension of the hidden layer representations.
                output_dim (int): Dimension of the output predictions.
                num_layers (int): Number of message-passing layers (default: 16).
                encoder (GCNConv): Graph convolution layer for initial feature encoding.
                processor_layers (nn.ModuleList): List of GATConv layers for message passing.
                decoder (GCNConv): Graph convolution layer for decoding hidden node features to output space.

    Methods:
        forward(data):
            Defines the forward pass of the model, applying the encoder, processor, and decoder.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16, lr=1e-4):
        """
        Initializes the GNN model.

        Parameters:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of hidden dimensions in the graph layers.
            output_dim (int): Number of output features per node.
            num_layers (int, optional): Number of GATConv message-passing layers (default: 16).
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Encoder: Maps data nodes to hidden nodes
        self.encoder = GCNConv(input_dim, hidden_dim)

        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder = GCNConv(hidden_dim, output_dim)

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Hidden node connections

        # Encoding: Input → Hidden
        x_hidden = self.encoder(x, edge_index)
        x_hidden = F.relu(x_hidden)

        # Message Passing: Hidden ↔ Hidden
        for layer in self.processor_layers:
            x_hidden = layer(x_hidden, edge_index)
            x_hidden = F.relu(x_hidden)

        # Decoding: Hidden → Target (Use Target Edge Index)
        target_indices = torch.unique(data.edge_index_target[1])  # Get target node indices
        x_out = self.decoder(x_hidden, data.edge_index_target)  # Decode using target edges
        x_out = x_out[target_indices]  # Extract only target predictions

        return x_out

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_true = batch.y[:y_pred.shape[0], :]
        loss = self.loss_fn(y_pred, y_true)
        self.log("train_loss", loss)  # prog_bar=True)
        return {'loss': loss}

    # def validation_step(self, batch, batch_idx):
    #     y_pred = self(batch)
    #     y_true = batch.y  # batch.y[:y_hat.shape[0], :]
    #     loss = self.loss_fn(y_pred, y_true)
    #     self.log("val_loss", loss)
    #     return {'val_loss': loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class GNNDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path,
            start_date,
            end_date,
            satellite_id,
            batch_size=1,
            mesh_resolution=2,
            cutoff_factor=0.6,
            num_neighbors=3,
            num_hops=3
    ):
        super().__init__()
        # Store parameters
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.satellite_id = satellite_id
        self.batch_size = batch_size

        # Graph parameters
        self.mesh_resolution = mesh_resolution
        self.cutoff_factor = cutoff_factor
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops

        # Will be set in setup()
        self.data_summary = None
        self.train_data = None
        self.val_data = None
        self.z = None

    def prepare_data(self):
        """
        Check if Zarr dataset exists.
        """
        try:
            zarr.open(self.data_path, mode="r")
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr dataset at {self.data_path}: {e}")

    def setup(self, stage=None):
        """
        Prepare data for training/validation.
        """
        # Open Zarr dataset
        self.z = zarr.open(self.data_path, mode="r")

        # Process time bins and features
        self.data_summary = organize_bins_times(
            self.z, self.start_date, self.end_date, self.satellite_id
        )
        self.data_summary = extract_features(self.z, self.data_summary)

        # Create graph structure for first bin (can be extended to handle multiple bins)
        hetero_data = self._create_graph_structure(self.data_summary['bin1'])

        # # Generate icosahedral mesh
        # hetero_data, mesh_graph, mesh_latlon_rad, mesh_coords = create_icosahedral_mesh(resolution=2)

        # Split data into train/val
        total_samples = len(hetero_data["hidden"].x)
        train_size = int(0.8 * total_samples)

        # Create train/val splits
        self.train_data = self._create_data_object(hetero_data, slice(0, total_samples))
        # self.val_data = self._create_data_object(hetero_data, slice(train_size, total_samples))
        # TODO make validation work

    def _create_graph_structure(self, bin_data):
        """
        Create the graph structure for the model.
        """
        # Create icosahedral mesh
        hetero_data, mesh_graph, mesh_latlon_rad, _ = create_icosahedral_mesh(
            resolution=self.mesh_resolution
        )

        # Get input and target coordinates
        obs_latlon_rad = bin_data['input_features_final'][:, -2:]
        target_latlon_rad = bin_data['target_features_final'][:, -2:]

        # Create encoder edges
        encoder_graph = nx.DiGraph()
        cutoff_encoder = CutOffEdges(cutoff_factor=self.cutoff_factor)
        adj_matrix_encoding = cutoff_encoder.add_edges(
            encoder_graph, obs_latlon_rad, mesh_latlon_rad
        )

        # Create processor edges
        multi_scale_processor = MultiScaleEdges(
            source_name="hidden",
            target_name="hidden",
            x_hops=self.num_hops
        )
        hetero_data, mesh_graph = multi_scale_processor.update_graph(hetero_data, mesh_graph)

        # Generate adjacency matrix of processor graph
        adj_matrix = multi_scale_processor.get_adjacency_matrix(mesh_graph)

        # Convert adjacency matrix to PyG edge index
        edge_index = torch.tensor(np.vstack([adj_matrix.row, adj_matrix.col]), dtype=torch.long)

        # Assign multi-scale edges to the HeteroData object
        hetero_data["hidden", "to", "hidden"].edge_index = edge_index
        mesh_graph = mesh_graph.to_undirected()

        # Create decoder edges
        knn_decoder = KNNEdges(num_nearest_neighbours=self.num_neighbors)
        edge_index_knn, edge_attr_knn = knn_decoder.add_edges(
            mesh_graph, target_latlon_rad, mesh_latlon_rad
        )

        # Assign to HeteroData
        hetero_data["hidden"].x = bin_data['input_features_final']
        hetero_data["hidden", "to", "target"].edge_index = edge_index_knn
        hetero_data["hidden", "to", "target"].edge_attr = edge_attr_knn
        hetero_data["hidden"].y = bin_data['target_features_final']
        return hetero_data

    def _create_data_object(self, hetero_data, idx_slice):
        """
        Create a PyG Data object from HeteroData for the given indices.
        """
        return Data(
            x=hetero_data["hidden"].x[idx_slice],
            edge_index=hetero_data["hidden", "to", "hidden"].edge_index,
            edge_index_target=hetero_data["hidden", "to", "target"].edge_index,
            y=hetero_data["hidden"].y[idx_slice]  # Using same features as target for now
        )

    def train_dataloader(self):
        return DataLoader([self.train_data], batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader([self.val_data], batch_size=self.batch_size)


@timing_resource_decorator
def main():
    # Data parameters
    data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/atms.zarr"
    start_date = "2024-04-01"
    end_date = "2024-04-07"
    satellite_id = 224

    mesh_resolution = 2

    # Define model parameters
    input_dim = 27
    hidden_dim = 256
    output_dim = 24
    num_layers = 16
    lr = 1e-4

    # Training parameters
    max_epochs = 100
    batch_size = 1

    # Instantiate model & data module
    model = GNNLightning(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        lr=lr
    )

    data_module = GNNDataModule(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        satellite_id=satellite_id,
        batch_size=batch_size,
        mesh_resolution=mesh_resolution
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='gnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # Train with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        # callbacks=[checkpoint_callback, early_stopping],  #TODO make this work
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
