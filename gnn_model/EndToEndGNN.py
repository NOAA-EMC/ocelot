import psutil
import os
import time
import zarr
import numpy as np
import pandas as pd
import torch
import trimesh
from networkx import Graph
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch_geometric.data import HeteroData, Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
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
            edge_set.update([
                (face[0], face[1]), (face[1], face[0]),
                (face[1], face[2]), (face[2], face[1]),
                (face[2], face[0]), (face[0], face[2]),
            ])

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
        assert source_name in self.VALID_NODES, f"Invalid source_name: {source_name}. Must be one of: {self.VALID_NODES}"
        assert target_name in self.VALID_NODES, f"Invalid target_name: {target_name}. Must be one of: {self.VALID_NODES}"
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target names to be the same."



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

    def update_graph(self, graph: HeteroData, mesh_graph: Graph) -> tuple[HeteroData, Graph]:
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
        edge_index = torch.tensor(np.vstack([adj_matrix.row, adj_matrix.col]), dtype=torch.long)

        print(f"Added {edge_index.shape[1]} intra-mesh edges to graph: {self.source_name} -> {self.target_name}")


        # Assign edges to graph
        graph[self.source_name, self.relation, self.target_name].edge_index = edge_index


        return graph, mesh_graph


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
    A Graph Neural Network (GNN) model for processing structured spatiotemporal data.

    The model consists of:
    - An MLP-based encoder that embeds observation nodes (with edge attributes) into hidden mesh nodes.
    - A processor that applies multiple GATConv layers for message passing between mesh nodes.
    - An MLP-based decoder that maps hidden mesh node features to target nodes via KNN-based edges.

    Key Features:
    - Encoder and decoder use distance information (as edge attributes).
    - Decoder output is aggregated using inverse-distance weighted averaging.
    - Includes LayerNorm and Dropout in both encoder and decoder for regularization.

    Methods:
        forward(data):
            Runs the forward pass, including encoding, message passing, decoding, and
            weighted aggregation to produce target predictions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=16, lr=1e-4):
        """
        Initializes the GNNLightning model with an encoder, processor, and decoder.

        Parameters:
        input_dim (int): Number of input features per observation node (before encoding).
        hidden_dim (int): Size of the hidden representation used in all layers.
        output_dim (int): Number of features to predict at each target node.
        num_layers (int, optional): Number of GATConv layers in the processor block (default: 16).
        lr (float, optional): Learning rate for the optimizer (default: 1e-4).
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Encoder: Maps data nodes to hidden nodes
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Processor: Message passing layers (Hidden ↔ Hidden)
        self.processor_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder: Maps hidden nodes back to target nodes
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        x, edge_index_encoder, edge_attr_encoder, edge_index_processor, edge_index_decoder, y = (
        data.x,
        data.edge_index_encoder,
        data.edge_attr_encoder,
        data.edge_index_processor,
        data.edge_index_decoder,
        data.y
    )

        
        # === Encoding: obs → mesh ===
        src_encoder = edge_index_encoder[0]
        tgt_encoder = edge_index_encoder[1]
        obs_feats = x[src_encoder]
        edge_feats = data.edge_attr_encoder.unsqueeze(1)  # Shape [E, 1]
        encoder_input = torch.cat([obs_feats, edge_feats], dim=1)

        # Pass through MLP
        encoded = self.encoder_mlp(encoder_input)
        x_hidden = scatter_mean(encoded, tgt_encoder, dim=0, dim_size=x.shape[0])
        x_hidden = F.relu(x_hidden)

        # === Processor: mesh ↔ mesh ===
        for layer in self.processor_layers:
            x_hidden = layer(x_hidden, edge_index_processor)
            x_hidden = F.relu(x_hidden)

        # Decoding: Hidden → Target (MLP using mesh → target edges with edge_attr)
        src_decoder = edge_index_decoder[0]  # mesh node indices
        tgt_decoder = edge_index_decoder[1]  # target node indices
        mesh_feats = x_hidden[src_decoder]   # Features of mesh nodes sending messages
        dist_feats = data.edge_attr_decoder.unsqueeze(1)  # Haversine distance
        
        decoder_input = torch.cat([mesh_feats, dist_feats], dim=1)
        decoded = self.decoder_mlp(decoder_input)

        # === Weighted aggregation using inverse distance ===
        # Aggregate per target using scatter mean
        weights = 1.0 / (dist_feats + 1e-8)  # Avoid division by zero
        weighted = decoded * weights
        
        # Normalize by total weights per target
        norm_weights = scatter_add(weights, tgt_decoder, dim=0, dim_size=y.shape[0])
        x_out = scatter_add(weighted, tgt_decoder, dim=0, dim_size=y.shape[0])
        x_out = x_out / (norm_weights + 1e-8) 

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
        data_dict = self._create_graph_structure(self.data_summary['bin1'])


        # Split data into train/val
        total_samples = len(data_dict["x"])
        train_size = int(0.8 * total_samples)

        # Create train/val splits
        self.train_data = self._create_data_object(data_dict, slice(0, len(data_dict["x"])))
        # self.val_data = self._create_data_object(hetero_data, slice(train_size, total_samples))
        # TODO make validation work

    def _create_graph_structure(self, bin_data):
        """
        Create the graph structure for the model with global indexing and padded features.
        """
        input_features = bin_data['input_features_final']
        target_features = bin_data['target_features_final']
        obs_latlon_rad = input_features[:, -2:]
        target_latlon_rad = target_features[:, -2:]

        # Create icosahedral mesh
        hetero_data, mesh_graph, mesh_latlon_rad, stats  = create_icosahedral_mesh(
            resolution=self.mesh_resolution
        )

        # Print mesh statistics
        print("\n Mesh Statistics:")
        print(f"  - Mesh Resolution: {stats['resolution']}")
        print(f"  - Number of Nodes: {stats['finest_nodes']}")
        print(f"  - Number of Faces: {stats['finest_faces']}")
        print(f"  - Multilevel Edges: {stats['multilevel_edges']}")
        print("  - Edge Counts per Level:")
        for reso_level, count in stats["edge_counts_per_level"].items():
            print(f"    {reso_level}: {count}")
        print("")

        # === ENCODER EDGES ===
        cutoff_encoder = ObsMeshCutoffConnector(cutoff_factor=self.cutoff_factor)
        edge_index_encoder, edge_attr_encoder  = cutoff_encoder.add_edges(
            obs_latlon_rad, mesh_latlon_rad, return_edge_attr=True
        )

        if edge_index_encoder.numel() == 0:
            raise ValueError("No encoder edges were created. Try increasing cutoff_factor.")

        # === PROCESSOR EDGES ===
        multi_scale_processor = MeshSelfConnectivity(
            source_name="hidden",
            target_name="hidden",
        )
        hetero_data, mesh_graph = multi_scale_processor.update_graph(hetero_data, mesh_graph)

        mesh_graph = mesh_graph.to_undirected()

        # Get edge index for processor edges
        edge_index_processor = hetero_data["hidden", "to", "hidden"].edge_index
        
        # === DECODER EDGES ===
        knn_decoder = MeshTargetKNNConnector(num_nearest_neighbours=self.num_neighbors)
        edge_index_knn, edge_attr_knn = knn_decoder.add_edges(
            mesh_graph, target_latlon_rad, mesh_latlon_rad
        )

        # === GLOBAL INDEXING ===
        num_obs_nodes = input_features.shape[0]
        num_mesh_nodes = hetero_data["hidden"].x.shape[0]
        mesh_offset = num_obs_nodes

        # Pad mesh features to match input_dim
        mesh_feats = hetero_data["hidden"].x  # e.g., [40962, 2]
        input_dim = input_features.shape[1]
        pad_feats = torch.zeros((num_mesh_nodes, input_dim - mesh_feats.shape[1]))
        mesh_feats_padded = torch.cat([mesh_feats, pad_feats], dim=1)

        # Stack input and mesh features
        stacked_x = torch.cat([input_features, mesh_feats_padded], dim=0)

        # Update edge indices with global offsets
        edge_index_encoder_global = edge_index_encoder.clone()
        edge_index_encoder_global[1] += mesh_offset

        edge_index_processor_global = edge_index_processor + mesh_offset

        edge_index_decoder_global = edge_index_knn.clone()
        edge_index_decoder_global[0] += mesh_offset

        # Store everything in a dictionary (to be converted to Data later)
        data_dict = {
            'x': stacked_x,
            'edge_index_encoder': edge_index_encoder_global.to(torch.long),
            'edge_attr_encoder': edge_attr_encoder,
            'edge_index_processor': edge_index_processor_global.to(torch.long),
            'edge_index_decoder': edge_index_decoder_global.to(torch.long),
            'edge_attr_decoder': edge_attr_knn,
            'y': target_features
        }
        return data_dict

    def _create_data_object(self, data_dict, idx_slice):
        """
        Create a PyG Data object from the combined data_dict.
        """
        return Data(
            x=data_dict["x"],
            edge_index_encoder=data_dict["edge_index_encoder"],
            edge_attr_encoder=data_dict["edge_attr_encoder"],
            edge_index_processor=data_dict["edge_index_processor"],
            edge_index_decoder=data_dict["edge_index_decoder"],
            edge_attr_decoder=data_dict["edge_attr_decoder"],
            y=data_dict["y"]
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

    mesh_resolution = 6

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
