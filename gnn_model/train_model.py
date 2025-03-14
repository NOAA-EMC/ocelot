import numpy as np
import torch.nn as nn
import networkx as nx
import torch
import zarr
import torch.optim as optim
from torch_geometric.data import HeteroData, Data

from create_edges import CutOffEdges, MultiScaleEdges, KNNEdges
from create_features import extract_features, organize_bins_times
from graph_models import GNNModel
from utils import timing_decorator, create_icosahedral_mesh


@timing_decorator
def train_model(model, data, target_edge_index, target_y, epochs=10, lr=0.001):
    """
    Trains a Graph Neural Network (GNN) model using Mean Squared Error (MSE) loss.

    Parameters:
        model (nn.Module): The GNN model to be trained.
        data (torch_geometric.data.Data): Input graph data containing:
            - x (torch.Tensor): Node feature matrix.
            - edge_index (torch.Tensor): Edge connectivity matrix.
        target_edge_index (torch.Tensor): Edge index tensor connecting hidden nodes to target nodes.
        target_y (torch.Tensor): Ground truth values for target nodes.
        epochs (int, optional): Number of training epochs (default: 10).
        lr (float, optional): Learning rate for the optimizer (default: 0.001).

    Returns:
        nn.Module: The trained GNN model.

    Notes:
        - Uses Adam optimizer and MSE loss.
        - Selects only unique target nodes for training.
        - Performs a forward pass, computes loss, and updates gradients in each epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data)  

        # Select only **unique** target nodes
        # target_indices = torch.unique(target_edge_index[1])  # Get unique target node indices
        predicted_target = output  # Select corresponding predictions
        
        # Keep lat/lon separate for evaluation
        target_latlon = target_y[:predicted_target.shape[0], :2]
        
        # Ensure `target_y` only contains relevant rows
        target_y = target_y[:predicted_target.shape[0], :]  

        # Compute loss
        loss = criterion(predicted_target, target_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model


@timing_decorator
def main():
    ############################################################################################
    # Define parameters
    start_date = "2024-04-01"
    end_date = "2024-04-07"
    selected_satellite_id = 224
    
    # Open Zarr dataset
    z = zarr.open("/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/atms.zarr", mode="r")
    
    # make a pair of input-target data for each time step
    data_summary = organize_bins_times(z, start_date, end_date, selected_satellite_id)
    data_summary = extract_features(z, data_summary)
    
    # for now, we only train for first time step--bin1 which its data are available in data_summary['bin1']
    
    # Generates an icosahedral mesh
    hetero_data, mesh_graph, mesh_latlon_rad, mesh_coords = create_icosahedral_mesh(resolution=2)
    print("\n--- Generated Mesh Info ---")
    print(f"Total Nodes: {len(mesh_latlon_rad)}")  # Number of nodes
    print(f"Total Edges: {mesh_graph.number_of_edges()}")  # Number of edges
    print(f"Edge Index Shape: {hetero_data['hidden', 'to', 'hidden'].edge_index.shape}")  # PyG edge index shape
    print("\n--- HeteroData Structure ---")
    print(hetero_data)
    
    # now start making graphs using above icosahedral mesh
    encoder_graph = nx.DiGraph()  # Ensure only directed (No bidirectional edges) connections for encoder graph
    
    # Initialize CutOffEdges
    cutoff_encoder = CutOffEdges(cutoff_factor=0.6)
    # get lat and lon from input features
    obs_latlon_rad = data_summary['bin1']['input_features_final'][:, -2:]
    # Apply encoding (mapping observations to hidden mesh) 
    adj_matrix_encoding = cutoff_encoder.add_edges(encoder_graph, obs_latlon_rad, mesh_latlon_rad)
    
    # For processor graph, apply multiscale edges with x_hops=3 (3 number of hops for connectivity between nodes)
    multi_scale_processor = MultiScaleEdges(source_name="hidden", target_name="hidden", x_hops=3)
    
    # Pass the updated mesh_graph-- update processor graph
    hetero_data, mesh_graph = multi_scale_processor.update_graph(hetero_data, mesh_graph)
    
    # Generate adjacency matrix of processor graph
    adj_matrix = multi_scale_processor.get_adjacency_matrix(mesh_graph)
    
    # Convert adjacency matrix to PyG edge index
    edge_index = torch.tensor(np.vstack([adj_matrix.row, adj_matrix.col]), dtype=torch.long)
    
    # Assign multiscale edges to the HeteroData object
    hetero_data["hidden", "to", "hidden"].edge_index = edge_index
    mesh_graph = mesh_graph.to_undirected()
    
    # Extract lat/lon of target needd for decoder graph
    target_latlon_rad = data_summary['bin1']['target_features_final'][:, -2:]
    
    # Initialize KNNEdges for decoding
    knn_decoder = KNNEdges(num_nearest_neighbours=3)
    
    # add decoder edges from mesh to target node
    edge_index_knn, edge_attr_knn = knn_decoder.add_edges(mesh_graph, target_latlon_rad, mesh_latlon_rad)
    
    # Assign to HeteroData for PyTorch Geometric
    hetero_data["hidden", "to", "target"].edge_index = edge_index_knn
    hetero_data["hidden", "to", "target"].edge_attr = edge_attr_knn
    
    # here we are done defining our graphs and can not start to train the model using pytorch
    
    # Define pytorch model parameters
    input_dim = 27
    hidden_dim = 128  # Reduced hidden dimension to avoid memory issues
    output_dim = 24 
    num_layers = 16
    
    # Instantiate the model
    gnn_model = GNNModel(input_dim, hidden_dim, output_dim, num_layers)
    stacked_x = data_summary['bin1']['input_features_final']
    stacked_y = data_summary['bin1']['target_features_final']
    
    # Assign to HeteroData
    hetero_data["hidden"].x = stacked_x  # Use the structured tensor
    hetero_data["hidden", "to", "hidden"].edge_index = edge_index.to(torch.long)  # Processor edges
    hetero_data["hidden", "to", "target"].edge_index = edge_index_knn.to(torch.long)  # Decoder edges
    
    # Convert to PyG Data object using stacked_x
    hidden_data = Data(
        x=stacked_x,  
        edge_index=hetero_data["hidden", "to", "hidden"].edge_index,
        edge_index_target=hetero_data["hidden", "to", "target"].edge_index, 
        y=stacked_y   
    )
    
    # Assign filtered target edges & ground truth to `hidden_data`
    edge_index_target = hetero_data["hidden", "to", "target"].edge_index 

    # Train Model
    trained_model = train_model(gnn_model, hidden_data, edge_index_target, hidden_data.y, epochs=10, lr=1e-4)


if __name__ == "__main__":
    main()
