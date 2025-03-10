### March 2025
### Azadeh Gholoubi
# End to End Graph Neural Network for Direct Observation Prediction
## Overview
`EndToEndGNN.py` contains a Graph Neural Network (GNN) implementation designed to process observation data using PyTorch Geometric. The project leverages an icosahedral mesh representation for spatial structuring, encoding  observations, multi-scale edge connections,
and K-nearest neighbor decoding.

## Installation
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scipy torch trimesh networkx torch-geometric scikit-learn zarr joblib
```
## Usage
### 1. Load Satellite Data
Modify the start_date, end_date, and selected_satelliteId parameters to specify the required time range and satellite ID.
```python
start_date = "2024-04-01"
end_date = "2024-04-07"
selected_satelliteId = 224

z = zarr.open("/path/to/.zarr", mode="r")
data_summary = organize_bins_times(z, start_date, end_date, selected_satelliteId)
data_summary = extract_features(z, data_summary)
```
### 2. Generate Graph Structure
Create an icosahedral mesh and map observation data to it:
```python
hetero_data, mesh_graph, mesh_latlon_rad, mesh_coords = create_icosahedral_mesh(resolution=2)
```
Define encoder, processor, and decoder graphs:
```python
cutoff_encoder = CutOffEdges(cutoff_factor=0.6)
adj_matrix_encoding = cutoff_encoder.add_edges(encoder_graph, data_summary['bin1']['input_features_final'][:,-2:], mesh_latlon_rad)

multi_scale_processor = MultiScaleEdges(source_name="hidden", target_name="hidden", x_hops=3)
hetero_data, mesh_graph = multi_scale_processor.update_graph(hetero_data, mesh_graph)

knn_decoder = KNNEdges(num_nearest_neighbours=3)
edge_index_knn, edge_attr_knn = knn_decoder.add_edges(mesh_graph, data_summary['bin1']['target_features_final'][:,-2:], mesh_latlon_rad)
```
### 3. Train the GNN Model
```python
input_dim = 27
hidden_dim = 128
output_dim = 24
num_layers = 16

gnn_model = GNNModel(input_dim, hidden_dim, output_dim, num_layers)

trained_model = train_model(gnn_model, hidden_data, hetero_data["hidden", "to", "target"].edge_index, hidden_data.y, epochs=10, lr=1e-4)
```
### Model Architecture
The Graph Neural Network (GNN) consists of:

1. Encoder: GCNConv layer to project input features into hidden representations.
2. Processor: GATConv layers for multi-scale message passing across the icosahedral mesh.
3. Decoder: GCNConv layer to map hidden representations back to predicted target values.
