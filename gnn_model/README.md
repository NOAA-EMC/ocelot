### March 2025
### Azadeh Gholoubi
# End to End Graph Neural Network for Direct Observation Prediction
## Overview
This project implements a Graph Neural Network (GNN) pipeline to predict observations directly from raw input features. It uses an icosahedral mesh to structure spatial information, and builds edge connections via:

- Distance-based cutoff for obs → mesh

- Multi-scale mesh ↔ mesh

- KNN-based mesh → target

The pipeline uses PyTorch Lightning and PyTorch Geometric, with modular components for preprocessing, mesh generation, edge creation, model definition, and training.

## Modular Structure (Flat Files)
- `gnn_model.py`             GNNLightning model (encoder → processor → decoder)
- `gnn_datamodule.py`        Graph construction and PyTorch Lightning DataModule
- `train_gnn.py`             Main training script
- `obs_to_mesh.py`           Cutoff-based encoder edges
- `mesh_to_mesh.py`          Multi-scale mesh processor edges
- `mesh_to_target.py`        KNN-based decoder edges
- `mesh_creation.py`         Mesh and graph construction
- `process_timeseries.py`    Zarr binning + feature extraction
- `timing_utils.py`          Resource-logging decorator

## Installation
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scipy torch trimesh networkx torch-geometric scikit-learn zarr joblib lightning psutil

```
## Usage 
### 1. Load Data
Edit `train_gnn.py` to modify the start_date, end_date, and selected_satelliteId parameters to specify the required time range and satellite ID.
```python
start_date = "2024-04-01"
end_date = "2024-04-07"
z = zarr.open("/path/to/.zarr", mode="r")
```

### 3. Train the GNN Model
```bash
python train_gnn.py
```
### Model Architecture
The Graph Neural Network (GNN) consists of:

1. Encoder: MLP to project observation features (with distance edge_attr) → mesh nodes
2. Processor: Multiple GATConv layers on the icosahedral mesh
3. Decoder: MLP decoder that maps mesh → target nodes using inverse-distance weighted
