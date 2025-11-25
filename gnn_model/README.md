### March 2025
### Azadeh Gholoubi
# End to End Graph Neural Network for Direct Observation Prediction
## Overview
This project implements a Graph Neural Network (GNN) for weather prediction, inspired by DeepMind's Graphcast model. It uses a heterogeneous graph structure to seamlessly integrate and process multiple, diverse observation types (e.g., satellite, surface, and radiosonde data) on a global icosahedral mesh.

The pipeline is built with PyTorch Lightning and PyTorch Geometric and features a modular architecture that separates data processing, model definition, and training into clean, maintainable components.

## Key Features
**Heterogeneous Graph Structure**: The model uses torch_geometric.HeteroData to represent the Earth's atmosphere, with distinct node and edge types for the mesh grid and each observational instrument. This allows for flexible and powerful multi-instrument data fusion.

**Encoder-Processor-Decoder Architecture**:
- An Encoder projects raw features from each observation type onto the shared mesh.
- A deep Processor uses multiple layers of InteractionNetwork blocks for complex message-passing across the mesh graph.
- A Decoder maps the processed mesh state back to the observation locations to make predictions.

**Scalable & Efficient Training**:

- Supports multi-node, multi-GPU distributed training using PyTorch Lightning's DDPStrategy.
- Implements gradient checkpointing to reduce memory usage, allowing for deeper models and larger batch sizes.
- Features a flexible data pipeline with random window resampling for robust, generalized training on massive time-series datasets.

## Core Scripts
- `train_gnn.py`: The main script for launching training and evaluation.
- `gnn_model.py`: Defines the GNNLightning module, which contains the complete model architecture (embedding, encoders, processor, decoders).
- `gnn_datamodule.py`: Handles all data loading, processing, and graph construction, preparing HeteroData batches for the model.
- `process_timeseries.py`: Performs the initial data extraction, time-binning, Quality Control (QC) filtering, and feature engineering.
- `callbacks.py`: Contains custom PyTorch Lightning callbacks for data resampling (ResampleDataCallback).
- `configs/`: Directory for managing observation configurations, such as instrument and channel weights.
## Installation
Create a Conda environment and install the necessary packages.
```bash
pip install -r requirements.txt

```
Or a minimalist install
```bash
pip install numpy pandas scipy torch trimesh networkx torch-geometric scikit-learn zarr joblib lightning psutil

```
## Usage 
### Configure Your Experiment
Modify `train_gnn.py` to set the hyperparameters for your run:
- Set the full date range for the experiment (FULL_START_DATE, FULL_END_DATE).
- Configure the observation_config dictionary to define which instruments and features to use.
- Adjust model hyperparameters like mesh_resolution, hidden_dim, and num_layers.

### Launch training
Use the provided SLURM script (run_gnn.sh) to launch a multi-node training job, or run directly for a single-machine test:
- To start a new run use this from `sbatch run_gnn.sh`:
```bash
srun --cpu-bind=map_cpu:0,1,2,3 python train_gnn.py
```
- To Resume a run from the last saved checkpoint form `sbatch run_gnn.sh`:
 ```bash
python train_gnn.py --resume_from_checkpoint checkpoints/last.ckpt
```
- Then run the script:
```bash
sbatch run_gnn.sh
```
### Debug & plots (optional)
Pass the `--verbose` flag to `train_gnn.py`:
```bash
sbatch run_gnn.sh --verbose
```
### Model Architecture
The GNN uses a multi-stage process that flows from the observations to the mesh, through the processor, and back to the observations.

<img width="410" height="603" alt="image" src="https://github.com/user-attachments/assets/e3028735-1a97-4076-a61f-e9035d1f2d7d" />
