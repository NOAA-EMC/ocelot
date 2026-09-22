# OCELOT

Author: Azadeh Gholoubi

OCELOT is an end-to-end graph neural network workflow for direct observation
prediction in weather applications. The repository includes data preparation,
GNN training/inference, mesh-grid prediction, and evaluation utilities for
OCELOT, persistence, truth, and GFS comparisons.

The `ocelot-v1.0` tag is the reference code snapshot for the OCELOT v1 AIES
submission.

## Repository Layout

- `data_prep/`: readers, mappings, and Zarr/Parquet data preparation tools.
- `gnn_model/`: PyTorch Lightning/PyTorch Geometric model, training scripts,
  prediction workflows, and evaluation scripts.
- `gnn_model/evaluation/`: OCELOT-vs-truth, persistence, and GFS comparison
  documentation and plotting utilities.

## v1.0 Highlights

- Heterogeneous graph neural network for multi-instrument observation prediction.
- Obs-space and mesh-space prediction outputs.
- Persistence-baseline columns (`persist_*`) for obs-space and mesh-grid CSVs.
- Mesh-grid GFS forecast and optional GFS analysis interpolation (`gfs_*`,
  `anl_*`).
- 3-panel and 6-panel mesh plotting for OCELOT/GFS workflows and GFS analysis
  diagnostics.
- 2025 and seasonal Slurm workflows for prediction/evaluation campaigns.

## Quick Start

For model details and training commands, see:

- [`gnn_model/README.md`](gnn_model/README.md)

For evaluation workflows, see:

- [`gnn_model/evaluation/README_EVALUATION.md`](gnn_model/evaluation/README_EVALUATION.md)
