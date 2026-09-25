# Ocelot GNN Model Architecture

[← Repository overview](../README.md)

This directory contains the core implementation of the Ocelot Graph Neural Network (GNN), a model designed for direct-to-observation weather prediction.

## Model Overview

The model leverages a heterogeneous graph structure to process various observation types (e.g., satellite, surface) and predict future weather states. The core components are:

1.  **Encoder**: Maps input observation features onto a global icosahedral mesh.
2.  **Processor**: Utilizes multiple message-passing steps (Interaction Networks) to propagate information across the mesh.
3.  **Decoder**: Maps the processed mesh features back to the target observation locations to make predictions.

This modular architecture, built with PyTorch Geometric and PyTorch Lightning, allows for flexible and scalable experimentation.

## Running the training script

Entry point: `python -m ocelot.train_hetero_model` (run from the repo root so imports resolve).

Typical flags you must set (or override defaults in `train_hetero_model.py`’s `args_dict`):

| Flag | Role |
|------|------|
| `--model_path` | Dotted path to the model class, e.g. `ocelot.hetero_observation_interaction.HeteroObservationGraphModel` |
| `--data_path` | Root directory of the prepared graph/zarr data |
| `--start_date` / `--end_date` | Data window (strings, e.g. `2024-04-01`) |
| `--exp_name` / `--version` | Experiment name and logger subfolder |

Observation selection and stats come from `ocelot/observation_config.yaml` unless you inject `observation_config` / `feature_stats` programmatically.

### Example: train (single GPU)

```bash
python -m ocelot.train_hetero_model \
  --action train \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name my_run \
  --version v1 \
  --accelerator gpu \
  --devices 1 \
  --batch_size 2 \
  --max_epochs 40 \
  --lr 0.001
```

### Example: train then test on best checkpoint

```bash
python -m ocelot.train_hetero_model \
  --action train_and_test \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name my_run \
  --load_ckpt_path /path/to/resume.ckpt   # optional resume for training
```

After training, testing uses the best `ModelCheckpoint` path when available.

### Example: test only (no training)

Requires a checkpoint file.

```bash
python -m ocelot.train_hetero_model \
  --action test \
  --load_ckpt_path /path/to/checkpoint.ckpt \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name my_run
```

### Example: predict only and save outputs to disk

Runs `predict_dataloader` (same batches as `test_dataloader` in this project), loads weights from the checkpoint, and writes a `.pt` file on **rank 0** only in DDP.

```bash
python -m ocelot.train_hetero_model \
  --action predict \
  --load_ckpt_path /path/to/checkpoint.ckpt \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name my_run \
  --version v1 \
  --predict_output /path/to/my_predictions.pt
```

If `--predict_output` is omitted, the default is:

`predict_outputs/<exp_name>/<version>/predictions.pt`

Load saved predictions in Python:

```python
import torch
bundle = torch.load("predict_outputs/my_run/v1/predictions.pt", map_location="cpu")
predictions = bundle["predictions"]   # list of per-batch outputs from predict_step
meta = {k: bundle[k] for k in ("load_ckpt_path", "exp_name", "version")}
```

### Cluster jobs

The repository root includes shell scripts such as `run_hetero_single.sh`, `run_hetero_gpu.sh`, and `submit_experiments_from_yaml.sh` for batch schedulers; adjust paths and modules for your site.

**YAML inference templates** (`experiment_configs.yaml`): `test` (`action: test`) and `inference_predict_example` (`action: predict`, optional `predict_output`). Edit paths, then e.g. `bash submit_experiments_from_yaml.sh - test --dry-run`.

**Early stopping** (`experiment_configs.yaml`): `train_no_early_stop` (`no_early_stopping: true`) and `train_early_stop_patience` (`early_stopping_patience: 10`). Submit e.g. `bash submit_experiments_from_yaml.sh - train_no_early_stop --dry-run`.