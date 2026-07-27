# Ocelot GNN Model

This repository contains an implementation of the Ocelot Graph Neural Network model for weather prediction.

**Training CLI, actions (`train` / `test` / `predict`), and detailed `train_hetero_model` examples:** see [`ocelot/README.md`](ocelot/README.md).

## Quick start

To run a demonstration of the model:

```bash
qsub ./run_hetero_single.sh
```

## Example commands

### Submit experiments from YAML (SLURM)

[`experiment_configs.yaml`](experiment_configs.yaml) defines named experiments (resources, model, data, inference-only jobs). [`submit_experiments_from_yaml.sh`](submit_experiments_from_yaml.sh) turns them into `sbatch` jobs that run `python -m ocelot.train_hetero_model` with the right flags.

Requires **PyYAML** (`pip install pyyaml`) on the machine where you run the submit script.

```bash
# Submit all experiments in the YAML
bash submit_experiments_from_yaml.sh

# Use a specific config file
bash submit_experiments_from_yaml.sh my_configs.yaml

# Filter by substring in experiment keys (e.g. only "baseline" runs)
bash submit_experiments_from_yaml.sh - baseline

# Dry-run: generate job scripts under ./job_scripts/ but do not sbatch
bash submit_experiments_from_yaml.sh - quick_train --dry-run

# Pin Lightning logger version (folder logs/<exp_name>/<version>/)
bash submit_experiments_from_yaml.sh - baseline_standard --version v1

# Debug: cap wall time at 30 minutes in the generated scripts
bash submit_experiments_from_yaml.sh - quick_train --debug
```

#### Hierarchical mesh experiments

Examples in `experiment_configs.yaml` use keys prefixed with `hier_` (multi-level icosahedral mesh + `interaction` or `transformer` hierarchical processor).

```bash
# Preview hierarchical jobs only (no submission)
bash submit_experiments_from_yaml.sh - hier_ --dry-run

# Submit one experiment by key
bash submit_experiments_from_yaml.sh - hier_transformer_smoke
```

### Train locally (no SLURM)

From the **repository root** (so `python -m ocelot.train_hetero_model` resolves):

```bash
python -m ocelot.train_hetero_model \
  --action train \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name my_run \
  --accelerator gpu \
  --devices 1 \
  --batch_size 2 \
  --max_epochs 40 \
  --lr 0.001
```

#### Hierarchical mesh + processor (optional)

```bash
python -m ocelot.train_hetero_model \
  --action train \
  --model_path ocelot.hetero_observation_interaction.HeteroObservationGraphModel \
  --data_path /path/to/data \
  --start_date 2024-04-01 \
  --end_date 2024-05-30 \
  --exp_name hier_run \
  --accelerator gpu \
  --devices 1 \
  --hierarchical \
  --hierarchical_processor_type interaction \
  --mesh_splits 6 \
  --levels 4 \
  --batch_size 2 \
  --max_epochs 5 \
  --lr 0.0001
```

Use `--hierarchical_processor_type transformer` plus `--processor_window`, `--processor_depth`, `--processor_heads`, etc., for the sliding-window transformer path (see `train_hetero_model.py` / `experiment_configs.yaml`).

### Other submission scripts (site-specific)

```bash
sbatch input_summary_submit.sh 3 parquet /path/to/data/root
```

```bash
qsub ./analysis_submit.sh
```

Define `exp_name` (e.g. `baseline_standard`) inside the script. Input: `debug_outputs`. Analysis name and output paths are set in the script (e.g. figures under `figures/<analysis_name>/<exp_name>`).

```bash
qsub ./metrics_submit.sh
```

Define `exp_name` inside the script. Input: `logs`. Output: `plots`.
