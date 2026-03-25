## Current repo layout

Run commands from the `gnn_model/` directory.

- Code: `FSOI/*.py`
- Configs: `FSOI/configs/*.yaml`
- SLURM scripts: `FSOI/scripts/*.sh`
- Docs: `FSOI/docs/*.md`

Key entrypoints:

```bash
python FSOI/test_fsoi.py --checkpoint /path/to/model.ckpt

python FSOI/fsoi_inference.py \
  --checkpoint /path/to/model.ckpt \
  --config FSOI/configs/fsoi_config.yaml \
  --obs_config configs/observation_config.yaml

python FSOI/visualize_fsoi.py \
  --input ./FSOI/fsoi_outputs/<run_name>/csv \
  --output ./FSOI/fsoi_outputs/<run_name>/figures
```

Notes:
- `FSOI/fsoi_inference.py` uses `--start_date`, `--end_date`, `--output_dir` (underscore flags).
- Outputs are written under `<output_dir>/csv/` and `<output_dir>/logs/`.
- `FSOI/visualize_fsoi.py` accepts either `--input <output_dir>` (it will look in `<output_dir>/csv`) or `--input <output_dir>/csv`.

## Important implementation corrections

### 1) Compute FSOI w.r.t. INPUT observations

FSOI attribution must be with respect to the *inputs* (analysis/background in observation space), not the verification truth.

Current behavior:
- **xa (analysis)** is extracted from `*_input.x` (current window inputs)
- **xb (background)** is produced by forecasting the previous window forward and decoding at the current INPUT locations
- **δx = xa - xb** is computed in the same observation space
- Forecast error is computed against the verification targets, but gradients are taken w.r.t. xa/xb

This is the conceptual alignment required for meaningful observation impact.

### 2) Gradient validation

There is a hard validation step (`validate_gradients`) used during inference to catch broken computational graphs early:
- gradients must be finite (no NaN/Inf)
- gradients must be non-trivial (norm > ~1e-12)

If validation fails for a pair, inference skips that FSOI pair rather than producing misleading outputs.

### 3) Pressure-level stratification

The pipeline can extract pressure metadata (when present in the batch) and stratify channel-level aggregates by pressure.

When enabled, channel aggregation adds:
- `pressure_level_idx`
- `pressure_hpa`

This supports pressure × instrument heatmaps and vertical structure analysis.

## Visualization outputs (what exists today)

Depending on which CSVs are present, visualization can generate:
- Relative contribution by instrument/data type
- Pressure × instrument relative contribution heatmaps
- Per-variable heatmaps (when `target_variable` exists in channel CSV)
- Combined (variable, pressure) heatmaps
- Innovation vs. FSOI scatter plots (from `scatter_samples.csv` if written)