# FSOI (Forecast Sensitivity to Observations Impact) for OCELOT

## Overview

This implementation provides **FSOI** (Forecast Sensitivity to Observations Impact) diagnostics for the OCELOT GNN weather forecasting model. FSOI is a post-training, inference-time diagnostic that uses gradients with respect to inputs (not weights) to attribute forecast error to individual observations.

## What is FSOI?

FSOI quantifies how much each observation impacts forecast error through the formula:

```
FSOI(k) = δx(k) ⊙ (ga(k) + gb(k))
```

Where:
- **δx(k)**: Innovation (analysis - background) for observation k
- **ga(k)**: Gradient of forecast error w.r.t. analysis input
- **gb(k)**: Gradient of forecast error w.r.t. background input
- **⊙**: Element-wise multiplication

**Key insight**: FSOI tells you which observations help reduce forecast error (negative FSOI = beneficial) and which increase it (positive FSOI = detrimental).

## Implementation Structure

### Files Created

1. **FSOI/configs/fsoi_config.yaml** - FSOI configuration (lead times, weights, output options)
2. **FSOI/fsoi_dataset.py** - Sequential paired dataset for (previous, current) window pairs
3. **FSOI/fsoi_utils.py** - Helper functions for gradient computation, aggregation, validation
4. **FSOI/fsoi_model_extensions.py** - Model extensions for background prediction
5. **FSOI/fsoi_inference.py** - Main FSOI computation script
6. **FSOI/scripts/run_fsoi.sh** - SLURM batch script for running FSOI
7. **FSOI_README.md** - This file

### Files Modified

1. **gnn_datamodule.py** - Added `fsoi_dataloader()` method for sequential data loading

## Requirements

### Critical Requirement: Sequential Data Order

**FSOI MUST use sequential, deterministic data ordering.** This is because:

1. Background xb(k) is computed by forecasting from the previous window k-1
2. Random shuffling breaks the temporal chain between windows
3. Each window must know what came before it

The implementation enforces:
- `shuffle=False` in dataloader
- Chronologically sorted bin names
- Sequential consistency verification

### Model Requirements

- Trained OCELOT GNN model checkpoint
- Model must be in eval mode with frozen weights
- Gradients are computed w.r.t. inputs, not model parameters

## Usage

### Step 1: Configure FSOI

Edit `FSOI/configs/fsoi_config.yaml`:

```yaml
forecast:
  lead_steps: [0]  # Which forecast steps to score (0 = +12h, 1 = +24h, etc.)
  target_instruments: "all"  # Or list: ["atms", "amsua"]
  use_instrument_weights: true
  use_channel_weights: true
  use_area_weights: true

data:
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  output_dir: "./FSOI/fsoi_outputs"
```

### Step 2: Run FSOI Inference

#### Interactive Mode

```bash
python FSOI/fsoi_inference.py \
    --checkpoint path/to/model.ckpt \
  --config FSOI/configs/fsoi_config.yaml \
  --obs_config configs/observation_config.yaml \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
    --output_dir ./FSOI/fsoi_outputs \
    --device cuda:0
```

#### Batch Mode (SLURM)

1. Edit `FSOI/scripts/run_fsoi.sh` to set:
   - `CHECKPOINT_PATH`: Path to your trained model
   - `START_DATE` and `END_DATE`: Period for FSOI computation
   - SLURM options (account, partition, time, etc.)

2. Submit job:
```bash
chmod +x FSOI/scripts/run_fsoi.sh
sbatch FSOI/scripts/run_fsoi.sh
```

### Step 3: Analyze Results

Results are saved in the output directory:

```
FSOI/fsoi_outputs/
├── csv/
│   ├── fsoi_by_channel.csv      # Per-channel FSOI values
│   ├── fsoi_by_instrument.csv   # Per-instrument aggregates
│   └── fsoi_summary.csv         # Overall summary statistics
├── zarr/                         # Optional: per-observation impacts
└── logs/
    └── fsoi_config_used.yaml    # Config snapshot for reproducibility
```

## Output Format

### fsoi_by_instrument.csv

Columns:
- `instrument`: Instrument name (e.g., "atms", "amsua")
- `n_observations`: Number of observations
- `n_channels`: Number of channels
- `mean_impact`: Average FSOI per observation
- `sum_impact`: Total FSOI (larger magnitude = more impact)
- `positive_frac`: Fraction of observations with positive FSOI
- `pair_idx`: Time window pair index
- `prev_bin`, `curr_bin`: Time window identifiers
- `lead_step`: Forecast lead step
- `ea`, `eb`: Analysis and background forecast errors

### fsoi_by_channel.csv

Same as above but broken down by individual channels within each instrument.

### Interpretation

- **Negative FSOI**: Observation helps reduce forecast error (beneficial)
- **Positive FSOI**: Observation increases forecast error (detrimental)
- **Large |FSOI|**: High impact (good or bad)
- **Small FSOI**: Low impact

**Example**: ATMS channel 8 with sum_impact = -1500 means this channel collectively reduces forecast error by 1500 units across all observations.

## Technical Details

### Algorithm Flow

For each time window pair (k-1, k):

1. **Extract analysis inputs xa(k)**: Actual observations from current window
2. **Compute background xb(k)**: Forecast current-window observations from previous window
3. **Verify alignment**: Ensure xa and xb refer to same observation locations
4. **Compute analysis adjoint ga(k)**:
   - Forward pass with xa
   - Compute forecast error ea
   - Backpropagate: ga = ∇_xa(ea)
5. **Compute background adjoint gb(k)**:
   - Forward pass with xb
   - Compute forecast error eb
   - Backpropagate: gb = ∇_xb(eb)
6. **Compute FSOI**: FSOI = (xa - xb) * (ga + gb)
7. **Aggregate**: By instrument, channel, region, time

### Background Prediction Strategy

Two approaches implemented (you can choose in `fsoi_model_extensions.py`):

**Method 1 (Simple)**: Create hybrid batch
- Use input observations from window k-1
- Use target locations from window k
- Model predicts what window k observations should be

**Method 2 (General)**: Decode at specified targets
- Run encoder on window k-1
- Decode at window k target locations
- More flexible for arbitrary target sets

The code uses Method 1 by default (`predict_at_targets_simple`) as it's simpler and leverages existing batch structure.

### Validation Checks

The implementation includes several validation checks (configurable in `fsoi_config.yaml`):

1. **Alignment verification**: Ensures xa and xb refer to same observations
2. **Gradient verification**: Checks for null, NaN, or zero gradients
3. **Reproducibility check**: Rerunning same case yields same FSOI
4. **Sequential consistency**: Verifies uniform time spacing between windows

## Performance Considerations

### Memory

- FSOI requires keeping computation graphs for backpropagation
- Peak memory ≈ 2× forward pass (stores gradients)
- Use single GPU to avoid synchronization issues
- Process one time pair at a time (batch_size=1)

### Speed

- FSOI is ~3× slower than inference (requires 2 forward passes + 2 backward passes per pair)
- Typical: ~10-20 time pairs per hour on V100/A100
- For 1 month (60 pairs): ~3-6 hours

### Recommended Settings

```yaml
# In FSOI/configs/fsoi_config.yaml
data:
  batch_size: 1  # Must be 1
  num_workers: 1  # Low for stability

# In FSOI/scripts/run_fsoi.sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # Single GPU only
#SBATCH --time=04:00:00  # For 1 month of data
```

## Troubleshooting

### Problem: "No background predictions"

**Cause**: Model cannot predict at current window locations from previous window.

**Solution**: 
- Check that edge connections are being created correctly
- Verify that decoder is receiving target metadata
- Enable verbose logging in `FSOI/configs/fsoi_config.yaml`

### Problem: "Alignment check failed"

**Cause**: xa and xb have different shapes or refer to different observations.

**Solution**:
- Verify both windows have same instruments available
- Check for missing data in either window
- Ensure observation locations haven't changed between windows

### Problem: "Gradients are null or zero"

**Cause**: Model parameters are frozen but computation graph is broken.

**Solution**:
- Ensure NOT using `torch.no_grad()` context
- Check that input tensors have `requires_grad=True`
- Verify forecast error is non-zero and requires_grad=True

### Problem: "Out of memory"

**Cause**: FSOI requires more memory than inference.

**Solution**:
- Use smaller model or reduce hidden_dim (requires retraining)
- Process fewer instruments (use `target_instruments` filter)
- Reduce forecast lead steps (fewer steps to score)

## Extending FSOI

### Add Spatial Aggregation

To create spatial maps of FSOI:

1. Enable in config:
```yaml
output:
  save_zarr: true  # Saves per-observation impacts
```

2. Extract lat/lon from batch and aggregate to grid:
```python
# In fsoi_utils.py, add function:
def aggregate_fsoi_spatially(fsoi_values, batch, grid_res=2.5):
    # Bin observations by lat/lon
    # Sum FSOI within each grid cell
    # Return gridded impact map
```

### Add Temporal Aggregation

Daily/monthly aggregates:

```python
# In post-processing:
df = pd.read_csv("fsoi_by_instrument.csv")
df['date'] = pd.to_datetime(df['curr_bin'].str[:10])
daily = df.groupby(['date', 'instrument'])['sum_impact'].sum()
```

### Add Multiple Lead Times

To score multiple forecast steps:

```yaml
forecast:
  lead_steps: [0, 1, 2]  # +12h, +24h, +36h
```

The code will compute FSOI for each step independently.

## References

1. Langland & Baker (2004): "Estimation of observation impact using the NRL atmospheric variational data assimilation adjoint system"
2. Gelaro et al. (2010): "Evaluation of the 7-km GEOS-5 Nature Run"
3. Cardinali (2009): "Monitoring the observation impact on the short-range forecast"

## Citation

If you use this FSOI implementation, please cite:

```
@software{ocelot_fsoi_2024,
  title={FSOI Implementation for OCELOT GNN Weather Forecasting},
  author={NOAA-EMC OCELOT Team},
  year={2024},
  url={https://github.com/NOAA-EMC/ocelot}
}
```

## Support

For issues or questions:
1. Check this README
2. Review `FSOI/configs/fsoi_config.yaml` for all options
3. Enable verbose logging for detailed diagnostics
4. Open an issue on GitHub with:
   - Config files used
   - Error messages
   - Model checkpoint info
   - Data date range

## License

Same as OCELOT main project (see top-level LICENSE file).
