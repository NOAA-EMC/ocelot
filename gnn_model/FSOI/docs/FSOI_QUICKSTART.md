# FSOI Quick Start Guide

## What You Need

1. Trained OCELOT model checkpoint (`.ckpt` file)
2. Observation data (Zarr files)
3. Single GPU with ~16GB+ memory
4. Python environment with PyTorch, PyTorch Geometric, Lightning

## Quick Start (3 Steps)

### Step 1: Test the Setup

```bash
cd /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/FSOI/ocelot/gnn_model

# Test FSOI implementation (takes ~1 minute)
python FSOI/test_fsoi.py --checkpoint /path/to/your/model.ckpt
```

Expected output:
```
✓ PASS Config Loading
✓ PASS Model Loading
✓ PASS Gradient Computation
✓ PASS FSOI Formula
```

### Step 2: Configure

Edit `FSOI/configs/fsoi_config.yaml`:

```yaml
forecast:
  lead_steps: [0]  # 0 = first forecast step (+12h)

data:
  start_date: "2024-01-01"  # Your date range
  end_date: "2024-01-31"
  output_dir: "./FSOI/fsoi_outputs"
```

Edit `FSOI/scripts/run_fsoi.sh`:

```bash
# Line ~30: Set your checkpoint path
CHECKPOINT_PATH="/path/to/your/trained/model.ckpt"
```

### Step 3: Run FSOI

```bash
# Make executable
chmod +x FSOI/scripts/run_fsoi.sh

# Submit job
sbatch FSOI/scripts/run_fsoi.sh

# Monitor progress
tail -f logs/fsoi_*.out
```

## Understanding Results

Results saved in `FSOI/fsoi_outputs/csv/`:

### fsoi_by_instrument.csv
One row per instrument per time window:

| Column | Meaning |
|--------|---------|
| `instrument` | Instrument name (e.g., "atms") |
| `sum_impact` | Total FSOI (negative = helpful) |
| `mean_impact` | Average FSOI per observation |
| `positive_frac` | % of observations that increased error |

**Key metric**: `sum_impact`
- **Negative = Good**: Observations reduced forecast error
- **Positive = Bad**: Observations increased forecast error
- **Large magnitude**: High impact (important observations)

### Example Interpretation

```csv
instrument,sum_impact,mean_impact,positive_frac
atms,-1234.5,-0.0012,0.35
amsua,+456.2,+0.0008,0.62
surface_obs,-789.1,-0.0015,0.28
```

**Interpretation**:
- **ATMS**: Very helpful! Reduced error by 1234.5 units. 35% had positive impact, 65% helpful.
- **AMSU-A**: Slightly harmful. Increased error. 62% of obs had positive (bad) impact.
- **Surface obs**: Helpful. Reduced error by 789.1 units. 28% harmful, 72% helpful.

## Common Issues

### "No background predictions"

**Fix**: Check that your model has data for consecutive time windows.

```bash
# Verify you have sequential data
python -c "
from process_timeseries import organize_bins_times
# Check bins exist for your date range
"
```

### "Alignment check failed"

**Fix**: Ensure observation locations are consistent between time windows.

Set in `FSOI/configs/fsoi_config.yaml`:
```yaml
validation:
  check_alignment: false  # Temporarily disable if unavoidable
```

### "Out of memory"

**Fix**: Reduce batch size or filter instruments:

```yaml
forecast:
  target_instruments: ["atms", "amsua"]  # Only process these
```

Or reduce date range:
```yaml
data:
  start_date: "2024-01-01"
  end_date: "2024-01-07"  # Just 1 week
```

## Next Steps

1. **Visualize results**:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FSOI/fsoi_outputs/csv/fsoi_by_instrument.csv')

# Plot total impact by instrument
impacts = df.groupby('instrument')['sum_impact'].sum().sort_values()
impacts.plot(kind='barh')
plt.xlabel('Total FSOI Impact (negative = helpful)')
plt.title('Observation Impact on Forecast Error')
plt.tight_layout()
plt.savefig('fsoi_by_instrument.png')
```

2. **Time series analysis**:
```python
# Extract dates from bin names
df['date'] = pd.to_datetime(df['curr_bin'].str[:10])

# Plot impact over time
daily = df.groupby(['date', 'instrument'])['sum_impact'].sum().unstack()
daily.plot(figsize=(12, 6))
plt.ylabel('Daily FSOI Impact')
plt.title('Observation Impact Time Series')
plt.savefig('fsoi_timeseries.png')
```

3. **Channel analysis**:
```python
# Per-channel breakdown
df_ch = pd.read_csv('FSOI/fsoi_outputs/csv/fsoi_by_channel.csv')

# Top 10 most impactful channels
top_channels = df_ch.groupby(['instrument', 'channel'])['sum_impact'].sum()
print(top_channels.abs().nlargest(10))
```

## Performance Tips

### For Large Datasets

If processing >3 months of data:

1. **Split into chunks**:
```bash
# Process monthly
for month in 01 02 03; do
  python FSOI/fsoi_inference.py \
    --checkpoint /path/to/your/model.ckpt \
    --config FSOI/configs/fsoi_config.yaml \
    --obs_config configs/observation_config.yaml \
    --start_date 2024-${month}-01 \
    --end_date 2024-${month}-31 \
    --output_dir FSOI/fsoi_outputs/2024_${month}
done

# Combine results later
```

2. **Parallelize** (if you have multiple models/periods):
```bash
# Submit multiple jobs for different periods
sbatch FSOI/scripts/run_fsoi.sh  # Edit dates between submissions
```

### For Quick Tests

Process just 1 week:
```yaml
data:
  start_date: "2024-01-01"
  end_date: "2024-01-07"
```

Expected runtime: ~30 minutes for 1 week

## Files Reference

| File | Purpose |
|------|---------|
| `FSOI/fsoi_inference.py` | Main FSOI computation script |
| `FSOI/fsoi_utils.py` | Helper functions (gradients, aggregation) |
| `FSOI/fsoi_dataset.py` | Sequential dataset for time pairs |
| `FSOI/fsoi_model_extensions.py` | Background prediction |
| `FSOI/configs/fsoi_config.yaml` | Configuration |
| `FSOI/scripts/run_fsoi.sh` | SLURM batch script |
| `FSOI/test_fsoi.py` | Quick validation test |
| `FSOI_README.md` | Full documentation |

## Getting Help

1. **Check logs**: `logs/fsoi_*.out` and `logs/fsoi_*.err`
2. **Enable verbose mode**: Set `validation.verbose: true` in config
3. **Run test suite**: `python FSOI/test_fsoi.py --checkpoint your_model.ckpt`
4. **Check alignment**: Look for "ALIGNMENT ERROR" messages

## Success Indicators

✅ You should see in the output:

```
[FSOI] Model frozen for inference
[FSOI Dataset] Created with N sequential pairs
✓ FSOI computation complete for lead step 0
Passed: 5/5 tests
Results saved to: ./FSOI/fsoi_outputs
```

✅ CSV files should have:
- Multiple instruments listed
- Mix of positive and negative impacts
- `sum_impact` values in reasonable range (not all zeros)

## One-Line Summary

**FSOI tells you which observations help your forecast** (negative FSOI = good) **and which hurt it** (positive FSOI = bad), computed using gradients from your trained model.
