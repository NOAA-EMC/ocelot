# FSOI for OCELOT — Index

This folder documents and supports the FSOI (Forecast Sensitivity to Observations Impact) workflow.

Assumption: run commands from the `gnn_model/` directory.

## Start here

- `FSOI_QUICKSTART.md` — minimal steps to run
- `FSOI_README.md` — technical detail and interpretation
- `FSOI_RADIOSONDE_TEMP_GUIDE.md` — radiosonde-temperature-focused workflow
- `FSOI_DEV_NOTES.md` — consolidated implementation/validation notes

## Current layout (post-cleanup)

- Code: `FSOI/*.py`
- Configs: `FSOI/configs/*.yaml`
- SLURM: `FSOI/scripts/*.sh`
- Docs: `FSOI/docs/*.md`

## Canonical workflow

```bash
# 1) Quick sanity test
python FSOI/test_fsoi.py --checkpoint /path/to/model.ckpt

# 2) Run FSOI (interactive)
python FSOI/fsoi_inference.py \
  --checkpoint /path/to/model.ckpt \
  --config FSOI/configs/fsoi_config.yaml \
  --obs_config configs/observation_config.yaml \
  --start_date 2024-01-01 \
  --end_date 2024-01-07 \
  --output_dir ./FSOI/fsoi_outputs/my_run

# 3) Visualize
python FSOI/visualize_fsoi.py \
  --input ./FSOI/fsoi_outputs/my_run/csv \
  --output ./FSOI/fsoi_outputs/my_run/figures
```

Batch mode:

```bash
sbatch FSOI/scripts/run_fsoi.sh
```

## Output structure

Each run writes:

```
<output_dir>/
  csv/
    fsoi_by_instrument.csv
    fsoi_by_channel.csv
    fsoi_by_time.csv
    scatter_samples.csv            # optional
  logs/
```

## Key scripts

- `FSOI/fsoi_inference.py` — main computation (`--start_date/--end_date` underscore flags)
- `FSOI/test_fsoi.py` — quick checks (only requires `--checkpoint`)
- `FSOI/visualize_fsoi.py` — generates plots from CSVs
