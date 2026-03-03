# Evaluation & Plotting Guide (OCELOT vs Truth vs GFS)

This guide explains how to generate evaluation plots and summaries using:

- `run_evaluation.sh` (Slurm batch wrapper)
- `evaluations.py` (OCELOT vs Truth plots + pointwise metrics)
- `compare_to_gfs.py` (build `*_vs_gfs.csv` by interpolating GFS to obs)
- `plot_gfs_compare.py` (single entrypoint: RMSE-vs-fhr + per-fhr maps from `*_vs_gfs.csv`)

It also covers **mesh-grid** comparisons, where OCELOT and GFS are evaluated on the same OCELOT mesh points:

- `compare_mesh_to_gfs.py` (interpolate GFS to OCELOT mesh points)
- `plot_mesh_vs_gfs_maps.py` (maps: OCELOT_on_mesh vs GFS_on_mesh)

The most common workflow is:

1. Generate OCELOT prediction CSVs (training `val_csv/...` or testing `predictions/pred_csv/obs-space/...`).
2. Use `evaluations.py --mode plots` to make OCELOT-vs-Truth figures.
3. (Optional) Use `compare_to_gfs.py` to create `*_vs_gfs.csv`.
4. Use `plot_gfs_compare.py` (or `run_evaluation.sh` with `MODE=gfs_compare`) to plot OCELOT/GFS/Truth comparisons.

If you want a single entrypoint that generates *both* RMSE curves and map panels, use:

- `plot_gfs_compare.py`

---

## 1) What files do these scripts read?

### A) OCELOT vs Truth CSVs
These are the standard “obs-space” artifacts that contain `pred_*` and `true_*` columns.

Typical locations:

- Training validation outputs (created during training):
  - `val_csv/<run_name>/val_<instrument>_target_init_<YYYYMMDDHH>_epoch<E>_batch<B>.csv`
- Testing / inference outputs:
  - `predictions/pred_csv/obs-space/pred_<instrument>_target_init_<YYYYMMDDHH>.csv`

Important columns used by plotting/metrics:

- `lat`, `lon`
- `pred_<var>`, `true_<var>`
- optional QC: `mask_<var>`
- time metadata (newer format): `init_datetime` and `datetime` (valid time) and/or unix time columns
- lead metadata (newer format): `lead_hours_nominal`

**Multi-lead note:** newer `val_csv` files often contain multiple leads (e.g., 3/6/9/12h) in one CSV.
Use `--fhr` when plotting to filter by `lead_hours_nominal`.

### B) OCELOT vs GFS comparison CSVs (`*_vs_gfs.csv`)
These are produced by `compare_to_gfs.py` and include (at least):

- `pred_*`, `true_*` (OCELOT and truth)
- `gfs_*` (GFS interpolated to each observation)
- `lead_hours_nominal` and/or `fhr_used`

`plot_gfs_compare.py` expects files like:

- `.../pred_<instrument>_target_init_<YYYYMMDDHH>_vs_gfs.csv`


### C) Mesh-grid prediction CSVs (OCELOT_on_mesh)

If mesh-grid prediction is enabled, `predict_gnn.py` also writes one CSV per lead hour:

- `predictions/<experiment>/pred_csv/mesh-grid/<instrument>_init_<YYYYMMDDHH>_f<FFF>.csv`

These contain OCELOT predictions at fixed OCELOT mesh points, with strict alignment via `mesh_idx`.


### D) Mesh-grid comparison CSVs (GFS_on_mesh)

To compare OCELOT and GFS **on the same mesh points**, an additional CSV is produced:

- **GFS_on_mesh** (GFS interpolated onto the OCELOT mesh points)
  - Current name:
    - `.../<instrument>_init_<YYYYMMDDHH>_f<FFF>_gfs_on_ocelot_mesh.csv`
  - Legacy names that may exist from older runs:
    - `..._mesh_vs_gfs.csv`
    - `..._ocelot_vs_gfs_mesh.csv`

---

## 2) `evaluations.py` (OCELOT vs Truth plots + metrics)

### Dependencies
- Plotting mode requires `matplotlib` and `cartopy`.
- Metrics mode does **not** require plotting dependencies.

### A) Generate plots for a single init/epoch/batch/lead
Example (training validation under a specific run folder):

```bash
python evaluations.py --mode plots --has_ground_truth \
  --data_dir val_csv/Rand_TenYear_nl16 \
  --plot_dir figures/Rand_TenYear_nl16/epoch100/batch0/fhr3 \
  --epoch 100 --batch_idx 0 --init_time 2024020200 --fhr 3
```

Example (testing/inference obs-space predictions):

```bash
python evaluations.py --mode plots --has_ground_truth \
  --data_dir predictions/pred_csv/obs-space \
  --plot_dir figures/test/obs/init_2025030100 \
  --init_time 2025030100
```

### B) Generate plots for all leads (3/6/9/12h)
If your CSV contains multiple leads, run once per lead:

```bash
for fhr in 3 6 9 12; do
  python evaluations.py --mode plots --has_ground_truth \
    --data_dir val_csv/Rand_TenYear_nl16 \
    --plot_dir figures/Rand_TenYear_nl16/epoch100/batch0/fhr${fhr} \
    --epoch 100 --batch_idx 0 --init_time 2024020200 --fhr ${fhr}
done
```

### C) Pointwise metrics (RMSE/MAE/bias) from `val_csv`
This produces a single CSV with aggregated metrics grouped by keys like instrument and lead time.

```bash
python evaluations.py --mode metrics \
  --data_dir val_csv/Rand_TenYear_nl16 \
  --metrics_out metrics_Rand_TenYear_nl16_pointwise.csv \
  --metrics_groupby instrument,lead_hours_nominal
```

If you want to run over a whole tree (e.g., `val_csv/` containing many runs), use recursion:

```bash
python evaluations.py --mode metrics \
  --data_dir val_csv \
  --metrics_recursive \
  --metrics_pattern "**/val_*.csv" \
  --metrics_out metrics_all_pointwise.csv \
  --metrics_groupby instrument,lead_hours_nominal
```

---

## 3) `compare_to_gfs.py` (build `*_vs_gfs.csv`)

This script interpolates GFS grib fields to each observation location/time and writes a `*_vs_gfs.csv`.

You need a GFS grib directory like:

`<gfs_root>/<YYYYMMDD>/gfs.<YYYYMMDD>.t<HH>z.pgrb2.0p25.f<FFF>`

Example (surface observations for init 2025030100):

```bash
python compare_to_gfs.py \
  --instrument surface_obs \
  --ocelot_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100.csv \
  --gfs_root /path/to/gfs-rt25 \
  --out_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100_vs_gfs.csv \
  --init_mode from_csv \
  --interp nearest
```

Notes:
- OCELOT CSV includes `init_datetime` or `init_time_unix`, `--init_mode from_csv`
- For very large CSVs, consider splitting inputs or using a smaller `--chunk_size` (see script args).

Compared variables (depending on `--instrument` and which `pred_*`/`true_*` columns exist):
- `surface_obs`:
  - 10m winds: `gfs_u10`, `gfs_v10` (m/s)
  - 2m temperature: `gfs_t2m_C` (GFS K → °C)
  - surface pressure: `gfs_sp_hPa` (GFS Pa → hPa)
- `radiosonde` / `aircraft`:
  - isobaric winds: `gfs_u`, `gfs_v` (m/s)
  - isobaric temperature: `gfs_airTemperature_C` (GFS K → °C)

---

## 4) `plot_gfs_compare.py` (RMSE and/or maps vs forecast hour)

This script reads one `*_vs_gfs.csv` and can create:
- RMSE-vs-forecast-hour summary CSV + PNG (default)
- per-forecast-hour map panels (with `--maps`)

Example:

```bash
python plot_gfs_compare.py \
  --init_time 2025030100 \
  --data_dir predictions/pred_csv/obs-space \
  --plot_dir figures/gfs_compare/init_2025030100 \
  --instrument surface_obs \
  --vars wind \
  --chunksize 200000
```

Surface obs temperature + pressure example:

```bash
python plot_gfs_compare.py \
  --init_time 2025030100 \
  --data_dir predictions/pred_csv/obs-space \
  --plot_dir figures/gfs_compare/init_2025030100 \
  --instrument surface_obs \
  --vars temperature_pressure \
  --chunksize 200000
```

Per-forecast-hour maps example:

```bash
python plot_gfs_compare.py \
  --init_time 2025030100 \
  --data_dir predictions/pred_csv/obs-space \
  --plot_dir figures/gfs_compare/init_2025030100 \
  --instrument surface_obs \
  --vars wind_temperature_pressure \
  --chunksize 200000 \
  --maps \
  --fhrs 0 6 12 18 24
```

What gets plotted depends on instrument + available columns. In general you will see RMSE curves for:
- OCELOT − Truth
- OCELOT − GFS
- Truth − GFS


## 5) Mesh-grid evaluation (OCELOT_on_mesh / GFS_on_mesh)

This workflow produces maps on **OCELOT mesh points**, which avoids mixing different spatial samplings.

### A) OCELOT_on_mesh vs GFS_on_mesh

1) Build `*_gfs_on_ocelot_mesh.csv` by interpolating GFS onto the mesh points:

```bash
python compare_mesh_to_gfs.py \
  --mesh_csv predictions/<experiment>/pred_csv/mesh-grid/<instrument>_init_<YYYYMMDDHH>_f<FFF>.csv \
  --gfs_root /path/to/gfs-rt25 \
  --out_csv  predictions/<experiment>/pred_csv/mesh-grid/<instrument>_init_<YYYYMMDDHH>_f<FFF>_gfs_on_ocelot_mesh.csv \
  --interp nearest
```

2) Plot maps (u10/v10/t2m/sp):

```bash
python plot_mesh_vs_gfs_maps.py \
  --csv predictions/<experiment>/pred_csv/mesh-grid/<instrument>_init_<YYYYMMDDHH>_f<FFF>_gfs_on_ocelot_mesh.csv \
  --plot_dir predictions/<experiment>/figures/ocelot_on_mesh_vs_gfs_on_mesh/init_<YYYYMMDDHH>/fhr<HH> \
  --var t2m
```

Outputs include files like:
- `mesh_OCELOT_on_mesh_vs_GFS_on_mesh_t2m.png`

Mesh map note:
- On offline nodes, Cartopy NaturalEarth shapefiles may not be pre-cached; mesh map scripts will skip coastlines/land/borders
  to avoid network downloads.

---

## 6) `run_evaluation.sh` (Slurm wrapper)

`run_evaluation.sh` runs a loop over init times (12-hour increments) and calls either:

- `evaluations.py` (MODE=`standard`) for OCELOT vs Truth plots
- `plot_gfs_compare.py` (MODE=`gfs_compare`) for OCELOT/GFS/Truth RMSE-vs-fhr plots (and optional maps)

Note: `run_evaluation.sh` is for **obs-space** artifacts. For **mesh-grid** plots, use the commands in the “Mesh-grid evaluation” section
or the dedicated end-to-end wrapper `run_pred_eval_gfs.sh`.

### A) Submit with a date range

```bash
sbatch run_evaluation.sh 2025030100 2025030912
```

If you omit args it defaults to a single init time.

### B) Configure what it runs
Open `run_evaluation.sh` and edit the **CONFIGURATION** section:

- `MODE`:
  - `standard` (default): `evaluations.py`
  - `gfs_compare`: `plot_gfs_compare.py`
- `DATA_DIR`: where CSVs are (e.g., `val_csv/<run_name>` or `predictions/pred_csv/obs-space`)
- `PLOT_DIR`: output figure directory
- `HAS_GROUND_TRUTH`:
  - `true` for obs-space with `true_*` columns
  - `false` for forecast-only mesh-grid CSVs
- `FHR_LIST`:
  - `("" )` to run once per init time (no lead filtering)
  - `(3 6 9 12)` to run separate plot jobs per lead (recommended for multi-lead `val_csv`)
- Training validation selection:
  - `EPOCH_TO_PLOT` and `BATCH_IDX_TO_PLOT` (set empty for pure testing-mode files)

### C) Switch to GFS compare mode without editing the script
`MODE`, `INSTRUMENT`, `VARS`, and `CHUNKSIZE` are environment-overridable.

Example:

```bash
MODE=gfs_compare INSTRUMENT=surface_obs VARS=temperature_pressure CHUNKSIZE=200000 \
  sbatch run_evaluation.sh 2025030100 2025030912
```

(You still need `DATA_DIR`/`PLOT_DIR` to point at the directory containing `*_vs_gfs.csv` files.)

---

## 7) Troubleshooting

- **"Expected 1 file, found N"**
  - Add more filters: set `--epoch`, `--batch_idx`, and `--init_time`.
  - For multi-lead files, add `--fhr` to avoid mixing leads.

- **Cartopy/matplotlib import errors**
  - Plotting mode needs them. Metrics mode (`--mode metrics`) does not.

- **Plots look like mixed leads**
  - Use `--fhr 3` / `6` / `9` / `12`. Newer `val_csv` files contain multiple leads.

---

## 8) Quick command reference

Show all CLI flags:

```bash
python evaluations.py --help
python plot_gfs_compare.py --help
python compare_to_gfs.py --help
python compare_mesh_to_gfs.py --help
python plot_mesh_vs_gfs_maps.py --help
```
