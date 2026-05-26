# FSOI Framework — Changes Made (Sessions 2026-05-22, 2026-05-23, 2026-05-24)

---

## Session 2026-05-24 — All 17 Jobs Complete; FSOI Weights Computed

### Summary

All 17 fixed jobs (3 seasonal obs-space + 12 mesh-space + 2 OSE ATMS) confirmed
complete. `compute_fsoi_weights.py` run across all four obs-space seasonal CSVs
(Jan/Apr/Jul/Oct 2025). Weight outputs written to `FSOI/fsoi_weights/`.

### FSOI Weight Computation

**Command run:**
```bash
cd gnn_model/
python FSOI/compute_fsoi_weights.py \
    --fsoi_dirs \
        FSOI/fsoi_outputs/seasonal_fixed/radiosonde_jan2025/csv \
        FSOI/fsoi_outputs/seasonal_fixed/radiosonde_apr2025/csv \
        FSOI/fsoi_outputs/seasonal_fixed/radiosonde_oct2025/csv \
        FSOI/fsoi_outputs/full_eval_fixed_20250701_20250731/radiosonde/csv \
    --obs_config configs/observation_config.yaml \
    --output_dir FSOI/fsoi_weights \
    --min_pairs 20
```

**Input:** 131,624 rows; Jan=58 pairs, Apr=56, Oct=60, Jul=60.

**Instrument weights (Variant A / Variant B):**

| Instrument | mean_impact | pos_frac | Weight A | Weight B |
|---|---|---|---|---|
| radiosonde | −3.51e−05 | 0.458 | **8.178** | **8.179** |
| aircraft | −6.56e−07 | 0.470 | 0.153 | 0.146 |
| surface_obs | +4.55e−07 | 0.445 | 0.106 | 0.111 |
| amsua/ascat/avhrr/atms/seviri_asr/ssmis | ±1e−10–1e−08 | 0.21–0.28 | 0.094 | 0.094 |

Key observations:
- Radiosonde is 50× dominant; all satellites at the 0.094 minimum floor.
- Variant A ≈ Variant B across all instruments (reliability doesn't differentiate).
- Surface_obs, ATMS, AMSUA, SSMIS are net detrimental (positive mean_impact).

**Per-variable channel weights:** Wind variables (u, v) carry most impact for
radiosonde (u=2.20, v=1.60), aircraft (v=3.21), and ascat (u=1.24, v=2.15).
Dewpoint dominates for avhrr (2.82) and seviri_asr (1.15).

**Outputs written:**
```
FSOI/fsoi_weights/
├── fsoi_weight_summary.csv
├── observation_config_variant_a.yaml
├── observation_config_variant_b.yaml
└── observation_config_channel_weighted.yaml
```

### Documentation updates (2026-05-24)
- `CURRENT_STATE_2026_05_23.md`: updated to reflect all 17 jobs complete,
  added full weight table and channel weight table, updated paper status,
  removed stale "running" job entries and "when job finishes" steps.
- `CHANGES_SESSION_2026_05_22.md`: this entry added.

### Bias identified in obs-space weights (same session)

All 4 seasonal obs-space runs used `target_instruments: ["radiosonde"]`. The loss
e(xa) was evaluated only at ~3,000 NH-concentrated radiosonde sites, not globally.
This inflates radiosonde's own weight and suppresses satellites that improve
ocean/tropical/SH areas. Weights in `FSOI/fsoi_weights/` are biased and must
not be used for training.

**Resolution:** recomputed weights from the 12 completed mesh-space jobs (3 levels
× 4 seasons), which verify against GFS analysis at 40,962 global mesh nodes.

**Mesh-space weight computation:**
```bash
python FSOI/compute_fsoi_weights.py \
    --fsoi_dirs \
        FSOI/fsoi_outputs/mesh_seasonal/radiosonde_{250,500,850}hPa_{jan,apr,jul,oct}2025/csv \
    --obs_config configs/observation_config.yaml \
    --output_dir FSOI/fsoi_weights_mesh \
    --min_pairs 20
```

**Input:** 6,168 rows × 60 pairs from 12 files.

**Key result — weights nearly identical to biased obs-space set:**

| Instrument | Mesh Wt A | Obs-only Wt A | Δ |
|---|---|---|---|
| radiosonde | 8.166 | 8.178 | −0.01 |
| aircraft | 0.154 | 0.153 | +0.001 |
| surface_obs | 0.117 | 0.106 | +0.011 |
| all satellites | 0.094 | 0.094 | 0 |

Radiosonde dominance is **real**, not an artifact of verification bias.
Satellite positive_frac shifts upward in mesh-space (0.35–0.42 vs 0.21–0.28
obs-space), confirming that satellites appear less beneficial globally than near
radiosonde sites — but the floor minimum absorbs the difference.

**Primary weights for fine-tuning:** `FSOI/fsoi_weights_mesh/observation_config_variant_{a,b}.yaml`

### Two bugs found and fixed in `fsoi_ose.py` (same session)

#### Bug OSE-1 — Scale mismatch: `ea_control` is sum-of-strata, `ea_denied` is single-call

**Root cause:** When the stratified path runs
(`stratify_by_pressure: true`, `stratify_by_variable: true`), the caller
computes 64 per-(level, variable) loss values and sums them:

```python
ea_total = sum(lr['ea_p'] for lr in per_level_results)   # ≈ 632
```

This `ea_total` was passed as `ea_control` to `compute_ose_for_pair`.  But
`compute_ose_for_pair` calls a single unstratified `compute_forecast_error` for
`ea_denied` (≈ 9.95).  The ratio ea_total / ea_denied ≈ 64 = number of strata,
making every closure_ratio off by ~64×.

**Effect on saved CSVs:** `ose_atms_jul2025_fixed/evaluation/ose_results.csv` and
`ose_vs_fsoi_comparison.csv` have ea_control ≈ 632, ea_denied ≈ 9.9, which is
invalid.  These must be regenerated with the fixed code.

**Fix (`fsoi_ose.py`):** `compute_ose_for_pair` now always recomputes `ea_control`
from scratch using the same single `compute_forecast_error` call as `ea_denied`.
The caller-supplied `ea_control` is logged but not used for the impact calculation.
Both forward passes now use identical aggregation → closure_ratio is meaningful.

#### Bug OSE-2 — Sign convention: `sign_agree` always False for correctly-signed pairs

**Root cause:** FSOI and OSE use opposite sign conventions:
- FSOI > 0 means detrimental (instrument increases forecast error)
- ose_impact = ea_denied − ea_control < 0 means detrimental (removing it helped)

The comparison code used `sign(fsoi) == sign(ose_impact)` — which is False
whenever both agree the instrument is detrimental (the common case for ATMS).
`closure_ratio = fsoi / ose_impact` was also inverted (should aim for −1.0, not +1.0).

**Fix (`fsoi_ose.py`):** `compare_ose_vs_fsoi` now negates `ose_impact` to convert
it to the same "positive = detrimental" convention as FSOI before computing
`sign_agree` and `closure_ratio`.  A correct closure_ratio now aims for +1.0.

#### OSE reruns needed
Both OSE output directories contain invalid comparison CSVs:
- `ose_atms_jul2025_fixed/evaluation/ose_vs_fsoi_comparison.csv` — wrong scale
- `ose_atms_jan2025_fixed/evaluation/ose_vs_fsoi_comparison.csv` — wrong scale
  (Jan OSE has no `evaluation/` dir at all — comparison was never written)

Rerun both OSE jobs with the fixed code, then run `plot_ose_comparison.py` on
each output directory.

### Next step
Fine-tune OCELOT with mesh-space Variant A and Variant B weights and evaluate
RMSE vs baseline (M0). Configs ready at
`FSOI/fsoi_weights_mesh/observation_config_variant_{a,b}.yaml`.

---

Branch: `feature/FSOI_evaluation`  
Commits: `93d3721` (framework), `a65232c` (weights), `7f04963` (column-offset fix), `f8fdbe6` (FD precision fix)

---

## 0. Critical Bugs Found and Fixed (Session 2026-05-23)

### Bug 1 — `fsoi_validation.py`: `replace_batch_inputs` missing `observation_config` argument
**Affected file:** `FSOI/fsoi_validation.py`  
**Symptom:** All FD check runs crashed with `TypeError: replace_batch_inputs() missing 1 required positional argument: 'observation_config'`. The FD check had NEVER successfully run — every prior run died before completing a single forward pass.  
**Root cause:** `replace_batch_inputs` was refactored to require `observation_config` as its third argument, but the three call sites in `finite_difference_check` (lines 102, 134, 162) were not updated.  
**Fix:** Added `model.observation_config` as the third argument to all three calls.  
**Commit:** part of session; fixed inline before submitting first corrected FD job.

---

### Bug 2 — `fsoi_utils.py` / `fsoi_model_extensions.py`: CRITICAL column-offset bug in observation extraction
**Affected files:** `FSOI/fsoi_utils.py` (two functions), `FSOI/fsoi_model_extensions.py` (one function)  
**Commit:** `7f04963`  
**Impact:** **ALL prior FSOI CSV outputs are INVALID.** This bug corrupted `δx = xa − xb` for every instrument.

**Root cause:** `x_input` tensor (assembled by `process_timeseries.py`) has layout:
```
[7 geo/time | n_meta instrument-metadata | n_channels obs-features | optional trailing]
```
where geo/time = `[sin_lat, cos_lat, sin_lon, cos_lon, sin_time, cos_time, dayofyear]` (7 cols) and instrument-metadata = scan angles (satellites) or log_pressure (radiosonde).

All three functions assumed `x_input[:, :n_channels]` were the observation channels. They were not:

| Instrument | Wrong extraction (old) | Correct extraction (fixed) |
|---|---|---|
| ATMS (n_meta=3, n_ch=22) | cols 0–21 = geo/time + scan_angles + bt_ch1–12 | cols 10–31 = bt_ch1–22 |
| Radiosonde (n_meta=1, n_ch=4) | cols 0–3 = sin_lat, cos_lat, sin_lon, cos_lon | cols 8–11 = T, Td, u, v |
| Aircraft (n_meta=1, n_ch=3) | cols 0–2 = sin_lat, cos_lat, sin_lon | cols 8–10 = T, q, u |
| Surface_obs (n_meta=1, n_ch=5) | cols 0–4 = geo/time | cols 8–12 = T, Td, u, v, ps |
| AMSUA/SSMIS (n_meta=3, n_ch=15/24) | cols 0–14/23 = geo/time + scan_angles + some BT | cols 10–24/33 = full BT channels |

**Consequence for innovation diagnostics:** The ATMS "channel 8" anomaly (+1.74σ bias, 411% nRMSE, obs_range≈0.05) reported in `INNOVATION_QUALITY_REPORT.md` was actually col 7 = `cos(sensorZenithAngle)` — a scan angle, which is near-constant within a swath and has no business appearing in FSOI. The entire INNOVATION_QUALITY_REPORT is invalid.

**Three functions fixed:**

1. **`get_fsoi_inputs`** (`fsoi_utils.py`):
   ```python
   # BEFORE (wrong):
   x_channels = x_input[:, :n_channels]
   # AFTER (fixed):
   n_meta = len(cfg.get('metadata', []))
   bt_start = 7 + n_meta
   x_channels = x_input[:, bt_start:bt_start + n_channels]
   ```

2. **`replace_batch_inputs`** (`fsoi_utils.py`):
   ```python
   # BEFORE: torch.cat([full_channels, metadata_full])  — dropped geo/time
   # AFTER:  torch.cat([prefix, full_channels, metadata_full])
   # where prefix = x_orig[:, :bt_start]  (geo/time + inst-metadata, kept fixed)
   #       full_channels = new xa or xb channels
   #       metadata_full = x_orig[:, bt_start+n_channels:]  (trailing one-hot etc.)
   ```

3. **`predict_at_targets`** (`fsoi_model_extensions.py`):
   ```python
   # BEFORE: metadata = x_input[:, n_channels:]  → for ATMS: bt_ch13–22 + sat-id one-hot
   # AFTER:  metadata = x_input[:, 7:bt_start]   → actual scan angles / inst-metadata
   ```

**OSE also affected:** `fsoi_ose.py` calls `replace_batch_inputs` to substitute `xb[denied_inst]` for `xa[denied_inst]`. With the old bug, the "denial" was overwriting geo/time columns with predicted BT values while leaving the actual BT channels unchanged — the denial was completely ineffective. OSE results are also invalid.

---

### Bug 3 — FD check float32 precision failure
**Affected file:** `FSOI/fsoi_validation.py`, `FSOI/fsoi_inference.py`  
**Commit:** `f8fdbe6`  
**Symptom:** 30/30 FAIL in both `fd_check_rerun` (wrong columns) and `fd_check_fixed` (correct columns after Bug 2 fix). Pattern: `error_plus ≈ error_minus ≈ error_original` → FD ≈ 0, relative error ≈ 100%.  
**Root cause:** `compute_forecast_error` uses `loss_reduction='sum'`, producing error magnitude ~10⁵ (summed over thousands of verification targets). Perturbing one observation by ε=1e-4 changes the error by ~gradient × 1e-4 ≈ 10⁻³. But float32 ULP at scale 10⁵ ≈ 0.01 → change is invisible to FD.  
**Fix:**
- Added `loss_reduction='mean'` to all three `compute_forecast_error` calls in `finite_difference_check`. Mean error is O(1), making single-observation changes detectable.
- Increased default `epsilon` from 1e-4 to 1e-2 in `finite_difference_check` and `validate_fsoi_gradients`. ε=0.01 in normalized obs space is well within the linearization regime (<<1σ for all channels).
- Updated fallback default in `fsoi_inference.py` (`fd_epsilon` config key) from 1e-4 to 1e-2.

**FD check jobs:**
- Job 14148440 (`fd_check_rerun`): ran with wrong columns + ε=1e-4 → 30/30 FAIL (expected)
- Job 14149704 (`fd_check_fixed`): ran with correct columns + ε=1e-4 → 30/30 FAIL (float32 precision)
- Job 14151602 (`fd_check_precision`): running with correct columns + ε=1e-2 + `loss_reduction='mean'` → expected PASS

---

## 1. Core Bug Fixes

### `fsoi_utils.py`
- **`_default_target_channel_names`**: Added `surface_obs` (and aliases `surface`, `synop`, `metar`, `sfcship`) with 5 channels: temperature, specific_humidity, u_wind, v_wind, surface_pressure. Previously fell back to uninformative `channel_1` etc.
- **`sample_innovation_vs_fsoi`**: Added `obs_coords: Optional[Dict[str, Tuple]]` parameter. When lat/lon arrays are passed (already subsampled to match xa), appends `lat` and `lon` columns to scatter_samples output. Required for gridded FSOI maps and regional analysis.
- **`get_fsoi_metadata`**: Fixed lat/lon extraction — previously only checked `node_data.metadata` (a combined [N,2] tensor that most batches don't have). Now first checks `node_data.lat` and `node_data.lon` directly (which all batches do have).
- **`compute_per_level_fsoi_by_variable`**:
  - Replaced hard `raise ValueError` for missing `pressure_level` with a soft path: instruments without pressure levels (e.g. `surface_obs`) get `unique_levels = [None]` — a single "surface level" using all observations.
  - Updated `_loss_for` inner function to skip pressure masking when `p_idx is None`.
  - Updated results assembly to emit `p_idx=None` and `p_hpa=NaN` for surface targets.
- **`compute_forecast_error_on_mesh`** *(new function)*: Computes MSE between OCELOT mesh decoder output and a GFS analysis tensor [N_mesh, C]. Calls `model._decode_one_step_to_mesh()` directly (bypassing the `no_grad` wrapper in `_decode_all_steps_to_mesh`) so gradients flow. NaN channels in GFS reference are automatically excluded from MSE. Supports cosine-latitude area weighting using mesh node coordinates.
- **`_unwrap_predictions_and_mesh`** *(new)*: Returns `(predictions, mesh_features_per_step)` without stripping mesh features.

### `fsoi_inference.py`
- **`use_instrument_weights` / `use_channel_weights` flags now honored**: Were silently ignored — weights from YAML always applied regardless of config. Fixed: empty dict `{}` is now passed when flags are False (all weights = 1.0 when disabled). All three configs set these to False, so this has zero numerical effect currently (all YAML weights happen to be 1.0) but is correct for future use.
- **`compute_fsoi_for_pair` signature extended** with `run_fd_check`, `run_diagnostics`, `run_repro_check`, `ose_instruments`, `gfs_reference`, `mesh_instrument`, `mesh_pressure_level_idx` parameters.
- **`_compute_innovation_diagnostics`** *(new)*: Per-(instrument, channel, pair) statistics: mean, std, skewness, RMSE, obs_range, normalized_rmse of δx = xa − xb. Invoked by `--diagnostics` flag.
- **`_run_ose_check`** *(new helper)*: Calls `fsoi_ose.compute_ose_for_pair`, appends to `results['ose_records']`.
- **`_run_repro_check`**: Implemented — was in config (`check_reproducibility: true`) but never coded. Re-clones xa, re-runs forward+backward, compares ea and ga. Saves `reproducibility_check.csv`. Runs on pair_idx==0 only.
- **FD check across 3 dates**: Changed from `pair_idx == 0` hard-code to configurable `fd_pair_set = {0, N//3, 2N//3}` (or from config `fd_check_pairs`). Per-sample FD records accumulate and write to `evaluation/fd_validation.csv` at end of run.
- **`obs_coords` construction**: After xa subsampling alignment, builds per-instrument `(lat_1d, lon_1d)` arrays (applying same subsample indices) for passing to scatter samples.
- **`--verification_target mesh`** *(new)*: When set, loads GFS analysis for each pair's valid time via `fsoi_gfs_loader.load_gfs_on_mesh()`, builds normalized [N_mesh, C] tensor, replaces both `ea` and `eb` computations with `compute_forecast_error_on_mesh`. Innovation δx, xb computation, and FSOI formula are unchanged.
- **`--ose_instruments`** *(new)*: Space-separated list of instruments to run OSE denial on. Adds 1 no-grad forward pass per pair.
- **`--diagnostics`** *(new)*: Enables innovation diagnostics output.
- **`--gfs_root`, `--mesh_instrument`, `--mesh_pressure_level_idx`** *(new)*.
- **End-of-run outputs**: Writes `evaluation/fd_validation.csv`, `evaluation/innovation_diagnostics.csv`, `evaluation/ose_results.csv`, `evaluation/ose_vs_fsoi_comparison.csv`, `evaluation/reproducibility_check.csv`.

### `fsoi_model_extensions.py`
- **`_stratified_spatial_subsample`** *(new)*: Replaces `torch.randperm` for decoder subsampling. Divides globe into 10°×10° cells, samples proportionally from each non-empty cell. Ensures geographic representativeness in `sum_impact_scaled` scaling — the old random sampling caused geographic concentration bias, worst for AVHRR at scale 20.5×.

### `evaluate_fsoi_results.py`
- **`_closure_quality_flag`** *(new)*: Returns PASS/WARN/FAIL/INSUF based on `sign_agreement_frac ≥ 0.90` AND `|median_closure_ratio − 1| ≤ 0.15`.
- **`_closure_summary_records`**: Each row now includes `quality_flag`. Per-(variable, pressure) rows separated into `fsoi_closure_per_level_summary.csv`.
- **`write_closure_diagnostics`**: Prints full per-level table (not just `head(12)`), lists WARN/FAIL cells explicitly.
- **`compute_beneficial_fraction`** *(new)*: Per-pair: `helpful = sum(negative FSOI)`, `helpful_fraction_of_abs_total = |helpful| / (|helpful|+|harmful|)` (target 50–80%). Fixed formula — old version computed `|helpful_fsoi| / ea_total` which gives nonsensical values (~2%) because ea is orders of magnitude larger than FSOI increments.
- **`_DEFAULT_REGIONS`** and **`compute_regional_impact`** *(new)*: Reads scatter_samples.csv lat/lon, filters by 5 geographic regions (tropics, mid-lat NH/SH, polar NH/SH), computes per-instrument sum/mean FSOI. Writes `fsoi_regional_summary.csv`. Requires lat/lon in scatter_samples (populated by `get_fsoi_metadata` fix above).
- **`evaluate()` and `main()`**: Call `compute_beneficial_fraction` and `compute_regional_impact` automatically.

### `visualize_fsoi.py`
- **`plot_positive_frac_timeseries`** *(new)*: Time series of `positive_frac` per instrument. Green reference band [20–50%], red dashed lines at 20%/70%. Instruments with `std > 0.15` drawn dashed with warning. Writes `positive_frac_timeseries.png` and `positive_frac_timeseries.csv`.

### `configs/fsoi_config_radiosonde_all.yaml`
- `use_area_weights: false` → **`true`** — required for globally representative FSOI; without this, NH mid-latitude radiosonde sites dominate.
- `finite_difference_check: false` → **`true`** with `fd_num_samples: 10` (10 samples × 3 dates = 30 total triples across the 30-day window).

### `scripts/run_fsoi_radiosonde_all.sh`
- Added `--diagnostics` flag to `fsoi_inference.py` call.
- Added `plot_fsoi_maps.py` call (gridded FSOI maps) after scatter_samples.csv exists.
- Added `plot_innovation_diagnostics.py` call.

---

## 2. New Files

### New Configs
| File | Purpose |
|---|---|
| `configs/fsoi_config_aircraft.yaml` | Aircraft as verification target (T, q, u, v; pressure-stratified; area-weighted; FD check enabled; aircraft humidity NOT masked) |
| `configs/fsoi_config_surface_obs.yaml` | Surface obs as verification target (T, q, u, v, ps; no pressure stratification; area-weighted) |
| `configs/fsoi_config_mesh_radiosonde.yaml` | Mesh-space FSOI at configurable pressure level vs GFS analysis |

### New Python Modules
| File | Purpose |
|---|---|
| `fsoi_gfs_loader.py` | Load GFS GRIB2 analysis, bilinear interpolation to mesh nodes, normalize to model feature space. Supports `radiosonde` (T, u, v at isobaric level) and `surface_obs` (2m T, 10m u/v, MSLP). Time-interpolates between 6-hourly GFS cycles. |
| `fsoi_ose.py` | OSE computation: `compute_ose_for_pair()` replaces `xa[X]→xb[X]`, runs 1 no-grad forward pass, returns ea_control/ea_denied/ose_impact. `compare_ose_vs_fsoi()` merges with FSOI CSV to compute closure_ratio and sign_agree. |
| `plot_fsoi_maps.py` | 6 global map types from scatter_samples.csv: signed total, absolute, relative contribution %, beneficial fraction, per-instrument, per-variable. Cartopy optional. |
| `plot_innovation_diagnostics.py` | 4 plots from innovation_diagnostics.csv + scatter_samples.csv: δx histograms vs Gaussian fit, bias time series, normalized RMSE bar chart, skewness heatmap. |
| `plot_ose_comparison.py` | 3 plots from ose_results.csv + fsoi_by_instrument.csv: scatter OSE vs FSOI predicted, time series, closure ratio histogram. Writes `ose_summary.csv` with PASS/WARN flag. |
| `compare_obs_vs_mesh_fsoi.py` | 6-panel comparison of obs-space vs mesh-space rankings: scatter, side-by-side bars, sign-flip table, vertical profile of impact vs pressure level, seasonal stability. |
| `compute_fsoi_weights.py` | Reads FSOI CSVs, computes Variant A (∝ |mean_impact|) and Variant B (∝ |mean_impact| × reliability²) training weights, writes `observation_config_variant_{a,b}.yaml` and channel-weighted config. |

### New Run Scripts
| File | Purpose |
|---|---|
| `scripts/run_fsoi_aircraft.sh` | SLURM job for aircraft-target FSOI |
| `scripts/run_fsoi_surface_obs.sh` | SLURM job for surface_obs-target FSOI |
| `scripts/run_fsoi_mesh.sh` | SLURM job for mesh-space FSOI (accepts `--pressure IDX`) |
| `scripts/run_fsoi_comparison.sh` | Post-processing job (u1-service, no GPU) — runs `compare_obs_vs_mesh_fsoi.py` |
| `scripts/submit_fsoi_full_evaluation.sh` | Master script: submits radiosonde + aircraft + surface_obs jobs in parallel |
| `scripts/submit_fsoi_mesh_seasonal.sh` | Submits 4-season mesh-space FSOI jobs at one or more pressure levels |
| `scripts/submit_fsoi_seasonal_rerun.sh` | Submits Jan/Apr/Oct 2025 obs-space re-runs with all fixes |

---

## 3. FSOI Weights (Preliminary — to be updated)

`FSOI/fsoi_weights/radiosonde_4seasons/`  
Computed from **old runs (no area weighting)** — to be regenerated after fixed seasonal runs complete.

| Instrument | Weight A | Weight B | Notes |
|---|---|---|---|
| radiosonde | 5.48 | 5.47 | Dominant |
| aircraft | 2.57 | 2.44 | Second |
| surface_obs | 0.37 | 0.37 | Moderate |
| All satellites | 0.097 | 0.10–0.18 | Near floor |

---

## 4. Key Scientific Findings (Jul 2025 obs-space, area-weighted)

> ⚠️ **These findings are based on pre-Bug-2-fix runs and are INVALID.** Rankings, magnitudes, and innovation diagnostics were computed with geo/time encoding as "observations". All jobs must be rerun after commit `7f04963`.

- **Aircraft** most beneficial (mean impact −4.5), **ATMS** appears net detrimental (+3.1) — **DO NOT USE for paper**
- **Helpful fraction 55.5%**, **closure ratio 1.064** — **DO NOT USE for paper** (δx was geo-encoding, not observations)
- These numbers will be regenerated from corrected runs.

---

## 5. Rerun Plan (Session 2026-05-23)

All FSOI output directories listed below were generated with the Bug 2 column-offset error and must be recomputed:

All invalid directories were **deleted on 2026-05-23**. Only two directories remain in `fsoi_outputs/`:

| Directory | Type | Status |
|---|---|---|
| `fd_check_skip/` | FD check (ε=1e-2 + SKIP logic) | **RUNNING — job 14152884** |
| `full_eval_fixed_20250701_20250731/radiosonde/` | Jul 2025 obs-space, all fixes applied | **RUNNING — job 14152885** |

Previously deleted (all had column-offset bug; some also lacked area weighting):
`full_eval_20250701_20250731/`, `seasonal_fixed/`, `mesh_seasonal/`, `ose_atms_jul2025/`, `ose_atms_jan2025/`, `comparison/`, all `radiosonde_allvars_impact_*` dirs, `fd_check_rerun/`, `fd_check_fixed/`, `fd_check_precision/`

### FD check — fundamental limitation for high-N instruments

Per-observation FD is not feasible in float32 for instruments with large observation counts:

| Instrument | N_obs/pair | Typical gradient | Δerror @ ε=0.01 | Float32 ULP | Detectable? |
|---|---|---|---|---|---|
| ATMS | ~100k | ~1e-8 | ~1e-10 | ~8e-9 (at loss~0.07) | **NO** |
| AVHRR | ~250k | ~1e-8 | ~1e-10 | ~2e-8 (at loss~0.2) | **NO** |
| SSMIS | ~50k | ~1e-8 | ~1e-10 | ~1e-8 (at loss~0.15) | **NO** |
| Radiosonde | ~3k | ~0.002 | ~2e-5 | ~1e-6 (at loss~10) | **YES** |
| Aircraft | ~20k | ~1e-5 | ~1e-7 | ~1e-7 (at loss~0.9) | **MARGINAL** |

**Fix (commit 40f6bc6):** Added SKIP status for |gradient| < max(1e-4, 1e-5/epsilon). High-N instruments get SKIP (not FAIL); radiosonde/aircraft/surface_obs cases with detectable gradients get PASS/WARN/FAIL.

**Primary validation**: FSOI closure ratio (Σ FSOI_i ≈ ea − eb) is the correct aggregate check. The FD check is a secondary diagnostic for a few well-conditioned cases.

### Rerun order (updated)
1. **Job 14152884** (`fd_check_skip`): expect radiosonde PASS/WARN; ATMS/AVHRR/SSMIS SKIP
2. **Job 14152885** (`full_eval_fixed_20250701_20250731/radiosonde`): primary result — check closure ratio, instrument rankings
3. After #2 completes: run seasonal obs-space (Jan/Apr/Oct 2025)
4. Run mesh-space (12 jobs)
5. Run OSE ATMS (Jul + Jan)
6. Recompute FSOI weights from corrected seasonal runs
7. Update `FSOI/docs/FSOI_EVALUATION_RESULTS.md` with corrected numbers

### FSOI Weights status
Current weights in `FSOI/fsoi_weights/radiosonde_4seasons/` are from pre-fix runs — **do not use for training**.
