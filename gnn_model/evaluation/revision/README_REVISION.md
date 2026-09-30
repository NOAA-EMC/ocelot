# AIES-D-26-0089 revision experiments (OCELOT v1)

Branch: `revision/aies-d-26-0089`, created from `release/ocelot-v1.0` so that the manuscript checkpoint
runs with exactly the code it was trained with. All commands run from `gnn_model/` on Ursa
(`mkdir -p logs` first). Every experiment uses the
manuscript checkpoint `/scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt` (override with `CKPT=`).
Nothing below changes OCELOT v1 itself; the code changes are opt-in flags:

| File | Change |
|---|---|
| `gnn_model.py` | `model.denied_input_instruments`: skip those encoders (default empty = v1 behaviour) |
| `predict_gnn.py` | `--rollout_windows N` (targets span N x 12 h; default 1 = v1), `--deny_input_instruments` |
| `process_timeseries.py`, `gnn_datamodule.py` | `rollout_windows` (default 1 reproduces v1 binning exactly; tested) |
| `train_gnn.py` | `--processor_type {sliding_transformer,interaction}`, `--deny_input_instruments` |

The Python analysis scripts in this directory were tested on synthetic CSVs in the exact
`pred_*_target_init_*.csv` format; the Slurm launchers pass `bash -n` but have not been run on Ursa.

## Experiments, in priority order

| ID | Addresses | What | Cost | Script |
|---|---|---|---|---|
| E0 | R2 minor (dataset size) | Obs counts per instrument/split, archive size, parameter count | CPU, <1 h | `data_inventory.py` |
| E1 | R1 c2, R1 c3 | Metrics for **all** targets/channels/levels from the existing 730-init 2025 predictions: RMSE, bias, corr, ACC, MSESS vs climatology & persistence, error-coherence index, bootstrap CIs | CPU, few h (needs E2 for ACC/clim) | `revision_metrics.py` |
| E2 | R1 c2 (ACC), R2 (climatology baseline) | Training-period (2015-2023) obs-space climatology: 648 inits dumped, then aggregated | 324 x 1-GPU tasks (<=2 h) + CPU | `run_climatology_dump.sh`, `build_obs_climatology.py` |
| E3 | R2 (rollout > 4 steps) | Frozen v1 rolled to 48 h (16 steps) on 122 inits | 122 x 1-GPU tasks (<=3 h) | `run_extended_rollout.sh` |
| E4 | R1 c5 (source ablation) | 9 observation-denial groups + matched control, 122 inits each | 1,220 x 1-GPU tasks (<=2 h) | `submit_denial_all.sh`, `summarize_denial.py` |
| E5 | R1 c4, R2 (ML baseline) | NOAA GraphCastGFS verified at the same surface/radiosonde/aircraft obs as OCELOT, GFS, persistence, climatology | download + 730 CPU tasks | `stage_graphcastgfs.sh`, `run_graphcastgfs_compare.sh`, `baseline_table.py` |
| E6 (NOT RUN) | R1 c4/c5 (design choices; training-time denial) | 5 retrainings at equal budget: v1, no spatial mixing, GraphCast-style interaction processor, satellite-denied, conventional-denied | 5 x (4 nodes x 8 H100) x budget | `run_train_ablation.sh` |

E0-E5 need no retraining and fit comfortably before the 28 Nov 2026 deadline. E6 was the only
expensive item and was **not run** for this revision: the reviewers' ablation request (R1 c5) is
answered by the E4 denial experiments, and the response states explicitly that they measure what the
trained model relies on rather than what a model trained without a system could reach. All E6
passages have been removed from the response and revision documents, and the preliminary processor
comparison in Section 7 stands unchanged. `run_train_ablation.sh` is kept for v2.
To size it later: `MAX_EPOCHS=1000` is ~1/3 of the v1 run; estimate wall time from the v1 logs.

## Commands

```bash
# E0
python evaluation/revision/data_inventory.py --data_path $DATA_PATH \
  --ckpt /scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt --out evaluation/revision/results/data_inventory.csv

# E2 -> climatology
sbatch evaluation/revision/run_climatology_dump.sh
python evaluation/revision/build_obs_climatology.py --pred_dir predictions/clim_dump_2015_2023 \
  --recursive --out_dir evaluation/revision/climatology

# E1 (all 730 inits already on disk)
python evaluation/revision/revision_metrics.py \
  --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
  --clim_dir evaluation/revision/climatology --out_dir evaluation/revision/results/v1_2025

# E3
sbatch evaluation/revision/run_extended_rollout.sh            # ROLLOUT_WINDOWS=4 (48 h) default
python evaluation/revision/revision_metrics.py --pred_dir predictions/rollout_48h/pred_csv/obs-space \
  --clim_dir evaluation/revision/climatology --out_dir evaluation/revision/results/rollout_48h

# E4  (prints the scoring commands when it submits)
bash evaluation/revision/submit_denial_all.sh
# Score each finished group as a CPU array, with the same verification QC as the manuscript run
# so that the control stays comparable to results/v1_2025. Groups are independent; submit as
# many as are finished.
for g in control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional; do
  PRED_DIR=predictions/denial_2025/$g/pred_csv/obs-space   OUT_DIR=evaluation/revision/results/denial/$g   CLIM_DIR= N_BOOT=0 N_SHARDS=8 VERIFY_QC=evaluation/revision/verify_qc.yaml     bash evaluation/revision/run_metrics_array.sh submit
done
python evaluation/revision/check_denial_control.py   --control evaluation/revision/results/denial/control --reference evaluation/revision/results/v1_2025
python evaluation/revision/summarize_denial.py   --control evaluation/revision/results/denial/control   $(for g in mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional; do
      echo --exp $g=evaluation/revision/results/denial/$g; done)   --out evaluation/revision/results/denial/denial_summary.csv

# E5
bash evaluation/revision/stage_graphcastgfs.sh                 # on a node with internet
sbatch evaluation/revision/run_graphcastgfs_compare.sh
python evaluation/revision/baseline_table.py \
  --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
  --ref GFS=predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space:vs_gfs \
  --ref GraphCastGFS=predictions/ocelot_v1_2025_graphcastgfs/obs-space:vs_graphcastgfs \
  --clim_dir evaluation/revision/climatology --out evaluation/revision/results/baselines.csv

# E6 (optional)
for v in v1_budget no_spatial_mix interaction deny_satellite deny_conventional; do
  sbatch -J abl_$v --export=ALL,VARIANT=$v evaluation/revision/run_train_ablation.sh; done
```

## Checks before trusting the numbers

1. **E1 sanity:** surface T2m and 10-m wind RMSE from `metrics_summary.csv` must reproduce Fig. 6
   (3.23 -> 3.64 K) when restricted to the persistence-valid rows (`rmse_model_on_pers`).
2. **E4 control:** `denial_2025/control` must match the manuscript predictions for the same inits
   (tiny FP16 differences only). If it doesn't, stop and check CKPT/config. Run
   `check_denial_control.py --control results/denial/control --reference results/v1_2025`: it
   compares RMSE per target on the shared initializations. Targets whose sample sizes differ were
   scored under different `--verify_qc` and are excluded, and the QC summary each run wrote is
   printed so you can see which run applied what. The verdict is about systematic offsets, not
   about any single target: a wrong checkpoint or config moves a whole instrument one way, while
   reduced precision scatters in sign and looks largest, in relative terms, on low-variance
   radiance channels whose RMSE is only a degree or two. It fails on a non-zero overall median,
   on an instrument offset consistently one way, or on any target beyond 3%.
3. **E5 GFS coverage:** `_vs_gfs.csv` files exist for surface_obs for all 730 inits. Radiosonde/aircraft
   GFS files exist only if they were built; build them with `INSTRUMENT=radiosonde|aircraft
   RUN_PREDICTION=0 CSV_ONLY=1` in `run_pred_eval_gfs_2025.sh` (CPU).
4. **E5 GraphCastGFS:** confirm the bucket path first (see header of `stage_graphcastgfs.sh`); the
   +3 h and +9 h values are time-interpolated from 6-hourly output, so report +6 h and +12 h as primary.
5. **Climatology:** `build_obs_climatology.py` refuses any file with init year > 2023 (leakage guard).

## Metric definitions (for the manuscript)

* ACC: per initialization, lead and variable, centered Pearson correlation between forecast and
  observed anomalies relative to the 2015-2023 observation-space climatology (stratified by
  1 deg [conventional] / 2 deg [satellite] cell, month, 3-h UTC bin, pressure level and scan-angle
  bin, with fallback to cell-month and 5-deg-band-month when a stratum has < 10 samples).
  ACC is averaged over initializations; CIs are from bootstrap over initializations.
* MSESS_ref = 1 - MSE_OCELOT / MSE_ref on the rows where the reference exists.
* Error-coherence index C_LS = (S_LS - S_0)/(SSE - S_0), where S_LS = sum over observations of the
  squared mean error of the observation's 5-deg box and S_0 its expectation under random permutation
  of the same errors across locations. It is 0 for spatially random (noise-like) errors and approaches
  1 when errors are organized at >= 5-deg scales. This quantifies the "spatially coherent residual" statements.
