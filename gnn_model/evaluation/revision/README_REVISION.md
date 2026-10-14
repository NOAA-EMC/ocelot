# OCELOT v1 observation-space evaluation

Scripts that reproduce the verification, baseline comparison, and observation-denial results for
OCELOT v1. All commands run from `gnn_model/` on Ursa (`mkdir -p logs` first) and use the OCELOT v1
checkpoint set in `revision_env.sh` (override with `CKPT=`). The model is not retrained; the only
model-code change is the opt-in `--deny_input_instruments` option of `predict_gnn.py`.

## Contents

| Script | Purpose |
|---|---|
| `revision_env.sh` | Shared paths, denial groups, and the 2025 evaluation subset for the Slurm launchers |
| `run_climatology_dump.sh`, `build_obs_climatology.py` | 2015-2023 observation-space climatology |
| `revision_metrics.py`, `run_metrics_array.sh` | RMSE, bias, ACC, MSESS, C_LS and bootstrap intervals for every target |
| `verify_qc.yaml` | Verification-time QC applied to the verifying observations only |
| `summarize_metrics.py` | Condensed tables from a `revision_metrics.py` results directory |
| `window_counts.py` | Observations per 12-h window after QC and subsampling |
| `stage_graphcastgfs.sh`, `run_graphcastgfs_compare.sh`, `baseline_table.py` | GFS, GraphCastGFS, persistence, and climatology on common observations |
| `run_denial_ose.sh`, `submit_denial_all.sh`, `summarize_denial.py` | Observation-denial experiments |
| `revision_plots.py` | Scorecard, baseline, and denial figures |
| `revision_common.py` | Shared readers, climatology keys, and verification QC |

## Commands

```bash
# Observations per window (training sample and 2025)
python evaluation/revision/window_counts.py --pred_dir predictions/clim_dump_2015_2023 --recursive \
  --out evaluation/revision/results/window_counts_train.csv
python evaluation/revision/window_counts.py --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
  --out evaluation/revision/results/window_counts_test2025.csv

# Climatology from 2015-2023 observations
sbatch evaluation/revision/run_climatology_dump.sh
python evaluation/revision/build_obs_climatology.py --pred_dir predictions/clim_dump_2015_2023 --recursive \
  --verify_qc evaluation/revision/verify_qc.yaml --out_dir evaluation/revision/climatology

# Verification of every target over the 730 initializations of 2025
VERIFY_QC=evaluation/revision/verify_qc.yaml bash evaluation/revision/run_metrics_array.sh submit
python evaluation/revision/summarize_metrics.py --results evaluation/revision/results/v1_2025

# Baselines on the observations of the surface comparison
bash evaluation/revision/stage_graphcastgfs.sh            # on a node with internet access
sbatch evaluation/revision/run_graphcastgfs_compare.sh
python evaluation/revision/baseline_table.py --base OCELOT,GFS,Persistence \
  --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
  --ref GFS=predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space:vs_gfs \
  --ref GraphCastGFS=predictions/ocelot_v1_2025_graphcastgfs/obs-space:vs_graphcastgfs \
  --clim_dir evaluation/revision/climatology --out evaluation/revision/results/baselines_fig6.csv

# Observation-denial experiments (prints the scoring and summary commands)
bash evaluation/revision/submit_denial_all.sh

# Figures
python evaluation/revision/revision_plots.py scorecard --summary evaluation/revision/results/v1_2025/metrics_summary.csv \
  --exclude "ssmis:1,2,3,4" --out fig8_scorecard_msess.png
python evaluation/revision/revision_plots.py scorecard --summary evaluation/revision/results/v1_2025/metrics_summary.csv \
  --metric acc_centered_mean --exclude "ssmis:1,2,3,4" --out fig8_scorecard_acc.png
python evaluation/revision/revision_plots.py baselines --table evaluation/revision/results/baselines_fig6.csv \
  --out fig6_baselines.png
python evaluation/revision/revision_plots.py denial --summary evaluation/revision/results/denial/denial_summary.csv \
  --zero_ns --out fig9_denial.png
```

SSMIS channels 1-4 are left out of the verification because of documented F17 striping noise
(Dong and Zou 2022).
