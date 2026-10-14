#!/bin/bash
# Submit the observation-denial experiments: the control and every denial group.
# Run from gnn_model/:  bash evaluation/revision/submit_denial_all.sh
set -euo pipefail
mkdir -p logs
GROUPS_TO_RUN=${GROUPS_TO_RUN:-"control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface"}
for g in ${GROUPS_TO_RUN}; do
  sbatch -J "deny_${g}" --export=ALL,DENY_GROUP="${g}" ${SBATCH_EXTRA:-} evaluation/revision/run_denial_ose.sh
done

cat <<'EOF'

When all arrays have finished, score each group with the verification QC and summarize:

  for g in control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface; do
    PRED_DIR=predictions/denial_2025/$g/pred_csv/obs-space OUT_DIR=evaluation/revision/results/denial/$g \
      CLIM_DIR= N_BOOT=0 VERIFY_QC=evaluation/revision/verify_qc.yaml \
      bash evaluation/revision/run_metrics_array.sh submit
  done
  python evaluation/revision/summarize_denial.py \
    --control evaluation/revision/results/denial/control \
    $(for g in mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface; do
        echo --exp $g=evaluation/revision/results/denial/$g; done) \
    --exclude "ssmis:1,2,3,4" --out evaluation/revision/results/denial/denial_summary.csv
EOF
