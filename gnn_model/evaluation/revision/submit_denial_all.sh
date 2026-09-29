#!/bin/bash
# Submit E4 for the matched control and every denial group, then (after completion) score.
# Run from gnn_model/:  bash evaluation/revision/submit_denial_all.sh
set -euo pipefail
mkdir -p logs
GROUPS_TO_RUN=${GROUPS_TO_RUN:-"control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional"}
# CPU=1 submits on CPU nodes through submit_cpu.sh (control and all groups on the same hardware).
for g in ${GROUPS_TO_RUN}; do
  if [[ "${CPU:-0}" == "1" ]]; then
    bash evaluation/revision/submit_cpu.sh evaluation/revision/run_denial_ose.sh -J "deny_${g}" --export=ALL,DENY_GROUP="${g}" ${SBATCH_EXTRA:-}
  else
    sbatch -J "deny_${g}" --export=ALL,DENY_GROUP="${g}" ${SBATCH_EXTRA:-} evaluation/revision/run_denial_ose.sh
  fi
done

cat <<'EOF'

When all arrays have finished, score each group and summarize (CPU node):

  for g in control mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional; do
    python evaluation/revision/revision_metrics.py --n_boot 0 \
      --pred_dir predictions/denial_2025/$g/pred_csv/obs-space \
      --out_dir  evaluation/revision/results/denial/$g
  done
  python evaluation/revision/summarize_denial.py \
    --control evaluation/revision/results/denial/control \
    $(for g in mw_sounders mw_imager ir_imagers scatterometer aircraft radiosonde surface all_satellite all_conventional; do
        echo --exp $g=evaluation/revision/results/denial/$g; done) \
    --out evaluation/revision/results/denial/denial_summary.csv
EOF
