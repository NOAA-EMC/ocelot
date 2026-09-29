#!/bin/bash
# Submit an OCELOT revision launcher on CPU nodes instead of GPUs (no GPU queue wait).
# predict_gnn.py falls back to CPU / FP32 automatically when no GPU is visible.
#
# Usage (from gnn_model/):
#   bash evaluation/revision/submit_cpu.sh evaluation/revision/run_climatology_dump.sh     # E2
#   bash evaluation/revision/submit_cpu.sh evaluation/revision/run_extended_rollout.sh     # E3
#   CPU=1 bash evaluation/revision/submit_denial_all.sh                                   # E4
# Time ONE task first and set CPU_TIME from it:
#   bash evaluation/revision/submit_cpu.sh evaluation/revision/run_extended_rollout.sh --array=0
# Extra arguments are passed to sbatch and override the defaults below.
set -euo pipefail
script=${1:?usage: submit_cpu.sh <launcher.sh> [extra sbatch args]}; shift
mkdir -p logs
CPU_ACCOUNT=${CPU_ACCOUNT:-da-cpu}
CPU_PARTITION=${CPU_PARTITION:-u1-compute}
CPU_QOS=${CPU_QOS:-batch}
CPU_CORES=${CPU_CORES:-32}
CPU_TIME=${CPU_TIME:-08:00:00}
sbatch -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" -q "${CPU_QOS}" --gres=none \
  --cpus-per-task="${CPU_CORES}" --mem=0 -t "${CPU_TIME}" "$@" "${script}"
