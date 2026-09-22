#!/bin/bash -l
#SBATCH -A gpu-ai4wp
#SBATCH -p u1-service
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J fsoi_compare
#SBATCH --output=fsoi_compare_%j.out
#SBATCH --error=fsoi_compare_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# FSOI Comparison Job — runs after all seasonal mesh + obs-space jobs finish
# Submitted with --dependency=afterany:<all_job_ids>
# No GPU needed — pure post-processing on completed CSVs.
# ============================================================================

set -e

CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate gnn-env

find_gnn_model_dir() {
    local d="$1"
    for _ in 1 2 3 4 5 6 7 8; do
        [ -f "$d/configs/observation_config.yaml" ] && [ -d "$d/FSOI" ] && echo "$d" && return 0
        d="$(dirname -- "$d")"
    done; return 1
}
GNN_MODEL_DIR_RESOLVED=""
[ -n "${GNN_MODEL_DIR:-}" ] && GNN_MODEL_DIR_RESOLVED="$GNN_MODEL_DIR"
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && \
    GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$SLURM_SUBMIT_DIR")" || true
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && \
    GNN_MODEL_DIR_RESOLVED="$(find_gnn_model_dir "$PWD")" || true
[ -z "$GNN_MODEL_DIR_RESOLVED" ] && { echo "ERROR: cannot find gnn_model dir"; exit 1; }
cd "$GNN_MODEL_DIR_RESOLVED"
echo "[COMPARE] Working directory: $(pwd)"
echo "[COMPARE] Start: $(date)"

# First re-run evaluate on any seasonal_fixed dirs that just completed
echo ""
echo "=== Step 1: Re-run evaluate on newly completed obs-space runs ==="
for d in FSOI/fsoi_outputs/seasonal_fixed/radiosonde_*/; do
    if [ -f "${d}csv/fsoi_by_instrument.csv" ] && \
       [ ! -f "${d}evaluation/fsoi_evaluation_summary.csv" ]; then
        echo "  Evaluating: $d"
        python FSOI/evaluate_fsoi_results.py \
            --input "$d" --output "${d}evaluation" 2>/dev/null || true
    fi
done

echo ""
echo "=== Step 2: Generate comparison figures ==="
python FSOI/compare_obs_vs_mesh_fsoi.py \
    --base  FSOI/fsoi_outputs \
    --output FSOI/fsoi_outputs/comparison \
    2>&1

echo ""
echo "=== Step 3: Print key result ==="
SUMMARY="FSOI/fsoi_outputs/comparison/obs_vs_mesh_summary.csv"
if [ -f "$SUMMARY" ]; then
    echo "Obs-space vs Mesh-space ranking comparison:"
    python3 -c "
import pandas as pd
df = pd.read_csv('$SUMMARY')
print(df.to_string(index=False))
flip_cols = [c for c in df.columns if c.startswith('sign_flip')]
if flip_cols:
    flippers = df[df[flip_cols].any(axis=1)]['instrument'].tolist()
    print(f'\nSign-flip instruments: {flippers}')
    print('These instruments changed sign between obs-space and mesh-space.')
    print('Most likely cause: NH geographic concentration bias or background field bias.')
"
fi

echo ""
echo "=== Step 4: Summary of all outputs ==="
ls FSOI/fsoi_outputs/comparison/ 2>/dev/null

echo ""
echo "============================================="
echo "FSOI comparison complete: $(date)"
echo "Outputs: FSOI/fsoi_outputs/comparison/"
echo "============================================="