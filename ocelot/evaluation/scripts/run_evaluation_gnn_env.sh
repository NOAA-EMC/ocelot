#!/bin/bash -l
#
# Launcher for OCELOT evaluation. Sets up the environment and forwards all
# arguments to evaluations.py. Plotting configuration lives in plotting.yaml.
#
#   sbatch run_evaluation_gnn_env.sh
#   sbatch run_evaluation_gnn_env.sh --instruments atms --max-files 4
#   sbatch run_evaluation_gnn_env.sh --mode metrics
#
# Dry run on the login node (no sbatch needed):
#   sh run_evaluation_gnn_env.sh --filedetection
#
#SBATCH --exclude=u22g09,u22g08,u22g10,u23g12
#SBATCH -A da-cpu
#SBATCH -p u1-service
#SBATCH -J gnn_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --output=slurm/gnn_eval_%j.out
#SBATCH --error=slurm/gnn_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# `sh script.sh` runs this under dash, where `set -o pipefail` is an error --
# and a failing `set` in a POSIX shell aborts the script silently. Re-exec
# under bash so the invocation works either way.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -uo pipefail

# ---------------------------------------------------------------------------
# Where this evaluation code lives. Set it, or export OCELOT_EVAL_DIR.
#
# Deriving it from $0 does not work under sbatch: Slurm runs a copy of this
# script from /var/spool/slurmd/job<N>/slurm_script, so $0 points at the spool
# directory rather than the repo. An explicit path is the only reliable option.
# ---------------------------------------------------------------------------
EVAL_DIR="${OCELOT_EVAL_DIR:-/scratch3/NCEPDEV/da/Mu-Chieh.Ko/OCELOT/DEV/window_hour_fix/ocelot/gnn_model/evaluation/refactored_scripts}"

CONDA_SH="${OCELOT_CONDA_SH:-/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${OCELOT_CONDA_ENV:-gnn-env}"

EVAL_SCRIPT="${EVAL_DIR}/evaluations.py"
CONFIG="${CONFIG:-${EVAL_DIR}/plotting.yaml}"

# MODE=gfs_compare dispatches to the standalone GFS comparison script.
MODE="${MODE:-standard}"
GFS_COMPARE_SCRIPT="${EVAL_DIR}/plot_gfs_compare.py"

fail() { echo "Error: $*" >&2; exit 1; }

[ -d "${EVAL_DIR}" ]    || fail "EVAL_DIR does not exist: ${EVAL_DIR}
       Edit EVAL_DIR at the top of this script, or export OCELOT_EVAL_DIR."
[ -f "${EVAL_SCRIPT}" ] || fail "evaluations.py not found in ${EVAL_DIR}"
[ -f "${CONFIG}" ]      || fail "config not found: ${CONFIG}"
[ -f "${CONDA_SH}" ]    || fail "conda profile not found: ${CONDA_SH}"

# Relative data_dir / plot_dir in the config resolve against the working
# directory unless io.base_dir is set.
cd "${SLURM_SUBMIT_DIR:-${EVAL_DIR}}" || fail "cannot cd to working directory"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}" || fail "could not activate conda env ${CONDA_ENV}"

# evaluations.py imports only eval_plots, which sits beside it.
export PYTHONPATH="${EVAL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "================================================"
echo "OCELOT evaluation"
echo "  host:    $(hostname)"
echo "  arch:    $(uname -m)"
echo "  started: $(date)"
echo "  evaldir: ${EVAL_DIR}"
echo "  workdir: $(pwd)"
echo "  config:  ${CONFIG}"
echo "  args:    $*"
echo "================================================"

if [ "${MODE}" == "gfs_compare" ]; then
    [ -f "${GFS_COMPARE_SCRIPT}" ] || fail "not found: ${GFS_COMPARE_SCRIPT}"
    python -u "${GFS_COMPARE_SCRIPT}" "$@"
    RC=$?
else
    echo "  python:  $(command -v python)"
    echo "  script:  ${EVAL_SCRIPT}"
    python -u "${EVAL_SCRIPT}" --config "${CONFIG}" "$@"
    RC=$?
fi

echo "================================================"
echo "Finished: $(date)  (exit ${RC})"
echo "================================================"
exit ${RC}
