#!/bin/bash
#SBATCH -J analysis
#SBATCH -A da-cpu
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00
#SBATCH -o jobs/analysis_%J.out
#SBATCH -e jobs/analysis_%J.err
#SBATCH --mail-user=$LOGNAME@noaa.gov
#SBATCH --mem=0
set -x 

# load environment needed to run python scripts
source  /home/Xin.C.Jin/modules/env_diatom.sh

# Parse command-line options
VERSION="latest"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-v|--version VERSION]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-v|--version VERSION]"
            exit 1
            ;;
    esac
done

# Define experiment parameters
#EXP_NAME="baseline_standard"
#EXP_NAME="regional_smoke"
#EXP_NAME="regional_standard"
#EXP_NAME="da_standard"
EXP_NAME="da_anal_standard"
#EXP_NAME="longrun_standard"
#EXP_NAME="baseline_8nodes"
#EXP_NAME="medium_hidden_256"
#EXP_NAME="medium_mesh_16"
#EXP_NAME="medium_mesh_16_no_net"
#EXP_NAME="longrun_hidden_layers_1"
RANK=0
ANALYSIS_NAME="my_analysis"
DEBUG_BASE_DIR="/scratch3/NCEPDEV/da/Xin.C.Jin/git/ocelot/gnn_model/REGIONAL/debug_outputs"

# List of instruments to analyze
INSTRUMENTS=(
    "satellite_atms"
    "conventional_surface_obs"
    "satellite_diag_atms"
    "satellite_diag_amsua"
    "satellite_diag_ssmis"
    "conventional_radiosonde"
    "conventional_diag_surface_obs_t"
    "conventional_diag_sst"
    "satellite_diag_avhrr"
    "satellite_diag_cris-fsr"
    "conventional_diag_t"
    "state_ges"
)

# Loop through each instrument
for instrument in "${INSTRUMENTS[@]}"; do
    echo "========================================"
    echo "Analyzing instrument: $instrument (version: $VERSION)"
    echo "========================================"
    
    python -m ocelot.analyze_outputs \
        --exp_name "$EXP_NAME" \
        --rank "$RANK" \
        --instrument "$instrument" \
        --analysis_name "$ANALYSIS_NAME" \
        --debug_base_dir "$DEBUG_BASE_DIR" \
        --version "$VERSION"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully analyzed: $instrument"
    else
        echo "✗ Failed to analyze: $instrument"
    fi
    echo ""
done

echo "========================================"
echo "All instruments analyzed!"
echo "========================================"
