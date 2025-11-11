#!/bin/bash
#
# Analyze FSOI results and generate GraphDOP-style plots
#
# Usage on HPC:
#   cd /scratch3/NCEPDEV/da/Azadeh.Gholoubi/add_fsoi/ocelot/gnn_model
#   bash run_fsoi_analysis.sh

# Results directory (where your CSV files are)
RESULTS_DIR="fsoi_results_conventional"

# Run the analysis
echo "Analyzing FSOI results from: $RESULTS_DIR"
python analyze_fsoi_results.py \
    --results_dir $RESULTS_DIR \
    --output_dir ${RESULTS_DIR}/analysis

echo ""
echo "✓ Analysis complete!"
echo "Results saved to: ${RESULTS_DIR}/analysis/"
echo ""
echo "Generated plots:"
echo "  - fsoi_per_channel_graphdop_style.png  ← This is the plot you want!"
echo "  - input_impact_on_conventional_obs.png ← Shows how inputs affect u,v,T,q,p"
echo "  - fsoi_by_instrument_mean.png"
echo "  - fsoi_analysis_summary.txt"
echo ""
echo "To view on HPC:"
echo "  ls -lh ${RESULTS_DIR}/analysis/"
echo "  # Transfer PNG files to your local machine for viewing"
