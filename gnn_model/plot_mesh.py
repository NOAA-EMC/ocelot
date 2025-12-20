#!/usr/bin/env python
"""
Mesh prediction plotting script - matches evaluations.py style

Usage:
    python plot_mesh_predictions.py --input_dir mesh_predictions --output_dir mesh_plots
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _add_land_boundaries(ax):
    """Add coastlines and borders to map - matches evaluations.py"""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)


def plot_mesh_prediction(
    df: pd.DataFrame,
    variable: str,
    title: str = None,
    output_path: str = None,
    vmin: float = None,
    vmax: float = None,
) -> plt.Figure:
    """
    Plot a single mesh prediction variable - matches evaluations.py style.
    
    Args:
        df: DataFrame with columns ['lat', 'lon', 'pred_{variable}']
        variable: Variable name (will look for 'pred_{variable}' column)
        title: Plot title (auto-generated if None)
        output_path: Path to save figure (None = don't save)
        vmin, vmax: Color scale limits (auto if None)
        
    Returns:
        matplotlib Figure object
    """
    # Get prediction column
    pred_col = f'pred_{variable}'
    if pred_col not in df.columns:
        raise ValueError(f"Column '{pred_col}' not found in DataFrame. Available: {df.columns.tolist()}")
    
    # Extract data
    lats = df['lat'].values
    lons = df['lon'].values
    values = df[pred_col].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(values)
    lats = lats[valid_mask]
    lons = lons[valid_mask]
    values = values[valid_mask]
    
    if len(values) == 0:
        print(f"Warning: No valid values for {variable}")
        return None
    
    # Auto color limits
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    
    # Create figure - single panel like evaluations.py prediction panels
    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    # Plot data - matching evaluations.py style exactly
    sc = ax.scatter(
        lons, lats,
        c=values,
        cmap="jet",  # Match evaluations.py
        s=7,         # Match evaluations.py
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    
    # Add colorbar - match evaluations.py style
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1).set_label("Value")
    
    # Add map features - match evaluations.py
    _add_land_boundaries(ax)
    ax.set_global()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Set title
    if title is None:
        title = f"{variable}"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150)  # Match evaluations.py dpi
        print(f"  -> Saved plot: {output_path}")
    
    return fig


def plot_all_variables_from_csv(
    csv_file: str,
    output_dir: str,
) -> None:
    """
    Plot all prediction variables from a single CSV file.
    
    Args:
        csv_file: Path to CSV file
        output_dir: Directory to save plots
    """
    df = pd.read_csv(csv_file)
    
    # Find all prediction columns
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    variables = [col.replace('pred_', '') for col in pred_cols]
    
    if not variables:
        print(f"No prediction columns found in {csv_file}")
        return
    
    # Extract info from filename: instrument_f{fhr:03d}_epoch{epoch}_batch{batch}.csv
    basename = os.path.basename(csv_file).replace('.csv', '')
    parts = basename.split('_')
    
    # Try to extract epoch and forecast hour for title
    epoch_str = ""
    fhr_str = ""
    for part in parts:
        if part.startswith('epoch'):
            epoch_str = f" • Epoch: {part.replace('epoch', '')}"
        elif part.startswith('f') and len(part) == 4:  # f003, f006, etc
            fhr_str = f" • Forecast Hour: {int(part[1:])}"
    
    # Plot each variable
    for var in variables:
        title = f"Mesh Prediction: {var}{fhr_str}{epoch_str}"
        output_path = os.path.join(output_dir, f"{basename}_{var}.png")
        
        try:
            plot_mesh_prediction(
                df=df,
                variable=var,
                title=title,
                output_path=output_path,
            )
            plt.close()
        except Exception as e:
            print(f"Error plotting {var}: {e}")


def batch_plot_directory(
    input_dir: str,
    output_dir: str,
    instrument: str = None,
    epoch: int = None,
    batch: int = None,
) -> None:
    """
    Plot all CSV files in a directory.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save plots
        instrument: Filter by instrument name (None = all)
        epoch: Filter by epoch (None = all)
        batch: Filter by batch (None = all)
    """
    # Build glob pattern
    pattern = os.path.join(input_dir, '*.csv')
    csv_files = glob.glob(pattern)
    
    # Filter files
    if instrument:
        csv_files = [f for f in csv_files if instrument in os.path.basename(f)]
    if epoch is not None:
        csv_files = [f for f in csv_files if f'epoch{epoch}_' in os.path.basename(f)]
    if batch is not None:
        csv_files = [f for f in csv_files if f'batch{batch}' in os.path.basename(f)]
    
    if not csv_files:
        print(f"No CSV files found matching criteria in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to plot")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each file
    for csv_file in sorted(csv_files):
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        plot_all_variables_from_csv(csv_file, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Plot mesh predictions (matches evaluations.py style)')
    parser.add_argument('--input_dir', type=str, default='mesh_predictions',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='mesh_plots',
                        help='Directory to save plots')
    parser.add_argument('--instrument', type=str, default=None,
                        help='Filter by instrument name')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Filter by epoch')
    parser.add_argument('--batch', type=int, default=None,
                        help='Filter by batch')
    
    args = parser.parse_args()
    
    # Plot all files
    batch_plot_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        instrument=args.instrument,
        epoch=args.epoch,
        batch=args.batch,
    )


if __name__ == '__main__':
    main()
