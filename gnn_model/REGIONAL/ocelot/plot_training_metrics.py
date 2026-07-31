#!/usr/bin/env python
"""
Script to plot training metrics from metrics.csv file.

Usage:
    python -m ocelot.plot_training_metrics --metrics_csv /path/to/metrics.csv --exp_name my_experiment
    python -m ocelot.plot_training_metrics --metrics_csv /path/to/metrics.csv --exp_name baseline_standard --output_dir ./my_plots
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_metrics_data(metrics_csv_path: str) -> pd.DataFrame:
    """Load metrics from CSV file.

    Args:
        metrics_csv_path: Path to the metrics.csv file

    Returns:
        DataFrame with metrics data
    """
    print(f"Reading metrics from: {metrics_csv_path}")
    df = pd.read_csv(metrics_csv_path)
    print(f"Loaded {len(df)} rows")
    print(f"DEBUG: Columns: {df.columns.tolist()}")

    # Find all loss columns
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    print(f"DEBUG: Loss-related columns found: {loss_cols}")

    print(f"DEBUG: Data distribution:")
    print(f"  - Rows with non-null step: {df['step'].notna().sum()}")
    print(f"  - Rows with non-null epoch: {df['epoch'].notna().sum()}")

    for col in loss_cols[:5]:  # Show first 5 loss columns
        non_null_count = df[col].notna().sum()
        print(f"  - Rows with non-null {col}: {non_null_count}")
        if non_null_count > 0:
            print(f"    First few values: {df[col].dropna().head(5).tolist()}")

    return df


def extract_instruments(df: pd.DataFrame) -> List[str]:
    """Extract instrument names from column names.

    Args:
        df: DataFrame with metrics

    Returns:
        Sorted list of instrument names
    """
    train_loss_cols = [col for col in df.columns if col.startswith(
        'train_loss_') and col.endswith('_target_epoch')]
    val_loss_cols = [col for col in df.columns if col.startswith(
        'val_loss_') and col.endswith('_target_epoch')]

    instruments = set()
    for col in train_loss_cols:
        parts = col.replace('train_loss_', '').replace('_target_epoch', '')
        instruments.add(parts)
    for col in val_loss_cols:
        parts = col.replace('val_loss_', '').replace('_target_epoch', '')
        instruments.add(parts)

    instruments = sorted(instruments)
    print(f"Found {len(instruments)} instruments:")
    for inst in instruments:
        print(f"  - {inst}")
    print()
    return instruments


def aggregate_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by epoch, taking the last value per epoch.

    Args:
        df: DataFrame with metrics

    Returns:
        DataFrame aggregated by epoch
    """
    df_with_epoch = df[df['epoch'].notna()].copy()

    def _last_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[-1] if len(s2) > 0 else pd.NA

    df_epoch = (
        df_with_epoch
        .sort_values(['epoch', 'step'], na_position='last')
        .groupby('epoch', as_index=False)
        .agg(_last_non_null)
    )
    print(f"Found {len(df_epoch)} epochs with data\n")
    return df_epoch


def prepare_plot_data(df: pd.DataFrame, min_epochs: int = 2,
                      max_points: int = 200) -> Tuple[pd.DataFrame, str, str]:
    """Prepare data for plotting, using steps if not enough epochs.

    Args:
        df: DataFrame with metrics
        min_epochs: Minimum number of epochs required to use epoch-based plotting
        max_points: Maximum number of points to plot for step-based plotting (default: 200)

    Returns:
        Tuple of (aggregated_df, x_axis_label, loss_column) where:
            - aggregated_df: DataFrame with data to plot
            - x_axis_label: 'Epoch' or 'Step'
            - loss_column: Name of the loss column to use ('train_loss_epoch' or 'train_loss')
    """
    # Try epoch-based aggregation first
    df_epoch = aggregate_by_epoch(df)

    if len(df_epoch) >= min_epochs:
        print(
            f"Using epoch-based plotting ({len(df_epoch)} epochs available)\n")
        return df_epoch, 'Epoch', 'train_loss_epoch'
    else:
        print(
            f"⚠ Only {len(df_epoch)} epoch(s) available (minimum {min_epochs} required)")
        print(f"Falling back to step-based plotting\n")

        # Debug: Check original dataframe
        print(f"DEBUG: Original df has {len(df)} rows")
        print(f"DEBUG: Columns in df: {df.columns.tolist()}")

        # For step-based plotting, we need to find a loss column that's logged per step
        # train_loss_epoch is only logged at epoch boundaries, so look for
        # step-level losses

        # Strategy: Look for per-step loss columns
        # 1. Try 'train_loss' (per-step overall loss)
        # 2. Try any 'train_loss_*_target' columns (per-instrument, per-step)
        # 3. Fall back to 'train_loss_epoch' (will only have epoch boundary
        # points)

        loss_col = None

        # First priority: overall per-step loss
        if 'train_loss_step' in df.columns:
            per_step_count = df['train_loss_step'].notna().sum()
            print(
                f"DEBUG: train_loss_step has {per_step_count} non-null values")
            if per_step_count > len(
                    df_epoch) * 2:  # Need significantly more than just epoch boundaries
                loss_col = 'train_loss_step'
                print(
                    f"DEBUG: Using 'train_loss_step' column for step-based plotting ({per_step_count} points)")

        if loss_col is None and 'train_loss' in df.columns:
            # Check if it has per-step data
            per_step_count = df['train_loss'].notna().sum()
            if per_step_count > len(df_epoch):
                loss_col = 'train_loss'
                print(
                    f"DEBUG: Using 'train_loss' column for step-based plotting ({per_step_count} points)")

        if loss_col is None:
            # Look for per-step instrument losses (columns like
            # 'train_loss_satellite_atms_target_step')
            step_loss_cols = [col for col in df.columns
                              if col.startswith('train_loss_')
                              and col.endswith('_target_step')]

            if step_loss_cols:
                # Use the first one we find, or compute average
                loss_col = step_loss_cols[0]
                per_step_count = df[loss_col].notna().sum()
                print(
                    f"DEBUG: Using '{loss_col}' column for step-based plotting ({per_step_count} points)")
                print(
                    f"DEBUG: Found {len(step_loss_cols)} per-step instrument loss columns")
            else:
                # Last resort: use train_loss_epoch (will only have 2 points)
                loss_col = 'train_loss_epoch'
                print(
                    f"DEBUG: WARNING - Using 'train_loss_epoch' column (only has {df[loss_col].notna().sum()} points)")
                print(
                    f"DEBUG: This will result in very few data points. Consider logging per-step losses.")

        # Use all rows with step and loss data
        print(f"DEBUG: Before filtering - df has {len(df)} rows")
        print(f"DEBUG: Rows with step.notna(): {df['step'].notna().sum()}")
        print(
            f"DEBUG: Rows with {loss_col}.notna(): {df[loss_col].notna().sum()}")

        df_steps = df[(df['step'].notna()) & (
            df[loss_col].notna())].copy().reset_index(drop=True)

        print(
            f"DEBUG: After filtering for step.notna() & {loss_col}.notna(), df_steps has {len(df_steps)} rows")

        if len(df_steps) <= 2:
            print(f"\n⚠️  WARNING: Only {len(df_steps)} data points found!")
            print(f"DEBUG: Showing full df_steps:")
            print(df_steps[['step', loss_col]].to_string())
            print(f"\nDEBUG: Checking original df for {loss_col}:")
            print(
                f"  Non-null {loss_col} in original df: {df[loss_col].notna().sum()}")
            print(f"  Sample of non-null {loss_col} rows:")
            sample_df = df[df[loss_col].notna(
            )][['step', 'epoch', loss_col]].head(10)
            print(sample_df.to_string())
        if len(df_steps) > 0:
            print(
                f"DEBUG: First few step values: {df_steps['step'].head(10).tolist()}")
            print(
                f"DEBUG: Last few step values: {df_steps['step'].tail(10).tolist()}")
            print(
                f"DEBUG: Step range: {df_steps['step'].min()} to {df_steps['step'].max()}")

        # Downsample if too many points to avoid matplotlib rendering issues
        if len(df_steps) > max_points:
            original_len = len(df_steps)
            # Sample evenly across the range
            indices = [0]  # Always include first point
            step_size = len(df_steps) / (max_points - 1)
            for i in range(1, max_points - 1):
                indices.append(int(i * step_size))
            indices.append(len(df_steps) - 1)  # Always include last point

            print(f"DEBUG: Sampling indices (first 10): {indices[:10]}")
            print(f"DEBUG: Total sampled indices: {len(indices)}")

            df_steps = df_steps.iloc[indices].reset_index(drop=True)
            print(
                f"Downsampled from {original_len} to {len(df_steps)} points for plotting")
            print(
                f"DEBUG: After downsampling, step values (first 10): {df_steps['step'].head(10).tolist()}")
            print(
                f"DEBUG: After downsampling, step values (last 10): {df_steps['step'].tail(10).tolist()}\n")
        else:
            print(
                f"Using all {len(df_steps)} training steps (no downsampling needed)\n")

        return df_steps, 'Step', loss_col


def plot_overall_loss(
        df_data: pd.DataFrame,
        plot_dir: str,
        exp_name: str,
        x_axis: str = 'Epoch',
        loss_col: str = 'train_loss_epoch') -> None:
    """Plot overall training and validation loss.

    Args:
        df_data: DataFrame with metrics (aggregated by epoch or step)
        plot_dir: Directory to save plot
        exp_name: Experiment name for title
        x_axis: X-axis label ('Epoch' or 'Step')
        loss_col: Name of the training loss column to plot
    """
    print("Creating Plot 1: Overall Loss...")

    # Determine x-axis column
    x_col = 'epoch' if x_axis == 'Epoch' else 'step'

    print(f"  DEBUG: df_data has {len(df_data)} rows")
    print(f"  DEBUG: x_col = '{x_col}', loss_col = '{loss_col}'")
    print(f"  DEBUG: x values (first 10): {df_data[x_col].head(10).tolist()}")
    print(f"  DEBUG: x values (last 10): {df_data[x_col].tail(10).tolist()}")
    print(
        f"  DEBUG: {loss_col} values (first 10): {df_data[loss_col].head(10).tolist()}")
    print(
        f"  DEBUG: Number of non-null {loss_col}: {df_data[loss_col].notna().sum()}")

    # Drop rows with NA/NaN in plot columns so matplotlib can plot
    plot_cols = [x_col, loss_col]
    df_clean = df_data[plot_cols].dropna()
    if len(df_clean) == 0:
        print("  ⚠ No valid (non-NA) data for overall loss plot; skipping.")
        return
    x_vals = df_clean[x_col].astype(float)
    y_vals = df_clean[loss_col].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_vals, y_vals, 'b-',
            label='Training Loss', linewidth=2, marker='o', markersize=5)

    # Plot validation loss
    # For step-based plotting, validation might be in val_loss_step column
    # For epoch-based plotting, it's in val_loss_epoch column
    val_col = 'val_loss_step' if x_axis == 'Step' else 'val_loss_epoch'

    if val_col in df_data.columns:
        df_val = df_data[[x_col, val_col]].dropna()
        if len(df_val) > 0:
            ax.plot(
                df_val[x_col].astype(float),
                df_val[val_col].astype(float),
                'r-',
                label='Validation Loss',
                linewidth=2,
                marker='s',
                markersize=5)
            print(f"  ✓ Plotted {len(df_val)} validation points")
        else:
            print("  ⚠ No validation data available yet")
    else:
        print(f"  ⚠ Validation column '{val_col}' not found")

    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Overall Training and Validation Loss - {exp_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir,
            'overall_loss.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: overall_loss.png")


def plot_learning_rate(df: pd.DataFrame, plot_dir: str, exp_name: str) -> None:
    """Plot learning rate schedule.

    Args:
        df: DataFrame with metrics (all steps)
        plot_dir: Directory to save plot
        exp_name: Experiment name for title
    """
    print("Creating Plot 2: Learning Rate...")
    if 'lr' in df.columns and 'step' in df.columns:
        lr_data = df[['step', 'lr']].dropna()
        print(f"  DEBUG: Found {len(lr_data)} rows with learning rate data")
        if len(lr_data) > 0:
            print(
                f"  DEBUG: LR step range: {lr_data['step'].min()} to {lr_data['step'].max()}")
            print(
                f"  DEBUG: LR value range: {lr_data['lr'].min():.2e} to {lr_data['lr'].max():.2e}")

            fig, ax = plt.subplots(figsize=(10, 6))
            x_lr = lr_data['step'].astype(float)
            y_lr = lr_data['lr'].astype(float)
            if len(lr_data) > 50:
                ax.plot(x_lr, y_lr, 'g-', linewidth=2)
            else:
                ax.plot(
                    x_lr,
                    y_lr,
                    'g-',
                    linewidth=2,
                    marker='o',
                    markersize=4)

            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title(
                f'Learning Rate Schedule - {exp_name}',
                fontsize=14,
                fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    plot_dir,
                    'learning_rate.png'),
                dpi=150,
                bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: learning_rate.png with {len(lr_data)} points")
        else:
            print(f"  ⚠ No learning rate data available")
    else:
        print(f"  ⚠ Learning rate column not found in metrics")


def plot_train_loss_by_instrument(
        df_data: pd.DataFrame,
        instruments: List[str],
        plot_dir: str,
        exp_name: str,
        x_axis: str = 'Epoch') -> None:
    """Plot training loss for all instruments.

    Args:
        df_data: DataFrame with metrics (aggregated by epoch or step)
        instruments: List of instrument names
        plot_dir: Directory to save plot
        exp_name: Experiment name for title
        x_axis: X-axis label ('Epoch' or 'Step')
    """
    print("Creating Plot 3: Training Loss by Instrument...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine x-axis column and loss column suffix
    x_col = 'epoch' if x_axis == 'Epoch' else 'step'
    loss_suffix = 'target_epoch' if x_axis == 'Epoch' else 'target_step'

    for instrument in instruments:
        train_col = f'train_loss_{instrument}_{loss_suffix}'
        if train_col in df_data.columns:
            df_line = df_data[[x_col, train_col]].dropna()
            if len(df_line) > 0:
                ax.plot(
                    df_line[x_col].astype(float),
                    df_line[train_col].astype(float),
                    label=instrument,
                    linewidth=2,
                    marker='o',
                    markersize=4)

    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(
        f'Training Loss by Instrument - {exp_name}',
        fontsize=14,
        fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir,
            'train_loss_by_instrument.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: train_loss_by_instrument.png")


def plot_val_loss_by_instrument(
        df_data: pd.DataFrame,
        instruments: List[str],
        plot_dir: str,
        exp_name: str,
        x_axis: str = 'Epoch') -> None:
    """Plot validation loss for all instruments.

    Args:
        df_data: DataFrame with metrics (aggregated by epoch or step)
        instruments: List of instrument names
        plot_dir: Directory to save plot
        exp_name: Experiment name for title
        x_axis: X-axis label ('Epoch' or 'Step')
    """
    print("Creating Plot 4: Validation Loss by Instrument...")
    fig, ax = plt.subplots(figsize=(12, 6))
    has_val_data = False

    # Determine x-axis column and loss column suffix
    x_col = 'epoch' if x_axis == 'Epoch' else 'step'
    loss_suffix = 'target_epoch' if x_axis == 'Epoch' else 'target_step'

    for instrument in instruments:
        val_col = f'val_loss_{instrument}_{loss_suffix}'
        if val_col in df_data.columns:
            df_line = df_data[[x_col, val_col]].dropna()
            if len(df_line) > 0:
                has_val_data = True
                ax.plot(
                    df_line[x_col].astype(float),
                    df_line[val_col].astype(float),
                    label=instrument,
                    linewidth=2,
                    marker='o',
                    markersize=4)

    if has_val_data:
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(
            f'Validation Loss by Instrument - {exp_name}',
            fontsize=14,
            fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir,
                'val_loss_by_instrument.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: val_loss_by_instrument.png")
    else:
        plt.close()
        print(f"  ⚠ Skipped: No validation data available yet")


def plot_individual_instruments(df_data: pd.DataFrame, instruments: List[str],
                                plot_dir: str, x_axis: str = 'Epoch') -> None:
    """Plot individual train vs validation loss for each instrument.

    Args:
        df_data: DataFrame with metrics (aggregated by epoch or step)
        instruments: List of instrument names
        plot_dir: Directory to save plots
        x_axis: X-axis label ('Epoch' or 'Step')
    """
    print("Creating Plot 5: Individual Instrument Plots...")

    # Determine x-axis column and loss column suffix
    x_col = 'epoch' if x_axis == 'Epoch' else 'step'
    loss_suffix = 'target_epoch' if x_axis == 'Epoch' else 'target_step'

    for instrument in instruments:
        train_col = f'train_loss_{instrument}_{loss_suffix}'
        val_col = f'val_loss_{instrument}_{loss_suffix}'

        if train_col in df_data.columns or val_col in df_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot training loss
            if train_col in df_data.columns:
                df_train = df_data[[x_col, train_col]].dropna()
                if len(df_train) > 0:
                    ax.plot(
                        df_train[x_col].astype(float),
                        df_train[train_col].astype(float),
                        'b-',
                        label='Training',
                        linewidth=2,
                        marker='o',
                        markersize=5)

            # Plot validation loss (only where available)
            if val_col in df_data.columns:
                df_val = df_data[[x_col, val_col]].dropna()
                if len(df_val) > 0:
                    ax.plot(
                        df_val[x_col].astype(float),
                        df_val[val_col].astype(float),
                        'r-',
                        label='Validation',
                        linewidth=2,
                        marker='s',
                        markersize=5)

            ax.set_xlabel(x_axis, fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{instrument} - Training vs Validation Loss',
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            safe_name = instrument.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(plot_dir, f'{safe_name}_loss.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {safe_name}_loss.png")


def plot_training_metrics(
        metrics_csv_path,
        output_dir='plots',
        exp_name='training_metrics',
        version='version_0',
        min_epochs=4):
    """
    Plot training and validation losses from metrics.csv file.

    Args:
        metrics_csv_path: Path to the metrics.csv file
        output_dir: Directory to save plots
        exp_name: Experiment name for plot titles
        version: Version name for output directory structure
        min_epochs: Minimum number of epochs to use epoch-based plotting (default: 2)

    Returns:
        str: Path to the directory where plots were saved
    """
    # Create output directory
    plot_dir = os.path.join(output_dir, exp_name, version)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to: {plot_dir}\n")

    # Step 1: Load data
    df = load_metrics_data(metrics_csv_path)

    # Step 2: Extract instruments
    instruments = extract_instruments(df)

    # Step 3: Prepare plot data (epoch or step based)
    df_plot, x_axis, loss_col = prepare_plot_data(df, min_epochs=min_epochs)

    # Step 4: Create plots
    plot_overall_loss(df_plot, plot_dir, exp_name, x_axis, loss_col)
    plot_learning_rate(df, plot_dir, exp_name)
    plot_train_loss_by_instrument(
        df_plot, instruments, plot_dir, exp_name, x_axis)
    plot_val_loss_by_instrument(
        df_plot,
        instruments,
        plot_dir,
        exp_name,
        x_axis)
    plot_individual_instruments(df_plot, instruments, plot_dir, x_axis)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {plot_dir}")
    print(f"{'='*60}")
    return plot_dir


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Plot training metrics from metrics.csv file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using exp_name (automatically finds latest version)
  python -m ocelot.plot_training_metrics --exp_name baseline_standard

  # Specify a particular version
  python -m ocelot.plot_training_metrics --exp_name baseline_standard --version 1

  # With custom base directory
  python -m ocelot.plot_training_metrics \\
      --base_dir /scratch3/NCEPDEV/da/Xin.C.Jin/git/staging/ocelot/logs \\
      --exp_name baseline_standard

  # Using explicit metrics_csv path (overrides base_dir)
  python -m ocelot.plot_training_metrics \\
      --metrics_csv /path/to/version_1/metrics.csv \\
      --exp_name baseline_standard

  # Custom output directory
  python -m ocelot.plot_training_metrics \\
      --exp_name my_exp \\
      --output_dir ./my_plots
        """
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='Experiment name (used for plot titles and output directory)')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/scratch3/NCEPDEV/da/Xin.C.Jin/git/staging/ocelot/logs',
        help='Base directory containing experiment subdirectories (default: logs)')
    parser.add_argument(
        '--metrics_csv',
        type=str,
        default=None,
        help='Path to the metrics.csv file (if not specified, uses base_dir/exp_name/version_X/metrics.csv)')
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Version number (e.g., "1" for version_1). If not specified, uses latest version.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Base output directory for plots (default: plots)')
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=4,
        help='Minimum number of epochs to use epoch-based plotting. If fewer epochs, uses step-based plotting (default: 2)')

    args = parser.parse_args()

    version_name = args.version

    # Determine metrics CSV path
    if args.metrics_csv:
        metrics_csv_path = args.metrics_csv
        if version_name is None:
            version_name = 'version_custom'
    else:
        exp_dir = os.path.join(args.base_dir, args.exp_name)

        # Find version directory
        if args.version:
            version_dir = os.path.join(exp_dir, f'{args.version}')
        else:
            # Find the latest version directory
            import glob
            version_dirs = glob.glob(os.path.join(exp_dir, 'version_*'))
            if not version_dirs:
                print(f"Error: No version directories found in {exp_dir}")
                return 1

            # Sort by version number
            version_dirs.sort(key=lambda x: int(x.split('version_')[-1]))
            version_dir = version_dirs[-1]
            version_num = version_dir.split('version_')[-1]
            version_name = f"version_{version_num}"
            print(
                f"Found {len(version_dirs)} version(s), using latest: {version_name}")

        metrics_csv_path = os.path.join(version_dir, 'metrics.csv')

    print(f"Looking for metrics at: {metrics_csv_path}")

    # Check if metrics file exists
    if not os.path.exists(metrics_csv_path):
        print(f"Error: Metrics file not found: {metrics_csv_path}")
        if not args.metrics_csv:
            print(
                f"Hint: Check if experiment '{args.exp_name}' exists in {args.base_dir}")
        return 1

    # Generate plots
    try:
        plot_training_metrics(
            metrics_csv_path,
            args.output_dir,
            args.exp_name,
            version=version_name,
            min_epochs=args.min_epochs)
        return 0
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
