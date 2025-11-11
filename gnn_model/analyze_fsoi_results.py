#!/usr/bin/env python
"""
Analyze operational FSOI results from CSV files.

This script helps you investigate FSOI results by:
1. Loading and summarizing CSV files from multiple epochs
2. Computing statistics by instrument and channel
3. Generating diagnostic plots
4. Creating comparison tables

Usage:
    python analyze_fsoi_results.py --results_dir fsoi_results_operational
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def format_channel_label(instrument: str, channel: str) -> str:
    """
    Format channel label handling both numeric and string channel names.
    
    Examples:
        'amsua', 'bt_channel_9' -> 'amsua ch9'
        'surface_obs', 'airTemperature' -> 'surface_obs airTemperature'
        'atms', '5' -> 'atms ch5'
    """
    channel_str = str(channel)
    
    # Extract numeric part if channel is like 'bt_channel_9', 'channel_1', etc.
    if 'channel' in channel_str.lower():
        match = re.search(r'(\d+)', channel_str)
        if match:
            ch_num = match.group(1)
            return f"{instrument} ch{ch_num}"
    
    # Try to convert to int if it's a numeric string
    try:
        ch_num = int(float(channel_str))
        return f"{instrument} ch{ch_num}"
    except:
        # If all else fails, use the channel name as-is (e.g., 'airTemperature')
        return f"{instrument} {channel_str}"


def load_fsoi_csvs(results_dir: str) -> pd.DataFrame:
    """
    Load all FSOI CSV files from the detailed directory.
    
    Args:
        results_dir: Base results directory (e.g., fsoi_results_operational)
        
    Returns:
        Combined DataFrame with all epochs
    """
    detailed_dir = Path(results_dir) / "detailed"
    
    if not detailed_dir.exists():
        print(f"❌ Directory not found: {detailed_dir}")
        return None
    
    # Find all CSV files
    csv_files = sorted(detailed_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {detailed_dir}")
        return None
    
    print(f"\n📁 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # Load and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract epoch from filename (e.g., fsoi_operational_epoch5_batch0.csv)
            parts = csv_file.stem.split('_')
            epoch_part = [p for p in parts if p.startswith('epoch')]
            if epoch_part:
                epoch = int(epoch_part[0].replace('epoch', ''))
                df['epoch'] = epoch
            dfs.append(df)
            print(f"   ✓ Loaded {csv_file.name}: {len(df)} observations")
        except Exception as e:
            print(f"   ✗ Error loading {csv_file.name}: {e}")
    
    if not dfs:
        print("❌ No data loaded successfully")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(combined_df)} total observations")
    print(f"  Epochs: {sorted(combined_df['epoch'].unique())}")
    print(f"  Instruments: {sorted(combined_df['instrument'].unique())}")
    
    return combined_df


def summarize_fsoi_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics by instrument and epoch.
    
    Returns:
        DataFrame with statistics (mean, std, min, max FSOI per instrument/epoch)
    """
    print("\n" + "="*80)
    print("FSOI STATISTICS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print("\n📊 Overall Statistics:")
    print(f"   Total observations: {len(df)}")
    print(f"   Mean FSOI: {df['fsoi'].mean():.6f}")
    print(f"   Std FSOI: {df['fsoi'].std():.6f}")
    print(f"   Min FSOI: {df['fsoi'].min():.6f}")
    print(f"   Max FSOI: {df['fsoi'].max():.6f}")
    
    # Statistics by instrument
    print("\n📊 By Instrument:")
    instrument_stats = df.groupby('instrument')['fsoi'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ]).round(6)
    print(instrument_stats.to_string())
    
    # Statistics by epoch
    if 'epoch' in df.columns:
        print("\n📊 By Epoch:")
        epoch_stats = df.groupby('epoch')['fsoi'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(6)
        print(epoch_stats.to_string())
    
    # Statistics by instrument and epoch
    if 'epoch' in df.columns:
        print("\n📊 By Instrument and Epoch:")
        inst_epoch_stats = df.groupby(['instrument', 'epoch'])['fsoi'].agg([
            ('count', 'count'),
            ('mean', 'mean')
        ]).round(6)
        print(inst_epoch_stats.to_string())
    
    return instrument_stats


def check_fsoi_components(df: pd.DataFrame):
    """
    Check the FSOI components (sensitivities, innovations, background).
    """
    print("\n" + "="*80)
    print("FSOI COMPONENTS CHECK")
    print("="*80)
    
    required_cols = ['observation', 'background', 'innovation', 'sensitivity', 'fsoi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Check innovations (O-B)
    print("\n📊 Innovations (Observation - Background):")
    print(f"   Mean: {df['innovation'].mean():.6f}")
    print(f"   Std: {df['innovation'].std():.6f}")
    print(f"   Min: {df['innovation'].min():.6f}")
    print(f"   Max: {df['innovation'].max():.6f}")
    
    # Check if innovations are computed correctly
    if 'observation' in df.columns and 'background' in df.columns:
        computed_innov = df['observation'] - df['background']
        innov_match = np.allclose(computed_innov, df['innovation'], rtol=1e-3)
        if innov_match:
            print("   ✓ Innovations match (O-B) formula")
        else:
            print("   ⚠️  Innovations don't match (O-B) formula")
    
    # Check sensitivities
    print("\n📊 Sensitivities (Adjoint ∇J/∂obs):")
    print(f"   Mean: {df['sensitivity'].mean():.6f}")
    print(f"   Std: {df['sensitivity'].std():.6f}")
    print(f"   Min: {df['sensitivity'].min():.6f}")
    print(f"   Max: {df['sensitivity'].max():.6f}")
    
    # Check FSOI sign (GraphDOP convention: negative = beneficial)
    print("\n📊 FSOI Sign Distribution:")
    print("   (GraphDOP convention: Negative FSOI = beneficial, Positive FSOI = detrimental)")
    positive_fsoi = (df['fsoi'] > 0).sum()
    negative_fsoi = (df['fsoi'] < 0).sum()
    zero_fsoi = (df['fsoi'] == 0).sum()
    
    print(f"   Positive FSOI (detrimental): {positive_fsoi} ({100*positive_fsoi/len(df):.1f}%)")
    print(f"   Negative FSOI (beneficial):  {negative_fsoi} ({100*negative_fsoi/len(df):.1f}%)")
    print(f"   Zero FSOI:                   {zero_fsoi} ({100*zero_fsoi/len(df):.1f}%)")
    
    if negative_fsoi > positive_fsoi:
        print("   ✓ Most observations are beneficial (negative FSOI)")
    else:
        print("   ⚠️  Most observations are detrimental (positive FSOI) - check if expected")


def plot_fsoi_per_channel_graphdop_style(df: pd.DataFrame, output_dir: str):
    """
    Generate GraphDOP-style plot showing FSOI per channel with horizontal bars.
    Similar to the ECMWF GraphDOP sensitivity analysis visualization.
    """
    print("\n📈 Creating GraphDOP-style per-channel FSOI plot...")
    
    # Check if channel information exists
    if 'channel' not in df.columns:
        print("   ⚠️  No channel information found, skipping per-channel plot")
        return
    
    # Use 'fsoi_value' if it exists (new format), else 'fsoi' (old format)
    fsoi_col = 'fsoi_value' if 'fsoi_value' in df.columns else 'fsoi'
    
    # Compute mean FSOI per instrument-channel combination
    # Convert to percentage by multiplying by 100
    channel_stats = df.groupby(['instrument', 'channel']).agg({
        fsoi_col: ['mean', 'count']
    }).reset_index()
    
    channel_stats.columns = ['instrument', 'channel', 'fsoi_mean', 'count']
    channel_stats['fsoi_pct'] = channel_stats['fsoi_mean'] * 100  # Convert to percentage
    
    # Sort by instrument, then by FSOI value
    channel_stats = channel_stats.sort_values(['instrument', 'fsoi_pct'], ascending=[True, True])
    
    # Create labels (instrument + channel)
    channel_stats['label'] = channel_stats.apply(
        lambda row: format_channel_label(row['instrument'], row['channel']), axis=1
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(channel_stats) * 0.25)))
    
    # Color bars: green for beneficial (negative FSOI), red for detrimental (positive FSOI)
    colors = ['green' if x < 0 else 'red' for x in channel_stats['fsoi_pct']]
    
    # Create horizontal bar chart
    y_positions = np.arange(len(channel_stats))
    bars = ax.barh(y_positions, channel_stats['fsoi_pct'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize appearance
    ax.set_yticks(y_positions)
    ax.set_yticklabels(channel_stats['label'], fontsize=8)
    ax.set_xlabel('FSOI per observation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity tools to study GraphDOP physical consistency\n' + 
                 'FSOI estimates how every observation contributes at reducing the forecast error',
                 fontsize=13, fontweight='bold', pad=20)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Beneficial (reduces error)'),
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Detrimental (increases error)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'fsoi_per_channel_graphdop_style.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n   Channel Statistics:")
    print(f"   Total channels analyzed: {len(channel_stats)}")
    beneficial = (channel_stats['fsoi_pct'] < 0).sum()
    detrimental = (channel_stats['fsoi_pct'] > 0).sum()
    print(f"   Beneficial channels: {beneficial} ({100*beneficial/len(channel_stats):.1f}%)")
    print(f"   Detrimental channels: {detrimental} ({100*detrimental/len(channel_stats):.1f}%)")
    
    # Print top 5 most beneficial and detrimental channels
    print(f"\n   Top 5 Most Beneficial Channels:")
    top_beneficial = channel_stats.nsmallest(5, 'fsoi_pct')[['label', 'fsoi_pct', 'count']]
    for idx, row in top_beneficial.iterrows():
        print(f"      {row['label']:25s}: {row['fsoi_pct']:8.4f}% (n={int(row['count'])})")
    
    print(f"\n   Top 5 Most Detrimental Channels:")
    top_detrimental = channel_stats.nlargest(5, 'fsoi_pct')[['label', 'fsoi_pct', 'count']]
    for idx, row in top_detrimental.iterrows():
        print(f"      {row['label']:25s}: {row['fsoi_pct']:8.4f}% (n={int(row['count'])})")


def plot_input_impact_on_target(df: pd.DataFrame, output_dir: str, target_instrument: str = 'surface_obs'):
    """
    Create plots showing how different INPUT observation types
    contribute to predicting TARGET observations.
    
    This answers: "How do all observation types (satellite + surface) affect
    the prediction of conventional observations (u, v, T, q, p)?"
    
    Args:
        df: FSOI DataFrame with 'instrument' (input type), 'fsoi' columns
        output_dir: Where to save the plot
        target_instrument: Which target to analyze (ignored if target_variable exists)
    """
    print(f"\n📈 Creating input impact analysis...")
    
    # Check if this is conventional obs analysis (has target_variable column)
    if 'target_variable' in df.columns:
        print("   ✓ Detected conventional obs analysis (u, v, T, q, p)")
        plot_conventional_obs_impact(df, output_dir)
    else:
        print("   ⚠️  Showing overall FSOI (all targets combined)")
        plot_overall_impact(df, output_dir)


def plot_conventional_obs_impact(df: pd.DataFrame, output_dir: str):
    """
    Plot how all inputs affect conventional observations (prognostic variables).
    """
    print("   Creating conventional obs impact plots...")
    
    # 1. Overall impact by input type (averaged across all conventional targets)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Mean FSOI by input type
    inst_stats = df.groupby('instrument')['fsoi_value'].mean().sort_values() * 100
    colors = ['green' if x < 0 else 'red' for x in inst_stats.values]
    
    inst_stats.plot(kind='barh', color=colors, ax=ax1, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Mean FSOI Impact (%)\n(on conventional obs: u,v,T,q,p)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Input Observation Type', fontsize=11, fontweight='bold')
    ax1.set_title('Impact on Prognostic Variables\n(Surface + Radiosonde)', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Heatmap of input × target
    pivot = df.pivot_table(
        values='fsoi_value',
        index='instrument',
        columns='target_variable',
        aggfunc='mean'
    ) * 100
    
    # Plot heatmap using matplotlib's imshow
    im = ax2.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', 
                    vmin=-np.abs(pivot.values).max(), vmax=np.abs(pivot.values).max())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Mean FSOI (%)', rotation=270, labelpad=20)
    
    # Add annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax2.text(j, i, f'{pivot.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    # Set ticks and labels
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax2.set_yticklabels(pivot.index)
    
    ax2.set_xlabel('Target Variable', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Input Observation Type', fontsize=11, fontweight='bold')
    ax2.set_title('Input × Target Breakdown', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'input_impact_on_conventional_obs.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # 2. Individual plots for each conventional target
    for target in df['target_variable'].unique():
        target_df = df[df['target_variable'] == target]
        
        plt.figure(figsize=(10, 6))
        inst_stats = target_df.groupby('instrument')['fsoi_value'].mean().sort_values() * 100
        colors = ['green' if x < 0 else 'red' for x in inst_stats.values]
        
        inst_stats.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
        plt.xlabel('Mean FSOI Impact (%)', fontsize=11, fontweight='bold')
        plt.ylabel('Input Observation Type', fontsize=11, fontweight='bold')
        plt.title(f'Impact on {target.upper()}\n' + 
                 f'How all observations affect predicting {target}',
                 fontsize=12, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'input_impact_on_{target}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {output_file}")
        plt.close()


def plot_overall_impact(df: pd.DataFrame, output_dir: str):
    """
    Plot overall impact (when target_variable column doesn't exist).
    """
    plt.figure(figsize=(12, 6))
    
    # Use 'fsoi' column if available, otherwise 'fsoi_value'
    fsoi_col = 'fsoi_value' if 'fsoi_value' in df.columns else 'fsoi'
    
    inst_stats = df.groupby('instrument')[fsoi_col].mean().sort_values() * 100
    colors = ['green' if x < 0 else 'red' for x in inst_stats.values]
    
    inst_stats.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('Mean FSOI Impact (%)\n(on ALL target predictions)', fontsize=12, fontweight='bold')
    plt.ylabel('Input Observation Type', fontsize=12, fontweight='bold')
    plt.title('Input Observation Impact on Forecast Error\n' +
              '(All targets combined)',
              fontsize=13, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'input_impact_on_all_targets.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()


def plot_fsoi_by_instrument(df: pd.DataFrame, output_dir: str):
    """
    Generate plots of FSOI by instrument.
    """
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 0. Input impact on targets (overall)
    plot_input_impact_on_target(df, output_dir)
    
    # 1. GraphDOP-style per-channel plot (if channel data exists)
    plot_fsoi_per_channel_graphdop_style(df, output_dir)
    
    # 1. Box plot of FSOI by instrument
    print("\n📈 Creating box plot by instrument...")
    plt.figure(figsize=(12, 6))
    instruments = sorted(df['instrument'].unique())
    
    data_by_inst = [df[df['instrument'] == inst]['fsoi'].values for inst in instruments]
    
    plt.boxplot(data_by_inst, labels=instruments)
    plt.ylabel('FSOI', fontsize=12)
    plt.xlabel('Instrument', fontsize=12)
    plt.title('FSOI Distribution by Instrument', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fsoi_by_instrument_boxplot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # 2. Mean FSOI by instrument (bar plot)
    print("\n📈 Creating mean FSOI bar plot...")
    plt.figure(figsize=(12, 6))
    
    inst_means = df.groupby('instrument')['fsoi'].mean().sort_values()
    colors = ['green' if x < 0 else 'red' for x in inst_means.values]
    
    inst_means.plot(kind='barh', color=colors)
    plt.xlabel('Mean FSOI', fontsize=12)
    plt.ylabel('Instrument', fontsize=12)
    plt.title('Mean FSOI by Instrument\n(Negative = Beneficial)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fsoi_by_instrument_mean.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # 3. FSOI evolution by epoch (if multiple epochs)
    if 'epoch' in df.columns and len(df['epoch'].unique()) > 1:
        print("\n📈 Creating FSOI evolution plot...")
        plt.figure(figsize=(14, 8))
        
        for inst in instruments:
            inst_data = df[df['instrument'] == inst]
            epoch_means = inst_data.groupby('epoch')['fsoi'].mean()
            plt.plot(epoch_means.index, epoch_means.values, marker='o', label=inst, linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean FSOI', fontsize=12)
        plt.title('FSOI Evolution During Training', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'fsoi_evolution_by_epoch.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {output_file}")
        plt.close()
    
    # 4. Innovation vs Sensitivity scatter plot
    print("\n📈 Creating innovation vs sensitivity scatter...")
    plt.figure(figsize=(10, 8))
    
    for inst in instruments:
        inst_data = df[df['instrument'] == inst]
        plt.scatter(inst_data['innovation'], inst_data['sensitivity'], 
                   alpha=0.3, s=20, label=inst)
    
    plt.xlabel('Innovation (O-B)', fontsize=12)
    plt.ylabel('Sensitivity (∇J/∂obs)', fontsize=12)
    plt.title('Innovation vs Sensitivity by Instrument', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'innovation_vs_sensitivity.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # 5. FSOI histogram
    print("\n📈 Creating FSOI histogram...")
    plt.figure(figsize=(12, 6))
    
    plt.hist(df['fsoi'], bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('FSOI', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('FSOI Distribution', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero FSOI')
    plt.axvline(x=df['fsoi'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean = {df["fsoi"].mean():.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fsoi_histogram.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    print(f"\n✓ All plots saved to: {output_dir}/")


def create_summary_report(df: pd.DataFrame, output_file: str):
    """
    Create a text summary report.
    """
    print(f"\n📄 Creating summary report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPERATIONAL FSOI ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total observations: {len(df)}\n")
        f.write(f"Mean FSOI: {df['fsoi'].mean():.6f}\n")
        f.write(f"Std FSOI: {df['fsoi'].std():.6f}\n")
        f.write(f"Min FSOI: {df['fsoi'].min():.6f}\n")
        f.write(f"Max FSOI: {df['fsoi'].max():.6f}\n")
        f.write(f"Median FSOI: {df['fsoi'].median():.6f}\n\n")
        
        # By instrument
        f.write("STATISTICS BY INSTRUMENT\n")
        f.write("-"*80 + "\n")
        inst_stats = df.groupby('instrument')['fsoi'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(6)
        f.write(inst_stats.to_string() + "\n\n")
        
        # Ranking
        f.write("INSTRUMENT RANKING (Most Beneficial Impact)\n")
        f.write("-"*80 + "\n")
        inst_means = df.groupby('instrument')['fsoi'].mean().sort_values()
        for i, (inst, mean_fsoi) in enumerate(inst_means.items(), 1):
            impact = "BENEFICIAL" if mean_fsoi < 0 else "DETRIMENTAL"
            f.write(f"{i:2d}. {inst:20s} Mean FSOI: {mean_fsoi:10.6f}  [{impact}]\n")
        
        # By epoch
        if 'epoch' in df.columns:
            f.write("\n\nSTATISTICS BY EPOCH\n")
            f.write("-"*80 + "\n")
            epoch_stats = df.groupby('epoch')['fsoi'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(6)
            f.write(epoch_stats.to_string() + "\n")
        
        # Sign distribution
        f.write("\n\nFSOI SIGN DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write("GraphDOP convention: Negative FSOI = beneficial, Positive FSOI = detrimental\n\n")
        positive = (df['fsoi'] > 0).sum()
        negative = (df['fsoi'] < 0).sum()
        zero = (df['fsoi'] == 0).sum()
        f.write(f"Positive FSOI (detrimental): {positive:8d} ({100*positive/len(df):5.1f}%)\n")
        f.write(f"Negative FSOI (beneficial):  {negative:8d} ({100*negative/len(df):5.1f}%)\n")
        f.write(f"Zero FSOI:                   {zero:8d} ({100*zero/len(df):5.1f}%)\n")
        
    print(f"   ✓ Report saved")


def main():
    parser = argparse.ArgumentParser(description="Analyze operational FSOI results")
    parser.add_argument('--results_dir', type=str, default='fsoi_results_operational',
                       help='Base results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/analysis)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    
    print("\n" + "="*80)
    print("OPERATIONAL FSOI ANALYSIS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    df = load_fsoi_csvs(args.results_dir)
    
    if df is None or len(df) == 0:
        print("\n❌ No data to analyze. Exiting.")
        return 1
    
    # Compute statistics
    summarize_fsoi_statistics(df)
    
    # Check components
    check_fsoi_components(df)
    
    # Generate plots
    if not args.no_plots:
        plot_fsoi_by_instrument(df, args.output_dir)
    
    # Create summary report
    report_file = os.path.join(args.output_dir, 'fsoi_analysis_summary.txt')
    create_summary_report(df, report_file)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated files:")
    if 'channel' in df.columns:
        print(f"  - fsoi_per_channel_graphdop_style.png  [GraphDOP-style visualization]")
    print(f"  - fsoi_by_instrument_boxplot.png")
    print(f"  - fsoi_by_instrument_mean.png")
    if 'epoch' in df.columns and len(df['epoch'].unique()) > 1:
        print(f"  - fsoi_evolution_by_epoch.png")
    print(f"  - innovation_vs_sensitivity.png")
    print(f"  - fsoi_histogram.png")
    print(f"  - fsoi_analysis_summary.txt")
    
    print("\n💡 Next steps:")
    print("  1. Review the plots to understand observation impact")
    print("  2. Check fsoi_analysis_summary.txt for detailed statistics")
    print("  3. Instruments with negative mean FSOI have beneficial impact")
    print("  4. Compare results across epochs to see FSOI evolution during training")
    print("  5. Update observation_errors.yaml if needed for EMC comparison\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
